import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..state import TINYNAV_DB_PATH, runner

router = APIRouter(prefix='/agent', tags=['agent'])

_LLAMA_BASE_URL = os.environ.get('TINYNAV_LLM_BASE_URL', 'http://127.0.0.1:8888')

_ACTIONS = {
    'chat',
    'current_map',
    'list_maps',
    'select_map',
    'start_localization',
    'stop_localization',
    'list_pois',
    'go_pois',
    'pause_nav',
    'resume_nav',
    'cancel_nav',
    'status',
    'unknown',
}


class AgentCommandRequest(BaseModel):
    text: str
    execute: bool = False


class AgentIntent(BaseModel):
    action: Literal[
        'chat',
        'current_map',
        'list_maps',
        'select_map',
        'start_localization',
        'stop_localization',
        'list_pois',
        'go_pois',
        'pause_nav',
        'resume_nav',
        'cancel_nav',
        'status',
        'unknown',
    ]
    map_name: str | None = None
    poi_names: list[str] = Field(default_factory=list)
    response: str | None = None
    reason: str | None = None


def _dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, 'model_dump'):
        return model.model_dump()
    return model.dict()


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


def _db_root() -> Path:
    return Path(TINYNAV_DB_PATH)


def _maps_root() -> Path:
    return _db_root() / 'maps'


def _active_map_link() -> Path:
    return _db_root() / 'map'


def _safe_name(name: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_.-]+$', name))


def _list_maps() -> list[str]:
    root = _maps_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _active_map_name() -> str | None:
    link = _active_map_link()
    if not link.exists():
        return None
    try:
        target = link.resolve()
    except FileNotFoundError:
        return None
    maps_root = _maps_root().resolve()
    try:
        target.relative_to(maps_root)
    except ValueError:
        return None
    return target.name


def _set_active_map(map_name: str) -> str:
    if not _safe_name(map_name):
        raise HTTPException(400, 'Invalid map name')
    src = _maps_root() / map_name
    if not src.is_dir():
        raise HTTPException(404, f'Map {map_name!r} not found')
    link = _active_map_link()
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    link.symlink_to(src)
    return map_name


def _load_active_pois() -> list[dict[str, Any]]:
    active = _active_map_name()
    if active is None:
        return []
    pois_path = _maps_root() / active / 'pois.json'
    if not pois_path.exists():
        return []
    with pois_path.open() as f:
        data = json.load(f)
    return list(data.values())


def _active_map_info() -> dict[str, Any]:
    active = _active_map_name()
    if active is None:
        return {
            'activeMap': None,
            'mapPath': None,
            'message': 'No active map selected',
        }
    return {
        'activeMap': active,
        'mapPath': str(_maps_root() / active),
    }


def _node_status() -> dict[str, Any]:
    node = runner.node
    if node is None:
        return {'ready': False}
    status = node.get_status()
    with node._lock:
        status['localized'] = bool(node._localized)
    status['activeMap'] = _active_map_name()
    return status


def _agent_context() -> dict[str, Any]:
    return {
        'available_maps': _list_maps(),
        'active_map': _active_map_name(),
        'nav_status': _node_status(),
        'pois': [
            {'id': int(p['id']), 'name': str(p['name'])}
            for p in _load_active_pois()
            if 'id' in p and 'name' in p
        ],
    }


def _contains_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def _asks_for_multiple_destinations(text: str) -> bool:
    return _contains_any(text, ['先', '再', '然后', '依次', '顺路', '顺便', '几个', '多个'])


def _looks_like_navigation_request(text: str) -> bool:
    lower_text = text.lower()
    return (
        _contains_any(text, ['去', '到', '带我', '导航'])
        or 'go to' in lower_text
        or 'take me to' in lower_text
        or 'navigate to' in lower_text
    )


def _exact_match_intent(text: str, context: dict[str, Any]) -> AgentIntent | None:
    lower_text = text.lower()
    compact_text = lower_text.strip(' ?!.。！？')

    if 'any map' in lower_text or compact_text in ('maps', 'list maps'):
        return AgentIntent(action='list_maps')

    if (
        _contains_any(text, ['地图'])
        and _contains_any(text, ['有什么', '有哪些', '列', '查看', '看看'])
    ):
        return AgentIntent(action='list_maps')

    if 'any poi' in lower_text or 'pois' in lower_text:
        return AgentIntent(action='list_pois')

    if _contains_any(text, ['当前地图', '现在地图', '什么地图', '哪个地图']):
        return AgentIntent(action='current_map')

    if _contains_any(text, ['状态', '怎么样', '什么情况']) or 'status' in lower_text:
        return AgentIntent(action='status')

    if _contains_any(text, ['暂停']):
        return AgentIntent(action='pause_nav')
    if _contains_any(text, ['继续', '恢复']):
        return AgentIntent(action='resume_nav')
    if _contains_any(text, ['取消', '停止导航', '结束导航']):
        return AgentIntent(action='cancel_nav')
    if _contains_any(text, ['关闭定位', '停止定位']):
        return AgentIntent(action='stop_localization')
    if (
        _contains_any(text, ['poi', 'POI'])
        and _contains_any(text, ['有什么', '有哪些', '列', '查看', '看看'])
    ):
        return AgentIntent(action='list_pois')

    if compact_text in ('hi', 'hello', 'hey') or _contains_any(text, ['你好', '您好', '哈喽']):
        return AgentIntent(action='chat', response='Hello! I can help with maps, POIs, localization, and navigation.')

    matched_map = next((name for name in context['available_maps'] if name in text), None)
    if matched_map is not None:
        if _contains_any(text, ['定位', '启动']):
            return AgentIntent(action='start_localization', map_name=matched_map)
        if _contains_any(text, ['选择', '切换', '地图', 'map']):
            return AgentIntent(action='select_map', map_name=matched_map)

    matched_pois = [p['name'] for p in context['pois'] if p['name'] in text]
    if matched_pois and _looks_like_navigation_request(text):
        return AgentIntent(action='go_pois', poi_names=matched_pois)

    return None


def _normalize_intent(text: str, intent: AgentIntent) -> AgentIntent:
    wants_localization = _contains_any(text, ['定位', '启动', '开始'])
    wants_map_selection = _contains_any(text, ['地图', 'map', '选择', '切换', '使用'])
    if (
        intent.action == 'start_localization'
        and intent.map_name is not None
        and wants_map_selection
        and not wants_localization
    ):
        return AgentIntent(action='select_map', map_name=intent.map_name)
    if intent.action == 'go_pois' and len(intent.poi_names) > 1 and not _asks_for_multiple_destinations(text):
        return AgentIntent(action='go_pois', poi_names=intent.poi_names[:1])
    return intent


def _build_prompt(text: str, context: dict[str, Any]) -> list[dict[str, str]]:
    system = """
You are a command parser for TinyNav robot navigation.

Return exactly one JSON object. Do not use markdown. Do not explain.

Allowed actions:
- current_map: show the current active map.
- list_maps: list available maps.
- select_map: choose an active localization map. Fields: action, map_name.
- start_localization: start localization/nav nodes. Optional field: map_name.
- stop_localization: stop localization/nav nodes.
- list_pois: list POIs in the active map.
- go_pois: navigate to one or more POIs by semantic matching. Fields: action, poi_names.
- pause_nav: pause active navigation.
- resume_nav: resume paused navigation.
- cancel_nav: cancel active navigation.
- status: query current status.
- chat: normal conversation without controlling the robot. Field: response.
- unknown: use when the command is ambiguous or unsupported. Field: reason.

Rules:
- Choose map_name only from available_maps.
- Choose poi_names only from pois[].name.
- If the user describes a place semantically, choose the best matching existing POI name.
- Return exactly one POI unless the user clearly asks for multiple destinations.
- Keep POI order only if the user clearly asks for multiple destinations.
- Only return go_pois when the user clearly asks to go, navigate, or be taken somewhere.
- For greetings or casual conversation, return chat with a short response.
- For asking available maps, return list_maps. For asking available POIs, return list_pois.
- Never invent maps or POIs.
- If no existing POI or map matches, return unknown.

Examples:
- If user_text is "选择 map_2026_05_07_10_56_41 地图" and that map is available,
  return {"action":"select_map","map_name":"map_2026_05_07_10_56_41"}.
- If user_text is "我想去打印" and a POI named "printer" exists,
  return {"action":"go_pois","poi_names":["printer"]}.
- If user_text is "先去 printer 再去 desk" and both POIs exist,
  return {"action":"go_pois","poi_names":["printer","desk"]}.
""".strip()
    user = json.dumps(
        {
            **context,
            'user_text': text,
        },
        ensure_ascii=False,
    )
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise HTTPException(502, f'LLM did not return JSON: {text[:200]}')
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise HTTPException(502, f'Invalid LLM JSON: {e}') from e


def _call_llama(text: str, context: dict[str, Any]) -> AgentIntent:
    payload = {
        'model': 'tinynav-local-llm',
        'messages': _build_prompt(text, context),
        'temperature': 0,
        'max_tokens': 256,
    }
    try:
        resp = httpx.post(
            f'{_LLAMA_BASE_URL}/v1/chat/completions',
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(503, f'LLM server unavailable: {e}') from e

    data = resp.json()
    try:
        content = data['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(502, 'Unexpected LLM response format') from e

    raw = _extract_json(content)
    if 'action' not in raw and 'tool' in raw:
        raw['action'] = raw.get('tool')
    action = raw.get('action')
    if action not in _ACTIONS:
        raw = {'action': 'unknown', 'reason': f'Unsupported action: {action}'}
    try:
        return AgentIntent(**raw)
    except Exception as e:
        raise HTTPException(502, f'Invalid LLM intent: {e}') from e


def _validate_intent(intent: AgentIntent, context: dict[str, Any]) -> AgentIntent:
    maps = set(context['available_maps'])
    poi_names = {p['name'] for p in context['pois']}

    if intent.map_name is not None and intent.map_name not in maps:
        return AgentIntent(action='unknown', reason=f'Map not found: {intent.map_name}')

    if intent.action == 'go_pois':
        # Safety: LLMs tend to over-eagerly interpret casual text as navigation.
        # A POI command must have a clear navigation verb in the user's text.
        # This is checked by callers via _normalize_intent before validation.
        if not intent.poi_names:
            return AgentIntent(action='unknown', reason='No POI selected')
        missing = [name for name in intent.poi_names if name not in poi_names]
        if missing:
            return AgentIntent(action='unknown', reason=f'POI not found: {", ".join(missing)}')

    return intent


def _resolve_poi_ids(poi_names: list[str], context: dict[str, Any]) -> list[int]:
    by_name = {p['name']: int(p['id']) for p in context['pois']}
    return [by_name[name] for name in poi_names]


def _execute_intent(intent: AgentIntent, context: dict[str, Any]) -> dict[str, Any]:
    if intent.action == 'chat':
        return {'ok': True, 'intent': _dump_model(intent), 'message': intent.response or 'Hello.'}

    node = _require_node()

    if intent.action == 'unknown':
        return {'ok': False, 'intent': _dump_model(intent), 'message': intent.reason or 'Unknown command'}

    if intent.action == 'status':
        return {
            'ok': True,
            'intent': _dump_model(intent),
            'status': _node_status(),
            'availableMaps': context['available_maps'],
            **_active_map_info(),
        }

    if intent.action == 'current_map':
        return {'ok': True, 'intent': _dump_model(intent), **_active_map_info()}

    if intent.action == 'list_maps':
        return {
            'ok': True,
            'intent': _dump_model(intent),
            'availableMaps': context['available_maps'],
            **_active_map_info(),
        }

    if intent.action == 'list_pois':
        return {'ok': True, 'intent': _dump_model(intent), **_active_map_info(), 'pois': context['pois']}

    if intent.action == 'select_map':
        if intent.map_name is None:
            raise HTTPException(400, 'map_name is required')
        active = _set_active_map(intent.map_name)
        return {'ok': True, 'intent': _dump_model(intent), 'activeMap': active}

    if intent.action == 'start_localization':
        if intent.map_name is not None:
            _set_active_map(intent.map_name)
        if _active_map_name() is None:
            return {'ok': False, 'intent': _dump_model(intent), 'message': 'No active map selected'}
        with node._lock:
            running = node._nav_nodes_running
        if running:
            node.cmd_stop_nav_nodes()
        node.cmd_start_nav_nodes()
        return {'ok': True, 'intent': _dump_model(intent), 'message': 'Localization started'}

    if intent.action == 'stop_localization':
        node.cmd_stop_nav_nodes()
        return {'ok': True, 'intent': _dump_model(intent), 'message': 'Localization stopped'}

    if intent.action == 'pause_nav':
        node.cmd_nav_pause()
        return {'ok': True, 'intent': _dump_model(intent), 'message': 'Navigation paused'}

    if intent.action == 'resume_nav':
        node.cmd_nav_resume()
        return {'ok': True, 'intent': _dump_model(intent), 'message': 'Navigation resumed'}

    if intent.action == 'cancel_nav':
        node.cmd_nav_cancel()
        return {'ok': True, 'intent': _dump_model(intent), 'message': 'Navigation cancelled'}

    if intent.action == 'go_pois':
        if _active_map_name() is None:
            return {'ok': False, 'intent': _dump_model(intent), 'message': 'No active map selected'}
        poi_ids = _resolve_poi_ids(intent.poi_names, context)
        with node._lock:
            running = node._nav_nodes_running
            localized = node._localized
        if not running:
            node.cmd_start_nav_nodes()
            return {
                'ok': False,
                'intent': _dump_model(intent),
                'poi_ids': poi_ids,
                'message': 'Localization started. Wait until localized before navigation.',
                'needLocalization': True,
            }
        if not localized:
            return {
                'ok': False,
                'intent': _dump_model(intent),
                'poi_ids': poi_ids,
                'message': 'Not localized yet. Wait until localization succeeds before navigation.',
                'needLocalization': True,
            }
        node.cmd_send_pois(poi_ids)
        return {'ok': True, 'intent': _dump_model(intent), 'poi_ids': poi_ids, 'message': 'Navigation started'}

    raise HTTPException(400, f'Unsupported action: {intent.action}')


@router.post('/parse')
def agent_parse(req: AgentCommandRequest):
    context = _agent_context()
    intent = _exact_match_intent(req.text, context)
    if intent is None:
        intent = _call_llama(req.text, context)
    intent = _normalize_intent(req.text, intent)
    if intent.action == 'go_pois' and not _looks_like_navigation_request(req.text):
        intent = AgentIntent(action='chat', response='I can help with TinyNav maps, POIs, localization, and navigation.')
    intent = _validate_intent(intent, context)
    return {
        'ok': intent.action != 'unknown',
        'intent': _dump_model(intent),
        'context': context,
    }


@router.post('/command')
def agent_command(req: AgentCommandRequest):
    context = _agent_context()
    intent = _exact_match_intent(req.text, context)
    if intent is None:
        intent = _call_llama(req.text, context)
    intent = _normalize_intent(req.text, intent)
    if intent.action == 'go_pois' and not _looks_like_navigation_request(req.text):
        intent = AgentIntent(action='chat', response='I can help with TinyNav maps, POIs, localization, and navigation.')
    intent = _validate_intent(intent, context)
    if intent.action in ('chat', 'current_map', 'list_maps', 'list_pois', 'status'):
        result = _execute_intent(intent, context)
        result['executed'] = False
        return result
    if not req.execute:
        return {
            'ok': intent.action != 'unknown',
            'intent': _dump_model(intent),
            'context': context,
            'executed': False,
        }
    result = _execute_intent(intent, context)
    result['executed'] = result.get('ok', False)
    return result
