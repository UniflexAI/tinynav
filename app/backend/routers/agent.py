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
    'sit',
    'stand',
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
        'sit',
        'stand',
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
    lower_text = text.lower()
    return any(word.lower() in lower_text for word in words)


def _normalize_intent(text: str, intent: AgentIntent) -> AgentIntent:
    compact_text = text.lower().strip(' \t\r\n?!.')
    if compact_text == 'stop':
        return AgentIntent(action='cancel_nav')
    if compact_text in {'stop localization', 'stop localized', 'stop localize', 'stop_localization'}:
        return AgentIntent(action='stop_localization')
    if compact_text in {'continue', 'go on'}:
        return AgentIntent(action='resume_nav')

    wants_localization = _contains_any(text, ['localize', 'localization', 'start'])
    wants_map_selection = _contains_any(text, ['map', 'select', 'use', 'switch'])
    if (
        intent.action == 'start_localization'
        and intent.map_name is not None
        and wants_map_selection
        and not wants_localization
    ):
        return AgentIntent(action='select_map', map_name=intent.map_name)
    if (
        intent.action == 'select_map'
        and intent.map_name is not None
        and wants_map_selection
        and wants_localization
    ):
        return AgentIntent(action='start_localization', map_name=intent.map_name)
    return intent


def _llm_context(text: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        'maps': context['available_maps'],
        'active_map': context['active_map'],
        'pois': [p['name'] for p in context['pois']],
        'user_text': text,
    }


def _build_prompt(
    text: str,
    context: dict[str, Any],
    correction: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    system = """
Parse TinyNav user text into one JSON object only.
Return valid JSON with double-quoted keys and strings. No markdown. No extra text.
Actions: chat,current_map,list_maps,select_map,start_localization,stop_localization,list_pois,go_pois,pause_nav,resume_nav,cancel_nav,sit,stand,status,unknown.
Fields: action, map_name, poi_names, response, reason.
Use map_name only from maps. Use poi_names only from POIs, copied exactly.
Do not invent POIs. Map semantic phrases to existing POIs: "back home" -> "home" if home exists.
Keep multiple destinations in user order: "go printer and back home" -> ["printer","home"] if both exist.
stop -> cancel_nav. pause/wait -> pause_nav. continue/go on/resume -> resume_nav.
stop localization/stop localized -> stop_localization.
sit/sit down -> sit. stand/stand up -> stand.
hello -> chat. any maps -> list_maps. any POIs -> list_pois.
If selecting a map and starting localization together, use start_localization with map_name.
Information queries do not require matching a specific map or POI.
Use unknown only for unsupported requests, missing requested map, or missing requested POI.
""".strip()
    llm_context = _llm_context(text, context)
    if correction is not None:
        llm_context['previous_output'] = correction.get('raw')
        llm_context['validation_error'] = correction.get('error')
        llm_context['instruction'] = 'Fix the previous output. Return one valid JSON object only.'
    user = json.dumps(
        llm_context,
        ensure_ascii=False,
        separators=(',', ':'),
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
        return {'action': 'unknown', 'reason': f'LLM did not return JSON: {text[:200]}'}
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = _repair_json(candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            return {'action': 'unknown', 'reason': f'Invalid LLM JSON: {e}'}


def _repair_json(text: str) -> str:
    repaired = text.strip().replace("'", '"')
    repaired = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', repaired)
    repaired = re.sub(r':\s*([A-Za-z_][A-Za-z0-9_]*)(\s*[,}])', r':"\1"\2', repaired)

    def quote_array_items(match: re.Match[str]) -> str:
        items = []
        for item in match.group(1).split(','):
            value = item.strip()
            if not value:
                continue
            if not (value.startswith('"') and value.endswith('"')):
                value = f'"{value}"'
            items.append(value)
        return '[' + ','.join(items) + ']'

    return re.sub(r'\[([A-Za-z0-9_\-.\s,]+)\]', quote_array_items, repaired)


def _call_llama(text: str, context: dict[str, Any]) -> AgentIntent:
    raw = _call_llama_raw(text, context)
    return _intent_from_raw(raw, text)


def _call_llama_raw(
    text: str,
    context: dict[str, Any],
    correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        'model': 'tinynav-local-llm',
        'messages': _build_prompt(text, context, correction=correction),
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

    return _extract_json(content)


def _intent_from_raw(raw: dict[str, Any], text: str) -> AgentIntent:
    if 'action' not in raw and 'tool' in raw:
        raw['action'] = raw.get('tool')
    if 'action' not in raw:
        raw = {'action': 'unknown', 'reason': 'LLM response missing action'}
    action = _normalize_action_alias(str(raw.get('action', '')), text)
    raw['action'] = action
    if action not in _ACTIONS:
        raw = {'action': 'unknown', 'reason': f'Unsupported action: {action}'}
    try:
        return AgentIntent(**raw)
    except Exception as e:
        raise HTTPException(502, f'Invalid LLM intent: {e}') from e


def _normalize_action_alias(action: str, text: str) -> str:
    normalized = action.strip().lower().replace('-', '_').replace(' ', '_')
    compact_text = text.lower().strip(' \t\r\n?!.')
    if normalized in {'map', 'maps', 'list_map', 'list_maps', 'available_maps', 'show_maps', 'get_maps'}:
        return 'list_maps'
    if normalized in {'poi', 'pois', 'list_poi', 'list_pois', 'available_pois', 'show_pois', 'get_pois'}:
        return 'list_pois'
    if normalized in {'current_map', 'active_map', 'selected_map'}:
        return 'current_map'
    if normalized in {'state', 'status', 'robot_status'}:
        return 'status'
    if normalized in {'stop', 'halt'}:
        if 'localiz' in compact_text or 'localized' in compact_text:
            return 'stop_localization'
        return 'cancel_nav'
    if normalized in {'stop_localize', 'stop_localized', 'stop_localisation'}:
        return 'stop_localization'
    if normalized in {'continue', 'go_on'}:
        return 'resume_nav'
    return normalized


def _validate_intent(intent: AgentIntent, context: dict[str, Any]) -> AgentIntent:
    maps = set(context['available_maps'])
    poi_names = {p['name'] for p in context['pois']}

    if intent.map_name is not None and intent.map_name not in maps:
        return AgentIntent(action='unknown', reason=f'Map not found: {intent.map_name}')

    if intent.action == 'go_pois':
        # Safety: LLMs tend to over-eagerly interpret casual text as navigation.
        if not intent.poi_names:
            return AgentIntent(action='unknown', reason='No POI selected')
        missing = [name for name in intent.poi_names if name not in poi_names]
        if missing:
            return AgentIntent(action='unknown', reason=f'POI not found: {", ".join(missing)}')

    return intent


def _validate_intent_error(intent: AgentIntent, context: dict[str, Any]) -> str | None:
    if intent.action == 'unknown':
        return intent.reason or 'Unknown action'
    maps = set(context['available_maps'])
    poi_names = {p['name'] for p in context['pois']}
    if intent.map_name is not None and intent.map_name not in maps:
        return f'Map not found: {intent.map_name}'
    if intent.action == 'go_pois':
        if not intent.poi_names:
            return 'No POI selected'
        missing = [name for name in intent.poi_names if name not in poi_names]
        if missing:
            return f'POI not found: {", ".join(missing)}'
    return None


def _parse_intent(text: str, context: dict[str, Any]) -> AgentIntent:
    raw = _call_llama_raw(text, context)
    intent = _normalize_intent(text, _intent_from_raw(dict(raw), text))
    intent = _fallback_query_intent(text, intent)
    error = _validate_intent_error(intent, context)
    if error is None:
        return intent

    retry_raw = _call_llama_raw(text, context, correction={'raw': raw, 'error': error})
    retry_intent = _normalize_intent(text, _intent_from_raw(dict(retry_raw), text))
    retry_intent = _fallback_query_intent(text, retry_intent)
    retry_error = _validate_intent_error(retry_intent, context)
    if retry_error is None:
        return retry_intent
    return AgentIntent(action='unknown', reason=retry_error)


def _fallback_query_intent(text: str, intent: AgentIntent) -> AgentIntent:
    compact_text = text.lower().strip(' \t\r\n?!.')
    if compact_text in {'maps', 'any maps', 'list maps', 'what maps do we have', 'what maps'}:
        return AgentIntent(action='list_maps')
    if compact_text in {'pois', 'any pois', 'list pois', 'what pois do we have', 'what pois'}:
        return AgentIntent(action='list_pois')
    if compact_text in {'current map', 'active map', 'selected map'}:
        return AgentIntent(action='current_map')
    if compact_text in {'status', 'state'}:
        return AgentIntent(action='status')
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

    if intent.action in ('sit', 'stand'):
        node.cmd_action(intent.action)
        return {'ok': True, 'intent': _dump_model(intent), 'message': f'Action sent: {intent.action}'}

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
                'message': 'Localization started, but TinyNav is not localized yet and cannot depart.',
                'needLocalization': True,
            }
        if not localized:
            return {
                'ok': False,
                'intent': _dump_model(intent),
                'poi_ids': poi_ids,
                'message': 'TinyNav is not localized yet and cannot depart.',
                'needLocalization': True,
            }
        node.cmd_send_pois(poi_ids)
        return {'ok': True, 'intent': _dump_model(intent), 'poi_ids': poi_ids, 'message': 'Navigation started'}

    raise HTTPException(400, f'Unsupported action: {intent.action}')


@router.post('/parse')
def agent_parse(req: AgentCommandRequest):
    context = _agent_context()
    intent = _parse_intent(req.text, context)
    return {
        'ok': intent.action != 'unknown',
        'intent': _dump_model(intent),
        'context': context,
    }


@router.post('/command')
def agent_command(req: AgentCommandRequest):
    context = _agent_context()
    intent = _parse_intent(req.text, context)
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
