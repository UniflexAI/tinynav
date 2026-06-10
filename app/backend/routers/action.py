from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import runner

router = APIRouter(prefix='/action', tags=['action'])

_ALLOWED = {'sit', 'stand'}


class ActionRequest(BaseModel):
    command: str


@router.post('/command')
def send_command(req: ActionRequest):
    if req.command not in _ALLOWED:
        raise HTTPException(400, f'Unknown command: {req.command}. Allowed: {_ALLOWED}')
    node = runner.node
    if node is None:
        raise HTTPException(503, 'Node not ready')
    node.cmd_action(req.command)
    return {'ok': True, 'command': req.command}
