from fastapi import APIRouter, Request

from ..manager_client import proxy_to_manager

router = APIRouter(tags=['manager-proxy'])


@router.api_route(
    '/{path:path}',
    methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
)
async def proxy_manager_route(request: Request, path: str):
    return await proxy_to_manager(request, path)
