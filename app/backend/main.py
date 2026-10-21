"""
TinyNav backend — FastAPI + uvicorn.

Usage:
    cd /tinynav
    TINYNAV_DB_PATH=/tinynav/tinynav_db uv run uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .state import runner
from .routers import action, bag, device, files, nav, sensor
from .routers import map as map_router
from .routers import poi
from . import ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner.start()
    yield
    runner.stop()


app = FastAPI(title='TinyNav API', version='0.1.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(device.router, prefix='/device')
app.include_router(bag.router, prefix='/bag')
app.include_router(map_router.router, prefix='/map')
app.include_router(poi.router)
app.include_router(nav.router, prefix='/nav')
app.include_router(sensor.router)
app.include_router(files.router)
app.include_router(action.router)
app.include_router(ws.router)


@app.middleware('http')
async def protect_manual_mode(request, call_next):
    node = runner.node
    allowed = {'/nav/manual/enter', '/nav/manual/exit', '/nav/cancel', '/action/command'}
    if (node is not None and node._manual_intervention
            and request.method in ('POST', 'PUT', 'DELETE', 'PATCH')
            and request.url.path not in allowed):
        return JSONResponse({'detail': 'Exit manual intervention first'}, status_code=409)
    return await call_next(request)
