"""
TinyNav backend — FastAPI + uvicorn.

Usage:
    cd /tinynav
    TINYNAV_DB_PATH=/tinynav/tinynav_db uv run uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .manager_client import BACKEND_ROLE, is_display_role
from .state import runner
from .routers import action, bag, device, files, nav, sensor
from .routers import map as map_router
from .routers import poi
from .routers import proxy
from . import ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner.start()
    yield
    runner.stop()


app = FastAPI(title=f'TinyNav API ({BACKEND_ROLE})', version='0.1.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(device.router, prefix='/device')
app.include_router(sensor.router)
app.include_router(ws.router)

if is_display_role():
    # Jetson display backend owns topic subscriptions/WebSockets and forwards
    # manager-owned HTTP routes to the insight9 manager backend.
    app.include_router(proxy.router)
else:
    app.include_router(bag.router, prefix='/bag')
    app.include_router(map_router.router, prefix='/map')
    app.include_router(poi.router)
    app.include_router(nav.router, prefix='/nav')
    app.include_router(files.router)
    app.include_router(action.router)
