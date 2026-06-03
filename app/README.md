# TinyNav App

Mobile / web control interface for the TinyNav visual navigation module.

## Quick start

### Backend

```bash
cd /tinynav
TINYNAV_DB_PATH=/tinynav/tinynav_db uv run uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd /tinynav/app/frontend
flutter pub get
flutter build web --release
# Then serve build/web/ on your preferred port, e.g.:
cd build/web && uv run python -m http.server 8080
```

## Device Startup: Jetson + insight9

In the deployed robot setup, the backend runs on insight9 and the frontend runs in
the Jetson devcontainer:

- insight9 backend: `169.254.10.1:8000`
- Jetson frontend: `http://192.168.123.220`
- Jetson backend proxy: `192.168.123.220:8000 -> 169.254.10.1:8000`

Start the backend from Jetson:

```bash
ssh dm@192.168.123.220
ssh root@169.254.10.1

cd /userdata/junlinp/tinynav
tmux kill-session -t backend 2>/dev/null || true
tmux new-session -d -s backend \
  "bash /userdata/junlinp/tinynav/run_backend.sh >> /userdata/junlinp/logs/backend_debug.log 2>&1"
```

Verify the backend from Jetson:

```bash
curl http://169.254.10.1:8000/device/status
curl http://169.254.10.1:8000/sensor/mode
curl http://169.254.10.1:8000/sensor/image-topics
```

For insight9, the expected sensor mode is `looper`, and the color stream should be
`/camera/camera/color/image_rect_raw/compressed`.

Start the Jetson devcontainer and frontend:

```bash
cd ~/workspace/junlinp/tinynav
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash

cd /tinynav/app/frontend
flutter pub get
flutter build web --release

cd /tinynav
bash scripts/start_app.sh
```

`scripts/start_app.sh` serves the frontend on port `80` and starts the proxy from
Jetson port `8000` to insight9 port `8000`.

Enable navigation nodes after the backend and frontend are running:

```bash
curl -X POST http://127.0.0.1:8000/nav/nodes/enable
curl http://127.0.0.1:8000/device/status
```

Useful verification commands:

```bash
curl http://192.168.123.220:8000/device/status
ssh root@169.254.10.1 'pgrep -af "uvicorn|looper_bridge_node|planning_node|map_node"'
ssh root@169.254.10.1 'tail -f /userdata/junlinp/logs/backend_debug.log'
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `TINYNAV_DB_PATH` | `<repo>/tinynav_db` | Root path for bag, map, and nav data |
| `BACKEND_PORT`   | `8000` | Override the backend port |
| `FRONTEND_PORT`  | `8080` | Override the frontend port |

---

## Sub-projects

| Directory | Description |
|---|---|
| [`backend/`](backend/README.md) | FastAPI server — REST + WebSocket bridge to ROS 2 |
| [`frontend/`](frontend/README.md) | Flutter web / mobile app |

---

## Architecture overview

```
┌─────────────────────┐       HTTP / WebSocket       ┌──────────────────────┐
│  Flutter Web App    │ ◄──────────────────────────► │  FastAPI Backend     │
│  (port 8080)        │                              │  (port 8000)         │
└─────────────────────┘                              └────────┬─────────────┘
                                                              │ rclpy spin thread
                                                    ┌─────────▼────────────┐
                                                    │  ROS 2 / TinyNav     │
                                                    │  (map_node, planning │
                                                    │   node, perception…) │
                                                    └──────────────────────┘
```

The backend runs a `BackendNode` (a ROS 2 node) in a background thread alongside the async FastAPI event loop. All sensor data, poses, and planning state are forwarded to the Flutter app over WebSocket.
