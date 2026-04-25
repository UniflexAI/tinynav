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
cd build/web && python3 -m http.server 8080
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
