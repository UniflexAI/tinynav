# TinyNav App

Mobile / web control interface for the TinyNav visual navigation module.

## Quick start

The fastest way to build and run everything is the provided shell script:

```bash
bash /tinynav/scripts/run_web_app.sh
```

What it does:

1. **Starts the backend** — launches FastAPI/uvicorn on port `8000` (sources ROS 2 Humble if available).
2. **Serves the frontend** — serves `app/frontend/build/web/` on port `8080` with COOP/COEP headers for SharedArrayBuffer support.
3. **Prints the device IP** and waits; `Ctrl+C` shuts both processes down cleanly.

> **Note:** You must build the frontend first: `cd app/frontend && flutter build web --release`

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `TINYNAV_DB_PATH` | `<repo>/tinynav_db` | Root path for bag, map, and nav data |
| `BACKEND_PORT`   | `8000` | Override the backend port |
| `FRONTEND_PORT`  | `8080` | Override the frontend port |

Example with overrides:

```bash
TINYNAV_DB_PATH=/data/mydb FRONTEND_PORT=9090 bash /tinynav/scripts/run_web_app.sh
```

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
                                                    ┌────────▼─────────────┐
                                                    │  ROS 2 / TinyNav     │
                                                    │  (map_node, planning │
                                                    │   node, perception…) │
                                                    └──────────────────────┘
```

The backend runs a `BackendNode` (a ROS 2 node) in a background thread alongside the async FastAPI event loop. All sensor data, poses, and planning state are forwarded to the Flutter app over WebSocket.
