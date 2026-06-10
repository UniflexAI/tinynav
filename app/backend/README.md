# TinyNav Backend

FastAPI server that exposes the TinyNav robot stack over HTTP and WebSocket.
It wraps `Ros2NodeManager` (a ROS 2 node) and runs `rclpy` in a background
thread alongside the async FastAPI/uvicorn event loop.

## Requirements

In addition to the project's existing dependencies, the backend needs:

```
fastapi
uvicorn[standard]
pillow
```

Install them with:

```bash
uv add fastapi "uvicorn[standard]" pillow
```

## Running

```bash
TINYNAV_DB_PATH=/tinynav/tinynav_db uv run uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

The interactive API docs are available at `http://<host>:8000/docs`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `TINYNAV_DB_PATH` | `/tinynav/tinynav_db` | Root path for bag, map, and nav output data |

## REST endpoints

### Device

| Method | Path | Description |
|---|---|---|
| `GET` | `/device/info` | Device ID, firmware version, capabilities |
| `GET` | `/device/status` | Online flag, bag / map / nav status, mapping percent |

### Bag recording

| Method | Path | Description |
|---|---|---|
| `POST` | `/bag/start` | Start RealSense bag recording |
| `POST` | `/bag/stop` | Stop bag recording |
| `GET` | `/bag/status` | Recording status and whether a bag file exists |

### Map

| Method | Path | Description |
|---|---|---|
| `POST` | `/map/build` | Trigger map build from the recorded bag |
| `GET` | `/map/current` | Map metadata (image URL, resolution, origin, size) |
| `GET` | `/map/image` | Occupancy grid rendered as a PNG image |

### Points of interest (POI)

| Method | Path | Description |
|---|---|---|
| `GET` | `/map/pois` | List all POIs |
| `POST` | `/map/pois` | Create a POI `{"name": "…", "position": [x, y, z]}` |
| `DELETE` | `/poi/{id}` | Delete a POI by integer ID |

### Navigation

| Method | Path | Description |
|---|---|---|
| `POST` | `/nav/go-to-poi` | Start navigation `{"poi_id": 0}` |
| `POST` | `/nav/cancel` | Cancel active navigation |
| `GET` | `/nav/status` | Navigation status |

## WebSocket endpoints

| Path | Push rate | Payload |
|---|---|---|
| `/ws/status` | ~1 s | Full device status JSON |
| `/ws/pose` | On every odometry message | `{x, y, z, yaw, timestamp, source}` |
| `/ws/map-update` | On map file change | `{event: "map_updated", timestamp}` |
| `/ws/preview?topic=<topic>` | ~5 fps | Raw JPEG bytes for the selected camera topic |
| `/ws/planning` | ~5 fps | `{localized, odom_pose, map_pose, esdf_image, obstacle_image, trajectory, grid_info}` |

### Sensor / camera

| Method | Path | Description |
|---|---|---|
| `GET` | `/sensor/mode` | Detected sensor mode: `looper`, `realsense`, or `unknown` |
| `GET` | `/sensor/image-topics` | List of available camera topics |

## File structure

```
app/backend/
├── main.py           # FastAPI application + lifespan
├── state.py          # NodeRunner singleton
├── node_manager.py   # BackendNode (extends Ros2NodeManager) + NodeRunner
├── map_renderer.py   # occupancy_grid.npy → PNG
├── ws.py             # WebSocket handlers
└── routers/
    ├── device.py
    ├── bag.py
    ├── map.py
    ├── poi.py
    └── nav.py
```

## Architecture notes

- **ROS 2 integration**: `BackendNode` inherits `Ros2NodeManager` and adds
  subscriptions for `/mapping/percent`, `/slam/odometry`, and
  `/mapping/current_pose_in_map`. `NodeRunner` spins the node in a daemon
  thread so the asyncio event loop is never blocked.
- **Thread safety**: pose callbacks use `loop.call_soon_threadsafe` to hand
  data from the rclpy spin thread to the asyncio event loop.
- **Map rendering**: the occupancy grid (3-D numpy array, values 0/1/2) is
  projected along the Z axis and saved as a grayscale PNG. Free space is light
  gray, occupied cells are dark, unknown cells are medium gray.
- **POI storage**: POIs are read from and written to `{map_path}/pois.json`,
  which is the same format used by `tool/poi_editor.py`.
