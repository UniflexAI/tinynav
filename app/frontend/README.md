# TinyNav App — Frontend

Flutter Web / Android app for controlling the TinyNav visual navigation module.

## Quick start (Docker — recommended)

Flutter is pre-installed in the TinyNav Docker image. Inside the container:

```bash
# Build the web app (first time or after source changes)
cd /tinynav/app/frontend
flutter build web

# Serve it
cd build/web
python3 -m http.server 8080
```

Then open `http://<device-ip>:8080` in a browser on the same network.

---

## Development setup (without Docker)

### Prerequisites

Install Flutter via git (works on x86_64 and ARM64):

```bash
git clone https://github.com/flutter/flutter.git -b stable --depth 1 ~/flutter
export PATH="$PATH:$HOME/flutter/bin"
git config --global --add safe.directory ~/flutter
flutter config --enable-web
flutter doctor
```

### Install dependencies

```bash
cd app/frontend
flutter pub get
```

### Build and serve

```bash
flutter build web
cd build/web && python3 -m http.server 8080
```

---

## Testing with mock backend (no hardware required)

Run the mock server on the host — no ROS2 needed:

```bash
cd /tinynav   # or ~/workspace/.../tinynav
uv run --with fastapi --with uvicorn app/backend/mock_server.py
```

Or with pip:

```bash
pip install fastapi uvicorn
python3 app/backend/mock_server.py
```

The mock server listens on port `8000` and simulates all API endpoints and WebSocket streams (device status, pose, map image, POIs, navigation). Enter `localhost` as the device IP in the app.

---

## App structure

```
lib/
├── main.dart              # Entry point — watches device IP, switches pages automatically
├── core/
│   ├── models.dart        # Data models: DeviceStatus, Pose, MapInfo, Poi
│   └── providers.dart     # Riverpod providers (REST + WebSocket)
└── pages/
    ├── setup_page.dart    # IP input + connection test
    ├── home_page.dart     # Main shell with bottom navigation
    ├── device_tab.dart    # Device status, bag recording, map build
    ├── map_tab.dart       # Map viewer + POI management
    ├── map_painter.dart   # CustomPainter: overlays robot pose and POIs on map image
    └── nav_tab.dart       # Navigate to POI, cancel navigation
```

## How it connects to the backend

| Protocol | Endpoint | Purpose |
|---|---|---|
| HTTP REST | `http://<ip>:8000` | Commands and one-shot queries |
| WebSocket | `ws://<ip>:8000/ws/status` | Device status pushed every 1 s |
| WebSocket | `ws://<ip>:8000/ws/pose` | Robot pose on every odometry message |
| WebSocket | `ws://<ip>:8000/ws/map-update` | Notification when map file changes |

CORS is enabled on the backend — browser access works without extra configuration.

## Typical workflow

1. Record a bag → **Device tab → Start / Stop**
2. Build the map → **Device tab → Build Map** (watch the progress bar)
3. View the map and add POIs at the robot's current position → **Map tab**
4. Send the robot to a POI → **Navigate tab → Go**
