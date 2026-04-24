# TinyNav App — Frontend

Flutter Web / Android / iOS app for controlling the TinyNav visual navigation module.

## Quick start

The easiest way is the top-level script (auto-installs Flutter if needed):

```bash
bash /tinynav/scripts/run_web_app.sh
```

Or build and serve manually inside the container:

```bash
cd /tinynav/app/frontend
flutter pub get
flutter build web --release
cd build/web && python3 -m http.server 8080
```

Open `http://<device-ip>:8080` in a browser on the same network.

---

## Development setup (without Docker)

Install Flutter via the tarball or git:

```bash
git clone https://github.com/flutter/flutter.git -b stable --depth 1 ~/flutter
export PATH="$PATH:$HOME/flutter/bin"
flutter config --enable-web
flutter doctor
```

Then:

```bash
cd app/frontend
flutter pub get
flutter build web --release
```

---

## Testing with mock backend (no hardware required)

```bash
cd /tinynav
uv run --with fastapi --with uvicorn app/backend/mock_server.py
```

The mock server listens on port `8000` and simulates all REST and WebSocket endpoints. Enter `localhost` as the device IP in the setup screen.

---

## Widgetbook (UI preview)

Use Widgetbook to preview UI states quickly without connecting to hardware.

```bash
cd app/frontend
flutter pub get
flutter run -t widgetbook/main.dart -d chrome
```

This project currently uses a manual Widgetbook directory setup in `widgetbook/tinynav_widgetbook.dart` with mocked Riverpod providers for:
- `HomePage` (idle / recording)
- `DeviceTab` (idle / map building)

---

## App structure

```
lib/
├── main.dart                  # Entry point — orange theme, switches Setup ↔ Home on IP change
├── core/
│   ├── models.dart            # DeviceStatus, Pose, MapInfo, Poi, PlanningState, TrajPoint, GridInfo
│   └── providers.dart         # Riverpod providers: REST (Dio) + WebSocket streams
└── pages/
    ├── setup_page.dart        # IP input + connection test + save
    ├── home_page.dart         # Menu home — three feature cards, pushes sub-pages
    ├── device_tab.dart        # Device status, bag recording, map build
    ├── map_tab.dart           # Camera panel (1/3) + map/planning view (2/3)
    │                          #   · _MapView       — global map PNG + POI overlay (post-mapping)
    │                          #   · _LocalPlanningView — ESDF / obstacle / trajectory (pre-mapping)
    │                          #   · _CameraPanel   — live camera feed with topic selector
    ├── map_painter.dart       # CustomPainter: robot arrow + POI dots on global map
    ├── planning_painter.dart  # CustomPainter: robot arrow + trajectory on local planning canvas
    └── nav_tab.dart           # Navigate to POI, cancel navigation
```

### Navigation pattern

`HomePage` shows three menu cards (Device, Map, Navigate). Tapping a card pushes the corresponding page onto the navigator stack; the system back button / back arrow returns to the menu. There is no persistent bottom tab bar.

---

## How it connects to the backend

| Protocol | Endpoint | Push rate | Description |
|---|---|---|---|
| HTTP REST | `http://<ip>:8000` | On demand | Commands and one-shot queries |
| WebSocket | `/ws/status` | ~1 s | Full device status |
| WebSocket | `/ws/pose` | Every odometry msg | `{x, y, z, yaw, timestamp, source}` |
| WebSocket | `/ws/map-update` | On map file change | `{event: "map_updated", timestamp}` |
| WebSocket | `/ws/preview` | ~5 fps | JPEG frames for a selected camera topic |
| WebSocket | `/ws/planning` | ~5 fps | ESDF image, obstacle mask, trajectory, localization state |

---

## Map tab layout

```
┌────────────────────────────┐
│   Camera panel  (1 / 3)    │  live video + topic selector (Off / color / left / right / depth)
├────────────────────────────┤
│                            │
│   Map / Planning  (2 / 3)  │  pinch-to-zoom + pan (InteractiveViewer)
│                            │
└────────────────────────────┘
```

**Before a map is built** (`_LocalPlanningView`):
- Background: ESDF heatmap (JET colormap, toggleable)
- Overlay: obstacle mask (semi-transparent, on by default)
- Canvas: planned trajectory polyline + robot arrow (trajectory off by default)
- Layer toggle panel in the top-right corner

**After a map is built** (`_MapView`):
- Background: global occupancy map PNG
- Overlay: ESDF grid centered on the robot (when localized)
- Canvas: robot arrow + POI markers with labels

---

## Typical workflow

1. **Record** a sensor bag → Device tab → Start / Stop
2. **Build** the map → Device tab → Build Map (watch the progress bar)
3. **View** the map and create POIs at target positions → Map tab
4. **Navigate** the robot to a POI → Navigate tab → Go
