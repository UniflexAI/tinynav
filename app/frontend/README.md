# TinyNav App — Frontend

Flutter Web / Android app for controlling the TinyNav visual navigation module.

## Prerequisites

Install Flutter: https://docs.flutter.dev/get-started/install

Verify the installation:

```bash
flutter doctor
```

You should see Flutter and Chrome (web) marked as OK. Android is not required for web testing.

## First-time setup

Run this once inside the `app/frontend/` directory to generate platform files:

```bash
flutter create . --project-name tinynav_app
flutter pub get
```

## Run in browser (recommended for development)

Make sure the TinyNav backend is running on the device (see `app/backend/README.md`), then:

```bash
flutter run -d chrome
```

A Chrome window will open. Enter the device IP address (e.g. `192.168.1.100`) and press **Connect**.

## Run on a physical Android device

1. Enable **Developer options** and **USB debugging** on the phone.
2. Connect the phone via USB.
3. Verify the device is detected:
   ```bash
   flutter devices
   ```
4. Run:
   ```bash
   flutter run
   ```

> **Note:** The first build takes a few minutes. Subsequent builds are much faster.

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

- **REST**: `http://<device-ip>:8000` — commands and one-shot queries
- **WebSocket**: `ws://<device-ip>:8000/ws/status` — device status pushed every 1 s
- **WebSocket**: `ws://<device-ip>:8000/ws/pose` — robot pose pushed on every odometry message

CORS is already enabled on the backend, so browser access works without extra configuration.

## Typical workflow

1. Record a bag → **Device tab → Start / Stop**
2. Build the map → **Device tab → Build Map** (watch the progress bar)
3. View the map and add POIs at the robot's current position → **Map tab**
4. Send the robot to a POI → **Navigate tab → Go**
