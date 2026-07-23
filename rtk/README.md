# RTK

RTK GNSS for TinyNav: publishes the robot's **position + heading in the saved
map** so navigation can relocalize even when visual relocalization fails (e.g. at
night). Decoupled from the vision stack — it talks to the rest of the system only
through ROS topics and a per-map calibration file, never by importing `tinynav`.

In the container the repo root is mounted at `/tinynav`, so this module lives at
`/tinynav/rtk`.

## Components

| File | Role |
|------|------|
| `rtk_bridge_node.py` | Reads NMEA from the receiver, feeds NTRIP RTCM back, publishes `/fix`, `/rtk/odom`, `/rtk/status`. Runs always; independent of any map. |
| `rtk_align_calibrate.py` | **Calibration** (one-off per map). Pairs `/map/relocalization` with RTK `/fix` and fits a planar Sim3 (map↔ENU) → writes `rtk_align.json` into the map dir. |
| `rtk_map_pose_node.py` | **Runtime.** Turns live `/fix` into `/rtk/map_pose` (position + heading in the map) and drives the init handshake `/rtk/init_status`. |
| `rtk_geo.py` | Shared geodesy + planar-Sim3 math (numpy only). |
| `service/start_rtk_bridge.sh` | Self-heal launcher for the bridge. |
| `service/rtk-bridge.service` | systemd unit to autostart the bridge on the host. |
| `.ntrip.env.example` | NTRIP credential template (copy to `.ntrip.env`, git-ignored). |

## Autostart

The bridge (and thus NTRIP) autostarts via the host systemd unit
`service/rtk-bridge.service`, which runs `service/start_rtk_bridge.sh` inside the
`tinynav` container (same idea as the web app launcher). Install once:

```bash
sudo cp /tinynav/service/rtk-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now rtk-bridge
journalctl -u rtk-bridge -f
```

NTRIP credentials live in `/tinynav/rtk/.ntrip.env` (copy from `.ntrip.env.example`).

---

# Interface for the navigation node

This is what a navigation/control integrator needs. RTK is decoupled: it only
reads/writes topics.

## Topics RTK consumes (inputs)

| Topic | Type | Source | Notes |
|-------|------|--------|-------|
| `/fix` | `sensor_msgs/NavSatFix` | rtk_bridge (internal) | RTK position + status. |
| `/map/current_map` | `std_msgs/String` | **YOU (map/nav owner)** | The active map **directory** path (the same one passed to `map_node` as `--tinynav_map_path`). **Must be latched** (`durability=TRANSIENT_LOCAL`, depth 1) so a late-starting RTK node still gets it. RTK loads `<map_dir>/rtk_align.json` from it (next to `nav_flow.json`). |

## Topics RTK provides (outputs)

### `/rtk/init_status` — `std_msgs/String` (JSON), ~2 Hz, always on
The init handshake. Tells you when to do the one-time yaw-init motion.

```json
{"state":"NEED_YAW_INIT","need_forward_init":true,"have_map":true,
 "map":"map_go_1","fix_ok":true,"navsat_status":2,"yaw_ready":false,"yaw_deg":null}
```

`state` is one of:
| state | meaning | nav node action |
|-------|---------|-----------------|
| `NO_MAP` | no active map, or the map has no `rtk_align.json` | nothing (map not RTK-calibrated) |
| `WAIT_FIX` | map ready, waiting for RTK FIXED/FLOAT (q4/5) | wait |
| `NEED_YAW_INIT` | map + q4/5, but heading not yet known | **drive forward ~1 m** (see below) |
| `ACTIVE` | heading acquired; `/rtk/map_pose` is live | consume `/rtk/map_pose` |

Other fields: `need_forward_init` (bool, = state==NEED_YAW_INIT), `fix_ok`
(RTK q4/5 and recent), `navsat_status` (2 = GBAS_FIX i.e. RTK q4/5, 0 = plain
fix, -1 = none), `yaw_ready`, `yaw_deg` (current heading estimate, debug).

### `/rtk/map_pose` — `nav_msgs/Odometry`, only while `ACTIVE`
The robot's pose in the map frame.

- `header.frame_id = "map"`, `child_frame_id = "rtk"`
- `pose.pose.position` x, y = map-frame position (z = 0)
- `pose.pose.orientation` = yaw about map +Z (motion-fit heading)
- `pose.covariance`: `[0]=[7]=0.25` (x,y, metres²; RTK cm-level), `[14]=1e6`
  (z unknown), `[21]=[28]=1e6` (roll/pitch unknown), `[35]` = yaw variance —
  `(5°)²` right after a heading fit, inflated when the fit goes stale (robot
  stopped / possibly turned in place — unobservable without IMU). **Weight the
  pose by this covariance.**
- Published **only at RTK FIXED/FLOAT (q4/5)** and **only after** the heading is
  acquired.

## What the navigation node must do

1. **Publish `/map/current_map`** (latched `std_msgs/String` = active map dir)
   whenever a map is loaded. Without it RTK stays in `NO_MAP` and never publishes.
2. **Do the yaw-init walk.** While `/rtk/init_status` reports
   `need_forward_init: true` (state `NEED_YAW_INIT`), drive the robot **slowly
   forward ~1 m**, using your own obstacle avoidance. RTK is passive — it never
   sends motion commands. Stop the init motion once state becomes `ACTIVE`.
   (Heading comes from the RTK course-over-ground of that ~1 m; it needs actual
   forward translation, not rotation in place.)
3. **Consume `/rtk/map_pose`** for relocalization once `ACTIVE`, weighting by its
   covariance. Heading keeps refining as the robot drives.

## Data flow

```
receiver ─NMEA─▶ rtk_bridge ─/fix,/rtk/odom,/rtk/status─▶ ...
                                   │
 (calibration, once per map)  map_node ─/map/relocalization─┐
                              /slam/keyframe_odom ──────────┤
                                                            ▼
                                       <map_dir>/rtk_align.json  (lives with the map)
                                                            │
 /fix ───────────▶ rtk_map_pose_node ───────────────────────┴─▶ /rtk/map_pose  (pose in map)
        /map/current_map (you, latched) ─▶ │ ◀── loads <map_dir>/rtk_align.json
                                           └─▶ /rtk/init_status  (handshake: when to walk 1 m)
```

---

# Calibration (one-off per map)

With `map_node` relocalizing in the target map and `/map/current_map` being
published, drive the robot through the map **including turns**, then Ctrl-C:

```bash
uv run python /tinynav/rtk/rtk_align_calibrate.py --ros-args \
    -p map_topic:=/map/current_map
```

Output auto-targets `<map_dir>/rtk_align.json` (map dir learned from the topic),
next to `nav_flow.json`, so the runtime node picks it up automatically. Bag replay
/ no topic: pass `-p out:=<path>` explicitly. `rtk_align.json` is per-map
calibration and lives with the map, not in git.

# Runtime

```bash
uv run python /tinynav/rtk/rtk_map_pose_node.py --ros-args \
    -p map_topic:=/map/current_map
```
Bench without the map topic: `-p align_json:=<map_dir>/rtk_align.json`.
Key params: `heading_min_dist_m` (default 1.0), `yaw_std_deg` (5.0),
`heading_stale_s` (3.0), `min_status` (GBAS_FIX = q4/5), `status_rate_hz` (2.0).
