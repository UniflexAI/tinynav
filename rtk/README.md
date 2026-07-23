# RTK

RTK GNSS for TinyNav. Decoupled from the vision stack — it talks to the rest of
the system only through ROS topics and a small per-map calibration file, never by
importing `tinynav` internals.

In the container the repo root is mounted at `/tinynav`, so this module lives at
`/tinynav/rtk`.

## Components

| File | Role |
|------|------|
| `rtk_bridge_node.py` | Reads NMEA from the receiver, feeds NTRIP RTCM back, publishes RTK topics (`/fix`, `/rtk/odom`, `/rtk/status`, …). Runs always; independent of any map. |
| `rtk_align_calibrate.py` | **Calibration.** Runs alongside `map_node`; pairs `/map/relocalization` with RTK `/fix` and fits a planar Sim3 (map↔ENU), writing `rtk_align.json`. |
| `rtk_map_pose_node.py` | **Runtime.** Loads a map's `rtk_align.json` and turns live `/fix` into the robot's position in that map, published as `/rtk/map_pose`. |
| `rtk_geo.py` | Shared geodesy + planar-Sim3 math (numpy only). |
| `rtk_fusion_node.py` | (Existing) fusion node. |
| `service/start_rtk_bridge.sh` | Autostart + self-heal wrapper for the bridge. |
| `.ntrip.env.example` | NTRIP credential template (copy to `.ntrip.env`, git-ignored). |

## Data flow

```
receiver ──NMEA──▶ rtk_bridge_node ──/fix,/rtk/odom,/rtk/status──▶ ...
                                          │
    map_node ──/map/relocalization────────┤ (calibration)
    /slam/keyframe_odom ──────────────────┘
                                          ▼
                             <map_dir>/rtk_align.json   (per map, lives with the map)
                                          │
   /fix ─────────▶ rtk_map_pose_node ─────┴────▶ /rtk/map_pose  (position in map)
                        ▲
   /map/current_map ────┘  (current map DIR path — published by the map owner)
```

`rtk_align.json` is calibration data tied to one map build, so it lives **in the
map directory** (e.g. `<tinynav_map_path>/rtk_align.json`), next to
`nav_flow.json` — not in git.

## Map gating — integration contract (⚠ needs the map owner)

`rtk_map_pose_node` does not hard-code a map. It subscribes to `map_topic`
(default `/map/current_map`) and loads `<map_dir>/rtk_align.json` from whatever
directory that topic carries. It publishes `/rtk/map_pose` **only** while a map
with an `rtk_align.json` is active; a map without one (not RTK-calibrated) leaves
it silent.

Publishing `/map/current_map` is **the map/navigation owner's responsibility,
not part of the RTK module.** That publisher must:

- publish a `std_msgs/String` whose `data` is the map **directory** (the same
  path given to `map_node` as `--tinynav_map_path`), and
- use a **latched** QoS (`durability = TRANSIENT_LOCAL`, depth 1) so this node
  receives the current map even when it starts late.

Until that topic exists, use the bench bypass `-p align_json:=<path>` to load a
fixed `rtk_align.json` directly.

## Quick start

```bash
# 1. NTRIP creds (once)
cp rtk/.ntrip.env.example rtk/.ntrip.env && chmod 600 rtk/.ntrip.env   # edit it

# 2. Bridge (autostart / self-heal) — always on, no map needed
docker exec tinynav-dev tmux new-session -d -s rtk /tinynav/service/start_rtk_bridge.sh
# verify (odom needs RTK FIXED/FLOAT):  ros2 topic hz /rtk/odom

# 3. Calibrate a map (with map_node relocalizing in it), drive with turns, Ctrl-C:
uv run python /tinynav/rtk/rtk_align_calibrate.py --ros-args \
    -p map_name:=<map_name> \
    -p out:=<tinynav_map_path>/rtk_align.json

# 4. Runtime position-in-map (topic-gated on the current map)
uv run python /tinynav/rtk/rtk_map_pose_node.py --ros-args \
    -p map_topic:=/map/current_map
# -> /rtk/map_pose (nav_msgs/Odometry, map frame). Position only; covariance
#    reflects RTK quality (tight at FIXED, loose at DGNSS). Orientation TODO
#    (single antenna): to be filled from VIO / motion-fit heading.
```
