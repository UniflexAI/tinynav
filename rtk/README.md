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
| `rtk_bridge_node.py` | Reads NMEA from the receiver, feeds NTRIP RTCM back, publishes `/fix`, `/rtk/odom`, `/rtk/status`, `/rtk/io_status`, `/rtk/receiver_status`. Runs always; independent of any map. |
| `rtk_align_calibrate.py` | **Calibration** (one-off per map). Pairs `/map/relocalization` with RTK `/fix` and fits a **local-weighted Sim3** (map↔ENU) → writes `rtk_align.json` into the map dir. |
| `rtk_map_pose_node.py` | **Runtime.** Turns live `/fix` into `/rtk/map_pose` (position + heading in the map) and drives the init handshake `/rtk/init_status`. |
| `rtk_heading.py` | Heading maintenance for the runtime node: `StraightYaw` (course-over-ground, emitted only while the track is straight) + `HeadingFilter` (complementary filter that fuses gyro/odom rate with those RTK observations). Pure numpy, unit-testable offline. |
| `rtk_geo.py` | Shared geodesy + Sim3 math, global and local-weighted (numpy only). |
| `scripts/run_rtk.sh` | Self-heal launcher for the bridge (repo-root `scripts/`, alongside `run_navigation.sh`). |
| `record_traj.sh` | Records a calibration bag (the topics `rtk_align_calibrate.py` needs). |
| `.ntrip.env.example` | NTRIP credential template (copy to `.ntrip.env`, git-ignored). |

## Autostart

The robot already autostarts on boot via the host systemd unit
`tinynav_start.service` (in `/etc/systemd/system/` on the dog — a host file, not
in this repo): it restarts the `tinynav-dev` container and `docker exec`s the
web-app launcher. The RTK bridge reuses **the same** unit — add one line after
the app's `ExecStart` (the `-` prefix makes an RTK failure not block the app):

```ini
ExecStart=-/usr/bin/docker exec -itd tinynav-dev /tinynav/scripts/run_rtk.sh
```

then `sudo systemctl daemon-reload` on the host. Because the unit lives on the
host it survives container restart/rebuild; `run_rtk.sh` ships in this repo (git),
so it survives too. Manual run with logs:
`docker exec -it tinynav-dev /tinynav/scripts/run_rtk.sh`.

NTRIP credentials live in `/tinynav/rtk/.ntrip.env` (copy from `.ntrip.env.example`).

### `TINYNAV_NTRIP_INITIAL_GGA` — set it, the default is wrong for you

The caster is a **VRS**: it generates corrections for whatever position the client
reports. Before the receiver has its own fix (i.e. every cold start indoors) the
bridge has nothing to report, so it falls back to `ntrip_initial_gga` — whose
built-in default is a **Shenzhen** sentence. Boot the bridge indoors in Beijing
and the caster serves corrections for a base ~1900 km away until the receiver
gets a position on its own.

Put a sentence for *your* site in `.ntrip.env`:

```bash
TINYNAV_NTRIP_INITIAL_GGA='$GNGGA,000000.00,3947.00929,N,11633.66606,E,1,12,1.0,24.2,M,-8.3,M,,*5F'
```

The NMEA checksum (after `*`) is XOR of every character between `$` and `*` — a
wrong one makes the caster ignore the sentence silently. Note the **single
quotes**: `$GNGGA` would otherwise be expanded to the empty string by the shell,
because `run_rtk.sh` sources this file.

`latest_position_gga` is never cleared once set, so a bridge that has already had
a fix keeps reporting the last real position — this default only bites on a cold
start. `/rtk/io_status` reports which one is in use as `ntrip_gga_source`
(`live` or `initial`).

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
  `yaw_base_std_deg²` (default 3°) right after an RTK heading observation, then
  growing at `yaw_drift_deg_per_s` (default 0.05°/s) for as long as the filter is
  coasting on rate alone. **Weight the pose by this covariance.**

**Heading survives spot turns.** A single antenna gives position, not heading, and
course-over-ground dies the moment the robot stops translating — which is exactly
what it does to turn in place or reverse. `rtk_heading.py` therefore runs a
complementary filter: an angular-rate source (odom by default, `/lidar_imu` gyro
if `heading_source:=imu`) carries the heading through turns, while `StraightYaw`
supplies drift-free absolute corrections **only** when the recent track is
straight enough (`straight_max_offline_m`, default 0.15 m over
`heading_min_dist_m`, default 1.0 m). A curved or rotating path is rejected
outright rather than force-fitted into a bogus direction.
- Published **only at RTK FIXED/FLOAT (q4/5)** and **only after** the heading is
  acquired.

### Diagnostics — `std_msgs/String` (JSON), 1 Hz each, not part of the interface

Nothing should *depend* on these; they exist so a human can tell why RTK is not
doing what was expected.

| Topic | Carries |
|-------|---------|
| `/rtk/status` | The fix as the bridge sees it: `gga_quality`, `latitude`/`longitude`/`altitude`, `enu`, `num_satellites`, `hdop`, `last_gga_age_s`, `last_rtcm_age_s`, `fix_stale`, `accepted`. |
| `/rtk/io_status` | The plumbing: `ntrip_connected`, `ntrip_gga_source` (`live`/`initial`), `rtcm_bytes` / `rtcm_written_bytes` / `rtcm_dropped_bytes`, `nmea_checksum_fail_count`. This is where you look when corrections are not reaching the receiver. |
| `/rtk/receiver_status` | Vendor decode: `receiver_stage`, `receiver_position_type`, `bestnav_*`, Unicore log counters. |

`last_rtcm_age_s` only advances on RTCM that was **successfully written to the
receiver**, so it goes stale even while the caster keeps streaming — that is the
field that tells you corrections are being received but not delivered.

## What the navigation node must do

1. **Publish `/map/current_map`** (latched `std_msgs/String` = active map dir)
   whenever a map is loaded. Without it RTK stays in `NO_MAP` and never publishes.
2. **Do the yaw-init walk.** While `/rtk/init_status` reports
   `need_forward_init: true` (state `NEED_YAW_INIT`), drive the robot **slowly
   forward ~1 m**, using your own obstacle avoidance. RTK is passive — it never
   sends motion commands. Stop the init motion once state becomes `ACTIVE`.
   The **first** heading needs real forward translation, not rotation in place —
   turning on the spot yields no course-over-ground to seed the filter. Once
   `ACTIVE`, spot turns are fine (see the heading note above).
   A yaw sweep aimed at *visual* relocalization is useless here and the web
   backend suppresses it in `replace` mode — see below.
3. **Consume `/rtk/map_pose`** for relocalization once `ACTIVE`, weighting by its
   covariance. Heading keeps refining as the robot drives.

## `replace` mode — letting RTK own localization outright

The above describes RTK as an *advisory* pose source. In practice a map is
usually switched to **`replace` mode**, where RTK is the only localizer and the
visual path is turned off. That is a per-map switch, in `nav_flow.json` next to
`rtk_align.json`:

```json
{
  "rtk": { "mode": "replace" }
}
```

Absent or `"off"` → unchanged VIO behaviour. Accepted as "on":
`replace`, `on`, `true`, `1`, `yes`. A bare `"rtk": true` also means `replace`.
Anything else logs a warning and falls back to `off`.

What flips when the mode is `replace`:

| Component | Behaviour |
|-----------|-----------|
| `map_node` | Consumes `/rtk/map_pose` and derives `T_from_map_to_odom` from it. **Visual relocalization never runs** — `keyframe_callback` returns early, so no competing `/map/relocalization` and no map rotation. Mapping itself still runs. |
| `app/backend` | Localization assist (the yaw-sweep that exists to bootstrap *visual* reloc) is not started, and is stopped if it was already running. The `/nav/loc-assist` endpoint reports the **effective** state, so the UI toggle cannot claim to be on while it is suppressed. |

**Consequence worth knowing:** in `replace` mode there is currently no fallback.
If RTK stops (loss of fix, serial drop), the visual path does *not* take over —
the last transform simply freezes and the robot keeps navigating on drifting
odometry, without an alarm. Do not run `replace` mode unattended where a
sustained RTK outage is plausible.

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

### Why the fit is local-weighted, and what that means for the file

A single global Sim3 cannot absorb VIO map warp: on a ~150 m route the map's own
scale and heading drift, so one rigid transform leaves ~1.5 m of residual. The
runtime therefore does a **Gaussian-weighted Sim3 refit around each query point**
(`bw_m` 5 m, at least `min_neighbors` correspondences, widening to `bw_max_m`),
which brings it to ~0.2 m. The global fit is kept only as the fallback for
queries with too few neighbours.

So the file is not just four numbers — it carries the correspondence cloud:

```json
{
  "model": "local_weighted_sim3",
  "origin_lla": {"lat": ..., "lon": ..., "alt": ...},
  "sim3":  {"yaw_deg": ..., "scale": ..., "tx": ..., "ty": ...},
  "local": {"bw_m": 5.0, "min_neighbors": 15, "bw_max_m": 40.0},
  "points": {"map_xy": [[x, y], ...], "enu_xy": [[e, n], ...]},
  "fit":    {"rmse_m": ..., "n_local_points": ..., "gga_quality_kept": [4]}
}
```

**The invariant that matters:** `points.enu_xy` must be derived from the very
`origin_lla` written into the same file. At runtime the node converts raw
`/fix` lat/lon to ENU using `origin_lla` and looks that up against `enu_xy`; if
the two were built from different origins, every query lands off by the distance
between the origins. This has bitten us once, for 10.8 m — the robot localized
onto a wall while standing in the middle of the road. Leave-one-out
cross-validation **cannot** catch it, because it stays inside `enu_xy → map_xy`
and is self-consistent either way. Only a check that starts from the raw lat/lon
exposes it.

# Runtime

```bash
uv run python /tinynav/rtk/rtk_map_pose_node.py --ros-args \
    -p map_topic:=/map/current_map
```
Bench without the map topic: `-p align_json:=<map_dir>/rtk_align.json`.

Key params:

| Param | Default | Meaning |
|-------|---------|---------|
| `heading_source` | `odom` | Rate source carrying heading through turns; `imu` uses `imu_topic`. |
| `imu_topic` | `/lidar_imu` | Gyro source when `heading_source:=imu`. |
| `odom_topic` | `/slam/odometry` | Rate source when `heading_source:=odom`. |
| `heading_min_dist_m` | 1.0 | Track window length for a course-over-ground observation. |
| `straight_max_offline_m` | 0.15 | Straightness gate over that window; a curved track is rejected, not fitted. |
| `yaw_base_std_deg` | 3.0 | Yaw σ right after an accepted RTK heading observation. |
| `yaw_drift_deg_per_s` | 0.05 | Rate at which that σ grows while coasting. |
| `min_status` | `GBAS_FIX` | Minimum fix quality to publish (q4/5). |
| `fix_timeout_s` | 2.0 | Fix older than this counts as lost. |
| `status_rate_hz` | 2.0 | `/rtk/init_status` publish rate. |

## Known issues

- **The bridge does not reopen the serial port.** `_open_serial()` runs once at
  startup and there is no reopen path. If the USB-serial adapter re-enumerates
  (the CH340 has done this mid-run: `ttyUSB0` disconnected and came back as
  `ttyUSB4`), the process keeps its file descriptor on the now-deleted device:
  reads return EOF forever without raising, so nothing is logged on that side,
  and writes fail as `Serial write timeout, dropped N/N RTCM bytes` — which reads
  like congestion but means the device is gone. The by-id symlink protects the
  `open()`, not an fd already open. Symptom: `last_gga_age_s` and
  `last_rtcm_age_s` climb together forever while `seq` keeps incrementing.
  Recovery today is a bridge restart.
- **`replace` mode has no fallback** if RTK drops — see the replace-mode section.
- Degenerate GGAs (`num_satellites: 0`, `hdop: 9999`) are currently reported as
  `accepted`, so a coasted receiver solution can look like a real DGNSS fix.
