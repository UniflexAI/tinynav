#!/usr/bin/env python3
"""Scripted obstacle scenes + goal publishing for the gz (Fortress) sim.

Reference: star-core modules/simulation/scripts/auto/auto_sim_node.py, adapted
for Ignition Fortress, which has no gz-transport Python bindings -- entity
create/remove/set_pose go through the `ign service` CLI instead.

Prereq: the sim stack is already up (bash scripts/run_simulator.sh), with the
gz server running a world that loads gz-sim-user-commands-system (every
world under worlds/ does).

Usage:
    uv run python tool/simulator/gazebo_scene/scene_runner.py l_corridor
    uv run python tool/simulator/gazebo_scene/scene_runner.py config/scenes/l_corridor.json --spawn-only

Scene JSON (see config/scenes/l_corridor.json):
    robot   : {name, pose[x,y,z,roll,pitch,yaw]} -- set_pose target at reset
    models  : [{name, type, pose} | {line: {name, type, from, to, spacing, yaw}}]
              type indexes config/obstacles.json; "line" expands to a row of models
              entries are packed into one static model and spawned with a
              single /create call; set "batch": false on an entry to spawn it
              individually (required if it must move independently later)
    targets : [{name, pose[x,y,z], tolerance, timeout}]  -- published in order
              to /sim/target_pose_gz (gazebo world coords); sim_gt_reloc
              transforms them into the SLAM frame on /control/target_pose.
              sequencing only, no pass/fail evaluation
    path    : {points: [[x,y],...], lookahead, tolerance, timeout, rate}
              alternative to targets: like map_node's nav_target_timer, the
              goal is re-published every cycle as the point `lookahead` meters
              ahead of the robot's projection along the polyline (map_node uses
              max_speed*5 = 2.5m), until the robot reaches the last point

Coordinate convention: scene coordinates == gazebo world frame. With reloc
(default), progress/reached checks run against the gz ground truth
(/world/$WORLD/pose/info) and sim_gt_reloc keeps the published goal glued to
the SLAM frame as it drifts; zeros are published on /cmd_vel during the reset
so a stale diff-drive twist cannot coast, then the start pose is
burst-published once as a hold target. With --no-reloc (map_node mode) goals
go straight to /control/target_pose in the caller's frame -- the raw SLAM
frame -- and progress falls back to /slam/odometry_visual; the caller is
responsible for a consistent anchor.
"""

import os
import argparse
import json
import math
import pathlib
import subprocess
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty
from tf2_msgs.msg import TFMessage

from tinynav.core.math_utils import tf2np

SCENE_ROOT = pathlib.Path(__file__).resolve().parent
MODEL_DB = json.load(open(SCENE_ROOT / "config/obstacles.json"))
SPAWN_STATE = pathlib.Path("/tmp/tinynav_gazebo_scene_spawned.json")
TMP_SDF_DIR = pathlib.Path("/tmp/tinynav_gazebo_scene_models")

# Same camera->robot rotation as platforms/simulator_control.py
T_ROBOT_TO_CAMERA = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return qx, qy, qz, qw


def run_cli(cmd, timeout=15):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def discover_world():
    """World name from the pose/info topic; the sim may run any world file."""
    code, out, _ = run_cli(["ign", "topic", "-l"])
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("/world/") and line.endswith("/pose/info"):
            return line.split("/")[2]
    return None


def ign_service(service, reqtype, req):
    code, out, err = run_cli([
        "ign", "service", "-s", service,
        "--reqtype", reqtype,
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "3000",
        "--req", req,
    ])
    ok = "data: true" in out
    return ok, out or err


def build_material(texture, color):
    if texture:
        tex_path = SCENE_ROOT / "models/textures" / texture
        return f"""<diffuse>1 1 1 1</diffuse>
                    <specular>0.05 0.05 0.05 1</specular>
                    <pbr><metal>
                        <albedo_map>file://{tex_path}</albedo_map>
                        <roughness>0.9</roughness>
                        <metalness>0.0</metalness>
                    </metal></pbr>"""
    r, g, b = color
    return f"""<ambient>{r} {g} {b} 1</ambient>
                    <diffuse>{r} {g} {b} 1</diffuse>
                    <specular>0.1 0.1 0.1 1</specular>"""


def build_link_xml(entry, idx=0):
    """One link per obstacle, posed inside the (origin-anchored) pack model."""
    name, mtype, pose = entry["name"], entry["type"], entry["pose"]
    spec = MODEL_DB[mtype]
    sx, sy, sz = spec["size"]
    texture = spec.get("texture")
    if "texture_prefix" in spec:
        # rotate variants so neighboring crates never share an identical texture
        texture = f"{spec['texture_prefix']}_{idx % spec['texture_variants']}.png"
    geometry = f"<box><size>{sx} {sy} {sz}</size></box>"
    pose_xml = " ".join(str(v) for v in pose)
    return f"""<link name="{name}">
            <pose>{pose_xml}</pose>
            <collision name="collision">
                <geometry>{geometry}</geometry>
            </collision>
            <visual name="visual">
                <geometry>{geometry}</geometry>
                <material>
                    {build_material(texture, spec.get("color"))}
                </material>
            </visual>
        </link>"""


def build_pack_sdf(name, entries):
    """All obstacles as links of one static model: a single /create call
    spawns the whole scene instead of one service round-trip per crate."""
    links = "\n        ".join(build_link_xml(e, i) for i, e in enumerate(entries))
    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
    <model name="{name}">
        <static>true</static>
        <pose>0 0 0 0 0 0</pose>
        {links}
    </model>
</sdf>
"""


def build_model_sdf(name, mtype):
    return build_pack_sdf(name, [{"name": "link", "type": mtype, "pose": [0.0] * 6}])


def expand_models(model_entries):
    """Expand {line: {...}} shorthand into individual model entries."""
    out = []
    for entry in model_entries:
        if "line" not in entry:
            out.append(entry)
            continue
        line = entry["line"]
        spec = MODEL_DB[line["type"]]
        x0, y0 = line["from"]
        x1, y1 = line["to"]
        length = math.hypot(x1 - x0, y1 - y0)
        # ceil: actual spacing never exceeds the requested one, so crates always
        # overlap slightly (round() could leave gaps wider than the grid cell)
        n = max(1, int(math.ceil(length / line["spacing"])) + 1)
        yaw = line.get("yaw", 0.0)
        for i in range(n):
            t = i / max(1, n - 1)
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            out.append({
                "name": f"{line['name']}_{i}",
                "type": line["type"],
                "pose": [x, y, spec["size"][2] / 2.0, 0.0, 0.0, yaw],
            })
    return out


class AutoNavSim(Node):
    def __init__(self, args):
        super().__init__("scene_runner")
        self.args = args
        # model name in gz (the launcher exports it; lekiwi is the default rig)
        self.robot_name = os.environ.get("TINYNAV_ROBOT_MODEL", "lekiwi")
        # reloc mode: goals are authored in gz coords and sim_gt_reloc
        # transforms them into the SLAM frame; without reloc the caller's
        # coordinates go to /control/target_pose unchanged.
        target_topic = "/sim/target_pose_gz" if not args.no_reloc else "/control/target_pose"
        self.pub_target = self.create_publisher(Odometry, target_topic, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        # VIO backend reset: reset_robot teleports the rig, which poisons the
        # perception window unless it re-anchors (see /slam/reset in
        # tinynav/core/perception_node.py)
        self.pub_vio_reset = self.create_publisher(Empty, "/slam/reset", 10)
        self.latest_out = None  # reloc-transformed goal echoed back from /control/target_pose
        if not args.no_reloc:
            self.create_subscription(Odometry, "/control/target_pose", self.out_cb, 10)
        self.latest_center = None  # robot center xy in the goal frame
        self.latest_yaw = None     # robot yaw in the goal frame (gt)
        self._zero_cmd_stop = threading.Event()
        self._zero_cmd_thread = None
        self.world = None

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        # Only translation is needed; robot center = R * t_r2c + p, and
        # T_ROBOT_TO_CAMERA has zero translation, so center == p here. Keep the
        # same convention as simulator_control for future offset changes.
        # Used only in --no-reloc mode; reloc mode tracks the gz truth instead.
        self.latest_center = np.array([p.x, p.y, p.z])

    def out_cb(self, msg):
        p = msg.pose.pose.position
        self.latest_out = (p.x, p.y, p.z)

    def gt_cb(self, msg):
        for t in msg.transforms:
            if t.child_frame_id.split("/")[-1] != self.robot_name:
                continue
            _, _, T = tf2np(t)
            self.latest_center = T[:3, 3].copy()
            self.latest_yaw = float(np.arctan2(T[1, 0], T[0, 0]))

    # ---- gz entity control (ign service CLI) ----

    def spawn_model(self, entry):
        name, mtype, pose = entry["name"], entry["type"], entry["pose"]
        TMP_SDF_DIR.mkdir(exist_ok=True)
        sdf_path = TMP_SDF_DIR / f"{name}.sdf"
        sdf_path.write_text(build_model_sdf(name, mtype))
        qx, qy, qz, qw = euler_to_quaternion(*pose[3:6])
        req = (f'sdf_filename: "{sdf_path}" name: "{name}" allow_renaming: false '
               f'pose {{ position {{ x: {pose[0]} y: {pose[1]} z: {pose[2]} }} '
               f'orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }} }}')
        ok, out = ign_service(f"/world/{self.world}/create", "ignition.msgs.EntityFactory", req)
        print(f"spawn {mtype} as {name}: {'ok' if ok else 'FAILED'} {out if not ok else ''}")
        return ok

    def spawn_pack(self, entries):
        name = "scene_obstacles"
        TMP_SDF_DIR.mkdir(exist_ok=True)
        sdf_path = TMP_SDF_DIR / f"{name}.sdf"
        sdf_path.write_text(build_pack_sdf(name, entries))
        req = f'sdf_filename: "{sdf_path}" name: "{name}" allow_renaming: false'
        ok, out = ign_service(f"/world/{self.world}/create", "ignition.msgs.EntityFactory", req)
        print(f"spawn pack {name} ({len(entries)} links): {'ok' if ok else 'FAILED'} {out if not ok else ''}")
        return ok

    def remove_model(self, name):
        req = f'name: "{name}" type: MODEL'
        ok, _ = ign_service(f"/world/{self.world}/remove", "ignition.msgs.Entity", req)
        return ok

    def set_model_pose(self, name, pose):
        # set_pose_vector/Pose_V is the only teleport that can target a named
        # model: /world/X/set_pose takes msgs::Pose which has no name field
        # (the old call here failed request creation silently)
        qx, qy, qz, qw = euler_to_quaternion(*pose[3:6])
        req = (f'pose {{ name: "{name}" position {{ x: {pose[0]} y: {pose[1]} z: {pose[2]} }} '
               f'orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }} }}')
        ok, out = ign_service(f"/world/{self.world}/set_pose_vector", "ignition.msgs.Pose_V", req)
        print(f"set_pose {name} -> {pose[:3]}: {'ok' if ok else 'FAILED'} {out if not ok else ''}")
        return ok

    # ---- scene setup ----

    def cleanup_previous(self):
        if not SPAWN_STATE.exists():
            return
        for name in json.load(open(SPAWN_STATE)):
            self.remove_model(name)
        SPAWN_STATE.unlink()
        print("removed models from previous run")

    def scene_initialize(self, models):
        self.remove_model("scene_obstacles")  # clear stale pack, if any
        packed = [e for e in models if e.get("batch", True)]
        solo = [e for e in models if not e.get("batch", True)]
        spawned = []
        if packed:
            t0 = time.monotonic()
            if self.spawn_pack(packed):
                spawned.append("scene_obstacles")
            print(f"pack spawn took {time.monotonic() - t0:.1f}s")
        for entry in solo:
            self.remove_model(entry["name"])  # clear stale same-name entity
            if self.spawn_model(entry):
                spawned.append(entry["name"])
            time.sleep(0.05)
        SPAWN_STATE.write_text(json.dumps(spawned))
        print(f"spawned {len(models)} obstacles ({len(packed)} packed, {len(solo)} individual)")

    # ---- robot / SLAM reset ----

    def start_zero_cmd(self):
        self._zero_cmd_stop.clear()

        def loop():
            while not self._zero_cmd_stop.is_set():
                self.pub_cmd.publish(Twist())
                time.sleep(0.05)

        self._zero_cmd_thread = threading.Thread(target=loop, daemon=True)
        self._zero_cmd_thread.start()

    def stop_zero_cmd(self):
        self._zero_cmd_stop.set()
        if self._zero_cmd_thread:
            self._zero_cmd_thread.join(timeout=2)

    def reset_robot(self, robot):
        pose = robot.get("pose", [0.0] * 6)
        name = robot.get("name", self.robot_name)
        self.start_zero_cmd()
        if self.set_model_pose(name, pose):
            # re-anchor perception at the new pose: repeated publishes ride
            # out the pub-matching race of a one-shot message
            for _ in range(3):
                self.pub_vio_reset.publish(Empty())
                time.sleep(0.3)
        time.sleep(0.5)
        # Hold target overwrites planning's stale target_pose from the last run.
        self.publish_target(pose[:3])
        self.stop_zero_cmd()

    def wait_reloc(self):
        """Block until sim_gt_reloc reports calibrated (latched status). The
        robot must not move before calibration; nothing has moved it yet."""
        if self.args.no_reloc:
            print("sim_gt_reloc wait skipped (--no-reloc)")
            return
        print("waiting for sim_gt_reloc (goal transform ready)...")
        state = {"ready": None}
        self.create_subscription(
            Bool, "/sim/gt_reloc/status",
            lambda m: state.update(ready=m.data),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.args.reloc_timeout:
            rclpy.spin_once(self, timeout_sec=0.5)
            if state["ready"]:
                print("sim_gt_reloc calibrated")
                return
        raise SystemExit(
            f"sim_gt_reloc did not report calibrated within {self.args.reloc_timeout}s. "
            "If map_node is running, rerun with --no-reloc.")

    # ---- targets ----

    def publish_target_once(self, xyz):
        msg = Odometry()
        msg.header.frame_id = "world"
        msg.pose.pose.position.x = float(xyz[0])
        msg.pose.pose.position.y = float(xyz[1])
        msg.pose.pose.position.z = float(xyz[2])
        msg.pose.pose.orientation.w = 1.0
        self.pub_target.publish(msg)

    def publish_target(self, xyz):
        for _ in range(5):
            self.publish_target_once(xyz)
            time.sleep(0.2)

    def run_path(self, path_cfg):
        """map_node-style carrot goal: every cycle publish the path point
        `lookahead` meters ahead of the robot's projection on the polyline."""
        pts = np.array(path_cfg["points"], dtype=float)
        lookahead = path_cfg.get("lookahead", 2.5)
        tol = path_cfg.get("tolerance", 0.5)
        timeout = path_cfg.get("timeout", 180)
        period = 1.0 / path_cfg.get("rate", 2)
        seg = np.diff(pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]
        print(f"\n== path mode: {len(pts)} waypoints, length {total:.1f}m, "
              f"lookahead {lookahead}m, end tol {tol}m, timeout {timeout}s")

        t0 = time.monotonic()
        last_report = 0.0
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=period)
            if self.latest_center is None:
                continue
            p = self.latest_center[:2]
            # project robot onto each segment; keep the closest projection's arc length
            rel = p - pts[:-1]
            t = np.clip((rel * seg).sum(axis=1) / np.maximum(seg_len ** 2, 1e-9), 0, 1)
            proj = pts[:-1] + t[:, None] * seg
            dists = np.hypot(*(proj - p).T)
            i = int(np.argmin(dists))
            s = cum[i] + t[i] * seg_len[i]
            s_goal = min(s + lookahead, total)
            idx = min(int(np.searchsorted(cum, s_goal, side="right")) - 1, len(seg_len) - 1)
            ratio = (s_goal - cum[idx]) / max(seg_len[idx], 1e-9)
            goal = pts[idx] + ratio * seg[idx]
            # ride at the robot's own height: 2D polyline points carry no z,
            # and a hardcoded 0 buries the goal under a raised deck (the
            # factory floor sits at z~0.26) where the ESDF reads it as an
            # obstacle
            self.publish_target_once([goal[0], goal[1], float(self.latest_center[2])])
            # debug: full transform chain per cycle -- authored goal (world),
            # robot (world), and the reloc-transformed goal (SLAM frame)
            o = self.latest_out
            if o is not None and self.latest_yaw is not None:
                route_yaw = math.degrees(math.atan2(seg[idx][1], seg[idx][0]))
                c = self.latest_center
                print(f"  [goalchain] goal_gz=({goal[0]:.2f},{goal[1]:.2f},{c[2]:.2f}) "
                      f"route_yaw={route_yaw:.0f}deg "
                      f"robot_gz=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f},{math.degrees(self.latest_yaw):.0f}) "
                      f"out_slam=({o[0]:.2f},{o[1]:.2f},{o[2]:.2f})")
            else:
                print("  [goalchain] robot/out not ready yet")
            end_dist = float(np.hypot(*(p - pts[-1])))
            if end_dist < tol:
                print(f"reached path end in {time.monotonic() - t0:.1f}s "
                      f"(dist {end_dist:.2f}m)")
                return True
            if time.monotonic() - last_report > 2.0:
                last_report = time.monotonic()
                print(f"  [{time.monotonic() - t0:5.1f}s] progress {s:.1f}/{total:.1f}m "
                      f"goal=({goal[0]:.2f},{goal[1]:.2f}) end_dist={end_dist:.2f}m "
                      f"pos=({p[0]:.2f},{p[1]:.2f})")
        print("WARN: path timeout")
        return False

    def run_targets(self, targets):
        for target in targets:
            name = target.get("name", "target")
            tol = target.get("tolerance", 0.5)
            timeout = target.get("timeout", 120)
            xyz = target["pose"]
            print(f"\n== target '{name}' -> {xyz}, tol {tol}m, timeout {timeout}s")
            self.publish_target(xyz)
            t0 = time.monotonic()
            last_report = 0.0
            while time.monotonic() - t0 < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.latest_center is None:
                    continue
                dist = float(np.hypot(*(self.latest_center[:2] - xyz[:2])))
                if dist < tol:
                    print(f"reached '{name}' in {time.monotonic() - t0:.1f}s "
                          f"(dist {dist:.2f}m)")
                    break
                if time.monotonic() - last_report > 2.0:
                    last_report = time.monotonic()
                    print(f"  [{time.monotonic() - t0:5.1f}s] dist={dist:.2f}m "
                          f"pos=({self.latest_center[0]:.2f},{self.latest_center[1]:.2f})")
            else:
                print(f"WARN: timeout waiting for '{name}', moving on")
        if self.latest_center is not None:
            print("\nall targets done; publishing current pose as hold target")
            self.publish_target(self.latest_center)

    # ---- entry ----

    def run(self, scene):
        t0 = time.monotonic()
        while self.world is None and time.monotonic() - t0 < 60:
            self.world = discover_world()
            if self.world is None:
                print("waiting for gz world...")
                time.sleep(2)
        if self.world is None:
            raise RuntimeError("no gz world found (is the sim running?)")
        print(f"world: {self.world}")

        # rig-agnostic scenes carry no robot name: the launcher's
        # TINYNAV_ROBOT_MODEL is the single source of truth
        self.robot_name = scene.get("robot", {}).get("name") or self.robot_name
        if not self.args.no_reloc:
            # progress/reached checks against the gz ground truth; the goals
            # themselves are transformed into the SLAM frame by sim_gt_reloc.
            self.create_subscription(
                TFMessage, f"/world/{self.world}/pose/info", self.gt_cb,
                QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE))
        else:
            self.create_subscription(Odometry, "/slam/odometry_visual", self.odom_cb, 10)

        self.wait_reloc()
        self.cleanup_previous()
        models = expand_models(scene.get("models", []))
        if models:
            self.scene_initialize(models)
        self.reset_robot(scene.get("robot", {}))
        if not self.args.spawn_only:
            if "path" in scene:
                if self.run_path(scene["path"]) and self.latest_center is not None:
                    self.publish_target(self.latest_center)  # hold at path end
            else:
                self.run_targets(scene.get("targets", []))
        print("scene setup done; world left running for inspection")


def resolve_scene_path(arg):
    p = pathlib.Path(arg)
    if p.is_file():
        return p
    for cand in (SCENE_ROOT / "config/scenes" / f"{arg}.json",
                 SCENE_ROOT / "config/scenes" / arg):
        if cand.is_file():
            return cand
    raise SystemExit(f"scene not found: {arg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene name (l_corridor) or path to scene JSON")
    ap.add_argument("--spawn-only", action="store_true", help="spawn walls + reset robot, no targets")
    ap.add_argument("--no-reloc", action="store_true", help="do not wait for sim_gt_reloc (map_node mode)")
    ap.add_argument("--reloc-timeout", type=float, default=120.0,
                    help="seconds to wait for sim_gt_reloc calibration")
    args = ap.parse_args()

    scene = json.load(open(resolve_scene_path(args.scene)))
    print(f"scene: {scene.get('desc', args.scene)}")
    from gen_textures import ensure_textures
    ensure_textures()
    rclpy.init(args=[])
    node = AutoNavSim(args)
    try:
        node.run(scene)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
