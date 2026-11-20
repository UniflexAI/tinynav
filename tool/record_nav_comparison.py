"""Record one navigation run with the active planner's configuration and video inputs."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node

from tinynav.core.planner_config import planner_mode, route_guidance, shared_safety


TOPICS = [
    "/cmd_vel", "/planning/rl_cmd", "/planning/nav_cmd", "/planning/decision", "/planning/trajectory_path",
    "/planning/obstacle_mask", "/planning/height_map", "/planning/footprint",
    "/planning/rl_bev", "/planning/timing",
    "/slam/odometry", "/slam/odometry_visual", "/slam/camera_info",
    "/control/target_pose", "/mapping/global_plan", "/mapping/current_pose_in_map",
    "/mapping/nav_progress", "/mapping/nav_done", "/mapping/poi_change",
    "/nav/active", "/nav/paused", "/tf", "/tf_static", "/rosout",
]
CAMERA_TOPICS = [
    "/camera/camera/color/image_rect_raw/compressed",
    "/camera/camera/color/image_raw/compressed",
    "/camera/camera/infra1/image_rect_raw/compressed",
    "/camera/camera/color/image_raw",
    "/camera/camera/color/image_rect_raw",
    "/camera/camera/infra1/image_rect_raw",
]
IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
PROBE = """
import json, os
from dataclasses import asdict
from tinynav.core.robot_specs import ROBOT_CONFIG
from tinynav.core.planner_config import go2_response_guard, planner_mode, route_guidance, policy_engine_path, shared_safety
from tinynav.core.navigation_safety import NavigationLimits
from tinynav.rl.ppo_planner import PPOGeometry, GUARD_MARGIN
limits = NavigationLimits.from_environment() if shared_safety() else None
geometry = None
if route_guidance():
    default = PPOGeometry()
    geometry = PPOGeometry(**{
        k: float(os.environ.get('TINYNAV_PPO_' + k.upper() + '_M', getattr(default, k)))
        for k in ('front', 'rear', 'half_width', 'control_x', 'control_y')
    })
    if limits is not None:
        geometry = PPOGeometry(**{k: getattr(limits, k) for k in
                               ('front', 'rear', 'half_width', 'control_x', 'control_y')})
print(json.dumps(dict(
    mode=planner_mode(), route_guidance=route_guidance(), robot=asdict(ROBOT_CONFIG),
    ppo_geometry_m=asdict(geometry) if geometry is not None else None,
    ppo_margin_m=GUARD_MARGIN if planner_mode() == 'rl' else None,
    diagnostic_clouds=os.environ.get('TINYNAV_PLANNING_DIAGNOSTIC_CLOUDS', '0') == '1',
    latest_pair_only=os.environ.get('TINYNAV_PLANNING_LATEST_PAIR_ONLY', '1') == '1',
    timing_enabled=os.environ.get('TINYNAV_PLANNING_TIMING') == '1',
    shared_safety=shared_safety(), go2_response_guard=go2_response_guard(ROBOT_CONFIG.name),
    navigation_limits=asdict(limits) if limits is not None else None,
    engine=policy_engine_path() if planner_mode() == 'rl' else None,
)))
"""


def file_sha256(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def running_nodes():
    found = {"planning_node": [], "cmd_vel_control": [], "map_node": []}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            executable = (proc / "exe").resolve(strict=True)
            if not executable.name.startswith("python"):
                continue
            args = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            for name in found:
                matches = [arg for arg in args if Path(arg).name == name + ".py"]
                if not matches:
                    continue
                cwd = (proc / "cwd").resolve(strict=True)
                source = Path(matches[0])
                source = source if source.is_absolute() else cwd / source
                environ = dict(item.split("=", 1) for item in
                               (proc / "environ").read_bytes().decode().split("\0") if "=" in item)
                if environ.get("ROS_DOMAIN_ID", "0") != os.environ.get("ROS_DOMAIN_ID", "0"):
                    continue
                python = Path(args[0])
                if not python.is_absolute():
                    python = Path(environ.get("VIRTUAL_ENV", "")) / "bin/python"
                python = python if python.is_file() else executable
                found[name].append(dict(pid=int(proc.name), executable=python,
                                        args=args, source=source.resolve(), cwd=cwd, environ=environ))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    for name in ("planning_node", "cmd_vel_control"):
        if len(found[name]) != 1:
            raise ValueError(f"Expected one running {name}, found {len(found[name])}. "
                             "Start the navigation nodes before recording.")
    return found


def capture_config(nodes):
    planner = nodes["planning_node"][0]
    controller = nodes["cmd_vel_control"][0]
    if planner_mode(planner["environ"]) != planner_mode(controller["environ"]):
        raise ValueError("Planning and velocity control selected different planners. Restart both nodes.")
    if shared_safety(planner["environ"]) != shared_safety(controller["environ"]):
        raise ValueError("Planning and velocity control selected different navigation safety modes.")
    repo = planner["source"].parents[2]
    env = dict(planner["environ"])
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.check_output([str(planner["executable"]), "-c", PROBE], cwd=repo, env=env, text=True)
    effective = json.loads(result)
    control_env = dict(controller["environ"])
    control_env["PYTHONPATH"] = str(repo) + os.pathsep + control_env.get("PYTHONPATH", "")
    control_result = subprocess.check_output([str(controller["executable"]), "-c", PROBE],
                                            cwd=repo, env=control_env, text=True)
    control_limits = json.loads(control_result)["navigation_limits"]
    if effective["navigation_limits"] != control_limits:
        raise ValueError("Planning and velocity control have different navigation limits.")
    if (controller["environ"].get("ROBOT_TYPE", "go2").strip().lower()
            != planner["environ"].get("ROBOT_TYPE", "go2").strip().lower()):
        raise ValueError("Planning and velocity control selected different robot profiles.")
    engine = effective["engine"]
    if engine:
        path = Path(engine)
        path = path if path.is_absolute() else planner["cwd"] / path
        effective["engine"] = dict(path=str(path), sha256=file_sha256(path))

    tree = ast.parse(controller["source"].read_text())
    constants = {}
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
            for target in statement.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    constants[target.attr] = statement.value.value
    effective["controller_constants"] = constants
    snapshot = {}
    for name, items in nodes.items():
        snapshot[name] = [dict(
            pid=item["pid"], command=item["args"], source=str(item["source"]),
            source_sha256=file_sha256(item["source"]),
            environment={k: v for k, v in item["environ"].items()
                         if k.startswith("TINYNAV_") or k in
                         ("ROBOT_TYPE", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "ROS_LOCALHOST_ONLY")},
        ) for item in items]
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True)
    return dict(effective=effective, nodes=snapshot, git_commit=git.stdout.strip() or None,
                git_status=dirty.stdout.splitlines(), git_error=git.stderr.strip() or None), repo


def front_camera(node, requested):
    available = dict(node.get_topic_names_and_types())
    names = [requested] if requested else CAMERA_TOPICS
    for topic in names:
        if IMAGE_TYPES.intersection(available.get(topic, [])) and node.count_publishers(topic):
            return topic
    raise ValueError("No front camera image publisher found. Supply --image-topic TOPIC, "
                     "or use --no-video for a navigation-data-only recording.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="run", help="For example poi5-r1 or crossing-r2")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get(
        "TINYNAV_RECORD_DIR", Path.home() / ".local/share/tinynav/rosbags")))
    parser.add_argument("--image-topic", help="Front camera Image or CompressedImage topic")
    parser.add_argument("--no-video", action="store_true", help="Skip the front camera")
    parser.add_argument("--depth", action="store_true", help="Also record raw /slam/depth for perception replay")
    args = parser.parse_args()
    nodes = running_nodes()
    config, repo = capture_config(nodes)
    mode = "ppo" if config["effective"]["mode"] == "rl" and config["effective"]["route_guidance"] else config["effective"]["mode"]
    mode = "dwa" if mode == "library" else mode

    rclpy.init(args=[])
    node = Node("nav_comparison_recorder")
    try:
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)
        camera = None if args.no_video else front_camera(node, args.image_topic)
        available = dict(node.get_topic_names_and_types())
        publishers = {topic: node.count_publishers(topic) for topic in TOPICS}
    finally:
        node.destroy_node()
        rclpy.shutdown()

    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.label).strip("-.") or "run"
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"tinynav-compare-{mode}-{label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir()
    sources = out / "sources"
    sources.mkdir()
    paths = {item["source"] for items in nodes.values() for item in items}
    paths.update(repo / name for name in ("tinynav/core/robot_specs.py", "tinynav/core/planner_config.py",
                                        "tinynav/core/map_node.py",
                                        "tinynav/core/navigation_safety.py", "tinynav/rl/route_rejoin.py",
                                        "tinynav/rl/calibrated_guard.py", "tinynav/rl/go2_response.json",
                                        "tinynav/rl/ppo_planner.py"))
    for path in sorted(paths):
        (sources / path.name).write_bytes(path.read_bytes())
    topics = [*TOPICS]
    for topic in (os.environ.get("TINYNAV_EKF_TOPIC", "/go2/ekf/odometry"), "/camera/camera/vio_image"):
        if topic in available and topic not in topics:
            topics.append(topic)
    if camera:
        topics.append(camera)
    if args.depth:
        topics.append("/slam/depth")
    config.update(started_at=datetime.now(timezone.utc).isoformat(), label=args.label,
                  front_camera=camera, topics=topics, publishers_at_start=publishers,
                  topic_types_at_start=available)
    manifest = out / "runtime-config.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    command = ["ros2", "bag", "record", "--output", str(out / "bag"),
               "--storage", "sqlite3", "--compression-mode", "file", "--compression-format", "zstd",
               "--max-cache-size", "67108864", *topics]
    version = config["git_commit"] or "unavailable (source snapshots saved)"
    print(f"Active planner: {mode}; commit: {version}", flush=True)
    print(f"Front camera: {camera}; raw depth: {args.depth}", flush=True)
    print(f"Output: {out}", flush=True)
    print("Wait for rosbag2 to report 'Recording...' before sending the POI. Stop with Ctrl+C.", flush=True)
    config["record_command"] = command
    config["record_exit_code"] = None

    with (out / "recorder.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        previous_handlers = {}
        def stop_recording(signum, frame):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, stop_recording)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            config["record_exit_code"] = proc.returncode
            config["finished_at"] = datetime.now(timezone.utc).isoformat()
            config["metadata_present"] = (out / "bag/metadata.yaml").is_file()
            manifest.write_text(json.dumps(config, indent=2) + "\n")
    if not config["metadata_present"]:
        raise RuntimeError(f"The recorder did not finalize a bag. See {out / 'recorder.log'}")
    print(f"Saved bag and runtime configuration to {out}", flush=True)
    print(f"Upload the complete {out.name} directory, including runtime-config.json and sources/.", flush=True)
    return 0 if code in (0, -signal.SIGINT, 128 + signal.SIGINT) else code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
