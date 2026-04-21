#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish POIs to /mapping/cmd_pois")
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--pois", default=None, help="Comma-separated POI ids, for example 2,1,0")
    parser.add_argument("--maps-dir", default=str(Path.home() / ".local/share/tinynav/maps"))
    return parser.parse_args()


def parse_pois_arg(pois: str) -> list[str]:
    values = [value.strip() for value in pois.split(",") if value.strip()]
    if not values:
        raise ValueError("--pois must be a comma-separated list like 2,1,0")
    return values


def load_selected_pois(maps_dir: Path, map_name: str, pois: str | None) -> dict[str, object]:
    pois_path = maps_dir / map_name / "pois.json"
    if not pois_path.exists():
        raise FileNotFoundError(f"POI file not found: {pois_path}")
    data = json.loads(pois_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("pois.json must be a JSON object")
    if pois is None:
        return data
    selected = {}
    for index, poi_key in enumerate(parse_pois_arg(pois)):
        if poi_key not in data:
            raise KeyError(f"POI {poi_key} not found in {pois_path}")
        selected[str(index)] = data[poi_key]
    return selected


def publish(payload: dict[str, object]) -> None:
    payload_json = json.dumps(payload, separators=(",", ":"))
    ros_msg_yaml = json.dumps({"data": payload_json}, separators=(",", ":"))
    cmd = [
        "bash",
        "-lc",
        "source /opt/ros/*/setup.bash >/dev/null 2>&1 && "
        f"ros2 topic pub --once /mapping/cmd_pois std_msgs/msg/String {shlex.quote(ros_msg_yaml)}",
    ]
    result = subprocess.run(cmd, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to publish POIs to /mapping/cmd_pois")


def main() -> int:
    args = parse_args()
    try:
        payload = load_selected_pois(Path(args.maps_dir), args.map_name, args.pois)
        publish(payload)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Published POIs for map: {args.map_name}")
    print(f"POIs: {args.pois or 'all'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
