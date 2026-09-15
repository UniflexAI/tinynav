#!/usr/bin/env python3
"""Quick looper_bridge QoS comparison on a Looper-connected host."""

import argparse
import subprocess
import sys
import threading
import time


def stress_subscriber(seconds: float):
    code = f"""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

rclpy.init()
node = Node("stress_sub")
qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
)
count = 0

def cb(_msg):
    global count
    count += 1

node.create_subscription(Image, "/camera/camera/depth/image_rect_raw", cb, qos)
t0 = time.monotonic()
while time.monotonic() - t0 < {seconds:.1f}:
    rclpy.spin_once(node, timeout_sec=0.05)
print(f"stress_rx={{count}}")
node.destroy_node()
rclpy.shutdown()
"""
    subprocess.run(
        ["uv", "run", "python", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_case(name: str, bridge_args: list[str], seconds: float, with_stress: bool):
    cmd = [
        "uv",
        "run",
        "python",
        "/tinynav/tool/looper_bridge_node.py",
        "--sync-watchdog-s",
        "0",
        *bridge_args,
    ]
    print(f"\n===== {name} =====")
    print("cmd:", " ".join(cmd))
    if with_stress:
        print("(with extra RELIABLE depth subscriber depth=50)")

    stress = None
    if with_stress:
        stress = threading.Thread(target=stress_subscriber, args=(seconds + 2.0,))
        stress.start()
        time.sleep(0.5)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    sync_times: list[float] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        line = line.strip()
        if "Image QoS:" in line:
            print(line)
        if "sync_callback:" in line:
            sync_times.append(time.monotonic())

    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()

    if stress is not None:
        stress.join()

    if len(sync_times) < 2:
        print(f"sync_count={len(sync_times)}  avg_hz=n/a  (stalled or no match)")
        return

    gaps = [sync_times[i] - sync_times[i - 1] for i in range(1, len(sync_times))]
    span = sync_times[-1] - sync_times[0]
    hz = (len(sync_times) - 1) / span if span > 0 else 0.0
    print(
        f"sync_count={len(sync_times)}  span={span:.1f}s  avg_hz={hz:.2f}  "
        f"gap_max={max(gaps):.3f}s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()

    cases = [
        (
            "OLD depth=50 queue=20",
            ["--image-reliability", "reliable", "--image-qos-depth", "50", "--sync-queue-size", "20"],
        ),
        (
            "NEW depth=5 queue=10",
            ["--image-reliability", "reliable", "--image-qos-depth", "5", "--sync-queue-size", "10"],
        ),
        (
            "NEW depth=5 + approx slop=0.05",
            [
                "--image-reliability",
                "reliable",
                "--image-qos-depth",
                "5",
                "--sync-queue-size",
                "10",
                "--sync-slop",
                "0.05",
            ],
        ),
    ]
    for name, bridge_args in cases:
        run_case(name, bridge_args, args.seconds, args.stress)


if __name__ == "__main__":
    main()
