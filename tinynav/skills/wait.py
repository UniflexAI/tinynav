#!/usr/bin/env python3
"""
wait — Sleep for N seconds. The simplest blocking skill.

Usage:
  ros2 run tinynav wait --seconds 5

Example:
  ros2 run tinynav wait --seconds 3
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for N seconds (blocking)")
    parser.add_argument("--seconds", type=float, default=1.0, help="Seconds to wait")
    args = parser.parse_args()
    print(f"wait: sleeping {args.seconds}s ...")
    time.sleep(args.seconds)
    print("wait: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
