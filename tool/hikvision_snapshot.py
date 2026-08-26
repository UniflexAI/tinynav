#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Hikvision snapshot utility for upper-layer callers.

This module does one thing: connect to a Hikvision RTSP stream and return one
frame as an OpenCV image. Uploading, LLM analysis, mission orchestration, and
other business logic stay outside this module.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ENV_PATH = _REPO_ROOT / ".env"


class HikvisionSnapshotError(RuntimeError):
    """Raised when the camera cannot provide a usable frame."""


@dataclass(frozen=True)
class HikvisionCameraConfig:
    ip: str
    username: str
    password: str
    port: int = 554
    channel: str = "101"

    @property
    def rtsp_url(self) -> str:
        return (
            f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}"
            f"/Streaming/Channels/{self.channel}"
        )


def capture_frame(
    config: HikvisionCameraConfig,
    *,
    read_attempts: int = 5,
) -> object:
    """Return one camera frame as an OpenCV image.

    `read_attempts` gives the stream a few chances to warm up before failing.
    """

    if read_attempts < 1:
        raise ValueError("read_attempts must be >= 1")

    cap = cv2.VideoCapture(config.rtsp_url)
    if not cap.isOpened():
        raise HikvisionSnapshotError(
            f"failed to connect to Hikvision camera at {config.ip}"
        )

    try:
        for _ in range(read_attempts):
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        raise HikvisionSnapshotError(
            f"connected to {config.ip} but failed to read a frame"
        )
    finally:
        cap.release()


def save_frame(frame: object, output_path: str | Path) -> Path:
    """Save one OpenCV image to disk and return the resolved path."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise HikvisionSnapshotError(f"failed to save image to {path}")
    return path


def capture_frame_to_file(
    config: HikvisionCameraConfig,
    *,
    output_dir: str | Path = ".",
    filename_prefix: str = "hik",
    read_attempts: int = 5,
) -> Path:
    """Capture one frame and save it to a timestamped JPEG file."""

    frame = capture_frame(config, read_attempts=read_attempts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{filename_prefix}_{timestamp}.jpg"
    return save_frame(frame, output_path)


def _load_env_file(path: Path = _DEFAULT_ENV_PATH) -> None:
    """Best-effort `.env` loader for local manual testing.

    Keeps the module dependency-free and never overrides environment variables
    that are already set by the shell or the container runtime.
    """

    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one image from a Hikvision RTSP camera."
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("HIK_CAMERA_IP"),
        help="Camera IP address (or HIK_CAMERA_IP)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("HIK_CAMERA_USERNAME"),
        help="Camera username (or HIK_CAMERA_USERNAME)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("HIK_CAMERA_PASSWORD"),
        help="Camera password (or HIK_CAMERA_PASSWORD)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HIK_CAMERA_PORT", "554")),
        help="RTSP port (default: 554, or HIK_CAMERA_PORT)",
    )
    parser.add_argument(
        "--channel",
        default=os.environ.get("HIK_CAMERA_CHANNEL", "101"),
        help="RTSP channel, such as 101 or 102 (default: 101, or HIK_CAMERA_CHANNEL)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("HIK_OUTPUT_DIR", "."),
        help="Directory used to save the captured JPEG (or HIK_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--filename-prefix",
        default=os.environ.get("HIK_FILENAME_PREFIX", "hik"),
        help="Output filename prefix (default: hik, or HIK_FILENAME_PREFIX)",
    )
    parser.add_argument(
        "--read-attempts",
        type=int,
        default=int(os.environ.get("HIK_READ_ATTEMPTS", "5")),
        help="Frame read attempts before failing (default: 5, or HIK_READ_ATTEMPTS)",
    )
    return parser


def _main() -> int:
    _load_env_file()
    args = _build_arg_parser().parse_args()
    missing = [
        name for name, value in (
            ("ip", args.ip),
            ("username", args.username),
            ("password", args.password),
        )
        if not value
    ]
    if missing:
        print(
            "Missing required camera settings: "
            + ", ".join(missing)
            + ". Provide them via CLI args or .env / HIK_CAMERA_* environment variables."
        )
        return 2
    config = HikvisionCameraConfig(
        ip=args.ip,
        username=args.username,
        password=args.password,
        port=args.port,
        channel=args.channel,
    )
    try:
        path = capture_frame_to_file(
            config,
            output_dir=args.output_dir,
            filename_prefix=args.filename_prefix,
            read_attempts=args.read_attempts,
        )
    except (HikvisionSnapshotError, ValueError) as exc:
        print(f"Failed to capture image: {exc}")
        return 1

    print(f"Saved image to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

