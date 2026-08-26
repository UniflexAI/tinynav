#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capture one image from a Hikvision camera and save it locally.

Configuration comes from HIK_CAMERA_* (falling back to tinynav/.env, which is
gitignored) — the same source hikvision_snapshot.py and CaptureHikShot use, so
there is one place to change the camera and no credentials in the tree.
"""

import os
import sys
from datetime import datetime

import cv2

from hikvision_snapshot import _load_env_file

# Main stream is 101; switch to 102 for the lighter sub stream.
CHANNEL = os.environ.get("HIK_CAMERA_CHANNEL", "101")


def _config():
    """(ip, username, password, port), or exit with what is missing."""
    _load_env_file()
    ip = os.environ.get("HIK_CAMERA_IP")
    username = os.environ.get("HIK_CAMERA_USERNAME")
    password = os.environ.get("HIK_CAMERA_PASSWORD")
    port = os.environ.get("HIK_CAMERA_PORT", "554")

    missing = [
        name for name, value in (
            ("HIK_CAMERA_IP", ip),
            ("HIK_CAMERA_USERNAME", username),
            ("HIK_CAMERA_PASSWORD", password),
        ) if not value
    ]
    if missing:
        sys.exit(f"Not configured: {', '.join(missing)} (set them or put them in tinynav/.env)")
    return ip, username, password, port


def capture_one_frame():
    ip, username, password, port = _config()
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{CHANNEL}"

    print(f"Connecting to camera: {ip} ...")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Failed to connect. Please check:")
        print("  1. HIK_CAMERA_IP is correct and the camera is powered")
        print("  2. HIK_CAMERA_USERNAME / HIK_CAMERA_PASSWORD are correct")
        print(f"  3. This machine is on the same subnet as the camera ({ip})")
        return

    ret, frame = cap.read()
    cap.release()

    if ret:
        filename = f"hik_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image captured successfully and saved as: {filename}")
        print(f"Image size: {frame.shape[1]} x {frame.shape[0]}")
    else:
        print("Failed to read a frame from the camera.")


if __name__ == "__main__":
    capture_one_frame()
