#!/usr/bin/env python3
"""
Convert TinyNav map artifacts into NeRF/Nerfstudio transforms.json format.

Usage:
    python tool/convert_to_nerf_format.py --map-dir tinynav_map
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from tool.video_db import VideoDB


def convert_nerf_format(
    output_dir: Path,
    poses: Dict[int, np.ndarray],
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    t_rgb_to_infra1: np.ndarray,
) -> None:
    camera_model = "PINHOLE"
    fl_x = float(intrinsics[0, 0])
    fl_y = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    h = int(image_size[0])
    w = int(image_size[1])
    frames = []
    opencv_to_opengl_convention = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )

    for timestamp, camera_to_world_pose in poses.items():
        camera_to_world_opengl = (
            camera_to_world_pose @ t_rgb_to_infra1 @ opencv_to_opengl_convention
        )
        frames.append(
            {
                "file_path": f"images/image_{timestamp}.png",
                "transform_matrix": camera_to_world_opengl.tolist(),
            }
        )

    data = {
        "camera_model": camera_model,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": frames,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "transforms.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def infer_image_size(images_dir: Path) -> Tuple[int, int]:
    image_candidates = sorted(images_dir.glob("image_*.png"))
    if not image_candidates:
        raise FileNotFoundError(f"No image_*.png found under: {images_dir}")

    image = cv2.imread(str(image_candidates[0]), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_candidates[0]}")
    return image.shape[:2]


def export_rgb_images(
    map_dir: Path,
    images_dir: Path,
    timestamps: list[int],
) -> Tuple[int, int]:
    images_dir.mkdir(parents=True, exist_ok=True)
    image_size = None
    rgb_db_dir = map_dir / "rgb_images_db"
    rgb_video_db = VideoDB(dir_path=str(rgb_db_dir), mode="read")
    try:
        for timestamp in tqdm(timestamps, desc="Exporting RGB images", unit="img"):
            rgb_image = rgb_video_db.read(timestamp)
            if rgb_image is None:
                raise KeyError(f"Missing rgb image timestamp {timestamp} in {rgb_db_dir / 'meta.json'}")
            if image_size is None:
                image_size = rgb_image.shape[:2]
            cv2.imwrite(str(images_dir / f"image_{timestamp}.png"), rgb_image)
    finally:
        rgb_video_db.close()

    if image_size is None:
        raise RuntimeError("No RGB images exported; poses may be empty")
    return image_size


def infer_t_rgb_to_infra1_from_tf_messages(
    tf_messages: Dict[int, Dict[str, np.ndarray]]
) -> np.ndarray:
    latest_tf: Dict[str, np.ndarray] = {}
    for timestamp_ns in sorted(tf_messages.keys()):
        for edge_key, transform in tf_messages[timestamp_ns].items():
            latest_tf[edge_key] = np.asarray(transform, dtype=np.float64)

    # Looper bags use cam_left/cam_rgb directly as camera frames.
    if "cam_left->cam_rgb" in latest_tf:
        return latest_tf["cam_left->cam_rgb"]

    required_keys = [
        "camera_link->camera_infra1_frame",
        "camera_infra1_frame->camera_infra1_optical_frame",
        "camera_link->camera_color_frame",
        "camera_color_frame->camera_color_optical_frame",
    ]
    missing = [k for k in required_keys if k not in latest_tf]
    if missing:
        raise KeyError(
            "Missing TF edges in tf_messages.npy required to derive T_rgb_to_infra1: "
            + ", ".join(missing)
        )

    t_infra1_to_link = latest_tf["camera_link->camera_infra1_frame"]
    t_infra1_optical_to_infra1 = latest_tf[
        "camera_infra1_frame->camera_infra1_optical_frame"
    ]
    t_rgb_to_link = latest_tf["camera_link->camera_color_frame"]
    t_rgb_optical_to_rgb = latest_tf["camera_color_frame->camera_color_optical_frame"]
    return (
        np.linalg.inv(t_infra1_optical_to_infra1)
        @ np.linalg.inv(t_infra1_to_link)
        @ t_rgb_to_link
        @ t_rgb_optical_to_rgb
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TinyNav map directory to NeRF transforms.json"
    )
    parser.add_argument(
        "--map-dir",
        required=True,
        help="TinyNav map directory containing poses.npy/intrinsics/images",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Override images directory (default: <map-dir>/images)",
    )
    parser.add_argument(
        "--reuse-existing-images",
        action="store_true",
        help="Do not export from rgb_images_db/video.mp4; reuse existing image_*.png files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_dir = Path(args.map_dir)
    images_dir = Path(args.images_dir) if args.images_dir else map_dir / "images"

    poses = np.load(map_dir / "poses.npy", allow_pickle=True).item()
    intrinsics = np.load(map_dir / "rgb_camera_intrinsics.npy", allow_pickle=True)
    tf_messages_path = map_dir / "tf_messages.npy"
    if tf_messages_path.exists():
        tf_messages = np.load(tf_messages_path, allow_pickle=True).item()
        t_rgb_to_infra1 = infer_t_rgb_to_infra1_from_tf_messages(tf_messages)
        print(f"Using TF data from {tf_messages_path} to derive T_rgb_to_infra1")
    else:
        t_rgb_to_infra1 = np.load(map_dir / "T_rgb_to_infra1.npy", allow_pickle=True)
        print("tf_messages.npy not found; using T_rgb_to_infra1.npy")
    timestamps = sorted(int(k) for k in poses.keys())
    if args.reuse_existing_images:
        image_size = infer_image_size(images_dir)
    else:
        image_size = export_rgb_images(map_dir, images_dir, timestamps)

    convert_nerf_format(
        output_dir=map_dir,
        poses=poses,
        intrinsics=intrinsics,
        image_size=image_size,
        t_rgb_to_infra1=t_rgb_to_infra1,
    )
    print(f"Wrote NeRF transforms to {map_dir / 'transforms.json'}")


if __name__ == "__main__":
    main()
