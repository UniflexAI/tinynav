#!/usr/bin/env python3
"""
tool/qr_odom/board_detect.py

ROS-free AprilTag GridBoard detection helpers, shared by any qr_odom script
that needs to find one or more of the tag_grid_2x2_s76mm*.json boards
(see generate.py --id-offset) in a single image.
"""

import json
from pathlib import Path

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def load_boards(qrcode_dir: Path, pattern: str = "tag_grid_2x2_s76mm*.json") -> dict[str, cv2.aruco.GridBoard]:
    """Load every board config matching `pattern` in `qrcode_dir`.

    Returns {label: GridBoard}, label = "<first_tag_id>-<last_tag_id>" (e.g. "4-7").
    """
    boards = {}
    for path in sorted(qrcode_dir.glob(pattern)):
        d = json.loads(path.read_text())
        tag_ids = d["tag_ids"]
        label = f"{tag_ids[0]}-{tag_ids[-1]}"
        boards[label] = cv2.aruco.GridBoard(
            size=(2, 2),
            markerLength=float(d["size_m"]),
            markerSeparation=float(d["spacing_m"]),
            dictionary=ARUCO_DICT,
            ids=np.array(tag_ids, dtype=np.int32),
        )
    return boards


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(ARUCO_DICT, params)


def detect_board_poses(
    img: np.ndarray,
    K: np.ndarray,
    boards: dict[str, cv2.aruco.GridBoard],
    detector: cv2.aruco.ArucoDetector,
    dist: np.ndarray | None = None,
    min_tags: int = 2,
    max_reproj_px: float = 3.0,
) -> dict[str, tuple[np.ndarray, float, int]]:
    """Detect every board from `boards` visible in `img`.

    Returns {label: (T_camera_board, reproj_err_px, n_tags_visible)} for boards
    with >= min_tags visible tags and a solvePnPRansac reprojection error below
    max_reproj_px. Boards not visible (or below quality gate) are omitted.
    """
    results: dict[str, tuple[np.ndarray, float, int]] = {}
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return results

    ids_flat = ids.flatten()
    for label, board in boards.items():
        board_ids = set(board.getIds().flatten().tolist())
        n_visible = sum(1 for tid in ids_flat if tid in board_ids)
        if n_visible < min_tags:
            continue

        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            continue

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None:
            continue

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        reproj_err = float(np.mean(
            np.linalg.norm(img_pts.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        ))
        if reproj_err > max_reproj_px:
            continue

        R_mat, _ = cv2.Rodrigues(rvec)
        T_camera_board = np.eye(4)
        T_camera_board[:3, :3] = R_mat
        T_camera_board[:3,  3] = tvec.ravel()
        results[label] = (T_camera_board, reproj_err, n_visible)

    return results
