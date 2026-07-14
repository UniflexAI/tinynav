from __future__ import annotations
import sys
sys.path.append("/tinynav/tinynav/core")
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
import numpy.typing as npt
import rclpy
import tyro
import viser
import viser.transforms as vtf
from nav_msgs.msg import Odometry
import nav_msgs
from plyfile import PlyData
from rclpy.node import Node
from scipy.ndimage import distance_transform_edt

from math_utils import msg2np, matrix_to_quat
from map_node import search_close_to_sdf_map, search_within_sdf_map
from tool.video_db import VideoDB


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PathPoint = npt.NDArray[np.floating]


class SplatFile(TypedDict):
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]


# ---------------------------------------------------------------------------
# Splat / PLY / point-cloud loaders  (from poi_editor)
# ---------------------------------------------------------------------------

def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = 3 * 4 + 3 * 4 + 4 + 4  # position + scale + rgba + ijkl
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = vtf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds")
    return {
        "centers": centers,
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    start_time = time.time()
    SH_C0 = 0.28209479177387814
    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))
    Rs = vtf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)
    print(f"PLY file with {len(v)=} loaded in {time.time() - start_time} seconds")
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }


def load_pointcloud_ply(ply_file_path: Path, center: bool = False) -> dict:
    start_time = time.time()
    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    try:
        if "red" in v.data.dtype.names:
            colors = np.stack([v["red"], v["green"], v["blue"]], axis=-1)
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float32) / 255.0
        elif "r" in v.data.dtype.names:
            colors = np.stack([v["r"], v["g"], v["b"]], axis=-1)
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float32) / 255.0
        else:
            colors = np.ones((len(v), 3), dtype=np.float32)
    except Exception:
        colors = np.ones((len(v), 3), dtype=np.float32)
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)
    print(f"Point cloud PLY with {len(v)=} loaded in {time.time() - start_time} seconds")
    return {"positions": positions, "colors": colors}


# ---------------------------------------------------------------------------
# VideoDB helpers  (from poi_editor)
# ---------------------------------------------------------------------------

def _open_infra1_video_db(map_dir: Path) -> VideoDB | None:
    db_dir = map_dir / "infra1_images_db"
    if not db_dir.exists():
        return None
    try:
        return VideoDB(dir_path=str(db_dir), mode="read")
    except Exception as e:
        print(f"Warning: failed to open infra1 VideoDB at {db_dir}: {e}")
        return None


def _load_infra1_camera_image(infra1_db: VideoDB | None, timestamp: str) -> np.ndarray | None:
    if infra1_db is None:
        return None
    try:
        ts_key = int(float(timestamp))
    except (TypeError, ValueError):
        return None
    img = infra1_db.read(ts_key)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Grid helpers (shared by path editor + SDF path planner)
# ---------------------------------------------------------------------------

def _grid_to_world(indices: np.ndarray, origin: np.ndarray, resolution: float) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return origin[None, :] + indices.astype(np.float32) * float(resolution)


def _world_to_grid(points: np.ndarray, origin: np.ndarray, resolution: float, shape: tuple[int, int, int]) -> np.ndarray:
    indices = np.rint((points - origin[None, :]) / float(resolution)).astype(np.int32)
    max_index = np.asarray(shape, dtype=np.int32) - 1
    return np.clip(indices, 0, max_index[None, :])


def _world_to_grid_index(point: np.ndarray, origin: np.ndarray, resolution: float) -> tuple[int, int, int]:
    p = np.asarray(point, dtype=np.float32)
    return (
        int((p[0] - origin[0]) / float(resolution)),
        int((p[1] - origin[1]) / float(resolution)),
        int((p[2] - origin[2]) / float(resolution)),
    )


def _grid_index_in_bounds(idx: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    return 0 <= idx[0] < shape[0] and 0 <= idx[1] < shape[1] and 0 <= idx[2] < shape[2]


def _grid_indices_to_world(path: list[tuple[int, int, int]], origin: np.ndarray, resolution: float) -> np.ndarray:
    if len(path) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(path, dtype=np.float32) * float(resolution) + origin[None, :]


def _line_segments_from_points(points: list[PathPoint]) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _sample_polyline_voxels(
    points: list[PathPoint], origin: np.ndarray, resolution: float, shape: tuple[int, int, int]
) -> np.ndarray:
    if not points:
        return np.empty((0, 3), dtype=np.int32)
    if len(points) == 1:
        return _world_to_grid(np.asarray(points, dtype=np.float32), origin, resolution, shape)
    sampled: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        start_np = np.asarray(start, dtype=np.float32)
        end_np = np.asarray(end, dtype=np.float32)
        distance = float(np.linalg.norm(end_np - start_np))
        steps = max(2, int(np.ceil(distance / max(float(resolution) * 0.5, 1e-6))) + 1)
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
        sampled.append(start_np[None, :] * (1.0 - t) + end_np[None, :] * t)
    world_points = np.concatenate(sampled, axis=0)
    indices = _world_to_grid(world_points, origin, resolution, shape)
    return np.unique(indices, axis=0)


def build_sdf_from_paths(
    paths: dict[int, dict[str, Any]], origin: np.ndarray, resolution: float, shape: tuple[int, int, int]
) -> np.ndarray:
    seed_mask = np.ones(shape, dtype=np.uint8)
    seed_count = 0
    for item in paths.values():
        indices = _sample_polyline_voxels(item["points"], origin, resolution, shape)
        if len(indices) == 0:
            continue
        seed_mask[indices[:, 0], indices[:, 1], indices[:, 2]] = 0
        seed_count += len(indices)
    if seed_count == 0:
        return np.full(shape, np.inf, dtype=np.float32)
    return distance_transform_edt(seed_mask, sampling=(resolution, resolution, resolution)).astype(np.float32)


# ---------------------------------------------------------------------------
# SDF path planning helpers  (from feat/editor-3d-occupancy-toggle)
# ---------------------------------------------------------------------------

def _segment_is_shortcut_safe(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    sdf_map: np.ndarray,
    occupancy_map: np.ndarray,
    resolution: float,
    max_segment_m: float = 1.0,
    sdf_margin_m: float = 0.2,
) -> bool:
    start_np = np.asarray(start, dtype=np.float32)
    goal_np = np.asarray(goal, dtype=np.float32)
    delta = goal_np - start_np
    distance_m = float(np.linalg.norm(delta) * resolution)
    if distance_m <= 1e-6:
        return True
    if distance_m > max_segment_m:
        return False
    steps = max(1, int(np.ceil(float(np.max(np.abs(delta))))))
    max_allowed_sdf = max(float(sdf_map[start]), float(sdf_map[goal]), 0.5) + sdf_margin_m
    for t in np.linspace(0.0, 1.0, steps + 1):
        idx = tuple(np.rint(start_np + delta * t).astype(np.int32).tolist())
        if not _grid_index_in_bounds(idx, occupancy_map.shape):
            return False
        if occupancy_map[idx] == 2:
            return False
        if not np.isfinite(sdf_map[idx]) or float(sdf_map[idx]) > max_allowed_sdf:
            return False
    return True


def _shortcut_prune_path(
    path: list[tuple[int, int, int]],
    sdf_map: np.ndarray,
    occupancy_map: np.ndarray,
    resolution: float,
    max_segment_m: float = 1.0,
    max_skip_nodes: int = 30,
    max_prune_nodes: int = 100,
) -> list[tuple[int, int, int]]:
    if len(path) <= 2:
        return path
    prune_end = min(len(path), max_prune_nodes)
    prune_path = path[:prune_end]
    tail = path[prune_end:]
    pruned = [prune_path[0]]
    i = 0
    while i < len(prune_path) - 1:
        farthest = i + 1
        upper = min(len(prune_path) - 1, i + max_skip_nodes)
        for j in range(upper, i, -1):
            if _segment_is_shortcut_safe(
                prune_path[i], prune_path[j], sdf_map, occupancy_map, resolution, max_segment_m=max_segment_m
            ):
                farthest = j
                break
        pruned.append(prune_path[farthest])
        i = farthest
    return pruned + tail


def _plan_sdf_path_between_points(
    start_position: np.ndarray,
    goal_position: np.ndarray,
    occupancy_map: np.ndarray,
    occupancy_meta: np.ndarray,
    sdf_map: np.ndarray,
    stop_distance: float = 0.2,
) -> tuple[np.ndarray | None, str]:
    origin = occupancy_meta[:3].astype(np.float32)
    resolution = float(occupancy_meta[3])
    start_idx = _world_to_grid_index(start_position, origin, resolution)
    goal_idx = _world_to_grid_index(goal_position, origin, resolution)
    if not _grid_index_in_bounds(start_idx, occupancy_map.shape):
        return None, f"Start out of map bounds: {start_idx}"
    if not _grid_index_in_bounds(goal_idx, occupancy_map.shape):
        return None, f"Goal out of map bounds: {goal_idx}"
    if occupancy_map[start_idx] == 2:
        return None, f"Start is occupied: {start_idx}"
    if occupancy_map[goal_idx] == 2:
        return None, f"Goal is occupied: {goal_idx}"
    sdf_start_path = search_close_to_sdf_map(start_idx, sdf_map, occupancy_map, stop_distance)
    sdf_goal_path = search_close_to_sdf_map(goal_idx, sdf_map, occupancy_map, stop_distance)
    if len(sdf_start_path) == 0:
        return None, f"No low-SDF corridor reachable from start: {start_idx}"
    if len(sdf_goal_path) == 0:
        return None, f"No low-SDF corridor reachable from goal: {goal_idx}"
    sdf_start = sdf_start_path[-1]
    sdf_goal = sdf_goal_path[-1]
    path_sdf = search_within_sdf_map(sdf_start, sdf_goal, sdf_map, occupancy_map, resolution)
    if len(path_sdf) == 0:
        return None, f"No SDF path between corridor voxels: {sdf_start} -> {sdf_goal}"
    raw_path = sdf_start_path + path_sdf + sdf_goal_path[::-1]
    pruned_path = _shortcut_prune_path(raw_path, sdf_map, occupancy_map, resolution)
    world_path = _grid_indices_to_world(pruned_path, origin, resolution)
    return world_path, f"SDF path OK: raw={len(raw_path)}, pruned={len(pruned_path)}, start_idx={start_idx}, goal_idx={goal_idx}"


# ---------------------------------------------------------------------------
# Path JSON helpers  (from path_editor)
# ---------------------------------------------------------------------------

def _random_color() -> tuple[int, int, int]:
    return tuple(int(x) for x in np.random.randint(40, 255, size=3))


def _load_paths_json(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        raw = json.load(f)
    paths: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        path_id = int(key)
        points = [np.asarray(p, dtype=np.float32) for p in value.get("points", [])]
        paths[path_id] = {
            "id": int(value.get("id", path_id)),
            "name": str(value.get("name", f"Path_{path_id}")),
            "points": points,
            "color": tuple(value.get("color", _random_color())),
        }
    return paths


def _save_paths_json(path: Path, paths: dict[int, dict[str, Any]]) -> None:
    serializable = {}
    for path_id, item in paths.items():
        serializable[str(path_id)] = {
            "id": int(item["id"]),
            "name": item["name"],
            "points": [np.asarray(p).tolist() for p in item["points"]],
            "color": list(item["color"]),
        }
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)


# ---------------------------------------------------------------------------
# POI UI helpers  (from feat/editor-3d-occupancy-toggle)
# ---------------------------------------------------------------------------

def _poi_path_role_label(poi_index: int, nav_state: dict | None) -> str:
    if nav_state is None:
        return "-"
    is_start = nav_state.get("start_poi_id") == poi_index
    is_goal = nav_state.get("goal_poi_id") == poi_index
    if is_start and is_goal:
        return "Start & End"
    if is_start:
        return "Start"
    if is_goal:
        return "End"
    return "-"


def create_poi_ui(
    server,
    poi_list_container,
    poi_index: int,
    poi_points: dict,
    sphere_handle: viser.SceneHandle,
    nav_state: dict | None = None,
    refresh_nav_markers=None,
):
    with poi_list_container:
        with server.gui.add_folder(f"POI_{poi_index}") as poi_container:
            role_label = None
            if nav_state is not None:
                role_label = server.gui.add_text(
                    "Path Role",
                    initial_value=_poi_path_role_label(poi_index, nav_state),
                )
                nav_state.setdefault("poi_role_labels", {})[poi_index] = role_label
            gui_vector3 = server.gui.add_vector3(
                "Position",
                initial_value=poi_points[poi_index]['position'],
                step=0.25,
            )
            scale = server.gui.add_slider("Scale", min=0.1, max=5.0, step=0.05, initial_value=0.1)
            color_r_slider = server.gui.add_slider("Color R", min=0, max=255, step=1, initial_value=int(sphere_handle.color[0]))
            color_g_slider = server.gui.add_slider("Color G", min=0, max=255, step=1, initial_value=int(sphere_handle.color[1]))
            color_b_slider = server.gui.add_slider("Color B", min=0, max=255, step=1, initial_value=int(sphere_handle.color[2]))
            set_start_button = None
            set_goal_button = None
            if nav_state is not None:
                set_start_button = server.gui.add_button("Set as Start", color=(40, 220, 80))
                set_goal_button = server.gui.add_button("Set as Goal", color=(255, 80, 80))
            delete_button = server.gui.add_button("Delete POI", color=(255, 0, 0))

    if set_start_button is not None:
        @set_start_button.on_click
        def _(_) -> None:
            nav_state["start_poi_id"] = poi_index
            if refresh_nav_markers is not None:
                refresh_nav_markers()

    if set_goal_button is not None:
        @set_goal_button.on_click
        def _(_) -> None:
            nav_state["goal_poi_id"] = poi_index
            if refresh_nav_markers is not None:
                refresh_nav_markers()

    def update_scale(event):
        sphere_handle.radius = scale.value
    scale.on_update(update_scale)

    def update_color(event):
        sphere_handle.color = (color_r_slider.value, color_g_slider.value, color_b_slider.value)
    color_r_slider.on_update(update_color)
    color_g_slider.on_update(update_color)
    color_b_slider.on_update(update_color)

    gizmo = server.scene.add_transform_controls(
        f"/{poi_points[poi_index]['name']}_gizmo",
        position=poi_points[poi_index]['position'],
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    def on_gizmo_update(event):
        sphere_handle.position = event.target.position
        gui_vector3.value = event.target.position
        poi_points[poi_index]['position'] = np.asarray(event.target.position, dtype=np.float32)
        if refresh_nav_markers is not None and nav_state is not None and (
            nav_state.get("start_poi_id") == poi_index or nav_state.get("goal_poi_id") == poi_index
        ):
            refresh_nav_markers()
    gizmo.on_update(on_gizmo_update)

    def on_vector3_update(event):
        new_pos = np.asarray(gui_vector3.value, dtype=np.float32)
        sphere_handle.position = new_pos
        gizmo.position = new_pos
        poi_points[poi_index]['position'] = new_pos
        if refresh_nav_markers is not None and nav_state is not None and (
            nav_state.get("start_poi_id") == poi_index or nav_state.get("goal_poi_id") == poi_index
        ):
            refresh_nav_markers()
    gui_vector3.on_update(on_vector3_update)

    @delete_button.on_click
    def _(_) -> None:
        del poi_points[poi_index]
        if nav_state is not None:
            if nav_state.get("start_poi_id") == poi_index:
                nav_state["start_poi_id"] = None
            if nav_state.get("goal_poi_id") == poi_index:
                nav_state["goal_poi_id"] = None
            nav_state.setdefault("poi_role_labels", {}).pop(poi_index, None)
        poi_container.remove()
        sphere_handle.remove()
        gizmo.remove()
        if refresh_nav_markers is not None:
            refresh_nav_markers()


# ---------------------------------------------------------------------------
# ROS2 node  (from poi_editor)
# ---------------------------------------------------------------------------

class RelocalizationPose(Node):
    def __init__(self, viser_server: viser.ViserServer):
        super().__init__('relocalization_pose')
        self.viser_server = viser_server
        self.relocalization_pose_sub = self.create_subscription(
            Odometry, '/map/relocalization', self.relocalization_pose_callback, 10
        )
        self.global_plan_sub = self.create_subscription(
            nav_msgs.msg.Path, '/mapping/global_plan', self.global_plan_callback, 10
        )
        self.planning_path_sub = self.create_subscription(
            nav_msgs.msg.Path, '/planning/trajectory_path', self.planning_path_callback, 10
        )
        self.targegt_pose_sub = self.create_subscription(
            Odometry, "/control/target_pose", self.target_pose_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/slam/odometry", self.odometry_callback, 10
        )
        self.current_pose_in_map_sub = self.create_subscription(
            Odometry, "/mapping/current_pose_in_map", self.current_pose_in_map_callback, 10
        )

    def relocalization_pose_callback(self, msg: Odometry):
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.viser_server.scene.add_icosphere(
            "/relocalization_pose", color=(255, 0, 0), position=position, radius=0.1
        )

    def global_plan_callback(self, msg: Path):
        points = [
            np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
            for p in msg.poses
        ]
        if len(points) < 2:
            return
        line_segments = np.array([[points[i - 1], points[i]] for i in range(1, len(points))])
        N = line_segments.shape[0]
        colors = np.zeros((N, 2, 3))
        colors[:, 0, :] = (0, 255, 0)
        colors[:, 1, :] = (0, 255, 0)
        self.viser_server.scene.add_line_segments(
            "/global_plan", points=line_segments, colors=colors, line_width=3
        )

    def planning_path_callback(self, msg: Path):
        points = [
            np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
            for p in msg.poses
        ]
        if len(points) < 2:
            return
        line_segments = np.array([[points[i - 1], points[i]] for i in range(1, len(points))])
        N = line_segments.shape[0]
        colors = np.zeros((N, 2, 3))
        colors[:, 0, :] = (0, 0, 255)
        colors[:, 1, :] = (0, 0, 255)
        self.viser_server.scene.add_line_segments(
            "/planning_path", points=line_segments, colors=colors, line_width=3
        )

    def odometry_callback(self, msg: Odometry):
        odom, _ = msg2np(msg)
        xyzw = matrix_to_quat(odom[:3, :3])
        position = odom[:3, 3]
        self.viser_server.scene.add_transform_controls(
            "/odom_gizmo", position=position, wxyz=(xyzw[3], xyzw[0], xyzw[1], xyzw[2])
        )

    def target_pose_callback(self, msg: Odometry):
        odom, _ = msg2np(msg)
        xyzw = matrix_to_quat(odom[:3, :3])
        position = odom[:3, 3]
        self.viser_server.scene.add_transform_controls(
            "/target_pose_gizmo", position=position, wxyz=(xyzw[3], xyzw[0], xyzw[1], xyzw[2])
        )

    def current_pose_in_map_callback(self, msg: Odometry):
        odom, _ = msg2np(msg)
        xyzw = matrix_to_quat(odom[:3, :3])
        position = odom[:3, 3]
        self.viser_server.scene.add_transform_controls(
            "/current_pose_in_map_gizmo",
            position=position,
            wxyz=(xyzw[3], xyzw[0], xyzw[1], xyzw[2]),
        )


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Args:
    tinynav_map_path: Path
    """Tinynav map directory containing occupancy_grid.npy, occupancy_meta.npy, sdf_map.npy, etc."""

    paths_json_name: str = "paths.json"
    """Editable path control-point file saved under tinynav_map_path."""

    sdf_map_name: str = "sdf_map.npy"
    """Active SDF path-map file used by Tinynav; Build/Replace overwrites this file."""

    default_sdf_map_name: str = "sdf_map.default.npy"
    """Backup of the original/default SDF path-map, used by Restore Default."""

    path_radius_m: float = 0.2
    """SDF distance threshold; voxels near edited paths become low-SDF corridor."""

    max_preview_points: int = 300_000
    """Maximum number of low-SDF voxels rendered in the Viser preview."""

    host: str = "0.0.0.0"
    port: int = 8080


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    map_dir = args.tinynav_map_path

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------
    poi_points: dict[int, dict] = {}
    poi_id_counter = 0
    nav_state: dict = {
        "start_poi_id": None,
        "goal_poi_id": None,
        "start_marker": None,
        "goal_marker": None,
        "path_handle": None,
        "poi_role_labels": {},
    }

    def refresh_poi_role_labels() -> None:
        for poi_id, label in list(nav_state.get("poi_role_labels", {}).items()):
            if poi_id in poi_points:
                label.value = _poi_path_role_label(poi_id, nav_state)
            else:
                nav_state["poi_role_labels"].pop(poi_id, None)

    def refresh_nav_markers() -> None:
        refresh_poi_role_labels()
        for marker_key, poi_key, color, radius in [
            ("start_marker", "start_poi_id", (0, 255, 0), 0.25),
            ("goal_marker", "goal_poi_id", (255, 0, 0), 0.25),
        ]:
            handle = nav_state.get(marker_key)
            if handle is not None:
                handle.remove()
                nav_state[marker_key] = None
            poi_id = nav_state.get(poi_key)
            if poi_id is None or poi_id not in poi_points:
                continue
            nav_state[marker_key] = server.scene.add_icosphere(
                f"/sdf_path_test/{poi_key}",
                radius=radius,
                color=color,
                position=np.asarray(poi_points[poi_id]["position"], dtype=np.float32),
            )

    # ------------------------------------------------------------------
    # Load POIs
    # ------------------------------------------------------------------
    if os.path.exists(f"{map_dir}/pois.json"):
        with open(f"{map_dir}/pois.json", "r") as f:
            poi_points = json.load(f)
            poi_points = {int(k): v for k, v in poi_points.items()}
            for k, v in poi_points.items():
                v['position'] = np.array(v['position'])
            poi_id_counter = max(map(lambda x: int(x), poi_points.keys())) + 1

    # ------------------------------------------------------------------
    # POI management UI
    # ------------------------------------------------------------------
    with server.gui.add_folder("Points of Interest (POI)") as _:
        add_poi_button = server.gui.add_button("Add POI Point")
        add_save_poi_button = server.gui.add_button("Save POI")

        @add_save_poi_button.on_click
        def _(_) -> None:
            with open(f"{map_dir}/pois.json", "w") as f:
                json.dump(
                    poi_points, f, indent=2,
                    default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                )

        poi_list_container = server.gui.add_folder("POI List")
        for poi_id, poi_point in poi_points.items():
            sphere_handle = server.scene.add_icosphere(
                f"/{poi_point['name']}",
                radius=0.1,
                color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                position=poi_point['position'],
            )
            create_poi_ui(server, poi_list_container, int(poi_id), poi_points, sphere_handle, nav_state, refresh_nav_markers)

        @add_poi_button.on_click
        def _(_) -> None:
            nonlocal poi_id_counter
            poi_id = poi_id_counter
            poi_id_counter += 1
            poi_name = f"POI_{poi_id}"
            if len(poi_points) > 0:
                previous_poi_id = max(poi_points.keys())
                previous_position = np.asarray(poi_points[previous_poi_id]['position'], dtype=float)
                poi_position = previous_position + np.array([0.3, 0.0, 0.0])
            else:
                poi_position = np.random.randn(3)
            poi_points[poi_id] = {
                'id': poi_id,
                'name': poi_name,
                'position': poi_position,
            }
            sphere_handle = server.scene.add_icosphere(
                f"/{poi_name}",
                radius=0.1,
                color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                position=poi_points[poi_id]['position'],
            )
            create_poi_ui(server, poi_list_container, poi_id, poi_points, sphere_handle, nav_state, refresh_nav_markers)

    # ------------------------------------------------------------------
    # Occupancy grid + SDF map
    # ------------------------------------------------------------------
    occupancy_grid_path = map_dir / "occupancy_grid.npy"
    occupancy_meta_path = map_dir / "occupancy_meta.npy"
    sdf_map_path = map_dir / args.sdf_map_name
    default_sdf_map_path = map_dir / args.default_sdf_map_name

    occupancy_grid = None
    occupancy_meta = None
    sdf_map = None

    if occupancy_grid_path.exists() and occupancy_meta_path.exists():
        print(f"Loading occupancy grid from {map_dir}")
        occupancy_grid = np.load(occupancy_grid_path)
        occupancy_meta = np.load(occupancy_meta_path)
        origin = occupancy_meta[:3]
        resolution = float(occupancy_meta[3])
        shape = tuple(int(v) for v in occupancy_grid.shape)
        print(f"Occupancy grid shape: {occupancy_grid.shape}")
        print(f"Origin: ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})")
        print(f"Resolution: {resolution:.3f} m")

        x_y_plane = np.max(occupancy_grid, axis=2)
        unknown_indices = np.argwhere(x_y_plane == 0)
        free_indices = np.argwhere(x_y_plane == 1)
        occupied_indices = np.argwhere(x_y_plane == 2)
        z_plane = float(origin[2])

        def _xy_to_world_points(xy_indices: np.ndarray) -> np.ndarray:
            if len(xy_indices) == 0:
                return np.array([]).reshape(0, 3)
            points = np.zeros((len(xy_indices), 3), dtype=np.float32)
            points[:, 0] = float(origin[0]) + xy_indices[:, 0] * resolution
            points[:, 1] = float(origin[1]) + xy_indices[:, 1] * resolution
            points[:, 2] = z_plane
            return points

        def _xyz_to_world_points(xyz_indices: np.ndarray) -> np.ndarray:
            if len(xyz_indices) == 0:
                return np.array([]).reshape(0, 3)
            points = np.zeros((len(xyz_indices), 3), dtype=np.float32)
            points[:, 0] = float(origin[0]) + xyz_indices[:, 0] * resolution
            points[:, 1] = float(origin[1]) + xyz_indices[:, 1] * resolution
            points[:, 2] = float(origin[2]) + xyz_indices[:, 2] * resolution
            return points

        # --- 2D projection handles ---
        unknown_handle = None
        free_handle = None
        occupied_handle = None
        sdf_search_handle = None

        unknown_points = _xy_to_world_points(unknown_indices)
        free_points = _xy_to_world_points(free_indices)
        occupied_points = _xy_to_world_points(occupied_indices)

        if len(unknown_points) > 0:
            unknown_handle = server.scene.add_point_cloud(
                "/occupancy_2d/unknown",
                points=unknown_points,
                colors=np.zeros((len(unknown_points), 3), dtype=np.float32),
                point_size=resolution * 0.8,
                point_shape="rounded",
            )
        if len(free_points) > 0:
            free_handle = server.scene.add_point_cloud(
                "/occupancy_2d/free",
                points=free_points,
                colors=np.tile(np.array([[0.2, 0.4, 1.0]], dtype=np.float32), (len(free_points), 1)),
                point_size=resolution * 0.8,
                point_shape="rounded",
            )
        if len(occupied_points) > 0:
            occupied_column_height = 0.8
            z_levels = np.arange(
                z_plane + resolution * 0.5, z_plane + occupied_column_height, resolution, dtype=np.float32
            )
            occupied_column_points = np.repeat(occupied_points, len(z_levels), axis=0)
            occupied_column_points[:, 2] = np.tile(z_levels, len(occupied_points))
            wall_zero_bgr = cv2.applyColorMap(np.array([[255]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            wall_zero_rgb = wall_zero_bgr[::-1].astype(np.float32) / 255.0
            wall_light_rgb = np.clip(0.55 * wall_zero_rgb + 0.45 * np.ones(3, dtype=np.float32), 0.0, 1.0)
            occupied_handle = server.scene.add_point_cloud(
                "/occupancy_2d/occupied",
                points=occupied_column_points,
                colors=np.tile(wall_light_rgb[None, :], (len(occupied_column_points), 1)),
                point_size=resolution * 0.8,
                point_shape="rounded",
            )

        # --- SDF search region (sdf < threshold) ---
        if sdf_map_path.exists():
            sdf_map = np.load(sdf_map_path).astype(np.float32)
            if sdf_map.shape == occupancy_grid.shape:
                traversable_mask = occupancy_grid != 2
                sdf_valid_mask = np.logical_and(traversable_mask, np.isfinite(sdf_map))
                sdf_search_threshold = 0.2
                sdf_search_mask = np.logical_and(sdf_valid_mask, sdf_map < sdf_search_threshold)
                sdf_search_indices_all = np.argwhere(sdf_search_mask)
                max_search_points = 300_000
                if len(sdf_search_indices_all) > max_search_points:
                    stride = int(np.ceil(len(sdf_search_indices_all) / max_search_points))
                    sdf_search_indices = sdf_search_indices_all[::stride]
                else:
                    sdf_search_indices = sdf_search_indices_all
                sdf_search_points = _xyz_to_world_points(sdf_search_indices)
                if len(sdf_search_points) > 0:
                    sdf_search_handle = server.scene.add_point_cloud(
                        "/occupancy_2d/sdf_search_region",
                        points=sdf_search_points,
                        colors=np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (len(sdf_search_points), 1)),
                        point_size=resolution * 0.8,
                        point_shape="rounded",
                    )

        # --- 3D occupancy voxels (true per-voxel height) ---
        max_3d_points = 300_000

        def _capped_indices(mask: np.ndarray) -> np.ndarray:
            indices_all = np.argwhere(mask)
            if len(indices_all) > max_3d_points:
                stride = int(np.ceil(len(indices_all) / max_3d_points))
                return indices_all[::stride]
            return indices_all

        occupied_3d_indices = _capped_indices(occupancy_grid == 2)
        free_3d_indices = _capped_indices(occupancy_grid == 1)
        occupied_3d_points = _xyz_to_world_points(occupied_3d_indices)
        free_3d_points = _xyz_to_world_points(free_3d_indices)

        occupied_3d_handle = None
        free_3d_handle = None
        if len(occupied_3d_points) > 0:
            occupied_3d_handle = server.scene.add_point_cloud(
                "/occupancy_3d/occupied",
                points=occupied_3d_points,
                colors=np.tile(np.array([[0.6, 0.6, 0.6]], dtype=np.float32), (len(occupied_3d_points), 1)),
                point_size=resolution * 0.8,
                point_shape="rounded",
            )
            occupied_3d_handle.visible = False
        if len(free_3d_points) > 0:
            free_3d_handle = server.scene.add_point_cloud(
                "/occupancy_3d/free",
                points=free_3d_points,
                colors=np.tile(np.array([[0.2, 0.4, 1.0]], dtype=np.float32), (len(free_3d_points), 1)),
                point_size=resolution * 0.8,
                point_shape="rounded",
            )
            free_3d_handle.visible = False

        # Default visibility
        if unknown_handle is not None:
            unknown_handle.visible = False
        if free_handle is not None:
            free_handle.visible = True
        if occupied_handle is not None:
            occupied_handle.visible = True
        if sdf_search_handle is not None:
            sdf_search_handle.visible = False

        # --- Occupancy GUI folder ---
        point_size_init = resolution * 0.8
        point_size_max = max(0.1, point_size_init)
        with server.gui.add_folder("Occupancy Map") as _:
            show_free = server.gui.add_checkbox("Show Free", initial_value=True)
            show_occupied = server.gui.add_checkbox("Show Occupied", initial_value=True)
            show_sdf_search_region = server.gui.add_checkbox("Show SDF<0.2m Region", initial_value=False)
            show_occupied_3d = server.gui.add_checkbox("Show 3D Occupied (true height)", initial_value=False)
            show_free_3d = server.gui.add_checkbox("Show 3D Free (true height)", initial_value=False)
            point_size_slider = server.gui.add_slider(
                "Point Size", min=0.001, max=point_size_max, step=0.001, initial_value=point_size_init
            )

            @show_free.on_update
            def _(_) -> None:
                if free_handle is not None:
                    free_handle.visible = show_free.value

            @show_occupied.on_update
            def _(_) -> None:
                if occupied_handle is not None:
                    occupied_handle.visible = show_occupied.value

            @show_sdf_search_region.on_update
            def _(_) -> None:
                if sdf_search_handle is not None:
                    sdf_search_handle.visible = show_sdf_search_region.value

            @show_occupied_3d.on_update
            def _(_) -> None:
                if occupied_3d_handle is not None:
                    occupied_3d_handle.visible = show_occupied_3d.value

            @show_free_3d.on_update
            def _(_) -> None:
                if free_3d_handle is not None:
                    free_3d_handle.visible = show_free_3d.value

            @point_size_slider.on_update
            def _(_) -> None:
                if unknown_handle is not None:
                    unknown_handle.point_size = point_size_slider.value
                if free_handle is not None:
                    free_handle.point_size = point_size_slider.value
                if occupied_handle is not None:
                    occupied_handle.point_size = point_size_slider.value
                if sdf_search_handle is not None:
                    sdf_search_handle.point_size = point_size_slider.value
                if occupied_3d_handle is not None:
                    occupied_3d_handle.point_size = point_size_slider.value
                if free_3d_handle is not None:
                    free_3d_handle.point_size = point_size_slider.value
    else:
        print(f"Warning: Occupancy grid files not found in {map_dir}")
        origin = None
        resolution = None
        shape = None

    # ------------------------------------------------------------------
    # SDF Path Editor (merged from path_editor.py)
    # ------------------------------------------------------------------
    path_json_path = map_dir / args.paths_json_name
    path_sdf_map_path = sdf_map_path  # same file: sdf_map.npy
    # Load default SDF backup
    default_sdf_map = None
    if default_sdf_map_path.exists():
        default_sdf_map = np.load(default_sdf_map_path).astype(np.float32)
    elif sdf_map is not None:
        default_sdf_map = sdf_map.copy()

    paths_data = _load_paths_json(path_json_path)
    path_id_counter = (max(paths_data.keys()) + 1) if paths_data else 0
    selected_path_id: int | None = next(iter(paths_data.keys()), None)

    path_line_handles: dict[int, viser.SceneHandle] = {}
    path_point_handles: dict[tuple[int, int], viser.SceneHandle] = {}
    path_gizmo_handles: dict[tuple[int, int], viser.SceneHandle] = {}
    sdf_preview_handle: viser.SceneHandle | None = None

    def _default_new_path_point() -> np.ndarray:
        if occupancy_grid is not None:
            traversable = np.argwhere(occupancy_grid != 2)
            if len(traversable) > 0:
                center_idx = traversable[len(traversable) // 2]
                return _grid_to_world(center_idx[None, :], origin, resolution)[0]
        return np.asarray(origin, dtype=np.float32).copy()

    def _refresh_path_line(pid: int) -> None:
        if pid in path_line_handles:
            path_line_handles[pid].remove()
            del path_line_handles[pid]
        item = paths_data[pid]
        segments = _line_segments_from_points(item["points"])
        if len(segments) == 0:
            return
        color = np.asarray(item["color"], dtype=np.float32) / 255.0
        colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
        colors[:, :, :] = color
        path_line_handles[pid] = server.scene.add_line_segments(
            f"/paths/{item['name']}/line", points=segments, colors=colors, line_width=4.0
        )

    def _remove_path_handles(pid: int) -> None:
        if pid in path_line_handles:
            path_line_handles[pid].remove()
            del path_line_handles[pid]
        for key in list(path_point_handles.keys()):
            if key[0] == pid:
                path_point_handles[key].remove()
                del path_point_handles[key]
        for key in list(path_gizmo_handles.keys()):
            if key[0] == pid:
                path_gizmo_handles[key].remove()
                del path_gizmo_handles[key]

    def _refresh_path(pid: int) -> None:
        _remove_path_handles(pid)
        item = paths_data[pid]
        color = tuple(int(c) for c in item["color"])
        points = item["points"]
        segments = _line_segments_from_points(points)
        if len(segments) > 0:
            colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
            colors[:, :, :] = np.asarray(color, dtype=np.float32) / 255.0
            path_line_handles[pid] = server.scene.add_line_segments(
                f"/paths/{item['name']}/line", points=segments, colors=colors, line_width=4.0
            )
        for point_idx, point in enumerate(points):
            key = (pid, point_idx)
            point_name = f"/paths/{item['name']}/point_{point_idx}"
            path_point_handles[key] = server.scene.add_icosphere(
                point_name, radius=max(0.05, resolution or 0.05), color=color, position=np.asarray(point, dtype=np.float32)
            )
            gizmo = server.scene.add_transform_controls(
                f"{point_name}_gizmo", position=np.asarray(point, dtype=np.float32), wxyz=(1.0, 0.0, 0.0, 0.0)
            )
            path_gizmo_handles[key] = gizmo

            @gizmo.on_update
            def _(event, p_id=pid, p_idx=point_idx, handle=path_point_handles[key]) -> None:
                new_pos = np.asarray(event.target.position, dtype=np.float32)
                paths_data[p_id]["points"][p_idx] = new_pos
                handle.position = new_pos
                _refresh_path_line(p_id)

    def _refresh_all_paths() -> None:
        for pid in list(paths_data.keys()):
            _refresh_path(pid)

    def _refresh_sdf_preview() -> None:
        nonlocal sdf_preview_handle
        if sdf_preview_handle is not None:
            sdf_preview_handle.remove()
            sdf_preview_handle = None
        if sdf_map is None:
            return
        traversable_mask = occupancy_grid != 2
        mask = np.logical_and.reduce((
            traversable_mask, np.isfinite(sdf_map), sdf_map < args.path_radius_m
        ))
        indices_all = np.argwhere(mask)
        if len(indices_all) == 0:
            return
        if len(indices_all) > args.max_preview_points:
            stride = int(np.ceil(len(indices_all) / args.max_preview_points))
            indices = indices_all[::stride]
        else:
            indices = indices_all
        points = _grid_to_world(indices, origin, resolution)
        colors = np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (len(points), 1))
        sdf_preview_handle = server.scene.add_point_cloud(
            "/sdf_path_preview/low_sdf_voxels",
            points=points, colors=colors,
            point_size=resolution * 0.8, point_shape="rounded",
        )

    def _ensure_default_backup() -> None:
        if default_sdf_map_path.exists():
            return
        if sdf_map is None:
            return
        np.save(default_sdf_map_path, sdf_map)

    # Build SDF path editor UI
    with server.gui.add_folder("SDF Path Editor") as _:
        path_status = server.gui.add_text("Status", initial_value="Ready")
        selected_path_gui = server.gui.add_number(
            "Selected Path ID", initial_value=selected_path_id or 0, step=1
        )
        add_path_btn = server.gui.add_button("Add Path")
        add_waypoint_btn = server.gui.add_button("Add Waypoint To Selected")
        delete_last_waypoint_btn = server.gui.add_button("Delete Last Waypoint")
        delete_path_btn = server.gui.add_button("Delete Selected Path", color=(255, 80, 80))
        replace_sdf_btn = server.gui.add_button("Build / Replace Path Map", color=(80, 200, 80))
        restore_default_sdf_btn = server.gui.add_button("Restore Default Map", color=(80, 120, 255))

        @selected_path_gui.on_update
        def _(_) -> None:
            nonlocal selected_path_id
            pid = int(selected_path_gui.value)
            selected_path_id = pid if pid in paths_data else None
            path_status.value = f"Selected path: {selected_path_id}"

        @add_path_btn.on_click
        def _(_) -> None:
            nonlocal path_id_counter, selected_path_id
            pid = path_id_counter
            path_id_counter += 1
            selected_path_gui.value = pid
            selected_path_id = pid
            paths_data[pid] = {
                "id": pid,
                "name": f"Path_{pid}",
                "points": [_default_new_path_point()],
                "color": _random_color(),
            }
            _refresh_path(pid)
            path_status.value = f"Added Path_{pid}"

        @add_waypoint_btn.on_click
        def _(_) -> None:
            if selected_path_id is None:
                path_status.value = "No selected path"
                return
            points = paths_data[selected_path_id]["points"]
            if points:
                new_point = np.asarray(points[-1], dtype=np.float32) + np.array([resolution * 5, 0.0, 0.0], dtype=np.float32)
            else:
                new_point = _default_new_path_point()
            points.append(new_point)
            _refresh_path(selected_path_id)
            path_status.value = f"Added waypoint to Path_{selected_path_id}"

        @delete_last_waypoint_btn.on_click
        def _(_) -> None:
            if selected_path_id is None:
                path_status.value = "No selected path"
                return
            points = paths_data[selected_path_id]["points"]
            if points:
                points.pop()
            _refresh_path(selected_path_id)
            path_status.value = f"Deleted last waypoint from Path_{selected_path_id}"

        @delete_path_btn.on_click
        def _(_) -> None:
            nonlocal selected_path_id
            if selected_path_id is None:
                path_status.value = "No selected path"
                return
            pid = selected_path_id
            _remove_path_handles(pid)
            del paths_data[pid]
            selected_path_id = next(iter(paths_data.keys()), None)
            selected_path_gui.value = selected_path_id or 0
            path_status.value = f"Deleted Path_{pid}"

        @replace_sdf_btn.on_click
        def _(_) -> None:
            nonlocal sdf_map
            _save_paths_json(path_json_path, paths_data)
            _ensure_default_backup()
            sdf_map = build_sdf_from_paths(paths_data, origin, resolution, shape)
            np.save(path_sdf_map_path, sdf_map)
            _refresh_sdf_preview()
            path_status.value = f"Built path map: {path_sdf_map_path.name}; saved {path_json_path.name}"

        @restore_default_sdf_btn.on_click
        def _(_) -> None:
            nonlocal sdf_map
            if default_sdf_map is None:
                path_status.value = "No default map available to restore"
                return
            np.save(path_sdf_map_path, default_sdf_map)
            sdf_map = default_sdf_map.copy()
            _refresh_sdf_preview()
            path_status.value = f"Restored default map into {path_sdf_map_path.name}"

    # Initial path rendering
    if occupancy_grid is not None:
        _refresh_all_paths()
        _refresh_sdf_preview()

    # ------------------------------------------------------------------
    # SDF POI Path Test (from feat/editor-3d-occupancy-toggle)
    # ------------------------------------------------------------------
    if occupancy_grid is not None and occupancy_meta is not None and sdf_map_path.exists():
        if sdf_map is None:
            loaded_sdf_map = np.load(sdf_map_path).astype(np.float32)
            if loaded_sdf_map.shape == occupancy_grid.shape:
                sdf_map = loaded_sdf_map
        with server.gui.add_folder("SDF POI Path Test") as _:
            poi_path_status = server.gui.add_text(
                "Status", initial_value="Pick POI start/goal, then Plan SDF Path"
            )
            stop_distance_slider = server.gui.add_slider(
                "SDF Stop Distance", min=0.05, max=1.0, step=0.05, initial_value=0.2
            )
            line_width_slider = server.gui.add_slider(
                "Path Line Width", min=1.0, max=20.0, step=1.0, initial_value=8.0
            )
            plan_button = server.gui.add_button("Plan SDF Path", color=(80, 200, 80))
            clear_button = server.gui.add_button("Clear Planned Path")

            @plan_button.on_click
            def _(_) -> None:
                start_id = nav_state.get("start_poi_id")
                goal_id = nav_state.get("goal_poi_id")
                if start_id is None or goal_id is None:
                    poi_path_status.value = "Need both start and goal POIs"
                    print(poi_path_status.value)
                    return
                if start_id not in poi_points or goal_id not in poi_points:
                    poi_path_status.value = "Start/goal POI was deleted"
                    print(poi_path_status.value)
                    refresh_nav_markers()
                    return
                if sdf_map is None:
                    poi_path_status.value = f"Missing/invalid sdf_map.npy: {sdf_map_path}"
                    print(poi_path_status.value)
                    return
                if nav_state.get("path_handle") is not None:
                    nav_state["path_handle"].remove()
                    nav_state["path_handle"] = None
                start_position = np.asarray(poi_points[start_id]["position"], dtype=np.float32)
                goal_position = np.asarray(poi_points[goal_id]["position"], dtype=np.float32)
                t0 = time.time()
                world_path, message = _plan_sdf_path_between_points(
                    start_position, goal_position, occupancy_grid, occupancy_meta, sdf_map,
                    stop_distance=float(stop_distance_slider.value),
                )
                elapsed_ms = (time.time() - t0) * 1000.0
                if world_path is None or len(world_path) < 2:
                    poi_path_status.value = f"Plan failed ({elapsed_ms:.0f} ms): {message}"
                    print(poi_path_status.value)
                    return
                segments = np.stack([world_path[:-1], world_path[1:]], axis=1)
                colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
                colors[:, :, :] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
                nav_state["path_handle"] = server.scene.add_line_segments(
                    "/sdf_path_test/planned_path",
                    points=segments, colors=colors,
                    line_width=float(line_width_slider.value),
                )
                refresh_nav_markers()
                path_len = float(np.sum(np.linalg.norm(np.diff(world_path, axis=0), axis=1)))
                poi_path_status.value = f"{message}; len={path_len:.2f}m; time={elapsed_ms:.0f}ms"
                print(poi_path_status.value)

            @clear_button.on_click
            def _(_) -> None:
                if nav_state.get("path_handle") is not None:
                    nav_state["path_handle"].remove()
                    nav_state["path_handle"] = None
                poi_path_status.value = "Cleared planned path"
                print(poi_path_status.value)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    poses = np.load(map_dir / "poses.npy", allow_pickle=True).item()
    if (map_dir / "intrinsics.npy").exists():
        camera_K = np.load(map_dir / "intrinsics.npy", allow_pickle=True)
    elif (map_dir / "rgb_camera_intrinsics.npy").exists():
        camera_K = np.load(map_dir / "rgb_camera_intrinsics.npy", allow_pickle=True)
    else:
        raise FileNotFoundError("Neither intrinsics.npy nor rgb_camera_intrinsics.npy exists.")

    fx, _, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]
    infra1_db = _open_infra1_video_db(map_dir)
    max_camera_poses = 500
    sample_interval = max(1, math.ceil(len(poses) / max_camera_poses))
    with server.gui.add_folder("cameras") as _:
        for pose_idx, (timestamp, camera_pose) in enumerate(poses.items()):
            if pose_idx % sample_interval != 0:
                continue
            R = vtf.SO3.from_matrix(camera_pose[:3, :3])
            t = camera_pose[:3, 3]
            timestamp_str = str(timestamp)
            frustum = server.scene.add_camera_frustum(
                name=f"/cameras/camera_{timestamp}",
                fov=float(2 * np.arctan((cx / fx))),
                scale=0.01,
                aspect=float(cx / cy),
                image=None,
                wxyz=R.wxyz,
                position=t,
                format="jpeg",
                jpeg_quality=50,
            )

            @frustum.on_click
            def _(event, cam_pose=camera_pose, ts=timestamp_str, fr=frustum):
                q = matrix_to_quat(cam_pose[:3, :3])  # xyzw
                target_position = tuple(cam_pose[:3, 3].tolist())
                target_wxyz = (q[3], q[0], q[1], q[2])
                img = _load_infra1_camera_image(infra1_db, ts)
                if img is not None:
                    fr.image = img
                else:
                    print(f"timestamp {ts} don't found image")
                client = getattr(event, "client", None)
                if client is not None:
                    client.camera.position = target_position
                    client.camera.wxyz = target_wxyz
                    return
                for c in server.get_clients().values():
                    c.camera.position = target_position
                    c.camera.wxyz = target_wxyz

    # ------------------------------------------------------------------
    # Splat / point cloud
    # ------------------------------------------------------------------
    splat_path = Path(f"{map_dir}/splat.ply")
    pointcloud_path = Path(f"{map_dir}/pointcloud.ply")

    if splat_path.exists():
        print(f"Loading Gaussian splat from {splat_path}")
        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=True)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=False)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")
        gs_handle = server.scene.add_gaussian_splats(
            "/0/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )
        remove_button = server.gui.add_button("Remove splat object")

        @remove_button.on_click
        def _(_, gs_handle=gs_handle, remove_button=remove_button) -> None:
            gs_handle.remove()
            remove_button.remove()

    elif pointcloud_path.exists():
        print(f"Loading point cloud from {pointcloud_path}")
        pc_data = load_pointcloud_ply(pointcloud_path, center=False)
        pc_handle = server.scene.add_point_cloud(
            "/0/point_cloud",
            points=pc_data["positions"],
            colors=pc_data["colors"],
            point_size=0.01,
            point_shape="rounded",
        )
        with server.gui.add_folder("Point Cloud Settings") as _:
            point_size_slider = server.gui.add_slider(
                "Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
            )

            @point_size_slider.on_update
            def _(_) -> None:
                pc_handle.point_size = point_size_slider.value

        remove_button = server.gui.add_button("Remove point cloud")

        @remove_button.on_click
        def _(_, pc_handle=pc_handle, remove_button=remove_button) -> None:
            pc_handle.remove()
            remove_button.remove()
    else:
        print(f"Warning: Neither {splat_path} nor {pointcloud_path} exists. No 3D representation loaded.")

    # ------------------------------------------------------------------
    # ROS2 spin
    # ------------------------------------------------------------------
    print(f"Map editor is running at http://{args.host}:{args.port}")
    rclpy.init()
    relocalization_pose_node = RelocalizationPose(server)
    try:
        rclpy.spin(relocalization_pose_node)
        relocalization_pose_node.destroy_node()
        rclpy.shutdown()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:
        pass
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
        if infra1_db is not None:
            infra1_db.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
