from __future__ import annotations

import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import viser


@dataclass(frozen=True)
class Args:
    tinynav_map_path: Path
    """Tinynav map directory containing poses.npy."""

    output_name: str = "relocalization_mask.json"
    """Mask file saved under tinynav_map_path."""

    host: str = "0.0.0.0"
    port: int = 8082

    point_size: float = 0.05
    """Keyframe point size in meters."""

    max_occupancy_points: int = 300_000
    """Maximum occupancy/free points shown in the map background."""


def _load_poses(map_dir: Path) -> tuple[list[int], np.ndarray]:
    poses_path = map_dir / "poses.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"poses.npy not found: {poses_path}")
    poses = np.load(poses_path, allow_pickle=True).item()
    timestamps = sorted(int(t) for t in poses.keys())
    positions = np.asarray([poses[t][:3, 3] for t in timestamps], dtype=np.float32)
    return timestamps, positions


def _load_mask(mask_path: Path) -> dict[str, Any]:
    if not mask_path.exists():
        return {"version": 1, "excluded_timestamps": [], "zones": []}
    with mask_path.open("r") as f:
        raw = json.load(f)
    raw.setdefault("version", 1)
    raw.setdefault("excluded_timestamps", [])
    raw.setdefault("zones", [])
    return raw


def _save_mask(mask_path: Path, mask: dict[str, Any]) -> None:
    mask["excluded_timestamps"] = sorted({int(t) for t in mask.get("excluded_timestamps", [])})
    for zone in mask.get("zones", []):
        zone["excluded_timestamps"] = sorted({int(t) for t in zone.get("excluded_timestamps", [])})
    with mask_path.open("w") as f:
        json.dump(mask, f, indent=2)
        f.write("\n")


def _ensure_port_available(host: str, port: int) -> None:
    bind_host = "" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use. Please stop the old editor or choose another port.") from exc


def _box_line_segments(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = size.astype(np.float32) * 0.5
    c = center.astype(np.float32)
    corners = np.asarray(
        [
            [c[0] - half[0], c[1] - half[1], c[2] - half[2]],
            [c[0] + half[0], c[1] - half[1], c[2] - half[2]],
            [c[0] + half[0], c[1] + half[1], c[2] - half[2]],
            [c[0] - half[0], c[1] + half[1], c[2] - half[2]],
            [c[0] - half[0], c[1] - half[1], c[2] + half[2]],
            [c[0] + half[0], c[1] - half[1], c[2] + half[2]],
            [c[0] + half[0], c[1] + half[1], c[2] + half[2]],
            [c[0] - half[0], c[1] + half[1], c[2] + half[2]],
        ],
        dtype=np.float32,
    )
    edges = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int32,
    )
    return corners[edges]


def _points_in_box(points: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = size * 0.5
    return np.all((points >= center - half) & (points <= center + half), axis=1)


class RelocalizationMaskEditor:
    def __init__(self, args: Args):
        self.args = args
        self.map_dir = args.tinynav_map_path
        self.mask_path = self.map_dir / args.output_name
        self.timestamps, self.positions = _load_poses(self.map_dir)
        self.z_plane = float(np.median(self.positions[:, 2])) if len(self.positions) > 0 else 0.0

        self.mask = _load_mask(self.mask_path)
        self.current_selection: set[int] = set()
        extent = np.ptp(self.positions, axis=0) if len(self.positions) > 0 else np.ones(3, dtype=np.float32)
        self.default_box_center = (
            np.mean(self.positions, axis=0).astype(np.float32) if len(self.positions) > 0 else np.zeros(3)
        )
        self.default_box_size = np.maximum(extent * 0.25, np.array([1.0, 1.0, 0.5], dtype=np.float32)).astype(
            np.float32
        )
        self.boxes = self._load_boxes_from_mask()
        if not self.boxes:
            self.boxes.append({"center": self.default_box_center.copy(), "size": self.default_box_size.copy()})
        self.selected_box_idx = 0
        self.excluded = self._compute_excluded_from_boxes()

        _ensure_port_available(args.host, args.port)
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.world_axes.visible = True
        self.server.scene.set_up_direction("+z")

        self.available_handle: viser.SceneHandle | None = None
        self.excluded_handle: viser.SceneHandle | None = None
        self.selected_handle: viser.SceneHandle | None = None
        self.box_line_handles: list[viser.SceneHandle] = []
        self.box_gizmo_handles: list[viser.TransformControlsHandle] = []
        self.selected_box_number = None
        self.box_size_x_slider = None
        self.box_size_y_slider = None
        self.box_size_z_slider = None
        self.status = None

    def run(self) -> None:
        self._add_static_map_layers()
        self._add_keyframe_layers()
        self._add_ui()
        self._refresh_all()
        print(f"Relocalization mask editor is running at http://{self.args.host}:{self.args.port}")
        while True:
            time.sleep(1.0)

    def _add_static_map_layers(self) -> None:
        occupancy_path = self.map_dir / "occupancy_grid.npy"
        meta_path = self.map_dir / "occupancy_meta.npy"
        if not occupancy_path.exists() or not meta_path.exists():
            return

        occupancy = np.load(occupancy_path)
        meta = np.load(meta_path)
        origin = meta[:3].astype(np.float32)
        resolution = float(meta[3])
        xy_plane = np.max(occupancy, axis=2)

        def xy_world(indices: np.ndarray) -> np.ndarray:
            points = np.zeros((len(indices), 3), dtype=np.float32)
            points[:, 0] = float(origin[0]) + indices[:, 0] * resolution
            points[:, 1] = float(origin[1]) + indices[:, 1] * resolution
            points[:, 2] = self.z_plane - 0.03
            return points

        def xyz_world(indices: np.ndarray) -> np.ndarray:
            points = np.zeros((len(indices), 3), dtype=np.float32)
            points[:, 0] = float(origin[0]) + indices[:, 0] * resolution
            points[:, 1] = float(origin[1]) + indices[:, 1] * resolution
            points[:, 2] = float(origin[2]) + indices[:, 2] * resolution
            return points

        rng = np.random.default_rng(0)
        free_indices = np.argwhere(xy_plane == 1)
        occupied_indices = np.argwhere(xy_plane == 2)

        if len(free_indices) > self.args.max_occupancy_points:
            free_indices = free_indices[rng.choice(len(free_indices), self.args.max_occupancy_points, replace=False)]
        if len(occupied_indices) > self.args.max_occupancy_points:
            occupied_indices = occupied_indices[
                rng.choice(len(occupied_indices), self.args.max_occupancy_points, replace=False)
            ]

        free_handle = None
        occupied_handle = None
        if len(free_indices) > 0:
            free_handle = self.server.scene.add_point_cloud(
                "/map/free",
                points=xy_world(free_indices),
                colors=np.tile(np.array([[120, 120, 120]], dtype=np.uint8), (len(free_indices), 1)),
                point_size=max(0.01, resolution * 0.35),
                point_shape="square",
            )
        if len(occupied_indices) > 0:
            occupied_handle = self.server.scene.add_point_cloud(
                "/map/occupied",
                points=xy_world(occupied_indices),
                colors=np.tile(np.array([[40, 40, 40]], dtype=np.uint8), (len(occupied_indices), 1)),
                point_size=max(0.01, resolution * 0.45),
                point_shape="square",
            )

        free_3d_indices = np.argwhere(occupancy == 1)
        occupied_3d_indices = np.argwhere(occupancy == 2)
        if len(free_3d_indices) > self.args.max_occupancy_points:
            free_3d_indices = free_3d_indices[
                rng.choice(len(free_3d_indices), self.args.max_occupancy_points, replace=False)
            ]
        if len(occupied_3d_indices) > self.args.max_occupancy_points:
            occupied_3d_indices = occupied_3d_indices[
                rng.choice(len(occupied_3d_indices), self.args.max_occupancy_points, replace=False)
            ]

        free_3d_handle = None
        occupied_3d_handle = None
        if len(free_3d_indices) > 0:
            free_3d_handle = self.server.scene.add_point_cloud(
                "/map_3d/free",
                points=xyz_world(free_3d_indices),
                colors=np.tile(np.array([[80, 110, 255]], dtype=np.uint8), (len(free_3d_indices), 1)),
                point_size=max(0.01, resolution * 0.35),
                point_shape="rounded",
            )
            free_3d_handle.visible = False
        if len(occupied_3d_indices) > 0:
            occupied_3d_handle = self.server.scene.add_point_cloud(
                "/map_3d/occupied",
                points=xyz_world(occupied_3d_indices),
                colors=np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (len(occupied_3d_indices), 1)),
                point_size=max(0.01, resolution * 0.45),
                point_shape="rounded",
            )
            occupied_3d_handle.visible = False

        with self.server.gui.add_folder("Map View"):
            show_free = self.server.gui.add_checkbox("Show 2D Free", initial_value=True)
            show_occupied = self.server.gui.add_checkbox("Show 2D Occupied", initial_value=True)
            show_free_3d = self.server.gui.add_checkbox("Show 3D Free", initial_value=False)
            show_occupied_3d = self.server.gui.add_checkbox("Show 3D Occupied", initial_value=False)

            @show_free.on_update
            def _(_) -> None:
                if free_handle is not None:
                    free_handle.visible = show_free.value

            @show_occupied.on_update
            def _(_) -> None:
                if occupied_handle is not None:
                    occupied_handle.visible = show_occupied.value

            @show_free_3d.on_update
            def _(_) -> None:
                if free_3d_handle is not None:
                    free_3d_handle.visible = show_free_3d.value

            @show_occupied_3d.on_update
            def _(_) -> None:
                if occupied_3d_handle is not None:
                    occupied_3d_handle.visible = show_occupied_3d.value

    def _add_keyframe_layers(self) -> None:
        if len(self.positions) >= 2:
            segments = np.stack([self.positions[:-1], self.positions[1:]], axis=1)
            colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
            colors[:, :, :] = np.array([0.3, 0.3, 0.3], dtype=np.float32)
            self.server.scene.add_line_segments("/keyframes/trajectory", points=segments, colors=colors, line_width=1.5)

    def _add_ui(self) -> None:
        size_slider_max = np.maximum(np.ptp(self.positions, axis=0) + 5.0, np.array([1.0, 1.0, 1.0]))
        with self.server.gui.add_folder("Relocalization Mask Editor"):
            self.status = self.server.gui.add_text("Status", initial_value="Ready")
            self.selected_box_number = self.server.gui.add_number(
                "Selected Box ID", initial_value=self.selected_box_idx, step=1
            )
            add_box = self.server.gui.add_button("Add Box")
            delete_box = self.server.gui.add_button("Delete Selected Box", color=(255, 80, 80))
            self.box_size_x_slider = self.server.gui.add_slider(
                "Box Size X",
                min=0.1,
                max=float(size_slider_max[0]),
                step=0.1,
                initial_value=float(self.boxes[self.selected_box_idx]["size"][0]),
            )
            self.box_size_y_slider = self.server.gui.add_slider(
                "Box Size Y",
                min=0.1,
                max=float(size_slider_max[1]),
                step=0.1,
                initial_value=float(self.boxes[self.selected_box_idx]["size"][1]),
            )
            self.box_size_z_slider = self.server.gui.add_slider(
                "Box Size Z",
                min=0.1,
                max=float(size_slider_max[2]),
                step=0.1,
                initial_value=float(self.boxes[self.selected_box_idx]["size"][2]),
            )
            reset_mask = self.server.gui.add_button("Reset Mask", color=(255, 80, 80))
            save_mask = self.server.gui.add_button("Save Mask", color=(80, 200, 80))

            @self.selected_box_number.on_update
            def _(_) -> None:
                self.selected_box_idx = int(np.clip(int(self.selected_box_number.value), 0, len(self.boxes) - 1))
                self.selected_box_number.value = self.selected_box_idx
                self._sync_size_sliders()
                self._refresh_all()

            @add_box.on_click
            def _(_) -> None:
                self._add_box()

            @delete_box.on_click
            def _(_) -> None:
                self._delete_selected_box()

            @self.box_size_x_slider.on_update
            def _(_) -> None:
                self._selected_box()["size"][0] = float(self.box_size_x_slider.value)
                self._refresh_after_box_change()

            @self.box_size_y_slider.on_update
            def _(_) -> None:
                self._selected_box()["size"][1] = float(self.box_size_y_slider.value)
                self._refresh_after_box_change()

            @self.box_size_z_slider.on_update
            def _(_) -> None:
                self._selected_box()["size"][2] = float(self.box_size_z_slider.value)
                self._refresh_after_box_change()

            @reset_mask.on_click
            def _(_) -> None:
                self.boxes = [{"center": self.default_box_center.copy(), "size": self.default_box_size.copy()}]
                self.selected_box_idx = 0
                if self.selected_box_number is not None:
                    self.selected_box_number.value = 0
                self._sync_size_sliders()
                self.excluded = self._compute_excluded_from_boxes()
                self._set_status("Reset mask in memory. Click Save Mask to write file.")
                self._refresh_all()

            @save_mask.on_click
            def _(_) -> None:
                self._save()

    def _save(self) -> None:
        self.excluded = self._compute_excluded_from_boxes()
        self.mask = {"version": 2, "excluded_timestamps": sorted(self.excluded), "zones": []}
        for idx, box in enumerate(self.boxes):
            center = box["center"]
            size = box["size"]
            box_min = center - size * 0.5
            box_max = center + size * 0.5
            inside = _points_in_box(self.positions, center, size)
            excluded_timestamps = {self.timestamps[i] for i, flag in enumerate(inside) if flag}
            self.mask["zones"].append(
                {
                    "name": f"box_{idx}",
                    "type": "box_xyz",
                    "center_xyz": [float(v) for v in center],
                    "size_xyz": [float(v) for v in size],
                    "min_xyz": [float(v) for v in box_min],
                    "max_xyz": [float(v) for v in box_max],
                    "excluded_timestamps": sorted(excluded_timestamps),
                }
            )
        _save_mask(self.mask_path, self.mask)
        self._set_status(f"Saved {len(self.excluded)} excluded keyframes to {self.mask_path}")

    def _refresh_all(self) -> None:
        self.excluded = self._compute_excluded_from_boxes()
        self._refresh_keyframe_points()
        self._refresh_box_handles()
        self._refresh_selection_preview()

    def _refresh_keyframe_points(self) -> None:
        if self.available_handle is not None:
            self.available_handle.remove()
        if self.excluded_handle is not None:
            self.excluded_handle.remove()

        available_indices = [i for i, t in enumerate(self.timestamps) if t not in self.excluded]
        excluded_indices = [i for i, t in enumerate(self.timestamps) if t in self.excluded]

        if available_indices:
            points = self.positions[available_indices]
            self.available_handle = self.server.scene.add_point_cloud(
                "/keyframes/available",
                points=points,
                colors=np.tile(np.array([[40, 130, 255]], dtype=np.uint8), (len(points), 1)),
                point_size=self.args.point_size,
                point_shape="rounded",
            )
        if excluded_indices:
            points = self.positions[excluded_indices]
            self.excluded_handle = self.server.scene.add_point_cloud(
                "/keyframes/excluded",
                points=points,
                colors=np.tile(np.array([[255, 60, 60]], dtype=np.uint8), (len(points), 1)),
                point_size=self.args.point_size * 1.8,
                point_shape="rounded",
            )

    def _refresh_selection_preview(self) -> None:
        if self.selected_handle is not None:
            self.selected_handle.remove()
            self.selected_handle = None

        box = self._selected_box()
        inside = _points_in_box(self.positions, box["center"], box["size"])
        self.current_selection = {self.timestamps[i] for i, flag in enumerate(inside) if flag}
        selected_indices = [i for i, t in enumerate(self.timestamps) if t in self.current_selection]
        if selected_indices:
            points = self.positions[selected_indices]
            self.selected_handle = self.server.scene.add_point_cloud(
                "/keyframes/current_selection",
                points=points,
                colors=np.tile(np.array([[255, 180, 0]], dtype=np.uint8), (len(points), 1)),
                point_size=self.args.point_size * 2.2,
                point_shape="rounded",
            )
        self._update_status_counts()

    def _refresh_box_handles(self) -> None:
        for handle in self.box_line_handles:
            handle.remove()
        for handle in self.box_gizmo_handles:
            handle.remove()
        self.box_line_handles.clear()
        self.box_gizmo_handles.clear()

        for idx, box in enumerate(self.boxes):
            segments = _box_line_segments(box["center"], box["size"])
            colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
            color = np.array([0.0, 1.0, 0.5], dtype=np.float32)
            if idx == self.selected_box_idx:
                color = np.array([1.0, 0.85, 0.0], dtype=np.float32)
            colors[:, :, :] = color
            line_handle = self.server.scene.add_line_segments(
                f"/mask_boxes/box_{idx}/line", points=segments, colors=colors, line_width=4.0
            )
            gizmo = self.server.scene.add_transform_controls(
                f"/mask_boxes/box_{idx}/center_gizmo",
                position=box["center"],
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            self.box_line_handles.append(line_handle)
            self.box_gizmo_handles.append(gizmo)

            @gizmo.on_update
            def _(event, box_idx=idx) -> None:
                self.boxes[box_idx]["center"] = np.asarray(event.target.position, dtype=np.float32)
                self.selected_box_idx = box_idx
                if self.selected_box_number is not None:
                    self.selected_box_number.value = box_idx
                self._sync_size_sliders()
                self._refresh_after_box_change()

    def _refresh_after_box_change(self) -> None:
        self.excluded = self._compute_excluded_from_boxes()
        self._refresh_keyframe_points()
        self._refresh_box_handles()
        self._refresh_selection_preview()

    def _selected_box(self) -> dict[str, np.ndarray]:
        self.selected_box_idx = int(np.clip(self.selected_box_idx, 0, len(self.boxes) - 1))
        return self.boxes[self.selected_box_idx]

    def _sync_size_sliders(self) -> None:
        if self.box_size_x_slider is None or self.box_size_y_slider is None or self.box_size_z_slider is None:
            return
        size = self._selected_box()["size"]
        self.box_size_x_slider.value = float(size[0])
        self.box_size_y_slider.value = float(size[1])
        self.box_size_z_slider.value = float(size[2])

    def _add_box(self) -> None:
        new_center = self._selected_box()["center"].copy() + np.array([0.5, 0.0, 0.0], dtype=np.float32)
        self.boxes.append({"center": new_center, "size": self._selected_box()["size"].copy()})
        self.selected_box_idx = len(self.boxes) - 1
        if self.selected_box_number is not None:
            self.selected_box_number.value = self.selected_box_idx
        self._sync_size_sliders()
        self._refresh_all()
        self._set_status(f"Added box {self.selected_box_idx}")

    def _delete_selected_box(self) -> None:
        if len(self.boxes) <= 1:
            self._set_status("Keep at least one box. Use Reset Mask to clear and restore default box.")
            return
        deleted_idx = self.selected_box_idx
        self.boxes.pop(deleted_idx)
        self.selected_box_idx = min(deleted_idx, len(self.boxes) - 1)
        if self.selected_box_number is not None:
            self.selected_box_number.value = self.selected_box_idx
        self._sync_size_sliders()
        self._refresh_all()
        self._set_status(f"Deleted box {deleted_idx}")

    def _compute_excluded_from_boxes(self) -> set[int]:
        excluded: set[int] = set()
        for box in self.boxes:
            inside = _points_in_box(self.positions, box["center"], box["size"])
            excluded |= {self.timestamps[i] for i, flag in enumerate(inside) if flag}
        return excluded

    def _load_boxes_from_mask(self) -> list[dict[str, np.ndarray]]:
        boxes = []
        for zone in self.mask.get("zones", []):
            if zone.get("type") != "box_xyz":
                continue
            center = np.asarray(zone.get("center_xyz", []), dtype=np.float32)
            size = np.asarray(zone.get("size_xyz", []), dtype=np.float32)
            if center.shape == (3,) and size.shape == (3,):
                boxes.append({"center": center, "size": np.maximum(size, 0.1).astype(np.float32)})
        return boxes

    def _update_status_counts(self) -> None:
        self._set_status(
            f"keyframes={len(self.timestamps)} | excluded={len(self.excluded)} | "
            f"selected={len(self.current_selection)} | boxes={len(self.boxes)} | "
            f"active_box={self.selected_box_idx} | mask={self.mask_path}"
        )

    def _set_status(self, value: str) -> None:
        print(value)
        if self.status is not None:
            self.status.value = value


def main(args: Args) -> None:
    RelocalizationMaskEditor(args).run()


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
