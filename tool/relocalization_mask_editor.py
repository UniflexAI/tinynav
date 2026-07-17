from __future__ import annotations

import json
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


def _line_segments_from_points(points: list[np.ndarray], close: bool) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    if close and len(points) >= 3:
        pts = np.vstack([pts, pts[:1]])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _points_in_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    if len(polygon_xy) < 3:
        return np.zeros(len(points_xy), dtype=bool)

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    poly_x = polygon_xy[:, 0]
    poly_y = polygon_xy[:, 1]
    inside = np.zeros(len(points_xy), dtype=bool)

    j = len(polygon_xy) - 1
    for i in range(len(polygon_xy)):
        intersects = ((poly_y[i] > y) != (poly_y[j] > y)) & (
            x < (poly_x[j] - poly_x[i]) * (y - poly_y[i]) / (poly_y[j] - poly_y[i] + 1e-12) + poly_x[i]
        )
        inside ^= intersects
        j = i
    return inside


class RelocalizationMaskEditor:
    def __init__(self, args: Args):
        self.args = args
        self.map_dir = args.tinynav_map_path
        self.mask_path = self.map_dir / args.output_name
        self.timestamps, self.positions = _load_poses(self.map_dir)
        self.xy = self.positions[:, :2]
        self.z_plane = float(np.median(self.positions[:, 2])) if len(self.positions) > 0 else 0.0

        self.mask = _load_mask(self.mask_path)
        self.excluded = {int(t) for t in self.mask.get("excluded_timestamps", [])}
        self.current_selection: set[int] = set()
        self.polygon_points: list[np.ndarray] = []

        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.world_axes.visible = True
        self.server.scene.set_up_direction("+z")

        self.available_handle: viser.SceneHandle | None = None
        self.excluded_handle: viser.SceneHandle | None = None
        self.selected_handle: viser.SceneHandle | None = None
        self.polygon_line_handle: viser.SceneHandle | None = None
        self.polygon_point_handles: list[viser.SceneHandle] = []
        self.polygon_gizmo_handles: list[viser.TransformControlsHandle] = []
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

        rng = np.random.default_rng(0)
        free_indices = np.argwhere(xy_plane == 1)
        occupied_indices = np.argwhere(xy_plane == 2)

        if len(free_indices) > self.args.max_occupancy_points:
            free_indices = free_indices[rng.choice(len(free_indices), self.args.max_occupancy_points, replace=False)]
        if len(occupied_indices) > self.args.max_occupancy_points:
            occupied_indices = occupied_indices[
                rng.choice(len(occupied_indices), self.args.max_occupancy_points, replace=False)
            ]

        if len(free_indices) > 0:
            self.server.scene.add_point_cloud(
                "/map/free",
                points=xy_world(free_indices),
                colors=np.tile(np.array([[120, 120, 120]], dtype=np.uint8), (len(free_indices), 1)),
                point_size=max(0.01, resolution * 0.35),
                point_shape="square",
            )
        if len(occupied_indices) > 0:
            self.server.scene.add_point_cloud(
                "/map/occupied",
                points=xy_world(occupied_indices),
                colors=np.tile(np.array([[40, 40, 40]], dtype=np.uint8), (len(occupied_indices), 1)),
                point_size=max(0.01, resolution * 0.45),
                point_shape="square",
            )

    def _add_keyframe_layers(self) -> None:
        if len(self.positions) >= 2:
            segments = np.stack([self.positions[:-1], self.positions[1:]], axis=1)
            colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
            colors[:, :, :] = np.array([0.3, 0.3, 0.3], dtype=np.float32)
            self.server.scene.add_line_segments("/keyframes/trajectory", points=segments, colors=colors, line_width=1.5)

    def _add_ui(self) -> None:
        with self.server.gui.add_folder("Relocalization Mask Editor"):
            self.status = self.server.gui.add_text("Status", initial_value="Ready")
            add_vertex = self.server.gui.add_button("Add Polygon Vertex")
            apply_polygon = self.server.gui.add_button("Apply Polygon To Mask", color=(255, 170, 60))
            clear_polygon = self.server.gui.add_button("Clear Polygon")
            reset_mask = self.server.gui.add_button("Reset Mask", color=(255, 80, 80))
            save_mask = self.server.gui.add_button("Save Mask", color=(80, 200, 80))

            @add_vertex.on_click
            def _(_) -> None:
                self._add_polygon_vertex()

            @apply_polygon.on_click
            def _(_) -> None:
                self._apply_polygon_to_mask()

            @clear_polygon.on_click
            def _(_) -> None:
                self._clear_polygon()

            @reset_mask.on_click
            def _(_) -> None:
                self.excluded = set()
                self.current_selection = set()
                self.mask = {"version": 1, "excluded_timestamps": [], "zones": []}
                self._set_status("Reset mask in memory. Click Save Mask to write file.")
                self._refresh_all()

            @save_mask.on_click
            def _(_) -> None:
                self._save()

    def _default_new_vertex(self) -> np.ndarray:
        if self.polygon_points:
            return self.polygon_points[-1] + np.array([0.5, 0.0, 0.0], dtype=np.float32)
        center = np.mean(self.positions, axis=0).astype(np.float32)
        center[2] = self.z_plane + 0.05
        return center

    def _add_polygon_vertex(self) -> None:
        self.polygon_points.append(self._default_new_vertex())
        self._refresh_polygon_handles()
        self._refresh_selection_preview()
        self._set_status(f"Polygon vertices: {len(self.polygon_points)}")

    def _clear_polygon(self) -> None:
        self.polygon_points.clear()
        self.current_selection = set()
        self._refresh_polygon_handles()
        self._refresh_all()
        self._set_status("Cleared polygon")

    def _apply_polygon_to_mask(self) -> None:
        if len(self.polygon_points) < 3:
            self._set_status("Need at least 3 polygon vertices")
            return

        self._refresh_selection_preview()
        self.excluded |= self.current_selection
        polygon_xy = np.asarray(self.polygon_points, dtype=np.float32)[:, :2]
        self.mask.setdefault("zones", []).append(
            {
                "name": f"zone_{len(self.mask.get('zones', []))}",
                "polygon_xy": [[float(x), float(y)] for x, y in polygon_xy],
                "excluded_timestamps": sorted(self.current_selection),
            }
        )
        self._set_status(f"Applied polygon: excluded {len(self.current_selection)} keyframes")
        self._refresh_all()

    def _save(self) -> None:
        self.mask["version"] = 1
        self.mask["excluded_timestamps"] = sorted(self.excluded)
        _save_mask(self.mask_path, self.mask)
        self._set_status(f"Saved {len(self.excluded)} excluded keyframes to {self.mask_path}")

    def _refresh_all(self) -> None:
        self._refresh_keyframe_points()
        self._refresh_polygon_handles()
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

        if len(self.polygon_points) < 3:
            self.current_selection = set()
            self._update_status_counts()
            return

        polygon_xy = np.asarray(self.polygon_points, dtype=np.float32)[:, :2]
        inside = _points_in_polygon(self.xy, polygon_xy)
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

    def _refresh_polygon_handles(self) -> None:
        if self.polygon_line_handle is not None:
            self.polygon_line_handle.remove()
            self.polygon_line_handle = None
        for handle in self.polygon_point_handles:
            handle.remove()
        for handle in self.polygon_gizmo_handles:
            handle.remove()
        self.polygon_point_handles.clear()
        self.polygon_gizmo_handles.clear()

        segments = _line_segments_from_points(self.polygon_points, close=True)
        if len(segments) > 0:
            colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
            colors[:, :, :] = np.array([1.0, 0.6, 0.0], dtype=np.float32)
            self.polygon_line_handle = self.server.scene.add_line_segments(
                "/mask_polygon/line", points=segments, colors=colors, line_width=4.0
            )

        for idx, point in enumerate(self.polygon_points):
            point_handle = self.server.scene.add_icosphere(
                f"/mask_polygon/point_{idx}",
                radius=self.args.point_size * 1.8,
                color=(255, 170, 0),
                position=point,
            )
            gizmo = self.server.scene.add_transform_controls(
                f"/mask_polygon/point_{idx}_gizmo",
                position=point,
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            self.polygon_point_handles.append(point_handle)
            self.polygon_gizmo_handles.append(gizmo)

            @gizmo.on_update
            def _(event, point_idx=idx, handle=point_handle) -> None:
                new_pos = np.asarray(event.target.position, dtype=np.float32)
                new_pos[2] = self.z_plane + 0.05
                self.polygon_points[point_idx] = new_pos
                handle.position = new_pos
                self._refresh_polygon_line()
                self._refresh_selection_preview()

    def _refresh_polygon_line(self) -> None:
        if self.polygon_line_handle is not None:
            self.polygon_line_handle.remove()
            self.polygon_line_handle = None
        segments = _line_segments_from_points(self.polygon_points, close=True)
        if len(segments) == 0:
            return
        colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
        colors[:, :, :] = np.array([1.0, 0.6, 0.0], dtype=np.float32)
        self.polygon_line_handle = self.server.scene.add_line_segments(
            "/mask_polygon/line", points=segments, colors=colors, line_width=4.0
        )

    def _update_status_counts(self) -> None:
        self._set_status(
            f"keyframes={len(self.timestamps)} | excluded={len(self.excluded)} | "
            f"selected={len(self.current_selection)} | mask={self.mask_path}"
        )

    def _set_status(self, value: str) -> None:
        print(value)
        if self.status is not None:
            self.status.value = value


def main(args: Args) -> None:
    RelocalizationMaskEditor(args).run()


if __name__ == "__main__":
    main(tyro.cli(Args))
