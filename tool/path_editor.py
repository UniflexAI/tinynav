from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tyro
import viser
from plyfile import PlyData
from scipy.ndimage import distance_transform_edt


PathPoint = npt.NDArray[np.floating]


@dataclass(frozen=True)
class Args:
    tinynav_map_path: Path
    """Tinynav map directory containing occupancy_grid.npy, occupancy_meta.npy and sdf_map.npy."""

    paths_json_name: str = "paths.json"
    """Editable path control-point file saved under tinynav_map_path."""

    sdf_map_name: str = "sdf_map.npy"
    """Active SDF path-map file used by Tinynav; Replace Current SDF Map overwrites this file."""

    default_sdf_map_name: str = "sdf_map.default.npy"
    """Backup of the original/default SDF path-map, used by Restore Default SDF Map."""

    path_radius_m: float = 0.2
    """SDF distance threshold used by map_node; voxels near edited paths become low-SDF corridor."""

    max_preview_points: int = 300_000
    """Maximum number of low-SDF voxels rendered in the Viser preview."""

    host: str = "0.0.0.0"
    port: int = 8080


def _load_json(path: Path) -> dict[int, dict[str, Any]]:
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


def _save_json(path: Path, paths: dict[int, dict[str, Any]]) -> None:
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


def _random_color() -> tuple[int, int, int]:
    return tuple(int(x) for x in np.random.randint(40, 255, size=3))


def _grid_to_world(indices: np.ndarray, origin: np.ndarray, resolution: float) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return origin[None, :] + indices.astype(np.float32) * float(resolution)


def _world_to_grid(points: np.ndarray, origin: np.ndarray, resolution: float, shape: tuple[int, int, int]) -> np.ndarray:
    indices = np.rint((points - origin[None, :]) / float(resolution)).astype(np.int32)
    max_index = np.asarray(shape, dtype=np.int32) - 1
    return np.clip(indices, 0, max_index[None, :])


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


def _load_pointcloud_ply(ply_file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    names = v.data.dtype.names or ()
    if "red" in names:
        colors = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.float32) / 255.0
    elif "r" in names:
        colors = np.stack([v["r"], v["g"], v["b"]], axis=-1).astype(np.float32) / 255.0
    else:
        colors = np.ones((len(v), 3), dtype=np.float32)
    return positions, colors


class PathEditor:
    def __init__(self, args: Args):
        self.args = args
        self.map_dir = args.tinynav_map_path
        self.paths_json_path = self.map_dir / args.paths_json_name
        self.sdf_map_path = self.map_dir / args.sdf_map_name
        self.default_sdf_map_path = self.map_dir / args.default_sdf_map_name

        self.occupancy_grid = np.load(self.map_dir / "occupancy_grid.npy")
        self.occupancy_meta = np.load(self.map_dir / "occupancy_meta.npy")
        self.origin = self.occupancy_meta[:3].astype(np.float32)
        self.resolution = float(self.occupancy_meta[3])
        self.shape = tuple(int(v) for v in self.occupancy_grid.shape)
        self.current_sdf_map = self._load_sdf_file(self.sdf_map_path)
        self.default_sdf_map = self._load_sdf_file(self.default_sdf_map_path)
        if self.default_sdf_map is None:
            # Before the first replacement there is no separate backup, so the active file is the default map.
            self.default_sdf_map = self.current_sdf_map

        self.paths = _load_json(self.paths_json_path)
        self.path_id_counter = (max(self.paths.keys()) + 1) if self.paths else 0
        self.selected_path_id: int | None = next(iter(self.paths.keys()), None)

        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.world_axes.visible = True
        self.server.scene.set_up_direction("+z")

        self.path_line_handles: dict[int, viser.SceneHandle] = {}
        self.point_handles: dict[tuple[int, int], viser.SceneHandle] = {}
        self.gizmo_handles: dict[tuple[int, int], viser.SceneHandle] = {}
        self.sdf_preview_handle: viser.SceneHandle | None = None
        self.status = None

    def _load_sdf_file(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        sdf_map = np.load(path).astype(np.float32)
        if sdf_map.shape != self.occupancy_grid.shape:
            raise ValueError(
                f"{path} shape {sdf_map.shape} does not match occupancy_grid shape {self.occupancy_grid.shape}"
            )
        return sdf_map

    def _edited_sdf_map(self) -> np.ndarray:
        return build_sdf_from_paths(self.paths, self.origin, self.resolution, self.shape)

    def _ensure_default_backup(self) -> None:
        if self.default_sdf_map_path.exists():
            return
        if self.current_sdf_map is None:
            return
        np.save(self.default_sdf_map_path, self.current_sdf_map)
        self.default_sdf_map = self.current_sdf_map.copy()
        print(f"Saved default SDF backup to {self.default_sdf_map_path}")

    def run(self) -> None:
        self._add_static_map_layers()
        self._add_optional_pointcloud()
        self._add_path_editor_ui()
        self._refresh_all_paths()
        self._refresh_sdf_preview()
        print(f"Path editor is running at http://{self.args.host}:{self.args.port}")
        while True:
            time.sleep(1.0)

    def _add_static_map_layers(self) -> None:
        x_y_plane = np.max(self.occupancy_grid, axis=2)
        z_plane = float(self.origin[2])

        def xy_world(xy_indices: np.ndarray) -> np.ndarray:
            points = np.zeros((len(xy_indices), 3), dtype=np.float32)
            points[:, 0] = float(self.origin[0]) + xy_indices[:, 0] * self.resolution
            points[:, 1] = float(self.origin[1]) + xy_indices[:, 1] * self.resolution
            points[:, 2] = z_plane
            return points

        def xyz_world(xyz_indices: np.ndarray) -> np.ndarray:
            points = np.zeros((len(xyz_indices), 3), dtype=np.float32)
            points[:, 0] = float(self.origin[0]) + xyz_indices[:, 0] * self.resolution
            points[:, 1] = float(self.origin[1]) + xyz_indices[:, 1] * self.resolution
            points[:, 2] = float(self.origin[2]) + xyz_indices[:, 2] * self.resolution
            return points

        free_indices = np.argwhere(x_y_plane == 1)
        occupied_indices = np.argwhere(x_y_plane == 2)
        free_handle = None
        occupied_handle = None
        if len(free_indices) > 0:
            free_handle = self.server.scene.add_point_cloud(
                "/occupancy_2d/free",
                points=xy_world(free_indices),
                colors=np.tile(np.array([[0.2, 0.4, 1.0]], dtype=np.float32), (len(free_indices), 1)),
                point_size=self.resolution * 0.8,
                point_shape="rounded",
            )
        if len(occupied_indices) > 0:
            occupied_base = xy_world(occupied_indices)
            z_levels = np.arange(z_plane + self.resolution * 0.5, z_plane + 0.8, self.resolution, dtype=np.float32)
            occupied_points = np.repeat(occupied_base, len(z_levels), axis=0)
            occupied_points[:, 2] = np.tile(z_levels, len(occupied_base))
            # Light red, matching poi_editor's occupied-column feel without requiring OpenCV.
            wall_rgb = np.array([1.0, 0.45, 0.45], dtype=np.float32)
            occupied_handle = self.server.scene.add_point_cloud(
                "/occupancy_2d/occupied",
                points=occupied_points,
                colors=np.tile(wall_rgb[None, :], (len(occupied_points), 1)),
                point_size=self.resolution * 0.8,
                point_shape="rounded",
            )

        # Full 3D occupancy voxels (true per-voxel height, not the flattened Z-max projection).
        # Off by default: useful when placing waypoints at a specific height, but can be a lot of points.
        max_3d_points = 300_000

        def capped_indices(mask: np.ndarray) -> np.ndarray:
            indices_all = np.argwhere(mask)
            if len(indices_all) > max_3d_points:
                stride = int(np.ceil(len(indices_all) / max_3d_points))
                return indices_all[::stride]
            return indices_all

        free_3d_points = xyz_world(capped_indices(self.occupancy_grid == 1))
        occupied_3d_points = xyz_world(capped_indices(self.occupancy_grid == 2))
        free_3d_handle = None
        occupied_3d_handle = None
        if len(free_3d_points) > 0:
            free_3d_handle = self.server.scene.add_point_cloud(
                "/occupancy_3d/free",
                points=free_3d_points,
                colors=np.tile(np.array([[0.2, 0.4, 1.0]], dtype=np.float32), (len(free_3d_points), 1)),
                point_size=self.resolution * 0.8,
                point_shape="rounded",
            )
            free_3d_handle.visible = False
        if len(occupied_3d_points) > 0:
            occupied_3d_handle = self.server.scene.add_point_cloud(
                "/occupancy_3d/occupied",
                points=occupied_3d_points,
                colors=np.tile(np.array([[0.6, 0.6, 0.6]], dtype=np.float32), (len(occupied_3d_points), 1)),
                point_size=self.resolution * 0.8,
                point_shape="rounded",
            )
            occupied_3d_handle.visible = False

        with self.server.gui.add_folder("Occupancy 2D Map"):
            show_free = self.server.gui.add_checkbox("Show Free", initial_value=True)
            show_occupied = self.server.gui.add_checkbox("Show Occupied", initial_value=True)
            show_free_3d = self.server.gui.add_checkbox("Show 3D Free (true height)", initial_value=False)
            show_occupied_3d = self.server.gui.add_checkbox("Show 3D Occupied (true height)", initial_value=False)
            point_size = self.server.gui.add_slider(
                "Point Size", min=0.001, max=max(0.1, self.resolution), step=0.001, initial_value=self.resolution * 0.8
            )

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

            @point_size.on_update
            def _(_) -> None:
                if free_handle is not None:
                    free_handle.point_size = point_size.value
                if occupied_handle is not None:
                    occupied_handle.point_size = point_size.value
                if free_3d_handle is not None:
                    free_3d_handle.point_size = point_size.value
                if occupied_3d_handle is not None:
                    occupied_3d_handle.point_size = point_size.value

    def _add_optional_pointcloud(self) -> None:
        pointcloud_path = self.map_dir / "pointcloud.ply"
        if not pointcloud_path.exists():
            return
        try:
            points, colors = _load_pointcloud_ply(pointcloud_path)
        except Exception as exc:
            print(f"Warning: failed to load {pointcloud_path}: {exc}")
            return
        pc_handle = self.server.scene.add_point_cloud(
            "/0/point_cloud", points=points, colors=colors, point_size=0.01, point_shape="rounded"
        )
        with self.server.gui.add_folder("Point Cloud Settings"):
            show_pc = self.server.gui.add_checkbox("Show Point Cloud", initial_value=True)
            point_size = self.server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01)

            @show_pc.on_update
            def _(_) -> None:
                pc_handle.visible = show_pc.value

            @point_size.on_update
            def _(_) -> None:
                pc_handle.point_size = point_size.value

    def _add_path_editor_ui(self) -> None:
        with self.server.gui.add_folder("SDF Path Editor"):
            self.status = self.server.gui.add_text("Status", initial_value="Ready")
            selected = self.server.gui.add_number("Selected Path ID", initial_value=self.selected_path_id or 0, step=1)
            add_path = self.server.gui.add_button("Add Path")
            add_waypoint = self.server.gui.add_button("Add Waypoint To Selected")
            delete_last_waypoint = self.server.gui.add_button("Delete Last Waypoint")
            delete_path = self.server.gui.add_button("Delete Selected Path", color=(255, 80, 80))
            replace_sdf = self.server.gui.add_button("Replace Current SDF Map", color=(80, 200, 80))
            restore_default_sdf = self.server.gui.add_button("Restore Default SDF Map", color=(80, 120, 255))

            @selected.on_update
            def _(_) -> None:
                path_id = int(selected.value)
                self.selected_path_id = path_id if path_id in self.paths else None
                self._set_status(f"Selected path: {self.selected_path_id}")

            @add_path.on_click
            def _(_) -> None:
                path_id = self.path_id_counter
                self.path_id_counter += 1
                selected.value = path_id
                self.selected_path_id = path_id
                self.paths[path_id] = {
                    "id": path_id,
                    "name": f"Path_{path_id}",
                    "points": [self._default_new_point()],
                    "color": _random_color(),
                }
                self._refresh_path(path_id)
                self._set_status(f"Added Path_{path_id}")

            @add_waypoint.on_click
            def _(_) -> None:
                if self.selected_path_id is None:
                    self._set_status("No selected path")
                    return
                points = self.paths[self.selected_path_id]["points"]
                if points:
                    new_point = np.asarray(points[-1], dtype=np.float32) + np.array([self.resolution * 5, 0.0, 0.0], dtype=np.float32)
                else:
                    new_point = self._default_new_point()
                points.append(new_point)
                self._refresh_path(self.selected_path_id)
                self._set_status(f"Added waypoint to Path_{self.selected_path_id}")

            @delete_last_waypoint.on_click
            def _(_) -> None:
                if self.selected_path_id is None:
                    self._set_status("No selected path")
                    return
                points = self.paths[self.selected_path_id]["points"]
                if points:
                    points.pop()
                self._refresh_path(self.selected_path_id)
                self._set_status(f"Deleted last waypoint from Path_{self.selected_path_id}")

            @delete_path.on_click
            def _(_) -> None:
                if self.selected_path_id is None:
                    self._set_status("No selected path")
                    return
                path_id = self.selected_path_id
                self._remove_path_handles(path_id)
                del self.paths[path_id]
                self.selected_path_id = next(iter(self.paths.keys()), None)
                selected.value = self.selected_path_id or 0
                self._set_status(f"Deleted Path_{path_id}")

            @replace_sdf.on_click
            def _(_) -> None:
                # Save a permanent default backup before the first replacement, then write a fully new path map.
                _save_json(self.paths_json_path, self.paths)
                self._ensure_default_backup()
                if self.sdf_map_path.exists():
                    backup_path = self.sdf_map_path.with_suffix(f".bak-{time.strftime('%Y%m%d-%H%M%S')}.npy")
                    shutil.copy2(self.sdf_map_path, backup_path)
                    print(f"Backed up active SDF map to {backup_path}")
                self.current_sdf_map = self._edited_sdf_map()
                np.save(self.sdf_map_path, self.current_sdf_map)
                self._refresh_sdf_preview()
                self._set_status(f"Replaced active SDF map with edited path map; saved {self.paths_json_path.name}")

            @restore_default_sdf.on_click
            def _(_) -> None:
                if self.default_sdf_map is None:
                    self._set_status("No default SDF map available to restore")
                    return
                if self.sdf_map_path.exists():
                    backup_path = self.sdf_map_path.with_suffix(f".bak-{time.strftime('%Y%m%d-%H%M%S')}.npy")
                    shutil.copy2(self.sdf_map_path, backup_path)
                    print(f"Backed up active SDF map to {backup_path}")
                np.save(self.sdf_map_path, self.default_sdf_map)
                self.current_sdf_map = self.default_sdf_map.copy()
                self._refresh_sdf_preview()
                self._set_status("Restored default SDF map to active sdf_map.npy")

    def _default_new_point(self) -> np.ndarray:
        traversable = np.argwhere(self.occupancy_grid != 2)
        if len(traversable) == 0:
            return self.origin.copy()
        center_idx = traversable[len(traversable) // 2]
        return _grid_to_world(center_idx[None, :], self.origin, self.resolution)[0]

    def _refresh_all_paths(self) -> None:
        for path_id in list(self.paths.keys()):
            self._refresh_path(path_id)

    def _refresh_path(self, path_id: int) -> None:
        self._remove_path_handles(path_id)
        item = self.paths[path_id]
        color = tuple(int(c) for c in item["color"])
        points = item["points"]
        segments = _line_segments_from_points(points)
        if len(segments) > 0:
            colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
            colors[:, :, :] = np.asarray(color, dtype=np.float32) / 255.0
            self.path_line_handles[path_id] = self.server.scene.add_line_segments(
                f"/paths/{item['name']}/line", points=segments, colors=colors, line_width=4.0
            )
        for point_idx, point in enumerate(points):
            key = (path_id, point_idx)
            point_name = f"/paths/{item['name']}/point_{point_idx}"
            self.point_handles[key] = self.server.scene.add_icosphere(
                point_name, radius=max(0.05, self.resolution), color=color, position=np.asarray(point, dtype=np.float32)
            )
            gizmo = self.server.scene.add_transform_controls(
                f"{point_name}_gizmo", position=np.asarray(point, dtype=np.float32), wxyz=(1.0, 0.0, 0.0, 0.0)
            )
            self.gizmo_handles[key] = gizmo

            @gizmo.on_update
            def _(event, pid=path_id, pidx=point_idx, handle=self.point_handles[key]) -> None:
                new_pos = np.asarray(event.target.position, dtype=np.float32)
                self.paths[pid]["points"][pidx] = new_pos
                handle.position = new_pos
                self._refresh_path_line(pid)

    def _refresh_path_line(self, path_id: int) -> None:
        if path_id in self.path_line_handles:
            self.path_line_handles[path_id].remove()
            del self.path_line_handles[path_id]
        item = self.paths[path_id]
        segments = _line_segments_from_points(item["points"])
        if len(segments) == 0:
            return
        color = np.asarray(item["color"], dtype=np.float32) / 255.0
        colors = np.zeros((len(segments), 2, 3), dtype=np.float32)
        colors[:, :, :] = color
        self.path_line_handles[path_id] = self.server.scene.add_line_segments(
            f"/paths/{item['name']}/line", points=segments, colors=colors, line_width=4.0
        )

    def _remove_path_handles(self, path_id: int) -> None:
        if path_id in self.path_line_handles:
            self.path_line_handles[path_id].remove()
            del self.path_line_handles[path_id]
        for key in list(self.point_handles.keys()):
            if key[0] == path_id:
                self.point_handles[key].remove()
                del self.point_handles[key]
        for key in list(self.gizmo_handles.keys()):
            if key[0] == path_id:
                self.gizmo_handles[key].remove()
                del self.gizmo_handles[key]

    def _refresh_sdf_preview(self) -> None:
        if self.sdf_preview_handle is not None:
            self.sdf_preview_handle.remove()
            self.sdf_preview_handle = None
        preview_sdf = self.current_sdf_map
        if preview_sdf is None:
            print("No active SDF map available for preview")
            return
        traversable_mask = self.occupancy_grid != 2
        mask = np.logical_and.reduce((traversable_mask, np.isfinite(preview_sdf), preview_sdf < self.args.path_radius_m))
        indices_all = np.argwhere(mask)
        if len(indices_all) == 0:
            return
        if len(indices_all) > self.args.max_preview_points:
            stride = int(np.ceil(len(indices_all) / self.args.max_preview_points))
            indices = indices_all[::stride]
        else:
            indices = indices_all
        points = _grid_to_world(indices, self.origin, self.resolution)
        colors = np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (len(points), 1))
        self.sdf_preview_handle = self.server.scene.add_point_cloud(
            "/sdf_path_preview/low_sdf_voxels",
            points=points,
            colors=colors,
            point_size=self.resolution * 0.8,
            point_shape="rounded",
        )
        print(
            f"Previewing {len(points)} low-SDF voxels from active SDF map "
            f"(threshold={self.args.path_radius_m:.3f}m, total={len(indices_all)})"
        )

    def _set_status(self, value: str) -> None:
        print(value)
        if self.status is not None:
            self.status.value = value


def main(args: Args) -> None:
    PathEditor(args).run()


if __name__ == "__main__":
    main(tyro.cli(Args))
