"""Viser editor for 3D relocalization candidate masks (oriented boxes)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
import viser

sys.path.append("/tinynav/tinynav/core")

from relocalization_mask import (
    DEFAULT_RELOCALIZATION_MASK_FILENAME,
    MaskRegion,
    allowed_keyframe_timestamps,
    load_nav_flow_dict,
    load_relocalization_mask,
    resolve_relocalization_mask_path,
    save_relocalization_mask,
)


@dataclass
class Args:
    tinynav_map_path: Path
    """Map directory containing poses.npy."""

    output_name: str = DEFAULT_RELOCALIZATION_MASK_FILENAME
    """Mask JSON filename written under tinynav_map_path."""

    host: str = "0.0.0.0"
    port: int = 8081


@dataclass
class RegionUi:
    region: MaskRegion
    folder: object
    gizmo: object
    box: object
    scale_handles: list = field(default_factory=list)
    center_gui: object | None = None
    half_size_gui: object | None = None
    delete_button: object | None = None
    updating: bool = False


def _gizmo_scale(half_size: np.ndarray) -> float:
    return float(max(0.45, np.max(half_size) * 0.7))


def _default_map_center(map_poses: dict) -> np.ndarray:
    if not map_poses:
        return np.zeros(3, dtype=np.float64)
    positions = np.array([np.asarray(pose, dtype=np.float64)[:3, 3] for pose in map_poses.values()])
    return positions.mean(axis=0)


def _add_occupancy_background(server: viser.ViserServer, map_dir: Path, *, max_points: int = 300_000) -> bool:
    occupancy_grid_path = map_dir / "occupancy_grid.npy"
    occupancy_meta_path = map_dir / "occupancy_meta.npy"
    if not occupancy_grid_path.exists() or not occupancy_meta_path.exists():
        return False

    occupancy_grid = np.load(occupancy_grid_path)
    occupancy_meta = np.load(occupancy_meta_path)
    origin = occupancy_meta[:3].astype(np.float64)
    resolution = float(occupancy_meta[3])

    occupied_indices = np.argwhere(occupancy_grid == 2)
    if len(occupied_indices) == 0:
        occupied_indices = np.argwhere(occupancy_grid != 0)
    if len(occupied_indices) == 0:
        print(f"Occupancy grid in {map_dir} has no occupied voxels to display")
        return False

    if len(occupied_indices) > max_points:
        stride = int(np.ceil(len(occupied_indices) / max_points))
        occupied_indices = occupied_indices[::stride]

    points = np.zeros((len(occupied_indices), 3), dtype=np.float32)
    points[:, 0] = origin[0] + occupied_indices[:, 0] * resolution
    points[:, 1] = origin[1] + occupied_indices[:, 1] * resolution
    points[:, 2] = origin[2] + occupied_indices[:, 2] * resolution
    colors = np.tile(np.array([[0.75, 0.78, 0.82]], dtype=np.float32), (len(points), 1))
    server.scene.add_point_cloud(
        "/background/occupancy",
        points=points,
        colors=colors,
        point_size=max(resolution * 0.85, 0.05),
        point_shape="rounded",
    )
    print(
        f"Loaded occupancy background from {map_dir} "
        f"({len(points)} voxels, shape={tuple(int(v) for v in occupancy_grid.shape)}, "
        f"resolution={resolution:.3f}m)"
    )
    return True


def _frame_camera_to_map(server: viser.ViserServer, map_poses: dict) -> None:
    center = _default_map_center(map_poses).astype(np.float64)
    positions = np.array([np.asarray(pose, dtype=np.float64)[:3, 3] for pose in map_poses.values()])
    extent = float(np.max(np.linalg.norm(positions - center, axis=1))) if len(positions) else 5.0
    extent = max(extent, 2.0)
    eye = center + np.array([extent * 1.2, extent * 1.2, extent * 0.8], dtype=np.float64)

    @server.on_client_connect
    def _(client) -> None:
        client.camera.look_at = tuple(float(v) for v in center)
        client.camera.position = tuple(float(v) for v in eye)


def _resolve_output_path(map_dir: Path, output_name: str) -> Path:
    nav_flow = load_nav_flow_dict(str(map_dir))
    try:
        configured = resolve_relocalization_mask_path(str(map_dir), nav_flow)
    except ValueError:
        configured = None
    if configured:
        return Path(configured)
    return map_dir / output_name


def main(args: Args) -> None:
    map_dir = args.tinynav_map_path
    poses = np.load(map_dir / "poses.npy", allow_pickle=True).item()
    output_path = _resolve_output_path(map_dir, args.output_name)

    regions: list[MaskRegion] = []
    if output_path.exists():
        regions = load_relocalization_mask(str(output_path))

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")

    splat_path = map_dir / "splat.ply"
    pointcloud_path = map_dir / "pointcloud.ply"
    if splat_path.exists():
        from tool.map_editor import load_ply_file, load_splat_file

        splat_data = load_ply_file(splat_path, center=False) if splat_path.suffix == ".ply" else load_splat_file(splat_path, center=True)
        server.scene.add_gaussian_splats(
            "/background/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )
        print(f"Loaded Gaussian splat from {splat_path}")
    elif pointcloud_path.exists():
        from tool.map_editor import load_pointcloud_ply

        pc_data = load_pointcloud_ply(pointcloud_path, center=False)
        server.scene.add_point_cloud(
            "/background/point_cloud",
            points=pc_data["positions"],
            colors=pc_data["colors"],
            point_size=0.01,
            point_shape="rounded",
        )
        print(f"Loaded point cloud from {pointcloud_path}")
    elif _add_occupancy_background(server, map_dir):
        pass
    else:
        print(
            f"Warning: no splat.ply, pointcloud.ply, or occupancy_grid.npy found under {map_dir}. "
            "Only keyposes and mask boxes will be shown."
        )

    _frame_camera_to_map(server, poses)

    region_uis: list[RegionUi] = []
    allowed_handle = None
    status_gui = server.gui.add_text("Status", initial_value="Ready")
    help_gui = server.gui.add_markdown(
        "**Gizmo:** arrows = move, rings = rotate. "
        "RGB handles on the box faces = scale X/Y/Z. "
        "Or use Center / Half size in the region folder."
    )

    def refresh_allowed_keyposes() -> None:
        nonlocal allowed_handle
        if allowed_handle is not None:
            allowed_handle.remove()
            allowed_handle = None
        if not regions:
            status_gui.value = "No regions — add a box to define allowed keyposes"
            return
        allowed = allowed_keyframe_timestamps(poses, regions)
        if not allowed:
            status_gui.value = f"0/{len(poses)} keyposes inside {len(regions)} region(s)"
            return
        points = np.array(
            [np.asarray(poses[timestamp], dtype=np.float64)[:3, 3] for timestamp in allowed],
            dtype=np.float32,
        )
        colors = np.tile(np.array([[0.2, 0.95, 0.35]], dtype=np.float32), (len(points), 1))
        allowed_handle = server.scene.add_point_cloud(
            "/relocalization/allowed_keyposes",
            points=points,
            colors=colors,
            point_size=0.06,
            point_shape="rounded",
        )
        status_gui.value = f"{len(allowed)}/{len(poses)} keyposes allowed ({len(regions)} region(s))"

    def apply_region(ui: RegionUi, region: MaskRegion, *, from_gui: bool = False, from_gizmo: bool = False) -> None:
        ui.region = region
        idx = region_uis.index(ui)
        regions[idx] = region
        ui.updating = True
        try:
            if not from_gizmo:
                ui.gizmo.position = tuple(float(x) for x in region.center)
                ui.gizmo.wxyz = tuple(float(x) for x in region.wxyz)
            ui.gizmo.scale = _gizmo_scale(region.half_size)
            ui.box.dimensions = tuple(float(x) for x in region.dimensions)
            for axis, handle in enumerate(ui.scale_handles):
                pos = [0.0, 0.0, 0.0]
                pos[axis] = float(region.half_size[axis])
                handle.position = tuple(pos)
            if not from_gui:
                ui.center_gui.value = tuple(float(x) for x in region.center)
                ui.half_size_gui.value = tuple(float(x) for x in region.half_size)
        finally:
            ui.updating = False
        refresh_allowed_keyposes()

    def remove_region_ui(ui: RegionUi) -> None:
        idx = region_uis.index(ui)
        for handle in ui.scale_handles:
            handle.remove()
        ui.box.remove()
        ui.gizmo.remove()
        try:
            ui.folder.remove()
        except Exception:
            if ui.center_gui is not None:
                ui.center_gui.remove()
            if ui.half_size_gui is not None:
                ui.half_size_gui.remove()
            if ui.delete_button is not None:
                ui.delete_button.remove()
        region_uis.pop(idx)
        regions.pop(idx)
        refresh_allowed_keyposes()

    def add_region_ui(region: MaskRegion) -> None:
        name = region.name
        gizmo = server.scene.add_transform_controls(
            f"/relocalization/regions/{name}",
            position=tuple(float(x) for x in region.center),
            wxyz=tuple(float(x) for x in region.wxyz),
            scale=_gizmo_scale(region.half_size),
            depth_test=False,
        )
        box = server.scene.add_box(
            f"/relocalization/regions/{name}/box",
            color=(40, 220, 90),
            dimensions=tuple(float(x) for x in region.dimensions),
            wireframe=True,
            cast_shadow=False,
            receive_shadow=False,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        scale_handles = []
        axis_active = ((True, False, False), (False, True, False), (False, False, True))
        for axis, active in enumerate(axis_active):
            pos = [0.0, 0.0, 0.0]
            pos[axis] = float(region.half_size[axis])
            handle = server.scene.add_transform_controls(
                f"/relocalization/regions/{name}/scale_{axis}",
                position=tuple(pos),
                scale=0.32,
                disable_rotations=True,
                disable_sliders=True,
                active_axes=active,
                depth_test=False,
            )
            scale_handles.append(handle)

        folder = server.gui.add_folder(name)
        with folder:
            center_gui = server.gui.add_vector3(
                "Center",
                initial_value=tuple(float(x) for x in region.center),
                step=0.05,
            )
            half_size_gui = server.gui.add_vector3(
                "Half size",
                initial_value=tuple(float(x) for x in region.half_size),
                step=0.05,
            )
            delete_button = server.gui.add_button("Delete region", color=(255, 80, 80))

        ui = RegionUi(
            region=region,
            folder=folder,
            gizmo=gizmo,
            box=box,
            scale_handles=scale_handles,
            center_gui=center_gui,
            half_size_gui=half_size_gui,
            delete_button=delete_button,
        )
        region_uis.append(ui)

        @gizmo.on_update
        def _(_) -> None:
            if ui.updating:
                return
            apply_region(
                ui,
                MaskRegion.from_center_half_size(
                    ui.region.name,
                    np.asarray(ui.gizmo.position),
                    ui.region.half_size,
                    np.asarray(ui.gizmo.wxyz),
                ),
                from_gizmo=True,
            )

        for axis, handle in enumerate(scale_handles):
            @handle.on_update
            def _(_, axis=axis, handle=handle) -> None:
                if ui.updating:
                    return
                half = np.array(ui.region.half_size, dtype=np.float64)
                half[axis] = max(0.05, abs(float(np.asarray(handle.position)[axis])))
                apply_region(
                    ui,
                    MaskRegion.from_center_half_size(
                        ui.region.name,
                        ui.region.center,
                        half,
                        ui.region.wxyz,
                    ),
                )

        @center_gui.on_update
        def _(_) -> None:
            if ui.updating:
                return
            apply_region(
                ui,
                MaskRegion.from_center_half_size(
                    ui.region.name,
                    np.asarray(center_gui.value),
                    ui.region.half_size,
                    ui.region.wxyz,
                ),
                from_gui=True,
            )

        @half_size_gui.on_update
        def _(_) -> None:
            if ui.updating:
                return
            apply_region(
                ui,
                MaskRegion.from_center_half_size(
                    ui.region.name,
                    ui.region.center,
                    np.asarray(half_size_gui.value),
                    ui.region.wxyz,
                ),
                from_gui=True,
            )

        @delete_button.on_click
        def _(_) -> None:
            remove_region_ui(ui)

    with server.gui.add_folder("Relocalization mask") as _:
        add_button = server.gui.add_button("Add region", color=(80, 200, 120))
        save_button = server.gui.add_button("Save mask", color=(80, 160, 255))
        save_status = server.gui.add_text("Save path", initial_value=str(output_path))

        @add_button.on_click
        def _(_) -> None:
            center = _default_map_center(poses)
            existing_names = {r.name for r in regions}
            idx = 0
            while f"region_{idx}" in existing_names:
                idx += 1
            name = f"region_{idx}"
            region = MaskRegion.from_center_half_size(name, center, np.array([1.0, 1.0, 0.5]))
            regions.append(region)
            add_region_ui(region)
            refresh_allowed_keyposes()

        @save_button.on_click
        def _(_) -> None:
            save_relocalization_mask(str(output_path), regions)
            save_status.value = f"Saved {len(regions)} region(s) -> {output_path}"
            print(save_status.value)

    for region in regions:
        add_region_ui(region)
    refresh_allowed_keyposes()

    print(f"Relocalization mask editor running at http://{args.host}:{args.port}")
    print(f"Save target: {output_path}")
    print("Drag the colored gizmo to move/rotate the box; drag face handles to scale.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Args))
