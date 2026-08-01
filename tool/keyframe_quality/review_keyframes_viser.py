#!/usr/bin/env python3
"""Interactive 3D review of find_confusing_keyframes.py's flagged keyframes, viser-based --
same occupancy-grid-as-point-cloud + clickable camera-frustum pattern as tool/poi_editor.py and
tool/path_editor.py, so browsing/selecting keyframes here feels like those existing tools.

Every other (non-flagged) map keyframe is drawn as a small amber point for trajectory context.
Every flagged keyframe gets a full camera frustum, clickable to toggle it between "will be
removed" (red) and "kept despite the flag" (green) -- exactly mirroring the checkbox in
generate_review_page.py's HTML report, and starting in the same default state (all flagged
keyframes start red/excluded). Clicking a frustum also loads its actual camera image onto it,
same as tool/poi_editor.py's frustum click handler.

This tool is for spatial browsing and selection. Comparing a flagged keyframe against the actual
retrieval candidates that got it flagged (side-by-side thumbnails) is still generate_review_page.py's
job -- viser's GUI sidebar is too narrow for that kind of wide image comparison, so the two tools
are complementary rather than merged.

"Save Exclusion List" writes the current selection to --out_selection_json, in the same
{"map_path", "excluded_timestamps"} shape prune_map.py's --exclude_file line format expects a
plain newline list, so this writes a JSON with an explicit note on how to turn it into that.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
import viser
import viser.transforms as vtf

from tool.video_db import VideoDB

EXCLUDED_COLOR = (210, 69, 45)
KEPT_COLOR = (45, 154, 78)
CONTEXT_COLOR = (255, 190, 60)


@dataclass(frozen=True)
class Args:
    tinynav_map_path: Path
    """Map directory find_confusing_keyframes.py was run against."""

    flagged_json: Path
    """find_confusing_keyframes.py --out_json output."""

    out_selection_json: Path | None = None
    """Where 'Save Exclusion List' writes. Defaults to <tinynav_map_path>/keyframe_review_selection.json"""

    max_context_keyframes: int = 2000
    """Cap on how many non-flagged keyframes get a context point (perf on very large maps)."""

    host: str = "0.0.0.0"
    port: int = 8092


def _open_infra1_video_db(map_dir: Path) -> VideoDB | None:
    db_dir = map_dir / "infra1_images_db"
    if not db_dir.exists():
        return None
    try:
        return VideoDB(dir_path=str(db_dir), mode="read")
    except Exception as e:
        print(f"Warning: failed to open infra1 VideoDB at {db_dir}: {e}")
        return None


def _load_camera_image(db: VideoDB | None, ts: int) -> np.ndarray | None:
    if db is None:
        return None
    img = db.read(ts)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _add_occupancy_point_cloud(server: viser.ViserServer, map_path: Path) -> None:
    occ_path = map_path / "occupancy_grid.npy"
    meta_path = map_path / "occupancy_meta.npy"
    if not occ_path.exists() or not meta_path.exists():
        print(f"Warning: no occupancy_grid.npy/occupancy_meta.npy in {map_path}, skipping floor plan")
        return
    occupancy_grid = np.load(occ_path)
    occupancy_meta = np.load(meta_path)
    origin = occupancy_meta[:3].astype(np.float32)
    resolution = float(occupancy_meta[3])

    x_y_plane = np.max(occupancy_grid, axis=2)
    z_plane = float(origin[2])

    def to_world(xy_indices: np.ndarray) -> np.ndarray:
        if len(xy_indices) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        pts = np.zeros((len(xy_indices), 3), dtype=np.float32)
        pts[:, 0] = origin[0] + xy_indices[:, 0] * resolution
        pts[:, 1] = origin[1] + xy_indices[:, 1] * resolution
        pts[:, 2] = z_plane
        return pts

    free_pts = to_world(np.argwhere(x_y_plane == 1))
    occupied_pts = to_world(np.argwhere(x_y_plane == 2))
    if len(free_pts):
        server.scene.add_point_cloud(
            "/occupancy/free", points=free_pts,
            colors=np.tile(np.array([[0.2, 0.4, 1.0]], dtype=np.float32), (len(free_pts), 1)),
            point_size=resolution * 0.8, point_shape="rounded",
        )
    if len(occupied_pts):
        server.scene.add_point_cloud(
            "/occupancy/occupied", points=occupied_pts,
            colors=np.tile(np.array([[0.6, 0.6, 0.6]], dtype=np.float32), (len(occupied_pts), 1)),
            point_size=resolution * 0.8, point_shape="rounded",
        )


def main(args: Args) -> None:
    map_path = args.tinynav_map_path
    with args.flagged_json.open(encoding="utf-8") as f:
        summary = json.load(f)
    flagged = {int(kf["timestamp_ns"]): kf for kf in summary["flagged_keyframes"] if kf["bad_participation"] > 0}

    poses = np.load(map_path / "poses.npy", allow_pickle=True).item()
    poses = {int(ts): np.asarray(p) for ts, p in poses.items()}

    if (map_path / "intrinsics.npy").exists():
        camera_K = np.load(map_path / "intrinsics.npy", allow_pickle=True)
    elif (map_path / "rgb_camera_intrinsics.npy").exists():
        camera_K = np.load(map_path / "rgb_camera_intrinsics.npy", allow_pickle=True)
    else:
        raise FileNotFoundError("Neither intrinsics.npy nor rgb_camera_intrinsics.npy exists.")
    fx, fy, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]

    infra1_db = _open_infra1_video_db(map_path)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")

    _add_occupancy_point_cloud(server, map_path)

    context_ts = [ts for ts in poses if ts not in flagged]
    if len(context_ts) > args.max_context_keyframes:
        step = max(1, len(context_ts) // args.max_context_keyframes)
        context_ts = context_ts[::step]
    if context_ts:
        context_points = np.stack([poses[ts][:3, 3] for ts in context_ts])
        server.scene.add_point_cloud(
            "/keyframes/context", points=context_points,
            colors=np.tile(np.array([CONTEXT_COLOR], dtype=np.float32) / 255.0, (len(context_points), 1)),
            point_size=0.06, point_shape="circle",
        )

    out_selection_json = args.out_selection_json or (map_path / "keyframe_review_selection.json")
    excluded: set[int] = set(flagged.keys())
    frustum_handles: dict[int, viser.SceneHandle] = {}
    checkbox_handles: dict[int, viser.GuiCheckboxHandle] = {}

    with server.gui.add_folder("Keyframe Review") as _:
        status = server.gui.add_text(
            "Status", initial_value=f"{len(excluded)}/{len(flagged)} selected for removal"
        )
        save_button = server.gui.add_button("Save Exclusion List", color=(45, 125, 210))

        @save_button.on_click
        def _(_) -> None:
            payload = {
                "map_path": str(map_path),
                "excluded_timestamps": sorted(excluded),
                "exclude_timestamps_csv": ",".join(str(t) for t in sorted(excluded)),
            }
            with out_selection_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            status.value = f"saved {len(excluded)} timestamps -> {out_selection_json}"
            print(status.value)

        def set_excluded(ts: int, is_excluded: bool, *, sync_checkbox: bool) -> None:
            if is_excluded:
                excluded.add(ts)
            else:
                excluded.discard(ts)
            frustum_handles[ts].color = EXCLUDED_COLOR if is_excluded else KEPT_COLOR
            if sync_checkbox:
                checkbox_handles[ts].value = is_excluded
            status.value = f"{len(excluded)}/{len(flagged)} selected for removal"

        poi_list_container = server.gui.add_folder("Flagged Keyframes (worst first)")
        for ts, kf in sorted(flagged.items(), key=lambda kv: kv[1]["badness_ratio"], reverse=True):
            pose = poses[ts]
            rotation = vtf.SO3.from_matrix(pose[:3, :3])
            frustum = server.scene.add_camera_frustum(
                name=f"/flagged/{ts}",
                fov=float(2 * np.arctan(cx / fx)),
                aspect=float(cx / cy),
                scale=0.25,
                color=EXCLUDED_COLOR,
                wxyz=rotation.wxyz,
                position=pose[:3, 3],
                image=None,
            )
            frustum_handles[ts] = frustum

            with poi_list_container:
                with server.gui.add_folder(f"ts={ts}  badness={kf['badness_ratio'] * 100:.0f}%") as _:
                    server.gui.add_text(
                        "participation",
                        initial_value=f"{kf['bad_participation']}/{kf['total_participation']} bad retrievals",
                    )
                    toggle = server.gui.add_checkbox("Exclude (remove from map)", initial_value=True)
                    checkbox_handles[ts] = toggle

                    @toggle.on_update
                    def _(_, ts=ts, toggle=toggle) -> None:
                        set_excluded(ts, toggle.value, sync_checkbox=False)

            @frustum.on_click
            def _(event, ts=ts, frustum=frustum) -> None:
                img = _load_camera_image(infra1_db, ts)
                if img is not None:
                    frustum.image = img
                set_excluded(ts, ts not in excluded, sync_checkbox=True)

    print(f"viser server running at http://{args.host}:{args.port} (map={map_path}, flagged={len(flagged)})")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if infra1_db is not None:
            infra1_db.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
