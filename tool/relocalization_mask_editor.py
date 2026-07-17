from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, PolygonSelector


@dataclass(frozen=True)
class Args:
    tinynav_map_path: Path
    """Tinynav map directory containing poses.npy."""

    output_name: str = "relocalization_mask.json"
    """Mask file saved under tinynav_map_path."""

    show_occupancy: bool = True
    """Show occupancy_grid.npy as a 2D background when available."""

    point_size: float = 8.0
    """Keyframe scatter point size."""


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


def _draw_occupancy_background(ax, map_dir: Path) -> None:
    occupancy_path = map_dir / "occupancy_grid.npy"
    meta_path = map_dir / "occupancy_meta.npy"
    if not occupancy_path.exists() or not meta_path.exists():
        return

    occupancy = np.load(occupancy_path)
    meta = np.load(meta_path)
    origin = meta[:3].astype(np.float32)
    resolution = float(meta[3])

    grid_xy = np.max(occupancy, axis=2).T
    image = np.zeros_like(grid_xy, dtype=np.float32)
    image[grid_xy == 1] = 0.35
    image[grid_xy == 2] = 0.85

    extent = [
        origin[0],
        origin[0] + occupancy.shape[0] * resolution,
        origin[1],
        origin[1] + occupancy.shape[1] * resolution,
    ]
    ax.imshow(image, origin="lower", extent=extent, cmap="gray", alpha=0.35)


class RelocalizationMaskEditor:
    def __init__(self, args: Args):
        self.args = args
        self.map_dir = args.tinynav_map_path
        self.mask_path = self.map_dir / args.output_name
        self.timestamps, self.positions = _load_poses(self.map_dir)
        self.xy = self.positions[:, :2]
        self.mask = _load_mask(self.mask_path)
        self.excluded = {int(t) for t in self.mask.get("excluded_timestamps", [])}
        self.current_selection: set[int] = set()

        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        plt.subplots_adjust(bottom=0.16)

        if args.show_occupancy:
            _draw_occupancy_background(self.ax, self.map_dir)

        self.ax.plot(self.xy[:, 0], self.xy[:, 1], color="0.55", linewidth=0.8, alpha=0.7)
        self.all_scatter = self.ax.scatter(
            self.xy[:, 0],
            self.xy[:, 1],
            s=args.point_size,
            c="tab:blue",
            alpha=0.55,
            label="available keyframes",
        )
        self.excluded_scatter = self.ax.scatter([], [], s=args.point_size * 2.0, c="tab:red", label="excluded")
        self.selected_scatter = self.ax.scatter([], [], s=args.point_size * 2.0, c="orange", label="current selection")

        self.ax.set_title(
            "Relocalization Mask Editor\n"
            "Draw a polygon to select keyframes. Save writes relocalization_mask.json."
        )
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.axis("equal")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="upper right")

        self.selector = PolygonSelector(self.ax, self.on_select, useblit=True)
        self.status_text = self.fig.text(0.02, 0.03, "", fontsize=10)

        self._add_button("Save", 0.60, self.save)
        self._add_button("Clear Selection", 0.69, self.clear_selection)
        self._add_button("Reset Mask", 0.82, self.reset_mask)
        self._add_button("Quit", 0.92, self.quit)

        self.refresh()

    def _add_button(self, label: str, left: float, callback) -> None:
        ax_button = self.fig.add_axes([left, 0.04, 0.08, 0.055])
        button = Button(ax_button, label)
        button.on_clicked(callback)
        setattr(self, f"_button_{label.replace(' ', '_').lower()}", button)

    def on_select(self, vertices: list[tuple[float, float]]) -> None:
        if len(vertices) < 3:
            self.current_selection = set()
            self.refresh()
            return

        polygon = MplPath(np.asarray(vertices, dtype=np.float32))
        inside = polygon.contains_points(self.xy)
        selected = {self.timestamps[i] for i, flag in enumerate(inside) if flag}
        self.current_selection = selected
        self.excluded |= selected
        self.mask.setdefault("zones", []).append(
            {
                "name": f"zone_{len(self.mask.get('zones', []))}",
                "polygon_xy": [[float(x), float(y)] for x, y in vertices],
                "excluded_timestamps": sorted(selected),
            }
        )
        self.refresh()

    def refresh(self) -> None:
        excluded_indices = [i for i, t in enumerate(self.timestamps) if t in self.excluded]
        selected_indices = [i for i, t in enumerate(self.timestamps) if t in self.current_selection]

        self.excluded_scatter.set_offsets(self.xy[excluded_indices] if excluded_indices else np.empty((0, 2)))
        self.selected_scatter.set_offsets(self.xy[selected_indices] if selected_indices else np.empty((0, 2)))
        self.status_text.set_text(
            f"map={self.map_dir} | keyframes={len(self.timestamps)} | "
            f"excluded={len(self.excluded)} | selected={len(self.current_selection)} | "
            f"output={self.mask_path}"
        )
        self.fig.canvas.draw_idle()

    def save(self, _event=None) -> None:
        self.mask["version"] = 1
        self.mask["excluded_timestamps"] = sorted(self.excluded)
        _save_mask(self.mask_path, self.mask)
        print(f"Saved {len(self.excluded)} excluded keyframes to {self.mask_path}")
        self.refresh()

    def clear_selection(self, _event=None) -> None:
        self.current_selection = set()
        self.refresh()

    def reset_mask(self, _event=None) -> None:
        self.excluded = set()
        self.current_selection = set()
        self.mask = {"version": 1, "excluded_timestamps": [], "zones": []}
        self.refresh()

    def quit(self, _event=None) -> None:
        plt.close(self.fig)

    def run(self) -> None:
        plt.show()


def main(args: Args) -> None:
    RelocalizationMaskEditor(args).run()


if __name__ == "__main__":
    main(tyro.cli(Args))
