"""3D box whitelist for map keyframe relocalization candidates.

Regions are oriented boxes (OBB). Axis-aligned JSON from the first editor
version still loads and is treated as an OBB with identity rotation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

DEFAULT_RELOCALIZATION_MASK_FILENAME = "relocalization_mask.json"


def _as_xyz(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    return arr.copy()


def _as_wxyz(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return arr / norm


def rotmat_from_wxyz(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _as_wxyz(wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MaskRegion:
    name: str
    center: np.ndarray
    half_size: np.ndarray
    wxyz: np.ndarray

    @property
    def min_xyz(self) -> np.ndarray:
        return self.center - self.half_size

    @property
    def max_xyz(self) -> np.ndarray:
        return self.center + self.half_size

    @property
    def dimensions(self) -> np.ndarray:
        return self.half_size * 2.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": "obb",
            "center": [float(x) for x in self.center],
            "half_size": [float(x) for x in self.half_size],
            "wxyz": [float(x) for x in self.wxyz],
        }

    def contains(self, point: np.ndarray) -> bool:
        p = np.asarray(point, dtype=np.float64).reshape(3)
        local = rotmat_from_wxyz(self.wxyz).T @ (p - self.center)
        return bool(np.all(np.abs(local) <= self.half_size + 1e-9))

    @classmethod
    def from_center_half_size(
        cls,
        name: str,
        center: np.ndarray,
        half_size: np.ndarray,
        wxyz: np.ndarray | None = None,
    ) -> "MaskRegion":
        half = np.abs(_as_xyz(half_size))
        half = np.maximum(half, 0.05)
        quat = _as_wxyz((1.0, 0.0, 0.0, 0.0) if wxyz is None else wxyz)
        return cls(name=name, center=_as_xyz(center), half_size=half, wxyz=quat)

    @classmethod
    def from_aabb(cls, name: str, min_xyz: np.ndarray, max_xyz: np.ndarray) -> "MaskRegion":
        min_xyz = _as_xyz(min_xyz)
        max_xyz = _as_xyz(max_xyz)
        if np.any(min_xyz > max_xyz):
            raise ValueError(f"region {name!r} min must be <= max on every axis")
        center = (min_xyz + max_xyz) * 0.5
        half_size = (max_xyz - min_xyz) * 0.5
        return cls.from_center_half_size(name, center, half_size)


# Backward-compatible name used by the first editor revision.
AabbRegion = MaskRegion


def _parse_xyz(value, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of 3 numbers")
    return np.asarray([float(v) for v in value], dtype=np.float64)


def _parse_wxyz(value, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field_name} must be a list of 4 numbers")
    return _as_wxyz([float(v) for v in value])


def load_relocalization_mask(path: str) -> list[MaskRegion]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("relocalization mask root must be a JSON object")
    regions_raw = data.get("regions")
    if not isinstance(regions_raw, list):
        raise ValueError("relocalization mask must contain a 'regions' list")
    regions: list[MaskRegion] = []
    for idx, region in enumerate(regions_raw):
        if not isinstance(region, dict):
            raise ValueError(f"region[{idx}] must be an object")
        name = str(region.get("name", f"region_{idx}"))
        region_type = str(region.get("type", "aabb")).strip().lower()
        if region_type in ("aabb", ""):
            min_xyz = _parse_xyz(region["min"], f"regions[{idx}].min")
            max_xyz = _parse_xyz(region["max"], f"regions[{idx}].max")
            regions.append(MaskRegion.from_aabb(name, min_xyz, max_xyz))
            continue
        if region_type != "obb":
            raise ValueError(f"region[{idx}] unsupported type: {region_type!r}")
        center = _parse_xyz(region["center"], f"regions[{idx}].center")
        half_size = _parse_xyz(region["half_size"], f"regions[{idx}].half_size")
        wxyz = _parse_wxyz(region.get("wxyz", [1.0, 0.0, 0.0, 0.0]), f"regions[{idx}].wxyz")
        regions.append(MaskRegion.from_center_half_size(name, center, half_size, wxyz))
    return regions


def save_relocalization_mask(path: str, regions: list[MaskRegion], *, frame: str = "map") -> None:
    payload = {
        "version": 1,
        "frame": frame,
        "regions": [region.to_dict() for region in regions],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def point_in_aabb(point: np.ndarray, region: MaskRegion) -> bool:
    return region.contains(point)


def point_in_any_aabb(point: np.ndarray, regions: list[MaskRegion]) -> bool:
    return any(region.contains(point) for region in regions)


def allowed_keyframe_timestamps(map_poses: dict, regions: list[MaskRegion]) -> set[int]:
    allowed: set[int] = set()
    for timestamp, pose in map_poses.items():
        position = np.asarray(pose, dtype=np.float64)[:3, 3]
        if point_in_any_aabb(position, regions):
            allowed.add(int(timestamp))
    return allowed


def resolve_relocalization_mask_path(tinynav_map_path: str, nav_flow_config: dict | None) -> str | None:
    if not isinstance(nav_flow_config, dict):
        return None
    raw = nav_flow_config.get("relocalization_mask")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Invalid relocalization_mask: {raw!r}")
    raw = raw.strip()
    if os.path.isabs(raw):
        return raw
    return os.path.join(tinynav_map_path, raw)


def load_nav_flow_dict(tinynav_map_path: str) -> dict | None:
    config_path = os.path.join(tinynav_map_path, "nav_flow.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None
