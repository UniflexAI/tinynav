"""Backfill the two capture-path priors that older maps predate.

`build_map_node` grew these after the first maps were captured:

  - `path_climb.npy`  — stair hint (map_node -> /planning/on_stairs)
  - `path_speed.npy`  — capture-speed prior (map_node -> /planning/speed_cap)

Both are pure functions of `poses.npy`, so a map captured before they existed can
be brought up to date without re-capturing.

This lives in the fork because the priors do: `tinynav.core.path_speed` and
`tinynav.core.stair_hint` compute them and only this branch's map_node reads them.
Its companion — the DINOv2 patch VLAD relocalization index, which upstream
map_node requires — is backfilled by `deploy/backfill_map_priors.py` in
tinynav-pilot, which stays pin-agnostic by depending on nothing this branch adds.
Run both when a pre-existing map needs to be navigable on this branch.

No ROS, no GPU, no TensorRT: just numpy over the stored poses. Safe to run against
a live stack, though `map_node` reads these at map load, so restart it afterwards
to pick them up.

    python3 tool/backfill_path_priors.py

Idempotent: each artifact is skipped when already present (`--force` recomputes).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

DEFAULT_MAPS_ROOT = "/tinynav/tinynav_db/maps"


def _map_dirs(maps_root: str, names: list[str] | None) -> list[str]:
    if names:
        return [os.path.join(maps_root, n) for n in names]
    out = []
    for name in sorted(os.listdir(maps_root)):
        path = os.path.join(maps_root, name)
        # Skip the 'map'/'active' symlinks and any half-built capture.
        if os.path.islink(path) or not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "poses.npy")):
            out.append(path)
    return out


def backfill_path_priors(map_path: str, poses: dict, force: bool, dry_run: bool) -> None:
    from tinynav.core.path_speed import compute_path_speed
    from tinynav.core.stair_hint import compute_path_climb

    for filename, compute, describe in (
        ("path_climb.npy", compute_path_climb,
         lambda a: f"{int((a[:, 3] >= 0.5).sum())}/{len(a)} samples climbing"),
        ("path_speed.npy", compute_path_speed,
         lambda a: f"median capture speed {np.nanmedian(a[:, 3]):.2f} m/s"),
    ):
        out_path = os.path.join(map_path, filename)
        if os.path.exists(out_path) and not force:
            print(f"  {filename}: present, skipping")
            continue
        if dry_run:
            print(f"  {filename}: WOULD compute")
            continue
        array = compute(poses)
        np.save(out_path, array)
        print(f"  {filename}: saved ({describe(array)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--maps-root", default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--map", action="append", dest="maps",
                        help="map directory name; repeatable. Default: every map under --maps-root")
    parser.add_argument("--force", action="store_true",
                        help="recompute artifacts that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is missing without writing")
    args = parser.parse_args()

    map_paths = _map_dirs(args.maps_root, args.maps)
    if not map_paths:
        print(f"no maps found under {args.maps_root}", file=sys.stderr)
        return 1

    failed = []
    for map_path in map_paths:
        print(f"{map_path}")
        try:
            poses = np.load(f"{map_path}/poses.npy", allow_pickle=True).item()
            backfill_path_priors(map_path, poses, args.force, args.dry_run)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed.append(map_path)

    if failed:
        print(f"\n{len(failed)}/{len(map_paths)} maps failed: {failed}", file=sys.stderr)
        return 1
    print(f"\n{len(map_paths)} map(s) up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
