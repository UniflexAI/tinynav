"""Bake a map's capture-path priors — `path_climb.npy` and `path_speed.npy`.

Both are pure functions of `poses.npy`. `build_map_node` writes them at capture time
and `map_node.load_map_priors` rebakes any that are missing or older than the poses,
so a normal capture needs nothing manual. This exists for the two cases that are not
a normal capture:

  - a map built before a prior existed, or by an older/upstream build_map
  - baking without starting the stack at all

Pure numpy over a file every stock map already has, so it needs neither the stack
stopped nor the TRT engines.

    python3 tool/bake_path_priors.py --map_path /tinynav/tinynav_db/maps/map_X
    python3 tool/bake_path_priors.py --maps-root /tinynav/tinynav_db/maps

It lives here rather than in tinynav-pilot's deploy/ because the priors are the fork's:
only the fork's nodes read them, and core_runtime importing them would tie it to a
tinynav pin that is not on upstream main.
"""

from __future__ import annotations

import argparse
import os
import sys

from tinynav.core.path_climb import bake as bake_path_climb
from tinynav.core.path_speed import bake as bake_path_speed

_BAKERS = (('path_climb.npy', bake_path_climb), ('path_speed.npy', bake_path_speed))


def bake_map(map_path: str, *, force: bool = False, dry_run: bool = False) -> None:
    """Bake both priors for one map, reporting a line each. Never raises for one bad
    prior — each `bake` reports its own refusal (no poses.npy, already baked) rather
    than failing the run."""
    print(map_path)
    for filename, bake in _BAKERS:
        if dry_run:
            print(f'  {filename}: WOULD bake')
            continue
        print(f'  {bake(map_path, force=force)}')


def _map_dirs(maps_root: str, names) -> list[str]:
    if names:
        return [os.path.join(maps_root, n) for n in names]
    if not os.path.isdir(maps_root):
        return []
    return sorted(os.path.join(maps_root, n) for n in os.listdir(maps_root)
                  if os.path.isdir(os.path.join(maps_root, n)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--map_path', help='a single map directory')
    parser.add_argument('--maps-root', help='bake every map under this directory')
    parser.add_argument('--map', action='append', dest='maps',
                        help='map directory name under --maps-root; repeatable')
    parser.add_argument('--force', action='store_true',
                        help='rebake even when the prior is present and current')
    parser.add_argument('--dry-run', action='store_true',
                        help='report what would be baked without writing')
    args = parser.parse_args()

    if args.map_path:
        paths = [args.map_path]
    elif args.maps_root:
        paths = _map_dirs(args.maps_root, args.maps)
    else:
        parser.error('one of --map_path or --maps-root is required')

    if not paths:
        print('no maps found', file=sys.stderr)
        return 1

    failed = []
    for map_path in paths:
        if not os.path.isdir(map_path):
            print(f'not a directory: {map_path}', file=sys.stderr)
            failed.append(map_path)
            continue
        try:
            bake_map(map_path, force=args.force, dry_run=args.dry_run)
        except Exception as e:  # noqa: BLE001 - one bad map must not stop the rest
            print(f'  FAILED: {e}', file=sys.stderr)
            failed.append(map_path)

    if failed:
        print(f'{len(failed)} map(s) failed', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
