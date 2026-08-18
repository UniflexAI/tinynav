"""Keep only the climb runs a map really has; clear the rest of `path_climb.npy`.

`compute_path_climb` infers climbing from path z, and VIO z drift fakes it: on
n2n3-backward 21 runs were labelled where 4 are real, relaxing 61% of the route's
obstacle span filter -- which is how an obstacle taller than the strict threshold but
shorter than the relaxed one becomes invisible to the planner while occupancy still
shows it. Until the labels are authored rather than inferred, this prunes them by hand.

  python3 tool/prune_path_climb.py --map_path <map> --list
  python3 tool/prune_path_climb.py --map_path <map> --keep 20-71,1139-1182

Runs are given as inclusive sample-index ranges, as printed by --list. Writes nothing
without --keep; back up the file yourself first (it is map data, not source).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

CLIMBING = 0.5


def _runs(flags) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _arclength(xy) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def _coverage(xy, flags, radius_m: float) -> float:
    """Fraction of samples inside a relaxed square, i.e. what the planner actually
    loosens. Chebyshev because the region is grown with a square maximum_filter."""
    keyed = xy[flags]
    if not len(keyed):
        return 0.0
    d = np.abs(xy[:, None, :] - keyed[None, :, :]).max(2).min(1)
    return float(100.0 * (d <= radius_m).mean())


def describe(labels) -> str:
    xy, z, flags = labels[:, :2], labels[:, 2], labels[:, 3] >= CLIMBING
    s = _arclength(xy)
    lines = [f'path {s[-1]:.1f}m, {int(flags.sum())}/{len(flags)} samples climbing']
    for a, b in _runs(flags):
        lines.append(
            f'  {a}-{b}\ts {s[a]:.1f}-{s[b]:.1f}m ({100 * s[a] / s[-1]:.0f}%)\t'
            f'dz={z[b] - z[a]:+.2f} over {s[b] - s[a]:.1f}m')
    for r in (0.75, 1.5):
        lines.append(f'  relaxed coverage at radius {r}m: {_coverage(xy, flags, r):.0f}%')
    return '\n'.join(lines)


def parse_keep(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition('-')
        out.append((int(a), int(b or a)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--map_path', required=True)
    ap.add_argument('--keep', help='inclusive index ranges to keep, e.g. 20-71,1139-1182')
    ap.add_argument('--list', action='store_true', help='print the runs and exit')
    args = ap.parse_args()

    path = os.path.join(args.map_path, 'path_climb.npy')
    labels = np.load(path, allow_pickle=True)
    print(describe(labels))
    if args.list or not args.keep:
        return 0

    kept = np.zeros(len(labels), dtype=bool)
    for a, b in parse_keep(args.keep):
        kept[a:b + 1] = True
    out = labels.copy()
    out[:, 3] = np.where(kept, 1.0, 0.0).astype(labels.dtype)
    np.save(path, out)
    print('\nwrote', path)
    print(describe(np.load(path, allow_pickle=True)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
