"""Rewrite a built map's depths and SuperPoint descriptors in `map_store`'s encoding.

Maps built before it hold depth as float32 metres (1.36 MB a keyframe at 640x544)
and descriptors as float32; this stores depth as uint16 millimetres and the
descriptors in FP16 when that is exact, which halves both. Readers decode through
`TinyNavDB`, so nothing else changes. Depth on a mm grid (Looper's) comes back
bit-identical; other depth within 0.5 mm.

The shelves are Berkeley DB files that do not shrink when an entry is overwritten
with a smaller one, so each store is written anew beside the map, checked entry by
entry against the original, and only then moved over it. Idempotent: a store that is
already encoded is left alone. Touches nothing else in the map.

    python3 tool/slim_map.py /tinynav/tinynav_db/maps/<name> [--dry-run]
"""
import argparse
import os
import shelve
import shutil
import sys
import time

import numpy as np

from tinynav.core.map_store import DEPTH_SCALE, decode_depth, encode_depth, half_descriptors

#: An encoded depth is the original rounded to whole mm (plus float32's rounding).
_DEPTH_TOL_M = 0.5 / float(DEPTH_SCALE) + 1e-6


def _store_bytes(map_dir: str, name: str) -> int:
    return sum(os.path.getsize(os.path.join(map_dir, f)) for f in os.listdir(map_dir)
               if f == name or f.startswith(name + '.'))


def _slim_depth(value):
    value = np.asarray(value)
    return value if value.dtype == np.uint16 else encode_depth(value)


def _depth_matches(original, stored) -> bool:
    """Every depth kept within rounding to the mm, and only unusable ones dropped."""
    original = np.asarray(original)
    if original.dtype == np.uint16:
        return np.array_equal(original, stored)
    decoded = decode_depth(stored)
    kept = np.asarray(stored) != 0
    mm = np.rint(original.astype(np.float64) * float(DEPTH_SCALE))
    usable = np.isfinite(mm) & (mm > 0) & (mm <= np.iinfo(np.uint16).max)
    return (original.shape == decoded.shape
            and np.array_equal(kept, usable)
            and np.all(np.abs(decoded[kept] - original[kept]) <= _DEPTH_TOL_M))


def _same(a, b) -> bool:
    """Equal as values: an FP16 array against the float32 it was halved from counts."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        b = np.asarray(b)
        return a.shape == b.shape and (np.array_equal(a, b) if a.dtype == b.dtype else
                                       np.array_equal(a.astype(np.float32), b.astype(np.float32)))
    return a == b


def _nbytes(value) -> int:
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    return value.nbytes if isinstance(value, np.ndarray) else 0


#: store -> (encode one entry, returning it unchanged when already encoded;
#:           does the stored entry hold the original)
STORES = {
    'depths': (_slim_depth, _depth_matches),
    'features': (half_descriptors, _same),
}


def _pending(map_dir: str, name: str) -> tuple[int, int, int]:
    """(entries, entries to re-encode, bytes saved) of one store."""
    encode, _ = STORES[name]
    total = todo = saved = 0
    with shelve.open(os.path.join(map_dir, name), flag='r') as db:
        for key in db.keys():
            value = db[key]
            slim = encode(value)
            total += 1
            if slim is not value:
                todo += 1
                saved += _nbytes(value) - _nbytes(slim)
    return total, todo, saved


def _rewrite(map_dir: str, name: str) -> None:
    encode, matches = STORES[name]
    tmp_dir = os.path.join(map_dir, '.slim-tmp')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    src_path, dst_path = os.path.join(map_dir, name), os.path.join(tmp_dir, name)
    try:
        started = time.monotonic()
        with shelve.open(src_path, flag='r') as src, shelve.open(dst_path, flag='n') as dst:
            keys = list(src.keys())
            for i, key in enumerate(keys, 1):
                dst[key] = encode(src[key])
                if i % 500 == 0:
                    print(f'    {name}: {i}/{len(keys)} written in {time.monotonic() - started:.0f}s', flush=True)
        with shelve.open(src_path, flag='r') as src, shelve.open(dst_path, flag='r') as dst:
            if sorted(src.keys()) != sorted(dst.keys()):
                raise RuntimeError(f'{name}: the rewritten store has other keys; the map is unchanged')
            for key in src.keys():
                if not matches(src[key], dst[key]):
                    raise RuntimeError(f'{name}: entry {key} does not round-trip; the map is unchanged')
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    for f in os.listdir(tmp_dir):
        os.replace(os.path.join(tmp_dir, f), os.path.join(map_dir, f))
    os.rmdir(tmp_dir)
    print(f'    {name}: {len(keys)} entries rewritten and checked in {time.monotonic() - started:.0f}s')


def slim(map_dir: str, dry_run: bool) -> None:
    for name in STORES:
        if not os.path.exists(os.path.join(map_dir, name + '.db')) and not os.path.exists(os.path.join(map_dir, name)):
            print(f'  {name}: absent, skipped')
            continue
        total, todo, saved = _pending(map_dir, name)
        before = _store_bytes(map_dir, name)
        print(f'  {name}: {todo}/{total} entries to re-encode, ~{saved / 2**30:.2f} of {before / 2**30:.2f} GiB saved')
        if not todo or dry_run:
            continue
        free = shutil.disk_usage(map_dir).free
        if free < before - saved + (1 << 30):
            raise RuntimeError(f'{name}: {free / 2**30:.1f} GiB free, the rewrite needs '
                               f'{(before - saved) / 2**30:.1f} GiB beside the map')
        _rewrite(map_dir, name)
        print(f'  {name}: {before / 2**30:.2f} -> {_store_bytes(map_dir, name) / 2**30:.2f} GiB')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('map_dir')
    parser.add_argument('--dry-run', action='store_true', help='only say what would be re-encoded')
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.map_dir, 'poses.npy')):
        print(f'{args.map_dir} is not a map (no poses.npy)', file=sys.stderr)
        return 1
    print(f'{args.map_dir}:')
    slim(args.map_dir, args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
