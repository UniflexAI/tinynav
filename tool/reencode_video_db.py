"""Re-encode a built map's image video with a short GOP, so a random read decodes a
few frames instead of up to x264's default 250 (`video_db.GOP`).

    python3 tool/reencode_video_db.py <map_dir> [--sub infra1_images_db] [--gop 15]
    python3 tool/reencode_video_db.py <map_dir> --restore

Every frame is read the way relocalization reads it (`VideoDB.read`), in index order,
and written again through `VideoDB.write`, so the new video holds the same frames under
the same timestamps; meta.json is not touched. Before the new video takes the old one's
place it is checked: the same frame count, the same timestamp -> index map, and a
sample of frames close to the old ones (x264 at crf 18 again: a level or two).

The old video is moved, not deleted, to `<db>/video-backup/<map>/<sub>/video.mp4`
(outside maps/, so nothing that lists or copies maps sees it). `--restore` puts it back.
Nothing that reads the map should hold it open meanwhile: run it with nav off.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import av
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_db  # noqa: E402
from video_db import VideoDB  # noqa: E402

#: The worst mean absolute difference a re-encoded frame may have from the old one, in
#: grey levels.
MAX_MEAN_DIFF = 3.0
#: Every this-many-th frame is compared, and the last one.
CHECK_EVERY = 97


def backup_path(map_dir: str, sub: str) -> str:
    map_dir = os.path.realpath(map_dir)
    db = os.path.dirname(os.path.dirname(map_dir))          # <db>/maps/<map>
    return os.path.join(db, 'video-backup', os.path.basename(map_dir), sub, 'video.mp4')


def keyframe_gaps(path: str) -> tuple[int, int, float]:
    """(frames, largest gap between keyframes, mean gap), from the packets alone."""
    with av.open(path) as c:
        stream = c.streams.video[0]
        keys, n = [], 0
        for pkt in c.demux(stream):
            if pkt.size == 0:
                continue
            if pkt.is_keyframe:
                keys.append(n)
            n += 1
    gaps = [b - a for a, b in zip(keys, keys[1:])] or [n]
    return n, max(gaps), sum(gaps) / len(gaps)


def reencode(map_dir: str, sub: str, gop: int) -> int:
    src = os.path.join(map_dir, sub)
    video = os.path.join(src, 'video.mp4')
    backup = backup_path(map_dir, sub)
    if not os.path.isfile(video):
        print(f'[reencode] {src}: no video.mp4')
        return 1
    if os.path.exists(backup):
        print(f'[reencode] {src}: already re-encoded (the old one is at {backup}); --restore first')
        return 1
    frames, gap, mean_gap = keyframe_gaps(video)
    print(f'[reencode] {src}: {frames} frames, keyframes every {mean_gap:.0f} (at most {gap}), '
          f'{os.path.getsize(video) / 1e6:.0f} MB')
    if gap <= gop:
        print(f'[reencode] {src}: already at most {gop} apart; nothing to do')
        return 0

    old = VideoDB(src, 'read')
    order = sorted(old.ts_to_idx.items(), key=lambda kv: kv[1])
    if [idx for _, idx in order] != list(range(len(order))) or len(order) != frames:
        print(f'[reencode] {src}: meta.json does not number the {frames} frames 0..n-1; not touched')
        return 1
    tmp = src + '.reencode-tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    video_db.GOP = gop
    new = VideoDB(tmp, 'write', fps=old.fps)
    started = time.monotonic()
    try:
        for i, (ts, _) in enumerate(order):
            frame = old.read(ts)
            if frame is None:
                raise RuntimeError(f'frame {i} (ts {ts}) did not read')
            new.write(ts, frame)
            if i and i % 1000 == 0:
                print(f'[reencode]   {i}/{frames} in {time.monotonic() - started:.0f}s')
        new.close()
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    old.close()

    out = os.path.join(tmp, 'video.mp4')
    n2, gap2, mean2 = keyframe_gaps(out)
    with open(os.path.join(tmp, 'meta.json')) as f:
        meta_new = json.load(f)['ts_to_idx']
    same_meta = {int(k): int(v) for k, v in meta_new.items()} == dict(order)
    a, b = VideoDB(src, 'read'), VideoDB(tmp, 'read')
    worst = 0.0
    for i in sorted({*range(0, frames, CHECK_EVERY), frames - 1}):
        ts = order[i][0]
        fa, fb = a.read(ts), b.read(ts)
        if fa is None or fb is None or fa.shape != fb.shape:
            worst = float('inf')
            break
        worst = max(worst, float(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).mean()))
    a.close(), b.close()
    ok = n2 == frames and same_meta and gap2 <= gop and worst <= MAX_MEAN_DIFF
    print(f'[reencode] {src}: new {n2} frames, keyframes every {mean2:.0f} (at most {gap2}), '
          f'{os.path.getsize(out) / 1e6:.0f} MB, timestamps {"same" if same_meta else "DIFFERENT"}, '
          f'worst mean difference {worst:.2f} levels, {time.monotonic() - started:.0f}s')
    if not ok:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f'[reencode] {src}: check failed; the old video stays')
        return 1
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    shutil.move(video, backup)
    os.replace(out, video)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'[reencode] {src}: done; the old video is at {backup}')
    return 0


def restore(map_dir: str, sub: str) -> int:
    src = os.path.join(map_dir, sub)
    backup = backup_path(map_dir, sub)
    if not os.path.isfile(backup):
        print(f'[reencode] {src}: no backup at {backup}; nothing to restore')
        return 0
    os.replace(backup, os.path.join(src, 'video.mp4'))
    print(f'[reencode] {src}: the old video is back')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('map_dir')
    p.add_argument('--sub', action='append', help='image DB to re-encode (default infra1_images_db, '
                   'the one relocalization reads)')
    p.add_argument('--gop', type=int, default=15)
    p.add_argument('--restore', action='store_true')
    args = p.parse_args(argv)
    subs = args.sub or ['infra1_images_db']
    rc = 0
    for sub in subs:
        rc |= restore(args.map_dir, sub) if args.restore else reencode(args.map_dir, sub, args.gop)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
