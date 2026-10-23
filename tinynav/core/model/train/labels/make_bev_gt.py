"""Per-frame BEV labels from teacher depth aggregated over neighbouring frames (sliding time window).

For frame t, points from frames within +-WINDOW_S (and close in 3D) are moved into t's gravity-aligned
heading frame (x forward, y left, z up, origin on the ground under the camera) and voxelized.
Layers per 5 cm cell (all heights relative to the ground under the camera at t):
  known    : occupied voxels seen by >= MIN_FRAMES frames in the robot's height band
  ground   : lowest occupied z in the band (support surface)
  span     : highest - lowest occupied z in the band (walls, railings, boxes)
  step     : max |ground - neighbour ground| within STEP_RADIUS cells (curbs, stairs, drop edges)
  rail     : known cell with voxels from railing/fence pixels
  traversed: the camera passed over this cell within the window
  visible  : cell observed by frame t alone (seen now vs. filled in from other frames)
"""
import argparse
import json
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion, maximum_filter
from scipy.spatial.transform import Rotation as R

from ..bev_geometry import BEV_N, BEV_X, BEV_Y, RES

RAIL_WORDS = ('railing', 'rail', 'bannister', 'balustrade', 'fence', 'handrail')
ZMAX = 4.5            # m, teacher depth beyond this is too noisy to aggregate
PIX_STEP = 1
MIN_FRAMES = 3
BAND = (-1.5, 1.2)    # m relative to ground under the camera
STEP_RADIUS = 2       # cells
WINDOW_S = 30.0
FREE_DS = 4          # depth downsampling for free-space tests
MAX_DIST = 10.0       # m, neighbour camera must be this close
NZ = int(round((BAND[1] - BAND[0]) / RES))


def load_frames(d, need_teacher=True):
    meta = json.loads((d / 'meta.json').read_text())
    poses = np.load(d / 'poses.npy')
    frames = []
    for f in meta['frames']:
        if (need_teacher and not (d / f"{f['name']}_fsdepth.npy").exists()) or not len(poses):
            continue
        j = int(np.argmin(np.abs(poses[:, 0] - f['stamp'])))
        if abs(poses[j, 0] - f['stamp']) > 30e6:
            continue
        T = np.eye(4)
        T[:3, :3] = R.from_quat(poses[j, 4:8]).as_matrix()
        T[:3, 3] = poses[j, 1:4]
        frames.append(dict(f, T=T))
    return meta, poses, frames


def frame_points(d, f, K, rail_ids):
    """World points of one frame, voxel-downsampled; returns (N,3) float32 and rail flags."""
    depth = np.load(d / f"{f['name']}_fsdepth.npy").astype(np.float32)[::PIX_STEP, ::PIX_STEP]
    seg = cv2.imread(str(d / f"{f['name']}_seg.png"), cv2.IMREAD_UNCHANGED)[::PIX_STEP, ::PIX_STEP]
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]] * PIX_STEP
    ok = (depth > 0.2) & (depth < ZMAX)
    z = depth[ok]
    pc = np.stack([(u[ok] - K[2]) / K[0] * z, (v[ok] - K[5]) / K[4] * z, z], 1)
    pw = pc @ f['T'][:3, :3].T + f['T'][:3, 3]
    rail = np.isin(seg[ok], rail_ids)
    _, idx = np.unique(np.floor(pw / RES).astype(np.int64), axis=0, return_index=True)
    full = np.load(d / f"{f['name']}_fsdepth.npy").astype(np.float32)
    return pw[idx].astype(np.float32), rail[idx], full[::FREE_DS, ::FREE_DS]


def camera_points(d, f, K):
    """World points from the camera's own depth (for frames without teacher depth)."""
    depth = cv2.imread(str(d / f"{f['name']}_depth.png"), cv2.IMREAD_UNCHANGED)[::4, ::4].astype(np.float32) / 1000.0
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]] * 4
    ok = (depth > 0.2) & (depth < ZMAX)
    z = depth[ok]
    pc = np.stack([(u[ok] - K[2]) / K[0] * z, (v[ok] - K[5]) / K[4] * z, z], 1)
    return (pc @ f['T'][:3, :3].T + f['T'][:3, 3]).astype(np.float32)


def heading_frame(T, cam_h):
    """World->BEV transform: rotation (3x3) and origin (ground under camera)."""
    fwd = T[:3, 2].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd) + 1e-9
    left = np.array([-fwd[1], fwd[0], 0.0])
    Rb = np.stack([fwd, left, [0, 0, 1.0]])
    origin = T[:3, 3] - np.array([0, 0, cam_h])
    return Rb, origin


def to_grid(pb):
    res_b = (BEV_X[1] - BEV_X[0]) / BEV_N
    r = np.floor((BEV_X[1] - pb[:, 0]) / res_b).astype(np.int64)   # row 0 = far forward
    q = np.floor((BEV_Y[1] - pb[:, 1]) / res_b).astype(np.int64)   # col 0 = far left
    k = np.floor((pb[:, 2] - BAND[0]) / RES).astype(np.int64)
    ok = (r >= 0) & (r < BEV_N) & (q >= 0) & (q < BEV_N) & (k >= 0) & (k < NZ)
    return r, q, k, ok


def estimate_cam_height(frames, pts):
    hs = []
    for f, (pw, *_) in zip(frames, pts):
        Rb, _ = heading_frame(f['T'], 0.0)
        rel = (pw - f['T'][:3, 3]) @ Rb.T
        near = (rel[:, 0] > 0.5) & (rel[:, 0] < 1.5) & (np.abs(rel[:, 1]) < 0.3)
        if near.sum() > 30:
            hs.append(-np.percentile(rel[near, 2], 10))
    return float(np.median(hs)) if hs else 0.5


def nan_median3(a):
    p = np.pad(a, 1, constant_values=np.nan)
    stack = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-nan windows
        m = np.nanmedian(stack, 0)
    return np.where(np.isnan(a), np.nan, m)


def seen_through(occv, occ, near, Rb, origin, K):
    """Voxels that more frames saw *through* (depth beyond them) than saw them: moving objects, noise."""
    r, q, k = np.nonzero(occv)
    res_b = (BEV_X[1] - BEV_X[0]) / BEV_N
    pb = np.stack([BEV_X[1] - (r + 0.5) * res_b, BEV_Y[1] - (q + 0.5) * res_b, BAND[0] + (k + 0.5) * RES], 1)
    pw = pb @ Rb + origin
    free = np.zeros(len(r), np.int32)
    fx, fy, cx, cy = K[0] / FREE_DS, K[4] / FREE_DS, K[2] / FREE_DS, K[5] / FREE_DS
    for g, _, _, dep in near:
        Rg, tg = g['T'][:3, :3], g['T'][:3, 3]
        pc = (pw - tg) @ Rg
        z = pc[:, 2]
        u = np.round(fx * pc[:, 0] / np.maximum(z, 1e-3) + cx).astype(np.int64)
        v = np.round(fy * pc[:, 1] / np.maximum(z, 1e-3) + cy).astype(np.int64)
        ok = (z > 0.3) & (z < ZMAX) & (u >= 0) & (u < dep.shape[1]) & (v >= 0) & (v < dep.shape[0])
        dm = np.zeros(len(z), np.float32)
        dm[ok] = dep[v[ok], u[ok]]
        free += ok & (dm > z + 0.15 + 0.02 * z * z)
    out = np.zeros_like(occv)
    out[r, q, k] = free > occ[r, q, k]
    return out


def label_for(f, window, poses_xyz, cam_h, K, own_points=None):
    Rb, origin = heading_frame(f['T'], cam_h)
    occ = np.zeros((BEV_N, BEV_N, NZ), np.uint8)       # frames seeing each voxel (saturating)
    railv = np.zeros((BEV_N, BEV_N, NZ), bool)
    visible = np.zeros((BEV_N, BEV_N), bool)
    near = [(g, pw, rail, dep) for g, (pw, rail, dep) in window
            if np.linalg.norm(g['T'][:3, 3] - f['T'][:3, 3]) <= MAX_DIST]
    for g, pw, rail, _ in near:
        r, q, k, ok = to_grid((pw - origin) @ Rb.T)
        r, q, k, rl = r[ok], q[ok], k[ok], rail[ok]
        occ[r, q, k] = np.minimum(occ[r, q, k], 254) + 1
        railv[r[rl], q[rl], k[rl]] = True
        if g is f:
            visible[r, q] = True
    if own_points is not None:  # target frame has no teacher depth: 'visible' from its camera depth
        r, q, k, ok = to_grid((own_points - origin) @ Rb.T)
        visible[r[ok], q[ok]] = True
    occv = occ >= MIN_FRAMES
    occv &= ~seen_through(occv, occ, near, Rb, origin, K)
    known = occv.any(2)
    zc = BAND[0] + (np.arange(NZ) + 0.5) * RES
    ground = np.where(known, zc[np.argmax(occv, 2)], np.nan)
    top = np.where(known, zc[NZ - 1 - np.argmax(occv[..., ::-1], 2)], np.nan)
    span = np.nan_to_num(top - ground)
    gs = nan_median3(ground)  # single low/high outlier voxels must not create steps
    size = 2 * STEP_RADIUS + 1
    g_hi = grey_dilation(np.where(known, gs, -1e3), size=size)
    g_lo = grey_erosion(np.where(known, gs, 1e3), size=size)
    step = np.where(known, np.maximum(g_hi - gs, gs - g_lo), 0)
    rail = (railv & occv).any(2)

    traversed = np.zeros((BEV_N, BEV_N), np.uint8)
    pb = (poses_xyz - origin) @ Rb.T
    pb[:, 2] -= cam_h                                 # pose ground height rel. current ground
    pb = pb[(pb[:, 2] > BAND[0]) & (pb[:, 2] < 0.5)]  # drop passes on other floors
    pb[:, 2] = 0
    r, q, _, ok = to_grid(pb)
    traversed[r[ok], q[ok]] = 1
    traversed = maximum_filter(traversed, size=int(0.2 / RES) * 2 + 1).astype(bool)
    return dict(known=known, ground=np.nan_to_num(ground).astype(np.float16), span=span.astype(np.float16),
                step=step.astype(np.float16), rail=rail, traversed=traversed, visible=visible)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames_dir')
    ap.add_argument('--out', default='bev')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--cam_h', type=float, default=0.0, help='camera height above ground; estimated when 0 (unreliable on stairs)')
    ap.add_argument('--targets', default=None,
                    help='folder of extra frames (left + camera depth, same bag) to label from the teacher frames in frames_dir')
    args = ap.parse_args()
    d = Path(args.frames_dir)
    meta, poses, frames = load_frames(d)
    if args.max_frames:
        frames = frames[:args.max_frames]
    td = Path(args.targets) if args.targets else d
    targets = load_frames(td, need_teacher=False)[2] if args.targets else frames
    labels = json.loads((d / 'seg_labels.json').read_text())
    rail_ids = [int(k) for k, v in labels.items() if any(w in v.lower() for w in RAIL_WORDS)]
    K = meta['K']
    out = td / args.out
    out.mkdir(exist_ok=True)

    sample = frames[:: max(1, len(frames) // 50)]
    cam_h = args.cam_h or estimate_cam_height(sample, [frame_points(d, f, K, rail_ids) for f in sample])
    # frames observing a cell ~ teacher rate * depth range / speed; fast, sparsely sampled bags lose most of
    # the ground with the default MIN_FRAMES, so they accept two views
    global MIN_FRAMES
    dur = (frames[-1]['stamp'] - frames[0]['stamp']) / 1e9 if len(frames) > 1 else 1.0
    speed = np.linalg.norm(np.diff(poses[:, 1:3], axis=0), axis=1).sum() / max((poses[-1, 0] - poses[0, 0]) / 1e9, 1e-3)
    views = len(frames) / dur * ZMAX / max(speed, 0.05)
    MIN_FRAMES = 3 if views >= 6 else 2
    window = deque()  # (frame, points) sorted by time
    nxt = 0
    for f in targets:
        t = f['stamp']
        while nxt < len(frames) and frames[nxt]['stamp'] <= t + WINDOW_S * 1e9:
            window.append((frames[nxt], frame_points(d, frames[nxt], K, rail_ids)))
            nxt += 1
        while window and window[0][0]['stamp'] < t - WINDOW_S * 1e9:
            window.popleft()
        near = np.abs(poses[:, 0] - t) < WINDOW_S * 1e9
        own = camera_points(td, f, K) if args.targets else None
        lab = label_for(f, window, poses[near, 1:4], cam_h, K, own)
        np.savez_compressed(out / f"{f['name']}.npz", **lab)
    (out / 'info.json').write_text(json.dumps(dict(cam_h=cam_h, res=RES, bev_x=BEV_X, bev_y=BEV_Y, n=BEV_N,
                                                  window_s=WINDOW_S, frames=len(targets), min_frames=MIN_FRAMES)))
    print(f'{td.name}: {len(targets)} labels, cam_h={cam_h:.2f} m, views~{views:.1f} -> MIN_FRAMES={MIN_FRAMES}')


if __name__ == '__main__':
    main()
