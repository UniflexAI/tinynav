"""Offline replay of planning_node on recorded frames, with and without BevNet obstacles, plus a video.

Each frame runs the planner twice with planning_node's own functions and methods: once on its obstacle map,
once on that map OR the model's obstacles (the model only adds). Needs ROS 2 on the path for the import:

  source /opt/ros/humble/setup.bash
  python -m tinynav.core.model.train.replay_planning <frames_dir> --labels <teacher_frames_dir> --video out.mp4

frames_dir: extract_frames output (left, camera depth, poses) with pred/ from predict.py.
--labels: optional teacher label folder of the same bag, to check chosen paths against the labelled obstacles.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial.transform import Rotation as R

from tinynav.core.planning_node import (PlanningNode, build_obstacle_map, generate_trajectories,
                                        score_trajectories_by_ESDF)
from tinynav.core.robot_specs import ROBOT_CONFIG

from .bev_geometry import BEV_N, BEV_X, BEV_Y, RES, heading_rotation
from .cost import OBSTACLE, PRESETS, classify

MODEL_THR = 0.5
MODEL_DECAY = 0.98   # model evidence outside the current view fades like the occupancy grid
SELF_CLEAR = 0.35    # m around the robot centre: the robot stands there, so it cannot be an obstacle


class Planner:
    """Duck-typed stand-in for PlanningNode's state, so its methods can run without ROS."""

    def __init__(self, use_model, fov_only=False, self_clear=False, hfov=None):
        self.grid_shape = (100, 100, 10)
        self.resolution = 0.05
        self.origin = np.array(self.grid_shape) * self.resolution / -2.
        self.step = 10
        self.occupancy_grid = np.zeros(self.grid_shape)
        self.obstacle_config = ROBOT_CONFIG.obstacle
        self.last_param = (0.0, 0.0)
        self.reverse_enter_threshold, self.reverse_exit_threshold = 0.30, 0.45
        self.reverse_engaged = False
        self.use_model = use_model
        self.fov_only, self.self_clear, self.hfov = fov_only, self_clear, hfov
        self.model_prob = np.zeros(self.grid_shape[:2], np.float32)

    camera_to_robot_center = PlanningNode.camera_to_robot_center
    update_occupancy_grid = PlanningNode.update_occupancy_grid
    trajectory_cost = PlanningNode.trajectory_cost
    _front_obstacle_dist = PlanningNode._front_obstacle_dist

    def cell_world_xy(self):
        i, j = np.mgrid[0:self.grid_shape[0], 0:self.grid_shape[1]]
        return self.origin[0] + (i + 0.5) * self.resolution, self.origin[1] + (j + 0.5) * self.resolution

    def shift_model(self, old_origin):
        d = np.round((self.origin[:2] - old_origin[:2]) / self.resolution).astype(int)
        if not d.any():
            return
        out = np.zeros_like(self.model_prob)
        H, W = out.shape
        src = self.model_prob[max(d[0], 0):H + min(d[0], 0), max(d[1], 0):W + min(d[1], 0)]
        out[max(-d[0], 0):max(-d[0], 0) + src.shape[0], max(-d[1], 0):max(-d[1], 0) + src.shape[1]] = src
        self.model_prob = out

    def fuse_model(self, obs_bev, T):
        """Write the model's BEV prediction into the planner's world-aligned grid."""
        X, Y = self.cell_world_xy()
        Rh = heading_rotation(T[:3, :3])
        dx, dy = X - T[0, 3], Y - T[1, 3]
        xb = Rh[0, 0] * dx + Rh[0, 1] * dy
        yb = Rh[1, 0] * dx + Rh[1, 1] * dy
        r = np.floor((BEV_X[1] - xb) / RES).astype(int)
        q = np.floor((BEV_Y[1] - yb) / RES).astype(int)
        inside = (r >= 0) & (r < BEV_N) & (q >= 0) & (q < BEV_N)
        if self.fov_only:  # outside the camera's view the model guesses; only trust what it can see
            inside &= (xb > 0.3) & (np.abs(np.arctan2(yb, xb)) < self.hfov / 2)
        self.model_prob *= MODEL_DECAY
        self.model_prob[inside] = obs_bev[r[inside], q[inside]]

    def step_frame(self, depth, T, K, obs_bev, target):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        old_origin = self.origin.copy()
        self.update_occupancy_grid(depth, T, fx, fy, cx, cy)
        self.shift_model(old_origin)
        mask = build_obstacle_map(self.occupancy_grid, self.origin, self.resolution, robot_z=T[2, 3],
                                  config=self.obstacle_config)
        base_mask = mask.copy()
        if self.use_model:
            self.fuse_model(obs_bev, T)
            extra = self.model_prob >= MODEL_THR
            if self.obstacle_config.dilation_cells > 0 and extra.any():
                extra = binary_dilation(extra, iterations=self.obstacle_config.dilation_cells)
            if self.self_clear:
                X, Y = self.cell_world_xy()
                c = self.camera_to_robot_center(T)
                extra &= (X - c[0]) ** 2 + (Y - c[1]) ** 2 > SELF_CLEAR ** 2
            mask = mask | extra
        esdf = distance_transform_edt(~mask).astype(np.float32) * self.resolution
        init_p = self.camera_to_robot_center(T)
        init_q = R.from_matrix(T[:3, :3]).as_quat()
        trajs, params = generate_trajectories(init_p, init_q)
        fl, rl, hw = ROBOT_CONFIG.footprint_from_control()
        scores, _ = score_trajectories_by_ESDF(trajs, esdf, self.origin, self.resolution, ROBOT_CONFIG.safety_radius, fl, rl, hw)
        front = self._front_obstacle_dist(T, mask)
        thr = self.reverse_exit_threshold if self.reverse_engaged else self.reverse_enter_threshold
        should_reverse = front <= thr
        self.reverse_engaged = should_reverse
        costs = [self.trajectory_cost(trajs[i], params[i], scores[i], target, should_reverse) for i in range(len(trajs))]
        k = int(np.argsort(np.array(costs), kind='stable')[0])
        self.last_param = params[k]
        stop = all(s == float('inf') for s in scores)
        return dict(traj=trajs[k], stop=stop, mask=mask, base_mask=base_mask, reverse=bool(should_reverse and params[k][0] < 0))


def carrot(poses, t, dist=2.5):
    """map_node.select_target_position on the recorded future path: the point ~dist ahead."""
    fut = poses[poses[:, 0] >= t, 1:4]
    if len(fut) < 2:
        return poses[-1, 1:4]
    seg = np.linalg.norm(np.diff(fut[:, :2], axis=0), axis=1)
    k = int(np.searchsorted(np.cumsum(seg), dist))
    return fut[min(k + 1, len(fut) - 1)]


def label_index(label_dir):
    meta = json.loads((label_dir / 'meta.json').read_text())
    info = json.loads((label_dir / 'bev' / 'info.json').read_text())
    poses = np.load(label_dir / 'poses.npy')
    out = []
    for f in meta['frames']:
        p = label_dir / 'bev' / f"{f['name']}.npz"
        if p.exists():
            j = int(np.argmin(np.abs(poses[:, 0] - f['stamp'])))
            T = np.eye(4)
            T[:3, :3] = R.from_quat(poses[j, 4:8]).as_matrix()
            T[:3, 3] = poses[j, 1:4]
            out.append((f['stamp'], p, T))
    return out, info['cam_h']


def path_hits_label(traj, lab_path, T_lab, horizon=21, radius=0.2):
    """Does the first `horizon` steps of a trajectory pass within `radius` of a labelled obstacle?"""
    z = np.load(lab_path)
    lab = classify(z['known'].astype(bool), z['span'].astype(np.float32), z['step'].astype(np.float32),
                   z['rail'].astype(bool), PRESETS['quadruped']) == OBSTACLE
    obs = binary_dilation(lab, iterations=int(radius / RES))
    Rh = heading_rotation(T_lab[:3, :3])
    d = traj[:horizon, :2] - T_lab[:2, 3]
    xb, yb = d @ Rh[0, :2], d @ Rh[1, :2]
    r = np.floor((BEV_X[1] - xb) / RES).astype(int)
    q = np.floor((BEV_Y[1] - yb) / RES).astype(int)
    ok = (r >= 0) & (r < BEV_N) & (q >= 0) & (q < BEV_N)
    return bool(obs[r[ok], q[ok]].any())


# --- video -----------------------------------------------------------------------------------------
COL_BASE, COL_MODEL, COL_PATH, COL_TGT = (40, 140, 255), (255, 220, 0), (255, 255, 255), (0, 230, 255)  # BGR
BEV_PX, BEV_SPAN = 480, 5.0


def project(points, T, K, img_shape):
    pc = (points - T[:3, 3]) @ T[:3, :3]
    ok = pc[:, 2] > 0.3
    u = K[0, 0] * pc[:, 0] / np.maximum(pc[:, 2], 1e-3) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.maximum(pc[:, 2], 1e-3) + K[1, 2]
    ok &= (u > -500) & (u < img_shape[1] + 500) & (v > -500) & (v < img_shape[0] + 500)
    return np.stack([u, v], 1)[ok].astype(np.int32)


def draw_camera(img, T, K, cam_h, res_b, res_m, future, scale=0.75):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ground = lambda p: np.column_stack([p[:, 0], p[:, 1], np.full(len(p), T[2, 3] - cam_h)])
    for pts, col, th in ((future, COL_PATH, 1), (res_b['traj'][:, :3], COL_BASE, 3), (res_m['traj'][:, :3], COL_MODEL, 3)):
        uv = project(ground(pts), T, K, img.shape)
        if len(uv) > 1:
            cv2.polylines(vis, [uv], False, col, th, cv2.LINE_AA)
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return vis


def draw_bev(planner, res_b, res_m, T, future, target):
    """Planner grid in the robot heading frame (forward up): grey = planning obstacles, magenta = added by model."""
    Rh = heading_rotation(T[:3, :3])
    px = (np.arange(BEV_PX) + 0.5) / BEV_PX * BEV_SPAN
    fwd = BEV_SPAN * 0.7 - px                    # rows: 3.5 m ahead (top) to 1.5 m behind
    lat = BEV_SPAN / 2 - px                      # cols: left to right
    F, L = np.meshgrid(fwd, lat, indexing='ij')
    wx = T[0, 3] + F * Rh[0, 0] + L * Rh[1, 0]
    wy = T[1, 3] + F * Rh[0, 1] + L * Rh[1, 1]
    i = np.floor((wx - planner.origin[0]) / planner.resolution).astype(int)
    j = np.floor((wy - planner.origin[1]) / planner.resolution).astype(int)
    ok = (i >= 0) & (i < planner.grid_shape[0]) & (j >= 0) & (j < planner.grid_shape[1])
    img = np.full((BEV_PX, BEV_PX, 3), 35, np.uint8)
    base = np.zeros((BEV_PX, BEV_PX), bool)
    full = np.zeros((BEV_PX, BEV_PX), bool)
    base[ok] = res_b['base_mask'][i[ok], j[ok]]
    full[ok] = res_m['mask'][i[ok], j[ok]]
    img[ok] = (60, 60, 60)
    img[base] = (200, 200, 200)
    img[full & ~base] = (200, 60, 220)

    def to_px(p):
        d = p[:, :2] - T[:2, 3]
        f, l = d @ Rh[0, :2], d @ Rh[1, :2]
        return np.stack([(BEV_SPAN / 2 - l) / BEV_SPAN * BEV_PX, (BEV_SPAN * 0.7 - f) / BEV_SPAN * BEV_PX], 1).astype(np.int32)
    cv2.polylines(img, [to_px(future)], False, COL_PATH, 1, cv2.LINE_AA)
    for res, col in ((res_b, COL_BASE), (res_m, COL_MODEL)):
        cv2.polylines(img, [to_px(res['traj'][:, :3])], False, col, 3, cv2.LINE_AA)
    t = to_px(target[None])[0]
    cv2.circle(img, tuple(int(v) for v in t), 6, COL_TGT, -1)
    c = to_px(T[None, :3, 3])[0]
    cv2.drawMarker(img, tuple(int(v) for v in c), (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    return img


def legend(w, t, res_b, res_m, hit_b, hit_m):
    bar = np.full((84, w, 3), 20, np.uint8)
    items = [((200, 200, 200), 'planning obstacles'), ((200, 60, 220), 'added by model'), (COL_TGT, 'target'),
             (COL_PATH, 'driven path'), (COL_BASE, 'path: planning only'), (COL_MODEL, 'path: planning + model')]
    for n, (col, txt) in enumerate(items):
        x, y = 10 + (n % 3) * (w // 3), 18 + (n // 3) * 22
        cv2.rectangle(bar, (x, y - 11), (x + 13, y + 2), col, -1)
        cv2.putText(bar, txt, (x + 19, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    status = f't={t:5.1f}s'
    for name, res, hit in (('planning only', res_b, hit_b), ('+ model', res_m, hit_m)):
        s = 'STOP (all paths blocked)' if res['stop'] else ('reverse' if res['reverse'] else '')
        if hit:
            s = (s + ' path hits labelled obstacle').strip()
        if s:
            status += f'  | {name}: {s}'
    cv2.putText(bar, status, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 200, 255), 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames_dir')
    ap.add_argument('--labels', default=None)
    ap.add_argument('--video', default=None)
    ap.add_argument('--start', type=float, default=0.0, help='seconds into the bag')
    ap.add_argument('--duration', type=float, default=1e9)
    ap.add_argument('--fov_only', action='store_true', help='model may only add obstacles inside the camera view')
    ap.add_argument('--self_clear', action='store_true', help='drop model obstacles under the robot')
    args = ap.parse_args()
    d = Path(args.frames_dir)
    meta = json.loads((d / 'meta.json').read_text())
    poses = np.load(d / 'poses.npy')
    K = np.array(meta['K'], np.float64).reshape(3, 3)
    labels, cam_h = label_index(Path(args.labels)) if args.labels else ([], 0.5)
    lab_stamps = np.array([s for s, _, _ in labels]) if labels else np.zeros(0)
    hfov = 2 * np.arctan(meta['width'] / 2 / K[0, 0])
    planners = dict(base=Planner(False), model=Planner(True, args.fov_only, args.self_clear, hfov))
    t0 = meta['frames'][0]['stamp']
    stats = dict(frames=0, labelled=0, hit_base=0, hit_model=0, stop_base=0, stop_model=0, differ=0, idle_base=0, idle_model=0)
    writer = None
    for f in meta['frames']:
        t = (f['stamp'] - t0) / 1e9
        j = int(np.argmin(np.abs(poses[:, 0] - f['stamp'])))
        T = np.eye(4)
        T[:3, :3] = R.from_quat(poses[j, 4:8]).as_matrix()
        T[:3, 3] = poses[j, 1:4]
        depth = cv2.imread(str(d / f"{f['name']}_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        pred = np.load(d / 'pred' / f"{f['name']}.npz")['obs_quadruped'].astype(np.float32)
        target = carrot(poses, f['stamp'])
        res = {k: p.step_frame(depth, T, K, pred, target) for k, p in planners.items()}  # both keep warming up
        if not (args.start <= t <= args.start + args.duration):
            continue
        stats['frames'] += 1
        stats['stop_base'] += res['base']['stop']
        stats['stop_model'] += res['model']['stop']
        stats['differ'] += np.linalg.norm(res['base']['traj'][20, :2] - res['model']['traj'][20, :2]) > 0.3
        for m in ('base', 'model'):  # chosen path moves < 0.15 m in 3 s: the robot would hesitate / turn in place
            stats[f'idle_{m}'] += not res[m]['stop'] and np.linalg.norm(res[m]['traj'][-1, :2] - res[m]['traj'][0, :2]) < 0.15
        hit = {'base': False, 'model': False}
        if len(lab_stamps):
            k = int(np.argmin(np.abs(lab_stamps - f['stamp'])))
            if abs(lab_stamps[k] - f['stamp']) < 0.15e9:
                stats['labelled'] += 1
                for m in hit:
                    hit[m] = not res[m]['stop'] and path_hits_label(res[m]['traj'], labels[k][1], labels[k][2])
                    stats[f'hit_{m}'] += hit[m]
        if args.video:
            future = poses[(poses[:, 0] >= f['stamp']) & (poses[:, 0] <= f['stamp'] + 4e9), 1:4]
            img = cv2.imread(str(d / f"{f['name']}_left.png"), cv2.IMREAD_GRAYSCALE)
            cam = draw_camera(img, T, K, cam_h, res['base'], res['model'], future)
            bev = draw_bev(planners['model'], res['base'], res['model'], T, future, target)
            h = max(cam.shape[0], bev.shape[0])
            row = np.hstack([np.pad(cam, ((0, h - cam.shape[0]), (0, 0), (0, 0))), np.pad(bev, ((0, h - bev.shape[0]), (0, 0), (0, 0)))])
            frame = np.vstack([row, legend(row.shape[1], t, res['base'], res['model'], hit['base'], hit['model'])])
            frame = frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2]
            if writer is None:
                import imageio_ffmpeg
                writer = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                                           '-pix_fmt', 'bgr24', '-s', f'{frame.shape[1]}x{frame.shape[0]}', '-r', '5', '-i', '-',
                                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26', '-movflags', '+faststart',
                                           args.video], stdin=subprocess.PIPE)
            writer.stdin.write(np.ascontiguousarray(frame).tobytes())
    if writer:
        writer.stdin.close()
        writer.wait()
    print(json.dumps(dict(bag=d.name, fov_only=args.fov_only, self_clear=args.self_clear, **{k: int(v) for k, v in stats.items()})))


if __name__ == '__main__':
    main()
