#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sensor_msgs.msg import CameraInfo, Image, Imu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay EuRoC mav0 for TinyNav perception and evaluate trajectory.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to EuRoC mav0 directory, e.g. /.../V2_01_easy/mav0",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="Playback speed multiplier (1.0 = realtime).")
    parser.add_argument("--max-stereo-frames", type=int, default=0, help="Limit stereo frames (0 = all).")
    parser.add_argument("--tail-seconds", type=float, default=3.0, help="Extra spin time after final publish.")
    parser.add_argument("--eval-topic", type=str, default="/slam/odometry_visual", help="Estimated odometry topic.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/euroc_eval"), help="Report output directory.")
    parser.add_argument(
        "--alignment-method",
        choices=("se3", "first_origin"),
        default="se3",
        help="Trajectory alignment for ATE metrics. se3 removes global yaw/frame offset; first_origin only removes translation.",
    )
    return parser.parse_args()


def load_csv_rows(csv_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            rows.append(row)
    return rows


def load_camera_sensor_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_camera_info(width: int, height: int, intrinsics: list[float], baseline: float, is_right: bool) -> CameraInfo:
    fx, fy, cx, cy = intrinsics
    msg = CameraInfo()
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    tx = -fx * baseline if is_right else 0.0
    msg.p = [fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def make_camera_info_from_rectified_P(width: int, height: int, P: np.ndarray, is_right: bool, baseline: float) -> CameraInfo:
    msg = CameraInfo()
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [
        float(P[0, 0]), 0.0, float(P[0, 2]),
        0.0, float(P[1, 1]), float(P[1, 2]),
        0.0, 0.0, 1.0,
    ]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    tx = -float(P[0, 0]) * float(baseline) if is_right else 0.0
    msg.p = [
        float(P[0, 0]), 0.0, float(P[0, 2]), tx,
        0.0, float(P[1, 1]), float(P[1, 2]), float(P[1, 3]),
        0.0, 0.0, 1.0, 0.0,
    ]
    return msg


def build_euroc_rectifier(cam0_yaml: dict, cam1_yaml: dict):
    intr0 = cam0_yaml["intrinsics"]
    intr1 = cam1_yaml["intrinsics"]
    dist0 = cam0_yaml["distortion_coefficients"]
    dist1 = cam1_yaml["distortion_coefficients"]
    width = int(cam0_yaml["resolution"][0])
    height = int(cam0_yaml["resolution"][1])

    K0 = np.array(
        [[intr0[0], 0.0, intr0[2]], [0.0, intr0[1], intr0[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    K1 = np.array(
        [[intr1[0], 0.0, intr1[2]], [0.0, intr1[1], intr1[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D0 = np.array(dist0, dtype=np.float64)
    D1 = np.array(dist1, dtype=np.float64)

    T_BS0 = np.asarray(cam0_yaml["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    T_BS1 = np.asarray(cam1_yaml["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    T_C1_C0 = np.linalg.inv(T_BS1) @ T_BS0
    R = T_C1_C0[:3, :3]
    t = T_C1_C0[:3, 3]

    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
        K0,
        D0,
        K1,
        D1,
        (width, height),
        R,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map0_x, map0_y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (width, height), cv2.CV_32FC1)
    map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    return (map0_x, map0_y, map1_x, map1_y, P0, P1)


def pose_matrix_from_t_q(t_xyz: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t_xyz
    return T


def odom_to_pose_matrix(msg: Odometry) -> np.ndarray:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float64)
    T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return T


def umeyama_rigid(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)
    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1.0
        R_opt = Vt.T @ U.T
    t_opt = dst_mean - R_opt @ src_mean
    return R_opt, t_opt


def make_T(R_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mat
    T[:3, 3] = t_vec
    return T


def se3_pose_pair_align(est_poses: list[np.ndarray], gt_poses: list[np.ndarray], T_init: np.ndarray) -> np.ndarray:
    def xi_to_T(xi: np.ndarray) -> np.ndarray:
        rot = R.from_rotvec(xi[:3]).as_matrix()
        trans = xi[3:]
        return make_T(rot, trans)

    def residuals(xi: np.ndarray) -> np.ndarray:
        T_align = xi_to_T(xi)
        out = []
        for T_est, T_gt in zip(est_poses, gt_poses):
            delta = np.linalg.inv(T_gt) @ T_align @ T_est
            out.extend(R.from_matrix(delta[:3, :3]).as_rotvec().tolist())
            out.extend(delta[:3, 3].tolist())
        return np.asarray(out, dtype=np.float64)

    xi0 = np.zeros(6, dtype=np.float64)
    xi0[:3] = R.from_matrix(T_init[:3, :3]).as_rotvec()
    xi0[3:] = T_init[:3, 3]
    opt = least_squares(residuals, xi0, loss='huber', f_scale=1.0, max_nfev=200)
    return xi_to_T(opt.x)


def interpolate_gt_pose(gt_ts: np.ndarray, gt_pos: np.ndarray, gt_rot: R, ts_query: int) -> np.ndarray | None:
    idx = np.searchsorted(gt_ts, ts_query)
    if idx == 0 or idx >= len(gt_ts):
        return None
    t0 = gt_ts[idx - 1]
    t1 = gt_ts[idx]
    alpha = float(ts_query - t0) / float(t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    pos = (1.0 - alpha) * gt_pos[idx - 1] + alpha * gt_pos[idx]
    slerp = Slerp([0.0, 1.0], R.from_quat([gt_rot[idx - 1].as_quat(), gt_rot[idx].as_quat()]))
    rot = slerp([alpha])[0]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T


class EuRoCPerceptionEvalNode(Node):
    def __init__(self, eval_topic: str):
        super().__init__("euroc_perception_eval")
        qos_sensor = QoSProfile(depth=500, reliability=ReliabilityPolicy.RELIABLE)
        self.bridge = CvBridge()
        self.left_pub = self.create_publisher(Image, "/camera/camera/infra1/image_rect_raw", qos_sensor)
        self.right_pub = self.create_publisher(Image, "/camera/camera/infra2/image_rect_raw", qos_sensor)
        self.info_left_pub = self.create_publisher(CameraInfo, "/camera/camera/infra1/camera_info", 10)
        self.info_right_pub = self.create_publisher(CameraInfo, "/camera/camera/infra2/camera_info", 10)
        self.imu_pub = self.create_publisher(Imu, "/camera/camera/imu", qos_sensor)
        self.est_sub = self.create_subscription(Odometry, eval_topic, self._est_callback, 200)
        self.est_poses: dict[int, np.ndarray] = {}

    def _est_callback(self, msg: Odometry) -> None:
        ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.est_poses[ts_ns] = odom_to_pose_matrix(msg)

    def publish_stereo(self, ts_ns: int, left_img: np.ndarray, right_img: np.ndarray, info_left: CameraInfo, info_right: CameraInfo) -> None:
        sec = int(ts_ns // 1_000_000_000)
        nsec = int(ts_ns % 1_000_000_000)
        stamp = type(info_left.header.stamp)()
        stamp.sec = sec
        stamp.nanosec = nsec

        left_msg = self.bridge.cv2_to_imgmsg(left_img, encoding="mono8")
        right_msg = self.bridge.cv2_to_imgmsg(right_img, encoding="mono8")
        left_msg.header.stamp = stamp
        right_msg.header.stamp = stamp
        left_msg.header.frame_id = "camera"
        right_msg.header.frame_id = "camera"

        info_left_msg = CameraInfo()
        info_left_msg = info_left
        info_left_msg.header.stamp = stamp
        info_left_msg.header.frame_id = "camera"

        info_right_msg = CameraInfo()
        info_right_msg = info_right
        info_right_msg.header.stamp = stamp
        info_right_msg.header.frame_id = "camera"

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)
        self.info_left_pub.publish(info_left_msg)
        self.info_right_pub.publish(info_right_msg)

    def publish_imu(self, ts_ns: int, gyro_xyz: np.ndarray, accel_xyz: np.ndarray) -> None:
        msg = Imu()
        msg.header.stamp.sec = int(ts_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(ts_ns % 1_000_000_000)
        msg.header.frame_id = "camera"
        msg.angular_velocity.x = float(gyro_xyz[0])
        msg.angular_velocity.y = float(gyro_xyz[1])
        msg.angular_velocity.z = float(gyro_xyz[2])
        msg.linear_acceleration.x = float(accel_xyz[0])
        msg.linear_acceleration.y = float(accel_xyz[1])
        msg.linear_acceleration.z = float(accel_xyz[2])
        self.imu_pub.publish(msg)


def run_replay_and_collect(node: EuRoCPerceptionEvalNode, args: argparse.Namespace) -> dict[int, np.ndarray]:
    root = args.dataset_root
    cam0_csv = root / "cam0" / "data.csv"
    cam1_csv = root / "cam1" / "data.csv"
    imu_csv = root / "imu0" / "data.csv"
    cam0_yaml = load_camera_sensor_yaml(root / "cam0" / "sensor.yaml")
    cam1_yaml = load_camera_sensor_yaml(root / "cam1" / "sensor.yaml")
    imu_yaml = load_camera_sensor_yaml(root / "imu0" / "sensor.yaml")

    rows0 = load_csv_rows(cam0_csv)
    rows1 = load_csv_rows(cam1_csv)
    rows_imu = load_csv_rows(imu_csv)

    right_by_ts = {int(r[0]): r[1] for r in rows1}
    stereo_pairs: list[tuple[int, Path, Path]] = []
    for r in rows0:
        ts = int(r[0])
        if ts in right_by_ts:
            left_path = root / "cam0" / "data" / r[1]
            right_path = root / "cam1" / "data" / right_by_ts[ts]
            if left_path.exists() and right_path.exists():
                stereo_pairs.append((ts, left_path, right_path))
    stereo_pairs.sort(key=lambda x: x[0])
    if args.max_stereo_frames > 0:
        stereo_pairs = stereo_pairs[: args.max_stereo_frames]
    if not stereo_pairs:
        raise RuntimeError("No stereo pairs found.")

    imu_data = []
    T_B_C0 = np.asarray(cam0_yaml["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    T_B_I = np.asarray(imu_yaml["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    R_B_C0 = T_B_C0[:3, :3]
    R_B_I = T_B_I[:3, :3]
    R_C0_I = R_B_C0.T @ R_B_I
    for r in rows_imu:
        ts = int(r[0])
        gyro_imu = np.array([float(r[1]), float(r[2]), float(r[3])], dtype=np.float64)
        accel_imu = np.array([float(r[4]), float(r[5]), float(r[6])], dtype=np.float64)
        gyro = R_C0_I @ gyro_imu
        accel = R_C0_I @ accel_imu
        imu_data.append((ts, gyro, accel))
    imu_data.sort(key=lambda x: x[0])

    width = int(cam0_yaml["resolution"][0])
    height = int(cam0_yaml["resolution"][1])
    map0_x, map0_y, map1_x, map1_y, P0, P1 = build_euroc_rectifier(cam0_yaml, cam1_yaml)
    baseline = abs(float(P1[0, 3]) / float(P1[0, 0]))
    info_left = make_camera_info_from_rectified_P(width, height, P0, is_right=False, baseline=baseline)
    info_right = make_camera_info_from_rectified_P(width, height, P1, is_right=True, baseline=baseline)

    events = []
    for ts, g, a in imu_data:
        events.append(("imu", ts, g, a))
    for ts, lp, rp in stereo_pairs:
        events.append(("stereo", ts, lp, rp))
    events.sort(key=lambda x: x[1])

    first_ts = events[0][1]
    t0_wall = time.monotonic()
    for event in events:
        kind = event[0]
        ts_ns = event[1]
        target_elapsed = (ts_ns - first_ts) / 1e9 / args.rate
        while True:
            now_elapsed = time.monotonic() - t0_wall
            remaining = target_elapsed - now_elapsed
            if remaining <= 0:
                break
            rclpy.spin_once(node, timeout_sec=min(remaining, 0.002))
        if kind == "imu":
            node.publish_imu(ts_ns, event[2], event[3])
        else:
            left_img = cv2.imread(str(event[2]), cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(str(event[3]), cv2.IMREAD_GRAYSCALE)
            if left_img is None or right_img is None:
                continue
            left_rect = cv2.remap(left_img, map0_x, map0_y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, map1_x, map1_y, cv2.INTER_LINEAR)
            node.publish_stereo(ts_ns, left_rect, right_rect, info_left, info_right)
        rclpy.spin_once(node, timeout_sec=0.0)

    end_wait = time.monotonic() + args.tail_seconds
    while time.monotonic() < end_wait:
        rclpy.spin_once(node, timeout_sec=0.01)
    return node.est_poses


def evaluate_and_report(args: argparse.Namespace, est_poses: dict[int, np.ndarray]) -> Path:
    gt_csv = args.dataset_root / "state_groundtruth_estimate0" / "data.csv"
    cam0_yaml = load_camera_sensor_yaml(args.dataset_root / "cam0" / "sensor.yaml")
    T_B_C0 = np.asarray(cam0_yaml["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    gt_rows = load_csv_rows(gt_csv)
    gt_ts = np.array([int(r[0]) for r in gt_rows], dtype=np.int64)
    gt_pos = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in gt_rows], dtype=np.float64)
    gt_quat_wxyz = np.array([[float(r[4]), float(r[5]), float(r[6]), float(r[7])] for r in gt_rows], dtype=np.float64)
    gt_rots = R.from_quat(np.stack([gt_quat_wxyz[:, 1], gt_quat_wxyz[:, 2], gt_quat_wxyz[:, 3], gt_quat_wxyz[:, 0]], axis=1))

    est_ts_sorted = sorted(est_poses.keys())
    gt_pose_pairs = []
    est_pose_pairs = []
    est_points = []
    matched_ts = []
    matched_gt_t0 = []
    matched_gt_t1 = []
    matched_alpha = []
    for ts in est_ts_sorted:
        idx = np.searchsorted(gt_ts, ts)
        if idx == 0 or idx >= len(gt_ts):
            continue
        t0 = int(gt_ts[idx - 1])
        t1 = int(gt_ts[idx])
        alpha = float(np.clip((ts - t0) / max(t1 - t0, 1), 0.0, 1.0))

        gt_pose_body = interpolate_gt_pose(gt_ts, gt_pos, gt_rots, ts)
        if gt_pose_body is None:
            continue
        gt_pose_cam0 = gt_pose_body @ T_B_C0
        gt_pose_pairs.append(gt_pose_cam0)
        est_pose_pairs.append(est_poses[ts])
        est_points.append(est_poses[ts][:3, 3])
        matched_ts.append(ts)
        matched_gt_t0.append(t0)
        matched_gt_t1.append(t1)
        matched_alpha.append(alpha)
    if len(est_points) < 10:
        raise RuntimeError(f"Too few matched poses for evaluation: {len(est_points)}")

    gt_points = np.array([g[:3, 3] for g in gt_pose_pairs], dtype=np.float64)
    est_points = np.array(est_points, dtype=np.float64)

    gt_origin = gt_points[0]
    est_origin = est_points[0]
    gt_points_rel = gt_points - gt_origin

    if args.alignment_method == "first_origin":
        est_points_aligned = est_points - est_origin
        alignment_transform = np.eye(4, dtype=np.float64)
    else:
        R_init, t_init = umeyama_rigid(est_points, gt_points)
        T_init = make_T(R_init, t_init)
        alignment_transform = se3_pose_pair_align(est_pose_pairs, gt_pose_pairs, T_init)
        est_points_abs_aligned = np.array(
            [(alignment_transform @ pose)[:3, 3] for pose in est_pose_pairs],
            dtype=np.float64,
        )
        est_points_aligned = est_points_abs_aligned - gt_origin

    errors = np.linalg.norm(est_points_aligned - gt_points_rel, axis=1)

    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    max_err = float(np.max(errors))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_root": str(args.dataset_root),
        "estimated_pose_count": len(est_poses),
        "matched_pose_count": int(len(errors)),
        "ate_rmse_m": rmse,
        "ate_mean_m": mean_err,
        "ate_median_m": median_err,
        "ate_max_m": max_err,
        "alignment_method": args.alignment_method,
        "alignment_transform_est_to_gt": alignment_transform.tolist(),
        "gt_origin": gt_origin.tolist(),
        "est_origin": est_origin.tolist(),
    }
    report_path = out_dir / "trajectory_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    matched_ts_np = np.array(matched_ts, dtype=np.int64)
    rel_t = (matched_ts_np - matched_ts_np[0]) / 1e9
    np.savetxt(
        out_dir / "trajectory_errors.csv",
        np.column_stack([matched_ts_np, rel_t, errors]),
        delimiter=",",
        header="timestamp_ns,relative_time_s,position_error_m",
        comments="",
    )

    np.savetxt(
        out_dir / "trajectory_timestamp_pairs.csv",
        np.column_stack([
            np.array(matched_ts, dtype=np.int64),
            np.array(matched_gt_t0, dtype=np.int64),
            np.array(matched_gt_t1, dtype=np.int64),
            np.array(matched_alpha, dtype=np.float64),
        ]),
        delimiter=",",
        header="est_timestamp_ns,gt_t0_ns,gt_t1_ns,interp_alpha",
        comments="",
    )

    frame_idx = np.arange(len(matched_ts_np), dtype=np.int64)
    np.savetxt(
        out_dir / "trajectory_pose_pairs.csv",
        np.column_stack([
            frame_idx,
            matched_ts_np,
            rel_t,
            gt_points_rel[:, 0], gt_points_rel[:, 1], gt_points_rel[:, 2],
            est_points[:, 0], est_points[:, 1], est_points[:, 2],
            est_points_aligned[:, 0], est_points_aligned[:, 1], est_points_aligned[:, 2],
            errors,
        ]),
        delimiter=",",
        header=(
            "frame_idx,timestamp_ns,relative_time_s,"
            "gt_x,gt_y,gt_z,"
            "est_x,est_y,est_z,"
            "est_aligned_x,est_aligned_y,est_aligned_z,"
            "position_error_m"
        ),
        comments="",
    )

    preview_rows = min(300, len(errors))
    table_rows = []
    for i in range(preview_rows):
        table_rows.append(
            f"<tr><td>{i}</td><td>{int(matched_ts_np[i])}</td><td>{rel_t[i]:.3f}</td>"
            f"<td>{gt_points_rel[i,0]:.4f}</td><td>{gt_points_rel[i,1]:.4f}</td><td>{gt_points_rel[i,2]:.4f}</td>"
            f"<td>{est_points_aligned[i,0]:.4f}</td><td>{est_points_aligned[i,1]:.4f}</td><td>{est_points_aligned[i,2]:.4f}</td>"
            f"<td>{errors[i]:.4f}</td></tr>"
        )
    table_html = "\n".join(table_rows)

    gt_js = json.dumps(gt_points_rel.tolist())
    est_js = json.dumps(est_points_aligned.tolist())
    err_t_js = json.dumps(rel_t.tolist())
    err_v_js = json.dumps(errors.tolist())

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EuRoC Perception Trajectory Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; background: #fafafa; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    #traj3d {{ width: 100%; height: 560px; border: 1px solid #ddd; border-radius: 6px; }}
    #err2d {{ width: 100%; height: 260px; border: 1px solid #ddd; border-radius: 6px; }}
    a {{ color: #0a58ca; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>EuRoC Perception Trajectory Report</h1>
  <div class="card">
    <h2>ATE Metrics</h2>
    <table>
      <tr><th>Dataset</th><td><code>{args.dataset_root}</code></td></tr>
      <tr><th>Estimated Pose Count</th><td>{len(est_poses)}</td></tr>
      <tr><th>Matched Pose Count</th><td>{int(len(errors))}</td></tr>
      <tr><th>Alignment Method</th><td>{args.alignment_method}</td></tr>
      <tr><th>ATE RMSE (m)</th><td>{rmse:.6f}</td></tr>
      <tr><th>ATE Mean (m)</th><td>{mean_err:.6f}</td></tr>
      <tr><th>ATE Median (m)</th><td>{median_err:.6f}</td></tr>
      <tr><th>ATE Max (m)</th><td>{max_err:.6f}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Trajectory 3D (Three.js)</h2>
    <div id="traj3d"></div>
  </div>

  <div class="card">
    <h2>ATE Over Time</h2>
    <canvas id="err2d"></canvas>
  </div>

  <div class="card">
    <h2>Per-Frame Pose Pairs (Preview)</h2>
    <p>Showing first {preview_rows} rows. Full data is in <code>trajectory_pose_pairs.csv</code>.</p>
    <div style="overflow:auto; max-height: 480px; border: 1px solid #ddd;">
      <table>
        <tr>
          <th>frame_idx</th><th>timestamp_ns</th><th>t[s]</th>
          <th>gt_x</th><th>gt_y</th><th>gt_z</th>
          <th>est_aligned_x</th><th>est_aligned_y</th><th>est_aligned_z</th>
          <th>err[m]</th>
        </tr>
        {table_html}
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Raw Outputs</h2>
    <ul>
      <li><a href="trajectory_report.json">trajectory_report.json</a></li>
      <li><a href="trajectory_errors.csv">trajectory_errors.csv</a></li>
      <li><a href="trajectory_pose_pairs.csv">trajectory_pose_pairs.csv</a></li>
      <li><a href="trajectory_timestamp_pairs.csv">trajectory_timestamp_pairs.csv</a></li>
    </ul>
  </div>

  <script>
    const gtPoints = {gt_js};
    const estPoints = {est_js};
    const errT = {err_t_js};
    const errV = {err_v_js};

    function drawTrajectoryCanvas() {{
      const root = document.getElementById('traj3d');
      root.innerHTML = '';
      const canvas = document.createElement('canvas');
      canvas.style.width = '100%';
      canvas.style.height = '560px';
      root.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const w = root.clientWidth;
      const h = 560;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.scale(dpr, dpr);

      const pts = gtPoints.concat(estPoints);
      let minX=Infinity,minY=Infinity,minZ=Infinity,maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
      for (const p of pts) {{
        minX=Math.min(minX,p[0]); minY=Math.min(minY,p[1]); minZ=Math.min(minZ,p[2]);
        maxX=Math.max(maxX,p[0]); maxY=Math.max(maxY,p[1]); maxZ=Math.max(maxZ,p[2]);
      }}
      const cx=(minX+maxX)/2, cy=(minY+maxY)/2, cz=(minZ+maxZ)/2;
      const span=Math.max(maxX-minX,maxY-minY,maxZ-minZ,1e-6);
      const yaw = -0.9 + 0.35 * Math.sin(Date.now() * 0.001);
      const pitch = 0.55 + 0.15 * Math.cos(Date.now() * 0.0007);
      const sy=Math.sin(yaw), cyaw=Math.cos(yaw);
      const sp=Math.sin(pitch), cp=Math.cos(pitch);

      function project(p) {{
        let x=p[0]-cx, y=p[1]-cy, z=p[2]-cz;
        const x1 = cyaw*x + sy*z;
        const z1 = -sy*x + cyaw*z;
        const y2 = cp*y - sp*z1;
        const z2 = sp*y + cp*z1;
        const f = 1.8 / (z2 / (span + 1e-9) + 3.2);
        return [x1 * f, y2 * f];
      }}

      const proj = pts.map(project);
      let minU=Infinity,minV=Infinity,maxU=-Infinity,maxV=-Infinity;
      for (const q of proj) {{
        minU=Math.min(minU,q[0]); minV=Math.min(minV,q[1]);
        maxU=Math.max(maxU,q[0]); maxV=Math.max(maxV,q[1]);
      }}
      const pad=28;
      const sx=(w-2*pad)/Math.max(maxU-minU,1e-9);
      const sy2=(h-2*pad)/Math.max(maxV-minV,1e-9);
      const sc=Math.min(sx,sy2);
      const toPix=(q)=>[pad + (q[0]-minU)*sc, h-pad-(q[1]-minV)*sc];

      ctx.fillStyle='#ffffff';
      ctx.fillRect(0,0,w,h);
      ctx.strokeStyle='#ddd';
      ctx.strokeRect(0.5,0.5,w-1,h-1);

      function drawLine(points, color) {{
        ctx.beginPath();
        for (let i=0;i<points.length;i++) {{
          const uv = toPix(project(points[i]));
          if (i===0) ctx.moveTo(uv[0],uv[1]); else ctx.lineTo(uv[0],uv[1]);
        }}
        ctx.strokeStyle=color;
        ctx.lineWidth=2;
        ctx.stroke();
      }}

      drawLine(gtPoints, '#0066ff');
      drawLine(estPoints, '#ff3b30');

      ctx.fillStyle='#111';
      ctx.font='14px Arial';
      ctx.fillText('Trajectory (3D projected, offline HTML)', 12, 22);
      ctx.fillStyle='#0066ff'; ctx.fillRect(14, 34, 18, 3); ctx.fillStyle='#111'; ctx.fillText('GT', 38, 39);
      ctx.fillStyle='#ff3b30'; ctx.fillRect(86, 34, 18, 3); ctx.fillStyle='#111'; ctx.fillText('Estimated (aligned)', 110, 39);
    }}

    function buildErr2D() {{
      const canvas = document.getElementById('err2d');
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      const padL=48, padR=14, padT=12, padB=28;
      const pw=w-padL-padR, ph=h-padT-padB;
      const tMin=errT[0], tMax=errT[errT.length-1];
      const eMin=0.0, eMax=Math.max(...errV) * 1.05 + 1e-9;
      const tx = (t) => padL + (t - tMin) / Math.max(tMax - tMin, 1e-9) * pw;
      const ty = (e) => padT + (1.0 - (e - eMin) / Math.max(eMax - eMin, 1e-9)) * ph;

      ctx.strokeStyle = '#ddd';
      ctx.strokeRect(padL, padT, pw, ph);

      ctx.beginPath();
      for (let i=0;i<errT.length;i++) {{
        const x=tx(errT[i]), y=ty(errV[i]);
        if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }}
      ctx.strokeStyle = '#ff3b30';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle='#333';
      ctx.font='12px Arial';
      ctx.fillText('time [s]', w/2-20, h-6);
      ctx.save();
      ctx.translate(12, h/2+24);
      ctx.rotate(-Math.PI/2);
      ctx.fillText('position error [m]', 0,0);
      ctx.restore();
    }}

    drawTrajectoryCanvas();
    buildErr2D();
    setInterval(drawTrajectoryCanvas, 50);
    window.addEventListener('resize', () => {{
      drawTrajectoryCanvas();
      buildErr2D();
    }});
  </script>
</body>
</html>
"""
    html_path = out_dir / "trajectory_report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path



def main() -> int:
    args = parse_args()
    rclpy.init()
    node = EuRoCPerceptionEvalNode(args.eval_topic)
    try:
        est_poses = run_replay_and_collect(node, args)
        report_path = evaluate_and_report(args, est_poses)
        print(f"Saved trajectory report: {report_path}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
