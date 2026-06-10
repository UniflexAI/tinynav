import argparse
import csv
import json
import math
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from tinynav.core.math_utils import pose_msg2np


class CmdVelDebugRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir = self.output_dir / "trajectories"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._trajectory_count = 0

        self._odom_file = (self.output_dir / "odom.csv").open("w", newline="", buffering=1)
        self._cmd_file = (self.output_dir / "cmd_vel.csv").open("w", newline="", buffering=1)
        self._traj_file = (self.output_dir / "trajectories.jsonl").open("w", buffering=1)

        self._odom_writer = csv.DictWriter(
            self._odom_file,
            fieldnames=[
                "wall_time_ns",
                "odom_stamp",
                "camera_x_raw",
                "camera_y_raw",
                "camera_z_raw",
                "robot_x_raw",
                "robot_y_raw",
                "robot_z_raw",
                "robot_yaw_raw",
                "camera_x_filtered",
                "camera_y_filtered",
                "camera_z_filtered",
                "robot_x_filtered",
                "robot_y_filtered",
                "robot_z_filtered",
                "robot_yaw_filtered",
            ],
        )
        self._cmd_writer = csv.DictWriter(
            self._cmd_file,
            fieldnames=[
                "wall_time_ns",
                "odom_stamp",
                "cmd_vx",
                "cmd_vyaw",
                "reason",
                "detail",
                "target_idx",
                "target_x",
                "target_y",
                "target_yaw",
                "v_ref",
                "w_ref",
                "tx",
                "ty",
                "heading_err",
                "path_len",
                "path_start_t",
                "path_end_t",
                "query_t",
            ],
        )
        self._odom_writer.writeheader()
        self._cmd_writer.writeheader()

    @staticmethod
    def _yaw_from_rotation(rotation):
        forward = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return math.atan2(float(forward[1]), float(forward[0]))

    def record_odom(self, wall_time_ns, odom_stamp, raw_position, raw_rotation, filtered_position, filtered_rotation):
        camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        raw_robot = raw_position - raw_rotation @ camera_offset
        filtered_robot = filtered_position - filtered_rotation @ camera_offset
        self._odom_writer.writerow(
            {
                "wall_time_ns": int(wall_time_ns),
                "odom_stamp": float(odom_stamp),
                "camera_x_raw": float(raw_position[0]),
                "camera_y_raw": float(raw_position[1]),
                "camera_z_raw": float(raw_position[2]),
                "robot_x_raw": float(raw_robot[0]),
                "robot_y_raw": float(raw_robot[1]),
                "robot_z_raw": float(raw_robot[2]),
                "robot_yaw_raw": self._yaw_from_rotation(raw_rotation),
                "camera_x_filtered": float(filtered_position[0]),
                "camera_y_filtered": float(filtered_position[1]),
                "camera_z_filtered": float(filtered_position[2]),
                "robot_x_filtered": float(filtered_robot[0]),
                "robot_y_filtered": float(filtered_robot[1]),
                "robot_z_filtered": float(filtered_robot[2]),
                "robot_yaw_filtered": self._yaw_from_rotation(filtered_rotation),
            }
        )

    def record_cmd(self, wall_time_ns, odom_stamp, vx, vyaw, reason="", detail=None, target_idx=-1, target=None, tx=np.nan, ty=np.nan, heading_err=np.nan, path_ref=None, query_t=np.nan):
        row = {
            "wall_time_ns": int(wall_time_ns),
            "odom_stamp": "" if odom_stamp is None else float(odom_stamp),
            "cmd_vx": float(vx),
            "cmd_vyaw": float(vyaw),
            "reason": reason,
            "detail": "" if detail is None else str(detail),
            "target_idx": int(target_idx),
            "target_x": "",
            "target_y": "",
            "target_yaw": "",
            "v_ref": "",
            "w_ref": "",
            "tx": "" if math.isnan(tx) else float(tx),
            "ty": "" if math.isnan(ty) else float(ty),
            "heading_err": "" if math.isnan(heading_err) else float(heading_err),
            "path_len": 0 if path_ref is None else int(len(path_ref)),
            "path_start_t": "" if path_ref is None or len(path_ref) == 0 else float(path_ref[0, 5]),
            "path_end_t": "" if path_ref is None or len(path_ref) == 0 else float(path_ref[-1, 5]),
            "query_t": "" if math.isnan(query_t) else float(query_t),
        }
        if target is not None:
            row.update(
                {
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                    "target_yaw": float(target[2]),
                    "v_ref": float(target[3]),
                    "w_ref": float(target[4]),
                }
            )
        self._cmd_writer.writerow(row)

    def record_trajectory(self, wall_time_ns, accepted, pose_count, path_ref, reason=""):
        self._trajectory_count += 1
        path_file = ""
        duration_s = 0.0
        start_t = None
        end_t = None
        if path_ref is not None and len(path_ref) > 0:
            path_file = f"trajectories/trajectory_{self._trajectory_count:06d}.npy"
            np.save(self.output_dir / path_file, path_ref)
            start_t = float(path_ref[0, 5])
            end_t = float(path_ref[-1, 5])
            duration_s = end_t - start_t
        self._traj_file.write(
            json.dumps(
                {
                    "id": self._trajectory_count,
                    "wall_time_ns": int(wall_time_ns),
                    "accepted": bool(accepted),
                    "pose_count": int(pose_count),
                    "path_file": path_file,
                    "path_start_t": start_t,
                    "path_end_t": end_t,
                    "duration_s": duration_s,
                    "reason": reason,
                },
                sort_keys=True,
            )
            + "\n"
        )

    def close(self):
        self._odom_file.close()
        self._cmd_file.close()
        self._traj_file.close()


class CmdVelControlNode(Node):
    def __init__(self, debug_dir=None):
        super().__init__('cmd_vel_control_node')
        self.logger = self.get_logger()  # Use ROS2 logger
        self._debug_recorder = CmdVelDebugRecorder(debug_dir) if debug_dir else None
        if self._debug_recorder is not None:
            self.logger.info(f"cmd_vel debug recorder enabled: {self._debug_recorder.output_dir}")

        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self._odom_pose_initialized = False

        self._odom_stamp_sec = None

        # columns: x, y, yaw, v_ref, w_ref, t_abs
        self._path_ref = None
        self._track_idx = 0
        self._last_traj_update_sec = None
        self._last_traj_log_sec = None
        self._last_zero_log_sec = {}
        self._time_lookahead_s = 0.15
        self._trajectory_expire_grace_s = 0.05
        self._vx_gain_comp = 1.2

        self.create_subscription(PoseStamped, "/insight/vio_100hz", self._odom_cb, 50)
        self.create_subscription(Path, "/planning/trajectory_path", self._traj_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _odom_cb(self, msg: PoseStamped):
        measured_pose = pose_msg2np(msg)
        measured_position = measured_pose[:3, 3]
        measured_rotation = measured_pose[:3, :3]
        odom_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self._odom_pose_initialized:
            self.position = measured_position
            self.rotation = measured_rotation
            self._odom_pose_initialized = True
        else:
            alpha = 0.35  # First-order odom low-pass filter; smaller is smoother but laggier.
            self.position = (1.0 - alpha) * self.position + alpha * measured_position
            self.rotation = measured_rotation

        self._odom_stamp_sec = odom_stamp_sec
        if self._debug_recorder is not None:
            self._debug_recorder.record_odom(
                time.time_ns(),
                odom_stamp_sec,
                measured_position,
                measured_rotation,
                self.position,
                self.rotation,
            )
        self._control_loop()

    def _traj_cb(self, msg: Path):
        now = self._now_sec()
        new_ref = self._rebuild_path(msg)
        if msg.poses and self._odom_stamp_sec is not None:
            path_start_sec = msg.poses[0].header.stamp.sec + msg.poses[0].header.stamp.nanosec * 1e-9
            path_lag_s = self._odom_stamp_sec - path_start_sec
            if path_lag_s > 0.0:
                self.logger.warning(
                    f"received stale /planning/trajectory_path: first pose stamp is "
                    f"{path_lag_s:.3f}s behind latest /insight/vio_100hz "
                    f"(path={path_start_sec:.3f}, odom={self._odom_stamp_sec:.3f}); "
                    "planning_node may be taking too long."
                )
        if (
            self._last_traj_update_sec is not None
            and now - self._last_traj_update_sec < 0.2  # Drop path updates faster than 5 Hz.
        ):
            if self._debug_recorder is not None:
                self._debug_recorder.record_trajectory(
                    time.time_ns(),
                    accepted=False,
                    pose_count=len(msg.poses),
                    path_ref=new_ref,
                    reason="rate_limited",
                )
            return

        if new_ref is None:
            self._path_ref = None
            self._track_idx = 0
            self._log_traj_update(now, len(msg.poses), 0.0, accepted=False)
            if self._debug_recorder is not None:
                self._debug_recorder.record_trajectory(
                    time.time_ns(),
                    accepted=False,
                    pose_count=len(msg.poses),
                    path_ref=None,
                    reason="empty_or_invalid",
                )
        elif self._path_ref is None or len(self._path_ref) == 0:
            self._path_ref = new_ref
            self._track_idx = 0
            self._log_traj_update(now, len(msg.poses), float(new_ref[-1, 5] - new_ref[0, 5]), accepted=True)
            if self._debug_recorder is not None:
                self._debug_recorder.record_trajectory(
                    time.time_ns(),
                    accepted=True,
                    pose_count=len(msg.poses),
                    path_ref=new_ref,
                    reason="accepted",
                )
        else:
            # Concatenate by timestamp: keep old segment strictly before new start,
            # then append the new trajectory (replaces overlapping future tail).
            new_start_t = float(new_ref[0, 5])
            keep_mask = self._path_ref[:, 5] < new_start_t
            kept = self._path_ref[keep_mask]
            if len(kept) == 0:
                self._path_ref = new_ref
            else:
                self._path_ref = np.vstack((kept, new_ref))
            self._track_idx = int(np.clip(np.searchsorted(self._path_ref[:, 5], now, side='left'), 0, len(self._path_ref) - 1))
            self._log_traj_update(now, len(msg.poses), float(new_ref[-1, 5] - new_ref[0, 5]), accepted=True)
            if self._debug_recorder is not None:
                self._debug_recorder.record_trajectory(
                    time.time_ns(),
                    accepted=True,
                    pose_count=len(msg.poses),
                    path_ref=new_ref,
                    reason="accepted_concat",
                )
        self._last_traj_update_sec = now

    def _log_traj_update(self, now, pose_count, duration_s, accepted):
        if self._last_traj_log_sec is not None and now - self._last_traj_log_sec < 1.0:
            return
        self._last_traj_log_sec = now
        state = "accepted" if accepted else "ignored"
        self.logger.info(
            f"received /planning/trajectory_path {state}: "
            f"poses={pose_count} duration={duration_s:.3f}s"
        )

    def _rebuild_path(self, path_msg: Path):
        n = len(path_msg.poses)
        if n == 0:
            return None

        xy_yaw = np.zeros((n, 3), dtype=np.float64)
        t = np.zeros(n, dtype=np.float64)
        last_yaw = 0.0
        for i, pose in enumerate(path_msg.poses):
            pose_np = pose_msg2np(pose)
            xy_yaw[i, :2] = pose_np[:2, 3]
            t[i] = pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9

            forward = pose_np[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if np.linalg.norm(forward[:2]) > 1e-6:
                last_yaw = math.atan2(float(forward[1]), float(forward[0]))
            xy_yaw[i, 2] = last_yaw

        path_ref = np.zeros((n, 6), dtype=np.float64)
        path_ref[:, :3] = xy_yaw
        path_ref[:, 5] = t
        if n > 1:
            dt_arr = np.diff(t)
            if np.min(dt_arr) <= 0.0:
                # Planner publishes every other 0.1s trajectory sample.
                t = t[0] + np.arange(n, dtype=np.float64) * 0.2

            yaw_u = np.unwrap(xy_yaw[:, 2])
            for i in range(n - 1):
                dt = max(1e-3, float(t[i + 1] - t[i]))
                ds = float(np.linalg.norm(xy_yaw[i + 1, :2] - xy_yaw[i, :2]))
                path_ref[i, 3] = ds / dt
                path_ref[i, 4] = (yaw_u[i + 1] - yaw_u[i]) / dt
            path_ref[-1, 3] = path_ref[-2, 3]
            path_ref[-1, 4] = path_ref[-2, 4]

        return path_ref

    def _control_loop(self):
        if self._path_ref is None:
            self._publish_zero("no /planning/trajectory_path arrived yet")
            return

        now_sec = self._now_sec()
        expired, query_t, path_end_t = self._trajectory_expired(now_sec)
        if expired:
            self._track_idx = len(self._path_ref) - 1
            self._publish_zero(
                "trajectory expired",
                f"query={query_t:.3f} end={path_end_t:.3f} over={query_t - path_end_t:.3f}s",
            )
            return

        # Keep this consistent with planning_node.camera_to_robot_center().
        camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)  # GO2 control center to camera.
        robot_pos = self.position - self.rotation @ camera_offset
        forward = self.rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        robot_yaw = math.atan2(forward[1], forward[0])

        target = self._find_tracking_target(robot_pos, robot_yaw, now_sec)
        if target is None:
            self._publish_zero("no valid tracking target")
            return

        tx, ty, heading_err = self._target_error(robot_pos, robot_yaw, target)
        v_ref = float(target[3])
        w_ref = float(target[4])

        b = 1.2
        zeta = 0.7
        k = 2.0 * zeta * math.sqrt(w_ref * w_ref + b * v_ref * v_ref)
        v = v_ref * math.cos(heading_err) + k * tx
        wz = w_ref + k * heading_err + b * v_ref * self._sinc(heading_err) * ty
        v *= self._vx_gain_comp
        v = float(np.clip(v, -0.2, 0.6))
        wz = float(np.clip(wz, -0.8, 0.8))

        heading_to_goal = self._wrap_angle(float(self._path_ref[-1, 2]) - robot_yaw)
        if (
            np.linalg.norm(robot_pos[:2] - self._path_ref[-1, :2]) < 0.1
            and abs(heading_to_goal) < 0.1
        ):
            self._publish_zero("local trajectory endpoint reached")
            return

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)
        if self._debug_recorder is not None:
            self._debug_recorder.record_cmd(
                time.time_ns(),
                self._odom_stamp_sec,
                v,
                wz,
                target_idx=self._track_idx,
                target=target,
                tx=float(tx),
                ty=float(ty),
                heading_err=float(heading_err),
                path_ref=self._path_ref,
                query_t=now_sec + self._time_lookahead_s,
            )

        self.logger.info(f"sent cmd_vel vx={v:.3f} vyaw={wz:.3f}")

    def _find_tracking_target(self, robot_pos, robot_yaw, now_sec):
        if self._path_ref is None or len(self._path_ref) == 0:
            return None

        start_idx = int(np.clip(self._track_idx, 0, len(self._path_ref) - 1))
        t_vec = self._path_ref[start_idx:, 5]
        if len(t_vec) > 1 and np.min(np.diff(t_vec)) > 0.0:
            query_t = now_sec + self._time_lookahead_s
            rel_idx = int(np.searchsorted(t_vec, query_t, side='left'))
            target_idx = min(start_idx + rel_idx, len(self._path_ref) - 1)
            self._track_idx = target_idx
            return self._path_ref[target_idx]

        delta = self._path_ref[start_idx:, :2] - robot_pos[:2]
        dist = np.linalg.norm(delta, axis=1)
        if float(np.max(dist)) < 0.05:
            yaw_err = np.abs([self._wrap_angle(float(yaw) - robot_yaw) for yaw in self._path_ref[start_idx:, 2]])
            nearest_idx = start_idx + int(np.argmin(yaw_err))
        else:
            nearest_idx = start_idx + int(np.argmin(dist))
        target_idx = min(nearest_idx + 1, len(self._path_ref) - 1)
        self._track_idx = nearest_idx
        return self._path_ref[target_idx]

    def _trajectory_expired(self, now_sec):
        if self._path_ref is None or len(self._path_ref) == 0:
            return True, now_sec, now_sec
        query_t = now_sec + self._time_lookahead_s
        path_end_t = float(self._path_ref[-1, 5])
        return query_t > path_end_t + self._trajectory_expire_grace_s, query_t, path_end_t

    def _target_error(self, robot_pos, robot_yaw, target):
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]

        cy = math.cos(robot_yaw)
        sy = math.sin(robot_yaw)

        tx = cy * dx + sy * dy
        ty = -sy * dx + cy * dy
        heading_err = self._wrap_angle(float(target[2]) - robot_yaw)

        return tx, ty, heading_err

    @staticmethod
    def _wrap_angle(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _sinc(a):
        if abs(a) < 1e-6:
            return 1.0
        return math.sin(a) / a

    def _publish_zero(self, reason, detail=None):
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        if self._debug_recorder is not None:
            self._debug_recorder.record_cmd(
                time.time_ns(),
                self._odom_stamp_sec,
                0.0,
                0.0,
                reason=reason,
                detail=detail,
                path_ref=self._path_ref,
                query_t=self._now_sec() + self._time_lookahead_s,
            )
        now = time.monotonic()
        last_log_sec = self._last_zero_log_sec.get(reason)
        if last_log_sec is None or now - last_log_sec >= 1.0:
            self._last_zero_log_sec[reason] = now
            msg = f"sent cmd_vel vx=0.000 vyaw=0.000 reason={reason}"
            if detail:
                msg = f"{msg} {detail}"
            self.logger.info(msg)

    def _now_sec(self):
        if self._odom_stamp_sec is not None:
            return self._odom_stamp_sec
        return self.get_clock().now().nanoseconds * 1e-9

    def destroy_node(self):
        if self._debug_recorder is not None:
            self._debug_recorder.close()
            self._debug_recorder = None
        self.logger.info("Destroying cmd_vel_control connection.")
        super().destroy_node()


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run cmd_vel controller.")
    parser.add_argument(
        "--debug-record",
        action="store_true",
        help="Record subscribed odometry, received trajectories, and published /cmd_vel to files.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Output directory for --debug-record. Defaults to ~/.local/share/tinynav/cmd_vel_debug/<timestamp>.",
    )
    return parser.parse_known_args(args)


def main(args=None):
    cli_args, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    debug_dir = None
    if cli_args.debug_record:
        debug_dir = cli_args.debug_dir
        if debug_dir is None:
            stamp = datetime.now().strftime("cmd_vel_debug_%Y%m%d_%H%M%S")
            debug_dir = Path.home() / ".local" / "share" / "tinynav" / "cmd_vel_debug" / stamp

    node = CmdVelControlNode(debug_dir=debug_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        
if __name__ == '__main__':
    main()
