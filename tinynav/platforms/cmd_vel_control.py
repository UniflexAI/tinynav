import argparse
import math
import logging
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R


class CmdVelControlNode(Node):
    DT = 1.0 / 50.0

    def __init__(self):
        super().__init__("cmd_vel_control_node")
        self.logger = logging.getLogger(__name__)

        # Camera-to-robot base transform (left-multiply camera pose to get robot pose)
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1],
        ], dtype=np.float64)

        # Previous published command for acceleration limiting
        self._prev_vx = 0.0
        self._prev_vyaw = 0.0
        self._prev_forward_err = 0.0
        self._prev_heading_err = 0.0
        self._forward_err_i = 0.0
        self._heading_err_i = 0.0

        # Propagated camera pose from perception /slam/odom_100hz
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self._odom_100hz_valid = False
        self._odom_stamp_sec: float | None = None

        # Trajectory
        self.trajectory: Path | None = None
        self._path_xy: np.ndarray | None = None
        self._path_yaw: np.ndarray | None = None
        self._path_t: np.ndarray | None = None
        self._path_v_ref: np.ndarray | None = None
        self._path_w_ref: np.ndarray | None = None

        # Subscribers
        self.create_subscription(Odometry, "/slam/odometry_100hz", self._odom_100hz_cb, 10)
        self.create_subscription(Path, "/planning/trajectory_path", self._traj_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, "/control/debug", 10)

        self.create_timer(self.DT, self._control_loop)

    def _odom_100hz_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = np.array([p.x, p.y, p.z])
        self.rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self._odom_stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self._odom_100hz_valid = True

    def _traj_cb(self, msg: Path):
        if msg.poses:
            self.trajectory = msg
            self._rebuild_path_cache(msg)
        else:
            self._clear_path_cache()

    def _control_loop(self):
        if not self._odom_100hz_valid or self._path_xy is None:
            self._publish_zero()
            return

        T_cam = np.eye(4)
        T_cam[:3, :3] = self.rotation
        T_cam[:3, 3] = self.position
        T_robot = T_cam @ self.T_robot_to_camera

        robot_pos = T_robot[:3, 3]
        forward_world = T_robot[:3, :3] @ np.array([0.0, 0.0, 1.0])
        robot_yaw = math.atan2(forward_world[1], forward_world[0])
        
        forward_err, lateral_err, heading_err, v_ff, w_ff, t_query = self._compute_tracking_error(robot_pos, robot_yaw)
        if forward_err is None:
            self._publish_zero()
            return

        dt = self.DT

        self._forward_err_i += forward_err * dt
        # FORWARD_I_CLAMP = 0.2
        self._forward_err_i = float(np.clip(self._forward_err_i, -0.2, 0.2))
        d_forward = (forward_err - self._prev_forward_err) / dt
        vx_cmd = (
            # FORWARD_KP = 1.2, FORWARD_KI = 0.02, FORWARD_KD = 0.05
            1.2 * forward_err
            + 0.02 * self._forward_err_i
            + 0.05 * d_forward
        )
        # Keep a small reverse capability for targets behind robot.
        vx_target = float(np.clip(v_ff + vx_cmd, -0.2, 0.5))

        # YAW_FROM_LATERAL_GAIN = 1.2
        heading_ctrl_err = heading_err + 1.2 * lateral_err
        self._heading_err_i += heading_ctrl_err * dt
        # HEADING_I_CLAMP = 0.3
        self._heading_err_i = float(np.clip(self._heading_err_i, -0.3, 0.3))
        d_heading = (heading_ctrl_err - self._prev_heading_err) / dt
        vyaw_cmd = (
            # HEADING_KP = 1.5, HEADING_KI = 0.01, HEADING_KD = 0.08
            1.0 * heading_ctrl_err
            + 0.01 * self._heading_err_i
            + 0.08 * d_heading
        )
        vyaw_target = float(np.clip(w_ff + vyaw_cmd, -0.5, 0.5))

        # Keep rate limits for smooth control
        # MAX_LINEAR_ACC = 0.3, MAX_ANGULAR_ACC = 2.5
        vx = self._clamp_step(vx_target, self._prev_vx, 0.3 * dt)
        vyaw = self._clamp_step(vyaw_target, self._prev_vyaw, 2.5 * dt)

        self._prev_vx = vx
        self._prev_vyaw = vyaw
        self._prev_forward_err = forward_err
        self._prev_heading_err = heading_ctrl_err

        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = vyaw
        self.cmd_pub.publish(cmd)

        dbg = Float32MultiArray()
        dbg.data = [float(heading_err), float(vx), float(vyaw), float(v_ff), float(w_ff)]
        self.debug_pub.publish(dbg)
        self.logger.info(
            "t=%.2f forward_err=%.4f lateral_err=%.4f heading_err=%.4f vff=%.3f wff=%.3f vx=%.3f wz=%.3f",
            t_query,
            forward_err,
            lateral_err,
            heading_err,
            v_ff,
            w_ff,
            vx,
            vyaw,
        )

    def _compute_tracking_error(self, robot_pos: np.ndarray, robot_yaw: float):
        if self._path_xy is None or self._path_yaw is None or self._path_t is None:
            return None, None, None, None, None, None

        if self._odom_stamp_sec is None:
            t_now = float(self.get_clock().now().nanoseconds) * 1e-9
        else:
            t_now = self._odom_stamp_sec

        t_rel = float(np.clip(t_now - float(self._path_t[0]), 0.0, float(self._path_t[-1] - self._path_t[0])))
        target_xy, target_yaw, v_ff, w_ff = self._interpolate_time_trajectory(t_rel)
        tx, ty, etheta = self._target_in_robot_frame_error(robot_pos, robot_yaw, target_xy, target_yaw)
        forward_err = tx
        lateral_err = ty
        heading_err = etheta
        return forward_err, lateral_err, heading_err, v_ff, w_ff, t_rel

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _rebuild_path_cache(self, path_msg: Path):
        n = len(path_msg.poses)
        if n == 0:
            self._clear_path_cache()
            return

        xy = np.zeros((n, 2), dtype=np.float64)
        yaw = np.zeros(n, dtype=np.float64)
        t = np.zeros(n, dtype=np.float64)
        last_yaw = 0.0
        for i, pose_stamped in enumerate(path_msg.poses):
            p = pose_stamped.pose.position
            xy[i, 0] = float(p.x)
            xy[i, 1] = float(p.y)
            t[i] = float(pose_stamped.header.stamp.sec) + float(pose_stamped.header.stamp.nanosec) * 1e-9

            q = pose_stamped.pose.orientation
            target_rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            target_fwd_w = target_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            n_fwd = np.linalg.norm(target_fwd_w[:2])
            if n_fwd > 1e-6:
                last_yaw = math.atan2(float(target_fwd_w[1]), float(target_fwd_w[0]))
            yaw[i] = last_yaw

        # Fallback to fixed dt if planner didn't provide increasing per-point stamps.
        if n > 1:
            dt_min = float(np.min(np.diff(t)))
            if dt_min <= 0.0:
                # TRAJ_DT_FALLBACK = 0.2
                t = t[0] + np.arange(n, dtype=np.float64) * 0.2

        self._path_xy = xy
        self._path_yaw = yaw
        self._path_t = t

        if n == 1:
            self._path_v_ref = np.zeros(1, dtype=np.float64)
            self._path_w_ref = np.zeros(1, dtype=np.float64)
            return

        yaw_u = np.unwrap(yaw)
        v_ref = np.zeros(n, dtype=np.float64)
        w_ref = np.zeros(n, dtype=np.float64)
        for i in range(n - 1):
            dt = max(1e-3, float(t[i + 1] - t[i]))
            ds = math.hypot(float(xy[i + 1, 0] - xy[i, 0]), float(xy[i + 1, 1] - xy[i, 1]))
            v_ref[i] = ds / dt
            w_ref[i] = float(yaw_u[i + 1] - yaw_u[i]) / dt
        v_ref[-1] = v_ref[-2]
        w_ref[-1] = w_ref[-2]
        self._path_v_ref = v_ref
        self._path_w_ref = w_ref

    def _interpolate_time_trajectory(self, t_rel: float):
        if (
            self._path_xy is None
            or self._path_yaw is None
            or self._path_t is None
            or self._path_v_ref is None
            or self._path_w_ref is None
        ):
            return np.array([0.0, 0.0], dtype=np.float64), 0.0, 0.0, 0.0

        n = len(self._path_xy)
        if n == 1:
            return self._path_xy[0].copy(), float(self._path_yaw[0]), 0.0, 0.0

        t0 = float(self._path_t[0])
        t_end = float(self._path_t[-1] - t0)
        tq = float(np.clip(t_rel, 0.0, t_end))
        t_abs = t0 + tq

        idx = int(np.searchsorted(self._path_t, t_abs, side="right") - 1)
        idx = int(np.clip(idx, 0, n - 2))
        ta = float(self._path_t[idx])
        tb = float(self._path_t[idx + 1])
        ratio = 0.0 if tb - ta < 1e-9 else (t_abs - ta) / (tb - ta)

        p0 = self._path_xy[idx]
        p1 = self._path_xy[idx + 1]
        xy = (1.0 - ratio) * p0 + ratio * p1

        y0 = float(self._path_yaw[idx])
        y1 = float(self._path_yaw[idx + 1])
        yaw = self._wrap_angle(y0 + ratio * self._wrap_angle(y1 - y0))
        v_ref = float((1.0 - ratio) * self._path_v_ref[idx] + ratio * self._path_v_ref[idx + 1])
        w_ref = float((1.0 - ratio) * self._path_w_ref[idx] + ratio * self._path_w_ref[idx + 1])
        return xy, yaw, v_ref, w_ref

    # Intuitive errors in robot frame: tx (forward), ty (lateral), etheta.
    def _target_in_robot_frame_error(
        self, robot_pos: np.ndarray, robot_yaw: float, target_xy: np.ndarray, target_yaw: float
    ):
        dx_w = float(target_xy[0]) - float(robot_pos[0])
        dy_w = float(target_xy[1]) - float(robot_pos[1])

        # World delta rotated into robot frame.
        cy = math.cos(robot_yaw)
        sy = math.sin(robot_yaw)
        tx = cy * dx_w + sy * dy_w
        ty = -sy * dx_w + cy * dy_w

        etheta = self._wrap_angle(target_yaw - robot_yaw)
        return tx, ty, etheta

    @staticmethod
    def _clamp_step(target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def _publish_zero(self):
        dt = self.DT
        self._prev_vx = self._clamp_step(0.0, self._prev_vx, 0.3 * dt)
        self._prev_vyaw = self._clamp_step(0.0, self._prev_vyaw, 2.5 * dt)
        # Decay integrators when tracking is unavailable to avoid windup on resume.
        self._forward_err_i *= 0.9
        self._heading_err_i *= 0.9
        cmd = Twist()
        cmd.linear.x = self._prev_vx
        cmd.angular.z = self._prev_vyaw
        self.cmd_pub.publish(cmd)

    def _clear_path_cache(self):
        self.trajectory = None
        self._path_xy = None
        self._path_yaw = None
        self._path_t = None
        self._path_v_ref = None
        self._path_w_ref = None

    def destroy_node(self):
        self.logger.info("Destroying cmd_vel_control node.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="cmd_vel_control.log", help="Path to the log file")
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(parsed_args.log_file)],
    )

    node = CmdVelControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
