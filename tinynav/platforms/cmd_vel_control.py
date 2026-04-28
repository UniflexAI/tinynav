import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
import numpy as np
import logging
import time
from tinynav.core.math_utils import np2msg

# Module-level logger for cases where self.get_logger() is not available
logger = logging.getLogger(__name__)


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(abs(integral_limit))
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float) -> float:
        dt = max(1e-3, float(dt))
        self.integral += float(error) * dt
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        derivative = 0.0 if not self.initialized else (float(error) - self.prev_error) / dt
        self.prev_error = float(error)
        self.initialized = True
        return self.kp * float(error) + self.ki * self.integral + self.kd * derivative


class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.logger = self.get_logger()  # Use ROS2 logger
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pose_pub = self.create_publisher(Odometry, '/control/target_pose_pid', 10)
        self.pose_sub = self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]]
        )
        self.last_path_time = 0.0
        self.pose = None
        self.path = None
        self.path_start_time = None
        self.path_duration = 0.0
        self.path_pos_spline = None
        self.path_ori_slerp = None

        # PID for tracking path against /slam/odometry.
        self.pos_pid = PIDController(kp=1.2, ki=0.05, kd=0.08, integral_limit=1.5)
        self.yaw_pid = PIDController(kp=1.8, ki=0.03, kd=0.10, integral_limit=1.0)
        # Cross-track coupling: map lateral error (robot-frame y) into yaw-rate correction.
        self.cross_track_gain = 1.2

        # === Control loop (ported from planning_node_compare style) ===
        self.cmd_rate_hz = 20.0
        self.path_stale_slow_s = 0.3
        self.path_stale_stop_s = 0.6
        self.max_linear_acc = 0.4   # m/s^2
        self.max_angular_acc = 0.8  # rad/s^2

        self.latest_cmd = Twist()
        self.prev_cmd = Twist()
        self.last_odom_time = None
        self.last_path_update_time = None
        
    def pose_callback(self, msg):
        self.pose = msg
        now = time.monotonic()
        if self.last_odom_time is None:
            dt = 1.0 / self.cmd_rate_hz
        else:
            dt = max(1e-3, now - self.last_odom_time)
        self.last_odom_time = now

        self._publish_pid_target_pose(msg.header.stamp)
        self._publish_cmd_on_odom(now, dt)

    def _publish_pid_target_pose(self, stamp):
        if self.path_start_time is None or self.path_duration <= 0.0:
            return
        now_mono = time.monotonic()
        t_ref = now_mono - self.path_start_time
        ref_pos, ref_quat = self._sample_pose_at(t_ref)
        if ref_pos is None:
            return
        T_robot_ref = self._world_from_pose_quat(ref_pos, ref_quat)
        target_msg = np2msg(T_robot_ref, stamp, "world", "camera_target")
        self.target_pose_pub.publish(target_msg)

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def _wrap_to_pi(self, angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _yaw_from_transform(self, T: np.ndarray) -> float:
        fwd = T[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return float(np.arctan2(fwd[1], fwd[0]))

    def _world_from_pose_quat(self, position: np.ndarray, quat: np.ndarray) -> np.ndarray:
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R.from_quat(quat).as_matrix()
        T_world_cam[:3, 3] = np.asarray(position, dtype=np.float64)
        return T_world_cam

    def _path_pose_camera_to_body(self, position: np.ndarray, quat: np.ndarray):
        """Convert path pose from world-camera to world-body by right multiply."""
        T_world_cam = self._world_from_pose_quat(position, quat)
        T_world_body = T_world_cam @ self.T_robot_to_camera
        pos_body = T_world_body[:3, 3].copy()
        quat_body = R.from_matrix(T_world_body[:3, :3]).as_quat()
        return pos_body, quat_body

    def _current_world_robot_from_odom(self):
        if self.pose is None:
            return None
        p = self.pose.pose.pose.position
        q = self.pose.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        return self._world_from_pose_quat(pos, quat) @ self.T_robot_to_camera

    def _build_path_interpolators(self, path_msg: Path):
        poses = path_msg.poses
        count = len(poses)
        if count < 2:
            return False

        stamp_times = np.array(
            [
                float(pose_stamped.header.stamp.sec)
                + float(pose_stamped.header.stamp.nanosec) * 1e-9
                for pose_stamped in poses
            ],
            dtype=np.float64,
        )
        if not np.all(np.diff(stamp_times) > 0):
            self.logger.warning("Path timestamps are invalid (non-monotonic); skip this path update.")
            return False
        times = stamp_times - stamp_times[0]

        positions = np.empty((count, 3), dtype=np.float64)
        quats = np.empty((count, 4), dtype=np.float64)
        for i, pose_stamped in enumerate(poses):
            p = pose_stamped.pose.position
            q = pose_stamped.pose.orientation
            pos_cam = np.array([p.x, p.y, p.z], dtype=np.float64)
            quat_cam = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            pos_body, quat_body = self._path_pose_camera_to_body(pos_cam, quat_cam)
            positions[i] = pos_body
            quats[i] = quat_body

        try:
            rotations = R.from_quat(quats)
            self.path_pos_spline = CubicSpline(times, positions, axis=0, bc_type='natural')
            self.path_ori_slerp = Slerp(times, rotations)
            self.path_duration = float(times[-1])
            self.path_start_time = time.monotonic()
            self.pos_pid.reset()
            self.yaw_pid.reset()
            return True
        except ValueError as exc:
            self.logger.warning(f"Failed to build path interpolators: {exc}")
            self.path_pos_spline = None
            self.path_ori_slerp = None
            self.path_duration = 0.0
            self.path_start_time = None
            return False

    def _sample_pose_at(self, t: float):
        if self.path_pos_spline is None or self.path_ori_slerp is None:
            return None, None
        t_clamped = float(np.clip(t, 0.0, self.path_duration))
        position = self.path_pos_spline(t_clamped)
        quat = self.path_ori_slerp([t_clamped]).as_quat()[0]
        return position, quat

    def _compute_cmd_from_sampled_path(self, now_mono: float):
        if self.path_start_time is None or self.path_duration <= 0.0:
            return None, None

        t0 = now_mono - self.path_start_time
        t1 = t0 + (1.0 / self.cmd_rate_hz)

        p0, q0 = self._sample_pose_at(t0)
        p1, q1 = self._sample_pose_at(t1)
        if p0 is None or p1 is None:
            return None, None

        T1 = np.eye(4)
        T1[:3, :3] = R.from_quat(q0).as_matrix()
        T1[:3, 3] = p0

        T2 = np.eye(4)
        T2[:3, :3] = R.from_quat(q1).as_matrix()
        T2[:3, 3] = p1

        T_robot_2_to_1 = np.linalg.inv(T1) @ T2

        dt = max(1e-3, t1 - t0)
        linear_velocity_vec = T_robot_2_to_1[:3, 3] / dt
        angular_velocity_vec = R.from_matrix(T_robot_2_to_1[:3, :3]).as_rotvec() / dt
        return linear_velocity_vec, angular_velocity_vec

    def _compute_tracking_pid_cmd(self, now_mono: float, dt: float):
        if self.path_start_time is None or self.path_duration <= 0.0:
            return None, None
        T_robot_now = self._current_world_robot_from_odom()
        if T_robot_now is None:
            return None, None

        t_ref = now_mono - self.path_start_time
        ref_pos, ref_quat = self._sample_pose_at(t_ref)
        if ref_pos is None:
            return None, None
        T_robot_ref = self._world_from_pose_quat(ref_pos, ref_quat)

        # Position error in current robot frame; control forward axis only.
        T_ref_in_robot = np.linalg.inv(T_robot_now) @ T_robot_ref
        pos_err_forward = float(T_ref_in_robot[0, 3])
        pos_err_lateral = float(T_ref_in_robot[1, 3])

        yaw_now = self._yaw_from_transform(T_robot_now)
        yaw_ref = self._yaw_from_transform(T_robot_ref)
        yaw_err = self._wrap_to_pi(yaw_ref - yaw_now)

        vx_correction = self.pos_pid.update(pos_err_forward, dt)
        wz_correction = self.yaw_pid.update(yaw_err, dt) + self.cross_track_gain * pos_err_lateral
        return vx_correction, wz_correction

    def _publish_cmd_on_odom(self, now: float, dt: float):
        pid_vx = 0.0
        pid_wz = 0.0
        pid_cmd = self._compute_tracking_pid_cmd(now, dt)
        if pid_cmd[0] is not None:
            pid_vx, pid_wz = pid_cmd

        vx = np.clip(pid_vx, -0.1, 0.3)
        vyaw = np.clip(pid_wz, -0.8, 0.8)
        self.latest_cmd.linear.x = float(vx)
        self.latest_cmd.linear.y = 0.0
        self.latest_cmd.angular.z = float(vyaw)

        # Stale-path protection: slow down, then stop if planner has not refreshed.
        age = float('inf') if self.last_path_update_time is None else (now - self.last_path_update_time)
        target_cmd = Twist()
        target_cmd.linear.x = self.latest_cmd.linear.x
        target_cmd.angular.z = self.latest_cmd.angular.z
        if age > self.path_stale_stop_s:
            target_cmd.linear.x = 0.0
            target_cmd.angular.z = 0.0
        elif age > self.path_stale_slow_s:
            target_cmd.linear.x *= 0.3
            target_cmd.angular.z *= 0.5

        # Acceleration limiting for smoother control.
        max_dv = self.max_linear_acc * dt
        max_dw = self.max_angular_acc * dt
        out = Twist()
        out.linear.x = self._clamp_step(target_cmd.linear.x, self.prev_cmd.linear.x, max_dv)
        out.angular.z = self._clamp_step(target_cmd.angular.z, self.prev_cmd.angular.z, max_dw)
        out.linear.y = 0.0

        self.cmd_pub.publish(out)
        self.prev_cmd = out
        
    def path_callback(self, msg):
        if msg is None:
            return
        if len(msg.poses) < 2:
            return

        # Save the latest path first.
        self.path = msg
        self.last_path_update_time = time.monotonic()
        self.last_path_time = self.get_clock().now().nanoseconds * 1e-9
        self._build_path_interpolators(msg)
        age = 0.0 if self.last_path_update_time is None else (time.monotonic() - self.last_path_update_time)
        self.logger.debug(f"path updated age={age:.2f}s points={len(msg.poses)}")

    def destroy_node(self):
        self.logger.info("Destroying cmd_vel_control connection.")
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
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
        
if __name__ == '__main__':
    main()
