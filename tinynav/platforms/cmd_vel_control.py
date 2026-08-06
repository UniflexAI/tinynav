import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation as R
import numpy as np
import logging
import time
from tinynav.core.planning_node import GO2_CONFIG

# Module-level logger for cases where self.get_logger() is not available
logger = logging.getLogger(__name__)

class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.logger = self.get_logger()  # Use ROS2 logger
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]]
        )
        self.last_path_time = 0.0
        # Heading must be referenced at the control center, not the camera, or the short
        # path lies behind it. From GO2_CONFIG so it tracks the planner's footprint.
        self.cam_forward_offset = float(GO2_CONFIG.cam_offset_3d[2])
        self.T_camera_to_control = self.T_robot_to_camera.copy()
        self.T_camera_to_control[2, 3] = -self.cam_forward_offset  # back along camera +z (=forward)
        self.pose = None
        self.path = None
        self._path_xy = None          # (N,2) cached path XY, updated in path_callback
        self._path_pose_yaw = None    # (N,) cached per-pose forward heading

        # === Control loop (ported from planning_node_compare style) ===
        # Planner input is typically 7-10 Hz; over-driving cmd publish rate amplifies jitter.
        self.cmd_rate_hz = 12.0
        # Use minima; actual stale thresholds are scaled by observed planner period.
        self.path_stale_slow_s = 0.35
        self.path_stale_stop_s = 0.8
        self.path_stale_slow_factor = 3.5
        self.path_stale_stop_factor = 5.0
        self.max_linear_acc = 0.6   # m/s^2
        self.max_angular_acc = 0.8  # rad/s^2
        self.max_angular_speed = 0.8  # rad/s
        self.planner_dt = 0.1       # trajectory dt in planning_node
        # planning_node publishes path with for j in range(..., step=10), so points are ~1.0 s apart.
        self.path_pose_stride = 10
        self.path_period_ema = 0.12
        self.path_filter_tau = 0.30
        self.lookahead_steps = 1
        # PI on heading drift: I learns this device's open-loop yaw bias, P damps (do
        # not set 0). The bias is per-metre (rad/m), not rad/s -- the drift is distance-
        # driven, so learning divides by vx and application multiplies it back.
        self.yaw_kp = 0.35             # proportional (damping) gain (rad/s per rad)
        self.yaw_bias_ki = 0.15        # integral gain (see cmd_timer_callback)
        self.yaw_bias_limit = 0.5      # clamp on the learned bias (rad/m)
        self.yaw_bias_min_vx = 0.15    # only learn/apply the bias above this forward speed (m/s)
        self.straight_ff_threshold = 0.05  # rad/s; |feedforward| below this == straight
        self._yaw_bias_per_m = 0.0     # learned bias / integral state (rad/m), relearned each run
        # Low-pass the drift: intended_yaw step-jumps when the nearest path pose
        # changes, which would spike P and corrupt the integral.
        self.drift_filter_tau = 0.15   # s
        self._drift_lp = None          # low-passed drift state (rad)
        # Static-friction compensation: very small vx often cannot move the robot.
        self.min_effective_linear_speed = 0.1
        # Yaw deadzone; must stay below the per-device yaw bias we cancel.
        self.min_effective_angular_speed = 0.03
        self.linear_engage_threshold = 0.04
        self.fixed_reverse_speed = 0.2
        # Rotate-first: too sharp a turn to drive through -> zero vx and turn in place.
        # 0.82 rad/s is upstream's 0.45 rad of heading change 1.0 s downrange, expressed
        # as the yaw rate of the equivalent constant-curvature arc.
        self.rotate_first_omega = 0.82          # rad/s of planned turn that trips it
        self.rotate_first_max_omega = 0.6       # rad/s turn-in-place rate
        self.max_forward_speed = 0.6            # execution ceiling, not the planner's vx_max

        self.latest_cmd = Twist()
        self.path_vyaw_ff = 0.0
        self.path_vyaw_raw = 0.0
        self.is_backward_segment = False
        self.prev_cmd = Twist()
        self.last_cmd_pub_time = time.monotonic()
        self.last_path_update_time = None
        self._paused = False
        self._nav_active = False
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/nav/paused', self._on_paused, _latched_qos)
        self.create_subscription(Bool, '/nav/active', self._on_nav_active, _latched_qos)
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self.cmd_timer_callback)

    def _on_paused(self, msg: Bool):
        self._paused = msg.data
        if not self._paused:
            # Reset prev_cmd so resume starts from zero cleanly
            self.prev_cmd = Twist()

    def _on_nav_active(self, msg: Bool):
        was_active = self._nav_active
        self._nav_active = bool(msg.data)
        if was_active and not self._nav_active:
            self.latest_cmd = Twist()
            self.prev_cmd = Twist()
            self.last_path_update_time = None
            # Send one stop when navigation is deactivated, then stay silent so
            # manual teleop can own /cmd_vel without being overwritten by zeros.
            self.cmd_pub.publish(Twist())

    def pose_callback(self, msg):
        self.pose = msg

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    @staticmethod
    def _pose_to_T(pose_msg) -> np.ndarray:
        T = np.eye(4)
        position = pose_msg.pose.position
        rot = pose_msg.pose.orientation
        quat = [rot.x, rot.y, rot.z, rot.w]
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = np.array([position.x, position.y, position.z]).ravel()
        return T

    def _actual_yaw(self):
        """World heading of the robot's measured forward axis (odometry)."""
        if self.pose is None:
            return None
        fwd = self._pose_to_T(self.pose.pose)[:3, :3] @ np.array([0.0, 0.0, 1.0])  # optical +z = forward
        return float(np.arctan2(fwd[1], fwd[0]))

    def _path_intended_yaw(self):
        """World heading the plan intends here: forward axis of the published trajectory
        pose nearest the control center. The reference for isolating open-loop drift."""
        if self.pose is None or self.path is None or self._path_xy is None or len(self._path_xy) == 0:
            return None
        ctrl_xy = (self._pose_to_T(self.pose.pose) @ self.T_camera_to_control)[:2, 3]
        d2 = np.sum((self._path_xy - ctrl_xy) ** 2, axis=1)
        best_i = int(np.argmin(d2))
        return float(self._path_pose_yaw[best_i])

    def cmd_timer_callback(self):
        now = time.monotonic()
        dt = max(1e-3, now - self.last_cmd_pub_time)
        self.last_cmd_pub_time = now

        if not self._nav_active:
            return

        if self._paused:
            self.cmd_pub.publish(Twist())
            self.prev_cmd = Twist()
            return

        # Stale-path protection: slow down, then stop if planner has not refreshed.
        age = float('inf') if self.last_path_update_time is None else (now - self.last_path_update_time)
        stale_slow_s = max(self.path_stale_slow_s, self.path_period_ema * self.path_stale_slow_factor)
        stale_stop_s = max(self.path_stale_stop_s, self.path_period_ema * self.path_stale_stop_factor)
        target_cmd = Twist()
        target_cmd.linear.x = self.latest_cmd.linear.x

        # Yaw = planner feedforward omega minus the learned per-device yaw bias, where
        # the bias is integrated from the heading drift (measured vs plan-intended).
        intended_yaw = self._path_intended_yaw()
        actual_yaw = self._actual_yaw()
        vx_now = float(self.latest_cmd.linear.x)
        # rad/m bias -> rad/s correction at the current speed; zero when stopped.
        bias_rate = self._yaw_bias_per_m * vx_now if vx_now > self.yaw_bias_min_vx else 0.0
        if intended_yaw is not None and actual_yaw is not None:
            drift = float(np.arctan2(np.sin(actual_yaw - intended_yaw),
                                     np.cos(actual_yaw - intended_yaw)))
            # Low-pass the raw drift before it feeds P/I (see drift_filter_tau).
            if self._drift_lp is None:
                self._drift_lp = drift
            else:
                a = dt / (self.drift_filter_tau + dt)
                self._drift_lp += a * (drift - self._drift_lp)
            # Learn the bias only on straight, non-backward segments moving fast enough
            # that vx is a reliable divisor (windup guard). "straight" also implies fresh,
            # since path_vyaw_ff comes from the latest path. Integrate in rad/m: divide the
            # rad/s drift update by vx so the estimate is speed-independent.
            straight = abs(self.path_vyaw_ff) < self.straight_ff_threshold
            if straight and not self.is_backward_segment and vx_now > self.yaw_bias_min_vx:
                self._yaw_bias_per_m += self.yaw_bias_ki * self._drift_lp * dt / vx_now
                self._yaw_bias_per_m = float(np.clip(self._yaw_bias_per_m,
                                                     -self.yaw_bias_limit, self.yaw_bias_limit))
            vyaw = self.path_vyaw_ff - (self.yaw_kp * self._drift_lp + bias_rate)
        else:
            # No path/pose yet: feedforward minus the bias learned so far.
            self._drift_lp = None
            vyaw = self.path_vyaw_ff - bias_rate
        target_cmd.angular.z = float(np.clip(vyaw, -self.max_angular_speed, self.max_angular_speed))

        # Ahead of the stale guards (those only attenuate, so they run last) and on the
        # UNCLAMPED omega -- the clamp sits below this threshold, so the clamped value
        # could never trip it. Overrides the drift PI: in place, there is no distance for
        # a per-metre bias to act over.
        if (target_cmd.linear.x > 0.0 and not self.is_backward_segment
                and abs(self.path_vyaw_raw) > self.rotate_first_omega):
            target_cmd.linear.x = 0.0
            target_cmd.angular.z = float(np.copysign(self.rotate_first_max_omega,
                                                     self.path_vyaw_raw))

        if age > stale_stop_s:
            target_cmd.linear.x = 0.0
            target_cmd.angular.z = 0.0
        elif age > stale_slow_s:
            target_cmd.linear.x *= 0.3
            target_cmd.angular.z *= 0.5

        out = Twist()
        out.linear.y = 0.0

        # Reverse is a predefined planner vocabulary: straight back at fixed speed.
        # Do not smooth or re-lock it here; just pass it through while stale/paused guards still work.
        if target_cmd.linear.x < 0.0:
            out.linear.x = target_cmd.linear.x
            out.angular.z = 0.0
            self.cmd_pub.publish(out)
            self.prev_cmd = out
            return

        # Forward/turning commands still get acceleration limiting and robot minimum-speed locks.
        max_dv = self.max_linear_acc * dt
        # If we just left reverse mode, do not let acceleration limiting leak another reverse command.
        prev_linear_x = 0.0 if self.prev_cmd.linear.x < 0.0 else self.prev_cmd.linear.x
        out.linear.x = self._clamp_step(target_cmd.linear.x, prev_linear_x, max_dv)
        # Do not acceleration-limit yaw. The planner/control layer already decides the turn rate,
        # and forced rotate-in-place should take effect immediately.
        out.angular.z = float(np.clip(target_cmd.angular.z, -self.max_angular_speed, self.max_angular_speed))

        # Linear x: robot cannot execute tiny non-zero speeds reliably.
        # Creep at +min for ANY positive target, else a sub-minimum target deadlocks.
        if 0.0 < out.linear.x < self.min_effective_linear_speed:
            out.linear.x = self.min_effective_linear_speed if target_cmd.linear.x > 0.0 else 0.0
        elif abs(out.linear.x) < self.min_effective_linear_speed:
            out.linear.x = 0.0

        # Angular z: same idea; tiny requested turns snap to executable min, decays snap to 0.
        if 0.0 < abs(out.angular.z) < self.min_effective_angular_speed:
            if abs(target_cmd.angular.z) >= self.min_effective_angular_speed:
                out.angular.z = float(np.sign(target_cmd.angular.z) * self.min_effective_angular_speed)
            else:
                out.angular.z = 0.0

        self.cmd_pub.publish(out)
        self.prev_cmd = out
        
    def path_callback(self, msg):
        if not self._nav_active:
            return
        if msg is None or self.pose is None:
            return
        if len(msg.poses) < 2:
            return
        self.path = msg

        ros_now = self.get_clock().now().to_msg()
        self.last_path_time = ros_now.sec + ros_now.nanosec * 1e-9

        # Cached so _path_intended_yaw (at cmd_rate_hz) is a vectorized argmin, not a
        # per-pose scipy Rotation build. Forward (optical +z) heading from quaternion:
        #   fwd_x = 2(xz + wy), fwd_y = 2(yz - wx).
        px = np.array([p.pose.position.x for p in msg.poses])
        py = np.array([p.pose.position.y for p in msg.poses])
        qx = np.array([p.pose.orientation.x for p in msg.poses])
        qy = np.array([p.pose.orientation.y for p in msg.poses])
        qz = np.array([p.pose.orientation.z for p in msg.poses])
        qw = np.array([p.pose.orientation.w for p in msg.poses])
        self._path_xy = np.stack([px, py], axis=1)
        self._path_pose_yaw = np.arctan2(2.0 * (qy * qz - qw * qx),
                                         2.0 * (qx * qz + qw * qy))

        now_mono = time.monotonic()
        if self.last_path_update_time is not None:
            period = np.clip(now_mono - self.last_path_update_time, 0.05, 0.5)
            self.path_period_ema = 0.85 * self.path_period_ema + 0.15 * float(period)
        self.last_path_update_time = now_mono

        # (vx, omega) from Path geometry: relative transform between the first published
        # pose and the one lookahead_steps downrange, over the time between them.
        T_robot_1 = self._pose_to_T(msg.poses[0]) @ self.T_robot_to_camera
        step_idx = int(min(self.lookahead_steps, len(msg.poses) - 1))
        T_robot_2 = self._pose_to_T(msg.poses[step_idx]) @ self.T_robot_to_camera
        T_robot_2_to_1 = np.linalg.inv(T_robot_1) @ T_robot_2
        dt = self.planner_dt * self.path_pose_stride * max(1, step_idx)
        raw_vx = float(T_robot_2_to_1[0, 3] / dt)
        raw_vyaw = float(R.from_matrix(T_robot_2_to_1[:3, :3]).as_rotvec()[2] / dt)
        vyaw = raw_vyaw

        is_backward_segment = raw_vx < 0.0
        if is_backward_segment:
            vx = -self.fixed_reverse_speed
        else:
            vx = float(np.clip(raw_vx, 0.0, self.max_forward_speed))
            # Preserve turn radius (vx/omega) when omega exceeds the cap: scale vx by the
            # same ratio instead of just clipping omega (which would widen the radius).
            if abs(vyaw) > self.max_angular_speed:
                vx *= self.max_angular_speed / abs(vyaw)
                vyaw = float(np.sign(vyaw) * self.max_angular_speed)

        # Feedforward yaw; the heading-drift PI is applied per-tick in the timer.
        self.is_backward_segment = is_backward_segment
        self.path_vyaw_ff = 0.0 if is_backward_segment else vyaw
        # What the plan asked for, BEFORE the clamp above -- rotate-first needs it.
        self.path_vyaw_raw = 0.0 if is_backward_segment else raw_vyaw
        self.latest_cmd.linear.x = float(vx)
        self.latest_cmd.linear.y = 0.0
        self.logger.debug(
            f"target_vx={self.latest_cmd.linear.x:.3f} vyaw_ff={self.path_vyaw_ff:.3f} "
            f"vyaw_raw={self.path_vyaw_raw:.3f} backward={self.is_backward_segment} "
            f"lookahead={step_idx}"
        )

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
