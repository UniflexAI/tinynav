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
        self.pose = None
        self.path = None

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
        # Static-friction compensation: very small vx often cannot move the robot.
        self.min_effective_linear_speed = 0.1
        self.min_effective_angular_speed = 0.1
        self.linear_engage_threshold = 0.04
        self.fixed_reverse_speed = 0.2
        # Heading-drift PI: cancel each device's constant open-loop yaw bias.
        # The bias is stored per-metre-travelled (rad/m), not rad/s: the drift is a
        # distance-driven asymmetry (e.g. a quadruped's gait bias), so its rad/s
        # contribution scales with vx. Learning divides by vx and application multiplies
        # by the current vx, so a bias learned on fast straight segments self-scales
        # down on slow turns instead of over-rotating.
        self.yaw_kp = 0.35                 # P damping gain (rad/s per rad)
        self.yaw_bias_ki = 0.15            # I gain
        self.yaw_bias_limit = 0.5          # clamp on the learned bias (rad/m)
        self.yaw_bias_min_vx = 0.15        # only learn/apply the bias above this vx (m/s)
        self.straight_ff_threshold = 0.05  # rad/s; |feedforward omega| below this == straight
        self._yaw_bias_per_m = 0.0         # learned integral state (rad/m), relearned each run
        # Low-pass the drift before P/I: intended_yaw comes from a sparse path and
        # step-jumps when the nearest pose changes, which would otherwise spike the
        # P term and corrupt the integral.
        self.drift_filter_tau = 0.15       # s
        self._drift_lp = None              # low-passed drift state (rad)
        # Hack: if path first segment points far away from robot heading,
        # rotate in place instead of publishing near-zero cmd_vel.
        self.force_turn_heading_threshold = np.deg2rad(80.0)

        self.latest_cmd = Twist()
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

    def _yaw_of(self, quat_xyzw):
        fwd = R.from_quat(quat_xyzw).as_matrix() @ np.array([0.0, 0.0, 1.0])  # optical +z = forward
        return float(np.arctan2(fwd[1], fwd[0]))

    def _actual_yaw(self):
        """World heading of the robot's measured forward axis (odometry)."""
        if self.pose is None:
            return None
        o = self.pose.pose.pose.orientation
        return self._yaw_of([o.x, o.y, o.z, o.w])

    def _path_intended_yaw(self):
        """World heading the plan intends here: forward axis of the published trajectory
        pose nearest the robot. The reference for isolating open-loop drift from turning."""
        if self.pose is None or self.path is None or len(self.path.poses) < 1:
            return None
        p = self.pose.pose.pose.position
        best_i, best_d2 = 0, float('inf')
        for i, ps in enumerate(self.path.poses):
            d2 = (ps.pose.position.x - p.x) ** 2 + (ps.pose.position.y - p.y) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        o = self.path.poses[best_i].pose.orientation
        return self._yaw_of([o.x, o.y, o.z, o.w])

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

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

        # Yaw = planner feedforward omega minus the learned drift correction (P damping
        # + per-device bias). Bias is learned only on straight, forward, fast-enough
        # segments (windup guard) and integrated in rad/m by dividing the rad/s update
        # by vx; applied everywhere by multiplying back by the current vx.
        vyaw_ff = float(self.latest_cmd.angular.z)
        vx_now = float(self.latest_cmd.linear.x)
        bias_rate = self._yaw_bias_per_m * vx_now if vx_now > self.yaw_bias_min_vx else 0.0
        intended_yaw, actual_yaw = self._path_intended_yaw(), self._actual_yaw()
        if intended_yaw is not None and actual_yaw is not None:
            drift = float(np.arctan2(np.sin(actual_yaw - intended_yaw),
                                     np.cos(actual_yaw - intended_yaw)))
            # Low-pass the raw drift before it feeds P/I (see drift_filter_tau).
            if self._drift_lp is None:
                self._drift_lp = drift
            else:
                a = dt / (self.drift_filter_tau + dt)
                self._drift_lp += a * (drift - self._drift_lp)
            if abs(vyaw_ff) < self.straight_ff_threshold and vx_now > self.yaw_bias_min_vx:
                self._yaw_bias_per_m += self.yaw_bias_ki * self._drift_lp * dt / vx_now
                self._yaw_bias_per_m = float(np.clip(self._yaw_bias_per_m,
                                                     -self.yaw_bias_limit, self.yaw_bias_limit))
            target_cmd.angular.z = vyaw_ff - (self.yaw_kp * self._drift_lp + bias_rate)
        else:
            self._drift_lp = None
            target_cmd.angular.z = vyaw_ff - bias_rate
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
        # When engaging forward motion, snap to +min; when stopping/decaying, snap to 0.
        if 0.0 < out.linear.x < self.min_effective_linear_speed:
            out.linear.x = self.min_effective_linear_speed if target_cmd.linear.x >= self.min_effective_linear_speed else 0.0
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
        now_mono = time.monotonic()
        if self.last_path_update_time is not None:
            period = np.clip(now_mono - self.last_path_update_time, 0.05, 0.5)
            self.path_period_ema = 0.85 * self.path_period_ema + 0.15 * float(period)
        self.last_path_update_time = now_mono

        def msg2np(msg):
            T = np.eye(4)
            position = msg.pose.position
            rot = msg.pose.orientation
            quat = [rot.x, rot.y, rot.z, rot.w]
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = np.array([position.x, position.y, position.z]).ravel()
            return T
        
        T1 = msg2np(self.path.poses[0])
        step_idx = int(min(self.lookahead_steps, len(self.path.poses) - 1))
        T2 = msg2np(self.path.poses[step_idx])
        T_robot_1 = T1 @ self.T_robot_to_camera
        T_robot_2 = T2 @ self.T_robot_to_camera
        T_robot_2_to_1 = np.linalg.inv(T_robot_1) @ T_robot_2
        p = T_robot_2_to_1[:3, 3]
        heading_err = float(np.arctan2(p[1], p[0]))
        # dt must match actual spacing between published Path poses, not raw trajectory dt.
        dt = self.planner_dt * self.path_pose_stride * max(1, step_idx)
        linear_velocity_vec = p / dt
        r = R.from_matrix(T_robot_2_to_1[:3, :3])
        angular_velocity_vec = r.as_rotvec() / dt

        raw_vx = float(linear_velocity_vec[0])
        if raw_vx < 0.0:
            vx = -self.fixed_reverse_speed
        else:
            vx = float(np.clip(raw_vx, 0.0, 0.5))
        vy = 0.0
        vyaw = np.clip(angular_velocity_vec[2], -self.max_angular_speed, self.max_angular_speed)
        is_backward_segment = raw_vx < 0.0
        if is_backward_segment:
            vyaw = 0.0

        # Hack: if path first segment points >80 deg away from robot heading,
        # force an in-place turn. Skip explicit backward segments because reverse
        # naturally has heading_err close to +/-pi.
        if (not is_backward_segment) and abs(heading_err) > self.force_turn_heading_threshold:
            vx = 0.0
            vyaw = float(np.clip(heading_err, -self.max_angular_speed, self.max_angular_speed))
        # Minimal rotate-first gate: apply only for forward motion.
        elif vx > 0.0 and abs(heading_err) > 0.45:
            vx = 0.0
            vyaw = float(np.clip(1.6 * heading_err, -0.6, 0.6))

        vyaw = float(np.clip(vyaw, -self.max_angular_speed, self.max_angular_speed))

        # Store the latest target command directly. Smoothing is intentionally kept
        # only in cmd_timer_callback via acceleration limiting, so planner/control
        # behavior stays easy to reason about during tuning.
        self.latest_cmd.linear.x = float(vx)
        self.latest_cmd.linear.y = float(vy)
        self.latest_cmd.angular.z = float(vyaw)
        age = 0.0 if self.last_path_update_time is None else (time.monotonic() - self.last_path_update_time)
        self.logger.debug(
            f"cmd vx={self.latest_cmd.linear.x:.3f} vyaw={self.latest_cmd.angular.z:.3f} "
            f"path_age={age:.2f}s path_dt_ema={self.path_period_ema:.2f}s lookahead={step_idx}"
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
