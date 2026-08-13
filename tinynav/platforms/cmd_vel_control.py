import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation as R
import numpy as np
import logging
import time
from tinynav.core.planning_node import ROBOT_CONFIG_TOPIC, RobotConfig

class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.logger = self.get_logger()  # Use ROS2 logger
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)
        self.create_subscription(Twist, '/planning/velocity_ff', self.velocity_ff_callback, 10)
        self.create_subscription(Float32, '/planning/forward_speed_cap', self._forward_speed_cap_callback, 10)
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]]
        )
        # Camera sits this far ahead of the control center; heading must be referenced
        # at the control center or the short path lies behind the camera. Same
        # RobotConfig the planner uses, published by the chassis bridge on
        # ROBOT_CONFIG_TOPIC, so it can't drift out of sync with the planner's
        # footprint geometry (cam_offset_3d forward component = camera_x - control_x).
        self.T_camera_to_control = self.T_robot_to_camera.copy()
        self._apply_robot_config(RobotConfig())  # fallback until the bridge publishes
        self.pose = None
        self.path = None
        self._path_xy = None          # (N,2) cached path XY, updated in path_callback
        self._path_pose_yaw = None    # (N,) cached per-pose forward heading

        self.cmd_rate_hz = 12.0
        # Minima; actual stale thresholds are scaled by observed planner period.
        self.path_stale_slow_s = 0.35
        self.path_stale_stop_s = 0.8
        self.path_stale_slow_factor = 3.5
        self.path_stale_stop_factor = 5.0
        self.max_linear_acc = 0.6   # m/s^2
        self.max_angular_acc = 0.8  # rad/s^2
        # Hardware execution limit (upstream's value). Deliberately BELOW the planner's
        # +/-pi/3 omega sampling: raising it to match the lattice made this clamp a
        # no-op, which also disabled the radius-preserving vx scale-down it feeds.
        self.max_angular_speed = 0.8  # rad/s
        self.path_period_ema = 0.12
        # Heading-drift control: PI on heading drift. I learns each device's open-loop
        # yaw bias (zero steady-state error); P provides damping (do not set 0).
        # The bias is stored per-metre-travelled (rad/m), not rad/s: the drift is a
        # distance-driven asymmetry, so its rad/s contribution scales with vx. Learning
        # divides by vx and application multiplies by the current vx, so a bias learned
        # on fast straight segments self-scales down on slow turns instead of over-rotating.
        self.yaw_kp = 0.35             # proportional (damping) gain (rad/s per rad)
        self.yaw_bias_ki = 0.15        # integral gain (see cmd_timer_callback)
        self.yaw_bias_limit = 0.5      # clamp on the learned bias (rad/m)
        self.yaw_bias_min_vx = 0.15    # only learn/apply the bias above this forward speed (m/s)
        # Integrate only while the plan is roughly straight (feedforward near zero);
        # this also implies the path is fresh, since path_vyaw_ff comes from the latest path.
        self.straight_ff_threshold = 0.05  # rad/s; |feedforward| below this == straight
        self._yaw_bias_per_m = 0.0     # learned bias / integral state (rad/m), relearned each run
        # Low-pass the drift before P/I: intended_yaw comes from a sparse path and
        # step-jumps when the nearest pose changes, which would otherwise inject a
        # spike into the P term and corrupt the integral.
        self.drift_filter_tau = 0.15   # s
        self._drift_lp = None          # low-passed drift state (rad)
        # Static-friction compensation: very small vx often cannot move the robot.
        # 0.1 (main's value): 0.2 doubled the step this deadzone injects at every stop
        # and doubled the creep speed the planner's sub-minimum targets get raised to,
        # which overshot the goal.
        self.min_effective_linear_speed = 0.1
        # Yaw deadzone; must stay below the per-device yaw bias we cancel.
        self.min_effective_angular_speed = 0.03
        self.linear_engage_threshold = 0.04
        self.fixed_reverse_speed = 0.2
        # Rotate-first (upstream's gate, keyed off the planner's own feedforward): when
        # the plan turns hard enough that driving forward would cut the corner, zero vx
        # and turn in place until it straightens out. Upstream measured the angle from
        # Path geometry because that is where it derived (vx, omega); here the planner
        # publishes its chosen omega, so gate on that -- same quantity, with no
        # dependence on the Path's decimation stride or lookahead horizon.
        # Threshold: upstream tripped at 0.45 rad of heading change 1.0 s downrange,
        # which for the lattice's constant-curvature arcs is |omega| * 1.1 / 2, i.e.
        # 0.82 rad/s. Its proportional term (1.6 * error) clipped to the max for every
        # omega the +/-pi/3 lattice can produce, so the turn was always the clamp value;
        # commanding the clamp directly is the same command with one fewer dead knob.
        self.rotate_first_omega = 0.82          # rad/s of planned turn that trips it
        self.rotate_first_max_omega = 0.6       # rad/s turn-in-place rate
        # Forward-speed cap tracks the planner's open-space target (capture-speed prior,
        # or its vx_max fallback) via /planning/forward_speed_cap, so a prior that raises
        # speed above the old static default is executed here too instead of being clipped.
        # When the stream is absent/stale, fall back to this static ceiling -- a
        # deliberate conservative floor, intentionally independent of the planner's
        # vx_max (we clip to a known-safe speed rather than trust a value we no
        # longer receive), so it does not track vx_max if that is retuned.
        self.max_forward_speed_fallback = 0.6
        self._forward_speed_cap = None
        self._forward_speed_cap_time = None
        self.forward_speed_cap_ttl_s = 2.0

        self.latest_cmd = Twist()
        self.path_vyaw_ff = 0.0
        self.is_backward_segment = False
        self.prev_cmd = Twist()
        self.last_cmd_pub_time = time.monotonic()
        self.last_path_update_time = None
        self._paused = False
        self._nav_active = False
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/nav/paused', self._on_paused, _latched_qos)
        self.create_subscription(Bool, '/nav/active', self._on_nav_active, _latched_qos)
        self.create_subscription(String, ROBOT_CONFIG_TOPIC, self._on_robot_config, _latched_qos)
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self.cmd_timer_callback)

    def _apply_robot_config(self, robot: RobotConfig):
        self.robot = robot
        self.cam_forward_offset = float(robot.cam_offset_3d[2])
        self.T_camera_to_control[2, 3] = -self.cam_forward_offset  # back along camera +z (=forward)

    def _on_robot_config(self, msg: String):
        try:
            robot = RobotConfig.from_json(msg.data)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Bad {ROBOT_CONFIG_TOPIC} payload ({e}); keeping {self.robot.name}")
            return
        self._apply_robot_config(robot)
        self.logger.info(f"Robot ({ROBOT_CONFIG_TOPIC}): {robot.name}, "
                         f"cam_forward_offset={self.cam_forward_offset:.3f}m")

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

    def _forward_speed_cap_callback(self, msg):
        self._forward_speed_cap = float(msg.data)
        self._forward_speed_cap_time = time.monotonic()

    def _current_forward_cap(self):
        """Planner's open-space target speed if a fresh, finite value is available,
        else the static fallback ceiling. _forward_speed_cap and its timestamp are set
        together, so the value None-check also guards the timestamp."""
        if (self._forward_speed_cap is not None and np.isfinite(self._forward_speed_cap)
                and time.monotonic() - self._forward_speed_cap_time <= self.forward_speed_cap_ttl_s):
            return self._forward_speed_cap
        return self.max_forward_speed_fallback

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

        # Rotate-first, ahead of the stale guards: those only ever attenuate, so they
        # must run last. Gated on the planner's UNCLAMPED omega (path_vyaw_raw), not the
        # clamped path_vyaw_ff -- the clamp sits at max_angular_speed, below this
        # threshold, so gating on the clamped value would make this unreachable. Reverse
        # is excluded (the fixed-speed vocabulary owns its own heading). Replaces the
        # drift-PI yaw outright: turning in place covers no distance for a per-metre
        # bias to apply over, and bias learning is already gated off below
        # yaw_bias_min_vx.
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

        # Reverse is a fixed-speed straight-back vocabulary; pass it through unsmoothed.
        if target_cmd.linear.x < 0.0:
            out.linear.x = target_cmd.linear.x
            out.angular.z = 0.0
            self.cmd_pub.publish(out)
            self.prev_cmd = out
            return

        # Forward/turning commands get acceleration limiting and minimum-speed locks.
        max_dv = self.max_linear_acc * dt
        # Just left reverse: don't let acceleration limiting leak another reverse command.
        prev_linear_x = 0.0 if self.prev_cmd.linear.x < 0.0 else self.prev_cmd.linear.x
        out.linear.x = self._clamp_step(target_cmd.linear.x, prev_linear_x, max_dv)
        # Don't acceleration-limit yaw; the turn rate is already decided upstream.
        out.angular.z = float(np.clip(target_cmd.angular.z, -self.max_angular_speed, self.max_angular_speed))

        # Tiny non-zero forward speeds aren't executable: creep at +min for any positive
        # target (else the robot freezes and deadlocks); non-positive target decays to 0.
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
        if len(msg.poses) < 1:
            return
        self.path = msg

        # Cache path XY and per-pose forward heading so _path_intended_yaw (called at
        # cmd_rate_hz) is a vectorized argmin instead of a Python loop over poses with
        # a scipy Rotation build per pose. Forward (optical +z) heading from quaternion:
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

    def velocity_ff_callback(self, msg):
        """Planner-selected instantaneous (vx, omega_y), published straight from the
        trajectory it picked -- no need to reverse-engineer it from path poses."""
        raw_vx = float(msg.linear.x)
        vyaw = float(msg.angular.z)

        # Reverse is a fixed-speed straight-back maneuver, explicitly flagged by the
        # planner (angular.x) rather than inferred from the sign of vx -- a real Twist
        # can legitimately have vx<0 with nonzero omega (reversing while turning).
        is_backward_segment = bool(msg.angular.x)
        if is_backward_segment:
            vx = -self.fixed_reverse_speed
        else:
            vx = float(np.clip(raw_vx, 0.0, self._current_forward_cap()))
            # Preserve turn radius (vx/omega) when omega exceeds the cap: scale vx by the
            # same ratio instead of just clipping omega (which would widen the radius).
            if abs(vyaw) > self.max_angular_speed:
                vx *= self.max_angular_speed / abs(vyaw)
                vyaw = float(np.sign(vyaw) * self.max_angular_speed)

        # Feedforward yaw rate; the heading-drift PI is applied per-tick in the timer.
        self.is_backward_segment = is_backward_segment
        self.path_vyaw_ff = 0.0 if is_backward_segment else vyaw
        # How hard the plan itself turns, BEFORE the max_angular_speed clamp above. The
        # clamp is what we can execute; this is what was asked for, and rotate-first
        # needs the latter to tell "too sharp to drive through" from "at the limit".
        self.path_vyaw_raw = 0.0 if is_backward_segment else float(msg.angular.z)
        self.latest_cmd.linear.x = float(vx)
        self.latest_cmd.linear.y = 0.0
        self.logger.debug(
            f"target_vx={self.latest_cmd.linear.x:.3f} vyaw_ff={self.path_vyaw_ff:.3f} "
            f"vyaw_raw={self.path_vyaw_raw:.3f} backward={self.is_backward_segment}"
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
