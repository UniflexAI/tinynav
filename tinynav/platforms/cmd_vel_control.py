import rclpy
import json
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
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
        self.planner_dt = 0.1       # trajectory dt in planning_node
        # planning_node publishes path with for j in range(..., step=10), so points are ~1.0 s apart.
        self.path_pose_stride = 10
        self.path_period_ema = 0.12
        self.path_filter_tau = 0.30
        self.lookahead_steps = 1
        # Static-friction compensation: very small vx often cannot move the robot.
        # Stairs need a higher floor; 0.1 was leaving the robot crawling / slipping.
        self.min_effective_linear_speed = 0.2
        self.min_effective_angular_speed = 0.1
        self.linear_engage_threshold = 0.04
        self.max_linear_speed = 0.5
        self.fixed_reverse_speed = 0.3
        self.max_reverse_angular_speed = 0.3
        self.terrain_mode = "normal"
        # Hack: if path first segment points far away from robot heading,
        # rotate in place instead of publishing near-zero cmd_vel.
        self.force_turn_heading_threshold = np.deg2rad(30.0)

        self.latest_cmd = Twist()
        self.prev_cmd = Twist()
        self.last_cmd_pub_time = time.monotonic()
        self.last_path_update_time = None
        self._last_cmd_debug_log_time = 0.0
        self._paused = False
        self._nav_active = False
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/nav/paused', self._on_paused, _latched_qos)
        self.create_subscription(Bool, '/nav/active', self._on_nav_active, _latched_qos)
        planning_config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(String, '/planning/config', self._on_planning_config, planning_config_qos)
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self.cmd_timer_callback)

    def _on_planning_config(self, msg: String):
        try:
            config = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.logger.warning(f"Failed to parse /planning/config: {exc}")
            return
        if not isinstance(config, dict):
            return
        if "terrain_mode" in config:
            terrain_mode = str(config["terrain_mode"]).strip().lower()
            if terrain_mode not in ("normal", "stairs"):
                self.logger.warning(f"Invalid planning terrain_mode: {config.get('terrain_mode')!r}")
            else:
                old = self.terrain_mode
                self.terrain_mode = terrain_mode
                if terrain_mode == "stairs":
                    self.fixed_reverse_speed = 0.15
                    self.max_reverse_angular_speed = 0.0
                else:
                    self.fixed_reverse_speed = 0.3
                    self.max_reverse_angular_speed = 0.3
                if old != terrain_mode:
                    self.logger.info(
                        f"Updated cmd_vel terrain_mode: {old} -> {terrain_mode} "
                        f"(fixed_reverse_speed={self.fixed_reverse_speed:.2f}, "
                        f"max_reverse_angular_speed={self.max_reverse_angular_speed:.2f})"
                    )

        if "max_linear_speed" in config:
            try:
                max_linear_speed = float(config["max_linear_speed"])
            except (TypeError, ValueError):
                self.logger.warning(f"Invalid planning max_linear_speed: {config.get('max_linear_speed')!r}")
            else:
                max_linear_speed = max(0.05, min(1.5, max_linear_speed))
                old = self.max_linear_speed
                self.max_linear_speed = max_linear_speed
                if abs(old - max_linear_speed) > 1e-6:
                    self.logger.info(
                        f"Updated cmd_vel max_linear_speed: {old:.2f} -> {max_linear_speed:.2f}"
                    )

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
        target_cmd.angular.z = self.latest_cmd.angular.z
        if age > stale_stop_s:
            target_cmd.linear.x = 0.0
            target_cmd.angular.z = 0.0
            stale_state = "stop"
        elif age > stale_slow_s:
            target_cmd.linear.x *= 0.3
            target_cmd.angular.z *= 0.5
            stale_state = "slow"
        else:
            stale_state = "fresh"

        out = Twist()
        out.linear.y = 0.0

        # Reverse is a predefined planner vocabulary. Pass it through while keeping
        # a conservative yaw-rate cap so reverse-left/right recovery trajectories
        # can actually be executed without spinning too aggressively.
        if target_cmd.linear.x < 0.0:
            out.linear.x = target_cmd.linear.x
            out.angular.z = float(np.clip(
                target_cmd.angular.z,
                -self.max_reverse_angular_speed,
                self.max_reverse_angular_speed,
            ))
            prev_vx = self.prev_cmd.linear.x
            prev_wz = self.prev_cmd.angular.z
            self.cmd_pub.publish(out)
            self.prev_cmd = out
            self._log_cmd_debug(now, age, stale_state, target_cmd, out, prev_vx, prev_wz, reverse_passthrough=True)
            return

        # Forward/turning commands still get acceleration limiting and robot minimum-speed locks.
        max_dv = self.max_linear_acc * dt
        # If we just left reverse mode, do not let acceleration limiting leak another reverse command.
        prev_linear_x = 0.0 if self.prev_cmd.linear.x < 0.0 else self.prev_cmd.linear.x
        out.linear.x = self._clamp_step(target_cmd.linear.x, prev_linear_x, max_dv)
        # Low-pass filter yaw: raw path-derived heading rate can spike between
        # consecutive replans (lookahead/occupancy changes); smooth it here instead
        # of hard acceleration-limiting so large sustained turns still track quickly.
        alpha = dt / (self.path_filter_tau + dt)
        out.angular.z = float(self.prev_cmd.angular.z + alpha * (target_cmd.angular.z - self.prev_cmd.angular.z))

        # Linear x: robot cannot execute tiny non-zero speeds reliably.
        # Any still-forward request below the floor snaps up to min; only a true
        # stop request (target <= 0) is allowed to decay to 0.
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

        prev_vx = self.prev_cmd.linear.x
        prev_wz = self.prev_cmd.angular.z
        self.cmd_pub.publish(out)
        self.prev_cmd = out
        self._log_cmd_debug(now, age, stale_state, target_cmd, out, prev_vx, prev_wz, reverse_passthrough=False)

    def _log_cmd_debug(self, now, age, stale_state, target_cmd, out, prev_vx, prev_wz, reverse_passthrough):
        if (
            now - self._last_cmd_debug_log_time < 0.5
            and stale_state == "fresh"
            and not reverse_passthrough
        ):
            return
        self._last_cmd_debug_log_time = now
        self.logger.info(
            "cmd_vel_debug "
            f"age={age:.2f} stale_state={stale_state} "
            f"path_dt_ema={self.path_period_ema:.2f} "
            f"target_vx={target_cmd.linear.x:.2f} target_wz={target_cmd.angular.z:.2f} "
            f"out_vx={out.linear.x:.2f} out_wz={out.angular.z:.2f} "
            f"prev_vx={prev_vx:.2f} prev_wz={prev_wz:.2f} "
            f"reverse_passthrough={reverse_passthrough}"
        )
        
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
            vx = float(np.clip(raw_vx, 0.0, self.max_linear_speed))
        vy = 0.0
        vyaw = np.clip(angular_velocity_vec[2], -0.8, 0.8)
        is_backward_segment = raw_vx < 0.0
        if is_backward_segment:
            vyaw = float(np.clip(vyaw, -self.max_reverse_angular_speed, self.max_reverse_angular_speed))

        # Hack: if path first segment points >80 deg away from robot heading,
        # force an in-place turn. Skip explicit backward segments because reverse
        # naturally has heading_err close to +/-pi.
        if (not is_backward_segment) and abs(heading_err) > self.force_turn_heading_threshold:
            vx = 0.0
            vyaw = float(heading_err)
        # Minimal rotate-first gate: apply only for forward motion.
        elif vx > 0.0 and abs(heading_err) > 0.45:
            vx = 0.0
            vyaw = float(np.clip(1.6 * heading_err, -0.6, 0.6))

        # Store the latest target command directly. Smoothing is intentionally kept
        # only in cmd_timer_callback via acceleration limiting, so planner/control
        # behavior stays easy to reason about during tuning.
        self.latest_cmd.linear.x = float(vx)
        self.latest_cmd.linear.y = float(vy)
        self.latest_cmd.angular.z = float(vyaw)
        age = 0.0 if self.last_path_update_time is None else (time.monotonic() - self.last_path_update_time)
        self.logger.info(
            "cmd_path_debug "
            f"raw_vx={raw_vx:.2f} heading_err={heading_err:.2f} "
            f"backward_segment={is_backward_segment} "
            f"cmd_vx={self.latest_cmd.linear.x:.2f} cmd_wz={self.latest_cmd.angular.z:.2f} "
            f"path_age={age:.2f} path_dt_ema={self.path_period_ema:.2f} lookahead={step_idx}"
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
