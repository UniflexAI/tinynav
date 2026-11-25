import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation as R
import numpy as np
import logging
import time
import json
from tinynav.core.robot_specs import ROBOT_CONFIG
from tinynav.core.vio_safety import safety_blocked, accept_safety

# Module-level logger for cases where self.get_logger() is not available
logger = logging.getLogger(__name__)

# stair_node publishes /stair/status every 0.5 s while stair mode is on (same rule as planning_node)
STAIR_MODE_TIMEOUT_S = 1.5
# Stair locomotion must clear the quadruped gait dead zone on both flights and
# landings. The planner samples the same minimum; stop/rotation remain zero.
STAIR_MIN_LINEAR_VEL = 0.25
# Below 1 mm a path segment has no reliable translation bearing (in-place turns).
PATH_TRANSLATION_EPS_M = 1e-3

class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.robot = ROBOT_CONFIG
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
        self.max_angular_speed = self.robot.max_angular_vel  # rad/s
        self.max_forward_speed = self.robot.max_linear_vel  # m/s
        self.planner_dt = 0.1       # trajectory dt in planning_node
        # planning_node publishes path with for j in range(..., step=10), so points are ~1.0 s apart.
        self.path_pose_stride = 10
        self.path_period_ema = 0.12
        self.path_filter_tau = 0.30
        self.lookahead_steps = 1
        # Static-friction compensation: very small vx often cannot move the robot.
        self.min_effective_linear_speed = self.robot.min_linear_vel
        self.min_effective_angular_speed = self.robot.min_angular_vel
        self.linear_engage_threshold = 0.04
        self.fixed_reverse_speed = 0.3
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
        self._stair_finishing = False
        self._stair_status_time = -float('inf')
        self._stair_blocked = False
        self.create_subscription(String, '/stair/status', self._on_stair_status, 10)
        self._vio_safety = None
        self._vio_safety_rx = 0.
        self._vio_epoch = None
        self.create_subscription(String, '/nav/vio_safety', self.vio_safety_callback, _latched_qos)
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

    def _on_stair_status(self, msg):
        self._stair_status_time = time.monotonic()
        parts = msg.data.split()
        turning = 'intent=turn' in parts
        changed = turning != getattr(self, '_stair_turning', False)
        self._stair_turning = turning
        self._stair_finishing = len(parts) > 1 and parts[1] == 'finishing'
        self._stair_blocked = len(parts) > 1 and parts[1] in (
            'no_seed', 'odom_invalid', 'depth_invalid', 'paused', 'arrived', 'landing', 'stopped', 'vio_locked', 'vio_wait')
        if (self._stair_blocked or changed) and self._nav_active:
            self.latest_cmd = Twist()
            self.prev_cmd = Twist()
            self.last_path_update_time = None
            self.cmd_pub.publish(Twist())

    @property
    def vio_blocked(self):
        return safety_blocked(getattr(self, '_vio_safety', None), getattr(self, '_vio_safety_rx', 0.), time.monotonic())

    def vio_safety_callback(self, msg):
        state = json.loads(msg.data)
        if not accept_safety(state, self._vio_safety):
            return
        epoch = (state['session'], state['epoch'])
        if epoch != self._vio_epoch or state['blocked']:
            self.latest_cmd = Twist()
            self.prev_cmd = Twist()
            self.path = None
            self.last_path_update_time = None
            self.cmd_pub.publish(Twist())
        self._vio_epoch = epoch
        self._vio_safety, self._vio_safety_rx = state, time.monotonic()

    @property
    def stair_mode(self):
        return time.monotonic() - self._stair_status_time < STAIR_MODE_TIMEOUT_S

    def pose_callback(self, msg):
        self.pose = msg

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def cmd_timer_callback(self):
        now = time.monotonic()
        dt = max(1e-3, now - self.last_cmd_pub_time)
        self.last_cmd_pub_time = now

        if self.vio_blocked:
            self.latest_cmd = self.prev_cmd = Twist()
            self.cmd_pub.publish(Twist())
            return

        if not self._nav_active:
            return

        if self._paused or (self.stair_mode and self._stair_blocked):
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
        elif age > stale_slow_s:
            target_cmd.linear.x *= 0.3
            target_cmd.angular.z *= 0.5
        if self.stair_mode and getattr(self, '_stair_turning', False):
            target_cmd.linear.x = 0.

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
        stair_forward = self.stair_mode and age <= stale_slow_s and target_cmd.linear.x > self.linear_engage_threshold
        if stair_forward:
            target_cmd.linear.x = max(target_cmd.linear.x, STAIR_MIN_LINEAR_VEL)
        max_dv = self.max_linear_acc * dt
        # If we just left reverse mode, do not let acceleration limiting leak another reverse command.
        prev_linear_x = 0.0 if self.prev_cmd.linear.x < 0.0 else self.prev_cmd.linear.x
        out.linear.x = 0. if self.stair_mode and target_cmd.linear.x == 0. else self._clamp_step(target_cmd.linear.x, prev_linear_x, max_dv)
        # Do not acceleration-limit yaw. The planner/control layer already decides the turn rate,
        # and forced rotate-in-place should take effect immediately.
        out.angular.z = float(np.clip(target_cmd.angular.z, -self.max_angular_speed, self.max_angular_speed))

        # Linear x: robot cannot execute tiny non-zero speeds reliably.
        # When engaging forward motion, snap to +min; when stopping/decaying, snap to 0.
        minimum = max(self.min_effective_linear_speed, STAIR_MIN_LINEAR_VEL) if self.stair_mode else self.min_effective_linear_speed
        if 0.0 < out.linear.x < minimum:
            out.linear.x = minimum if target_cmd.linear.x + 1e-6 >= minimum else 0.0
        elif abs(out.linear.x) < minimum:
            out.linear.x = 0.0

        # Angular z: same idea; tiny requested turns snap to executable min, decays snap to 0.
        turn_minimum = self.min_effective_angular_speed
        if self.stair_mode and out.linear.x == 0.:
            turn_minimum = max(turn_minimum, ROBOT_CONFIG.stair_min_turn_vel)
            if age <= stale_slow_s and abs(out.angular.z) > 1e-6:
                out.angular.z = float(np.sign(out.angular.z) * max(abs(out.angular.z), turn_minimum))
        if 0.0 < abs(out.angular.z) < turn_minimum:
            if abs(target_cmd.angular.z) >= turn_minimum:
                out.angular.z = float(np.sign(target_cmd.angular.z) * turn_minimum)
            else:
                out.angular.z = 0.0

        self.cmd_pub.publish(out)
        self.prev_cmd = out
        
    def path_callback(self, msg):
        if CmdVelControlNode.vio_blocked.fget(self):
            return
        cutoff = (getattr(self, '_vio_safety', None) or {}).get('accept_after')
        if cutoff is not None and msg is not None and msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 < cutoff:
            return
        if not self._nav_active or (self.stair_mode and self._stair_blocked):
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
        rotation_1 = T_robot_1[:3, :3]
        p = rotation_1.T @ (T_robot_2[:3, 3] - T_robot_1[:3, 3])
        has_translation = np.linalg.norm(p[:2]) > PATH_TRANSLATION_EPS_M
        if self.stair_mode and getattr(self, '_stair_turning', False) and has_translation:
            self.latest_cmd = self.prev_cmd = Twist()
            self.last_path_update_time = None
            self.cmd_pub.publish(Twist())
            return
        heading_err = float(np.arctan2(p[1], p[0]))
        # dt must match actual spacing between published Path poses, not raw trajectory dt.
        dt = self.planner_dt * self.path_pose_stride * max(1, step_idx)
        linear_velocity_vec = p / dt
        r = R.from_matrix(rotation_1.T @ T_robot_2[:3, :3])
        angular_velocity_vec = r.as_rotvec() / dt

        raw_vx = float(linear_velocity_vec[0])
        is_backward_segment = p[0] < -PATH_TRANSLATION_EPS_M
        if is_backward_segment:
            vx = -self.fixed_reverse_speed
        else:
            vx = float(np.clip(raw_vx, 0.0, self.max_forward_speed)) if p[0] > PATH_TRANSLATION_EPS_M else 0.0
        vy = 0.0
        vyaw = np.clip(angular_velocity_vec[2], -self.max_angular_speed, self.max_angular_speed)
        if is_backward_segment:
            vyaw = 0.0

        # Hack: if path first segment points >80 deg away from robot heading,
        # force an in-place turn. Skip explicit backward segments because reverse
        # naturally has heading_err close to +/-pi.
        if has_translation and (not is_backward_segment) and abs(heading_err) > self.force_turn_heading_threshold:
            vx = 0.0
            vyaw = float(np.clip(heading_err, -self.max_angular_speed, self.max_angular_speed))
        # Minimal rotate-first gate: apply only for forward motion.
        elif has_translation and vx > 0.0 and abs(heading_err) > 0.45:
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
