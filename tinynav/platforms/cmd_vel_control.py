import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float32
from rclpy.qos import DurabilityPolicy, QoSProfile
import numpy as np
import logging
import time
from tinynav.core.robot_specs import ROBOT_CONFIG

# Module-level logger for cases where self.get_logger() is not available
logger = logging.getLogger(__name__)

class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.robot = ROBOT_CONFIG
        self.logger = self.get_logger()  # Use ROS2 logger
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)
        self.create_subscription(Twist, '/planning/velocity_ff', self.velocity_ff_callback, 10)
        self.create_subscription(Float32, '/planning/forward_speed_cap', self._forward_speed_cap_callback, 10)
        self.path = None

        self.cmd_rate_hz = 12.0
        # Minima; actual stale thresholds are scaled by observed planner period.
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
        # Yaw is the planner's feedforward, passed through -- upstream's shape, no heading
        # feedback here. A drift PI used to sit at this point; its reference was the path
        # pose nearest the robot, which the replan puts on top of the robot, so it
        # measured ~nothing while subtracting enough to flatten real turns.
        # Static-friction compensation: very small vx often cannot move the robot.
        self.min_effective_linear_speed = self.robot.min_linear_vel
        self.min_effective_angular_speed = self.robot.min_angular_vel
        # Sub-floor yaw requests are banked as an angle and spent at the floor (see
        # cmd_timer_callback). The cap is roughly two ticks' worth at the floor.
        self._yaw_bank = 0.0           # rad
        self.yaw_bank_limit = 0.02     # rad
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
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self.cmd_timer_callback)

    def _on_paused(self, msg: Bool):
        self._paused = msg.data
        if not self._paused:
            # Reset prev_cmd so resume starts from zero cleanly
            self.prev_cmd = Twist()
            self._yaw_bank = 0.0

    def _on_nav_active(self, msg: Bool):
        was_active = self._nav_active
        self._nav_active = bool(msg.data)
        if was_active and not self._nav_active:
            self.latest_cmd = Twist()
            self.prev_cmd = Twist()
            self._yaw_bank = 0.0
            self.last_path_update_time = None
            # Send one stop when navigation is deactivated, then stay silent so
            # manual teleop can own /cmd_vel without being overwritten by zeros.
            self.cmd_pub.publish(Twist())

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

        # Yaw is the planner's feedforward, clipped to what the base can turn. Nothing
        # else touches it here -- see the note where the drift PI used to be configured.
        target_cmd.angular.z = float(np.clip(self.path_vyaw_ff,
                                             -self.max_angular_speed, self.max_angular_speed))

        # Rotate-first, ahead of the stale guards: those only ever attenuate, so they
        # must run last. Gated on the planner's UNCLAMPED omega (path_vyaw_raw), not the
        # clamped path_vyaw_ff -- the clamp sits at max_angular_speed, below this
        # threshold, so gating on the clamped value would make this unreachable. Reverse
        # is excluded (the fixed-speed vocabulary owns its own heading).
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

        # Angular z: a request below the executable floor used to be published as 0
        # while linear.x kept its accel-limited value, so the robot drove straight
        # through the turn -- measured on go2w, 43% of turn commands went out as exactly
        # 0.00 at a mean 0.53 m/s, and the drift they caused is not recoverable by a
        # controller that only measures heading. Bank the requested yaw ANGLE instead and
        # spend it as whole ticks at the floor: every tick is executable, and the mean
        # yaw rate over a few ticks is the one that was asked for. No guess at the real
        # floor is needed, which matters because min_angular_vel is still the shared
        # placeholder on every chassis.
        floor = self.min_effective_angular_speed
        if 0.0 < abs(out.angular.z) < floor:
            self._yaw_bank += out.angular.z * dt
            quantum = floor * dt
            if abs(self._yaw_bank) >= quantum:
                out.angular.z = float(np.copysign(floor, self._yaw_bank))
                self._yaw_bank -= np.copysign(quantum, self._yaw_bank)
            else:
                out.angular.z = 0.0
            # Bounded so a long sub-floor stretch cannot bank a turn it then spends
            # after the plan has moved on.
            self._yaw_bank = float(np.clip(self._yaw_bank, -self.yaw_bank_limit,
                                           self.yaw_bank_limit))
        else:
            self._yaw_bank = 0.0

        self.cmd_pub.publish(out)
        self.prev_cmd = out
        
    def path_callback(self, msg):
        if not self._nav_active:
            return
        if msg is None:
            return
        if len(msg.poses) < 1:
            return
        self.path = msg
        # Kept only as the freshness signal the stale guards read below. The old
        # `self.pose is None` gate went with the yaw reference that needed it -- while it
        # stood, a missing pose froze this timestamp and read as a stale plan.

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
