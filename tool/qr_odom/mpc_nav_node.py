#!/usr/bin/env python3
"""
tool/qr_odom/mpc_nav_node.py

Navigate to a fixed target pose defined relative to an AprilTag board, using a
Plan (turn-drive-turn reference trajectory) + MPC (receding-horizon tracking)
controller instead of the reactive PI controller in nav_node.py.

Robot model: unicycle with in-place rotation (v can be 0 while omega != 0).
The robot is treated as non-holonomic in the same sense as the general nav
stack (tinynav/platforms/cmd_vel_control.py): commands are (v, omega) only,
never a lateral velocity.

Frame chain (same as nav_node.py)
----------------------------------
  Goal (fixed in map frame):
    T_map_goal = T_map_qrworld @ T_qrworld_robot   (both predefined)

  Current pose in map frame:
    T_world_camera  <- /slam/odometry_fused
    T_world_map     <- TF world->map  (broadcast by map_node)
    T_map_camera    = inv(T_world_map) @ T_world_camera
    T_map_robot     = T_map_camera @ T_CAMERA_ROBOT

Plan
----
  Closed-form, time-parameterized "turn-drive-turn" reference trajectory,
  computed once from the current pose to the fixed goal:
    1. rotate in place to face the straight-line bearing to the goal
    2. drive straight along that bearing
    3. rotate in place to the final goal heading
  Each phase uses a trapezoidal (bang-coast-bang) velocity profile bounded by
  MAX_LINEAR/MAX_ANGULAR and MAX_LINEAR_ACC/MAX_ANGULAR_ACC. See make_plan().

MPC
---
  Single-shooting nonlinear MPC over a short horizon, solved every tick with
  scipy.optimize.minimize(method="SLSQP"): box bounds on (v, omega), a
  LinearConstraint enforcing per-step acceleration limits, quadratic tracking
  + control-effort + control-rate cost, warm-started from the previous
  solution (shift-and-pad). See solve_mpc().

  Runs on a fixed-rate timer (CONTROL_HZ), decoupled from the incoming
  odometry rate. If tracking error against the current plan grows too large,
  or the plan's nominal duration has elapsed without reaching the goal, the
  plan is recomputed from the current pose (see _control_tick).

Calibration files
------------------
  tinynav_db/qrcode/tag_mappose.json   T_map_qrworld
  tinynav_db/qrcode/tag_target.json    T_qrworld_robot  (existing)

Topics
------
  Subscribed:  /slam/odometry_fused    nav_msgs/Odometry
  TF lookup:   world -> map            (broadcast by map_node)
  Published:   /control/cmd_vel        geometry_msgs/Twist
               /qr_world/nav_done      std_msgs/Bool        (once, on reaching goal)
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.optimize import LinearConstraint, minimize
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from tinynav.core.math_utils import msg2np, tf2np
from tool.qr_odom.robot_frame import T_CAMERA_ROBOT

DB_DIR           = Path("tinynav_db/qrcode")
TAG_MAPPOSE_PATH = DB_DIR / "tag_mappose.json"
TARGET_PATH      = DB_DIR / "tag_target.json"

ODOM_TOPIC     = "/slam/odometry_fused"
CMD_VEL_TOPIC  = "/control/cmd_vel"
NAV_DONE_TOPIC = "/qr_world/nav_done"

# Control loop, decoupled from the (up to 100Hz) EKF-fused odometry rate.
CONTROL_HZ = 10.0
DT_MPC     = 1.0 / CONTROL_HZ
HORIZON_N  = 15
HORIZON_N_DEGRADED = 8   # fallback if solves are too slow for CONTROL_HZ
SOLVE_TIME_BUDGET_S = 0.07

# Velocity/acceleration limits — v/MAX_LINEAR/MAX_ANGULAR reused from
# nav_node.py; MAX_LINEAR_ACC/MAX_ANGULAR_ACC reused from
# tinynav/platforms/cmd_vel_control.py (the latter is defined there but
# currently unused for yaw).
MAX_LINEAR      = 0.3   # m/s  (v >= 0: the turn-drive-turn plan never needs reverse)
MAX_ANGULAR     = 0.5   # rad/s
MAX_LINEAR_ACC  = 0.6   # m/s^2
MAX_ANGULAR_ACC = 0.8   # rad/s^2

# Deadband floors, used only by the emergency P-controller fallback (see
# _fallback_cmd) — the MPC's own output is used as-is otherwise.
MIN_LINEAR  = 0.15
MIN_ANGULAR = 0.15
CMD_DEADBAND = 1e-3

DIST_THRESH    = 0.06   # m    — goal reached position tolerance
HEADING_THRESH = 0.06   # rad  — goal reached heading tolerance
CONVERGE_TICKS = 5      # consecutive in-tolerance ticks before latching "reached"

DIST_EPS = 0.03   # m    — below this, skip the drive phase entirely
TURN_EPS = 0.03   # rad  — below this, skip a turn phase entirely

REPLAN_DIST            = 0.15   # m    — tracking error vs. current plan that triggers a replan
REPLAN_HEADING         = 0.35   # rad
REPLAN_DEBOUNCE_TICKS  = 2
REPLAN_TIMEOUT_SLACK_S = 1.0    # replan if plan's nominal duration is exceeded by this much

# MPC cost weights.
Q_XY      = 5.0
Q_THETA   = 2.0
QF_XY     = 15.0
QF_THETA  = 8.0
R_V       = 0.05
R_OMEGA   = 0.02
RD_V      = 0.5
RD_OMEGA  = 0.3

SLSQP_MAXITER          = 30
SLSQP_MAXITER_DEGRADED = 15


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

def _wrap(angle):
    """Wrap angle(s) to [-pi, pi]. Works on scalars and numpy arrays alike."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _clip_with_min(value: float, min_abs: float, max_abs: float) -> float:
    if abs(value) < CMD_DEADBAND:
        return 0.0
    clipped = float(np.clip(value, -max_abs, max_abs))
    if abs(clipped) < min_abs:
        return float(np.sign(clipped) * min_abs)
    return clipped


def _unicycle_step(x: float, y: float, theta: float, v: float, w: float, dt: float):
    """Exact constant-(v,w) arc integration for one step (exact for a
    piecewise-constant-control unicycle; degrades to straight-line motion as
    w -> 0)."""
    if abs(w) < 1e-3:
        # Straight-line position update (avoids the 1/w division below), but
        # theta must still advance by w*dt — otherwise any rollout starting
        # from w==0 has an exactly-zero d(theta)/d(w) locally, which traps a
        # gradient-based solver (SLSQP) at w=0 even when rotating would
        # clearly reduce cost (e.g. right after a long straight-drive phase,
        # where the warm start's omega block is all zero).
        return x + v * dt * np.cos(theta), y + v * dt * np.sin(theta), theta + w * dt
    theta_new = theta + w * dt
    x_new = x + (v / w) * (np.sin(theta_new) - np.sin(theta))
    y_new = y - (v / w) * (np.cos(theta_new) - np.cos(theta))
    return x_new, y_new, theta_new


def _rollout(x0: float, y0: float, theta0: float, v_arr: np.ndarray, w_arr: np.ndarray, dt: float):
    n = len(v_arr)
    xs = np.empty(n + 1)
    ys = np.empty(n + 1)
    ths = np.empty(n + 1)
    xs[0], ys[0], ths[0] = x0, y0, theta0
    for k in range(n):
        xs[k + 1], ys[k + 1], ths[k + 1] = _unicycle_step(
            xs[k], ys[k], ths[k], v_arr[k], w_arr[k], dt)
    return xs, ys, ths


# ---------------------------------------------------------------------------
# Plan stage: turn-drive-turn reference trajectory
# ---------------------------------------------------------------------------

def _trapezoid_duration_and_peak(D: float, v_max: float, a_max: float):
    """Bang-coast-bang profile duration and peak speed for an unsigned
    displacement D. Single formula covers both the triangle (never reaches
    v_max) and full trapezoid cases."""
    D = abs(D)
    if D < 1e-9:
        return 0.0, 0.0
    v_peak = min(v_max, np.sqrt(a_max * D))
    t_acc = v_peak / a_max
    cruise_dist = max(0.0, D - v_peak ** 2 / a_max)
    t_cruise = cruise_dist / v_peak if v_peak > 1e-9 else 0.0
    return 2 * t_acc + t_cruise, v_peak


def _trapezoid_eval(t: float, D: float, v_max: float, a_max: float):
    """Return (progress s(t) in [0,|D|], speed w(t)>=0) for unsigned |D|."""
    D = abs(D)
    if D < 1e-9 or t <= 0.0:
        return 0.0, 0.0
    T, v_peak = _trapezoid_duration_and_peak(D, v_max, a_max)
    if t >= T:
        return D, 0.0
    t_acc = v_peak / a_max
    if t < t_acc:
        return 0.5 * a_max * t * t, a_max * t
    if t <= T - t_acc:
        return 0.5 * a_max * t_acc ** 2 + v_peak * (t - t_acc), v_peak
    tt = T - t
    return D - 0.5 * a_max * tt * tt, a_max * tt


@dataclasses.dataclass
class Plan:
    x0: float
    y0: float
    theta0: float
    xg: float
    yg: float
    thetag: float
    bearing: float
    turn1: float   # signed rotation (rad) to face `bearing`, 0 if skipped
    turn2: float   # signed rotation (rad) from `bearing` to `thetag`, 0 if skipped
    dist: float    # straight-line distance driven in phase 2, 0 if skipped
    T1: float
    T2: float
    T3: float

    @property
    def total_duration(self) -> float:
        return self.T1 + self.T2 + self.T3

    def eval(self, t: float):
        """Reference (x, y, theta, v, omega) at local plan time t (seconds
        since plan start)."""
        t = max(t, 0.0)
        t1 = self.T1
        t2 = self.T1 + self.T2
        t3 = self.T1 + self.T2 + self.T3
        if t < t1:
            s, w = _trapezoid_eval(t, self.turn1, MAX_ANGULAR, MAX_ANGULAR_ACC)
            sign = np.sign(self.turn1)
            return self.x0, self.y0, self.theta0 + sign * s, 0.0, sign * w
        if t < t2:
            s, w = _trapezoid_eval(t - t1, self.dist, MAX_LINEAR, MAX_LINEAR_ACC)
            return (self.x0 + s * np.cos(self.bearing),
                    self.y0 + s * np.sin(self.bearing),
                    self.bearing, w, 0.0)
        if t < t3:
            s, w = _trapezoid_eval(t - t2, self.turn2, MAX_ANGULAR, MAX_ANGULAR_ACC)
            sign = np.sign(self.turn2)
            return self.xg, self.yg, self.bearing + sign * s, 0.0, sign * w
        return self.xg, self.yg, self.thetag, 0.0, 0.0


def make_plan(x0: float, y0: float, theta0: float,
              xg: float, yg: float, thetag: float) -> Plan:
    dx, dy = xg - x0, yg - y0
    dist = float(np.hypot(dx, dy))

    if dist < DIST_EPS:
        # Goal position already reached (or within noise) — a single in-place
        # rotation to the final heading is the whole plan.
        turn = float(_wrap(thetag - theta0))
        T1, _ = _trapezoid_duration_and_peak(abs(turn), MAX_ANGULAR, MAX_ANGULAR_ACC)
        return Plan(x0, y0, theta0, xg, yg, thetag, bearing=theta0,
                    turn1=turn, turn2=0.0, dist=0.0, T1=T1, T2=0.0, T3=0.0)

    bearing = float(np.arctan2(dy, dx))
    turn1 = float(_wrap(bearing - theta0))
    if abs(turn1) < TURN_EPS:
        turn1 = 0.0
    turn2 = float(_wrap(thetag - bearing))
    if abs(turn2) < TURN_EPS:
        turn2 = 0.0

    T1, _ = _trapezoid_duration_and_peak(abs(turn1), MAX_ANGULAR, MAX_ANGULAR_ACC)
    T2, _ = _trapezoid_duration_and_peak(dist, MAX_LINEAR, MAX_LINEAR_ACC)
    T3, _ = _trapezoid_duration_and_peak(abs(turn2), MAX_ANGULAR, MAX_ANGULAR_ACC)
    return Plan(x0, y0, theta0, xg, yg, thetag, bearing, turn1, turn2, dist, T1, T2, T3)


# ---------------------------------------------------------------------------
# MPC stage
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Weights:
    q_xy: float
    q_theta: float
    qf_xy: float
    qf_theta: float
    r_v: float
    r_omega: float
    rd_v: float
    rd_omega: float


DEFAULT_WEIGHTS = Weights(Q_XY, Q_THETA, QF_XY, QF_THETA, R_V, R_OMEGA, RD_V, RD_OMEGA)


def _rate_constraint(n: int, u_prev: np.ndarray, dt: float,
                      max_linear_acc: float, max_angular_acc: float) -> LinearConstraint:
    """Per-step |u_k - u_{k-1}| <= max_acc*dt, expressed as a LinearConstraint
    over the flat decision vector [v_0..v_{n-1}, w_0..w_{n-1}]."""
    a = np.zeros((2 * n, 2 * n))
    lb = np.zeros(2 * n)
    ub = np.zeros(2 * n)
    dv = max_linear_acc * dt
    dw = max_angular_acc * dt

    a[0, 0] = 1.0
    lb[0], ub[0] = u_prev[0] - dv, u_prev[0] + dv
    for k in range(1, n):
        a[k, k], a[k, k - 1] = 1.0, -1.0
        lb[k], ub[k] = -dv, dv

    a[n, n] = 1.0
    lb[n], ub[n] = u_prev[1] - dw, u_prev[1] + dw
    for k in range(1, n):
        a[n + k, n + k], a[n + k, n + k - 1] = 1.0, -1.0
        lb[n + k], ub[n + k] = -dw, dw

    return LinearConstraint(a, lb, ub)


def _mpc_cost(u_flat: np.ndarray, x0: float, y0: float, theta0: float,
              ref_xr: np.ndarray, ref_yr: np.ndarray, ref_thr: np.ndarray,
              u_prev: np.ndarray, dt: float, weights: Weights) -> float:
    n = len(u_flat) // 2
    v, w = u_flat[:n], u_flat[n:]
    xs, ys, ths = _rollout(x0, y0, theta0, v, w, dt)

    pos_err2 = (xs[:n] - ref_xr[:n]) ** 2 + (ys[:n] - ref_yr[:n]) ** 2
    theta_err = _wrap(ths[:n] - ref_thr[:n])
    cost = weights.q_xy * np.sum(pos_err2) + weights.q_theta * np.sum(theta_err ** 2)
    cost += weights.r_v * np.sum(v ** 2) + weights.r_omega * np.sum(w ** 2)

    dv = np.diff(np.concatenate(([u_prev[0]], v)))
    dw = np.diff(np.concatenate(([u_prev[1]], w)))
    cost += weights.rd_v * np.sum(dv ** 2) + weights.rd_omega * np.sum(dw ** 2)

    term_pos = (xs[n] - ref_xr[n]) ** 2 + (ys[n] - ref_yr[n]) ** 2
    term_theta = _wrap(ths[n] - ref_thr[n]) ** 2
    cost += weights.qf_xy * term_pos + weights.qf_theta * term_theta
    return float(cost)


def plan_warm_start(plan: Plan, t0: float, n: int, dt: float) -> np.ndarray:
    v = np.empty(n)
    w = np.empty(n)
    for k in range(n):
        _, _, _, vr, wr = plan.eval(t0 + k * dt)
        v[k] = np.clip(vr, 0.0, MAX_LINEAR)
        w[k] = np.clip(wr, -MAX_ANGULAR, MAX_ANGULAR)
    return np.concatenate([v, w])


def shift_warm_start(u_opt: np.ndarray, n: int) -> np.ndarray:
    v, w = u_opt[:n], u_opt[n:]
    v_next = np.concatenate([v[1:], v[-1:]])
    w_next = np.concatenate([w[1:], w[-1:]])
    return np.concatenate([v_next, w_next])


def solve_mpc(x: float, y: float, theta: float,
              ref_xyth: tuple[np.ndarray, np.ndarray, np.ndarray],
              u_prev: np.ndarray, dt: float, n: int,
              warm_start: np.ndarray | None, weights: Weights = DEFAULT_WEIGHTS,
              max_linear: float = MAX_LINEAR, max_angular: float = MAX_ANGULAR,
              max_linear_acc: float = MAX_LINEAR_ACC, max_angular_acc: float = MAX_ANGULAR_ACC,
              maxiter: int = SLSQP_MAXITER):
    """Single-shooting nonlinear MPC solve. Returns (v0, w0, success, U_opt)."""
    ref_xr, ref_yr, ref_thr = ref_xyth
    u0 = warm_start if warm_start is not None else np.zeros(2 * n)
    bounds = [(0.0, max_linear)] * n + [(-max_angular, max_angular)] * n
    constraint = _rate_constraint(n, u_prev, dt, max_linear_acc, max_angular_acc)

    res = minimize(
        _mpc_cost, u0,
        args=(x, y, theta, ref_xr, ref_yr, ref_thr, u_prev, dt, weights),
        method="SLSQP", bounds=bounds, constraints=[constraint],
        options={"maxiter": maxiter, "ftol": 1e-6},
    )
    u_opt = res.x
    return float(u_opt[0]), float(u_opt[n]), bool(res.success), u_opt


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class MPCNavNode(Node):
    def __init__(self):
        super().__init__("qr_mpc_nav_node")

        d_map = json.loads(TAG_MAPPOSE_PATH.read_text())
        d_target = json.loads(TARGET_PATH.read_text())
        T_map_qrworld = np.array(d_map["T_map_qrworld"])
        T_qrworld_robot = np.array(d_target["T_qrworld_robot"])
        T_map_goal = T_map_qrworld @ T_qrworld_robot

        self._xg = float(T_map_goal[0, 3])
        self._yg = float(T_map_goal[1, 3])
        self._thetag = float(np.arctan2(T_map_goal[1, 0], T_map_goal[0, 0]))

        self._T_world_camera: np.ndarray | None = None
        self._plan: Plan | None = None
        self._t_plan = 0.0
        self._u_prev = np.zeros(2)
        self._warm_start: np.ndarray | None = None
        self._horizon_n = HORIZON_N
        self._maxiter = SLSQP_MAXITER
        self._replan_bad_ticks = 0
        self._converged_ticks = 0
        self._reached = False
        self._last_tick_time: float | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 100)
        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self._nav_done_pub = self.create_publisher(Bool, NAV_DONE_TOPIC, 10)
        self.create_timer(DT_MPC, self._control_tick)

        self.get_logger().info(
            f"qr_mpc_nav_node: goal fixed in map frame "
            f"(x={self._xg:.3f}, y={self._yg:.3f}, theta={np.degrees(self._thetag):.1f}deg), "
            f"control @ {CONTROL_HZ:.0f}Hz"
        )

    # ---- subscribers / TF ----

    def _odom_cb(self, msg: Odometry) -> None:
        self._T_world_camera, _ = msg2np(msg)

    def _lookup_T_world_map(self) -> np.ndarray | None:
        try:
            tf_msg = self._tf_buffer.lookup_transform("world", "map", rclpy.time.Time())
        except Exception:
            self.get_logger().warn("TF world→map not available", throttle_duration_sec=2.0)
            return None
        _, _, T_world_map = tf2np(tf_msg)
        return T_world_map

    def _current_pose(self):
        if self._T_world_camera is None:
            return None
        T_world_map = self._lookup_T_world_map()
        if T_world_map is None:
            return None
        T_map_robot = np.linalg.inv(T_world_map) @ self._T_world_camera @ T_CAMERA_ROBOT
        x = float(T_map_robot[0, 3])
        y = float(T_map_robot[1, 3])
        theta = float(np.arctan2(T_map_robot[1, 0], T_map_robot[0, 0]))
        return x, y, theta

    def _tick_dt(self) -> float:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_tick_time is None:
            self._last_tick_time = now
            return DT_MPC
        dt = now - self._last_tick_time
        self._last_tick_time = now
        return dt if dt > 0.0 else DT_MPC

    def _publish_reached_once(self) -> None:
        if self._reached:
            return
        self._reached = True
        self._nav_done_pub.publish(Bool(data=True))
        self.get_logger().info("qr target reached (MPC).")

    def _fallback_cmd(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Emergency pure-P controller, used only if SLSQP fails and there is
        no previous solution to fall back on (e.g. the very first tick)."""
        dx, dy = self._xg - x, self._yg - y
        dist = float(np.hypot(dx, dy))
        bearing_err = float(_wrap(np.arctan2(dy, dx) - theta))
        heading_err = float(_wrap(self._thetag - theta))
        if dist > DIST_THRESH:
            omega = _clip_with_min(1.0 * bearing_err, MIN_ANGULAR, MAX_ANGULAR) \
                if abs(bearing_err) > HEADING_THRESH else 0.0
            v = _clip_with_min(0.5 * dist, MIN_LINEAR, MAX_LINEAR) \
                if abs(bearing_err) < 0.5 else 0.0
            return v, omega
        if abs(heading_err) > HEADING_THRESH:
            return 0.0, _clip_with_min(1.0 * heading_err, MIN_ANGULAR, MAX_ANGULAR)
        return 0.0, 0.0

    def _sample_reference(self, t0: float):
        ref_xr = np.empty(self._horizon_n + 1)
        ref_yr = np.empty(self._horizon_n + 1)
        ref_thr = np.empty(self._horizon_n + 1)
        for k in range(self._horizon_n + 1):
            xr, yr, thr, _, _ = self._plan.eval(t0 + k * DT_MPC)
            ref_xr[k], ref_yr[k], ref_thr[k] = xr, yr, thr
        return ref_xr, ref_yr, ref_thr

    def _replan(self, x: float, y: float, theta: float) -> None:
        self._plan = make_plan(x, y, theta, self._xg, self._yg, self._thetag)
        self._t_plan = 0.0
        self._warm_start = plan_warm_start(self._plan, 0.0, self._horizon_n, DT_MPC)
        self._replan_bad_ticks = 0

    # ---- control loop ----

    def _control_tick(self) -> None:
        pose = self._current_pose()
        if pose is None:
            return
        x, y, theta = pose
        dt = self._tick_dt()

        if self._reached:
            self._cmd_pub.publish(Twist())
            return

        if self._plan is None:
            self._replan(x, y, theta)

        ref_xr, ref_yr, ref_thr = self._sample_reference(self._t_plan)
        pos_err0 = float(np.hypot(x - ref_xr[0], y - ref_yr[0]))
        head_err0 = float(abs(_wrap(theta - ref_thr[0])))
        self._replan_bad_ticks = self._replan_bad_ticks + 1 \
            if (pos_err0 > REPLAN_DIST or head_err0 > REPLAN_HEADING) else 0
        timed_out = self._t_plan > self._plan.total_duration + REPLAN_TIMEOUT_SLACK_S

        if self._replan_bad_ticks >= REPLAN_DEBOUNCE_TICKS or timed_out:
            self.get_logger().warn(
                f"replanning (pos_err={pos_err0:.3f}m, head_err={np.degrees(head_err0):.1f}deg, "
                f"timed_out={timed_out})", throttle_duration_sec=1.0)
            self._replan(x, y, theta)
            ref_xr, ref_yr, ref_thr = self._sample_reference(self._t_plan)

        t_solve0 = time.monotonic()
        v0, w0, success, u_opt = solve_mpc(
            x, y, theta, (ref_xr, ref_yr, ref_thr), self._u_prev, DT_MPC,
            self._horizon_n, self._warm_start, maxiter=self._maxiter)
        solve_dt = time.monotonic() - t_solve0

        if solve_dt > SOLVE_TIME_BUDGET_S and self._horizon_n > HORIZON_N_DEGRADED:
            self.get_logger().warn(
                f"MPC solve took {solve_dt * 1000:.0f}ms, degrading horizon "
                f"{self._horizon_n}->{HORIZON_N_DEGRADED}", throttle_duration_sec=2.0)
            self._horizon_n = HORIZON_N_DEGRADED
            self._maxiter = SLSQP_MAXITER_DEGRADED
            self._warm_start = None

        if success:
            self._warm_start = shift_warm_start(u_opt, self._horizon_n)
        elif self._warm_start is not None:
            self.get_logger().warn("MPC solve failed, reusing previous command",
                                   throttle_duration_sec=1.0)
            v0, w0 = float(self._u_prev[0]), float(self._u_prev[1])
        else:
            self.get_logger().warn("MPC solve failed on first tick, falling back to P control",
                                   throttle_duration_sec=1.0)
            v0, w0 = self._fallback_cmd(x, y, theta)

        self._u_prev = np.array([v0, w0])
        self._t_plan += dt

        dist_to_goal = float(np.hypot(x - self._xg, y - self._yg))
        head_to_goal = float(abs(_wrap(theta - self._thetag)))
        self._converged_ticks = self._converged_ticks + 1 \
            if (dist_to_goal < DIST_THRESH and head_to_goal < HEADING_THRESH) else 0

        if self._converged_ticks >= CONVERGE_TICKS:
            self._publish_reached_once()
            self._cmd_pub.publish(Twist())
            return

        cmd = Twist()
        cmd.linear.x = v0
        cmd.angular.z = w0
        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNavNode()
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
