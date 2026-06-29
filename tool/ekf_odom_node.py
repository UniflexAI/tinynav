#!/usr/bin/env python3
"""
tool/ekf_odom_node.py  —  Error-state EKF, quaternion orientation.

Formulation
-----------
  Nominal state  x  = [p(3), v(3), q(4)]   10-dim; q is a unit quaternion [x,y,z,w]
  Error state   δx  = [δp(3), δv(3), δθ(3)] 9-dim; δθ is a rotation-vector perturbation
  Covariance     P  : 9×9  (lives in error-state space)

  True state  =  nominal  ⊕  error:
    p_true  = p + δp
    v_true  = v + δv
    q_true  = q ⊗ Exp(δθ)      Exp(δθ) ≈ [1, δθ/2]ᵀ  for small δθ

Predict (SLAM odometry delta → T_delta)
-------
  Nominal propagation  (no noise):
    p' = p + R(q) @ Δp
    v' = R(q) @ Δp / dt
    q' = q ⊗ Δq    (renormalised)

  Error-state Jacobian F (9×9):
    F[δp, δθ] = -R(q) @ skew(Δp)
    F[δv, δθ] = -R(q) @ skew(Δp) / dt
    F[δθ, δθ] = ΔR.T
    (all other blocks: identity for self, zero otherwise)

  Covariance propagation:
    P' = F @ P @ F.T + Q

Update (pose6 or pos3 measurement)
-------
  Innovation in error-state space:
    δp_innov = p_meas - p_nom               (position)
    δθ_innov = (q_nom⁻¹ ⊗ q_meas).rotvec  (orientation; short-path enforced)

  H (6×9): δp rows → identity at cols 0:3; δθ rows → identity at cols 6:9
  H (3×9): δp rows → identity at cols 0:3  (RTK position-only)

  Joseph-form update for numerical stability:
    K     = P @ H.T @ inv(S)
    dx    = K @ innov
    IKH   = I - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T

  Correction injection:
    p += dx[0:3],  v += dx[3:6],  q = q ⊗ Exp(dx[6:9])  (renormalised)

All inputs must be  frame_id="world", child_frame_id="camera"  (T_world_camera).

Topics
------
  Predict:  /slam/odometry       → ekf_predict
  Update:   /wheel/odom_camera   → ekf_update_pose6  (6-DOF)
            /lidar/odom_camera   → ekf_update_pose6  (6-DOF)
            /qr/odom             → ekf_update_pose6  (6-DOF)
            /rtk/odom_camera     → ekf_update_pos3   (3-DOF, pos only)
  Output:   /slam/odometry_fused
"""

from __future__ import annotations

import dataclasses

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from tinynav.core.math_utils import msg2np, np2msg

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Process noise Q — error-state order: [δp(3), δv(3), δθ(3)]
Q_DIAG = np.array([
    0.01,  0.01,  0.005,   # δp   m²
    0.10,  0.10,  0.050,   # δv   (m/s)²
    0.005, 0.005, 0.010,   # δθ   rad²
], dtype=np.float64)

# Measurement noise R — order: [pos(3), angle(3)]
R_WHEEL = np.diag([0.030, 0.030, 0.010,  0.005, 0.005, 0.030])
R_LIDAR = np.diag([0.020, 0.020, 0.010,  0.005, 0.005, 0.020])
R_QR    = np.diag([0.005, 0.005, 0.005,  0.002, 0.002, 0.005])
R_RTK   = np.diag([0.010, 0.010, 0.040])   # position-only 3×3

GATE: dict[str, float] = {
    'wheel': 12.0,
    'lidar': 16.0,
    'qr':    10.0,
    'rtk':   16.0,
}

# Initial error-state covariance P0
P0_DIAG = np.array([
    1.0, 1.0, 0.5,    # δp
    0.5, 0.5, 0.2,    # δv
    0.1, 0.1, 0.2,    # δθ
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Observation matrices  (error-state space)
# ---------------------------------------------------------------------------

# Full 6-DOF: [δp_innov(3), δθ_innov(3)] observed from error state [δp, δv, δθ]
_H6 = np.zeros((6, 9))
_H6[0:3, 0:3] = np.eye(3)   # position
_H6[3:6, 6:9] = np.eye(3)   # orientation (δθ)

# Position-only (RTK)
_H3 = np.zeros((3, 9))
_H3[0:3, 0:3] = np.eye(3)


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NominalState:
    p: np.ndarray   # (3,)  position in world frame
    v: np.ndarray   # (3,)  velocity in world frame
    q: np.ndarray   # (4,)  quaternion [x,y,z,w]  (scipy convention)


@dataclasses.dataclass
class EKFState:
    nom:   NominalState
    P:     np.ndarray   # (9,9) error-state covariance
    stamp: float        # seconds


# ---------------------------------------------------------------------------
# Pure math utilities
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0 ],
    ])


def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply, scipy [x,y,z,w] convention."""
    return (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat()


def _qinv(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q).inv().as_quat()


def _Rmat(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q).as_matrix()


# ---------------------------------------------------------------------------
# EKF pure functions
# ---------------------------------------------------------------------------

def _predict_nominal(nom: NominalState, T_delta: np.ndarray,
                     dt: float) -> NominalState:
    Rq  = _Rmat(nom.q)
    dp  = T_delta[:3, 3]
    dq  = Rotation.from_matrix(T_delta[:3, :3]).as_quat()

    p_new = nom.p + Rq @ dp
    v_new = Rq @ dp / max(dt, 1e-4)
    q_new = _qmul(nom.q, dq)
    q_new = q_new / np.linalg.norm(q_new)
    return NominalState(p=p_new, v=v_new, q=q_new)


def _predict_F(nom: NominalState, T_delta: np.ndarray, dt: float) -> np.ndarray:
    """9×9 error-state Jacobian for the predict step."""
    Rq = _Rmat(nom.q)
    dp = T_delta[:3, 3]
    dR = T_delta[:3, :3]

    F = np.zeros((9, 9))
    F[0:3, 0:3] = np.eye(3)                          # δp  → δp
    F[0:3, 6:9] = -Rq @ _skew(dp)                    # δθ  → δp
    F[3:6, 6:9] = -Rq @ _skew(dp) / max(dt, 1e-4)   # δθ  → δv
    F[6:9, 6:9] = dR.T                               # δθ  → δθ
    return F


def ekf_predict(state: EKFState, T_delta: np.ndarray,
                Q: np.ndarray, dt: float) -> EKFState:
    nom_new = _predict_nominal(state.nom, T_delta, dt)
    F       = _predict_F(state.nom, T_delta, dt)
    P_new   = F @ state.P @ F.T + Q
    return EKFState(nom=nom_new, P=P_new, stamp=state.stamp + dt)


def _apply_correction(nom: NominalState, dx: np.ndarray) -> NominalState:
    """Inject 9-dim error-state correction into nominal state."""
    p_new = nom.p + dx[0:3]
    v_new = nom.v + dx[3:6]
    dq    = Rotation.from_rotvec(dx[6:9]).as_quat()
    q_new = _qmul(nom.q, dq)
    q_new = q_new / np.linalg.norm(q_new)
    return NominalState(p=p_new, v=v_new, q=q_new)


def _ekf_update(state: EKFState, innov: np.ndarray, H: np.ndarray,
                R_noise: np.ndarray, gate: float) -> tuple[EKFState, bool]:
    S  = H @ state.P @ H.T + R_noise
    d2 = float(innov @ np.linalg.solve(S, innov))
    if d2 > gate:
        return state, False

    K   = state.P @ H.T @ np.linalg.inv(S)
    dx  = K @ innov
    IKH = np.eye(9) - K @ H
    # Joseph form: numerically stable even when K is imprecise
    P_new   = IKH @ state.P @ IKH.T + K @ R_noise @ K.T
    nom_new = _apply_correction(state.nom, dx)
    return EKFState(nom=nom_new, P=P_new, stamp=state.stamp), True


def ekf_update_pose6(state: EKFState, T_meas: np.ndarray,
                     R_noise: np.ndarray, gate: float) -> tuple[EKFState, bool]:
    """6-DOF update from a 4×4 SE3 measurement (T_world_camera)."""
    dp = T_meas[:3, 3] - state.nom.p

    q_meas = Rotation.from_matrix(T_meas[:3, :3]).as_quat()
    q_err  = _qmul(_qinv(state.nom.q), q_meas)
    if q_err[3] < 0:   # enforce short-path (scalar part ≥ 0)
        q_err = -q_err
    dtheta = Rotation.from_quat(q_err).as_rotvec()

    innov = np.concatenate([dp, dtheta])
    return _ekf_update(state, innov, _H6, R_noise, gate)


def ekf_update_pos3(state: EKFState, p_meas: np.ndarray,
                    R_noise: np.ndarray, gate: float) -> tuple[EKFState, bool]:
    """Position-only update (e.g. RTK GPS)."""
    innov = p_meas - state.nom.p
    return _ekf_update(state, innov, _H3, R_noise, gate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _T_from_nominal(nom: NominalState) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _Rmat(nom.q)
    T[:3,  3] = nom.p
    return T


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class EKFOdomNode(Node):
    def __init__(self):
        super().__init__('ekf_odom_node')

        self._state: EKFState | None = None
        self._Q = np.diag(Q_DIAG)

        self._last_slam_T:     np.ndarray | None = None
        self._last_slam_stamp: float | None      = None

        self.create_subscription(
            Odometry, '/slam/odometry',     self._slam_cb,   100)
        self.create_subscription(
            Odometry, '/wheel/odom_camera', self._wheel_cb,  100)
        self.create_subscription(
            Odometry, '/lidar/odom_camera', self._lidar_cb,   50)
        self.create_subscription(
            Odometry, '/qr/odom',           self._qr_cb,      10)
        self.create_subscription(
            Odometry, '/rtk/odom_camera',   self._rtk_cb,     10)

        self._pub = self.create_publisher(Odometry, '/slam/odometry_fused', 10)
        self.get_logger().info(
            'ekf_odom_node ready  [error-state EKF, quaternion orientation]'
        )

    # ---- callbacks ----

    def _slam_cb(self, msg: Odometry) -> None:
        T, _  = msg2np(msg)
        stamp = _stamp_to_sec(msg.header.stamp)

        if self._last_slam_T is None:
            self._last_slam_T     = T
            self._last_slam_stamp = stamp
            if self._state is None:
                self._init(T, stamp)
            return

        dt = stamp - self._last_slam_stamp
        if dt <= 0.0:
            return

        T_delta               = np.linalg.inv(self._last_slam_T) @ T
        self._last_slam_T     = T
        self._last_slam_stamp = stamp

        if self._state is None:
            self._init(T, stamp)
            return

        self._state = ekf_predict(self._state, T_delta, self._Q, dt)
        self._publish(msg.header.stamp)

    def _wheel_cb(self, msg: Odometry) -> None:
        self._update_pose6(msg, R_WHEEL, GATE['wheel'], 'wheel')

    def _lidar_cb(self, msg: Odometry) -> None:
        self._update_pose6(msg, R_LIDAR, GATE['lidar'], 'lidar')

    def _qr_cb(self, msg: Odometry) -> None:
        self._update_pose6(msg, R_QR, GATE['qr'], 'qr')

    def _rtk_cb(self, msg: Odometry) -> None:
        T, _ = msg2np(msg)
        stamp = _stamp_to_sec(msg.header.stamp)
        if self._state is None:
            self._init(T, stamp)
            return
        self._state, ok = ekf_update_pos3(
            self._state, T[:3, 3], R_RTK, GATE['rtk'])
        if not ok:
            self.get_logger().warn('rtk: outlier rejected',
                                   throttle_duration_sec=1.0)
        else:
            self._publish(msg.header.stamp)

    # ---- helpers ----

    def _update_pose6(self, msg: Odometry, R_noise: np.ndarray,
                      gate: float, source: str) -> None:
        T, _ = msg2np(msg)
        if self._state is None:
            self._init(T, _stamp_to_sec(msg.header.stamp))
            return
        self._state, ok = ekf_update_pose6(self._state, T, R_noise, gate)
        if not ok:
            self.get_logger().warn(
                f'[{source}] outlier rejected', throttle_duration_sec=1.0)
        else:
            self._publish(msg.header.stamp)

    def _init(self, T: np.ndarray, stamp: float) -> None:
        q   = Rotation.from_matrix(T[:3, :3]).as_quat()
        nom = NominalState(p=T[:3, 3].copy(), v=np.zeros(3), q=q)
        self._state = EKFState(nom=nom, P=np.diag(P0_DIAG), stamp=stamp)
        self.get_logger().info('EKF state initialized.')

    def _publish(self, stamp) -> None:
        if self._state is None:
            return
        T = _T_from_nominal(self._state.nom)
        self._pub.publish(
            np2msg(T, stamp, 'world', 'camera',
                   velocity=self._state.nom.v))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = EKFOdomNode()
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
