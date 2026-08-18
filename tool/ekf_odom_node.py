#!/usr/bin/env python3
"""
Topics
------
  Predict:  /slam/odometry            → ekf_predict  (20Hz)  → /slam/odometry_fused
            /slam/odometry_100hz      → ekf_predict  (100Hz) → /slam/odometry_fused_100hz
  Update:   /robot/odom, /robot_odom  → ekf_update_pose6  (6-DOF, delta-referenced, huge-noise z)
            /wheel/odom_camera        → ekf_update_pose6  (6-DOF, delta-referenced)
            /lidar/odom               → ekf_update_pose6  (6-DOF, delta-referenced)
            /qr/odom                  → ekf_update_pose6  (6-DOF, absolute)
            /rtk/map_pose             → ekf_update_pos3   (3-DOF, pos only, absolute)
            (all updates publish to both /slam/odometry_fused and _100hz)

Always running (wired into scripts/start_app.sh), but map_node only consumes its
output when live-toggled: node_manager's 'VIO'/'EKF' operate-tab button publishes
{"odom_source": "vio"|"ekf"} on /localization/config, which map_node reads to pick
between /slam/odometry (default) and /slam/odometry_fused for continuous_odom_callback.

Ported from xinghan/qrcode for the beijing/A2 rig, which has no wheel-odometry/QR/robot_odom
publishers -- those subscriptions are left in place (harmless no-ops here) so this stays a
drop-in upgrade if any of those sources shows up later, exactly as on the reference branch.
Two real adaptations:

- RTK: beijing has no /rtk/odom_camera at all. The nearby /rtk/odom publishes in the
  receiver's own raw ENU frame, which is NOT the VIO/predict source's world frame, so feeding
  it straight into ekf_update_pos3 (which assumes the measurement is already in that same
  world frame) would silently corrupt the filter. /rtk/map_pose is the one already aligned
  into map/world frame (via rtk_align.json, see map_node.py's own RTK-replace logic), so
  that's the correct topic to subscribe to here.
- Lidar: subscribes to /lidar/odom (super_lio, ~/workspace/super-lio on the A2's onboard PC;
  not published by anything in this repo) instead of the reference branch's
  /lidar/odom_camera, conjugated through T_LIDAR_CAM since it reports poses/deltas in the
  lidar's own sensor frame. The A2's onboard hesai_ros_driver/lidar firmware was renamed from
  /lidar_points+/lidar_imu to /lidar/points+/lidar/imu; super_lio's hesai.yaml and this
  node's subscriptions were updated to match, confirmed flowing live on the A2 board
  (/lidar/points @10Hz, /lidar/imu @200Hz, /lidar/odom @10Hz).
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from tinynav.core.math_utils import msg2np, np2msg

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Process noise Q — error-state order: [δp(3), δv(3), δθ(3)]
Q_DIAG = np.array([
    0.01,  0.00,  0.00,     # δp   m²
    0.00,  0.10,  0.00,     # δv   (m/s)²
    0.00,  0.00,  0.0010,   # δθ   rad²
], dtype=np.float64)

# Measurement noise R.
# Robot odom now feeds the full 6-DOF pose6 update (all of x/y/z/roll/pitch/yaw),
# but a ground-robot base's z estimate is unreliable, so its observation noise
# is set very large — the Kalman gain on that row collapses to ~0, so z is
# effectively left to the other sources while x/y/roll/pitch/yaw still update.
R_ROBOT = np.diag([0.030, 0.030, 1.0e6,  0.005, 0.005, 0.030])
R_WHEEL = np.diag([0.030, 0.030, 0.010,  0.005, 0.005, 0.030])
R_LIDAR = np.diag([0.020, 0.020, 0.010,  0.005, 0.005, 0.020])
R_QR    = np.diag([0.005, 0.005, 0.005,  0.002, 0.002, 0.005])
R_RTK   = np.diag([0.010, 0.010, 0.040])   # position-only 3×3

GATE: dict[str, float] = {
    'robot': 12.0,
    'wheel': 12.0,
    'lidar': 16.0,
    'qr':    10.0,
    'rtk':   16.0,
}

ROBOT_ODOM_TOPICS = tuple(
    topic.strip()
    for topic in os.environ.get('EKF_ROBOT_ODOM_TOPICS', '/robot/odom,/robot_odom').split(',')
    if topic.strip()
)

# base_link -> camera extrinsic, hand-eye calibrated (AX=XB) from
# map_record_20260716_144806 (/robot_odom vs /camera/camera/vio_image);
# see tinynav_temp/calibrate_base_cam.py. /robot_odom reports T_odom_base, not
# T_world_camera like the other update sources — without this conjugation,
# base_link yaw leaks into apparent camera roll/pitch every time the robot
# turns (pre-calibration angular-rate mismatch ~38 deg/s mean vs VIO's own
# ~10 deg/s; post-calibration predicted-vs-actual residual ~0.4 deg mean).
T_BASE_CAM = np.array([
    [ 0.03345557, -0.02317757,  0.99917142,  0.27686093],
    [-0.99929625, -0.0177435 ,  0.03304816,  0.04623703],
    [ 0.01696282, -0.99957389, -0.02375488, -0.00515408],
    [ 0.0,         0.0,         0.0,         1.0        ],
])

# lidar -> camera extrinsic, same one planning_node.py's lidar_sync_callback already uses to
# raycast /lidar/points into the depth-camera-aligned occupancy grid (see T_lidar_to_cam
# there): Unitree's factory lidar->base_link extrinsic composed with the fixed camera<->body
# rotation and this rig's B2_CONFIG camera mount offset (camera_x=0.3), then the
# planning_node-documented on-robot-measured override for the translation. /lidar/odom (from
# super_lio, subscribed below) reports poses/deltas in the lidar's own sensor frame, not
# camera frame -- without this conjugation the lidar's own rotation axes leak into the wrong
# nominal-state axes, exactly like the T_BASE_CAM case above.
T_LIDAR_CAM = np.array([
    [-1.0,  0.0,  0.0,  0.0 ],
    [ 0.0, -1.0,  0.0,  0.07],
    [ 0.0,  0.0,  1.0, -0.02],
    [ 0.0,  0.0,  0.0,  1.0 ],
])

# Reject a SLAM predict step outright if the per-axis frame-to-frame delta
# exceeds this (m) — guards against VIO glitches/jumps corrupting predict,
# which has no Mahalanobis gate of its own (predict has no innovation/S).
SLAM_PREDICT_POS_GATE_M = 0.5

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


def _sec_to_stamp(t: float) -> TimeMsg:
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    return TimeMsg(sec=sec, nanosec=nanosec)


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


        self._last_robot_raw: np.ndarray | None = None
        self._last_robot_nom: np.ndarray | None = None
        self._last_robot_stamp: float | None = None
        self._last_wheel_raw: np.ndarray | None = None
        self._last_wheel_nom: np.ndarray | None = None
        self._last_lidar_raw: np.ndarray | None = None
        self._last_lidar_nom: np.ndarray | None = None

        self.create_subscription(
            Odometry, '/slam/odometry',        self._slam_20hz_cb,  100)
        self.create_subscription(
            Odometry, '/slam/odometry_100hz',  self._slam_100hz_cb, 200)
        for topic in ROBOT_ODOM_TOPICS:
            self.create_subscription(Odometry, topic, self._robot_cb, 100)
        self.create_subscription(
            Odometry, '/wheel/odom_camera', self._wheel_cb,  100)
        self.create_subscription(
            Odometry, '/lidar/odom',        self._lidar_cb,   50)
        self.create_subscription(
            Odometry, '/qr/odom',           self._qr_cb,      10)
        self.create_subscription(
            Odometry, '/rtk/map_pose',      self._rtk_cb,     10)

        self._pub        = self.create_publisher(Odometry, '/slam/odometry_fused',       10)
        self._pub_100hz  = self.create_publisher(Odometry, '/slam/odometry_fused_100hz', 10)
        self.get_logger().info(
            'ekf_odom_node ready  [error-state EKF, quaternion orientation]; '
            f'robot odom topics={ROBOT_ODOM_TOPICS}'
        )

    # ---- callbacks ----

    def _slam_20hz_cb(self, msg: Odometry) -> None:
        self._slam_predict(msg, 'slam-20hz', self._pub)

    def _slam_100hz_cb(self, msg: Odometry) -> None:
        self._slam_predict(msg, 'slam-100hz', self._pub_100hz)

    def _slam_predict(self, msg: Odometry, source: str, pub) -> None:
        """Predict step shared by both VIO rates. /slam/odometry (20Hz, optimized/
        corrected) and /slam/odometry_100hz (100Hz, IMU-propagated) are the same
        underlying trajectory at different rates/refinement"""
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

        dp = T_delta[:3, 3]
        if np.any(np.abs(dp) > SLAM_PREDICT_POS_GATE_M):
            self.get_logger().warn(
                f'[{source}] predict delta outlier rejected: dp={dp.tolist()} '
                f'(> {SLAM_PREDICT_POS_GATE_M} m per axis)', throttle_duration_sec=1.0)
            return

        self._state = ekf_predict(self._state, T_delta, self._Q, dt)
        self._publish(pub)

    def _robot_cb(self, msg: Odometry) -> None:
        stamp = _stamp_to_sec(msg.header.stamp)
        if stamp <= 0.0:
            self.get_logger().warn(
                '[robot] odom with zero timestamp ignored',
                throttle_duration_sec=1.0,
            )
            return
        if self._last_robot_stamp is not None and stamp <= self._last_robot_stamp:
            if stamp < self._last_robot_stamp:
                self.get_logger().warn(
                    '[robot] odom timestamp moved backwards; reset delta reference',
                    throttle_duration_sec=1.0,
                )
                self._last_robot_raw = None
                self._last_robot_nom = None
                self._last_robot_stamp = stamp
            return
        self._last_robot_stamp = stamp
        self._update_pose6_delta(
            msg, R_ROBOT, GATE['robot'], 'robot', '_last_robot_raw', '_last_robot_nom',
            extrinsic=T_BASE_CAM)

    def _wheel_cb(self, msg: Odometry) -> None:
        self._update_pose6_delta(
            msg, R_WHEEL, GATE['wheel'], 'wheel', '_last_wheel_raw', '_last_wheel_nom')

    def _lidar_cb(self, msg: Odometry) -> None:
        self._update_pose6_delta(
            msg, R_LIDAR, GATE['lidar'], 'lidar', '_last_lidar_raw', '_last_lidar_nom',
            extrinsic=T_LIDAR_CAM)

    def _qr_cb(self, msg: Odometry) -> None:
        self._update_pose6(msg, R_QR, GATE['qr'], 'qr')

    def _rtk_cb(self, msg: Odometry) -> None:
        return 
        # T, _ = msg2np(msg)
        # stamp = _stamp_to_sec(msg.header.stamp)
        # if self._state is None:
        #     self._init(T, stamp)
        #     return
        # self._state, ok = ekf_update_pos3(
        #     self._state, T[:3, 3], R_RTK, GATE['rtk'])
        # if not ok:
        #     self.get_logger().warn('rtk: outlier rejected',
        #                            throttle_duration_sec=1.0)
        # else:
        #     self._publish()

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
            self._publish()

    def _update_pose6_delta(self, msg: Odometry, R_noise: np.ndarray, gate: float,
                            source: str, raw_attr: str, nom_attr: str,
                            extrinsic: np.ndarray | None = None) -> None:
        """Update using only the motion delta since this source's last reading,
        re-anchored onto the fused nominal pose at that time. Avoids trusting
        the source's own absolute origin/heading, which need not agree with
        the EKF's world frame.

        extrinsic, if given, is T_base_cam: this source publishes in its own
        rigid-body frame (e.g. base_link), not the camera frame the nominal
        state lives in, so its raw pose/delta is conjugated into camera frame
        before use — otherwise the source's own rotation axes leak into the
        wrong nominal-state axes (e.g. base_link yaw appearing as camera
        roll/pitch whenever the robot turns)."""
        T_raw = msg2np(msg)[0]
        if self._state is None:
            T_init = T_raw @ extrinsic if extrinsic is not None else T_raw
            self._init(T_init, _stamp_to_sec(msg.header.stamp))
            setattr(self, raw_attr, T_raw)
            setattr(self, nom_attr, _T_from_nominal(self._state.nom))
            return

        last_raw = getattr(self, raw_attr)
        if last_raw is None:
            setattr(self, raw_attr, T_raw)
            setattr(self, nom_attr, _T_from_nominal(self._state.nom))
            return

        T_delta = np.linalg.inv(last_raw) @ T_raw
        if extrinsic is not None:
            T_delta = np.linalg.inv(extrinsic) @ T_delta @ extrinsic
        T_meas  = getattr(self, nom_attr) @ T_delta

        self._state, ok = ekf_update_pose6(self._state, T_meas, R_noise, gate)
        if not ok:
            self.get_logger().warn(
                f'[{source}] outlier rejected', throttle_duration_sec=1.0)
        else:
            self._publish()

        setattr(self, raw_attr, T_raw)
        setattr(self, nom_attr, _T_from_nominal(self._state.nom))

    def _init(self, T: np.ndarray, stamp: float) -> None:
        q   = Rotation.from_matrix(T[:3, :3]).as_quat()
        nom = NominalState(p=T[:3, 3].copy(), v=np.zeros(3), q=q)
        self._state = EKFState(nom=nom, P=np.diag(P0_DIAG), stamp=stamp)
        self.get_logger().info('EKF state initialized.')

    def _publish(self, pub=None) -> None:
        """pub selects which output topic(s) get this update: a single publisher
        (e.g. a predict step, which only publishes to its own triggering topic's
        rate-matched output) or, by default, both — since robot/wheel/lidar/qr/rtk
        corrections apply to the one shared EKF state, both external outputs
        should reflect them immediately rather than waiting for their own next
        predict tick."""
        if self._state is None:
            return
        pubs = (self._pub, self._pub_100hz) if pub is None else (pub,)
        T = _T_from_nominal(self._state.nom)
        # Always stamp with the EKF's own internal clock (self._state.stamp), which only
        # advances on predict and is left untouched by updates — never forward an input
        # message's own header.stamp here. Update sources (robot/wheel/lidar/qr/rtk) can run
        # on a different clock than the predict source (e.g. wall-clock vs a VIO-relative
        # clock); forwarding whichever source triggered this publish mixed both clocks onto
        # this one topic, making header.stamp jump backwards/forwards on ~1 in 6 messages.
        stamp = _sec_to_stamp(self._state.stamp)
        msg = np2msg(T, stamp, 'world', 'camera', velocity=self._state.nom.v)
        for p in pubs:
            p.publish(msg)


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
