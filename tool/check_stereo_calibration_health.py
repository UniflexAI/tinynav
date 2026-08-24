#!/usr/bin/env python3
"""Live stereo sensor-health check for the Looper camera.

Plug the Looper in, launch its driver so /camera/camera/infra{1,2}/image_rect_raw
are publishing, then run this node directly against that live ROS graph (no bag
needed) to triage a suspected-bad-sensor: for every synced left/right frame pair
it checks two independent failure modes:

  1. Blur: a Hessian-energy sharpness score per image (mean(Ixx^2 + 2*Ixy^2 +
     Iyy^2) from the Sobel second derivatives). A defocused/motion-blurred frame
     has small second derivatives everywhere, so its score drops well below an
     in-focus frame's -- alarm when either side's score falls below
     --blur-threshold. Blurry frames are skipped for the epipolar check below
     since their SuperPoint keypoints/matches aren't reliable.
  2. Epipolar (row) alignment: run SuperPoint+LightGlue (same TRT engines as
     perception_node.py/map_node.py) on the left/right pair and look at each
     match's row offset. For a correctly rectified pair every match should land
     on ~the same row; a systematic non-zero offset means the pair isn't
     row-aligned, which points at wrong/stale rectification or intrinsics
     rather than matcher noise.

  3. Left/color extrinsic drift: detect the same fiducial tag in the infra1
     (left) and color images and solve its pose in each camera frame
     independently (planar-square PnP, using each camera's own live
     CameraInfo). The two poses imply a left->color transform; comparing that
     against the transform the driver actually publishes on /tf_static (looked
     up live, not hardcoded, since the point is to catch a *this specific
     unit's* stale/wrong extrinsic) tells you whether the published
     left->color calibration matches what the cameras actually see. A single
     tag reading is not trustworthy on its own -- background clutter can
     misdecode as a valid tag -- so samples are gated (low reprojection error
     on both sides, plausible distance, and left/color agreeing on that
     distance since the two sensors are only ~cm apart) and aggregated
     (median) over --tag-min-samples before a verdict is logged.

Every frame's scores are logged (throttled) so thresholds can be calibrated
against a known-good sensor before trusting the alarms on a suspect one; a
running summary of alarm counts is logged every --summary-every frames.

Usage:
    uv run python tool/check_stereo_calibration_health.py
    uv run python tool/check_stereo_calibration_health.py --save-vis-dir /tmp/stereo_health
"""

import argparse
import asyncio
import os
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import Buffer, ExtrapolationException, LookupException, TransformListener

from tinynav.core.models_trt import LightGlueTRT, SuperPointTRT, save_matching_visualization

DEFAULT_LEFT_TOPIC = "/camera/camera/infra1/image_rect_raw"
DEFAULT_RIGHT_TOPIC = "/camera/camera/infra2/image_rect_raw"
DEFAULT_COLOR_TOPIC = "/camera/camera/color/image_rect_raw/compressed"
DEFAULT_LEFT_INFO_TOPIC = "/camera/camera/infra1/camera_info"
DEFAULT_COLOR_INFO_TOPIC = "/camera/camera/color/camera_info"


def hessian_sharpness_score(gray: np.ndarray) -> float:
    """No-reference sharpness score from the image's Hessian (second-derivative)
    energy. Higher = sharper; near-zero everywhere means the image is blurry.
    """
    gray = gray.astype(np.float32)
    ixx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=3)
    iyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=3)
    ixy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    return float(np.mean(ixx**2 + 2.0 * ixy**2 + iyy**2))


def epipolar_row_offsets(kpts0: np.ndarray, kpts1: np.ndarray, match_indices: np.ndarray) -> np.ndarray:
    """Per-match row offset (right_row - left_row) for valid SuperPoint+LightGlue
    matches. Should be ~0 everywhere for a properly rectified stereo pair.
    """
    valid_mask = match_indices != -1
    matched0 = kpts0[valid_mask]
    matched1 = kpts1[match_indices[valid_mask]]
    return matched1[:, 1] - matched0[:, 1]


@dataclass
class EpipolarCheck:
    ok: bool
    num_matches: int
    median_abs_dy: float
    outlier_fraction: float


def check_epipolar_alignment(dy: np.ndarray, max_row_offset_px: float, max_outlier_fraction: float) -> EpipolarCheck:
    if dy.size == 0:
        return EpipolarCheck(ok=False, num_matches=0, median_abs_dy=float("nan"), outlier_fraction=1.0)
    abs_dy = np.abs(dy)
    median_abs_dy = float(np.median(abs_dy))
    outlier_fraction = float(np.mean(abs_dy > max_row_offset_px))
    ok = median_abs_dy <= max_row_offset_px and outlier_fraction <= max_outlier_fraction
    return EpipolarCheck(ok=ok, num_matches=int(dy.size), median_abs_dy=median_abs_dy, outlier_fraction=outlier_fraction)


# Empirically the actual family printed on this board (an "A500 6x6" tag
# sheet, 5.5cm markers / 1.65cm spacing) -- a sweep of every OpenCV-predefined
# dictionary (including the standard AprilTag ones) against a real photo of
# the board only got clean, high-count matches from this one.
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_ARUCO_MIP_36H12


def build_aruco_detector(dictionary_id: int = ARUCO_DICTIONARY_ID) -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 0.5
    params.polygonalApproxAccuracyRate = 0.05
    # CORNER_REFINE_SUBPIX is iterative and has a known pathological-slow/non-converging
    # failure mode on certain degenerate candidate quads (observed: a single call hanging
    # for minutes on a real bag, burning CPU with no way to time out from the Python side
    # since the hang is inside the C++ call). CONTOUR is a single non-iterative pass --
    # slightly less precise corners, but detection/ID-decoding doesn't depend on it and it
    # can't hang.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    return cv2.aruco.ArucoDetector(dictionary, params)


def is_reasonable_quad(pts: np.ndarray, min_area_px2: float = 4.0) -> bool:
    """Cheap sanity check before handing 4 corners to solvePnP. IPPE_SQUARE has
    a pathological-slow/hanging failure mode on near-degenerate
    (collinear/self-intersecting/near-zero-area) quads -- reject those up
    front instead of ever calling into it.
    """
    quad = pts.astype(np.float32).reshape(4, 1, 2)
    if cv2.contourArea(quad) < min_area_px2:
        return False
    return bool(cv2.isContourConvex(cv2.convexHull(quad)))


def solve_best_marker_pose(img_pts: np.ndarray, marker_length_m: float, K: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Pose (4x4, marker->camera) of a single square tag of known side length,
    picking whichever of IPPE_SQUARE's two candidate solutions reprojects best.
    Single-square planar PnP has a well-known near-180-degree flip ambiguity:
    at typical calibration-target distances the two candidate solutions can
    reproject almost equally well, so "pick the lower error" alone silently
    coin-flips between two very different poses. Also returns the runner-up's
    error so callers can reject cases where the two weren't cleanly separated.
    Returns (identity, inf, inf) if the quad is degenerate or solvePnP fails --
    callers already reject on a large error, this just avoids ever calling
    into IPPE_SQUARE with input that can hang it.
    """
    obj = np.array([
        [-marker_length_m / 2, marker_length_m / 2, 0],
        [marker_length_m / 2, marker_length_m / 2, 0],
        [marker_length_m / 2, -marker_length_m / 2, 0],
        [-marker_length_m / 2, -marker_length_m / 2, 0],
    ], dtype=np.float32)
    if not is_reasonable_quad(img_pts):
        return np.eye(4), float("inf"), float("inf")
    try:
        _n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return np.eye(4), float("inf"), float("inf")
    flat_errs = sorted(float(np.asarray(e).flatten()[0]) for e in errs)
    best = int(np.argmin([float(np.asarray(e).flatten()[0]) for e in errs]))
    R, _ = cv2.Rodrigues(rvecs[best])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvecs[best].flatten()
    err_best = flat_errs[0]
    err_runner_up = flat_errs[1] if len(flat_errs) > 1 else float("inf")
    return T, err_best, err_runner_up


def transform_delta(T_sample: np.ndarray, T_reference: np.ndarray) -> tuple[float, float]:
    """Translation (m) and rotation-angle (deg) between two 4x4 transforms."""
    t_delta = float(np.linalg.norm(T_sample[:3, 3] - T_reference[:3, 3]))
    R_delta = T_sample[:3, :3] @ T_reference[:3, :3].T
    rvec_delta, _ = cv2.Rodrigues(R_delta)
    angle_delta_deg = float(np.degrees(np.linalg.norm(rvec_delta)))
    return t_delta, angle_delta_deg


@dataclass
class ExtrinsicSample:
    marker_id: int
    translation_delta_m: float
    rotation_delta_deg: float


def gate_marker_pair(
    T_left: np.ndarray, err_left: float, err_left_runner_up: float,
    T_color: np.ndarray, err_color: float, err_color_runner_up: float,
    max_reproj_err_px: float, min_distance_m: float, max_distance_m: float,
    max_distance_disagreement_frac: float, min_ambiguity_margin_px: float,
) -> bool:
    """Reject a candidate cross-camera marker match before it's allowed to
    influence the extrinsic estimate. A single low reprojection error does not
    mean the detection is real -- background clutter can misdecode into a
    geometrically-consistent-looking square -- so this also requires the
    implied distance to be plausible for a hand-held calibration target and,
    since infra1 and color are only ~cm apart, requires both cameras to agree
    on that distance (a false pairing of two unrelated clutter objects that
    happen to decode to the same id essentially never agrees on distance).
    It also requires each side's chosen IPPE_SQUARE solution to clearly beat
    its runner-up -- a near-tie means the well-known planar-square flip
    ambiguity could have picked the wrong (~180-degree-off) pose.
    """
    if err_left > max_reproj_err_px or err_color > max_reproj_err_px:
        return False
    if (err_left_runner_up - err_left) < min_ambiguity_margin_px or (err_color_runner_up - err_color) < min_ambiguity_margin_px:
        return False
    dist_left = float(np.linalg.norm(T_left[:3, 3]))
    dist_color = float(np.linalg.norm(T_color[:3, 3]))
    if not (min_distance_m <= dist_left <= max_distance_m and min_distance_m <= dist_color <= max_distance_m):
        return False
    disagreement = abs(dist_left - dist_color) / max(dist_left, dist_color)
    return disagreement <= max_distance_disagreement_frac


def quat_xyzw_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])


class StereoCalibrationHealthNode(Node):
    def __init__(self, args):
        super().__init__("check_stereo_calibration_health")
        self.args = args
        self.bridge = CvBridge()
        self.superpoint = SuperPointTRT()
        self.light_glue = LightGlueTRT()

        self.num_frames = 0
        self.num_blur_alarms = 0
        self.num_epipolar_alarms = 0

        if args.save_vis_dir:
            os.makedirs(args.save_vis_dir, exist_ok=True)

        self.left_sub = Subscriber(self, Image, args.left_topic)
        self.right_sub = Subscriber(self, Image, args.right_topic)
        self.sync = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=10, slop=args.sync_slop_s)
        self.sync.registerCallback(self.sync_callback)

        # Extrinsic (left<->color) tag-based cross-check.
        self.aruco_detector = build_aruco_detector()
        self.left_K = None
        self.left_D = None
        self.left_frame_id = None
        self.color_K = None
        self.color_D = None
        self.color_frame_id = None
        self.create_subscription(CameraInfo, args.left_camera_info_topic, self.left_info_callback, 10)
        self.create_subscription(CameraInfo, args.color_camera_info_topic, self.color_info_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.color_sub = Subscriber(self, CompressedImage, args.color_topic)
        self.extrinsic_sync = ApproximateTimeSynchronizer([self.left_sub, self.color_sub], queue_size=10, slop=args.extrinsic_sync_slop_s)
        self.extrinsic_sync.registerCallback(self.extrinsic_sync_callback)
        self.extrinsic_samples = deque(maxlen=max(args.tag_min_samples * 10, 100))
        self.num_extrinsic_frames = 0
        self.num_common_ids_seen = 0
        self.num_gated_samples = 0
        self.last_verdict_at_count = 0

        self.get_logger().info(f"Watching {args.left_topic} + {args.right_topic} for blur + epipolar-alignment alarms.")
        self.get_logger().info(f"Watching {args.left_topic} + {args.color_topic} (tag id={ARUCO_DICTIONARY_ID}) for left/color extrinsic drift.")

    def left_info_callback(self, msg: CameraInfo):
        if self.left_K is None:
            self.left_K = np.array(msg.k).reshape(3, 3)
            self.left_D = np.array(msg.d)
            self.left_frame_id = msg.header.frame_id
            self.get_logger().info(f"Got infra1 CameraInfo (frame={self.left_frame_id}).", once=True)

    def color_info_callback(self, msg: CameraInfo):
        if self.color_K is None:
            self.color_K = np.array(msg.k).reshape(3, 3)
            self.color_D = np.array(msg.d)
            self.color_frame_id = msg.header.frame_id
            self.get_logger().info(f"Got color CameraInfo (frame={self.color_frame_id}).", once=True)

    def sync_callback(self, left_msg: Image, right_msg: Image):
        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="mono8")
        right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="mono8")
        stamp_s = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        self.num_frames += 1

        sharpness_left = hessian_sharpness_score(left)
        sharpness_right = hessian_sharpness_score(right)
        if sharpness_left < self.args.blur_threshold or sharpness_right < self.args.blur_threshold:
            self.num_blur_alarms += 1
            self.get_logger().warn(
                f"[BLUR ALARM] t={stamp_s:.3f} sharp_L={sharpness_left:.1f} sharp_R={sharpness_right:.1f} "
                f"(threshold={self.args.blur_threshold}) -- skipping epipolar check for this frame"
            )
            self.maybe_log_summary()
            return

        feats0 = asyncio.run(self.superpoint.infer(left))
        feats1 = asyncio.run(self.superpoint.infer(right))
        image_shape = np.array([left.shape[1], left.shape[0]], dtype=np.int64)
        match_result = asyncio.run(self.light_glue.infer(
            feats0["kpts"], feats1["kpts"], feats0["descps"], feats1["descps"],
            feats0["mask"], feats1["mask"], image_shape, image_shape,
        ))
        match_indices = match_result["match_indices"][0]
        dy = epipolar_row_offsets(feats0["kpts"][0], feats1["kpts"][0], match_indices)
        check = check_epipolar_alignment(dy, self.args.max_row_offset_px, self.args.max_outlier_fraction)

        if check.ok:
            self.get_logger().info(
                f"[OK] t={stamp_s:.3f} sharp_L={sharpness_left:.1f} sharp_R={sharpness_right:.1f} "
                f"matches={check.num_matches} median|dy|={check.median_abs_dy:.2f}px",
                throttle_duration_sec=2.0,
            )
        else:
            self.num_epipolar_alarms += 1
            self.get_logger().error(
                f"[EPIPOLAR ALARM] t={stamp_s:.3f} matches={check.num_matches} "
                f"median|dy|={check.median_abs_dy:.2f}px outlier_frac={check.outlier_fraction:.2f} "
                f"-- left/right matches are not row-aligned, suspect bad rectification/intrinsics"
            )
            if self.args.save_vis_dir:
                save_matching_visualization(
                    left, right, feats0["kpts"], feats1["kpts"], match_result,
                    output_path=f"{self.args.save_vis_dir}/frame_{self.num_frames:06d}_t{stamp_s:.3f}.jpg",
                )

        self.maybe_log_summary()

    def maybe_log_summary(self):
        if self.num_frames % self.args.summary_every == 0:
            self.get_logger().info(
                f"[summary] frames={self.num_frames} blur_alarms={self.num_blur_alarms} epipolar_alarms={self.num_epipolar_alarms}"
            )

    def extrinsic_sync_callback(self, left_msg: Image, color_msg: CompressedImage):
        if self.left_K is None or self.color_K is None:
            self.get_logger().info("Waiting for infra1 + color CameraInfo before checking left/color extrinsic...", throttle_duration_sec=5.0)
            return

        left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="mono8")
        color = cv2.imdecode(np.frombuffer(color_msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        color_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        self.num_extrinsic_frames += 1
        try:
            left_corners, left_ids, _ = self.aruco_detector.detectMarkers(left)
            color_corners, color_ids, _ = self.aruco_detector.detectMarkers(color_gray)
        except cv2.error:
            # A degenerate candidate quad in this specific frame can make OpenCV's aruco
            # detector throw (CORNER_REFINE_CONTOUR trades the SUBPIX hang risk for this
            # rarer but real failure mode) -- skip just this one frame.
            self.maybe_log_extrinsic_debug()
            return
        if left_ids is None or color_ids is None:
            self.maybe_log_extrinsic_debug()
            return

        left_by_id = {mid: c for c, mid in zip(left_corners, left_ids.flatten().tolist())}
        color_by_id = {mid: c for c, mid in zip(color_corners, color_ids.flatten().tolist())}
        common_ids = set(left_by_id) & set(color_by_id)
        self.num_common_ids_seen += len(common_ids)

        for marker_id in common_ids:
            T_left, err_left, err_left_ru = solve_best_marker_pose(left_by_id[marker_id].reshape(4, 2), self.args.marker_length_m, self.left_K, self.left_D)
            T_color, err_color, err_color_ru = solve_best_marker_pose(color_by_id[marker_id].reshape(4, 2), self.args.marker_length_m, self.color_K, self.color_D)
            if not gate_marker_pair(
                T_left, err_left, err_left_ru, T_color, err_color, err_color_ru,
                self.args.tag_max_reproj_err_px, self.args.tag_min_distance_m, self.args.tag_max_distance_m,
                self.args.tag_max_distance_disagreement_frac, self.args.tag_min_ambiguity_margin_px,
            ):
                continue

            T_left_to_color_observed = T_color @ np.linalg.inv(T_left)
            try:
                tf_msg = self.tf_buffer.lookup_transform(self.color_frame_id, self.left_frame_id, Time())
            except (LookupException, ExtrapolationException):
                self.get_logger().info(f"No /tf_static {self.left_frame_id} -> {self.color_frame_id} yet, can't compare.", throttle_duration_sec=5.0)
                continue
            tr = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            T_published = np.eye(4)
            T_published[:3, :3] = quat_xyzw_to_R(q.x, q.y, q.z, q.w)
            T_published[:3, 3] = [tr.x, tr.y, tr.z]

            t_delta_m, angle_delta_deg = transform_delta(T_left_to_color_observed, T_published)
            self.extrinsic_samples.append(ExtrinsicSample(marker_id, t_delta_m, angle_delta_deg))
            self.num_gated_samples += 1
            self.get_logger().info(
                f"[tag sample] id={marker_id} err_left={err_left:.3f}px(margin={err_left_ru-err_left:.3f}) "
                f"err_color={err_color:.3f}px(margin={err_color_ru-err_color:.3f}) "
                f"t_delta={t_delta_m:.4f}m angle_delta={angle_delta_deg:.2f}deg"
            )

        self.maybe_log_extrinsic_debug()
        # num_gated_samples keeps counting past the deque's maxlen; compare
        # against the count at the last verdict (not a modulo on the raw
        # count) so this fires exactly once per new batch, not every callback
        # while the count sits idle on a multiple.
        if self.num_gated_samples >= self.last_verdict_at_count + self.args.tag_min_samples:
            self.last_verdict_at_count = self.num_gated_samples
            self.log_extrinsic_verdict()

    def maybe_log_extrinsic_debug(self):
        if self.num_extrinsic_frames % self.args.summary_every == 0:
            self.get_logger().info(
                f"[tag summary] synced left/color frames={self.num_extrinsic_frames} "
                f"raw common ids seen={self.num_common_ids_seen} gated samples kept={self.num_gated_samples}"
            )

    def log_extrinsic_verdict(self):
        t_deltas = np.array([s.translation_delta_m for s in self.extrinsic_samples])
        angle_deltas = np.array([s.rotation_delta_deg for s in self.extrinsic_samples])
        median_t = float(np.median(t_deltas))
        median_angle = float(np.median(angle_deltas))
        ok = median_t <= self.args.tag_max_translation_delta_m and median_angle <= self.args.tag_max_rotation_delta_deg
        level = self.get_logger().info if ok else self.get_logger().error
        tag = "OK" if ok else "EXTRINSIC ALARM"
        level(
            f"[{tag}] left/color extrinsic vs published /tf_static, over {len(self.extrinsic_samples)} gated tag samples: "
            f"median translation delta={median_t:.4f}m (std={float(np.std(t_deltas)):.4f}), "
            f"median rotation delta={median_angle:.2f}deg (std={float(np.std(angle_deltas)):.2f}) -- "
            f"large spread means the tag detections themselves are too noisy (background clutter) to trust this verdict yet"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-topic", default=DEFAULT_LEFT_TOPIC)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT_TOPIC)
    parser.add_argument("--sync-slop-s", type=float, default=0.02)
    parser.add_argument("--blur-threshold", type=float, default=30.0, help="Hessian-energy score below this on either side triggers a blur alarm; tune against a known-good sensor first")
    parser.add_argument("--max-row-offset-px", type=float, default=2.0, help="Median |dy| above this triggers an epipolar alarm")
    parser.add_argument("--max-outlier-fraction", type=float, default=0.2, help="Fraction of matches with |dy| > --max-row-offset-px above this triggers an epipolar alarm")
    parser.add_argument("--summary-every", type=int, default=30, help="Log a running alarm-count summary every N frames")
    parser.add_argument("--save-vis-dir", default=None, help="If set, save a left/right match visualization for every frame that fails the epipolar check")

    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--left-camera-info-topic", default=DEFAULT_LEFT_INFO_TOPIC)
    parser.add_argument("--color-camera-info-topic", default=DEFAULT_COLOR_INFO_TOPIC)
    parser.add_argument("--extrinsic-sync-slop-s", type=float, default=0.05)
    parser.add_argument("--marker-length-m", type=float, default=0.055, help="Physical side length (m) of one tag square on the calibration board")
    parser.add_argument("--tag-max-reproj-err-px", type=float, default=0.5, help="Per-camera solvePnP reprojection error gate for a tag sample to count")
    parser.add_argument("--tag-min-distance-m", type=float, default=0.2, help="Reject tag samples implying a camera-to-tag distance below this (likely a misdetection)")
    parser.add_argument("--tag-max-distance-m", type=float, default=3.0, help="Reject tag samples implying a camera-to-tag distance above this (likely background clutter misdecoded as a tag)")
    parser.add_argument("--tag-max-distance-disagreement-frac", type=float, default=0.15, help="Reject a tag sample if left/color distance-to-tag estimates disagree by more than this fraction")
    parser.add_argument("--tag-min-ambiguity-margin-px", type=float, default=0.2, help="Reject a tag sample unless each side's best IPPE_SQUARE solution beats its runner-up by at least this many px of reprojection error (guards against the planar-square pose-flip ambiguity). This margin is not a clean predictor at small values -- two samples with near-identical margins around 0.2-0.4px have been observed to give one correct (~2deg from published TF) and one wildly wrong (~110deg) result, so treat [tag sample] log lines as data to review, not a fully-solved filter")
    parser.add_argument("--tag-min-samples", type=int, default=10, help="Minimum gated tag samples before logging an extrinsic verdict; also the recompute interval")
    parser.add_argument("--tag-max-translation-delta-m", type=float, default=0.02, help="Median translation delta (m) vs published /tf_static above this triggers an extrinsic alarm")
    parser.add_argument("--tag-max-rotation-delta-deg", type=float, default=5.0, help="Median rotation delta (deg) vs published /tf_static above this triggers an extrinsic alarm")
    return parser.parse_args()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCalibrationHealthNode(parse_args())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"[final] frames={node.num_frames} blur_alarms={node.num_blur_alarms} epipolar_alarms={node.num_epipolar_alarms} "
            f"gated_tag_samples={len(node.extrinsic_samples)}"
        )
        if node.extrinsic_samples:
            node.log_extrinsic_verdict()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
