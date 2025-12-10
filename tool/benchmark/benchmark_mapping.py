import os
import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import rosbag2_py
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from reportlab.platypus import (
    SimpleDocTemplate,
    Image,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from launch import LaunchService, LaunchDescription

from launch.actions import ExecuteProcess, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from tabulate import tabulate


class BagMetadataExtractor:
    """
    Utility to extract metadata from ROS2 bags, particularly timing information
    for timestamp-based pose sampling.
    """

    @staticmethod
    def get_bag_time_range(bag_path: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract start and end timestamps from a ROS2 bag.

        Args:
            bag_path: Path to the ROS2 bag directory

        Returns:
            Tuple of (start_time_ns, end_time_ns) or (None, None) if failed
        """
        try:
            info = rosbag2_py.Info()
            metadata = info.read_metadata(bag_path, "")

            start_ns = metadata.starting_time.nanoseconds
            end_ns = start_ns + metadata.duration.nanoseconds

            print(
                f"Bag time range: {start_ns} to {end_ns} ns ({(end_ns - start_ns) / 1e9:.1f} seconds)"
            )
            return start_ns, end_ns

        except Exception as e:
            print(f"Failed to extract bag metadata from {bag_path}: {e}")
            return None, None

    @staticmethod
    def sample_timestamps_evenly(
        start_ns: int, end_ns: int, num_samples: int
    ) -> np.ndarray:
        """
        Generate evenly spaced timestamps between start and end.

        Args:
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            num_samples: Number of timestamps to generate

        Returns:
            Array of evenly spaced timestamps in nanoseconds
        """
        if start_ns >= end_ns:
            print(f"Invalid time range: start={start_ns}, end={end_ns}")
            return np.array([], dtype=np.int64)

        if num_samples <= 0:
            print(f"Invalid number of samples: {num_samples}")
            return np.array([], dtype=np.int64)

        # Generate evenly spaced timestamps
        timestamps = np.linspace(start_ns, end_ns, num_samples, dtype=np.int64)

        print(f"Generated {len(timestamps)} evenly spaced timestamps")
        print(f"Time range: {(end_ns - start_ns) / 1e9:.1f} seconds")
        print(f"Sample interval: {(timestamps[1] - timestamps[0]) / 1e9:.3f} seconds")

        return timestamps


class PoseQueryEngine:
    """
    Query system for timestamp-based pose lookup with SLERP interpolation for rotations.
    """

    def __init__(self):
        self.continuous_poses: Dict[int, np.ndarray] = (
            {}
        )  # timestamp_ns -> 4x4 pose matrix

    def load_continuous_poses(self, poses_file: str) -> bool:
        """
        Load continuous poses from a saved file.

        Args:
            poses_file: Path to the continuous poses .npy file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            loaded_poses = np.load(poses_file, allow_pickle=True).item()
            self.continuous_poses.update(loaded_poses)
            print(f"Loaded {len(loaded_poses)} continuous poses from {poses_file}")
            return True
        except Exception as e:
            print(f"Failed to load continuous poses from {poses_file}: {e}")
            return False

    def find_closest_poses(
        self, target_timestamp: int, poses_dict: Dict[int, np.ndarray]
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], Optional[Tuple[int, np.ndarray]]]:
        """
        Find the two closest poses (before and after) to the target timestamp.

        Args:
            target_timestamp: Target timestamp in nanoseconds
            poses_dict: Dictionary of timestamp -> pose matrix

        Returns:
            Tuple of (before_pose, after_pose) where each is (timestamp, pose_matrix) or None
        """
        if not poses_dict:
            return None, None

        timestamps = sorted(poses_dict.keys())

        # Find insertion point
        idx = np.searchsorted(timestamps, target_timestamp)

        before_pose = None
        after_pose = None

        if idx > 0:
            before_ts = timestamps[idx - 1]
            before_pose = (before_ts, poses_dict[before_ts])

        if idx < len(timestamps):
            after_ts = timestamps[idx]
            after_pose = (after_ts, poses_dict[after_ts])

        return before_pose, after_pose

    def interpolate_pose(
        self,
        timestamp1: int,
        pose1: np.ndarray,
        timestamp2: int,
        pose2: np.ndarray,
        target_timestamp: int,
    ) -> np.ndarray:
        """
        Interpolate pose at target timestamp using SLERP for rotation and linear interpolation for translation.

        Args:
            timestamp1: First timestamp in nanoseconds
            pose1: First 4x4 pose matrix
            timestamp2: Second timestamp in nanoseconds
            pose2: Second 4x4 pose matrix
            target_timestamp: Target timestamp in nanoseconds

        Returns:
            Interpolated 4x4 pose matrix
        """
        if timestamp1 == timestamp2:
            return pose1.copy()

        # Calculate interpolation factor
        t = (target_timestamp - timestamp1) / (timestamp2 - timestamp1)
        t = np.clip(t, 0.0, 1.0)

        # Extract rotations and translations
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        t1 = pose1[:3, 3]
        t2 = pose2[:3, 3]

        # SLERP for rotation
        try:
            # Create SLERP interpolator
            key_times = [0, 1]
            key_rots = R.from_matrix([R1, R2])
            slerp = Slerp(key_times, key_rots)

            # Interpolate rotation
            interpolated_rot = slerp(t)
            interpolated_R = interpolated_rot.as_matrix()

        except Exception as e:
            print(f"SLERP failed, using linear interpolation for rotation: {e}")
            # Fallback to linear interpolation for rotation matrix
            interpolated_R = (1 - t) * R1 + t * R2
            # Re-orthogonalize using SVD
            U, _, Vt = np.linalg.svd(interpolated_R)
            interpolated_R = U @ Vt

        # Linear interpolation for translation
        interpolated_t = (1 - t) * t1 + t * t2

        # Construct interpolated pose
        interpolated_pose = np.eye(4)
        interpolated_pose[:3, :3] = interpolated_R
        interpolated_pose[:3, 3] = interpolated_t

        return interpolated_pose

    def query_pose_at_timestamp(
        self, target_timestamp: int, max_time_diff_ns: int = 1000000000
    ) -> Optional[np.ndarray]:
        """
        Query pose at a specific timestamp with interpolation.

        Args:
            target_timestamp: Target timestamp in nanoseconds
            max_time_diff_ns: Maximum time difference for interpolation (default 1 second)

        Returns:
            4x4 pose matrix at target timestamp, or None if not found
        """
        if not self.continuous_poses:
            return None

        before_pose, after_pose = self.find_closest_poses(
            target_timestamp, self.continuous_poses
        )

        # Check if we can interpolate
        if before_pose and after_pose:
            before_ts, before_matrix = before_pose
            after_ts, after_matrix = after_pose

            # Check time difference constraints
            if (after_ts - before_ts) <= max_time_diff_ns:
                return self.interpolate_pose(
                    before_ts, before_matrix, after_ts, after_matrix, target_timestamp
                )

        # Try exact match or closest pose
        if before_pose:
            before_ts, before_matrix = before_pose
            if abs(target_timestamp - before_ts) <= max_time_diff_ns:
                return before_matrix.copy()

        if after_pose:
            after_ts, after_matrix = after_pose
            if abs(target_timestamp - after_ts) <= max_time_diff_ns:
                return after_matrix.copy()

        print(f"No pose found for timestamp {target_timestamp}")
        return None


# FIXME(yuance): Update database path
TINYNAV_DB = "tinynav_db"


def generate_launch_description_localization(
    bag_path: str,
    tinynav_db_path: str,
    tinynav_map_path: str,
    rate: float,
    data_saving_timeout: float,
    task_name: str,
    verbose_timer: bool,
):
    perception_cmd = [
        "python3",
        "/tinynav/tinynav/core/perception_node.py",
        "--log_file",
        f"{tinynav_db_path}/perception.log",
    ]
    if not verbose_timer:
        perception_cmd.append("--no_verbose_timer")

    perception = ExecuteProcess(
        cmd=perception_cmd,
        name=f"{task_name}_perception",
        output="screen",
    )

    localization_cmd = [
        "python3",
        "/tinynav/tinynav/core/map_node.py",
        "--tinynav_db_path",
        str(tinynav_db_path),
        "--tinynav_map_path",
        str(tinynav_map_path),
    ]
    if not verbose_timer:
        localization_cmd.append("--no_verbose_timer")

    localization = ExecuteProcess(
        cmd=localization_cmd,
        name=f"{task_name}_localization",
        output="screen",
    )
    bag_play = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "--rate",
            str(rate),
            "--clock",
        ],  # no --loop so it will exit at EOF
        output="screen",
    )
    coordinator = ExecuteProcess(
        cmd=[
            "python3",
            "/tinynav/tool/benchmark/data_saving_coordinator.py",
            str(data_saving_timeout),
        ],
        name=f"{task_name}_coordinator",
        output="screen",
    )

    # When rosbag play exits, trigger the coordinator and then shutdown
    on_bag_exit = RegisterEventHandler(
        OnProcessExit(target_action=bag_play, on_exit=[coordinator])
    )

    # Shutdown everything when coordinator finishes
    on_coordinator_exit = RegisterEventHandler(
        OnProcessExit(target_action=coordinator, on_exit=[EmitEvent(event=Shutdown())])
    )

    return LaunchDescription(
        [perception, localization, bag_play, on_bag_exit, on_coordinator_exit]
    )


def generate_launch_description_mapping(
    bag_path: str,
    map_save_path: str,
    rate: float,
    data_saving_timeout: float,
    task_name: str,
    verbose_timer: bool,
):
    perception_cmd = [
        "python3",
        "/tinynav/tinynav/core/perception_node.py",
        "--log_file",
        f"{map_save_path}/perception.log",
    ]
    if not verbose_timer:
        perception_cmd.append("--no_verbose_timer")

    perception = ExecuteProcess(
        cmd=perception_cmd,
        name=f"{task_name}_perception",
        output="screen",
    )

    mapping_cmd = [
        "python3",
        "/tinynav/tinynav/core/build_map_node.py",
        "--map_save_path",
        str(map_save_path),
        "--bag_file",
        str(bag_path),
        "--tinynav_temp_path",
        f"{map_save_path}/tinynav_temp",
    ]
    if not verbose_timer:
        mapping_cmd.append("--no_verbose_timer")

    mapping = ExecuteProcess(
        cmd=mapping_cmd,
        name=f"{task_name}_mapping",
        output="screen",
    )

    # Shutdown everything when mapping finishes
    on_mapping_exit = RegisterEventHandler(
        OnProcessExit(target_action=mapping, on_exit=[EmitEvent(event=Shutdown())])
    )

    return LaunchDescription(
        [perception, mapping, on_mapping_exit]
    )


class BenchmarkResults:
    """Container for benchmark results and metrics."""

    def __init__(self):
        self.total_poses = 0
        self.successful_localizations = 0
        self.localization_poses = {}  # timestamp -> pose from localizing B in map A
        self.ground_truth_poses = {}  # timestamp -> pose from map B
        self.transformation_matrix = np.eye(4)
        self.translation_errors = []
        self.rotation_errors = []
        self.precision_stats = {
            "high": {"threshold_trans": 0.05, "threshold_rot": 2.0, "count": 0},
            "medium": {"threshold_trans": 0.10, "threshold_rot": 5.0, "count": 0},
            "low": {"threshold_trans": 0.30, "threshold_rot": 10.0, "count": 0},
        }

    def add_pose_pair(
        self,
        timestamp: int,
        localization_pose: np.ndarray,
        ground_truth_pose: np.ndarray,
    ):
        self.localization_poses[timestamp] = localization_pose
        self.ground_truth_poses[timestamp] = ground_truth_pose
        self.total_poses += 1

    def add_failed_localization(self, timestamps: List[int]):
        self.total_poses += len(timestamps)

    # TODO(yuance): Make estimation based on 6DoF instead of translation only
    def compute_transformation(self) -> bool:
        """Estimate rigid transformation between coordinate systems using RANSAC."""
        if len(self.localization_poses) < 3:
            print("Error: Need at least 3 pose pairs for transformation estimation")
            return False

        points_a = []  # from localization in map A
        points_b = []  # from ground truth in map B

        for timestamp in self.localization_poses:
            if timestamp in self.ground_truth_poses:
                points_a.append(self.localization_poses[timestamp][:3, 3])
                points_b.append(self.ground_truth_poses[timestamp][:3, 3])

        points_a = np.array(points_a)
        points_b = np.array(points_b)

        if len(points_a) < 3:
            print("Error: Insufficient corresponding points for transformation")
            return False

        # Use RANSAC to find best rigid transformation
        best_inliers = 0
        best_transformation = np.eye(4)
        max_iterations = 1000
        inlier_threshold = 0.20  # 20cm threshold for inliers

        for _ in range(max_iterations):
            if len(points_a) < 3:
                break

            indices = np.random.choice(len(points_a), 3, replace=False)
            sample_a = points_a[indices]
            sample_b = points_b[indices]

            T = self._estimate_rigid_transform(sample_a, sample_b)
            if T is None:
                continue

            # Count inliers
            transformed_points = self._transform_points(points_a, T)
            distances = np.linalg.norm(transformed_points - points_b, axis=1)
            inliers = np.sum(distances < inlier_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_transformation = T

        # Refine with all inliers
        transformed_points = self._transform_points(points_a, best_transformation)
        distances = np.linalg.norm(transformed_points - points_b, axis=1)
        inlier_mask = distances < inlier_threshold

        if np.sum(inlier_mask) >= 3:
            refined_T = self._estimate_rigid_transform(
                points_a[inlier_mask], points_b[inlier_mask]
            )
            if refined_T is not None:
                best_transformation = refined_T

        self.transformation_matrix = best_transformation
        print(f"Estimated transformation with {best_inliers}/{len(points_a)} inliers")
        return True

    def _estimate_rigid_transform(
        self, points_a: np.ndarray, points_b: np.ndarray
    ) -> Optional[np.ndarray]:
        if len(points_a) != len(points_b) or len(points_a) < 3:
            return None

        centroid_a = np.mean(points_a, axis=0)
        centroid_b = np.mean(points_b, axis=0)

        centered_a = points_a - centroid_a
        centered_b = points_b - centroid_b

        # Compute rotation using SVD
        H = centered_a.T @ centered_b
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_b - R @ centroid_a

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def _transform_points(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed = (T @ homogeneous.T).T
        return transformed[:, :3]

    def evaluate_accuracy(self):
        if len(self.localization_poses) == 0:
            print("No poses to evaluate")
            return

        self.translation_errors = []
        self.rotation_errors = []

        for timestamp in self.localization_poses:
            if timestamp not in self.ground_truth_poses:
                continue

            loc_pose_transformed = (
                self.transformation_matrix @ self.localization_poses[timestamp]
            )
            gt_pose = self.ground_truth_poses[timestamp]

            trans_error = np.linalg.norm(loc_pose_transformed[:3, 3] - gt_pose[:3, 3])
            self.translation_errors.append(trans_error)

            R_error = loc_pose_transformed[:3, :3].T @ gt_pose[:3, :3]
            rot_error_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
            rot_error_deg = np.degrees(rot_error_rad)
            self.rotation_errors.append(rot_error_deg)

            # Count precision categories
            for precision, stats in self.precision_stats.items():
                if (
                    trans_error <= stats["threshold_trans"]
                    and rot_error_deg <= stats["threshold_rot"]
                ):
                    stats["count"] += 1

        self.successful_localizations = len(self.translation_errors)

    def plot_error_distribution(self, errors, title, filename, unit):
        abs_errors = np.abs(errors)
        mean_val = np.mean(abs_errors)

        plt.figure(figsize=(6, 4))
        plt.hist(abs_errors, bins=20, alpha=0.7, edgecolor="black")
        plt.axvline(
            mean_val,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean = {mean_val:.3f} {unit}",
        )
        plt.title(title)
        plt.xlabel(f"Error ({unit})")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def dump_visualization(self, output_dir: str):
        self.plot_error_distribution(
            self.translation_errors,
            "Translation Error Distribution",
            f"{output_dir}/translation_errors.png",
            "m",
        )
        self.plot_error_distribution(
            self.rotation_errors,
            "Rotation Error Distribution",
            f"{output_dir}/rotation_errors.png",
            "deg",
        )

        # --- Table 1: Basic statistics ---
        stats_data = [
            ["Metric", "Translation (m)", "Rotation (°)"],
            [
                "Mean",
                f"{np.mean(self.translation_errors):.4f}",
                f"{np.mean(self.rotation_errors):.2f}",
            ],
            [
                "Median",
                f"{np.median(self.translation_errors):.4f}",
                f"{np.median(self.rotation_errors):.2f}",
            ],
            [
                "Std",
                f"{np.std(self.translation_errors):.4f}",
                f"{np.std(self.rotation_errors):.2f}",
            ],
            [
                "Max",
                f"{np.max(self.translation_errors):.4f}",
                f"{np.max(self.rotation_errors):.2f}",
            ],
        ]
        stats_markdown = tabulate(stats_data, headers="firstrow", tablefmt="github")
        with open(f"{output_dir}/metrics_summary.md", "w") as f:
            f.write(stats_markdown)

        stats_table = Table(stats_data, hAlign="LEFT")
        stats_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )

        # --- Table 2: Precision analysis ---
        precision_data = [["Precision", "Count/Total", "Percentage"]]
        for precision, stats in self.precision_stats.items():
            pct = (
                (stats["count"] / self.total_poses) * 100 if self.total_poses > 0 else 0
            )
            precision_data.append(
                [
                    precision.capitalize(),
                    f"{stats['count']}/{self.total_poses}",
                    f"{pct:.1f}% (≤{stats['threshold_trans']*100:.0f}cm, ≤{stats['threshold_rot']:.0f}°)",
                ]
            )
        precision_markdown = tabulate(
            precision_data, headers="firstrow", tablefmt="github"
        )
        with open(f"{output_dir}/precision_summary.md", "w") as f:
            f.write(precision_markdown)

        precision_table = Table(precision_data, hAlign="LEFT")
        precision_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )

        # --- Build PDF ---
        doc = SimpleDocTemplate(output_dir + "/error_distributions.pdf", pagesize=A4)
        elements = [
            # Page 1: Plots
            Image(f"{output_dir}/translation_errors.png", width=400, height=300),
            Spacer(1, 20),
            Image(f"{output_dir}/rotation_errors.png", width=400, height=300),
            PageBreak(),  # new page
            # Page 2: Tables
            stats_table,
            Spacer(1, 20),
            precision_table,
        ]
        doc.build(elements)
        print(
            f"Saved error distribution plots and stats to {output_dir}/error_distributions.pdf"
        )

    def save_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        summary = {
            "total_poses": self.total_poses,
            "successful_localizations": self.successful_localizations,
            "transformation_matrix": self.transformation_matrix.tolist(),
            "translation_errors": self.translation_errors,
            "rotation_errors": self.rotation_errors,
            "precision_stats": self.precision_stats,
        }

        with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        np.save(
            os.path.join(output_dir, "transformation_matrix.npy"),
            self.transformation_matrix,
        )

        print(f"Results saved to {output_dir}")


def sample_timestamps_from_bag(bag_path: str, num_samples: int) -> np.ndarray:
    start_time, end_time = BagMetadataExtractor.get_bag_time_range(bag_path)

    if None in [start_time, end_time]:
        raise RuntimeError("Failed to extract time ranges from bags")

    timestamps = BagMetadataExtractor.sample_timestamps_evenly(
        start_time, end_time, num_samples
    )
    print(f"Sampled {len(timestamps)} timestamps from overlapping time range")
    print(f"Time range: {(end_time - start_time) / 1e9:.1f} seconds")

    return timestamps


def query_poses_at_timestamps(
    timestamps: np.ndarray, map_result_dir_b: str
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Query poses at the specified timestamps using keyframe/relocalization poses + odom deltas.

    Args:
        timestamps: Array of timestamps to query
        map_result_dir_b: Directory containing bag B's mapping and localization results

    Returns:
        Tuple of (ground_truth_poses, localization_poses) dictionaries
        - ground_truth_poses: From bag B's mapping (keyframe + odom delta)
        - localization_poses: From bag B's localization in map A (relocalization + odom delta)
    """
    ground_truth_poses = {}
    localization_poses = {}

    try:
        # Load continuous odom poses
        continuous_odom_file = f"{map_result_dir_b}/localization_continuous_odom.npy"
        if not os.path.exists(continuous_odom_file):
            print(f"Error: Continuous odom file not found: {continuous_odom_file}")
            return ground_truth_poses, localization_poses

        # Create pose query engine for odom interpolation
        odom_query = PoseQueryEngine()
        odom_loaded = odom_query.load_continuous_poses(continuous_odom_file)
        if not odom_loaded:
            print(
                f"Error: Continuous odom poses file not loaded: {continuous_odom_file}"
            )
            return ground_truth_poses, localization_poses

        continuous_odom_poses = np.load(continuous_odom_file, allow_pickle=True).item()
        print(f"Loaded {len(continuous_odom_poses)} continuous odom poses")

        # Load keyframe poses (ground truth from bag B's mapping)
        keyframe_poses_file = f"{map_result_dir_b}/poses.npy"
        if not os.path.exists(keyframe_poses_file):
            print(f"Error: Keyframe poses file not found: {keyframe_poses_file}")
            return ground_truth_poses, localization_poses

        keyframe_poses = np.load(keyframe_poses_file, allow_pickle=True).item()
        print(f"Loaded {len(keyframe_poses)} keyframe poses")

        # Load relocalization poses (from localizing bag B in map A)
        relocalization_poses_file = f"{map_result_dir_b}/relocalization_poses.npy"
        if not os.path.exists(relocalization_poses_file):
            print(
                f"Error: Relocalization poses file not found: {relocalization_poses_file}"
            )
            return ground_truth_poses, localization_poses

        relocalization_poses = np.load(
            relocalization_poses_file, allow_pickle=True
        ).item()
        print(f"Loaded {len(relocalization_poses)} relocalization poses")

        # Process each timestamp
        for timestamp in timestamps:
            timestamp_int = int(timestamp)

            # === GROUND TRUTH POSE ===
            closest_keyframe_ts, closest_keyframe_pose = find_closest_pose(
                timestamp_int, keyframe_poses
            )

            if closest_keyframe_ts is not None:
                odom_at_keyframe = odom_query.query_pose_at_timestamp(
                    closest_keyframe_ts
                )
                odom_at_target = odom_query.query_pose_at_timestamp(timestamp_int)

                if odom_at_keyframe is not None and odom_at_target is not None:
                    odom_delta = np.linalg.inv(odom_at_keyframe) @ odom_at_target
                    ground_truth_poses[timestamp_int] = (
                        closest_keyframe_pose @ odom_delta
                    )

            # === LOCALIZATION POSE ===
            closest_reloc_ts, closest_reloc_pose = find_closest_pose(
                timestamp_int, relocalization_poses
            )

            if closest_reloc_ts is not None:
                odom_at_reloc = odom_query.query_pose_at_timestamp(closest_reloc_ts)
                odom_at_target = odom_query.query_pose_at_timestamp(timestamp_int)

                if odom_at_reloc is not None and odom_at_target is not None:
                    odom_delta = np.linalg.inv(odom_at_reloc) @ odom_at_target
                    localization_poses[timestamp_int] = closest_reloc_pose @ odom_delta

        print(f"Successfully computed {len(ground_truth_poses)} ground truth poses")
        print(f"Successfully computed {len(localization_poses)} localization poses")

    except Exception as e:
        print(f"Error querying poses: {e}")

    return ground_truth_poses, localization_poses


def find_closest_pose(
    target_timestamp: int, poses_dict: Dict[int, np.ndarray]
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    if not poses_dict:
        return None, None

    timestamps = list(poses_dict.keys())
    closest_ts = min(timestamps, key=lambda ts: abs(ts - target_timestamp))
    return closest_ts, poses_dict[closest_ts]


def run_mapping_process(
    bag_path: str,
    map_save_path: str,
    rate: float,
    data_saving_timeout: float,
    verbose_timer: bool,
) -> bool:
    ld = generate_launch_description_mapping(
        bag_path, map_save_path, rate, data_saving_timeout, "mapping", verbose_timer
    )
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
    return True


def run_localization_process(
    bag_path: str,
    tinynav_db_path: str,
    tinynav_map_path: str,
    rate: float,
    data_saving_timeout: float,
    verbose_timer: bool,
) -> bool:
    ld = generate_launch_description_localization(
        bag_path,
        tinynav_db_path,
        tinynav_map_path,
        rate,
        data_saving_timeout,
        "localization",
        verbose_timer=verbose_timer,
    )
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
    return True


def run_benchmark(
    bag_a_path: str,
    bag_b_path: str,
    output_dir: str,
    rate: float,
    num_samples: int,
    timeout: float,
    verbose_timer: bool = False,
) -> bool:
    """
    Run benchmark using timestamp-based sampling instead of keyframe-based.

    Args:
        bag_a_path: Path to bag A (for creating reference map)
        bag_b_path: Path to bag B (for localization and ground truth)
        output_dir: Output directory for results
        rate: Playback rate for bags
        num_samples: Number of timestamps to sample for evaluation

    Returns:
        True if successful, False otherwise
    """
    print("Starting TinyNav Timestamp-Based Mapping Benchmark")
    print(f"Bag A: {bag_a_path}")
    print(f"Bag B: {bag_b_path}")
    print(f"Playback rate: {rate}x")
    print(f"Number of samples: {num_samples}")

    map_result_dir_a = f"{output_dir}/benchmark_map_a"
    os.makedirs(map_result_dir_a, exist_ok=True)
    map_result_dir_b = f"{output_dir}/benchmark_map_b"
    os.makedirs(map_result_dir_b, exist_ok=True)

    results = BenchmarkResults()

    print("\nStep 1: Creating map A from bag A as reference...")
    if not run_mapping_process(
        bag_a_path,
        map_save_path=map_result_dir_a,
        rate=rate,
        data_saving_timeout=timeout,
        verbose_timer=verbose_timer,
    ):
        print("Error: Failed to create map A")
        return False

    print("\nStep 2: Localizing bag B in map A...")
    if not run_localization_process(
        bag_b_path,
        tinynav_db_path=map_result_dir_b,
        tinynav_map_path=map_result_dir_a,
        rate=rate,
        data_saving_timeout=timeout,
        verbose_timer=verbose_timer,
    ):
        print("Error: Failed to localize bag B in map A")
        return False

    print(f"\nStep 3: Sampling {num_samples} timestamps from bag time ranges...")
    sampled_timestamps = sample_timestamps_from_bag(bag_b_path, num_samples)

    if sampled_timestamps is None or len(sampled_timestamps) == 0:
        print("Error: Timestamp sampling failed")
        return False

    print(f"\nStep 4: Querying poses at sampled timestamps...")
    ground_truth_poses, localization_poses = query_poses_at_timestamps(
        sampled_timestamps, map_result_dir_b
    )

    if len(ground_truth_poses) == 0 or len(localization_poses) == 0:
        print("Error: Pose querying failed")
        return False

    print(f"\nStep 5: Computing coordinate transformation...")
    # Find common timestamps
    common_timestamps = set(ground_truth_poses.keys()) & set(localization_poses.keys())
    print(f"Found {len(common_timestamps)} matching pose pairs")

    for timestamp in common_timestamps:
        results.add_pose_pair(
            timestamp, localization_poses[timestamp], ground_truth_poses[timestamp]
        )

    if results.total_poses == 0:
        print("Error: No matching pose pairs found")
        return False

    if not results.compute_transformation():
        print("Error: Failed to compute coordinate transformation")
        return False

    print("\nStep 6: Evaluating localization accuracy...")
    results.evaluate_accuracy()

    os.makedirs(output_dir, exist_ok=True)
    results.dump_visualization(output_dir)
    results.save_results(output_dir)

    print(f"\nTimestamp-based benchmark completed successfully!")
    print(
        f"Evaluated {len(common_timestamps)} poses sampled over {(sampled_timestamps[-1] - sampled_timestamps[0]) / 1e9:.1f} seconds"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="TinyNav Mapping Benchmark")
    parser.add_argument(
        "--bag_a", required=True, help="Path to ROS bag A (for mapping)"
    )
    parser.add_argument(
        "--bag_b",
        required=True,
        help="Path to ROS bag B (for localization and ground truth)",
    )
    parser.add_argument(
        "--output_dir",
        default="output/benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of evaluation samples (default: 100)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Playback rate for ROS bags (default: 1.0x)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout for each mapping process (seconds).",
    )
    parser.add_argument(
        "--verbose_timer",
        action="store_true",
        help="Enable verbose timer output for nodes",
    )
    parser.add_argument(
        "--no_verbose_timer",
        dest="verbose_timer",
        action="store_false",
        help="Disable verbose timer output for nodes (default)",
    )
    parser.set_defaults(verbose_timer=False)

    args = parser.parse_args()

    if not os.path.exists(args.bag_a):
        print(f"Error: Bag A not found: {args.bag_a}")
        return 1

    if not os.path.exists(args.bag_b):
        print(f"Error: Bag B not found: {args.bag_b}")
        return 1

    if args.rate <= 0:
        print("Error: Rate must be positive")
        return 1

    benchmark_return = run_benchmark(
        args.bag_a,
        args.bag_b,
        args.output_dir,
        args.rate,
        args.num_images,
        args.timeout,
        args.verbose_timer,
    )
    if benchmark_return:
        print("\nBenchmark completed!")
    else:
        print("\nBenchmark failed!")

    return 0


if __name__ == "__main__":
    main()
