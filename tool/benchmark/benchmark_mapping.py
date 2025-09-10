import os
import argparse
import subprocess
import time
import signal
import json
from typing import Dict, List, Optional
import shutil

import numpy as np
import rclpy
import rosbag2_py
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate,
    Image,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from tabulate import tabulate

# FIXME(yuance): Update database path
TINYNAV_DB = "tinynav_db"


class ProcessManager:
    """Manage subprocesses and ensure cleanup on exit."""

    def __init__(self):
        self.processes = {}  # name -> Popen
        self.cleaned_up = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start_process(self, name, cmd, cwd=None, use_pgroup=True):
        kwargs = {}
        if use_pgroup:
            kwargs["preexec_fn"] = os.setsid
        proc = subprocess.Popen(cmd, cwd=cwd, **kwargs)
        self.processes[name] = proc
        print(f"[ProcessManager] Started {name} (pid={proc.pid})")
        return proc

    def cleanup(self):
        if self.cleaned_up:
            return
        self.cleaned_up = True
        print("[ProcessManager] Cleaning up subprocesses...")
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                try:
                    print(f"Sending SIGINT to {name}...")
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception as e:
                    print(f"Error terminating {name}: {e}")

        time.sleep(15)

        # Force killing
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                print(f"Force killing {name}...")
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        print("[ProcessManager] Cleanup done.")

    def is_process_running(self, name):
        proc = self.processes.get(name)
        return proc and proc.poll() is None

    def _signal_handler(self, signum, frame):
        print(f"[ProcessManager] Received signal {signum}, cleaning up...")
        self.cleanup()
        os._exit(0)


class BenchmarkResults:
    """Container for benchmark results and metrics."""

    def __init__(self):
        self.total_images = 0
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
        self.total_images += 1

    def add_failed_localization(self, timestamps: List[int]):
        self.total_images += len(timestamps)

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
        stats_markdown = tabulate(stats_data, headers="firstrow", tablefmt="markdown")
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
                (stats["count"] / self.total_images) * 100
                if self.total_images > 0
                else 0
            )
            precision_data.append(
                [
                    precision.capitalize(),
                    f"{stats['count']}/{self.total_images}",
                    f"{pct:.1f}% (≤{stats['threshold_trans']*100:.0f}cm, ≤{stats['threshold_rot']:.0f}°)",
                ]
            )
        precision_markdown = tabulate(
            precision_data, headers="firstrow", tablefmt="markdown"
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
            "total_images": self.total_images,
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


def extract_bag_timestamps(bag_path: str) -> List[int]:
    """Extract all image timestamps from ROS bag."""
    timestamps = []

    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(bag_path), storage_id="sqlite3"
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        # Read messages
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()

            # Look for image topics (common patterns)
            if "/image" in topic or "/camera" in topic:
                if "sensor_msgs/msg/Image" in type_map.get(topic, ""):
                    timestamps.append(timestamp)

    except Exception as e:
        print(f"Error reading bag {bag_path}: {e}")
        return []

    timestamps.sort()
    return timestamps


def get_bag_duration(bag_path: str) -> float:
    try:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(bag_path), storage_id="sqlite3"
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader.open(storage_options, converter_options)

        first_timestamp = None
        last_timestamp = None

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

        if first_timestamp is not None and last_timestamp is not None:
            duration = (last_timestamp - first_timestamp) / 1e9
            return duration
        else:
            return 0.0

    except Exception as e:
        print(f"Error reading bag duration {bag_path}: {e}")
        return 0.0


def select_evaluation_timestamps(
    poses_file: str,
    sample_cnt: int = 100,
) -> Dict[int, np.ndarray]:
    """Select evenly spaced timestamps and poses from actual saved keyframes."""
    try:
        if not os.path.exists(poses_file):
            print(f"Error: Poses file not found at {poses_file}")
            return {}

        saved_poses = np.load(poses_file, allow_pickle=True).item()
        all_timestamps = sorted(saved_poses.keys())

        if len(all_timestamps) == 0:
            print("Error: No poses found in saved map")
            return {}

        if len(all_timestamps) < sample_cnt:
            print(f"Warning: Only {len(all_timestamps)} keyframes available, using all")
            # Return all available timestamp-pose pairs
            return {ts: saved_poses[ts] for ts in all_timestamps}

        if len(all_timestamps) < 2 * sample_cnt:
            print(
                f"Warning: Too few keyframes to select: {len(all_timestamps)}, only the first {sample_cnt} will be used."
            )

        # Select evenly spaced keyframe timestamps
        indices = np.linspace(0, len(all_timestamps) - 1, sample_cnt, dtype=int)
        selected_timestamp_pose_pairs = {}

        for i in indices:
            timestamp = all_timestamps[i]
            selected_timestamp_pose_pairs[timestamp] = saved_poses[timestamp]

        print(
            f"Selected {len(selected_timestamp_pose_pairs)} timestamp-pose pairs from {len(all_timestamps)} keyframes"
        )
        return selected_timestamp_pose_pairs

    except Exception as e:
        print(f"Error selecting timestamps from saved map: {e}")
        return {}


def run_mapping_process(
    bag_path: str, is_mapping_mode: bool, rate: float = 0.25
) -> bool:
    pm = ProcessManager()

    if is_mapping_mode and os.path.exists(TINYNAV_DB):
        shutil.rmtree(TINYNAV_DB)
        print(f"Cleaned up previous map data at {TINYNAV_DB}")

    print(
        f"Running {'mapping' if is_mapping_mode else 'localization'} on {bag_path} at {rate}x speed"
    )

    # Start perception node
    pm.start_process(
        "perception",
        ["uv", "run", "python", "/tinynav/tinynav/core/perception_node.py"],
        cwd="/tinynav",
    )

    # FIXME(yuance): Remove this workaround after map_node can run properly without pois.txt
    if not is_mapping_mode:
        pois_file = f"{TINYNAV_DB}/pois.txt"
        if not os.path.exists(pois_file):
            default_pois = np.array(
                [
                    [2.0, 1.0, 0.0],
                ]
            )
            np.savetxt(pois_file, default_pois, fmt="%.6f")

    # Start map node
    pm.start_process(
        "map",
        [
            "uv",
            "run",
            "python",
            "/tinynav/tinynav/core/map_node.py",
            "--mapping",
            str(is_mapping_mode).lower(),
        ],
        cwd="/tinynav",
    )

    # Wait for nodes to initialize
    time.sleep(3)

    # Play ROS bag
    bag_proc = pm.start_process(
        "bag",
        ["ros2", "bag", "play", str(bag_path), "--rate", str(rate)],
        use_pgroup=True,
    )

    bag_duration = get_bag_duration(bag_path)
    timeout = int(bag_duration / rate + 10)

    start_time = time.time()
    while True:
        return_code = bag_proc.poll()
        if return_code is not None:
            if return_code != 0:
                print(f"Bag playback exited with code {return_code}")
                return False
            else:
                print("Bag playback completed")
                break

        if not pm.is_process_running("perception") or not pm.is_process_running("map"):
            print(
                f"Error: {'map' if not pm.is_process_running('map') else 'perception'} node terminated unexpectedly before bag playback ends"
            )
            pm.cleanup()
            bag_proc.kill()
            return False

        if time.time() - start_time > timeout:
            print("Bag playback timed out")
            bag_proc.kill()
            return False

        time.sleep(1)

    print("Waiting for processing...")
    time.sleep(10)

    pm.cleanup()
    return True


def copy_map_result(from_dir: str, to_dir: str) -> bool:
    try:
        if os.path.exists(to_dir):
            print(f"Removing existing directory at {to_dir}")
            shutil.rmtree(to_dir)
        shutil.copytree(from_dir, to_dir)
        print(f"Copied from {from_dir} to {to_dir}")
        return True
    except Exception as e:
        print(f"Error copying map: {e}")
        return False


def extract_relocalization_poses(
    poses_file: str, timestamps: List[int]
) -> Dict[int, np.ndarray]:
    poses = {}

    try:
        if not os.path.exists(poses_file):
            print(f"Error: Relocalization poses file not found at {poses_file}")
            return poses

        saved_poses = np.load(poses_file, allow_pickle=True).item()

        # Match timestamps (approximate matching)
        for target_ts in timestamps:
            best_match = None
            min_diff = float("inf")

            for saved_ts, pose in saved_poses.items():
                diff = abs(saved_ts - target_ts)
                if diff < min_diff:
                    min_diff = diff
                    best_match = pose

            if min_diff < 50_000_000:  # Allow 50 ms tolerance
                poses[target_ts] = best_match

        print(f"Extracted {len(poses)} poses from saved map ({poses_file})")

    except Exception as e:
        print(f"Error extracting poses from saved map ({poses_file}): {e}")

    return poses


def extract_failed_localization_timestamps() -> List[int]:
    failed_reloc_file = TINYNAV_DB + "/failed_relocalizations.npy"

    if not os.path.exists(failed_reloc_file):
        raise Exception(f"No failed relocalizations file found at {failed_reloc_file}")

    return np.load(failed_reloc_file, allow_pickle=True).tolist()


def run_benchmark(
    bag_a_path: str, bag_b_path: str, output_dir: str, rate: float = 1.0
) -> bool:
    print("Starting TinyNav Mapping Benchmark")
    print(f"Bag A: {bag_a_path}")
    print(f"Bag B: {bag_b_path}")
    print(f"Playback rate: {rate}x")

    map_result_dir_b = f"{output_dir}/benchmark_map_b"
    os.makedirs(map_result_dir_b, exist_ok=True)

    results = BenchmarkResults()

    print("\nStep 1: Creating ground truth map B from bag B...")
    if not run_mapping_process(bag_b_path, is_mapping_mode=True, rate=rate):
        print("Error: Failed to create map B")
        return False

    print("\nStep 2: Selecting evaluation timestamps from map B keyframes...")
    ground_truth_poses = select_evaluation_timestamps(
        TINYNAV_DB + "/poses.npy",
        100,
    )

    if len(ground_truth_poses) == 0:
        print("Error: No keyframes extracted from saved map B")
        return False

    # Extract timestamps and ground truth poses
    evaluation_timestamps = list(ground_truth_poses.keys())
    print(f"Extracted {len(ground_truth_poses)} ground truth poses")

    copy_map_result(TINYNAV_DB, map_result_dir_b)

    print("\nStep 3: Creating map A from bag A...")
    if not run_mapping_process(bag_a_path, is_mapping_mode=True, rate=rate):
        print("Error: Failed to create map A")
        return False

    print("\nStep 4: Localizing bag B in map A...")
    if not run_mapping_process(
        bag_b_path,
        is_mapping_mode=False,
        rate=rate,
    ):
        print("Error: Failed to localize bag B in map A")
        return False

    # Extract localization results for the same timestamps
    localization_poses = extract_relocalization_poses(
        TINYNAV_DB + "/relocalization_poses.npy", evaluation_timestamps
    )
    print(f"Extracted {len(localization_poses)} localization poses")
    failed_reloc_timestamps = extract_failed_localization_timestamps()
    print(f"Extracted {len(failed_reloc_timestamps)} failed relocalizations")

    print("\nStep 5: Computing coordinate transformation...")
    for timestamp in evaluation_timestamps:
        if timestamp in localization_poses and timestamp in ground_truth_poses:
            results.add_pose_pair(
                timestamp, localization_poses[timestamp], ground_truth_poses[timestamp]
            )

    if results.total_images == 0:
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
        "--num_images", type=int, default=100, help="Number of evaluation images"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.25,
        help="Playback rate for ROS bags (default: 0.25x)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for each mapping process (seconds). If not specified, calculated from bag duration and rate",
    )

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

    try:
        rclpy.init()
        benchmark_return = run_benchmark(
            args.bag_a, args.bag_b, args.output_dir, args.rate
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        return 1
    finally:
        try:
            rclpy.shutdown()
        except:
            pass

    if benchmark_return:
        print("\nBenchmark completed!")
    else:
        print("\nBenchmark failed!")

    return 0


if __name__ == "__main__":
    main()
