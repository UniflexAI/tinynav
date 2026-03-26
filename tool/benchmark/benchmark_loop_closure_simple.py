#!/usr/bin/env python3
"""Simplified loop closure evaluation — no timestamp interpolation.

Approach:
- Bag A: build Map A (reference map)
- Bag B: every N images, localize in Map A → predicted pose
- Bag B: every N images, query Map B at the same image index → gt pose
- Compare directly (by image index, no interpolation needed)

Usage:
    uv run python tool/benchmark/benchmark_loop_closure_simple.py \
        --bag_a /path/to/bag_a \
        --bag_b /path/to/bag_b \
        --every_n 20 \
        --output_dir results/
"""
import os
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

from rosbag2_py import SequentialReader, StorageFilter
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

# ---------------------------------------------------------------------------
# Image index sampling
# ---------------------------------------------------------------------------

def sample_image_indices(msg_infos: List, every_n: int) -> List[int]:
    """Return image indices by stepping every_n images, skipping first/last 10%.

    Args:
        msg_infos: List of (timestamp, topic, type) tuples from rosbag
        every_n: sample every N images

    Returns:
        List of selected indices into msg_infos
    """
    # Filter to image topics
    image_msgs = [m for m in msg_infos if "image" in m[1].lower() or "infra" in m[1].lower()]
    total = len(image_msgs)

    # Skip first/last 10% (robot may be stationary)
    skip = max(1, int(total * 0.05))
    candidates = image_msgs[skip:-skip] if skip > 0 else image_msgs

    indices = list(range(skip, skip + len(candidates), every_n))
    return indices


def load_images_from_bag(
    bag_path: str, topic_filter: Optional[str] = None, max_images: int = 0
) -> List[Tuple[int, Image]]:
    """Load images from a rosbag.

    Returns:
        List of (timestamp_ns, Image msg)
    """
    reader = SequentialReader()
    reader.open(bag_path, "")

    if topic_filter:
        reader.set_filter(StorageFilter(topics=[topic_filter]))

    images: List[Tuple[int, Image]] = []
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg = reader._serializer.decompress_message(data)  # noqa: SLF001
        if isinstance(msg, Image):
            images.append((t, msg))
            if 0 < max_images <= len(images):
                break
    return images


# ---------------------------------------------------------------------------
# Simplified benchmark (placeholder — needs integration with actual mapping/localization)
# ---------------------------------------------------------------------------

class SimpleLoopClosureBenchmark:
    """Simplified loop closure evaluation without timestamp interpolation."""

    def __init__(self, bag_a: str, bag_b: str, every_n: int = 20, output_dir: str = "output/loop_closure"):
        self.bag_a = bag_a
        self.bag_b = bag_b
        self.every_n = every_n
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.predicted_poses: Dict[int, np.ndarray] = {}  # idx -> 4x4
        self.gt_poses: Dict[int, np.ndarray] = {}  # idx -> 4x4
        self.translation_errors: List[float] = []
        self.rotation_errors: List[float] = []

    def sample_indices(self, bag_path: str) -> List[int]:
        """Read bag metadata and return sampled image indices."""
        from rosbag2_py import Info
        info = Info()
        metadata = info.read_metadata(bag_path, "")
        msg_infos = [(m.timestamp, m.topic_metadata.name, m.topic_metadata.type) for m in metadata.topics_with_message_count]
        return sample_image_indices(msg_infos, self.every_n)

    def build_map_a(self):
        """Build Map A from bag A.

        This is a placeholder — integrate with the actual mapping pipeline
        (tinynav mapping node) to produce Map A.
        """
        # TODO: integrate with actual mapping
        raise NotImplementedError(
            "build_map_a not implemented yet. "
            "Needs to run the mapping pipeline on bag_a and save Map A."
        )

    def localize_in_map_a(self, image: Image, idx: int) -> Optional[np.ndarray]:
        """Localize a single image in Map A.

        Args:
            image: The image to localize
            idx: Image index in bag B

        Returns:
            4x4 predicted pose matrix, or None if localization failed
        """
        # TODO: integrate with actual localization node
        raise NotImplementedError(
            "localize_in_map_a not implemented yet. "
            "Needs to run the localization node using Map A + this image."
        )

    def query_gt_from_map_b(self, idx: int) -> Optional[np.ndarray]:
        """Query ground truth pose from Map B at image index idx.

        Args:
            idx: Image index in bag B

        Returns:
            4x4 ground truth pose matrix, or None if not found
        """
        # TODO: integrate with actual mapping output from bag B
        raise NotImplementedError(
            "query_gt_from_map_b not implemented yet. "
            "Needs Map B to have been built and stored with image index keys."
        )

    def estimate_transform_ransac(
        self, predicted: Dict[int, np.ndarray], gt: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, List[int]]:
        """Estimate rigid transformation between predicted and gt using RANSAC.

        Args:
            predicted: idx -> 4x4 predicted pose
            gt: idx -> 4x4 gt pose

        Returns:
            (4x4 transformation matrix, list of inlier indices)
        """
        common_idx = sorted(set(predicted.keys()) & set(gt.keys()))
        if len(common_idx) < 3:
            raise ValueError(f"Need at least 3 common indices, got {len(common_idx)}")

        pts_pred = np.array([predicted[i][:3, 3] for i in common_idx])
        pts_gt = np.array([gt[i][:3, 3] for i in common_idx])

        best_inliers = 0
        best_T = np.eye(4)
        inlier_threshold = 0.20  # 20cm
        max_iterations = 1000

        for _ in range(max_iterations):
            choice = np.random.choice(len(common_idx), 3, replace=False)
            T = self._rigid_transform_3pts(pts_pred[choice], pts_gt[choice])
            if T is None:
                continue

            pts_transformed = (T @ np.vstack([pts_pred.T, np.ones(len(pts_pred))]))[:3].T
            distances = np.linalg.norm(pts_transformed - pts_gt, axis=1)
            inliers = int(np.sum(distances < inlier_threshold))
            if inliers > best_inliers:
                best_inliers = inliers
                best_T = T

        # Refine with all inliers
        pts_pred_inlier = pts_pred[np.linalg.norm((best_T @ np.vstack([pts_pred.T, np.ones(len(pts_pred))]))[:3].T - pts_gt, axis=1) < inlier_threshold]
        pts_gt_inlier = pts_gt[np.linalg.norm((best_T @ np.vstack([pts_pred.T, np.ones(len(pts_pred))]))[:3].T - pts_gt, axis=1) < inlier_threshold]
        if len(pts_pred_inlier) >= 3:
            refined = self._rigid_transform_3pts(pts_pred_inlier, pts_gt_inlier)
            if refined is not None:
                best_T = refined

        inlier_mask = np.linalg.norm((best_T @ np.vstack([pts_pred.T, np.ones(len(pts_pred))]))[:3].T - pts_gt, axis=1) < inlier_threshold
        inlier_indices = [common_idx[i] for i in range(len(common_idx)) if inlier_mask[i]]
        return best_T, inlier_indices

    def _rigid_transform_3pts(
        self, pts_a: np.ndarray, pts_b: np.ndarray
    ) -> Optional[np.ndarray]:
        """Estimate rigid transform from 3 point pairs (Umeyama)."""
        if pts_a.shape != (3, 3) or pts_b.shape != (3, 3):
            return None
        centroid_a = pts_a.mean(axis=0)
        centroid_b = pts_b.mean(axis=0)
        aa = pts_a - centroid_a
        bb = pts_b - centroid_b
        H = aa.T @ bb
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = centroid_b - R @ centroid_a
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def compute_errors(
        self, transform: np.ndarray, inlier_indices: List[int]
    ) -> Tuple[List[float], List[float]]:
        """Compute translation and rotation errors after applying transform."""
        trans_errors = []
        rot_errors = []
        for i in inlier_indices:
            pred = self.predicted_poses[i]
            gt = self.gt_poses[i]
            pred_transformed = transform @ pred
            t_err = float(np.linalg.norm(pred_transformed[:3, 3] - gt[:3, 3]))

            R_pred = pred_transformed[:3, :3]
            R_gt = gt[:3, :3]
            R_rel = R_pred @ R_gt.T
            angle = float(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)) * 180 / np.pi)
            trans_errors.append(t_err)
            rot_errors.append(angle)
        return trans_errors, rot_errors

    def print_summary(self, transform: np.ndarray, inlier_indices: List[int]):
        """Print and save evaluation summary."""
        if not inlier_indices:
            print("No inliers — check data")
            return

        trans_errors, rot_errors = self.compute_errors(transform, inlier_indices)

        def pct(threshold_trans, threshold_rot):
            count = sum(
                (t < threshold_trans and r < threshold_rot)
                for t, r in zip(trans_errors, rot_errors)
            )
            return 100.0 * count / len(inlier_indices)

        print("\n========== Loop Closure Evaluation Summary ==========")
        print(f"Bag A: {self.bag_a}")
        print(f"Bag B: {self.bag_b}")
        print(f"Every-N images: {self.every_n}")
        print(f"Total localization attempts: {len(self.predicted_poses)}")
        print(f"Inliers (RANSAC, 20cm): {len(inlier_indices)}/{len(self.predicted_poses)}")
        print(f"\nPrecision @ 5cm/2° : {pct(0.05, 2.0):.1f}%")
        print(f"Precision @ 10cm/5°: {pct(0.10, 5.0):.1f}%")
        print(f"Precision @ 30cm/10°: {pct(0.30, 10.0):.1f}%")
        print(f"\nTranslation error (mean): {np.mean(trans_errors)*100:.2f} cm")
        print(f"Translation error (median): {np.median(trans_errors)*100:.2f} cm")
        print(f"Rotation error (mean): {np.mean(rot_errors):.2f} deg")
        print(f"\nEstimated transform (Map A -> Map B):\n{transform}")

        summary_path = os.path.join(self.output_dir, "metrics_summary.md")
        with open(summary_path, "w") as f:
            f.write("# Loop Closure Evaluation Summary\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Bag A | `{self.bag_a}` |\n")
            f.write(f"| Bag B | `{self.bag_b}` |\n")
            f.write(f"| Every-N | {self.every_n} |\n")
            f.write(f"| Inliers | {len(inlier_indices)}/{len(self.predicted_poses)} |\n")
            f.write(f"| ATE < 5cm & 2° | {pct(0.05, 2.0):.1f}% |\n")
            f.write(f"| ATE < 10cm & 5° | {pct(0.10, 5.0):.1f}% |\n")
            f.write(f"| ATE < 30cm & 10° | {pct(0.30, 10.0):.1f}% |\n")
            f.write(f"| Trans error mean | {np.mean(trans_errors)*100:.2f} cm |\n")
            f.write(f"| Rot error mean | {np.mean(rot_errors):.2f} ° |\n")
        print(f"\nSaved summary to {summary_path}")

    def run(self):
        """Run the full simplified evaluation."""
        indices = self.sample_indices(self.bag_b)
        print(f"Sampled {len(indices)} image indices from bag B (every {self.every_n})")

        # Step 1: build Map A
        print("Building Map A from bag A...")
        self.build_map_a()

        # Step 2: load bag B images and localize
        print("Loading images from bag B...")
        bag_b_images = load_images_from_bag(self.bag_b, max_images=0)
        print(f"Loaded {len(bag_b_images)} images from bag B")

        print(f"Localizing every {self.every_n}th image in Map A...")
        for idx in indices:
            if idx >= len(bag_b_images):
                break
            ts, image = bag_b_images[idx]
            try:
                pose = self.localize_in_map_a(image, idx)
                if pose is not None:
                    self.predicted_poses[idx] = pose
            except NotImplementedError:
                raise
            except Exception as e:
                print(f"  Localization failed at idx {idx}: {e}")

        # Step 3: query gt from Map B
        print("Querying ground truth from Map B...")
        for idx in indices:
            try:
                pose = self.query_gt_from_map_b(idx)
                if pose is not None:
                    self.gt_poses[idx] = pose
            except NotImplementedError:
                raise
            except Exception as e:
                print(f"  GT query failed at idx {idx}: {e}")

        # Step 4: estimate transform + report
        common = set(self.predicted_poses) & set(self.gt_poses)
        print(f"Common indices with both predicted and gt: {len(common)}")
        if len(common) < 3:
            raise RuntimeError(f"Not enough common indices ({len(common)}) for RANSAC — need at least 3")

        transform, inliers = self.estimate_transform_ransac(self.predicted_poses, self.gt_poses)
        self.print_summary(transform, inliers)


def main():
    parser = argparse.ArgumentParser(description="Simplified loop closure evaluation")
    parser.add_argument("--bag_a", required=True, help="Path to ROS bag A (reference map)")
    parser.add_argument("--bag_b", required=True, help="Path to ROS bag B (test bag)")
    parser.add_argument("--every_n", type=int, default=20, help="Sample every N images (default: 20)")
    parser.add_argument("--output_dir", default="output/loop_closure", help="Output directory")
    args = parser.parse_args()

    bench = SimpleLoopClosureBenchmark(
        bag_a=args.bag_a,
        bag_b=args.bag_b,
        every_n=args.every_n,
        output_dir=args.output_dir,
    )
    bench.run()


if __name__ == "__main__":
    main()
