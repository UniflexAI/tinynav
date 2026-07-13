from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import rclpy

from tinynav.core.bow_retrieval import BowIndex, build_bow_index
from tinynav.core.map_node import MapNode
from tinynav.core.math_utils import rerank_by_pnp_inliers


def _enhance_clahe_gamma(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8, gamma: float = 1.5) -> np.ndarray:
    """Apply CLAHE + gamma correction to a mono8 image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(image)
    # gamma correction
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    enhanced = cv2.LUT(enhanced, lut)
    return enhanced


class MapNodeBow(MapNode):
    """Map node variant that uses SuperPoint BoW retrieval for map relocalization.

    This keeps the original map node behavior for navigation, local pose-graph updates,
    and PnP verification. Only the map relocalization candidate generator changes:
    DINO global-descriptor Top-K is replaced by SuperPoint BoW Top-K.
    """

    def __init__(
        self,
        tinynav_db_path: str,
        tinynav_map_path: str,
        verbose_timer: bool = True,
        bow_score_threshold: float = 0.0,
    ):
        super().__init__(tinynav_db_path, tinynav_map_path, verbose_timer)
        self.bow_score_threshold = bow_score_threshold
        self.bow_index_path = f"{tinynav_map_path}/bow_index.npz"
        if not os.path.exists(self.bow_index_path):
            self.get_logger().info(f"BoW index not found at {self.bow_index_path}; building it from map features")
            build_bow_index(self.db, sorted(int(timestamp) for timestamp in self.map_poses.keys()), self.bow_index_path)
        self.bow_index = BowIndex(self.bow_index_path)

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        return _enhance_clahe_gamma(image)

    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray | None) -> tuple[bool, np.ndarray, float]:
        if K is None:
            return False, np.eye(4), -np.inf

        timestamp_and_score_array = [
            (timestamp, score)
            for timestamp, score in self.bow_index.query(keyframe_features, self.relocalization_loop_top_k)
            if score >= self.bow_score_threshold
        ]
        max_score = max((score for _, score in timestamp_and_score_array), default=0.0)
        if len(timestamp_and_score_array) == 0:
            self.get_logger().info(
                f"not enough BoW candidates to relocalize, {len(timestamp_and_score_array)}, max_score: {max_score}"
            )
            return False, np.eye(4), -np.inf

        pnp_candidates = []
        for timestamp_in_map, _bow_score in timestamp_and_score_array:
            reference_keyframe_pose = self.map_poses[timestamp_in_map]
            reference_depth, _, reference_features, _, _ = self.db.get_depth_embedding_features_images(timestamp_in_map)
            reference_matched_keypoints, keyframe_matched_keypoints, matches = self.match_keypoints(
                reference_features, keyframe_features
            )
            if len(matches) < 50:
                self.get_logger().info(f"not enough matched features to relocalize, {len(matches)} < 50")
                continue

            point_3d_in_world, inliers = self.keypoint_with_depth_to_3d(
                reference_matched_keypoints,
                reference_depth,
                reference_keyframe_pose,
                self.map_K,
            )
            point_3d_in_world_list = point_3d_in_world[inliers]
            point_2d_in_keyframe_list = keyframe_matched_keypoints[inliers]
            point_count = len(point_2d_in_keyframe_list)
            if point_count <= 80:
                self.get_logger().info(f"not enough landmarks to relocalize, {point_count}")
                continue
            pnp_candidates.append((point_3d_in_world_list, point_2d_in_keyframe_list))

        success, best_pose_in_camera, pose_cov_weight, _, _, _ = rerank_by_pnp_inliers(pnp_candidates, self.map_K)
        if success:
            self.get_logger().info(f"BoW relocalization pose: {best_pose_in_camera}")
            return True, best_pose_in_camera, pose_cov_weight

        self.get_logger().info("no valid BoW PnP relocalization candidate found")
        return False, np.eye(4), -np.inf


def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinynav_db_path", type=str, default="tinynav_temp")
    parser.add_argument("--tinynav_map_path", type=str, required=True)
    parser.add_argument("--bow_score_threshold", type=float, default=0.0)
    parser.add_argument("--verbose_timer", action="store_true", default=True, help="Enable verbose timer output")
    parser.add_argument("--no_verbose_timer", dest="verbose_timer", action="store_false", help="Disable verbose timer output")
    parsed_args, _unknown_args = parser.parse_known_args(sys.argv[1:])
    node = MapNodeBow(
        tinynav_db_path=parsed_args.tinynav_db_path,
        tinynav_map_path=parsed_args.tinynav_map_path,
        verbose_timer=parsed_args.verbose_timer,
        bow_score_threshold=parsed_args.bow_score_threshold,
    )

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
