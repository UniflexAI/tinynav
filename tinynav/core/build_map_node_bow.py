from __future__ import annotations

import logging
import os

import numpy as np
import rclpy
import tyro
from rclpy.executors import SingleThreadedExecutor

from tinynav.core.bow_retrieval import build_bow_index
from tinynav.core.build_map_node import (
    BagPlayer,
    BuildMapArgs,
    BuildMapNode,
    ImageTransportsNode,
    TinyNavDB,
)
from tinynav.core.map_node_bow import _enhance_clahe_gamma


class BuildMapNodeBow(BuildMapNode):
    """Build-map node variant that also writes a SuperPoint BoW retrieval index."""

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        return _enhance_clahe_gamma(image)

    def save_mapping(self):
        was_completed = self._save_completed
        super().save_mapping()
        if was_completed or not self._save_completed:
            return

        poses_path = f"{self.map_save_path}/poses.npy"
        if not os.path.exists(poses_path):
            self.get_logger().warning(f"Cannot build BoW index; poses not found: {poses_path}")
            return

        self.get_logger().info("Building SuperPoint BoW index...")
        with self.stage_timer.timed("build_bow_index"):
            poses = np.load(poses_path, allow_pickle=True).item()
            bow_db = TinyNavDB(self.map_save_path, is_scratch=False)
            build_bow_index(
                bow_db,
                sorted(int(timestamp) for timestamp in poses.keys()),
                f"{self.map_save_path}/bow_index.npz",
            )
            bow_db.close()
        self.get_logger().info(f"Saved SuperPoint BoW index to {self.map_save_path}/bow_index.npz")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    rclpy.init()

    parsed_args = tyro.cli(BuildMapArgs, use_underscores=True)

    exec_ = SingleThreadedExecutor()
    player_node = BagPlayer(parsed_args.bag_file, play_rate=parsed_args.play_rate)
    map_node = BuildMapNodeBow(
        parsed_args.map_save_path,
        verbose_timer=parsed_args.verbose_timer,
        global_frames_ratio=parsed_args.global_frames_ratio,
    )
    image_transports_node = ImageTransportsNode()
    exec_.add_node(player_node)
    exec_.add_node(map_node)
    exec_.add_node(image_transports_node)
    while rclpy.ok() and player_node.play_next():
        exec_.spin_once(timeout_sec=0.001)
    player_node._publish_percent(100.0)
    map_node.save_mapping()
