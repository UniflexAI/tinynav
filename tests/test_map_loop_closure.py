import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gtsam
import numpy as np

from tinynav.core import map_node


class MapLoopClosureTests(unittest.TestCase):
    def test_loop_setting_controls_search_without_stopping_odometry(self):
        for enabled in (True, False):
            with self.subTest(loop_closure=enabled):
                noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6))
                async def infer(_image):
                    return {}
                node = SimpleNamespace(
                    K=np.eye(3), loop_closure=enabled,
                    bridge=SimpleNamespace(imgmsg_to_cv2=lambda *_args, **_kwargs: np.ones((2, 2))),
                    nav_temp_db=SimpleNamespace(
                        set_entry=Mock(), get_embedding=lambda _t: np.ones(2),
                        get_depth_embedding_features_images=lambda _t: (None, None, {}, None, None),
                    ),
                    get_embeddings=lambda _image: np.ones(2),
                    super_point_extractor=SimpleNamespace(infer=infer),
                    isam=gtsam.ISAM2(), odometry_noise=noise, loop_noise=noise,
                    match_keypoints=lambda *_args: (np.zeros((120, 2)), np.zeros((120, 2)), None),
                    odom={}, pose_graph_used_pose={}, last_keyframe_timestamp=None,
                    pose_graph_trajectory_publish=Mock(), timer_logger=None,
                    loop_similarity_threshold=0.9, loop_top_k=1,
                )
                with patch.object(map_node, 'msg2np', return_value=(np.eye(4), None)), \
                     patch.object(map_node, 'find_loop', return_value=[(0, 0.95)]) as search, \
                     patch.object(map_node, 'estimate_pose', return_value=(True, np.eye(4), None, None, list(range(120)))):
                    for sec in (1, 13):
                        stamp = SimpleNamespace(sec=sec, nanosec=0)
                        msg = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
                        map_node.MapNode.keyframe_mapping(node, msg, msg, msg)
                self.assertEqual(search.call_count, int(enabled))
                self.assertEqual(node.isam.getFactorsUnsafe().size(), 2 + int(enabled))
                self.assertEqual(len(node.pose_graph_used_pose), 2)
                self.assertEqual(node.pose_graph_trajectory_publish.call_count, 2)

    def test_relocalization_continues_when_loop_closure_disabled(self):
        stamp = SimpleNamespace(sec=1, nanosec=0)
        msg = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
        node = SimpleNamespace(
            loop_closure=False, keyframe_mapping=Mock(),
            bridge=SimpleNamespace(imgmsg_to_cv2=lambda *_args, **_kwargs: np.ones((2, 2))),
            keyframe_relocalization=Mock(return_value=(True, np.eye(4))),
            compute_transform_from_map_to_odom=Mock(),
        )
        map_node.MapNode.keyframe_callback(node, msg, msg, msg)
        node.keyframe_relocalization.assert_called_once()
        node.compute_transform_from_map_to_odom.assert_called_once()


if __name__ == '__main__':
    unittest.main()
