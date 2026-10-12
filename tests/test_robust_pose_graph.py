import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation
from tinynav.core import build_map_node as mapping


def pose(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


class MemoryDB:
    def __init__(self):
        self.entries = {}

    def set_entry(self, timestamp, **data):
        self.entries.setdefault(timestamp, {}).update(data)

    def set_semantic_embedding(self, timestamp, embedding):
        self.entries[timestamp]['semantic_embedding'] = embedding

    def get_embedding(self, timestamp):
        return self.entries[timestamp]['embedding']

    def get_depth_embedding_features_images(self, timestamp):
        entry = self.entries[timestamp]
        return entry['depth'], entry['embedding'], entry['features'], None, None


def make_mapper():
    async def infer(image):
        return {}
    async def infer_semantic(image):
        return np.ones(2)
    async def infer_descriptors(image):
        return np.ones(2), np.ones((1, 2))
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1/30.]*3+[1/10.]*3))
    params = gtsam.ISAM2Params()
    params.setFactorization("QR")
    params.setRelinearizeThreshold(0.001)
    params.relinearizeSkip = 1
    params.setOptimizationParams(gtsam.ISAM2DoglegParams())
    node = SimpleNamespace(
        isam=gtsam.ISAM2(params), odometry_noise=noise,
        loop_noise=gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(5.), noise),
        odom={}, pose_graph_used_pose={}, last_keyframe_timestamp=None,
        timer_logger=None, stage_timer=mapping.StageTimer(),
        global_frames_ratio=1.1, _global_prev_num_frames=0,
        dinov2_model=SimpleNamespace(infer_global_and_patch_tokens=infer_descriptors),
        semantic_embedder=SimpleNamespace(encode_image=infer_semantic),
        K=np.eye(3), db=MemoryDB(),
        bridge=SimpleNamespace(imgmsg_to_cv2=lambda msg, **kwargs: np.ones((2, 2))),
        get_embeddings=lambda image: np.ones(2),
        super_point_extractor=SimpleNamespace(infer=infer),
        loop_similarity_threshold=.9, loop_top_k=1,
        match_keypoints=lambda a, b: (np.zeros((120, 2)), np.zeros((120, 2)), np.zeros((120, 2))),
        publish_local_map=Mock(), publish_all_transforms=Mock(),
        pose_graph_trajectory_publish=Mock(), get_logger=Mock(),
    )

    node.detect_loop_closure = lambda t, factors: mapping.BuildMapNode.detect_loop_closure(node, t, factors)
    node.maybe_run_global_refinement = lambda: mapping.BuildMapNode.maybe_run_global_refinement(node)
    return node


def process_keyframe(node, timestamp, odometry, loops=(), inliers=120):
    stamp = SimpleNamespace(sec=int(timestamp//10**9), nanosec=int(timestamp%10**9))
    msg = SimpleNamespace(header=SimpleNamespace(stamp=stamp), pose_np=odometry)
    valid = [t for t in node.pose_graph_used_pose if t+10*10**9 < timestamp]
    candidates = [(valid.index(previous), .95) for previous, _ in loops]
    measurements = [(True, T, None, None, list(range(inliers))) for _, T in loops]
    with patch.object(mapping, 'msg2np', side_effect=lambda msg: (msg.pose_np, np.zeros(3))), \
         patch.object(mapping, 'find_loop', return_value=candidates), \
         patch.object(mapping, 'estimate_pose', side_effect=measurements), \
         patch.object(mapping, 'depth_to_cloud', return_value=np.empty((0, 3))):
        mapping.BuildMapNode.process(node, msg, msg, msg, msg)


class DirectISAMTests(unittest.TestCase):
    def setUp(self):
        self.node = make_mapper()
        self.base = 1800000000000000000

    def chain(self, step=1.0):
        stamps = [self.base+i*10**9 for i in range(21)]
        for i, stamp in enumerate(stamps[:-1]):
            process_keyframe(self.node, stamp, pose(i*step))
        return stamps

    def test_timestamp_keys_fixed_anchor_and_duplicate_callback(self):
        process_keyframe(self.node, self.base, pose(0))
        process_keyframe(self.node, self.base+10**9, pose(1))
        count = self.node.isam.getFactorsUnsafe().size()
        process_keyframe(self.node, self.base+10**9, pose(50))
        self.assertEqual(self.node.isam.getFactorsUnsafe().size(), count)
        self.assertEqual(set(self.node.pose_graph_used_pose), {self.base, self.base+10**9})
        self.assertTrue(self.node.isam.valueExists(self.base))
        np.testing.assert_allclose(self.node.pose_graph_used_pose[self.base], pose(0), atol=1e-10)

    def test_direct_factor_types_and_false_loop(self):
        stamps = self.chain()
        process_keyframe(self.node, stamps[-1], pose(20), [(stamps[0], pose(0))])
        factors = self.node.isam.getFactorsUnsafe()
        self.assertEqual(factors.size(), 22)
        self.assertIsInstance(factors.at(1).noiseModel(), gtsam.noiseModel.Diagonal)
        self.assertIsInstance(factors.at(21).noiseModel(), gtsam.noiseModel.Robust)
        self.assertEqual(list(factors.at(21).keys()), [stamps[0], stamps[-1]])
        result = mapping.refine_pose_graph(self.node.isam)
        self.assertLess(abs(result[stamps[-1]][0, 3]-20), .3)

    def test_valid_loop_corrects_drift(self):
        stamps = self.chain(1.01)
        process_keyframe(self.node, stamps[-1], pose(20.2), [(stamps[0], pose(20))])
        result = mapping.refine_pose_graph(self.node.isam)
        self.assertLess(abs(result[stamps[-1]][0, 3]-20), .02)

    def test_pnp_inlier_requirement_preserved(self):
        stamps = self.chain()
        process_keyframe(self.node, stamps[-1], pose(20), [(stamps[0], pose(0))], inliers=99)
        self.assertEqual(self.node.isam.getFactorsUnsafe().size(), 21)

    def test_transform_direction_with_rotation_and_translation(self):
        previous = pose(2)
        previous[:3, :3] = Rotation.from_euler('xyz', [.3, -.2, .8]).as_matrix()
        relative = pose(.5)
        relative[:3, :3] = Rotation.from_euler('xyz', [-.2, .4, -.1]).as_matrix()
        relative[1:3, 3] = [-.3, 1.]
        process_keyframe(self.node, self.base, previous)
        process_keyframe(self.node, self.base+10**9, previous@relative)
        np.testing.assert_allclose(self.node.pose_graph_used_pose[self.base+10**9], previous@relative, atol=1e-8)
        measurement = self.node.isam.getFactorsUnsafe().at(1).measured().matrix()
        np.testing.assert_allclose(measurement, relative, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
