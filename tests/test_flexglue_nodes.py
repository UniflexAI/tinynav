"""Exercise both node adapters without starting ROS or loading GPU models."""
import inspect
import unittest
from unittest.mock import Mock

import numpy as np
import tyro

from tinynav.core.build_map_node import BuildMapArgs, BuildMapNode
from tinynav.core.map_node import MapNode


class NodeMatcherTest(unittest.TestCase):
    def test_default_selection_and_cli_option(self):
        for cls in (MapNode, BuildMapNode):
            self.assertEqual(inspect.signature(cls).parameters['feature_matcher'].default, 'lightglue')
        self.assertEqual(BuildMapArgs().feature_matcher, 'lightglue')
        self.assertEqual(tyro.cli(BuildMapArgs, args=['--feature_matcher', 'flexglue'], use_underscores=True).feature_matcher, 'flexglue')

    def test_default_match_path_preserves_lightglue_call(self):
        for cls in (MapNode, BuildMapNode):
            node = cls.__new__(cls); node.flex_glue_matcher = None
            class Light:
                async def infer(self, *args):
                    self.args = args
                    return {'match_indices': np.array([[1, -1]])}
            node.light_glue_matcher = Light()
            a = {'kpts':np.array([[[1., 2.], [3., 4.]]]), 'descps':np.zeros((1,2,4)), 'mask':np.ones((1,2,1),bool)}
            load = Mock(side_effect=AssertionError('LightGlue must not decode reference images'))
            p0,p1,matches=node.match_keypoints(a,a,image0=load)
            np.testing.assert_array_equal(matches, [[0,1]])
            np.testing.assert_array_equal(p0, [[1,2]])
            np.testing.assert_array_equal(p1, [[3,4]])
            np.testing.assert_array_equal(node.light_glue_matcher.args[-2], [848,480])
            load.assert_not_called()

    def test_selected_matcher_receives_lazy_reference_image(self):
        for cls in (MapNode, BuildMapNode):
            node=cls.__new__(cls);node.flex_glue_matcher=Mock();node.light_glue_matcher=None
            expected=(np.zeros((0,2)),np.zeros((0,2)),np.zeros((0,2),np.int64))
            node.flex_glue_matcher.match_keypoints.return_value=expected
            a,b,load,image={}, {}, Mock(), np.zeros((10,20),np.uint8)
            self.assertIs(node.match_keypoints(a,b,image0=load,image1=image), expected)
            args=node.flex_glue_matcher.match_keypoints.call_args.args
            self.assertIs(args[2],load);self.assertIs(args[3],image)


if __name__ == '__main__': unittest.main()
