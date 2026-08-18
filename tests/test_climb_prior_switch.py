"""map_node's `climb_prior` parameter — the switch back to upstream's obstacle handling.

The climb prior RELAXES the planner's span filter inside the region, so a falsely
labelled band makes flat ground more permissive, not less. Until the labels are
trustworthy the operator needs to be able to take the prior out of the loop entirely,
and "out of the loop" has to mean no labels loaded, none baked, and an empty region on
the wire — an empty region is what the planner reads as "strict default everywhere".

The methods are compiled on their own (see `_method`) so this needs neither rclpy nor
the TRT engines map_node imports at module scope.
"""
from __future__ import annotations

import ast
import os
import unittest

import numpy as np

_MAP_NODE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'tinynav', 'core', 'map_node.py')


def _method(name: str, **module_globals):
    """One MapNode method, compiled on its own so importing the module (and with it
    rclpy + TensorRT) isn't needed. `module_globals` stands in for the module
    constants and imports the method reads."""
    tree = ast.parse(open(_MAP_NODE).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == 'MapNode')
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {'os': os, 'np': np, **module_globals}
    exec(compile(module, _MAP_NODE, 'exec'), ns)   # noqa: S102 - our own source
    return ns[name]


class _Node:
    def __init__(self, climb_prior: bool = True):
        self.climb_index = 'stale'
        self.speed_index = None
        self._climb_prior = climb_prior
        self.T_from_map_to_odom = None
        self.latest_odom_pose = None
        self.lines: list[str] = []

    def get_logger(self):
        return self

    def info(self, msg):
        self.lines.append(msg)

    warning = error = info


def _load(map_path, *, prior: bool, bake=None, index=None):
    node = _Node(climb_prior=prior)
    fn = _method('load_map_priors',
                 bake_path_climb=bake or (lambda p: 'BAKED'),
                 PathClimbIndex=index,
                 PathSpeedIndex=None,
                 n_climbing=lambda pts: 0)
    fn(node, map_path)
    return node


class ClimbPriorSwitchTest(unittest.TestCase):
    def test_off_loads_nothing_and_bakes_nothing(self):
        baked = []
        node = _load('/no/such/map', prior=False,
                     bake=lambda p: baked.append(p) or 'BAKED')
        self.assertIsNone(node.climb_index)
        self.assertEqual(baked, [])          # a switched-off prior writes no labels
        self.assertTrue(any('climb_prior=false' in line for line in node.lines))

    def test_on_still_bakes_a_map_with_no_labels(self):
        """The switch must not have changed the default path."""
        baked = []
        node = _load('/no/such/map', prior=True,
                     bake=lambda p: baked.append(p) or 'BAKED')
        self.assertEqual(baked, ['/no/such/map'])
        self.assertIsNone(node.climb_index)   # bake could not write one either

    def test_the_default_is_on(self):
        """Reading the constant's own source: switching the fleet's default off by
        accident would silently drop stair support on every rig."""
        tree = ast.parse(open(_MAP_NODE).read())
        node = next(n for n in tree.body
                    if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], 'id', None) == 'CLIMB_PRIOR_DEFAULT')
        self.assertTrue(eval(ast.unparse(node.value), {}))  # noqa: S307

    def test_no_index_means_an_empty_region_not_a_silent_publisher(self):
        """The planner treats silence as staleness and an empty cloud as 'no region
        here'. With the prior off it must still publish, or the two are indistinguishable
        from a crashed map node."""
        region, stairs = [], []
        fn = _method('tick_map_priors',
                     PointCloud=lambda: type('M', (), {'header': type('H', (), {
                         'stamp': None, 'frame_id': ''})(), 'points': []})(),
                     Point32=None, Bool=lambda data: data,
                     CLIMB_REGION_CULL_M=3.5)
        node = _Node()
        node.climb_index = None
        node.climb_region_pub = type('P', (), {'publish': lambda _s, m: region.append(m)})()
        node.on_stairs_pub = type('P', (), {'publish': lambda _s, m: stairs.append(m)})()
        node.get_clock = lambda: type('C', (), {
            'now': staticmethod(lambda: type('T', (), {
                'to_msg': staticmethod(lambda: None)})())})()
        fn(node)
        self.assertEqual(len(region), 1)
        self.assertEqual(region[0].points, [])
        self.assertEqual(stairs, [False])


class ClimbIndexLoadTest(unittest.TestCase):
    """A map with no climb labels must degrade to "strict everywhere", not take
    map_node down with it.

    `bake` refuses to raise on a map it cannot label — a map-load path must never
    fail on a missing prior — so it reports "no poses.npy" by *returning* a message.
    That lands on the success path here, which the `except` below it cannot see. The
    node then ran on to log how many samples were labelled, on a None index: an
    AttributeError in __init__ (and in the reload path), i.e. a map that kills the
    node instead of one that plans strictly."""

    def _load(self, tmp, *, index=None, load_raises=None):
        logs: list[tuple[str, str]] = []

        def _load_index(path):
            if load_raises:
                raise load_raises
            return index

        load = _method(
            'load_map_priors',
            # What bake really does for a map it cannot label: report, do not raise.
            bake_path_climb=lambda p: 'path_climb.npy: skipped, no poses.npy',
            PathClimbIndex=type('I', (), {'load': staticmethod(_load_index)}),
            n_climbing=lambda pts: sum(1 for p in pts if p),
            PathSpeedIndex=None,
        )
        node = type('N', (), {})()
        node.speed_index = None
        node._climb_prior = True   # the switch off is ClimbPriorSwitchTest's business
        node.get_logger = lambda: type('L', (), {
            'info': lambda s, m: logs.append(('info', m)),
            'warning': lambda s, m: logs.append(('warning', m)),
            'error': lambda s, m: logs.append(('error', m)),
        })()
        load(node, tmp)
        return node, logs

    def test_a_map_with_no_poses_leaves_no_index_and_does_not_raise(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            node, logs = self._load(tmp)
            self.assertIsNone(node.climb_index)
            self.assertTrue(any(lvl == 'warning' and 'strict everywhere' in m
                                for lvl, m in logs), logs)

    def test_a_corrupt_file_is_still_an_error_not_a_warning(self):
        # The pre-existing branch: a file that exists but will not parse. It must stay
        # distinguishable from "this map was never labelled".
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, 'path_climb.npy'), 'wb').close()
            node, logs = self._load(tmp, load_raises=ValueError('bad header'))
            self.assertIsNone(node.climb_index)
            self.assertTrue(any(lvl == 'error' and 'bad header' in m
                                for lvl, m in logs), logs)

    def test_labels_present_are_reported_and_kept(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'path_climb.npy')
            open(path, 'wb').close()
            idx = type('X', (), {'pts': [True, False, True]})()
            node, logs = self._load(tmp, index=idx)
            self.assertIs(node.climb_index, idx)
            self.assertTrue(any(lvl == 'info' and '2/3' in m for lvl, m in logs), logs)


if __name__ == '__main__':
    unittest.main()
