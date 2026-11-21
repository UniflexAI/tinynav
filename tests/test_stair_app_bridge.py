import base64
from types import SimpleNamespace
import cv2
import numpy as np
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from tool.stair_app_bridge import StairAppBridge


def test_original_app_grid_orientation_and_metric_bounds():
    bridge = StairAppBridge()
    scalar = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
    msg = CvBridge().cv2_to_imgmsg(cv2.applyColorMap(scalar, cv2.COLORMAP_JET), encoding='bgr8')
    bridge._on_height_map(msg)
    mask = OccupancyGrid()
    mask.info.height, mask.info.width = 2, 3
    mask.info.resolution = 0.05
    mask.info.origin.position.x, mask.info.origin.position.y = -2., 3.
    blocked = np.array([[0, 100, 0], [100, 0, 100]], dtype=np.int8)
    mask.data = blocked.ravel(order='F').tolist()
    bridge._on_obstacle_mask(mask)
    snapshot = bridge.snapshot()
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(snapshot['esdf_image']), dtype=np.uint8), 0)
    obstacle = cv2.imdecode(np.frombuffer(base64.b64decode(snapshot['obstacle_image']), dtype=np.uint8), 0)
    assert np.array_equal(decoded, np.flipud(scalar.T))
    assert np.array_equal(obstacle, np.flipud((blocked > 50).T)*255)
    assert snapshot['grid_info'] == dict(origin_x=-2., origin_y=3., resolution=mask.info.resolution, width=2, height=3)
    assert not snapshot['localized'] and snapshot['map_pose'] is None


def test_app_voxels_come_only_from_observed_occupancy_and_are_bounded():
    bridge = StairAppBridge()
    grid = np.zeros((20, 20, 20), dtype=np.float32)
    grid[1, 2, 3] = .09
    grid[2, 3, 4] = 1
    planner = SimpleNamespace(occupancy_grid=grid, resolution=.1, origin=np.array([-1., 2., 3.]))
    bridge.capture_voxels(planner)
    assert bridge.snapshot()['voxel_points'] == [dict(x=-.8, y=2.3, z=3.4)]
    grid[:] = 1
    bridge.capture_voxels(planner)
    assert 0 < len(bridge.snapshot()['voxel_points']) <= 2500
