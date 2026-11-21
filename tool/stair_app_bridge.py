"""Feed original TinyNav App contracts from actual simulator node outputs.

Reuse BackendNode's serializers without constructing its robot lifecycle or
starting a second planner. All callbacks below run in the simulation loop.
"""
import threading
import numpy as np
from app.backend.node_manager import BackendNode


class StairAppBridge:
    _on_height_map = BackendNode._on_height_map
    _on_obstacle_mask = BackendNode._on_obstacle_mask
    _on_footprint = BackendNode._on_footprint
    _on_trajectory_path = BackendNode._on_trajectory_path
    _on_nav_target_pose = BackendNode._on_nav_target_pose
    _on_slam_odom = BackendNode._on_slam_odom
    _odom_to_dict = staticmethod(BackendNode._odom_to_dict)
    snapshot = BackendNode.get_planning_snapshot

    def __init__(self):
        self._lock = threading.Lock()
        self._esdf_bytes = self._obstacle_bytes = b''
        self._localized = False
        self._odom_pose = self._odom_pose_at_kf = self._map_pose = None
        self._nav_target_pose = self._grid_info = None
        self._global_path, self._trajectory, self._footprint, self._voxel_points = [], [], [], []
        self.pose_callbacks = []
        self.T_map_from_odom=None

    def _transform_path_via_tf(self, path):
        if self.T_map_from_odom is None:return []
        inv=np.linalg.inv(self.T_map_from_odom)
        return [dict(x=float(q[0]),y=float(q[1])) for q in (inv@np.array([p['x'],p['y'],0.,1.]) for p in path)]

    def capture_voxels(self, planner):
        occupied = np.argwhere(planner.occupancy_grid > 0.1)
        if len(occupied):
            occupied = occupied[::max(1, int(np.ceil(len(occupied)/2500)))]
            points = occupied * planner.resolution + planner.origin
            self._voxel_points = [dict(x=float(x), y=float(y), z=float(z)) for x, y, z in points]
        else:
            self._voxel_points = []
