"""The planning views, rebuilt outside planning_node's loop under their old topic names
and types, each only while it has a subscriber. Run one instance: two publish every view
twice. planning_node does not depend on it.
"""
import array

import cv2
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud, PointCloud2, PointField
from std_msgs.msg import Header

from tinynav.core.math_utils import msg2np
from tinynav.core.planning_node import (
    camera_to_robot_center,
    esdf_of,
    grid_from_msg,
    grid_origin,
    mask_from_msg,
)
from tinynav.core.robot_specs import ROBOT_CONFIG

ESDF_CLOUD_MAX_DIST = 1.0
ESDF_CLOUD_LAYER = 2
ESDF_CLOUD_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
]


def height_map_bgr(esdf):
    return cv2.applyColorMap(np.clip(esdf / 2.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)


def occupancy_grid_values(esdf):
    """100 inside obstacles, falling to 0 at 0.5 m of clearance."""
    return np.where(esdf <= 0.00, 100, np.clip(((1 - esdf / 0.5) * 120).astype(int), 0, 120)).astype(np.int8)


def voxel_points(grid, origin, resolution):
    return origin + np.argwhere(grid > 0.1) * resolution


def esdf_cloud_points(esdf, origin, resolution):
    X, Y = esdf.shape
    gx, gy = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    v = np.uint8((1 - np.clip(esdf.ravel(), 0, ESDF_CLOUD_MAX_DIST) / ESDF_CLOUD_MAX_DIST) * 255)
    colors = cv2.applyColorMap(v.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    points = np.zeros(X * Y, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.uint32)])
    points["x"] = gx.ravel() * resolution + origin[0]
    points["y"] = gy.ravel() * resolution + origin[1]
    points["z"] = ESDF_CLOUD_LAYER * resolution + origin[2]
    points["rgb"] = (colors[:, 2].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 0]
    return points


def footprint_points(T, samples=21):
    """The control-frame rectangle's outline in world, `samples` points per edge."""
    forward, left, center = T[:3, 2], T[:3, 0], camera_to_robot_center(T)
    fl, rl, hw = ROBOT_CONFIG.footprint_from_control()
    corners = [center + forward * fl + left * hw, center + forward * fl - left * hw,
               center - forward * rl - left * hw, center - forward * rl + left * hw]
    t = (np.arange(samples) / (samples - 1))[:, None]
    return np.concatenate([(1.0 - t) * corners[i] + t * corners[(i + 1) % 4] for i in range(4)])


class PlanningVisNode(Node):
    def __init__(self):
        super().__init__("planning_vis_node")
        self.bridge = CvBridge()
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, "/planning/occupancy_grid", 10)
        self.voxels_pub = self.create_publisher(PointCloud2, "/planning/occupied_voxels", 10)
        self.voxels_esdf_pub = self.create_publisher(PointCloud2, "/planning/occupied_voxels_with_esdf", 10)
        self.footprint_pub = self.create_publisher(PointCloud, "/planning/footprint", 10)

        self.mask_msg = None
        self.decoded = None  # the latest mask decoded, with its ESDF: built once per frame
        self.grid_msg = None
        self.grid_sub = None
        self.create_subscription(OccupancyGrid, "/planning/obstacle_mask", self.mask_callback, 10)
        self.create_subscription(Odometry, "/slam/odometry_visual", self.odom_callback, 10)
        # planning sends the grid only while it has a reader, so read it only while a cloud is watched.
        self.create_timer(1.0, self.sync_grid_subscription)

    @staticmethod
    def _wanted(pub):
        return pub.get_subscription_count() > 0

    def sync_grid_subscription(self):
        wanted = self._wanted(self.voxels_pub) or self._wanted(self.voxels_esdf_pub)
        if wanted and self.grid_sub is None:
            self.grid_sub = self.create_subscription(Image, "/planning/occupancy_3d", self.grid_callback, 1)
        elif not wanted and self.grid_sub is not None:
            self.destroy_subscription(self.grid_sub)
            self.grid_sub = None

    def decode(self):
        if self.decoded is None:
            mask, resolution, origin = mask_from_msg(self.mask_msg)
            self.decoded = (resolution, origin, esdf_of(mask, resolution))
        return self.decoded

    def mask_callback(self, msg):
        self.mask_msg, self.decoded = msg, None
        if self._wanted(self.height_map_pub):
            img = self.bridge.cv2_to_imgmsg(height_map_bgr(self.decode()[2]), encoding="bgr8")
            img.header = msg.header
            self.height_map_pub.publish(img)
        if self._wanted(self.occupancy_grid_pub):
            grid = OccupancyGrid(header=msg.header, info=msg.info)
            grid.data = array.array("b", occupancy_grid_values(self.decode()[2]).ravel(order="F").tobytes())
            self.occupancy_grid_pub.publish(grid)
        self.publish_clouds()

    def grid_callback(self, msg):
        self.grid_msg = msg
        self.publish_clouds()

    def publish_clouds(self):
        """Once the mask and the grid of the same planning step have both arrived."""
        if self.mask_msg is None or self.grid_msg is None or self.grid_msg.header.stamp != self.mask_msg.header.stamp:
            return
        grid, self.grid_msg = grid_from_msg(self.grid_msg), None
        resolution, mask_origin, esdf = self.decode()
        origin = grid_origin(mask_origin, resolution, grid.shape[2])
        header = self.mask_msg.header
        if self._wanted(self.voxels_pub):
            self.voxels_pub.publish(pc2.create_cloud_xyz32(header, voxel_points(grid, origin, resolution)))
        if self._wanted(self.voxels_esdf_pub):
            points = esdf_cloud_points(esdf, origin, resolution)
            self.voxels_esdf_pub.publish(pc2.create_cloud(header, ESDF_CLOUD_FIELDS, points))

    def odom_callback(self, msg):
        if not self._wanted(self.footprint_pub):
            return
        T, _ = msg2np(msg)
        cloud = PointCloud(header=Header(stamp=msg.header.stamp, frame_id="world"))
        cloud.points = [Point32(x=float(x), y=float(y), z=float(z)) for x, y, z in footprint_points(T)]
        self.footprint_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = PlanningVisNode()
    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
