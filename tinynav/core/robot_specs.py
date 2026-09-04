import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ObstacleConfig:
    """Height band + occupancy filters used by planning_node.build_obstacle_map.

    z-band is relative to camera height (T[2, 3]). Taller robots need a wider
    band so hanging obstacles and low walls still count as collisions.
    """
    robot_z_bottom: float = -0.4
    robot_z_top: float = 0.4
    occ_threshold: float = 0.1
    min_wall_span_m: float = 0.2
    dilation_cells: int = 2


@dataclass
class ObjectDetectionConfig:
    """Closed-set object detector settings for planning_node's object-voxel grid.

    kept_class_names empty means keep every COCO class the detector emits;
    otherwise only these names are kept (see models_trt.COCO_CLASS_NAMES for
    the full list of names the detector can emit).
    """
    enabled: bool = True
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    ttl_frames: int = 20
    detect_every_n_frames: int = 1
    kept_class_names: tuple[str, ...] = ("person", "car")


@dataclass
class RobotConfig:
    """Robot geometry + velocity limits. Body frame: +x forward, +y left.

    Shared between planning_node (trajectory sampling/collision footprint) and
    cmd_vel_control (final cmd_vel clamping) so both nodes read the same numbers
    for a given ROBOT_TYPE instead of keeping separate copies.
    """
    name: str = 'go2'
    shape: str = 'square'
    length: float = 0.7
    width: float = 0.3
    radius: float = 0.3
    camera_x: float = 0.35
    camera_y: float = 0.0
    control_x: float = 0.0
    control_y: float = 0.0
    safety_radius: float = 0.1
    # Bounds used to constrain trajectory-library velocity sampling and to clamp
    # the final published cmd_vel. Placeholder values, same for every robot until
    # real per-platform min/max linear & angular speeds are measured.
    min_linear_vel: float = 0.1
    max_linear_vel: float = 1.0
    min_angular_vel: float = 0.1
    max_angular_vel: float = 0.75
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    object_detection: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)

    @property
    def cam_offset_3d(self):
        """Offset [left, up, forward] from control center to camera in body frame."""
        return np.array([self.camera_y - self.control_y, 0.0, self.camera_x - self.control_x], dtype=np.float32)

    @property
    def half_size(self):
        if self.shape == 'circle':
            return (self.radius, self.radius)
        return (self.length / 2.0, self.width / 2.0)

    def footprint_from_control(self):
        """Returns (front_len, rear_len, half_w) relative to control center."""
        hl, hw = self.half_size
        return float(hl - self.control_x), float(hl + self.control_x), float(hw)


GO2_CONFIG = RobotConfig(
    name='go2', shape='square',
    length=0.4, width=0.3,
    camera_x=0.2, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.2,
    obstacle=ObstacleConfig(robot_z_bottom=-0.4, robot_z_top=0.4),
)

GO2W_CONFIG = RobotConfig(
    name='go2w', shape='square',
    length=0.4, width=0.3,
    camera_x=0.2, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.2,
    obstacle=ObstacleConfig(robot_z_bottom=-0.4, robot_z_top=0.4),
)

B2_CONFIG = RobotConfig(
    name='b2', shape='square',
    length=1.1, width=0.5,
    camera_x=0.3, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.1,
    obstacle=ObstacleConfig(robot_z_bottom=-0.6, robot_z_top=0.6),
)

B2W_CONFIG = RobotConfig(
    name='b2w', shape='square',
    length=1.1, width=0.5,
    camera_x=0.3, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.1,
    obstacle=ObstacleConfig(robot_z_bottom=-0.6, robot_z_top=0.6),
)

G1_CONFIG = RobotConfig(
    name='g1', shape='square',
    length=0.3, width=0.5,
    camera_x=0.1, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.15,
    min_linear_vel=0.2, min_angular_vel=0.3,
    obstacle=ObstacleConfig(robot_z_bottom=-0.8, robot_z_top=0.6),
)

ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "go2").strip().lower()
try:
    ROBOT_CONFIG = globals()[f"{ROBOT_TYPE.upper()}_CONFIG"]
except KeyError:
    raise ValueError(f"Unsupported ROBOT_TYPE: {ROBOT_TYPE!r}") from None
