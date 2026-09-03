import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ObstacleConfig:
    """Height band + occupancy filters used by planning_node.build_obstacle_map.

    z-band is relative to camera height (T[2, 3]). Taller robots need a wider
    band so hanging obstacles and low walls still count as collisions.

    The values here are this fork's, measured on its rigs, and they are NOT upstream's
    (which are -0.4/0.4, 0.1, 0.2, 2). They arrived as one global config inside
    planning_node -- `94e9871` tuned min_wall_span_m to 0.05 and `95ec80a` the
    occupancy threshold -- and moved here when upstream made the config per-robot.
    The move is upstream's structure with the fork's numbers. No robot below overrides
    the band: upstream gives the taller ones -0.6/0.6, and adopting that would widen
    what b2 treats as an obstacle -- a measured change, not a merge. The field is here
    per-robot when someone measures it.
    """
    robot_z_bottom: float = -0.45
    robot_z_top: float = 0.2
    occ_threshold: float = 0.05
    min_wall_span_m: float = 0.05
    #: Only cells whose lowest occupied voxel sits within this of robot_z_bottom are
    #: span-filtered: walls span, stair risers and ground bumps do not. Cells starting
    #: above the band are floating obstacles and keep a single-voxel noise floor.
    ground_band_m: float = 0.3
    dilation_cells: int = 0


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
    length=0.6, width=0.3,
    camera_x=0.35, camera_y=0.0,
    control_x=0.05, control_y=0.0,
    safety_radius=0.1,
)

GO2W_CONFIG = RobotConfig(
    name='go2w', shape='square',
    length=0.6, width=0.3,
    camera_x=0.35, camera_y=0.0,
    control_x=0.05, control_y=0.0,
    safety_radius=0.1,
)

B2_CONFIG = RobotConfig(
    name='b2', shape='square',
    length=0.8, width=0.3,
    camera_x=0.5, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.1,
)

B2W_CONFIG = RobotConfig(
    name='b2w', shape='square',
    length=0.8, width=0.3,
    camera_x=0.5, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.1,
)

G1_CONFIG = RobotConfig(
    name='g1', shape='square',
    length=0.3, width=0.5,
    camera_x=0.1, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.15,
    min_linear_vel=0.2, min_angular_vel=0.3,
)

ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "go2").strip().lower()
try:
    ROBOT_CONFIG = globals()[f"{ROBOT_TYPE.upper()}_CONFIG"]
except KeyError:
    raise ValueError(f"Unsupported ROBOT_TYPE: {ROBOT_TYPE!r}") from None
