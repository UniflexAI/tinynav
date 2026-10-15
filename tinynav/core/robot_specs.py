import os
import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict


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
class RobotConfig:
    """Robot geometry + velocity limits. Body frame: +x forward, +y left.

    Shared between planning_node (trajectory sampling/collision footprint) and
    cmd_vel_control (final cmd_vel clamping) so both nodes read the same numbers
    for a given ROBOT_TYPE instead of keeping separate copies.
    """
    name: str = 'go2'
    length: float = 0.7
    width: float = 0.3
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
        return (self.length / 2.0, self.width / 2.0)

    def footprint_from_control(self):
        """Returns (front_len, rear_len, half_w) relative to control center."""
        hl, hw = self.half_size
        return float(hl - self.control_x), float(hl + self.control_x), float(hw)


GO2_CONFIG = RobotConfig(
    name='go2',
    length=0.4, width=0.3,
    camera_x=0.2, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.2,
    obstacle=ObstacleConfig(robot_z_bottom=-0.4, robot_z_top=0.4),
)

GO2W_CONFIG = RobotConfig(
    name='go2w',
    length=0.4, width=0.3,
    camera_x=0.2, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.2,
    obstacle=ObstacleConfig(robot_z_bottom=-0.4, robot_z_top=0.4),
)

B2_CONFIG = RobotConfig(
    name='b2',
    length=1.1, width=0.5,
    camera_x=0.3, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.1,
    obstacle=ObstacleConfig(robot_z_bottom=-0.6, robot_z_top=0.6),
)

B2W_CONFIG = RobotConfig(
    name='b2w',
    length=1.1, width=0.5,
    camera_x=0.3, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.1,
    obstacle=ObstacleConfig(robot_z_bottom=-0.6, robot_z_top=0.6),
)

G1_CONFIG = RobotConfig(
    name='g1',
    length=0.3, width=0.5,
    camera_x=0.1, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.15,
    min_linear_vel=0.2, min_angular_vel=0.3,
    obstacle=ObstacleConfig(robot_z_bottom=-0.8, robot_z_top=0.6),
)

ROBOT_PRESETS = {config.name: config for config in (
    GO2_CONFIG, GO2W_CONFIG, B2_CONFIG, B2W_CONFIG, G1_CONFIG,
)}
ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "go2").strip().lower()
try:
    ROBOT_CONFIG = ROBOT_PRESETS[ROBOT_TYPE]
except KeyError:
    raise ValueError(f"Unsupported ROBOT_TYPE: {ROBOT_TYPE!r}") from None


def robot_config_from_dict(values):
    """Validate shared robot geometry and motion settings."""
    if not isinstance(values, dict):
        raise ValueError("robot configuration must be an object")
    values = dict(values)
    # Old saved configurations may contain the removed shape selector.
    values.pop("shape", None)
    values.pop("radius", None)
    obstacle = ObstacleConfig(**values.pop("obstacle", {}))
    config = RobotConfig(obstacle=obstacle, **values)
    if not isinstance(config.name, str) or config.name not in ROBOT_PRESETS:
        raise ValueError("unsupported robot name")
    for key, value in asdict(config).items():
        if key not in ("name", "obstacle") and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"invalid robot value: {key}")
    for key, value in asdict(obstacle).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid obstacle value: {key}")
    if min(config.length, config.width, config.max_linear_vel, config.max_angular_vel) <= 0:
        raise ValueError("dimensions and maximum velocities must be positive")
    if config.safety_radius < 0 or not 0 <= config.min_linear_vel <= config.max_linear_vel or not 0 <= config.min_angular_vel <= config.max_angular_vel:
        raise ValueError("invalid safety or velocity limits")
    if obstacle.robot_z_bottom >= obstacle.robot_z_top or obstacle.min_wall_span_m < 0 or obstacle.occ_threshold < 0:
        raise ValueError("invalid obstacle height band or threshold")
    if not isinstance(obstacle.dilation_cells, int) or obstacle.dilation_cells < 0:
        raise ValueError("dilation_cells must be a non-negative integer")
    return config


if "TINYNAV_ROBOT_CONFIG" in os.environ:
    ROBOT_CONFIG = robot_config_from_dict(json.loads(os.environ["TINYNAV_ROBOT_CONFIG"]))
