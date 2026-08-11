import numpy as np
from dataclasses import dataclass


@dataclass
class RobotConfig:
    """Robot geometry + velocity limits. Body frame: +x forward, +y left.

    Shared between planning_node (trajectory sampling/collision footprint) and
    cmd_vel_control (final cmd_vel clamping) so both nodes read the same numbers
    for a given --robot-model instead of keeping separate copies.
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
)

GO2W_CONFIG = RobotConfig(
    name='go2w', shape='square',
    length=0.4, width=0.3,
    camera_x=0.2, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.2,
)

B2_CONFIG = RobotConfig(
    name='b2', shape='square',
    length=1.1, width=0.5,
    camera_x=0.3, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.1,
)

B2W_CONFIG = RobotConfig(
    name='b2w', shape='square',
    length=1.1, width=0.5,
    camera_x=0.3, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.1,
)

G1_CONFIG = RobotConfig(
    name='g1', shape='square',
    length=0.3, width=0.5,
    camera_x=0.1, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.15,
    min_linear_vel=0.2,min_angular_vel=0.3
)

ROBOT_CONFIGS = {
    GO2_CONFIG.name: GO2_CONFIG,
    GO2W_CONFIG.name: GO2W_CONFIG,
    B2_CONFIG.name: B2_CONFIG,
    B2W_CONFIG.name: B2W_CONFIG,
    G1_CONFIG.name: G1_CONFIG,
}
