"""Robot configuration shared across planning, mapping, and editing tools.

Body frame convention: +x forward, +y left, +z up.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class RobotConfig:
    """Robot geometry. Body frame: +x forward, +y left."""
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

B2_CONFIG = RobotConfig(
    name='b2', shape='square',
    length=1.0, width=0.5,
    camera_x=0.5, camera_y=0.0,
    control_x=-0.5, control_y=0.0,
    safety_radius=0.1,
)

ROBOT_CONFIGS = {
    'go2': GO2_CONFIG,
    'b2': B2_CONFIG,
}

# ── 换机器人只改这一行 ──
ACTIVE_ROBOT = 'go2'
# ──────────────────────

ROBOT_CONFIG = ROBOT_CONFIGS[ACTIVE_ROBOT]


def get_robot_config(name: str = 'go2') -> RobotConfig:
    """Look up a robot config by name. Falls back to GO2_CONFIG."""
    return ROBOT_CONFIGS.get(name, GO2_CONFIG)


def camera_to_body_position(cam_pos: np.ndarray, cam_R: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Convert a camera-frame world position to robot body-center world position.

    Given a camera pose (position + rotation in world frame), compute where
    the robot's control center would be, using the camera-to-body offset
    defined in ``robot_config``.

    Args:
        cam_pos: Camera position in world frame, shape (3,).
        cam_R: Camera rotation matrix in world frame, shape (3, 3).
        robot_config: RobotConfig with camera/control offsets.

    Returns:
        Robot body-center position in world frame, shape (3,).
    """
    return cam_pos - cam_R @ robot_config.cam_offset_3d


def body_to_camera_position(body_pos: np.ndarray, cam_R: np.ndarray, robot_config: RobotConfig) -> np.ndarray:
    """Convert a robot body-center world position to camera world position.

    Inverse of :func:`camera_to_body_position`.

    Args:
        body_pos: Robot body-center position in world frame, shape (3,).
        cam_R: Camera rotation matrix in world frame, shape (3, 3).
        robot_config: RobotConfig with camera/control offsets.

    Returns:
        Camera position in world frame, shape (3,).
    """
    return body_pos + cam_R @ robot_config.cam_offset_3d
