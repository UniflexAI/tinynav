import json
import os
import sys
import numpy as np
from dataclasses import dataclass, field, replace


@dataclass
class ObstacleConfig:
    """Height band + occupancy filters used by planning_node.build_obstacle_map.

    z-band is relative to camera height (T[2, 3]). Taller robots need a wider
    band so hanging obstacles and low walls still count as collisions.

    The values here are this fork's, measured on its rigs, and they are NOT upstream's
    (which are -0.4/0.4, 0.1, 0.2, 2). They arrived as one global config inside
    planning_node -- `94e9871` tuned min_wall_span_m to 0.05 and `95ec80a` the
    occupancy threshold -- and moved here when upstream made the config per-robot.
    The move is upstream's structure with the fork's numbers. The band below is the
    default for every robot except b2, which the operator widened to -0.6/0.6 on
    2026-09-03 -- that is what this field is per-robot for, and it is a judgement about
    a rig rather than something a merge should have done on its own.
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
    """Robot geometry + velocity limits. Body frame: +x forward, +y left, origin at
    the CONTROL CENTRE -- the point the chassis turns about, which is what planning
    drives and what cmd_vel commands. Every geometry field is measured from it: the
    camera offset by a spin fit about the yaw axis, the footprint bounds by a tape
    measure from the turning point.

    Shared between planning_node (trajectory sampling/collision footprint) and
    cmd_vel_control (final cmd_vel clamping) so both nodes read the same numbers
    for a given ROBOT_TYPE instead of keeping separate copies.
    """
    name: str
    #: Footprint bounds from the control centre; front != rear is the normal case.
    front_len: float
    rear_len: float
    half_width: float
    #: Where the camera sits in the body frame. There is deliberately no vertical
    #: term: a spin about the vertical axis cannot observe one, and the z-band that
    #: would use it is defined relative to camera height (ObstacleConfig).
    camera_fwd: float
    camera_left: float
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
        return np.array([self.camera_left, 0.0, self.camera_fwd], dtype=np.float32)

    def footprint_from_control(self):
        """Returns (front_len, rear_len, half_w) relative to control center."""
        return self.front_len, self.rear_len, self.half_width


GO2_CONFIG = RobotConfig(
    name='go2',
    front_len=0.25, rear_len=0.35, half_width=0.15,
    camera_fwd=0.30, camera_left=0.0,
    safety_radius=0.1,
)

GO2W_CONFIG = replace(GO2_CONFIG, name='go2w')

B2_CONFIG = RobotConfig(
    name='b2',
    front_len=0.44, rear_len=0.44, half_width=0.15,
    camera_fwd=0.44, camera_left=0.0,
    safety_radius=0.075,
    # min_linear_vel stays the 0.1 default. 0.2 was run on 122 on 2026-09-04 and the
    # robot could barely move: cmd_vel_control DROPS a target below this rather than
    # raising it, so [0.1, 0.2) went from creeping to standing still -- every /cmd_vel
    # sample that drive was 0.000 or exactly 0.200. Raising it needs the planner's own
    # speed floor raised with it.
    # Taller than the default band covers: the operator's call, 2026-09-03. This is
    # the same band upstream ships for b2, adopted now that someone has judged it
    # rather than as a side effect of a merge.
    obstacle=ObstacleConfig(robot_z_bottom=-0.6, robot_z_top=0.6),
)

B2W_CONFIG = replace(B2_CONFIG, name='b2w', obstacle=ObstacleConfig())

G1_CONFIG = RobotConfig(
    name='g1',
    front_len=0.15, rear_len=0.15, half_width=0.25,
    camera_fwd=0.10, camera_left=0.0,
    safety_radius=0.15,
    min_linear_vel=0.2, min_angular_vel=0.3,
)

#: This rig's measured camera offset, over the robot type's. `forward_m` / `left_m`
#: in metres, in the body frame above; any other key is the writer's own record.
CAMERA_OFFSET_PATH = '/tinynav/tinynav_db/calib/camera_offset.json'


def with_measured_camera(config, path=CAMERA_OFFSET_PATH):
    """`config` with the camera offset from `path`, or `config` itself when there is
    no file or it cannot be read. An unreadable one is said on stderr, because every
    process that plans imports this and none of them should fail to start over it."""
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return config
    try:
        doc = json.loads(text)
        return replace(config, camera_fwd=float(doc['forward_m']),
                       camera_left=float(doc['left_m']))
    except (KeyError, TypeError, ValueError) as e:
        print(f"robot_specs: ignoring {path}, not a camera offset ({e!r})", file=sys.stderr)
        return config


ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "go2").strip().lower()
try:
    _TYPE_CONFIG = globals()[f"{ROBOT_TYPE.upper()}_CONFIG"]
except KeyError:
    raise ValueError(f"Unsupported ROBOT_TYPE: {ROBOT_TYPE!r}") from None


def robot_config(path=CAMERA_OFFSET_PATH):
    """This ROBOT_TYPE's config with the rig's measurement, read from disk now."""
    return with_measured_camera(_TYPE_CONFIG, path)


#: Read once, at import: a process picks up a new measurement at its next launch.
ROBOT_CONFIG = robot_config()
