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
    The move is upstream's structure with the fork's numbers. Since 2026-11-12 each
    model's bottom sits FLOOR_CLEARANCE_M above the floor at its standing camera height
    (the configs below say where each height comes from) -- a bottom set for go2's
    camera left taller bodies blind to anything under ~0.1 m -- and the top is the
    robot's highest point plus 0.05 m, so what it would pass under is not an obstacle.
    """
    robot_z_bottom: float = -0.45
    #: Every quadruped carries the same navcore box (130 x 130 x 55 mm) with the Looper
    #: in it, the lens 22.5 mm under the box's top in all three [CAD]; the highest
    #: point is 0.022-0.028 m over the lens (go2 0.028, as2 0.0225, b2 0.028 without
    #: the PRT0006 part one rig carries on top). 0.08 is that plus 0.05, for every
    #: quadruped (the operator's call, 2026-11-12; until then 0.2, and on b2 0.6, then
    #: 0.2 from 2026-11-13).
    robot_z_top: float = 0.08
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
    the robot's ROOT FRAME ORIGIN -- the URDF's base_link (pelvis on g1), which sits
    midway between the hip joints and is what planning drives and the chassis turns
    about. Every geometry field is measured from it (sources per model below): the
    footprint is the STANDING outline, legs, wheels and the navcore/Looper mount
    included; the camera is the Looper's left lens, whose pose /slam/odometry_visual is.
    A rig's spin calibration (CAMERA_OFFSET_PATH) replaces the camera numbers with
    ones measured from its yaw centre.

    Shared between planning_node (trajectory sampling/collision footprint) and
    cmd_vel_control (final cmd_vel clamping) so both nodes read the same numbers
    for a given ROBOT_TYPE instead of keeping separate copies.
    """
    name: str
    #: Standing footprint bounds from the root frame origin; front != rear is normal.
    front_len: float
    rear_len: float
    half_width: float
    #: Where the camera (the Looper's left lens) sits in the body frame: metres ahead
    #: of and to the left of the root frame origin. There is deliberately no vertical
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
    #: The slowest turn in place, where the gait needs more than min_angular_vel to
    #: take a step at all (`FREE_GAIT_FLOORS`); 0 is no floor beyond it.
    min_spin_vel: float = 0.0
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)

    @property
    def cam_offset_3d(self):
        """The offset from the root frame origin to the camera, in the CAMERA's frame
        (x right, y down, z forward -- the convention every caller multiplies it by:
        `T_world_camera[:3, :3] @ cam_offset_3d`). Left is the camera's -x, so
        `camera_left` goes in negated, as pilot/geometry.camera_from_robot has it.
        (Until 2026-11-12 it went in un-negated; every model's camera_left was 0 then,
        so nothing read it the wrong way round.)"""
        return np.array([-self.camera_left, 0.0, self.camera_fwd], dtype=np.float32)

    def footprint_from_control(self):
        """Returns (front_len, rear_len, half_w) relative to control center."""
        return self.front_len, self.rear_len, self.half_width

    @property
    def arrival_radius(self):
        """Radius of the two arrival circles (arrival_distance): two half-widths."""
        return 2.0 * self.half_width

    @property
    def planned_arrival_radius(self):
        """The radius planning rolls its trajectories into: one and a half half-widths,
        tighter than arrival_radius, so a row has to head for the target, not graze the
        circle."""
        return 1.5 * self.half_width

    def arrival_distance(self, T_camera, target_xy):
        """XY distance from `target_xy` to the nearer of the two arrival circle
        centres, the camera and the control centre, for a camera pose."""
        camera = T_camera[:3, 3]
        control = camera - T_camera[:3, :3] @ self.cam_offset_3d
        target = np.asarray(target_xy, dtype=float)[:2]
        return min(float(np.linalg.norm(camera[:2] - target)),
                   float(np.linalg.norm(control[:2] - target)))


# Where the numbers below come from (checked 2026-11-12; all distances from the root
# frame origin, standing):
#   [CAD]  the Looper mount assemblies, STEP files from the mechanical team: go2.stp,
#          b2_asm_xianggang_asm.stp, AS2.stp (a wheeled variant mounts like its legged
#          one). Read with OpenCascade; the root origin located as the centre of the
#          four hip/thigh motors, whose spacing matches the URDF's hip spacing (go2
#          0.386 m vs 0.387, b2 0.656 vs 0.657, as2 0.424 vs 0.424); the camera is the
#          insight9 module's left lens window (100 mm baseline).
#   [G1-URDF] nomad_description/nomad_29dof.urdf (insight7_joint on torso_link) and
#          nomad_g1b_full.glb, which agree.
#   [ROS]  github.com/unitreerobotics/unitree_ros @725b59b robots/<model>_description,
#          forward kinematics in the standing pose.
#   [SPEC] unitree.com/<model> product pages: standing L x W x H.
#   [DOC]  Unitree's developer docs, https://doc-cdn.unitree.com/<s>/<p>/en/<s>_<p>_en:
#          the StandUp body height -- go2 6_814 (0.33 m), go2-w 15_283 (0.44 m),
#          b2 9_136 (0.55 m).
#   [SDK2] github.com/unitreerobotics/unitree_sdk2 @aaf69cb.
#   [RLG]  github.com/unitreerobotics/unitree_rl_gym @276801e.
# A band bottom is the camera's standing height above the floor less
# FLOOR_CLEARANCE_M, negated: the origin's standing height plus the camera's height
# above the origin [CAD].

GO2_CONFIG = RobotConfig(
    name='go2',
    # [CAD] 0.342 / 0.325 / 0.169; [ROS] 0.33-0.34 / 0.36 (rear knees) / 0.16;
    # [SPEC] 700 x 310 x 400 mm.
    front_len=0.34, rear_len=0.36, half_width=0.17,
    # [CAD] left lens 0.304 ahead, 0.050 left, 0.127 above the origin.
    camera_fwd=0.304, camera_left=0.050,
    safety_radius=0.1,
    # Measured on looper 2026-09-29, in place: 0.19 rad/s does not yaw at all, 0.20 does.
    min_angular_vel=0.2,
    # Measured on 65 2026-10-16, in place, free gait: 1.0 rad/s drives 0.87 and 3.0
    # drives 2.87, so the chassis is not what bounds it. 2.0 swung the body into the
    # wall turning in place in tight spots; 1.0 is the operator's call. go2w and b2
    # take the same number unmeasured.
    max_angular_vel=1.0,
    # The band keeps the default -0.45: measured on 65, whose camera is 0.515 m up
    # (FLOOR_CLEARANCE_M). [DOC] 0.33 + [CAD] 0.127 would put it at 0.457 m (-0.39);
    # the rig's measurement wins over the documented StandUp height.
)

GO2W_CONFIG = replace(
    GO2_CONFIG, name='go2w',
    # [SPEC] 700 x 430 x 500 mm: the wheels stand out to 0.215 a side;
    # [ROS] go2w ~0.35 / 0.35-0.38.
    front_len=0.35, rear_len=0.37, half_width=0.215,
    # The camera mounts as on go2 [CAD].
    # The wheeled base's turn rates were not measured, so it keeps the default.
    min_angular_vel=RobotConfig.min_angular_vel,
    # [DOC 15_283] StandUp 0.44 m + [CAD] 0.127 = 0.567 m camera height -> -0.50.
    # go2's -0.45 left the bottom ~0.12 m above the floor on this base.
    obstacle=ObstacleConfig(robot_z_bottom=-0.50),
)

AS2_CONFIG = RobotConfig(
    name='as2',
    # [CAD] 0.354 / 0.359 / 0.228 (standing envelope with the mount);
    # [ROS] 0.335 / 0.39 / 0.19; [SPEC] 720 x 378 x 457 mm. The half-width is between
    # [CAD]'s envelope and [SPEC]'s width.
    front_len=0.36, rear_len=0.39, half_width=0.20,
    # [CAD] left lens 0.316 ahead, 0.050 left, 0.178 above the origin.
    camera_fwd=0.316, camera_left=0.050,
    safety_radius=0.2,
    # Origin ~0.32 m up ([CAD] envelope bottom 0.321; [SPEC] 0.457 less [ROS]'s shell
    # top 0.135) + [CAD] 0.178 = 0.50 m camera height -> -0.43. No documented height.
    obstacle=ObstacleConfig(robot_z_bottom=-0.43),
)

AS2W_CONFIG = replace(
    AS2_CONFIG, name='as2w',
    # [SPEC] 721 x 493 x 521 mm; [ROS] as2w ~0.337 / 0.37-0.39.
    front_len=0.34, rear_len=0.39, half_width=0.25,
    # The camera mounts as on as2 [CAD].
    # Origin ~0.39 m up ([SPEC] 0.521 less [ROS]'s shell top 0.135) + [CAD] 0.178 =
    # 0.57 m camera height -> -0.50. No documented height.
    obstacle=ObstacleConfig(robot_z_bottom=-0.50),
)

B2_CONFIG = RobotConfig(
    name='b2',
    # [CAD] 0.432 / 0.644 / 0.250 (the rear calves reach back of the body); [ROS]
    # 0.43 / 0.59 / 0.224; [SPEC] 1098 x 450 x 645 mm. The CAD's Piper arm (one rig's
    # payload) is not included.
    front_len=0.44, rear_len=0.65, half_width=0.25,
    # [CAD] left lens ~0.40 ahead, 0.050 left, 0.158 above the origin. That model draws
    # the camera without its lens windows: the glass centre, the left lens assumed at
    # the insight9's 0.05; +-0.01.
    camera_fwd=0.40, camera_left=0.050,
    safety_radius=0.05,
    # The operator's call, 2026-10-13, as go2's: a 15-degree turn row reaches the chassis
    # as 0.1-0.26 rad/s, and b2 yawed 0.06 rad/s on those (122, round r11, 90s stuck).
    min_angular_vel=0.2,
    # As go2's, unmeasured on b2: the operator's call, 2026-10-16.
    max_angular_vel=1.0,
    # min_linear_vel stays the 0.1 default. 0.2 was run on 122 on 2026-09-04 and the
    # robot could barely move: cmd_vel_control DROPS a target below this rather than
    # raising it, so [0.1, 0.2) went from creeping to standing still -- every /cmd_vel
    # sample that drive was 0.000 or exactly 0.200. Raising it needs the planner's own
    # speed floor raised with it.
    # [DOC 9_136] StandUp 0.55 m + [CAD] 0.158 = 0.708 m camera height -> -0.64; the
    # -0.6 it had left it ~0.11 m above the floor. The top is every quadruped's
    # (ObstacleConfig, 0.08): the operator's 0.6 (2026-09-03) took the charging
    # station's roof for an obstacle on 122 and became 0.2 on 2026-11-13; the body's
    # highest point plus 0.05 is lower still.
    obstacle=ObstacleConfig(robot_z_bottom=-0.64),
)

B2W_CONFIG = replace(
    B2_CONFIG, name='b2w',
    # [SPEC] 1098 x 550 x 758 mm: the wheels stand out to 0.275 a side.
    front_len=0.44, rear_len=0.67, half_width=0.275,
    # The camera mounts as on b2 [CAD].
    # Its standing height is not documented. [ROS] b2w in [SDK2]
    # example/b2w/b2w_stand_example.cpp's stand pose (0, 0.67, -1.3) puts the wheel
    # centre 0.557 m below the origin; + [SPEC]'s 225 mm wheel's radius = 0.670 m, and
    # + [CAD] 0.158 = 0.828 m camera height -> -0.76. The same pose gives b2 0.589 m
    # against its documented 0.55, so the real stand may be a few cm lower -- which
    # only puts the bottom under the floor, where the span filter drops the floor.
    obstacle=ObstacleConfig(robot_z_bottom=-0.76),
)

G1_CONFIG = RobotConfig(
    name='g1',
    # Arms down: [SPEC] 1320 x 450 x 200 mm (0.225 a side at the shoulders); feet at
    # [ROS] x -0.055..+0.125, +-0.154. Arms reaching forward ([G1-URDF]'s glb pose)
    # reach 0.37 ahead, which this does not cover.
    front_len=0.15, rear_len=0.15, half_width=0.25,
    # [G1-URDF] insight7_link, chained to the pelvis at zero waist: 0.064 ahead, 0.035
    # left (the module is 88 mm wide and centred: the left lens), 0.460 up, pitched
    # 10 deg down.
    camera_fwd=0.064, camera_left=0.035,
    safety_radius=0.15,
    min_linear_vel=0.2, min_angular_vel=0.3,
    # Pelvis 0.78 m up ([RLG] g1_config.py base_height_target; [ROS] FK 0.784 at the
    # default stance) + 0.460 = 1.24 m camera height -> -1.18. The default -0.45 left
    # g1 blind below ~0.79 m. The top: the head's top is 0.092 m over the lens
    # ([G1-URDF] glb, 1.344 m vs 1.252 m) + 0.05 -> 0.14.
    obstacle=ObstacleConfig(robot_z_bottom=-1.18, robot_z_top=0.14),
)

#: This rig's measured camera offset, over the robot type's. `forward_m` / `left_m`
#: in metres, in the body frame above; `up_m` the camera's height above the floor when
#: `floor_at` says it was measured from the floor; any other key is the writer's own record.
CAMERA_OFFSET_PATH = '/tinynav/tinynav_db/calib/camera_offset.json'
#: How far above the floor the obstacle band starts once the camera's height is
#: measured: what go2's -0.45 left on 65, whose camera is 0.515 m up (2026-11-04).
FLOOR_CLEARANCE_M = 0.065


def with_measured_camera(config, path=CAMERA_OFFSET_PATH):
    """`config` with the camera offset from `path` and, when its height was measured
    from the floor, the obstacle band's bottom FLOOR_CLEARANCE_M above that floor;
    `config` itself when there is no file or it cannot be read. An unreadable one is said on stderr, because every
    process that plans imports this and none of them should fail to start over it."""
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return config
    try:
        doc = json.loads(text)
        out = replace(config, camera_fwd=float(doc['forward_m']),
                      camera_left=float(doc['left_m']))
        if 'floor_at' in doc:
            out = replace(out, obstacle=replace(
                out.obstacle, robot_z_bottom=-(float(doc['up_m']) - FLOOR_CLEARANCE_M)))
        return out
    except (KeyError, TypeError, ValueError) as e:
        print(f"robot_specs: ignoring {path}, not a camera offset ({e!r})", file=sys.stderr)
        return config


ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "go2").strip().lower()
try:
    _TYPE_CONFIG = globals()[f"{ROBOT_TYPE.upper()}_CONFIG"]
except KeyError:
    raise ValueError(f"Unsupported ROBOT_TYPE: {ROBOT_TYPE!r}") from None


GAITS = ('classic', 'free')
#: The gait unitree_control walks in between pilot's switches (docker-compose.d/drivers.yml).
CHASSIS_GAIT = os.environ.get("CHASSIS_GAIT", "classic").strip().lower()
if CHASSIS_GAIT not in GAITS:
    raise ValueError(f"Unsupported CHASSIS_GAIT: {CHASSIS_GAIT!r}, expected one of {GAITS}")

#: In the free gait, the slowest walk and turn in place that take a step at all: the
#: thresholds as measured, no margin (pilot's docs/gotchas.md).
FREE_GAIT_FLOORS = {'go2': {'min_linear_vel': 0.25, 'min_spin_vel': 0.45}}


def robot_config(path=CAMERA_OFFSET_PATH, gait=CHASSIS_GAIT):
    """This ROBOT_TYPE's config in `gait`, with the rig's measurement read from disk now."""
    config = _TYPE_CONFIG
    if gait == 'free' and config.name in FREE_GAIT_FLOORS:
        config = replace(config, **FREE_GAIT_FLOORS[config.name])
    return with_measured_camera(config, path)


#: Read once, at import: a process picks up a new measurement at its next launch.
ROBOT_CONFIG = robot_config()
