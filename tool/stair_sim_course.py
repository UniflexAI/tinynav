"""Standalone geometry and front-camera calibration for stair_mode_sim.

Extracted from xiaole/rl-stairs at 598e459 without its navigation classes.
World coordinates are used only by the physics, renderer, and assertions.
"""
import math
from xml.etree import ElementTree

import mujoco
import numpy as np

RISE, RUN, STEPS = 0.16, 0.30, 8
LOWER = -(STEPS + 1) * RISE
LIP, FOOT = 0.0, RUN * STEPS
SPAWN = (-3.45, -0.06, 0.32)
LANDING_DEPTH = 1.75
SHAFT = (LIP, FOOT, 0.56, 0.70)
LANE_END = 1.82
MULTI_SHAFT = (LIP, FOOT, 0.90, 1.04)
MULTI_LANE_END = 2.84
DOOR_LOW, DOOR_HIGH = 0.80, 1.75
MAPPING_VIEWS = tuple((yaw, 44.0, 92.0) for yaw in (0, 90, 180, 270)) + ((0, 90.0, 104.0),)


def door_pose(swing):
    yaw = math.radians(-90.0 - swing)
    length = DOOR_HIGH - DOOR_LOW
    hinge = FOOT + LANDING_DEPTH + .07
    return hinge + length / 2 * math.cos(yaw), DOOR_HIGH + length / 2 * math.sin(yaw), yaw

def add_landing_door(box, wall_x, lower, upper, z, pose, wall_color, flight):
    x = wall_x[0]
    prefix = f"landing_{flight}_"
    low, high = DOOR_LOW, DOOR_HIGH
    brown = (0.59, 0.31, 0.15, 1)
    box(prefix + "door_frame_low_wall", *wall_x, lower, low, z - .20, z + .68, wall_color, True)
    box(prefix + "door_frame_high_wall", *wall_x, high, upper, z - .20, z + .68, wall_color, True)
    box(prefix + "side_room", x, x + 1.55, low, high, z - .20, z, (0.31, .36, .49, 1))
    box(prefix + "side_room_back_wall", x + 1.55, x + 1.67, low, high, z, z + .8, wall_color, True)
    for i, y in enumerate((low - .10, high)):
        box(prefix + f"side_room_{i}_wall", x + .12, x + 1.55, y, y + .10, z, z + .8, wall_color, True)
    cx, cy, yaw = door_pose({"room": -90, "landing": 165, "half": 85, "moving": 0}[pose])
    length = high - low
    box(prefix + "door_leaf", cx - length / 2, cx + length / 2, cy - .033, cy + .033,
        z + .015, z + 1.05, brown, True, yaw=yaw, moving=pose == "moving")

def add_floor_stack(box, flights, direction, doors, lane_width=1.8, landing_depth=LANDING_DEPTH):
    """Build two flights per floor around the same open stairwell."""
    rise = RISE * direction
    landing_height = (STEPS + 1) * rise
    shaft = shaft_bounds(flights, lane_width)
    lower_side, upper_side = -lane_width / 2, shaft[3] + lane_width
    near_lane = (lower_side, shaft[2])
    far_lane = (shaft[3], upper_side)
    wall_color = (0.57, 0.60, 0.64, 1)
    floor_color = (0.30, 0.51, 0.57, 1)
    box("upper", -5.1, LIP, *near_lane, -.20, 0, floor_color)
    for side, y0 in enumerate((lower_side - .12, near_lane[1])):
        box(f"entry_{side}_wall", -5.1, LIP, y0, y0 + .12, -.20, .68, wall_color, True)
    for flight in range(flights):
        z = flight * landing_height
        reverse = flight % 2 == 1
        lane = far_lane if reverse else near_lane
        outer = (upper_side, upper_side + .12) if reverse else (lower_side - .12, lower_side)
        for tread in range(STEPS):
            x0 = (STEPS - tread - 1) * RUN if reverse else tread * RUN
            x1 = x0 + RUN
            top = z + (tread + 1) * rise
            color = (0.40, 0.56, 0.66, 1) if tread % 2 else (0.35, 0.50, 0.60, 1)
            box(f"flight_{flight}_tread_{tread}", x0, x1, *lane, top - .22, top, color)
            box(f"flight_{flight}_outer_{tread}_wall", x0, x1, *outer,
                top - .19, top + .68, wall_color, True)
            rail_y = (shaft[3] - .035, shaft[3]) if reverse else (shaft[2], shaft[2] + .035)
            box(f"flight_{flight}_rail_{tread}", x0, x1, *rail_y, top + .30, top + .37,
                (0.93, 0.65, 0.22, 1), True)
        z += landing_height
        a, b = (-landing_depth, LIP) if reverse else (FOOT, FOOT + landing_depth)
        box(f"landing_{flight}", a, b, lower_side, upper_side, z - .20, z, floor_color)
        wall_x = (a - .12, a) if reverse else (b, b + .12)
        if flight in doors:
            add_landing_door(box, wall_x, lower_side, upper_side, z, doors[flight], wall_color, flight)
        else:
            box(f"landing_{flight}_end_wall", *wall_x, lower_side, upper_side,
                z - .20, z + .65, wall_color, True)
        for side, y0 in enumerate((lower_side - .12, upper_side)):
            box(f"landing_{flight}_{side}_wall", a, b, y0, y0 + .12,
                z - .20, z + .68, wall_color, True)

def shaft_bounds(flights, lane_width=None, landing_depth=LANDING_DEPTH):
    if lane_width is None and landing_depth != LANDING_DEPTH:
        lane_width = 1.8
    if lane_width is None:
        return MULTI_SHAFT if flights > 2 else SHAFT
    if not 0.8 <= lane_width <= 3.0:
        raise ValueError("Flight width must be in [0.8, 3.0] meters")
    return (LIP, FOOT, lane_width / 2, lane_width / 2 + 0.14)


def build_world(plant, robot, flights, *, direction=-1, mapping_rig=True, doors=None,
                lane_width=None, landing_depth=LANDING_DEPTH):
    spec = mujoco.MjSpec.from_string(
        '<mujoco model="stairs"><visual><headlight diffuse=".84 .84 .84" ambient=".38 .38 .38"/>'
        '</visual><worldbody/></mujoco>'
    )
    surfaces = []
    walls = []

    def box(name, x0, x1, y0, y1, bottom, top, rgba, wall=False, *, yaw=0.0, moving=False):
        body = spec.worldbody
        pos = ((x0 + x1) / 2, (y0 + y1) / 2, (bottom + top) / 2)
        quat = (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))
        if moving:
            body = body.add_body(name=name + "_body", mocap=True, pos=pos, quat=quat)
            pos, quat = (0., 0., 0.), (1., 0., 0., 0.)
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=pos, quat=quat,
                      size=((x1 - x0) / 2, (y1 - y0) / 2, (top - bottom) / 2),
                      friction=plant.TERRAIN_FRICTION, rgba=rgba,
                      group=1 if wall and "rail" not in name and not name.endswith("door_leaf") else 0)
        (walls if wall else surfaces).append((x0, x1, y0, y1, top))

    if not 1.0 <= landing_depth <= 3.0:
        raise ValueError("Landing depth must be in [1.0, 3.0] meters")
    shaft_bounds(flights, lane_width, landing_depth)
    if flights > 2 or lane_width is not None or landing_depth != LANDING_DEPTH:
        add_floor_stack(box, flights, direction, doors or {},
                        1.8 if lane_width is None else lane_width, landing_depth)
    else:
        stack = flights == 2
        step = RISE * direction
        landing_height = (STEPS + 1) * step
        bottom = min(0.0, flights * landing_height) - 0.25
        ymin, ymax = (-0.65, SHAFT[2]) if stack else (-0.90, 0.90)
        box("upper", -5.1, LIP, ymin, ymax, bottom, 0, (0.33, 0.47, 0.58, 1))
        for i in range(STEPS):
            color = (0.45, 0.60, 0.68, 1) if i % 2 else (0.37, 0.51, 0.61, 1)
            box(f"tread_{i+1}", i * RUN, (i + 1) * RUN, ymin, ymax,
                bottom, (i + 1) * step, color)
            if stack:
                box(f"next_tread_{i+1}", FOOT - (i + 1) * RUN, FOOT - i * RUN,
                    SHAFT[3], LANE_END, bottom, LOWER - (i + 1) * RISE, color)
        if stack:
            end = FOOT + LANDING_DEPTH
            box("turn_landing", FOOT, end, ymin, LANE_END, bottom, LOWER, (0.30, 0.51, 0.57, 1))
            box("lower", -3.3, LIP, SHAFT[3], LANE_END, bottom, 2 * LOWER, (0.25, 0.57, 0.50, 1))
            box("left_wall", -5.1, LIP, ymax, SHAFT[3], bottom, 0.65, (0.57, 0.60, 0.64, 1), True)
            box("right_wall", -5.1, end, ymin - 0.12, ymin, bottom, 0.55, (0.57, 0.60, 0.64, 0.25), True)
            box("outer_wall", -3.3, end, LANE_END, LANE_END + 0.12, bottom, LOWER + 0.50,
                (0.57, 0.60, 0.64, 1), True)
            box("end_wall", end, end + 0.12, ymin, LANE_END, bottom, LOWER + 0.65,
                (0.57, 0.60, 0.64, 1), True)
            box("shaft_rail", LIP, FOOT, 0.60, 0.66, LOWER + 0.35, LOWER + 0.42,
                (0.95, 0.65, 0.22, 1), True)
        else:
            box("lower", FOOT, 5.2, ymin, ymax, bottom, landing_height, (0.25, 0.57, 0.50, 1))
            box("left_wall", -5.1, 5.2, ymax, ymax + 0.12, bottom, max(0.0, landing_height) + 0.8, (0.57, 0.60, 0.64, 1), True)
            box("right_wall", -5.1, 5.2, ymin - 0.12, ymin, bottom, max(0.0, landing_height) + 0.8, (0.57, 0.60, 0.64, 0.25), True)

    # The optional asset bundle supplies visual meshes only. Keep all joints,
    # inertials and collision geoms from the registered Go2 training MJCF.
    if robot.mesh_dir() is None:
        root = ElementTree.parse(robot.mjcf).getroot()
        for parent in root.iter():
            for elem in list(parent):
                if elem.tag == "mesh" or (elem.tag == "geom" and "mesh" in elem.attrib):
                    parent.remove(elem)
        robot_spec = mujoco.MjSpec.from_string(ElementTree.tostring(root, encoding="unicode"))
        for geom in robot_spec.geoms:
            if (geom.name or "").endswith("_collision"):
                geom.rgba[:] = (0.95, 0.55, 0.16, 1) if geom.name.startswith("base") else (0.78, 0.80, 0.84, 1)
    else:
        robot_spec = plant._load_robot_spec(robot)
    base = next(body for body in robot_spec.worldbody.bodies if body.name == "base_link")
    spec.worldbody.add_frame().attach_body(base, "")
    base = next(body for body in spec.worldbody.bodies if body.name == "base_link")
    plant._add_depth_camera_and_shell(base, robot)
    if mapping_rig:
        add_mapping_cameras(base)
    plant._apply_plant_options(spec)
    model, bindings = plant._compile(spec, with_shell=True, robot=robot, scene_name="flat")
    return model, bindings, surfaces, walls

def add_mapping_cameras(base):
    for index, (yaw, pitch, fov) in enumerate(MAPPING_VIEWS):
        heading, tilt = np.radians((yaw, pitch))
        c, s = np.cos(heading), np.sin(heading)
        forward = np.array((np.cos(tilt) * c, np.cos(tilt) * s, -np.sin(tilt)))
        right = np.array((s, -c, 0.0))
        up = np.cross(-forward, right)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.column_stack((right, up, -forward)).ravel())
        base.add_camera(name=f"stairs_nav_{index}", pos=(0, 0, 1.55), quat=quat, fovy=fov)

def shaft_clearance(x, y, yaw, shaft=SHAFT):
    forward, left = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.25, 0.25, 11))
    c, s = math.cos(yaw), math.sin(yaw)
    fx, fy = x + c * forward - s * left, y + s * forward + c * left
    x0, x1, y0, y1 = shaft
    dx = np.maximum(np.maximum(x0 - fx, fx - x1), 0)
    dy = np.maximum(np.maximum(y0 - fy, fy - y1), 0)
    return float(np.min(np.hypot(dx, dy)))

def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

class FrontCameraBEV:
    """Apply only the measured ROS-left camera profile; BEV lives in stair_node."""

    def __init__(self, runtime, *, profile="ros-left"):
        if profile != "ros-left" or runtime.depth_spec.pie:
            raise ValueError("Expected the Go2 optical-Z policy and ROS-left calibration")
        self.camera_id = runtime.b.depth_camera_id
        self._apply_ros_left(runtime)
        mujoco.mj_camlight(runtime.model, runtime.data)

    def _apply_ros_left(self, runtime):
        model, data, spec = runtime.model, runtime.data, runtime.depth_spec
        camera = self.camera_id
        sx, sy = spec.width / 544, spec.height / 640
        fx, fy = 303.10 * sx, 303.10 * sy
        cx, cy = (269.37 + 0.5) * sx, (314.67 + 0.5) * sy
        model.cam_resolution[camera] = (spec.width, spec.height)
        # Use one length unit per output pixel. MuJoCo stores off-axis
        # offsets relative to the image center with the frustum's signs.
        model.cam_sensorsize[camera] = (spec.width, spec.height)
        model.cam_intrinsic[camera] = (fx, fy, spec.width / 2 - cx, spec.height / 2 - cy)

        # ROS measures height above the ground in stance, not relative to
        # base_link. Match it at the settled spawn on the upper landing.
        front, left, height = 0.2709, -0.0211, 0.4669
        body = model.body("base_link").id
        vertical = data.xmat[body].reshape(3, 3)[2]
        base_z = data.xpos[body, 2]
        dz = (height - base_z - vertical[0] * front - vertical[1] * left) / vertical[2]
        model.cam_pos[camera] = (front, left, dz)
        # The visual shell belongs to the nominal mount and can clip the
        # relocated lens. Hide that synthetic, non-colliding prop from depth.
        model.geom_group[model.geom("front_depth_camera_visual").id] = 4
        print(f"ROS left camera: rendered {spec.width}x{spec.height}, "
              f"fx={fx:.3f} fy={fy:.3f} cx={cx - 0.5:.3f} cy={cy - 0.5:.3f}, "
              f"base_link mount={model.cam_pos[camera].round(4)}", flush=True)
        return fx, fy, cx, cy
