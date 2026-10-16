"""The chassis driver: one pure-SDK process -- no rclpy here.

Why no rclpy: this process opens the Unitree SDK's cyclonedds domain 0 on the
dog-facing NIC, and cyclonedds 0.10.x allows a domain id to be created exactly
once per process. A Cyclone rmw in the same process would be the second
creator and die (measured 2026-10-15, both orders). Everything the wider bus
needs from this process is published as raw DDS on ROS-compatible topics --
`rt/battery`, `rt/robot_status`, `rt/unitree/odometry` use std_msgs types,
so rclpy subscribers see them as `/battery`, `/robot_status`,
`/unitree/odometry` like the old rclpy publishers -- /battery is a JSON
string sample (soc/status/current) where the old one was a Float32 soc. The rt/cmd_vel input is the same equivalence read
in reverse.

QoS note: the SDK's ChannelPublisher takes no QoS -- writers are RELIABLE
VOLATILE. /battery used to be TRANSIENT_LOCAL; it is now published on every
lf sample (~20 Hz) instead, so a late joiner waits one sample, not a change.
"""
import argparse
import logging
import math
import os
import threading
import time
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_ as SdkString_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as SdkOdometry_
from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_ as SdkTime_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import (
    Point_, Pose_, PoseWithCovariance_, Quaternion_, Twist_ as SdkTwist_,
    TwistWithCovariance_, Vector3_)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_ as SdkHeader_
from enum import Enum

try:
    from .robot_profiles import require_profile
except ImportError:
    from robot_profiles import require_profile

ROBOT_TYPE = os.environ["ROBOT_TYPE"].strip().lower()
require_profile(ROBOT_TYPE)


class RobotStatus(Enum):
    STANDUP = "standup"
    SITTING = "sitting"


# A reply RPC slower than this is logged. ClassicWalk waits up to the client
# timeout (10s) for a reply, so it runs on GaitWorker's thread, not a reader's.
_SLOW_RPC_S = 0.3
# A pause in rt/cmd_vel longer than this is a gap: logged if it came mid-motion,
# and the next motion re-asserts the gait.
_CMD_GAP_S = 0.5
# Chassis watchdog: a command at least this large, held this long, while the
# chassis reports no motion, is logged as "not executing".
_STALL_CMD_V = 0.1
_STALL_CMD_W = 0.2
_STALL_CHASSIS_V = 0.03
_STALL_CHASSIS_W = 0.05
_STALL_AFTER_S = 2.0
_REPEAT_S = 5.0
_SPORT_STATE_SILENT_S = 1.0
_ODOM_MIN_PERIOD_S = 0.02          # cap the odometry republish at 50 Hz
_STATUS_PERIOD_S = 1.0
_WATCH_PERIOD_S = 0.5


def _quat_xyzw_from_rpy(roll: float, pitch: float, yaw: float):
    """ZYX (Unitree rpy) -> quaternion, geometry_msgs xyzw order."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
            cy * cp * cr + sy * sp * sr)


def _ros_time(t: float) -> SdkTime_:
    return SdkTime_(sec=int(t), nanosec=int((t - int(t)) * 1e9))


def _odometry_msg(stamp: float, x: float, y: float, z: float,
                  qx: float, qy: float, qz: float, qw: float,
                  vx: float, vy: float, vz: float, wz: float) -> SdkOdometry_:
    """nav_msgs/Odometry on the wire type; zero covariances (the old rclpy
    publisher sent the message default, which is also zeros)."""
    return SdkOdometry_(
        header=SdkHeader_(stamp=_ros_time(stamp), frame_id="odom"),
        child_frame_id="base_link",
        pose=PoseWithCovariance_(
            pose=Pose_(position=Point_(x=x, y=y, z=z),
                       orientation=Quaternion_(x=qx, y=qy, z=qz, w=qw)),
            covariance=[0.0] * 36),
        twist=TwistWithCovariance_(
            twist=SdkTwist_(linear=Vector3_(x=vx, y=vy, z=vz),
                            angular=Vector3_(x=0.0, y=0.0, z=wz)),
            covariance=[0.0] * 36))


class GaitWorker:
    """Asserts the walking gait off the rt/cmd_vel reader thread.

    ClassicWalk waits for a reply up to the client timeout (10s). Called inline it
    holds the reader thread, and every queued Move waits behind it -- the robot walks
    two steps and stops for ten seconds. Worse, the stall shows up as a cmd_vel gap,
    which re-arms the gait and chains another one: 2026-09-11 15:24 was five in a row.

    Requests coalesce. What matters is that the gait is asserted soon, not how many
    times, so a request while a call is in flight is absorbed into the next one.
    """

    def __init__(self, call, log, name='ClassicWalk'):
        self._call = call
        self._log = log
        self._name = name
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='gait-worker', daemon=True)

    def start(self):
        self._thread.start()

    def request(self):
        """Non-blocking: safe to call from a reader thread."""
        self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self):
        while True:
            self._wake.wait()
            if self._stop.is_set():
                return
            self._wake.clear()
            self.run_once()

    def run_once(self):
        """One gait assertion, timed and logged. Separated out so it can be driven
        without a thread."""
        t0 = time.monotonic()
        try:
            code = self._call()
        except Exception:
            self._log.exception(f'[sport] {self._name} raised')
            return
        took = time.monotonic() - t0
        if code != 0 or took > _SLOW_RPC_S:
            self._log.warning(f'[sport] {self._name} code={code} took {took:.2f}s')


class ChassisWatch:
    """Compares what was commanded with what rt/lf/sportmodestate says the chassis is
    doing. Logs only; never changes a command."""

    def __init__(self, log):
        self.log = log
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_at = None
        self.state = None
        self.v = (0.0, 0.0)
        self.yaw_speed = 0.0
        self.range_obstacle = ()
        self.state_at = None
        self._stall_since = None
        self._stall_logged_at = None
        self._silent_logged = False

    def on_cmd(self, now, vx, vy, wz):
        self.cmd = (vx, vy, wz)
        self.cmd_at = now

    def on_sport_state(self, now, mode, gait, error_code, v, yaw_speed, range_obstacle):
        key = (mode, gait, error_code)
        if key != self.state:
            was = '' if self.state is None else f' (was mode={self.state[0]} gait={self.state[1]} error_code={self.state[2]})'
            (self.log.warning if error_code else self.log.info)(
                f'[chassis] mode={mode} gait={gait} error_code={error_code}{was}')
            self.state = key
        self.v = (v[0], v[1])
        self.yaw_speed = yaw_speed
        self.range_obstacle = range_obstacle
        self.state_at = now

    def check(self, now):
        if self.state_at is not None:
            silent = now - self.state_at
            if silent > _SPORT_STATE_SILENT_S and not self._silent_logged:
                self.log.warning(f'[chassis] rt/lf/sportmodestate silent for {silent:.1f}s')
                self._silent_logged = True
            elif silent <= _SPORT_STATE_SILENT_S and self._silent_logged:
                self.log.info('[chassis] rt/lf/sportmodestate back')
                self._silent_logged = False

        vx, vy, wz = self.cmd
        commanded = (self.cmd_at is not None and now - self.cmd_at < _CMD_GAP_S
                     and (math.hypot(vx, vy) >= _STALL_CMD_V or abs(wz) >= _STALL_CMD_W))
        still = (self.state_at is not None
                 and math.hypot(*self.v) < _STALL_CHASSIS_V
                 and abs(self.yaw_speed) < _STALL_CHASSIS_W)
        if commanded and still:
            if self._stall_since is None:
                self._stall_since = now
            held = now - self._stall_since
            if held >= _STALL_AFTER_S and (self._stall_logged_at is None
                                           or now - self._stall_logged_at >= _REPEAT_S):
                self._stall_logged_at = now
                mode, gait, err = self.state or (None, None, None)
                self.log.warning(
                    f'[chassis] not executing for {held:.1f}s: commanded vx={vx:.2f} vy={vy:.2f} '
                    f'wz={wz:.2f}, chassis v={math.hypot(*self.v):.3f} yaw_speed={self.yaw_speed:.3f} '
                    f'mode={mode} gait={gait} error_code={err} '
                    f'range_obstacle={[round(float(r), 2) for r in self.range_obstacle]}')
        else:
            if self._stall_logged_at is not None:
                self.log.info(f'[chassis] executing again after {now - self._stall_since:.1f}s')
            self._stall_since = None
            self._stall_logged_at = None


class Ros2UnitreeManagerNode:
    """Historical name: it stopped being an rclpy node when the process went
    pure-SDK (the class the tests build via __new__). Everything the bus needs
    leaves through raw-DDS publishers on ROS-compatible topics."""

    def __init__(self, networkInterface: str = "enP8p1s0", robot_model: str = ROBOT_TYPE,
                 log: logging.Logger | None = None):
        self.profile = require_profile(robot_model)
        self.robot_model = self.profile.type.value
        self.logger = log or logging.getLogger('unitree_control')

        self.channel = ChannelFactoryInitialize(0, networkInterface)
        self.sport_client = self.profile.client_factory()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        if self.profile.gait_assertion == 'classic_walk':
            self.sport_client.ClassicWalk(True)
        self._robot_status = RobotStatus.SITTING
        self.battery = None
        self.last_twist_time = None
        # The last command was non-zero (so the next non-zero is not a start).
        self._walking = False
        # The next Move re-asserts the gait first: set at every motion start.
        self._gait_due = False
        self._move_failures = 0
        self._move_failure_logged_at = None
        self.watch = ChassisWatch(self.logger)
        self.gait = None
        if self.profile.gait_assertion == 'classic_walk' and self.profile.needs_gait_worker:
            self.gait = GaitWorker(lambda: self.sport_client.ClassicWalk(True), self.logger)
            self.gait.start()

        self.twist_subscriber = ChannelSubscriber("rt/cmd_vel", Twist_)
        self.twist_subscriber.Init(self.TwistMessageHandler, 10)

        self.action_subscriber = ChannelSubscriber("rt/service/command", SdkString_)
        self.action_subscriber.Init(self.ActionMessageHandler, 10)

        # Bus publishers, raw DDS on ROS-compatible names (class docstring).
        self.publisher_robot_status = ChannelPublisher('rt/robot_status', SdkString_)
        self.publisher_robot_status.Init()

        self._init_battery_subscription()
        self._init_chassis_state(networkInterface)

        self._threads: list[threading.Thread] = []
        for target, name in ((self._publish_robot_status, 'robot-status'),
                             (self._watch_loop, 'chassis-watch')):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def _init_battery_subscription(self) -> None:
        battery = self.profile.battery
        if battery is None:
            return
        if battery.idl == 'unitree_go':
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as BattMsg
        else:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_ as BattMsg
        sub = ChannelSubscriber(battery.topic, BattMsg)
        sub.Init(self.BatteryMessageHandler, 10)
        self._battery_has_status = battery.has_status
        # The BMS bridge the bus reads (see BatteryMessageHandler).
        self.publisher_battery = ChannelPublisher('rt/battery', SdkString_)
        self.publisher_battery.Init()

    def _init_chassis_state(self, networkInterface: str) -> None:
        """Sport state (watch + AS2 odometry) exists for the sport API family;
        the go family additionally gets leg odometry off the wire."""
        if self.profile.motion_api != 'sport':
            return
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        self.sport_state_subscriber = ChannelSubscriber("rt/lf/sportmodestate",
                                                        SportModeState_)
        self.sport_state_subscriber.Init(self.SportStateMessageHandler, 10)

        self.publisher_chassis_odom = ChannelPublisher('rt/unitree/odometry',
                                                       SdkOdometry_)
        self.publisher_chassis_odom.Init()
        self._last_chassis_odom_time = 0.0

        if self.profile.odom == 'robot_odom':
            # Chassis odometry, republished onto the ROS bus. rt/utlidar/robot_odom
            # is the leg odometry wrapped as nav_msgs/Odometry, not lidar odometry:
            # it matches rt/sportmodestate.position exactly and keeps publishing with
            # the lidar removed.
            self.chassis_odom_subscriber = ChannelSubscriber("rt/utlidar/robot_odom",
                                                             SdkOdometry_)
            self.chassis_odom_subscriber.Init(self.ChassisOdomMessageHandler, 10)

    # -- twist message handler
    def TwistMessageHandler(self, msg: Twist_):
        # Runs on the SDK reader thread: an exception escaping here kills the
        # subscription (see ActionMessageHandler).
        try:
            self._on_twist(time.monotonic(), float(msg.linear.x), float(msg.linear.y),
                           float(msg.angular.z))
        except Exception:
            self.logger.exception("cmd_vel handling failed")
        time.sleep(0.02)

    def _on_twist(self, now, vx, vy, wz):
        gap = None if self.last_twist_time is None else now - self.last_twist_time
        self.last_twist_time = now
        if gap is not None and gap > _CMD_GAP_S:
            if self._walking:
                self.logger.warning(f'[cmd_vel] rt/cmd_vel silent for {gap:.2f}s mid-motion')
            self._gait_due = True

        if vx != 0 or vy != 0 or wz != 0:
            if not self._walking:
                self._gait_due = True
            # Handed off, never called here: it is a reply RPC, and this thread must
            # stay free to keep pushing Move at the chassis.
            if self._gait_due and self.gait is not None:
                self.gait.request()
            self._gait_due = False
            self._walking = True
        else:
            # A zero Move, not StopMove: StopMove is a reply RPC on this thread.
            self._walking = False
        code = self.sport_client.Move(vx, vy, wz)
        if code != 0:
            self._move_failed(now, code)
        self.watch.on_cmd(now, vx, vy, wz)

    def _move_failed(self, now, code):
        self._move_failures += 1
        if self._move_failure_logged_at is None or now - self._move_failure_logged_at >= _REPEAT_S:
            self.logger.warning(f'[sport] Move send failed code={code} '
                                f'({self._move_failures} failures so far)')
            self._move_failure_logged_at = now

    def SportStateMessageHandler(self, msg):
        # 20Hz on the SDK reader thread: keep it to field copies.
        try:
            if (int(msg.stamp.sec) == 0 and int(msg.stamp.nanosec) == 0
                    and int(msg.mode) == 0 and int(msg.gait_type) == 0
                    and not any(msg.position) and not any(msg.velocity)):
                return
            now = time.monotonic()
            self.watch.on_sport_state(now, int(msg.mode), int(msg.gait_type),
                                      int(msg.error_code), msg.velocity, float(msg.yaw_speed),
                                      msg.range_obstacle)
            if self.profile.odom == 'sportmodestate':
                self._republish_sportmodestate_odom(msg)
        except Exception as e:
            self.logger.error(f"Error in SportStateMessageHandler: {e}")

    def ActionMessageHandler(self, msg: SdkString_):
        self.logger.info(f"ActionMessageHandler received: {msg.data!r}")
        # unitree_sdk2py's reader thread calls this with no except around it, so an
        # exception escaping here kills that thread and the subscription goes deaf
        # for the rest of the run -- every later sit/stand silently dropped, while
        # the process still looks healthy. One bad action must not cost the channel.
        try:
            self._play_action(msg)
        except Exception:
            self.logger.exception("action failed")

    def _play_action(self, msg: SdkString_):
        if msg.data.split(" ")[0] != "play":
            return
        action_key = msg.data.split(" ")[1]
        if action_key == "sit":
            steps = [(name, self._resolve_step(name)) for name in self.profile.sit_steps]
            self._play_steps('Sitting', steps, RobotStatus.SITTING)
        elif action_key == "stand":
            steps = [(name, self._resolve_step(name)) for name in self.profile.stand_steps]
            self._play_steps('Standing', steps, RobotStatus.STANDUP)

    def _resolve_step(self, name: str):
        """Profile step name -> bound call. Names the sport/loco clients spell
        the same resolve by getattr; the three shaped calls are the exceptions
        the old per-model branches used to carry."""
        if name == 'ClassicWalk':
            return lambda: self.sport_client.ClassicWalk(True)
        if name == 'SwitchGait':
            return lambda: self.sport_client.SwitchGait(1)
        if name == 'Squat2StandUp':
            return self._squat_to_stand
        return getattr(self.sport_client, name)

    def _squat_to_stand(self):
        """The biped's FSM needs a moment after Damp before the stand takes."""
        time.sleep(0.5)
        return self.sport_client.Squat2StandUp()

    def _play_steps(self, what, steps, status):
        """Run the SDK calls in order and claim `status` only if every one returned 0.
        /robot_status is a statement about the chassis, and a false one is worse than
        none: a refusing sport service used to be reported as a successful stand."""
        codes = {name: call() for name, call in steps}
        said = ', '.join(f'{k} code={v}' for k, v in codes.items())
        if all(c == 0 for c in codes.values()):
            self.logger.info(f"{what}: {said}")
            self._robot_status = status
        else:
            self.logger.error(
                f"{what} REFUSED by the robot: {said}. The commands reached the sport "
                "service and it declined them.")

    def _publish_robot_status(self):
        """1 Hz thread: the status tree reads this to know sit from stand."""
        while True:
            self.publisher_robot_status.Write(SdkString_(data=self._robot_status.value))
            time.sleep(_STATUS_PERIOD_S)

    def _watch_loop(self):
        while True:
            self.watch.check(time.monotonic())
            time.sleep(_WATCH_PERIOD_S)

    def ChassisOdomMessageHandler(self, msg):
        # Already a nav_msgs Odometry on the wire; cap it at 50Hz.
        now = time.time()
        if now - self._last_chassis_odom_time < _ODOM_MIN_PERIOD_S:
            return
        self._last_chassis_odom_time = now
        try:
            # Restamped on the wall clock: the chassis stamp is its own timebase.
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            v, w = msg.twist.twist.linear, msg.twist.twist.angular
            self.publisher_chassis_odom.Write(_odometry_msg(
                now, float(p.x), float(p.y), float(p.z),
                float(q.x), float(q.y), float(q.z), float(q.w),
                float(v.x), float(v.y), float(v.z), float(w.z)))
        except Exception as e:
            self.logger.error(f"Error in ChassisOdomMessageHandler: {e}")

    def _republish_sportmodestate_odom(self, msg) -> None:
        """B2 and AS2 odometry source: same state the watch reads, on the wire type."""
        now = time.time()
        if now - self._last_chassis_odom_time < _ODOM_MIN_PERIOD_S:
            return
        self._last_chassis_odom_time = now
        try:
            qx, qy, qz, qw = _quat_xyzw_from_rpy(
                float(msg.imu_state.rpy[0]), float(msg.imu_state.rpy[1]),
                float(msg.imu_state.rpy[2]))
            self.publisher_chassis_odom.Write(_odometry_msg(
                now, float(msg.position[0]), float(msg.position[1]), float(msg.position[2]),
                qx, qy, qz, qw,
                float(msg.velocity[0]), float(msg.velocity[1]), float(msg.velocity[2]),
                float(msg.yaw_speed)))
        except Exception as e:
            self.logger.error(f"Error republishing sportmodestate odometry: {e}")

    def BatteryMessageHandler(self, msg):
        try:
            # One JSON sample carries everything the bus needs from the BMS: soc
            # for the status display, status+current for the charge judgement
            # (core_runtime reads /battery instead of opening its own SDK on
            # the dog NIC -- one cyclonedds domain-0 creator per process).
            if self._battery_has_status:
                status = int(msg.bms_state.status)
                ma = float(msg.bms_state.current)
                soc = float(msg.bms_state.soc)
            else:
                status, ma, soc = None, float(msg.current), float(msg.soc)
            self.publisher_battery.Write(SdkString_(
                data=f'{{"soc": {soc}, "status": {status if status is not None else "null"}, "ma": {ma}}}'))
            self.battery = soc
        except Exception as e:
            self.logger.error(f"Error in BatteryMessageHandler: {e}")
            import traceback
            traceback.print_exc()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-interface", default="enP8p1s0",
                        help="Network interface connected to the robot")
    parsed_args, _ = parser.parse_known_args(args=args)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    node = Ros2UnitreeManagerNode(parsed_args.network_interface)
    # Every thread is daemonized; blocking here is the process's whole job.
    threading.Event().wait()


if __name__ == "__main__":
    main()
