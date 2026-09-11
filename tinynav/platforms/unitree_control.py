import argparse
import math
import os
import rclpy
import threading
from rclpy.node import Node
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from std_msgs.msg import Float32, String
from nav_msgs.msg import Odometry
from enum import Enum
import time

# go2/b2 are quadrupeds sharing the same SportClient gait API (Move/StandUp/
# StandDown/BalanceStand/ClassicWalk). go2w/b2w are the wheeled variants of the
# same chassis — the vendored SDK has no separate go2w/b2w package, so they reuse
# the go2/b2 SportClient (same gait/lowstate API) as-is. g1 is a humanoid
# controlled through the FSM-based LocoClient instead, so it needs its own
# client, lowstate IDL, and stand/sit mapping.
_QUADRUPED_ROBOT_MODELS = ('go2', 'go2w', 'b2', 'b2w')
_SUPPORTED_ROBOT_MODELS = _QUADRUPED_ROBOT_MODELS + ('g1',)
# SportClient.SwitchGait exists only on b2; go2's client has no such method, so
# calling it there raises AttributeError -- inside a DDS reader callback, which is
# fatal for the whole subscription (see ActionMessageHandler).
_SWITCH_GAIT_ROBOT_MODELS = ('b2', 'b2w')
ROBOT_TYPE = os.environ["ROBOT_TYPE"].strip().lower()
if ROBOT_TYPE not in _SUPPORTED_ROBOT_MODELS:
    raise ValueError(f"Unsupported ROBOT_TYPE: {ROBOT_TYPE!r}, expected one of {_SUPPORTED_ROBOT_MODELS}")


def _build_sport_client(robot_model: str):
    if robot_model in ('go2', 'go2w'):
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        return SportClient()
    if robot_model in ('b2', 'b2w'):
        from unitree_sdk2py.b2.sport.sport_client import SportClient
        return SportClient()
    if robot_model == 'g1':
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        return LocoClient()
    raise ValueError(f"Unsupported robot model: {robot_model}")


def _lowstate_type_and_topic(robot_model: str):
    if robot_model in _QUADRUPED_ROBOT_MODELS:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        return LowState_, "rt/lowstate"
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    return LowState_, "rt/lowstate"

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
    """Compares what was commanded with what rt/sportmodestate says the chassis is
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
                self.log.warning(f'[chassis] rt/sportmodestate silent for {silent:.1f}s')
                self._silent_logged = True
            elif silent <= _SPORT_STATE_SILENT_S and self._silent_logged:
                self.log.info('[chassis] rt/sportmodestate back')
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


class Ros2UnitreeManagerNode(Node):
    def __init__(self, networkInterface: str = "enP8p1s0", robot_model: str = ROBOT_TYPE):
        super().__init__('ros2_unitree_manager')
        if robot_model not in _SUPPORTED_ROBOT_MODELS:
            raise ValueError(f"Unsupported robot model: {robot_model!r}, expected one of {_SUPPORTED_ROBOT_MODELS}")
        self.robot_model = robot_model
        self.is_quadruped = robot_model in _QUADRUPED_ROBOT_MODELS
        self.has_switch_gait = robot_model in _SWITCH_GAIT_ROBOT_MODELS

        self.channel = ChannelFactoryInitialize(0, networkInterface)
        self.sport_client = _build_sport_client(robot_model)
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        if self.is_quadruped:
            self.sport_client.ClassicWalk(True)
        self._robot_status = RobotStatus.SITTING
        self.battery = 0.0
        self.last_twist_time = None
        self.logger = self.get_logger()
        # The last command was non-zero (so the next non-zero is not a start).
        self._walking = False
        # The next Move re-asserts ClassicWalk first: set at every motion start.
        self._gait_due = False
        self._move_failures = 0
        self._move_failure_logged_at = None
        self.watch = ChassisWatch(self.logger)
        self.gait = None
        if self.is_quadruped:
            self.gait = GaitWorker(lambda: self.sport_client.ClassicWalk(True), self.logger)
            self.gait.start()

        self.twist_subscriber = ChannelSubscriber("rt/cmd_vel", Twist_)
        self.twist_subscriber.Init(self.TwistMessageHandler, 10)

        self.action_subscriber = ChannelSubscriber("rt/service/command", String_)
        self.action_subscriber.Init(self.ActionMessageHandler, 10)

        lowstate_type, lowstate_topic = _lowstate_type_and_topic(robot_model)
        lowstate_subscriber = ChannelSubscriber(lowstate_topic, lowstate_type)
        lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.publisher_battery = self.create_publisher(Float32, '/battery', 10)
        self.publisher_robot_status = self.create_publisher(String, '/robot_status', 10)

        # Chassis odometry, republished onto the ROS bus. rt/utlidar/robot_odom
        # is the leg odometry wrapped as nav_msgs/Odometry, not lidar odometry:
        # it matches rt/sportmodestate.position exactly and keeps publishing with
        # the lidar removed.
        if self.is_quadruped:
            from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
            self.publisher_chassis_odom = self.create_publisher(Odometry, '/unitree/odometry', 10)
            self._last_chassis_odom_time = 0.0
            self.chassis_odom_subscriber = ChannelSubscriber("rt/utlidar/robot_odom", Odometry_)
            self.chassis_odom_subscriber.Init(self.ChassisOdomMessageHandler, 10)

            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            self.sport_state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            self.sport_state_subscriber.Init(self.SportStateMessageHandler, 10)
            self._watch_timer = self.create_timer(0.5, lambda: self.watch.check(time.monotonic()))

        self._status_timer = self.create_timer(1.0, self._publish_robot_status)

    # twist message handler
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
        # 500Hz on the SDK reader thread: keep it to field copies.
        try:
            self.watch.on_sport_state(time.monotonic(), int(msg.mode), int(msg.gait_type),
                                      int(msg.error_code), msg.velocity, float(msg.yaw_speed),
                                      msg.range_obstacle)
        except Exception as e:
            self.logger.error(f"Error in SportStateMessageHandler: {e}")

    def ActionMessageHandler(self, msg: String_):
        self.logger.info(f"ActionMessageHandler received: {msg.data!r}")
        # unitree_sdk2py's reader thread calls this with no except around it, so an
        # exception escaping here kills that thread and the subscription goes deaf
        # for the rest of the run -- every later sit/stand silently dropped, while
        # the process still looks healthy. One bad action must not cost the channel.
        try:
            self._play_action(msg)
        except Exception:
            self.logger.exception("action failed")

    def _play_action(self, msg: String_):
        if msg.data.split(" ")[0] != "play":
            return
        action_key = msg.data.split(" ")[1]
        if action_key == "sit":
            if self.is_quadruped:
                steps = [('StandDown', self.sport_client.StandDown)]
            else:
                steps = [('StandUp2Squat', self.sport_client.StandUp2Squat)]
            self._play_steps('Sitting', steps, RobotStatus.SITTING)
        elif action_key == "stand":
            if self.is_quadruped:
                steps = [('StandUp', self.sport_client.StandUp),
                         ('BalanceStand', self.sport_client.BalanceStand),
                         ('ClassicWalk', lambda: self.sport_client.ClassicWalk(True))]
                if self.has_switch_gait:
                    steps.append(('SwitchGait', lambda: self.sport_client.SwitchGait(1)))
            else:
                steps = [('Damp', self.sport_client.Damp),
                         ('Squat2StandUp', self._squat_to_stand)]
            self._play_steps('Standing', steps, RobotStatus.STANDUP)

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
        msg = String()
        msg.data = self._robot_status.value
        self.publisher_robot_status.publish(msg)

    def ChassisOdomMessageHandler(self, msg):
        # Already a nav_msgs Odometry on the wire; cap it at 50Hz.
        now = time.time()
        if now - self._last_chassis_odom_time < 0.02:
            return
        self._last_chassis_odom_time = now
        try:
            odom = Odometry()
            # Restamped on the ROS clock: the chassis stamp is its own timebase.
            odom.header.stamp = self.get_clock().now().to_msg()
            # The chassis's own odom origin; unrelated to tinynav's "world".
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            odom.pose.pose.position.x = float(p.x)
            odom.pose.pose.position.y = float(p.y)
            odom.pose.pose.position.z = float(p.z)
            odom.pose.pose.orientation.x = float(q.x)
            odom.pose.pose.orientation.y = float(q.y)
            odom.pose.pose.orientation.z = float(q.z)
            odom.pose.pose.orientation.w = float(q.w)
            v, w = msg.twist.twist.linear, msg.twist.twist.angular
            odom.twist.twist.linear.x = float(v.x)
            odom.twist.twist.linear.y = float(v.y)
            odom.twist.twist.linear.z = float(v.z)
            odom.twist.twist.angular.z = float(w.z)
            self.publisher_chassis_odom.publish(odom)
        except Exception as e:
            self.logger.error(f"Error in ChassisOdomMessageHandler: {e}")

    def LowStateMessageHandler(self, msg):
        if not self.is_quadruped:
            # g1's lowstate has no battery field; skip battery reporting.
            return
        try:
            self.battery = float(msg.bms_state.soc)
            battery_msg = Float32()
            battery_msg.data = float(self.battery)
            self.publisher_battery.publish(battery_msg)
        except Exception as e:
            self.logger.error(f"Error in LowStateMessageHandler: {e}")
            import traceback
            traceback.print_exc()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-interface", default="enP8p1s0",
                        help="Network interface connected to the robot")
    parsed_args, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = Ros2UnitreeManagerNode(parsed_args.network_interface)
    try:
        rclpy.spin(node)
    finally:
        if node.gait is not None:
            node.gait.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
