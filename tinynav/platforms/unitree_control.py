import argparse
import os
import time
from enum import Enum

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import Float32, String
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from tinynav.platforms.command_watchdog import (
    VelocityCommandWatchdog,
    stop_unitree_motion,
)

# go2/b2 are quadrupeds sharing the same SportClient gait API (Move/StandUp/
# StandDown/BalanceStand/ClassicWalk). go2w/b2w are the wheeled variants of the
# same chassis — the vendored SDK has no separate go2w/b2w package, so they reuse
# the go2/b2 SportClient (same gait/lowstate API) as-is. g1 is a humanoid
# controlled through the FSM-based LocoClient instead, so it needs its own
# client, lowstate IDL, and stand/sit mapping.
_QUADRUPED_ROBOT_MODELS = ('go2', 'go2w', 'b2', 'b2w')
_SUPPORTED_ROBOT_MODELS = _QUADRUPED_ROBOT_MODELS + ('g1',)
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


class Ros2UnitreeManagerNode(Node):
    def __init__(self, networkInterface: str = "enP8p1s0", robot_model: str = ROBOT_TYPE,
                 cmd_vel_timeout_s: float = 0.5):
        super().__init__('ros2_unitree_manager')
        if robot_model not in _SUPPORTED_ROBOT_MODELS:
            raise ValueError(f"Unsupported robot model: {robot_model!r}, expected one of {_SUPPORTED_ROBOT_MODELS}")
        self.robot_model = robot_model
        self.is_quadruped = robot_model in _QUADRUPED_ROBOT_MODELS
        self.cmd_vel_watchdog = VelocityCommandWatchdog(cmd_vel_timeout_s)

        self.channel = ChannelFactoryInitialize(0, networkInterface)
        self.sport_client = _build_sport_client(robot_model)
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        # A blocked primary RPC must not prevent the watchdog from sending StopMove.
        self.safety_client = _build_sport_client(robot_model)
        self.safety_client.SetTimeout(min(0.1, cmd_vel_timeout_s / 4.0))
        self.safety_client.Init()
        if self.is_quadruped:
            self.sport_client.ClassicWalk(True)
        self._robot_status = RobotStatus.SITTING
        self.battery = 0.0
        self.last_twist_time = None
        self.logger = self.get_logger()
        self._watchdog_clock = Clock(clock_type=ClockType.STEADY_TIME)

        self.twist_subscriber = ChannelSubscriber("rt/cmd_vel", Twist_)
        self.twist_subscriber.Init(self.TwistMessageHandler, 10)

        self.action_subscriber = ChannelSubscriber("rt/service/command", String_)
        self.action_subscriber.Init(self.ActionMessageHandler, 10)

        lowstate_type, lowstate_topic = _lowstate_type_and_topic(robot_model)
        lowstate_subscriber = ChannelSubscriber(lowstate_topic, lowstate_type)
        lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.publisher_battery = self.create_publisher(Float32, '/battery', 10)
        self.publisher_robot_status = self.create_publisher(String, '/robot_status', 10)

        self._status_timer = self.create_timer(1.0, self._publish_robot_status)
        watchdog_period_s = min(0.05, cmd_vel_timeout_s / 2.0)
        self._cmd_vel_watchdog_timer = self.create_timer(
            watchdog_period_s, self._cmd_vel_watchdog_tick, clock=self._watchdog_clock)

    # twist message handler
    def TwistMessageHandler(self, msg: Twist_):
        current_time = time.time()
        if self.last_twist_time is not None:
            time_interval = current_time - self.last_twist_time
            self.logger.debug(f"cmd_vel callback time interval: {time_interval*1000:.2f} ms")
        self.last_twist_time = current_time

        if (msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0):
            self.logger.debug(f"Moving with velocity: {msg.linear.x}, {msg.linear.y}, {msg.angular.z}")
            self.cmd_vel_watchdog.observe_nonzero(time.monotonic())
            code = self.sport_client.Move(msg.linear.x, msg.linear.y, msg.angular.z)
            if code not in (0, None):
                self.logger.error(f"Move failed: code={code}")
        else:
            code = stop_unitree_motion(self.sport_client, self.robot_model)
            if code == 0:
                self.cmd_vel_watchdog.clear()
            else:
                self.logger.error(f"StopMove failed: code={code}")
        time.sleep(0.02)

    def _cmd_vel_watchdog_tick(self):
        if not self.cmd_vel_watchdog.consume_expiration(time.monotonic()):
            return
        code = stop_unitree_motion(self.safety_client, self.robot_model)
        if code == 0:
            self.logger.warning("cmd_vel stream stale; StopMove sent")
        else:
            self.logger.error(f"cmd_vel stream stale; StopMove failed: code={code}")
            self.cmd_vel_watchdog.observe_nonzero(time.monotonic())

    def ActionMessageHandler(self, msg: String_):
        self.logger.info(f"ActionMessageHandler received: {msg.data!r}")
        if msg.data.split(" ")[0] == "play":
            action_key = msg.data.split(" ")[1]
            if action_key == "sit":
                if self.is_quadruped:
                    code = self.sport_client.StandDown()
                    self.logger.info(f"Sitting: StandDown code={code}")
                else:
                    code = self.sport_client.StandUp2Squat()
                    self.logger.info(f"Sitting: StandUp2Squat code={code}")
                self._robot_status = RobotStatus.SITTING
            elif action_key == "stand":
                if self.is_quadruped:
                    code1 = self.sport_client.StandUp()
                    code2 = self.sport_client.BalanceStand()
                    self.logger.info(f"Standing: StandUp code={code1}, BalanceStand code={code2}")
                else:
                    code1 = self.sport_client.Damp()
                    time.sleep(0.5)
                    code2 = self.sport_client.Squat2StandUp()
                    self.logger.info(f"Standing: Damp code={code1}, Squat2StandUp code={code2}")
                self._robot_status = RobotStatus.STANDUP

    def _publish_robot_status(self):
        msg = String()
        msg.data = self._robot_status.value
        self.publisher_robot_status.publish(msg)

    def LowStateMessageHandler(self, msg):
        if not self.is_quadruped:
            # g1's lowstate has no battery field; skip battery reporting.
            return
        try:
            self.battery = float(msg.bms_state.soc)
            battery_msg = Float32()
            battery_msg.data = float(self.battery)
            self.publisher_battery.publish(battery_msg)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error in LowStateMessageHandler: {e}")
            import traceback
            traceback.print_exc()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-interface", default="enP8p1s0",
                        help="Network interface connected to the robot")
    parser.add_argument("--cmd-vel-timeout", type=float, default=0.5,
                        help="Stop after this many seconds without a fresh non-zero cmd_vel")
    parsed_args, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = Ros2UnitreeManagerNode(parsed_args.network_interface,
                                  cmd_vel_timeout_s=parsed_args.cmd_vel_timeout)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
