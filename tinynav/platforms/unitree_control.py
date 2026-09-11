import argparse
import math
import os
import threading
import rclpy
from rclpy.node import Node
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from std_msgs.msg import Float32, String
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
        if not math.isfinite(cmd_vel_timeout_s) or cmd_vel_timeout_s <= 0:
            raise ValueError(f"cmd_vel_timeout_s must be finite and positive, got {cmd_vel_timeout_s!r}")
        self.robot_model = robot_model
        self.is_quadruped = robot_model in _QUADRUPED_ROBOT_MODELS

        self.channel = ChannelFactoryInitialize(0, networkInterface)
        self.sport_client = _build_sport_client(robot_model)
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        self.safety_client = _build_sport_client(robot_model)
        self.safety_client.SetTimeout(min(0.1, cmd_vel_timeout_s / 4.0))
        self.safety_client.Init()
        self._safety_stop_lock = threading.Lock()
        if self.is_quadruped:
            self.sport_client.ClassicWalk(True)
        self._robot_status = RobotStatus.SITTING
        self.battery = 0.0
        self.last_twist_time = None
        self.logger = self.get_logger()

        self._cmd_vel_timeout_s = cmd_vel_timeout_s
        self._cmd_vel_lock = threading.Lock()
        self._cmd_vel_last_nonzero_at: float | None = None
        self._cmd_vel_armed = False

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
        self._cmd_vel_watchdog_timer = self.create_timer(watchdog_period_s, self._cmd_vel_watchdog_tick)

    # twist message handler
    def TwistMessageHandler(self, msg: Twist_):
        current_time = time.time()
        if self.last_twist_time is not None:
            time_interval = current_time - self.last_twist_time
            self.logger.debug(f"cmd_vel callback time interval: {time_interval*1000:.2f} ms")
        self.last_twist_time = current_time

        if  (msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0):
            self.logger.debug(f"Moving with velocity: {msg.linear.x}, {msg.linear.y}, {msg.angular.z}")
            dispatch_time = time.monotonic()
            with self._cmd_vel_lock:
                self._cmd_vel_last_nonzero_at = dispatch_time
                self._cmd_vel_armed = True
            self.sport_client.Move(msg.linear.x, msg.linear.y, msg.angular.z)
            
            if time.monotonic() - dispatch_time > self._cmd_vel_timeout_s:
                self._send_safety_stop("Move() completed after its own deadline", dispatch_time)
        else:
            with self._cmd_vel_lock:
                self._cmd_vel_armed = False
            self.sport_client.StopMove()
        time.sleep(0.02)

    def _cmd_vel_watchdog_tick(self):
        with self._cmd_vel_lock:
            last_nonzero_at = self._cmd_vel_last_nonzero_at
            stale = (
                self._cmd_vel_armed
                and last_nonzero_at is not None
                and time.monotonic() - last_nonzero_at > self._cmd_vel_timeout_s
            )
        if stale:
            self._send_safety_stop("cmd_vel stream stale", last_nonzero_at)

    def _send_safety_stop(self, reason: str, observed_at: float):
        with self._safety_stop_lock:
            code = self.safety_client.StopMove()
        with self._cmd_vel_lock:
            if self._cmd_vel_last_nonzero_at != observed_at:
                return
            if code == 0:
                self._cmd_vel_armed = False
        if code == 0:
            self.logger.warning(f"{reason}; StopMove sent")
        else:
            # _cmd_vel_armed stays True, so the next watchdog tick retries.
            self.logger.error(f"{reason}; StopMove failed: code={code}")

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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
