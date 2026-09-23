"""Unitree A2/A2W control bridge.

IMPORTANT:
1) Initialize Unitree ChannelFactory + SportClient BEFORE importing rclpy.
2) Keep ROS on FastDDS — Unitree already owns a Cyclone participant; a second
   Cyclone domain (especially with a missing CYCLONEDDS_URI file) fails with
   "Precondition Not Met".
"""
from __future__ import annotations

import logging
import os
import time
from enum import Enum

# Drop broken/image-default Cyclone URI so Unitree binds the NIC we pass in.
os.environ.pop("CYCLONEDDS_URI", None)

# A2W uses the A2 SportClient (there is no unitree_sdk2py.a2w).
from unitree_sdk2py.a2.sport.sport_client import SportClient as SportClientA2
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RobotStatus(Enum):
    STANDUP = "standup"
    SITTING = "sitting"


def _init_unitree_sport(network_interface: str) -> SportClientA2:
    print("ChannelFactoryInitialize...", flush=True)
    ChannelFactoryInitialize(0, network_interface)
    print("Channel initialized", flush=True)
    sport_client = SportClientA2()
    print("SportClient created", flush=True)
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    print("SportClient Init done", flush=True)
    time.sleep(1.0)

    # Do not StandUp on start. A2 has SwitchGait, not ContinuousGait.
    code = sport_client.StopMove()
    print(f"StopMove code={code}", flush=True)
    time.sleep(0.5)
    code = sport_client.SwitchGait(0)
    print(f"SwitchGait(0) code={code}", flush=True)
    return sport_client


def main(args=None):
    # Unitree DDS first — before any rclpy import.
    sport_client = _init_unitree_sport("enP8p1s0")

    # ROS side: FastDDS only (do not share Cyclone with Unitree SDK).
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    os.environ.pop("CYCLONEDDS_URI", None)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, String

    class Ros2UnitreeManagerNode(Node):
        def __init__(self, sport_client: SportClientA2):
            super().__init__('ros2_unitree_manager')
            self.sport_client = sport_client
            self._robot_status = RobotStatus.STANDUP
            self.battery = 0.0
            self.last_twist_time = None
            self.logger = logging.getLogger(__name__)

            self.twist_subscriber = ChannelSubscriber("rt/cmd_vel", Twist_)
            self.twist_subscriber.Init(self.TwistMessageHandler, 10)

            self.action_subscriber = ChannelSubscriber("rt/service/command", String_)
            self.action_subscriber.Init(self.ActionMessageHandler, 10)

            lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
            lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

            self.publisher_battery = self.create_publisher(Float32, '/battery', 10)
            self.publisher_robot_status = self.create_publisher(String, '/robot_status', 10)
            self._status_timer = self.create_timer(1.0, self._publish_robot_status)
            print("ROS node ready", flush=True)

        def TwistMessageHandler(self, msg: Twist_):
            current_time = time.time()
            if self.last_twist_time is not None:
                time_interval = current_time - self.last_twist_time
                self.logger.debug(
                    f"cmd_vel callback time interval: {time_interval*1000:.2f} ms"
                )
            self.last_twist_time = current_time

            if msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0:
                self.logger.debug(
                    f"Moving with velocity: {msg.linear.x}, {msg.linear.y}, {msg.angular.z}"
                )
                self.sport_client.Move(msg.linear.x, msg.linear.y, msg.angular.z)
            else:
                self.sport_client.StopMove()
            time.sleep(0.02)

        def ActionMessageHandler(self, msg: String_):
            if msg.data.split(" ")[0] == "play":
                action_key = msg.data.split(" ")[1]
                if action_key == "sit":
                    self.logger.info("Sitting")
                    self.sport_client.StandDown()
                    self._robot_status = RobotStatus.SITTING
                elif action_key == "stand":
                    self.logger.info("Standing")
                    self.sport_client.StandUp()
                    self.sport_client.BalanceStand()
                    self._robot_status = RobotStatus.STANDUP

        def _publish_robot_status(self):
            msg = String()
            msg.data = self._robot_status.value
            self.publisher_robot_status.publish(msg)

        def LowStateMessageHandler(self, msg: LowState_):
            try:
                self.battery = float(msg.bms_state.soc)
                battery_msg = Float32()
                battery_msg.data = float(self.battery)
                self.publisher_battery.publish(battery_msg)
            except Exception as e:
                self.logger.error(f"Error in LowStateMessageHandler: {e}")
                import traceback
                traceback.print_exc()

    rclpy.init(args=args)
    node = Ros2UnitreeManagerNode(sport_client)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
