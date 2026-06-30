import rclpy
from rclpy.node import Node
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.b2.sport.sport_client import SportClient as SportClientB2
from std_msgs.msg import Float32, String
from enum import Enum
import logging
import time
from tinynav.core.latency_trace import (
    TRACE_TOPIC,
    make_trace_publisher,
    parse_trace_event,
    publish_trace,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RobotStatus(Enum):
    STANDUP = "standup"
    SITTING = "sitting"


class Ros2UnitreeManagerNode(Node):
    def __init__(self, networkInterface: str = "enP8p1s0"):
        super().__init__('ros2_unitree_manager')
        self.channel = ChannelFactoryInitialize(0, networkInterface)
        self.sport_client = SportClientB2()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        self.sport_client.SwitchGait(1)
        self._robot_status = RobotStatus.SITTING
        self.battery = 0.0
        self.last_twist_time = None
        self.logger = logging.getLogger(__name__)
        self.active_trace_id = None

        self.twist_subscriber = ChannelSubscriber("rt/cmd_vel", Twist_)
        self.twist_subscriber.Init(self.TwistMessageHandler, 10)

        self.action_subscriber = ChannelSubscriber("rt/service/command", String_)
        self.action_subscriber.Init(self.ActionMessageHandler, 10)

        lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
        lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        
        self.publisher_battery = self.create_publisher(Float32, '/battery', 10)
        self.publisher_robot_status = self.create_publisher(String, '/robot_status', 10)
        self.latency_trace_pub = make_trace_publisher(self)
        self.create_subscription(String, TRACE_TOPIC, self.TraceMessageHandler, 10)

        self._status_timer = self.create_timer(1.0, self._publish_robot_status)

    def TraceMessageHandler(self, msg: String):
        event = parse_trace_event(msg.data)
        if not event:
            return
        if (
            event.get("stage") in ("backend", "planning", "cmd_vel_control")
            and event.get("trace_id")
        ):
            self.active_trace_id = event.get("trace_id")

    # twist message handler
    def TwistMessageHandler(self, msg: Twist_):
        current_time = time.time()
        if self.last_twist_time is not None:
            time_interval = current_time - self.last_twist_time
            self.logger.debug(f"cmd_vel callback time interval: {time_interval*1000:.2f} ms")
        self.last_twist_time = current_time
        publish_trace(
            self,
            self.latency_trace_pub,
            self.active_trace_id,
            "unitree_control",
            "cmd_vel_received",
            vx=float(msg.linear.x),
            vy=float(msg.linear.y),
            wz=float(msg.angular.z),
        )
        
        if  (msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0):
            self.logger.debug(f"Moving with velocity: {msg.linear.x}, {msg.linear.y}, {msg.angular.z}")
            self.sport_client.Move(msg.linear.x, msg.linear.y, msg.angular.z)
            command = "Move"
        else:
            self.sport_client.StopMove()
            command = "StopMove"
        publish_trace(
            self,
            self.latency_trace_pub,
            self.active_trace_id,
            "unitree_control",
            "robot_command_sent",
            command=command,
        )
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


def main(args=None):
    rclpy.init(args=args)
    node = Ros2UnitreeManagerNode("enP8p1s0")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
