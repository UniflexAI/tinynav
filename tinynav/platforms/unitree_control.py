import rclpy
from rclpy.node import Node
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from std_msgs.msg import Float32
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Ros2UnitreeManagerNode(Node):
    def __init__(self, networkInterface: str = "eno1"):
        super().__init__('ros2_unitree_manager')
        self.channel = ChannelFactoryInitialize(0, networkInterface)
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(1.0)
        self.sport_client.Init()

        self.twist_subscriber = ChannelSubscriber("rt/cmd_vel", Twist_)
        self.twist_subscriber.Init(self.TwistMessageHandler, 10)

        self.action_subscriber = ChannelSubscriber("rt/service/command", String_)
        self.action_subscriber.Init(self.ActionMessageHandler, 10)

        lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
        lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        
        self.publisher_battery = self.create_publisher(Float32, '/battery', 10)
        self.battery = 0.0
        self.last_twist_time = None

        self.logger = logging.getLogger(__name__)

    # twist message handler
    def TwistMessageHandler(self, msg: Twist_):
        current_time = time.time()
        if self.last_twist_time is not None:
            time_interval = current_time - self.last_twist_time
            self.logger.debug(f"cmd_vel callback time interval: {time_interval*1000:.2f} ms")
        self.last_twist_time = current_time
        
        if  (msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0):
            self.logger.debug(f"Moving with velocity: {msg.linear.x}, {msg.linear.y}, {msg.angular.z}")
            self.sport_client.Move(msg.linear.x, msg.linear.y, msg.angular.z)
        else:
            self.sport_client.StopMove()
        time.sleep(0.1)

    def ActionMessageHandler(self, msg: String_):
        if msg.data.split(" ")[0] == "play":
            action_key = msg.data.split(" ")[1]
            if action_key == "sit":
                self.logger.debug("Sitting")
                self.sport_client.StandDown()
            elif action_key == "stand":
                self.logger.debug("Standing")
                self.sport_client.StandUp()
                self.sport_client.BalanceStand()
    
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
    node = Ros2UnitreeManagerNode("eno1")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

