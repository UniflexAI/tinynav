import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.b2.sport.sport_client import SportClient as SportClientB2
from std_msgs.msg import Float32, String
from enum import Enum
import logging
import time
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_DEVICE_CMD_HZ = 50.0
MIN_DEVICE_CMD_INTERVAL_S = 1.0 / MAX_DEVICE_CMD_HZ
UNITREE_DEVICE_DRY_RUN = False


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
        # Startup stand is intentionally disabled; velocity/action commands still go to the device.
        # self.sport_client.SwitchGait(1)
        # self.sport_client.StandUp()
        # self.sport_client.BalanceStand()
        self._robot_status = RobotStatus.STANDUP
        self.battery = 0.0
        self.last_twist_time = None
        self.logger = logging.getLogger(__name__)
        self._cmd_lock = threading.Lock()
        self._pending_twist = None
        self._last_device_cmd_time = 0.0

        self.twist_subscriber = self.create_subscription(Twist, "/cmd_vel", self.TwistMessageHandler, 10)

        self.action_subscriber = ChannelSubscriber("rt/service/command", String_)
        self.action_subscriber.Init(self.ActionMessageHandler, 10)

        lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
        lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        
        self.publisher_battery = self.create_publisher(Float32, '/battery', 10)
        self.publisher_robot_status = self.create_publisher(String, '/robot_status', 10)

        self._status_timer = self.create_timer(1.0, self._publish_robot_status)
        self._cmd_flush_timer = self.create_timer(MIN_DEVICE_CMD_INTERVAL_S, self._flush_pending_twist)
        if UNITREE_DEVICE_DRY_RUN:
            self.get_logger().info("Unitree device dry-run enabled: motion commands are logged only.")
        else:
            self.get_logger().info("Unitree device command sending enabled.")
        self.logger.info(f"Unitree device command rate limited to {MAX_DEVICE_CMD_HZ:.0f} Hz.")
        self.get_logger().info("Unitree control initialization finished.")

    # twist message handler
    def TwistMessageHandler(self, msg: Twist):
        current_time = time.monotonic()
        if self.last_twist_time is not None:
            time_interval = current_time - self.last_twist_time
            self.logger.debug(f"cmd_vel callback time interval: {time_interval * 1000:.2f} ms")
        self.last_twist_time = current_time

        command = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
        self._queue_or_send_twist(command, current_time)

    def _queue_or_send_twist(self, command, current_time):
        send_now = False
        with self._cmd_lock:
            if current_time - self._last_device_cmd_time >= MIN_DEVICE_CMD_INTERVAL_S:
                self._last_device_cmd_time = current_time
                self._pending_twist = None
                send_now = True
            else:
                self._pending_twist = command

        if send_now:
            self._send_twist_command(command)

    def _flush_pending_twist(self):
        command = None
        current_time = time.monotonic()
        with self._cmd_lock:
            if (
                self._pending_twist is not None
                and current_time - self._last_device_cmd_time >= MIN_DEVICE_CMD_INTERVAL_S
            ):
                command = self._pending_twist
                self._pending_twist = None
                self._last_device_cmd_time = current_time

        if command is not None:
            self._send_twist_command(command)

    def _send_twist_command(self, command):
        vx, vy, wz = command
        if UNITREE_DEVICE_DRY_RUN:
            self.get_logger().info(f"dry-run before Unitree command vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}")
            return

        self.get_logger().info(f"before Unitree command vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}")
        if vx != 0.0 or vy != 0.0 or wz != 0.0:
            self.logger.debug(f"Moving with velocity: {vx}, {vy}, {wz}")
            self.sport_client.Move(vx, vy, wz)
            return

        self.sport_client.StopMove()

    def ActionMessageHandler(self, msg: String_):
        if msg.data.split(" ")[0] == "play":
            action_key = msg.data.split(" ")[1]
            if action_key == "sit":
                if UNITREE_DEVICE_DRY_RUN:
                    self.get_logger().info("dry-run Unitree action: StandDown")
                else:
                    self.get_logger().info("before Unitree action: StandDown")
                    self.sport_client.StandDown()
                self._robot_status = RobotStatus.SITTING
            elif action_key == "stand":
                if UNITREE_DEVICE_DRY_RUN:
                    self.get_logger().info("dry-run Unitree action: StandUp + BalanceStand")
                else:
                    self.get_logger().info("before Unitree action: StandUp + BalanceStand")
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
