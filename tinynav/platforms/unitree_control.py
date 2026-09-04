import argparse
import os
import rclpy
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

        self._status_timer = self.create_timer(1.0, self._publish_robot_status)

    # twist message handler
    def TwistMessageHandler(self, msg: Twist_):
        current_time = time.time()
        if self.last_twist_time is not None:
            time_interval = current_time - self.last_twist_time
            self.logger.debug(f"cmd_vel callback time interval: {time_interval*1000:.2f} ms")
        self.last_twist_time = current_time

        if  (msg.linear.x != 0 or msg.linear.y != 0 or msg.angular.z != 0):
            self.logger.debug(f"Moving with velocity: {msg.linear.x}, {msg.linear.y}, {msg.angular.z}")
            self.sport_client.ClassicWalk(True)
            self.sport_client.Move(msg.linear.x, msg.linear.y, msg.angular.z)
        else:
            self.sport_client.StopMove()
        time.sleep(0.02)

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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
