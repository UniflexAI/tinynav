import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pynput import keyboard
import time


class KeyboardTeleopNode(Node):
    """Global arrow-key teleop publishing /cmd_vel at 20 Hz.

    pynput keydown/keyup maintains the pressed-key set; the timer converts it
    to a Twist. Publishes only while keys are held, then keeps publishing zero
    for RELEASE_FLUSH_S after release (flushing any stale command in the gz
    diff-drive) and goes silent — a constant idle stream would fight the
    autonomous controller interleaving on /cmd_vel. Captures keys globally on
    the X display -- arrows drive the robot from any window.

    Only the four arrow keys are tracked: the global listener also sees keys
    typed in unrelated windows, and a key whose release event is lost would
    otherwise stay in the set forever (stuck arrow = robot never stops, stuck
    other key = endless zero stream).
    """

    LINEAR_VEL = 0.5
    ANGULAR_VEL = 0.5
    RELEASE_FLUSH_S = 1.0
    ARROWS = {keyboard.Key.up, keyboard.Key.down,
              keyboard.Key.left, keyboard.Key.right}

    def __init__(self):
        super().__init__('keyboard_teleop_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pressed = set()
        self.flush_until = 0.0
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.timer = self.create_timer(0.05, self.timer_callback)

    def on_press(self, key):
        if key in self.ARROWS:
            self.pressed.add(key)

    def on_release(self, key):
        if key in self.ARROWS:
            self.pressed.discard(key)
            self.flush_until = time.monotonic() + self.RELEASE_FLUSH_S

    def timer_callback(self):
        cmd = Twist()
        if keyboard.Key.up in self.pressed:
            cmd.linear.x = self.LINEAR_VEL
        if keyboard.Key.down in self.pressed:
            cmd.linear.x = -self.LINEAR_VEL
        if keyboard.Key.left in self.pressed:
            cmd.angular.z = self.ANGULAR_VEL
        if keyboard.Key.right in self.pressed:
            cmd.angular.z = -self.ANGULAR_VEL
        if self.pressed or time.monotonic() < self.flush_until:
            self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
