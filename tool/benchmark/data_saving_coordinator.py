#!/usr/bin/env python3

import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class DataSavingCoordinator(Node):
    """
    Coordinator node that handles data saving process coordination:
    1. Publishes stop signal when bag finishes
    2. Waits for save completion signal
    3. Signals benchmark completion
    """

    def __init__(self, timeout_seconds: float = 30.0):
        super().__init__("data_saving_coordinator")
        self.timeout_seconds = timeout_seconds
        self.save_completed = False

        # Publishers and subscribers
        self.mapping_stop_pub = self.create_publisher(Bool, "/benchmark/stop", 10)
        self.mapping_save_finished_sub = self.create_subscription(
            Bool, "/benchmark/data_saved", self.save_finished_callback, 10
        )

        self.get_logger().info(
            f"MappingCoordinator initialized with {timeout_seconds}s timeout"
        )

    def save_finished_callback(self, msg: Bool):
        self.get_logger().info(f"Received data save finished signal: {msg.data}")
        self.save_completed = True

    def send_stop_signal(self):
        stop_msg = Bool()
        stop_msg.data = True
        self.mapping_stop_pub.publish(stop_msg)
        self.get_logger().info("Sent benchmark stop signal")

    def wait_for_save_completion(self) -> bool:
        """
        Wait for mapping save completion or timeout.

        Returns:
            True if save completed successfully, False if timed out
        """
        start_time = time.time()

        self.get_logger().info("Waiting for data save completion...")
        while rclpy.ok() and not self.save_completed:
            if time.time() - start_time > self.timeout_seconds:
                self.get_logger().warn(
                    f"Timeout after {self.timeout_seconds}s waiting for data save completion"
                )
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

        if self.save_completed:
            self.get_logger().info("Data save completed successfully")
            return True

        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: data_saving_coordinator.py <timeout_seconds>")
        sys.exit(1)

    timeout_seconds = float(sys.argv[1])
    rclpy.init()
    coordinator = DataSavingCoordinator(timeout_seconds)

    try:
        time.sleep(1.0)
        coordinator.send_stop_signal()
        success = coordinator.wait_for_save_completion()
        coordinator.get_logger().info(f"Coordinator finished. Success: {success}")

    except Exception as e:
        coordinator.get_logger().error(f"Error in coordinator: {e}")
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
