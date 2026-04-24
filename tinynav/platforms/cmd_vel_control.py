import argparse
import math
import logging
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__("cmd_vel_control_node")
        self.logger = logging.getLogger(__name__)

        # Camera-to-robot base transform (left-multiply camera pose to get robot pose)
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1],
        ], dtype=np.float64)

        # Previous published command for acceleration limiting
        self._prev_vx = 0.0
        self._prev_vyaw = 0.0
        self._prev_forward_err = 0.0
        self._prev_heading_err = 0.0
        self._forward_err_i = 0.0
        self._heading_err_i = 0.0

        # Propagated camera pose from perception /slam/odom_100hz
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.rotation = np.eye(3)
        self._odom_100hz_valid = False

        # Trajectory
        self.trajectory: Path | None = None

        # Subscribers
        self.create_subscription(Odometry, "/slam/odom_100hz", self._odom_100hz_cb, 10)
        self.create_subscription(Path, "/planning/trajectory_path", self._traj_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, "/control/debug", 10)

        # TIMER_HZ = 50.0
        self.create_timer(1.0 / 50.0, self._control_loop)

    def _odom_100hz_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self.position = np.array([p.x, p.y, p.z])
        self.velocity = np.array([v.x, v.y, v.z])
        self.rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self._odom_100hz_valid = True

    def _traj_cb(self, msg: Path):
        if msg.poses:
            self.trajectory = msg

    def _control_loop(self):
        if not self._odom_100hz_valid or self.trajectory is None:
            self._publish_zero()
            return

        T_cam = np.eye(4)
        T_cam[:3, :3] = self.rotation
        T_cam[:3, 3] = self.position
        T_robot = T_cam @ self.T_robot_to_camera

        robot_pos = T_robot[:3, 3]
        robot_yaw = math.atan2(T_robot[1, 0], T_robot[0, 0])
        
        forward_err, heading_err = self._compute_path_error(robot_pos, robot_yaw)
        if forward_err is None:
            self._publish_zero()
            return

        # TIMER_HZ = 50.0
        dt = 1.0 / 50.0

        self._forward_err_i += forward_err * dt
        # FORWARD_I_CLAMP = 0.6
        self._forward_err_i = float(np.clip(self._forward_err_i, -0.6, 0.6))
        d_forward = (forward_err - self._prev_forward_err) / dt
        vx_cmd = (
            # FORWARD_KP = 1.2, FORWARD_KI = 0.1, FORWARD_KD = 0.05
            1.2 * forward_err
            + 0.1 * self._forward_err_i
            + 0.05 * d_forward
        )
        vx_target = float(np.clip(vx_cmd, 0.0, 0.5))

        self._heading_err_i += heading_err * dt
        # HEADING_I_CLAMP = 1.2
        self._heading_err_i = float(np.clip(self._heading_err_i, -1.2, 1.2))
        d_heading = (heading_err - self._prev_heading_err) / dt
        vyaw_cmd = (
            # HEADING_KP = 1.5, HEADING_KI = 0.05, HEADING_KD = 0.08
            1.5 * heading_err
            + 0.05 * self._heading_err_i
            + 0.08 * d_heading
        )
        vyaw_target = float(np.clip(vyaw_cmd, -0.5, 0.5))

        # Keep rate limits for smooth control
        # MAX_LINEAR_ACC = 0.3, MAX_ANGULAR_ACC = 2.5
        vx = self._clamp_step(vx_target, self._prev_vx, 0.3 * dt)
        vyaw = self._clamp_step(vyaw_target, self._prev_vyaw, 2.5 * dt)

        self._prev_vx = vx
        self._prev_vyaw = vyaw
        self._prev_forward_err = forward_err
        self._prev_heading_err = heading_err

        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = vyaw
        self.cmd_pub.publish(cmd)

        dbg = Float32MultiArray()
        dbg.data = [float(heading_err), float(vx), float(vyaw)]
        self.debug_pub.publish(dbg)
        self.logger.info(
            "forward_err=%.4f heading_err=%.4f vx=%.3f wz=%.3f",
            forward_err,
            heading_err,
            vx,
            vyaw,
        )

    def _compute_path_error(self, robot_pos: np.ndarray, robot_yaw: float):
        if self.trajectory is None or not self.trajectory.poses:
            return None, None

        poses = self.trajectory.poses

        rx, ry = robot_pos[0], robot_pos[1]

        best_idx = 0
        best_dist = float("inf")
        for i, pose_stamped in enumerate(poses):
            p = pose_stamped.pose.position
            x, y = float(p.x), float(p.y)
            d = (x - rx) ** 2 + (y - ry) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i

        target_pose = None
        for i in range(best_idx, len(poses)):
            p = poses[i].pose.position
            x, y = float(p.x), float(p.y)
            # LOOKAHEAD_DIST = 0.2
            if math.hypot(x - rx, y - ry) >= 0.2:
                target_pose = poses[i].pose
                break
        if target_pose is None:
            target_pose = poses[-1].pose

        dx = float(target_pose.position.x) - rx
        dy = float(target_pose.position.y) - ry
        forward_err = math.hypot(dx, dy)

        q = target_pose.orientation
        target_yaw = float(R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2])
        heading_err = self._wrap_angle(target_yaw - robot_yaw)
        return forward_err, heading_err

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _clamp_step(target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def _publish_zero(self):
        # TIMER_HZ = 50.0, MAX_LINEAR_ACC = 0.3, MAX_ANGULAR_ACC = 2.5
        dt = 1.0 / 50.0
        self._prev_vx = self._clamp_step(0.0, self._prev_vx, 0.3 * dt)
        self._prev_vyaw = self._clamp_step(0.0, self._prev_vyaw, 2.5 * dt)
        # Decay integrators when tracking is unavailable to avoid windup on resume.
        self._forward_err_i *= 0.9
        self._heading_err_i *= 0.9
        cmd = Twist()
        cmd.linear.x = self._prev_vx
        cmd.angular.z = self._prev_vyaw
        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        self.logger.info("Destroying cmd_vel_control node.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="cmd_vel_control.log", help="Path to the log file")
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(parsed_args.log_file)],
    )

    node = CmdVelControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
