#!/usr/bin/env python3
"""PI velocity servo for the Go2 skating gait.

The open-loop plant is stable but has a large, only roughly known gain
(~13x on vx from the stance-foot-speed arithmetic, ~0.7x on wz). This node
closes the loop on measured body twist:

    /cmd_vel (absolute) + gt_twist  ->  PI  ->  cmd_vel  (plant input, the trot
                                                          controller clamps at
                                                          +-0.045 m/s)

Feedforward fallback (des/13, des/0.7) whenever the measurement goes stale,
so the robot still tracks roughly if gz truth drops out.
"""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def clamp(v, lim):
    return max(-lim, min(lim, v))


class CmdVelServo(Node):
    DT = 0.05          # 20 Hz loop
    STALE = 1.0        # seconds without gt_twist before feedforward-only
    FF_X = 13.0        # plant gain estimate for feedforward
    FF_W = 0.7

    def __init__(self):
        super().__init__("cmd_vel_servo")
        self.declare_parameter("kp_x", 0.15)
        self.declare_parameter("ki_x", 0.25)
        self.declare_parameter("kp_w", 1.0)
        self.declare_parameter("ki_w", 0.5)
        self.declare_parameter("i_max_x", 0.12)   # integral headroom, plant units
        self.declare_parameter("i_max_w", 0.5)
        self.declare_parameter("u_max_x", 0.044)  # inside the trot controller's clamp
        self.declare_parameter("u_max_w", 0.9)
        g = lambda k: self.get_parameter(k).get_parameter_value().double_value
        self.kp_x, self.ki_x = g("kp_x"), g("ki_x")
        self.kp_w, self.ki_w = g("kp_w"), g("ki_w")
        self.i_max_x, self.i_max_w = g("i_max_x"), g("i_max_w")
        self.u_max_x, self.u_max_w = g("u_max_x"), g("u_max_w")

        self.des = (0.0, 0.0)
        self.meas = (0.0, 0.0)
        self.meas_t = 0.0
        self.ix = 0.0
        self.iw = 0.0
        # absolute: planners and teleop publish /cmd_vel with no namespace;
        # the plant input below stays namespaced (/robot1/cmd_vel) so this
        # node never hears its own output
        self.create_subscription(Twist, "/cmd_vel", self.cb_des, 10)
        self.create_subscription(Twist, "gt_twist", self.cb_meas, 10)
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_timer(self.DT, self.tick)

    def cb_des(self, m):
        self.des = (m.linear.x, m.angular.z)

    def cb_meas(self, m):
        self.meas = (m.linear.x, m.angular.z)
        self.meas_t = time.time()

    def tick(self):
        dx, dw = self.des
        out = Twist()
        if abs(dx) < 0.02 and abs(dw) < 0.05:
            self.ix = self.iw = 0.0
            self.pub.publish(out)
            return
        fresh = (time.time() - self.meas_t) < self.STALE
        if fresh:
            ex = dx - self.meas[0]
            ew = dw - self.meas[1]
        else:
            ex = ew = 0.0
        self.ix = clamp(self.ix + ex * self.DT, self.i_max_x)
        self.iw = clamp(self.iw + ew * self.DT, self.i_max_w)
        if fresh:
            ux = self.kp_x * ex + self.ki_x * self.ix
            uw = self.kp_w * ew + self.ki_w * self.iw
        else:
            ux = dx / self.FF_X
            uw = dw / self.FF_W
        out.linear.x = clamp(ux, self.u_max_x)
        out.angular.z = clamp(uw, self.u_max_w)
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(CmdVelServo())


if __name__ == "__main__":
    main()
