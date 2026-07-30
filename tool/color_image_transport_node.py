"""Standalone compressed->raw color image transport for map_node.py.

build_map_node.py's own __main__ instantiates ImageTransportsNode alongside its BagPlayer, but
map_node.py has no bag player of its own (it expects a live sensor or `ros2 bag play`), so this
just runs that same node standalone when testing relocalization against a played-back bag.
"""

import rclpy

from tinynav.core.build_map_node import ImageTransportsNode


def main(args=None):
    rclpy.init(args=args)
    node = ImageTransportsNode()
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
