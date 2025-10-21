import rclpy
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)
    node = Node('hello_node')
    node.get_logger().info('Hello from ROS 2!')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
