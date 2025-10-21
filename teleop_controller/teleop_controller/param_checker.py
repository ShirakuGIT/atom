import rclpy
from rclpy.node import Node
import time

def main(args=None):
    rclpy.init(args=args)
    node = Node('param_checker_node')

    node.declare_parameter('robot_description', '')

    urdf_string = ""
    retries = 0
    max_retries = 10 # Let's be extra patient
    while not urdf_string and retries < max_retries and rclpy.ok():
        urdf_string = node.get_parameter('robot_description').get_parameter_value().string_value
        if not urdf_string:
            node.get_logger().info("Waiting for 'robot_description' parameter...")
            time.sleep(1.0)
        retries += 1

    if urdf_string:
        node.get_logger().info("SUCCESS: Successfully got 'robot_description' parameter!")
    else:
        node.get_logger().error("FAILURE: Could not get 'robot_description' parameter.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
