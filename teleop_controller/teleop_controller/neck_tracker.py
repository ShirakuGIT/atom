#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class NeckTrackerNode(Node):
    def __init__(self):
        super().__init__('neck_tracker')
        
        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Neck controller publisher
        self.neck_pub = self.create_publisher(
            JointTrajectory,
            '/neck_joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Camera subscriber
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(
            Image,
            '/neck_camera/image_raw',
            self.camera_callback,
            10
        )
        
        # Video recording
        self.video_writer = None
        self.recording = False
        
        # Control loop (10 Hz)
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("🎥 Neck Tracker Started - Following Head Movements")
    
    def quaternion_to_euler(self, qx, qy, qz, qw):
        """Convert quaternion to roll, pitch, yaw"""
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (qw * qy - qz * qx)
        pitch = math.asin(np.clip(sinp, -1, 1))
        
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def control_loop(self):
        """Track human head and move neck"""
        try:
            head_tf = self.tf_buffer.lookup_transform(
                'world',
                'head_rod_meta',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            quat = head_tf.transform.rotation
            roll, pitch, yaw = self.quaternion_to_euler(quat.x, quat.y, quat.z, quat.w)
            
            # Map head orientation to neck (scale down)
            neck_yaw = yaw * 0.7
            neck_pitch = pitch * 0.6
            neck_roll = roll * 0.5
            
            # Clamp to joint limits
            neck_yaw = np.clip(neck_yaw, -1.57, 1.57)
            neck_pitch = np.clip(neck_pitch, -0.8, 0.8)
            neck_roll = np.clip(neck_roll, -0.5, 0.5)
            
            self.publish_neck_command([neck_yaw, neck_pitch, neck_roll])
            
        except TransformException:
            pass
    
    def publish_neck_command(self, joint_positions):
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = ['neck_yaw_joint', 'neck_pitch_joint', 'neck_roll_joint']
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0, 0.0, 0.0]
        point.time_from_start = Duration(sec=0, nanosec=100000000)
        
        trajectory.points.append(point)
        self.neck_pub.publish(trajectory)
    
    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            cv2.imshow('Neck Camera View', cv_image)
            cv2.waitKey(1)
            
            if not self.recording:
                self.start_recording(cv_image.shape)
            
            if self.video_writer:
                self.video_writer.write(cv_image)
                
        except Exception as e:
            self.get_logger().error(f'Camera error: {e}')
    
    def start_recording(self, frame_shape):
        height, width, _ = frame_shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            'neck_camera_recording.mp4',
            fourcc,
            30.0,
            (width, height)
        )
        self.recording = True
        self.get_logger().info("📹 Recording started: neck_camera_recording.mp4")


def main():
    rclpy.init()
    node = NeckTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.video_writer:
            node.video_writer.release()
        cv2.destroyAllWindows()
        node.get_logger().info("🛑 Neck tracker stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
