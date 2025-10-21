#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from tf2_ros import TransformException
from kdl_parser_py.urdf import treeFromString
from PyKDL import Chain, ChainIkSolverPos_LMA, ChainFkSolverPos_recursive, JntArray, Frame, Vector, Rotation
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
from collections import deque
import random
import math

class TeleopController(Node):
    def __init__(self):
        super().__init__('teleop_controller_node')
        
        # Frame configuration
        self.left_wrist_frame = 'left_wrist_pose_meta'
        self.right_wrist_frame = 'right_wrist_pose_meta'
        self.head_frame = 'head_stable_meta'  # Use stable for head tracking
        
        # Robot configuration
        self.left_robot_base = 'openarm_left_link0'
        self.left_robot_ee = 'openarm_left_hand_tcp'
        self.right_robot_base = 'openarm_right_link0'
        self.right_robot_ee = 'openarm_right_hand_tcp'
        self.neck_base = 'openarm_body_link0'
        self.neck_ee = 'neck_base'  
        
        # Parameters
        self.declare_parameter('control_rate', 20.0)  # Increased for smoother control
        self.declare_parameter('trajectory_duration', 0.1)  # Reduced for responsiveness
        self.declare_parameter('enable_smoothing', True)
        self.declare_parameter('smoothing_alpha', 0.7)  # For EMA
        self.declare_parameter('reach_tolerance', 1.2)  # Tolerance multiplier for reach check
        self.declare_parameter('ik_max_attempts', 10)  # Max IK attempts
        self.declare_parameter('ik_perturbation_range', 1.0)  # Perturbation range in rad
        self.declare_parameter('ik_eps', 0.01)  # IK solver epsilon - increased for looser convergence
        self.declare_parameter('ik_maxiter', 2000)  # IK solver max iterations - increased
        self.declare_parameter('ik_publish_on_failure', True)  # Publish best approximation on failure
        self.declare_parameter('ik_failure_tol', 0.05)  # Max error to publish on failure (m)
        self.declare_parameter('project_out_of_reach', True)  # Project target if out of reach
        self.declare_parameter('projection_factor', 0.95)  # Fraction of max reach to project to
        self.declare_parameter('log_projected', False)  # Log projection every time (spammy)
        self.declare_parameter('use_relative_scaling', True)  # Use relative motion scaling
        self.declare_parameter('motion_scale', 0.5)  # Scaling factor for relative motions
        
        # Get parameters
        self.control_rate = self.get_parameter('control_rate').value
        self.traj_duration = self.get_parameter('trajectory_duration').value
        self.enable_smoothing = self.get_parameter('enable_smoothing').value
        self.smoothing_alpha = self.get_parameter('smoothing_alpha').value
        self.reach_tolerance = self.get_parameter('reach_tolerance').value
        self.ik_max_attempts = self.get_parameter('ik_max_attempts').value
        self.ik_perturbation_range = self.get_parameter('ik_perturbation_range').value
        self.ik_eps = self.get_parameter('ik_eps').value
        self.ik_maxiter = self.get_parameter('ik_maxiter').value
        self.ik_publish_on_failure = self.get_parameter('ik_publish_on_failure').value
        self.ik_failure_tol = self.get_parameter('ik_failure_tol').value
        self.project_out_of_reach = self.get_parameter('project_out_of_reach').value
        self.projection_factor = self.get_parameter('projection_factor').value
        self.log_projected = self.get_parameter('log_projected').value
        self.use_relative_scaling = self.get_parameter('use_relative_scaling').value
        self.motion_scale = self.get_parameter('motion_scale').value
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Joint states
        self.left_kdl_joint_names = None
        self.right_kdl_joint_names = None
        self.neck_kdl_joint_names = None
        self.left_current_joints = None
        self.right_current_joints = None
        self.neck_current_joints = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        
        # Publishers
        self.left_joint_pub = self.create_publisher(JointTrajectory, '/left_joint_trajectory_controller/joint_trajectory', 10)
        self.right_joint_pub = self.create_publisher(JointTrajectory, '/right_joint_trajectory_controller/joint_trajectory', 10)
        self.neck_joint_pub = self.create_publisher(JointTrajectory, '/neck_joint_trajectory_controller/joint_trajectory', 10)
        
        # KDL chains and solvers
        self.left_chain = None
        self.right_chain = None
        self.neck_chain = None
        self.left_ik_solver = None
        self.right_ik_solver = None
        self.neck_ik_solver = None
        self.left_fk_solver = None
        self.right_fk_solver = None
        self.neck_fk_solver = None
        self.left_num_joints = 0
        self.right_num_joints = 0
        self.neck_num_joints = 0
        
        # Relative scaling state
        self.left_initial_pos = None
        self.left_initial_rot = None
        self.left_home_pos = None
        self.left_home_rot = None
        self.right_initial_pos = None
        self.right_initial_rot = None
        self.right_home_pos = None
        self.right_home_rot = None
        self.neck_initial_pos = None
        self.neck_initial_rot = None
        self.neck_home_pos = None
        self.neck_home_rot = None
        
        # Smoothing buffers (store last solution for EMA)
        self.left_last_joints = None
        self.right_last_joints = None
        self.neck_last_joints = None
        
        # Statistics
        self.left_success = 0
        self.left_failure = 0
        self.right_success = 0
        self.right_failure = 0
        self.neck_success = 0
        self.neck_failure = 0
        
        self.setup_complete = False
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("TELEOPERATION CONTROLLER STARTING")
        self.get_logger().info(f"Control rate: {self.control_rate} Hz")
        self.get_logger().info(f"Trajectory duration: {self.traj_duration}s")
        self.get_logger().info(f"Smoothing: {self.enable_smoothing}, alpha: {self.smoothing_alpha}")
        self.get_logger().info(f"Reach tolerance: {self.reach_tolerance}")
        self.get_logger().info(f"IK max attempts: {self.ik_max_attempts}, perturbation: ±{self.ik_perturbation_range} rad")
        self.get_logger().info(f"IK eps: {self.ik_eps}, maxiter: {self.ik_maxiter}")
        self.get_logger().info(f"IK publish on failure: {self.ik_publish_on_failure}, failure tol: {self.ik_failure_tol}")
        self.get_logger().info(f"Project out of reach: {self.project_out_of_reach}, factor: {self.projection_factor}")
        self.get_logger().info(f"Log projected every cycle: {self.log_projected}")
        self.get_logger().info(f"Use relative scaling: {self.use_relative_scaling}, motion scale: {self.motion_scale}")
        self.get_logger().info("=" * 60)
        
        # Setup robot description
        self.param_client = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                return
            self.get_logger().info('Waiting for robot_state_publisher...')
        
        self.request_robot_description()

    def request_robot_description(self):
        request = GetParameters.Request()
        request.names = ['robot_description']
        future = self.param_client.call_async(request)
        future.add_done_callback(self.on_robot_description)

    def joint_state_cb(self, msg: JointState):
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        if self.left_kdl_joint_names:
            self.left_current_joints = [name_to_pos.get(name, 0.0) for name in self.left_kdl_joint_names]
        if self.right_kdl_joint_names:
            self.right_current_joints = [name_to_pos.get(name, 0.0) for name in self.right_kdl_joint_names]
        if self.neck_kdl_joint_names:
            self.neck_current_joints = [name_to_pos.get(name, 0.0) for name in self.neck_kdl_joint_names]

    def extract_joint_names_from_chain(self, chain):
        joint_names = []
        for i in range(chain.getNrOfSegments()):
            seg = chain.getSegment(i)
            joint = seg.getJoint()
            if joint.getTypeName() != 'None':
                joint_names.append(joint.getName())
        return joint_names

    def on_robot_description(self, future):
        try:
            result = future.result()
            urdf_string = result.values[0].string_value
            (ok, kdl_tree) = treeFromString(urdf_string)
            if not ok:
                raise ValueError("Failed to parse URDF")

            # Setup LEFT arm
            self.left_chain = kdl_tree.getChain(self.left_robot_base, self.left_robot_ee)
            self.left_num_joints = self.left_chain.getNrOfJoints()
            self.left_kdl_joint_names = self.extract_joint_names_from_chain(self.left_chain)[:self.left_num_joints]
            if self.left_num_joints > 0:
                self.left_ik_solver = ChainIkSolverPos_LMA(self.left_chain, eps=self.ik_eps, maxiter=self.ik_maxiter)
                self.left_fk_solver = ChainFkSolverPos_recursive(self.left_chain)
            
            # Setup RIGHT arm
            self.right_chain = kdl_tree.getChain(self.right_robot_base, self.right_robot_ee)
            self.right_num_joints = self.right_chain.getNrOfJoints()
            self.right_kdl_joint_names = self.extract_joint_names_from_chain(self.right_chain)[:self.right_num_joints]
            if self.right_num_joints > 0:
                self.right_ik_solver = ChainIkSolverPos_LMA(self.right_chain, eps=self.ik_eps, maxiter=self.ik_maxiter)
                self.right_fk_solver = ChainFkSolverPos_recursive(self.right_chain)
            
            # Setup NECK
            self.neck_chain = kdl_tree.getChain(self.neck_base, self.neck_ee)
            self.neck_num_joints = self.neck_chain.getNrOfJoints()
            self.neck_kdl_joint_names = self.extract_joint_names_from_chain(self.neck_chain)[:self.neck_num_joints]
            if self.neck_num_joints > 0:
                self.neck_ik_solver = ChainIkSolverPos_LMA(self.neck_chain, eps=self.ik_eps, maxiter=self.ik_maxiter)
                self.neck_fk_solver = ChainFkSolverPos_recursive(self.neck_chain)
            else:
                self.get_logger().info("Neck has no controllable joints, skipping IK control.")

            left_reach = self.compute_max_reach(self.left_chain)
            right_reach = self.compute_max_reach(self.right_chain)
            neck_reach = self.compute_max_reach(self.neck_chain)
            
            self.get_logger().info(f"LEFT ARM:  {self.left_num_joints} joints, reach: {left_reach:.3f}m")
            self.get_logger().info(f"           Joints: {self.left_kdl_joint_names}")
            self.get_logger().info(f"RIGHT ARM: {self.right_num_joints} joints, reach: {right_reach:.3f}m")
            self.get_logger().info(f"           Joints: {self.right_kdl_joint_names}")
            self.get_logger().info(f"NECK:      {self.neck_num_joints} joints, reach: {neck_reach:.3f}m")
            self.get_logger().info(f"           Joints: {self.neck_kdl_joint_names}")
            
            self.setup_complete = True
            self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
            self.stats_timer = self.create_timer(5.0, self.print_stats)
            self.get_logger().info("✓ Setup complete! Starting teleoperation control...")

        except Exception as e:
            self.get_logger().error(f"Setup failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def compute_max_reach(self, chain):
        reach = 0.0
        for i in range(chain.getNrOfSegments()):
            seg = chain.getSegment(i)
            try:
                v = seg.getFrameToTip().p
                if v.Norm() > 0.001:
                    reach += v.Norm()
            except Exception:
                pass
        return reach

    def clamp_joints(self, joints):
        """Clamp joints to reasonable limits"""
        clamped = []
        for j in joints:
            clamped.append(max(min(j, math.pi), -math.pi))
        return clamped

    def smooth_joints(self, new_joints, last_joints, arm_name):
        """Apply EMA smoothing"""
        if not self.enable_smoothing:
            return new_joints
        if last_joints is None:
            setattr(self, f"{arm_name}_last_joints", new_joints)
            return new_joints
        smoothed = []
        for new, last in zip(new_joints, last_joints):
            smoothed.append(self.smoothing_alpha * new + (1 - self.smoothing_alpha) * last)
        setattr(self, f"{arm_name}_last_joints", smoothed)
        return smoothed

    def solve_arm_ik(self, target_frame, base_frame, chain, ik_solver, fk_solver, num_joints, current_joints, arm_name):
        """Solve IK with full pose and retries"""
        if num_joints == 0:
            return None, "No joints to control"
        best_solution = None
        best_error = float('inf')
        best_status = -1
        projected = False
        try:
            # Lookup transform directly from base to target
            t_base_target = self.tf_buffer.lookup_transform(base_frame, target_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            trans = t_base_target.transform.translation
            rot = t_base_target.transform.rotation
            
            current_pos = [trans.x, trans.y, trans.z]
            current_rot = Rotation.Quaternion(rot.x, rot.y, rot.z, rot.w)
            
            # Relative scaling setup and computation
            initial_pos_attr = f"{arm_name}_initial_pos"
            initial_rot_attr = f"{arm_name}_initial_rot"
            home_pos_attr = f"{arm_name}_home_pos"
            home_rot_attr = f"{arm_name}_home_rot"
            
            if self.use_relative_scaling and getattr(self, initial_pos_attr) is None:
                # Set initial on first call
                setattr(self, initial_pos_attr, current_pos)
                setattr(self, initial_rot_attr, current_rot)
                
                # Set home from current or zero joints
                home_j = JntArray(num_joints)
                if current_joints and len(current_joints) == num_joints:
                    for i, p in enumerate(current_joints):
                        home_j[i] = p
                # else zero already
                
                home_frame = Frame()
                fk_status = fk_solver.JntToCart(home_j, home_frame)
                if fk_status >= 0:
                    home_pos = [home_frame.p.x(), home_frame.p.y(), home_frame.p.z()]
                    home_rot = home_frame.M
                else:
                    home_pos = [0.0, 0.0, 0.0]
                    home_rot = Rotation.Identity()
                
                setattr(self, home_pos_attr, home_pos)
                setattr(self, home_rot_attr, home_rot)
                self.get_logger().info(f"{arm_name.upper()} relative scaling initialized (home at current/zero pose).")
            
            if self.use_relative_scaling and getattr(self, initial_pos_attr) is not None:
                initial_pos = getattr(self, initial_pos_attr)
                initial_rot = getattr(self, initial_rot_attr)
                home_pos = getattr(self, home_pos_attr)
                home_rot = getattr(self, home_rot_attr)
                
                rel_pos = np.array(current_pos) - np.array(initial_pos)
                scaled_rel_pos = rel_pos * self.motion_scale
                target_pos = np.array(home_pos) + scaled_rel_pos
                target_pos = target_pos.tolist()
                
                initial_inv = initial_rot.Inverse()
                rel_rot = initial_inv * current_rot
                target_rot = home_rot * rel_rot
            else:
                target_pos = current_pos
                target_rot = current_rot
            
            original_dist = np.linalg.norm(target_pos)
            max_reach = self.compute_max_reach(chain)
            if original_dist > max_reach:
                if self.project_out_of_reach:
                    scale = (max_reach * self.projection_factor) / original_dist
                    target_pos = [p * scale for p in target_pos]
                    projected_dist = np.linalg.norm(target_pos)
                    projected = True
                    if self.log_projected or (arm_name == 'left' and self.left_failure % 50 == 0) or (arm_name == 'right' and self.right_failure % 50 == 0):
                        self.get_logger().info(f"{arm_name.upper()} target projected from {original_dist:.2f}m to {projected_dist:.2f}m")
                else:
                    return None, f"Out of reach: {original_dist:.2f}m > {max_reach:.2f}m"
            
            target_dist = np.linalg.norm(target_pos)
            
            # Check with tolerance after projection
            if target_dist > max_reach * self.reach_tolerance:
                return None, f"Out of reach: {target_dist:.2f}m > {max_reach * self.reach_tolerance:.2f}m"
            
            # Log target position occasionally on failure buildup
            if (arm_name == 'left' and self.left_failure % 100 == 0) or (arm_name == 'right' and self.right_failure % 100 == 0):
                self.get_logger().info(f"{arm_name.upper()} target pos: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] (projected: {projected})")
            
            # Create target frame
            target_frame_kdl = Frame(target_rot, Vector(*target_pos))
            
            # [OLD DEPRECATED] Experiencing Null Space Freedome Initialize seed
            seed = current_joints if current_joints and len(current_joints) == num_joints else [0.0] * num_joints

            # Solve IK with retries
            for attempt in range(self.ik_max_attempts):
                initial_joints = JntArray(num_joints)
                for i in range(num_joints):
                    if attempt > 0:
                        initial_joints[i] = seed[i] + random.uniform(-self.ik_perturbation_range, self.ik_perturbation_range)
                    else:
                        initial_joints[i] = seed[i]
                
                result_joints = JntArray(num_joints)
                status = ik_solver.CartToJnt(initial_joints, target_frame_kdl, result_joints)
                
                # Compute FK error
                achieved = Frame()
                fk_status = fk_solver.JntToCart(result_joints, achieved)
                current_error = float('inf')
                if fk_status >= 0:
                    diff = achieved * target_frame_kdl.Inverse()
                    current_error = diff.p.Norm()  # position error in m
                
                # Track best
                if current_error < best_error:
                    best_error = current_error
                    best_solution = [result_joints[i] for i in range(num_joints)]
                    best_status = status
                
                if status >= 0:
                    solution = [result_joints[i] for i in range(num_joints)]
                    clamped_solution = self.clamp_joints(solution)
                    return clamped_solution, None
            
            # After retries, check if best is acceptable
            error_msg = f"IK failed after retries (best status: {best_status}, error: {best_error:.4f}m)"
            if self.ik_publish_on_failure and best_error <= self.ik_failure_tol:
                clamped_best = self.clamp_joints(best_solution)
                self.get_logger().info(f"{arm_name.upper()} publishing best approximation (error: {best_error:.4f}m)")
                return clamped_best, error_msg
            else:
                return None, error_msg

        except TransformException as ex:
            return None, f"TF error: {ex}"
        except Exception as ex:
            return None, f"Error: {ex}"

    def publish_trajectory(self, joint_names, joint_positions, publisher):
        if not joint_names:
            return
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(joint_positions)
        point.time_from_start = Duration(sec=int(self.traj_duration), nanosec=int((self.traj_duration % 1) * 1e9))
        trajectory.points.append(point)
        publisher.publish(trajectory)

    def control_loop(self):
        if not self.setup_complete:
            return
        
        # LEFT arm
        if self.left_num_joints > 0:
            left_solution, left_error = self.solve_arm_ik(
                self.left_wrist_frame, self.left_robot_base, self.left_chain,
                self.left_ik_solver, self.left_fk_solver, self.left_num_joints, self.left_current_joints, "left"
            )
            if left_solution:
                smoothed_left = self.smooth_joints(left_solution, self.left_last_joints, "left")
                self.publish_trajectory(self.left_kdl_joint_names, smoothed_left, self.left_joint_pub)
                self.left_success += 1
                self.left_failure = 0  # Reset on publish
            else:
                self.left_failure += 1
                if self.left_failure % 20 == 0:
                    self.get_logger().warn(f"LEFT arm: {left_error}")
        
        # RIGHT arm
        if self.right_num_joints > 0:
            right_solution, right_error = self.solve_arm_ik(
                self.right_wrist_frame, self.right_robot_base, self.right_chain,
                self.right_ik_solver, self.right_fk_solver, self.right_num_joints, self.right_current_joints, "right"
            )
            if right_solution:
                smoothed_right = self.smooth_joints(right_solution, self.right_last_joints, "right")
                self.publish_trajectory(self.right_kdl_joint_names, smoothed_right, self.right_joint_pub)
                self.right_success += 1
                self.right_failure = 0  # Reset on publish
            else:
                self.right_failure += 1
                if self.right_failure % 20 == 0:
                    self.get_logger().warn(f"RIGHT arm: {right_error}")
        
        # NECK
        if self.neck_num_joints > 0:
            neck_solution, neck_error = self.solve_arm_ik(
                self.head_frame, self.neck_base, self.neck_chain,
                self.neck_ik_solver, self.neck_fk_solver, self.neck_num_joints, self.neck_current_joints, "neck"
            )
            if neck_solution:
                smoothed_neck = self.smooth_joints(neck_solution, self.neck_last_joints, "neck")
                self.publish_trajectory(self.neck_kdl_joint_names, smoothed_neck, self.neck_joint_pub)
                self.neck_success += 1
                self.neck_failure = 0  # Reset on publish
            else:
                self.neck_failure += 1
                if self.neck_failure % 20 == 0:
                    self.get_logger().warn(f"NECK: {neck_error}")

    def print_stats(self):
        left_total = self.left_success + self.left_failure
        right_total = self.right_success + self.right_failure
        neck_total = self.neck_success + self.neck_failure
        
        if left_total > 0:
            left_rate = (self.left_success / left_total) * 100
            self.get_logger().info(f"LEFT:  {self.left_success}/{left_total} ({left_rate:.1f}%)")
        if right_total > 0:
            right_rate = (self.right_success / right_total) * 100
            self.get_logger().info(f"RIGHT: {self.right_success}/{right_total} ({right_rate:.1f}%)")
        if neck_total > 0:
            neck_rate = (self.neck_success / neck_total) * 100
            self.get_logger().info(f"NECK:  {self.neck_success}/{neck_total} ({neck_rate:.1f}%)")


def main(args=None):
    rclpy.init(args=args)
    node = TeleopController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down teleoperation controller...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()