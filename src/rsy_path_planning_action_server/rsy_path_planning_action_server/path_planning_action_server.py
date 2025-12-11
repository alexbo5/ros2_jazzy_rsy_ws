#!/usr/bin/env python3
"""
Path Planning Action Server for RSY dual robot setup.
Provides MoveJ (joint space) and MoveL (linear/Cartesian) motion actions.
Uses 4x4 transformation matrices for pose specification.

This server connects to the existing move_group node via the MoveGroup action interface.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    WorkspaceParameters,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from rsy_path_planning_action_server.action import MoveJ, MoveL

import numpy as np
import threading


class PathPlanningActionServer(Node):
    """Action server providing MoveJ and MoveL motion commands for dual UR robots."""

    def __init__(self):
        super().__init__('path_planning_action_server')

        self.get_logger().info('Initializing Path Planning Action Server...')

        # Robot configuration
        self.robot_configs = {
            'robot1': {
                'planning_group': 'robot1_manipulator',
                'base_frame': 'robot1_base_link',
                'ee_frame': 'robot1_tool0',
                'controller': 'robot1_joint_trajectory_controller',
            },
            'robot2': {
                'planning_group': 'robot2_manipulator',
                'base_frame': 'robot2_base_link',
                'ee_frame': 'robot2_tool0',
                'controller': 'robot2_joint_trajectory_controller',
            }
        }

        # Callback group for concurrent handling
        self.callback_group = ReentrantCallbackGroup()

        # Create MoveGroup action client to connect to existing move_group node
        self.get_logger().info('Connecting to move_group action server...')
        self._move_group_client = ActionClient(
            self,
            MoveGroup,
            'move_action',
            callback_group=self.callback_group
        )

        # Wait for move_group to be available
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('move_group action server not available!')
            raise RuntimeError('move_group action server not available')
        self.get_logger().info('Connected to move_group action server')

        # Current joint states (updated via subscription)
        self.current_joint_states = {}
        self.joint_state_lock = threading.Lock()
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )

        # Create action servers
        self._movej_action_server = ActionServer(
            self,
            MoveJ,
            'move_j',
            execute_callback=self.execute_movej_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        self._movel_action_server = ActionServer(
            self,
            MoveL,
            'move_l',
            execute_callback=self.execute_movel_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info('Path Planning Action Server ready!')
        self.get_logger().info('Available actions: /move_j (PTP), /move_l (LIN)')
        self.get_logger().info('Available robots: robot1, robot2')
        self.get_logger().info('Poses specified as 4x4 transformation matrices relative to frame_id')

    def joint_state_callback(self, msg: JointState):
        """Store current joint states for feedback."""
        with self.joint_state_lock:
            for i, name in enumerate(msg.name):
                self.current_joint_states[name] = msg.position[i]

    def goal_callback(self, goal_request):
        """Accept or reject incoming goal requests."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept cancel requests."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def get_robot_joint_positions(self, robot_name: str) -> list:
        """Get current joint positions for a specific robot."""
        prefix = f'{robot_name}_'
        joint_names = [
            f'{prefix}shoulder_pan_joint',
            f'{prefix}shoulder_lift_joint',
            f'{prefix}elbow_joint',
            f'{prefix}wrist_1_joint',
            f'{prefix}wrist_2_joint',
            f'{prefix}wrist_3_joint',
        ]
        with self.joint_state_lock:
            return [self.current_joint_states.get(name, 0.0) for name in joint_names]

    def get_robot_joint_names(self, robot_name: str) -> list:
        """Get joint names for a specific robot."""
        return [
            f'{robot_name}_shoulder_pan_joint',
            f'{robot_name}_shoulder_lift_joint',
            f'{robot_name}_elbow_joint',
            f'{robot_name}_wrist_1_joint',
            f'{robot_name}_wrist_2_joint',
            f'{robot_name}_wrist_3_joint',
        ]

    def transformation_matrix_to_pose_stamped(self, matrix_flat: list, frame_id: str) -> PoseStamped:
        """
        Convert a flat 4x4 transformation matrix (row-major) to a PoseStamped message.

        Matrix format (row-major, 16 elements):
        [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, 0, 0, 0, 1]

        Args:
            matrix_flat: List of 16 floats representing 4x4 transformation matrix
            frame_id: Reference frame for the pose

        Returns:
            PoseStamped message with position and orientation
        """
        # Reshape to 4x4 matrix
        matrix = np.array(matrix_flat).reshape(4, 4)

        # Extract translation
        tx, ty, tz = matrix[0, 3], matrix[1, 3], matrix[2, 3]

        # Extract rotation matrix
        rotation = matrix[:3, :3]

        # Convert rotation matrix to quaternion
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation)

        # Create PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        pose_stamped.header.stamp = self.get_clock().now().to_msg()

        pose_stamped.pose.position.x = float(tx)
        pose_stamped.pose.position.y = float(ty)
        pose_stamped.pose.position.z = float(tz)

        pose_stamped.pose.orientation.x = float(qx)
        pose_stamped.pose.orientation.y = float(qy)
        pose_stamped.pose.orientation.z = float(qz)
        pose_stamped.pose.orientation.w = float(qw)

        return pose_stamped

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> tuple:
        """
        Convert a 3x3 rotation matrix to quaternion (w, x, y, z).

        Args:
            R: 3x3 rotation matrix

        Returns:
            Tuple (w, x, y, z) quaternion
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        return (w/norm, x/norm, y/norm, z/norm)

    def pose_to_transformation_matrix(self, pose_stamped: PoseStamped) -> list:
        """
        Convert a PoseStamped to a flat 4x4 transformation matrix.

        Returns:
            List of 16 floats (row-major 4x4 matrix)
        """
        pos = pose_stamped.pose.position
        orient = pose_stamped.pose.orientation

        # Quaternion to rotation matrix
        qw, qx, qy, qz = orient.w, orient.x, orient.y, orient.z

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])

        # Build 4x4 matrix
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[0, 3] = pos.x
        matrix[1, 3] = pos.y
        matrix[2, 3] = pos.z

        return matrix.flatten().tolist()

    def get_identity_matrix(self) -> list:
        """Return identity transformation matrix as flat list."""
        return [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]

    def validate_transformation_matrix(self, matrix_flat: list) -> tuple:
        """
        Validate a transformation matrix.

        Returns:
            Tuple (is_valid, error_message)
        """
        if len(matrix_flat) != 16:
            return False, f'Expected 16 elements, got {len(matrix_flat)}'

        matrix = np.array(matrix_flat).reshape(4, 4)

        # Check bottom row
        if not np.allclose(matrix[3, :], [0, 0, 0, 1], atol=1e-6):
            return False, 'Bottom row must be [0, 0, 0, 1]'

        # Check rotation matrix is valid (orthonormal)
        R = matrix[:3, :3]
        if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-3):
            return False, 'Rotation matrix is not orthonormal'

        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
            return False, 'Rotation matrix determinant is not 1 (not a proper rotation)'

        return True, ''

    def create_move_group_goal(self, robot_name: str, target_pose: PoseStamped,
                                velocity_scaling: float, acceleration_scaling: float,
                                planner_id: str = 'PTP',
                                pipeline_id: str = 'pilz_industrial_motion_planner') -> MoveGroup.Goal:
        """Create a MoveGroup action goal for the specified robot and target pose."""
        config = self.robot_configs[robot_name]

        goal = MoveGroup.Goal()

        # Motion plan request
        goal.request.group_name = config['planning_group']
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = velocity_scaling
        goal.request.max_acceleration_scaling_factor = acceleration_scaling
        goal.request.pipeline_id = pipeline_id
        goal.request.planner_id = planner_id

        # Leave start_state empty to use current state from planning scene
        # goal.request.start_state is left as default (empty = use current)
        goal.request.start_state.is_diff = True  # Important: indicates to use current state

        # Set workspace parameters
        goal.request.workspace_parameters.header.frame_id = 'world'
        goal.request.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        goal.request.workspace_parameters.min_corner.x = -3.0
        goal.request.workspace_parameters.min_corner.y = -3.0
        goal.request.workspace_parameters.min_corner.z = -1.0
        goal.request.workspace_parameters.max_corner.x = 3.0
        goal.request.workspace_parameters.max_corner.y = 3.0
        goal.request.workspace_parameters.max_corner.z = 3.0

        # Create goal constraints from target pose
        constraints = Constraints()

        # Position constraint
        position_constraint = PositionConstraint()
        position_constraint.header = target_pose.header
        position_constraint.link_name = config['ee_frame']
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        # Bounding volume (small sphere around target)
        bounding_volume = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.01]  # 10mm tolerance
        bounding_volume.primitives.append(primitive)
        bounding_volume.primitive_poses.append(target_pose.pose)
        position_constraint.constraint_region = bounding_volume
        position_constraint.weight = 1.0

        constraints.position_constraints.append(position_constraint)

        # Orientation constraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = target_pose.header
        orientation_constraint.link_name = config['ee_frame']
        orientation_constraint.orientation = target_pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

        constraints.orientation_constraints.append(orientation_constraint)

        goal.request.goal_constraints.append(constraints)

        # Planning options
        goal.planning_options.plan_only = False  # Plan and execute
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5
        goal.planning_options.replan_delay = 0.5

        return goal

    async def execute_movej_callback(self, goal_handle):
        """Execute MoveJ action - joint space (PTP) motion to Cartesian pose."""
        self.get_logger().info('Executing MoveJ goal (PTP motion)...')

        request = goal_handle.request
        robot_name = request.robot_name

        # Validate robot name
        if robot_name not in self.robot_configs:
            self.get_logger().error(f'Unknown robot: {robot_name}')
            goal_handle.abort()
            result = MoveJ.Result()
            result.success = False
            result.message = f'Unknown robot: {robot_name}. Available: robot1, robot2'
            return result

        # Validate transformation matrix
        is_valid, error_msg = self.validate_transformation_matrix(list(request.transformation_matrix))
        if not is_valid:
            self.get_logger().error(f'Invalid transformation matrix: {error_msg}')
            goal_handle.abort()
            result = MoveJ.Result()
            result.success = False
            result.message = f'Invalid transformation matrix: {error_msg}'
            return result

        config = self.robot_configs[robot_name]

        # Set velocity and acceleration scaling
        velocity_scaling = request.velocity_scaling if request.velocity_scaling > 0 else 0.1
        acceleration_scaling = request.acceleration_scaling if request.acceleration_scaling > 0 else 0.1

        # Get frame_id (default to 'world')
        frame_id = request.frame_id if request.frame_id else 'world'

        # Convert transformation matrix to PoseStamped
        target_pose = self.transformation_matrix_to_pose_stamped(
            list(request.transformation_matrix), frame_id
        )

        self.get_logger().info(
            f'Planning MoveJ (PTP) for {robot_name}:\n'
            f'  Position: ({target_pose.pose.position.x:.4f}, {target_pose.pose.position.y:.4f}, {target_pose.pose.position.z:.4f})\n'
            f'  Orientation (xyzw): ({target_pose.pose.orientation.x:.4f}, {target_pose.pose.orientation.y:.4f}, '
            f'{target_pose.pose.orientation.z:.4f}, {target_pose.pose.orientation.w:.4f})\n'
            f'  Frame: {frame_id}\n'
            f'  Velocity scaling: {velocity_scaling}, Acceleration scaling: {acceleration_scaling}'
        )

        # Send initial feedback
        feedback_msg = MoveJ.Feedback()
        feedback_msg.progress = 0.0
        feedback_msg.current_transformation = self.get_identity_matrix()
        goal_handle.publish_feedback(feedback_msg)

        # Create MoveGroup goal using Pilz PTP planner
        move_group_goal = self.create_move_group_goal(
            robot_name, target_pose, velocity_scaling, acceleration_scaling,
            planner_id='PTP', pipeline_id='pilz_industrial_motion_planner'
        )

        self.get_logger().info('Sending goal to move_group (Pilz PTP)...')

        # Send goal to move_group
        send_goal_future = self._move_group_client.send_goal_async(move_group_goal)
        goal_response = await send_goal_future

        if not goal_response.accepted:
            self.get_logger().warn('Pilz PTP rejected, trying OMPL...')
            # Fallback to OMPL
            move_group_goal = self.create_move_group_goal(
                robot_name, target_pose, velocity_scaling, acceleration_scaling,
                planner_id='RRTConnect', pipeline_id='ompl'
            )
            send_goal_future = self._move_group_client.send_goal_async(move_group_goal)
            goal_response = await send_goal_future

            if not goal_response.accepted:
                self.get_logger().error('Goal rejected by move_group')
                goal_handle.abort()
                result = MoveJ.Result()
                result.success = False
                result.message = 'Motion goal rejected by move_group'
                return result

        self.get_logger().info('Goal accepted, waiting for result...')
        feedback_msg.progress = 25.0
        goal_handle.publish_feedback(feedback_msg)

        # Wait for result
        result_future = goal_response.get_result_async()
        move_group_result = await result_future

        # Check result
        if move_group_result.result.error_code.val == 1:  # SUCCESS
            feedback_msg.progress = 100.0
            feedback_msg.current_transformation = list(request.transformation_matrix)
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info('MoveJ (PTP) execution completed successfully')
            goal_handle.succeed()
            result = MoveJ.Result()
            result.success = True
            result.message = 'PTP motion completed successfully'
        else:
            feedback_msg.progress = 0.0
            goal_handle.publish_feedback(feedback_msg)

            error_code = move_group_result.result.error_code.val
            self.get_logger().error(f'Motion failed with error code: {error_code}')
            goal_handle.abort()
            result = MoveJ.Result()
            result.success = False
            result.message = f'Motion failed with error code: {error_code}'

        return result

    async def execute_movel_callback(self, goal_handle):
        """Execute MoveL action - linear (Cartesian) motion."""
        self.get_logger().info('Executing MoveL goal (LIN motion)...')

        request = goal_handle.request
        robot_name = request.robot_name

        # Validate robot name
        if robot_name not in self.robot_configs:
            self.get_logger().error(f'Unknown robot: {robot_name}')
            goal_handle.abort()
            result = MoveL.Result()
            result.success = False
            result.message = f'Unknown robot: {robot_name}. Available: robot1, robot2'
            return result

        # Validate transformation matrix
        is_valid, error_msg = self.validate_transformation_matrix(list(request.transformation_matrix))
        if not is_valid:
            self.get_logger().error(f'Invalid transformation matrix: {error_msg}')
            goal_handle.abort()
            result = MoveL.Result()
            result.success = False
            result.message = f'Invalid transformation matrix: {error_msg}'
            return result

        config = self.robot_configs[robot_name]

        # Set velocity and acceleration scaling
        velocity_scaling = request.velocity_scaling if request.velocity_scaling > 0 else 0.1
        acceleration_scaling = request.acceleration_scaling if request.acceleration_scaling > 0 else 0.1

        # Get frame_id (default to 'world')
        frame_id = request.frame_id if request.frame_id else 'world'

        # Convert transformation matrix to PoseStamped
        target_pose = self.transformation_matrix_to_pose_stamped(
            list(request.transformation_matrix), frame_id
        )

        self.get_logger().info(
            f'Planning MoveL (LIN) for {robot_name}:\n'
            f'  Position: ({target_pose.pose.position.x:.4f}, {target_pose.pose.position.y:.4f}, {target_pose.pose.position.z:.4f})\n'
            f'  Orientation (xyzw): ({target_pose.pose.orientation.x:.4f}, {target_pose.pose.orientation.y:.4f}, '
            f'{target_pose.pose.orientation.z:.4f}, {target_pose.pose.orientation.w:.4f})\n'
            f'  Frame: {frame_id}\n'
            f'  Velocity scaling: {velocity_scaling}, Acceleration scaling: {acceleration_scaling}'
        )

        # Send initial feedback
        feedback_msg = MoveL.Feedback()
        feedback_msg.progress = 0.0
        feedback_msg.current_transformation = self.get_identity_matrix()
        goal_handle.publish_feedback(feedback_msg)

        # Create MoveGroup goal using Pilz LIN planner
        move_group_goal = self.create_move_group_goal(
            robot_name, target_pose, velocity_scaling, acceleration_scaling,
            planner_id='LIN', pipeline_id='pilz_industrial_motion_planner'
        )

        self.get_logger().info('Sending goal to move_group (Pilz LIN)...')

        # Send goal to move_group
        send_goal_future = self._move_group_client.send_goal_async(move_group_goal)
        goal_response = await send_goal_future

        if not goal_response.accepted:
            self.get_logger().warn('Pilz LIN rejected, trying OMPL...')
            # Fallback to OMPL
            move_group_goal = self.create_move_group_goal(
                robot_name, target_pose, velocity_scaling, acceleration_scaling,
                planner_id='RRTConnect', pipeline_id='ompl'
            )
            send_goal_future = self._move_group_client.send_goal_async(move_group_goal)
            goal_response = await send_goal_future

            if not goal_response.accepted:
                self.get_logger().error('Goal rejected by move_group')
                goal_handle.abort()
                result = MoveL.Result()
                result.success = False
                result.message = 'Motion goal rejected by move_group'
                return result

        self.get_logger().info('Goal accepted, waiting for result...')
        feedback_msg.progress = 25.0
        goal_handle.publish_feedback(feedback_msg)

        # Wait for result
        result_future = goal_response.get_result_async()
        move_group_result = await result_future

        # Check result
        if move_group_result.result.error_code.val == 1:  # SUCCESS
            feedback_msg.progress = 100.0
            feedback_msg.current_transformation = list(request.transformation_matrix)
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info('MoveL (LIN) execution completed successfully')
            goal_handle.succeed()
            result = MoveL.Result()
            result.success = True
            result.message = 'Linear motion completed successfully'
        else:
            feedback_msg.progress = 0.0
            goal_handle.publish_feedback(feedback_msg)

            error_code = move_group_result.result.error_code.val
            self.get_logger().error(f'Motion failed with error code: {error_code}')
            goal_handle.abort()
            result = MoveL.Result()
            result.success = False
            result.message = f'Motion failed with error code: {error_code}'

        return result


def main(args=None):
    rclpy.init(args=args)

    action_server = PathPlanningActionServer()

    # Use multi-threaded executor for concurrent action handling
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
