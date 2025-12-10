#!/usr/bin/env python3
"""
Path Planning Action Server for RSY dual robot setup.
Provides MoveJ (joint space) and MoveL (linear/Cartesian) motion actions.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

from rsy_path_planning_action_server.action import MoveJ, MoveL

import numpy as np
import time


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
            },
            'robot2': {
                'planning_group': 'robot2_manipulator',
                'base_frame': 'robot2_base_link',
                'ee_frame': 'robot2_tool0',
            }
        }

        # Initialize MoveItPy
        self.get_logger().info('Initializing MoveItPy...')
        self.moveit = MoveItPy(node_name='moveit_py_planning')
        self.get_logger().info('MoveItPy initialized successfully')

        # Get planning components for each robot
        self.planning_components = {}
        for robot_name, config in self.robot_configs.items():
            self.planning_components[robot_name] = self.moveit.get_planning_component(
                config['planning_group']
            )
            self.get_logger().info(f'Planning component initialized for {robot_name}')

        # Callback group for concurrent action handling
        self.callback_group = ReentrantCallbackGroup()

        # Current joint states (updated via subscription)
        self.current_joint_states = {}
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
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
        self.get_logger().info('Available actions: /move_j, /move_l')
        self.get_logger().info('Available robots: robot1, robot2')

    def joint_state_callback(self, msg: JointState):
        """Store current joint states for feedback."""
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
        return [self.current_joint_states.get(name, 0.0) for name in joint_names]

    async def execute_movej_callback(self, goal_handle):
        """Execute MoveJ action - joint space motion."""
        self.get_logger().info('Executing MoveJ goal...')

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

        # Validate joint positions
        if len(request.joint_positions) != 6:
            self.get_logger().error(f'Expected 6 joint positions, got {len(request.joint_positions)}')
            goal_handle.abort()
            result = MoveJ.Result()
            result.success = False
            result.message = f'Expected 6 joint positions, got {len(request.joint_positions)}'
            return result

        # Get planning component
        planning_component = self.planning_components[robot_name]

        # Set velocity and acceleration scaling
        velocity_scaling = request.velocity_scaling if request.velocity_scaling > 0 else 0.1
        acceleration_scaling = request.acceleration_scaling if request.acceleration_scaling > 0 else 0.1

        # Set start state to current
        planning_component.set_start_state_to_current_state()

        # Create target robot state with joint positions
        robot_model = self.moveit.get_robot_model()
        robot_state = RobotState(robot_model)

        # Set joint values
        joint_names = [
            f'{robot_name}_shoulder_pan_joint',
            f'{robot_name}_shoulder_lift_joint',
            f'{robot_name}_elbow_joint',
            f'{robot_name}_wrist_1_joint',
            f'{robot_name}_wrist_2_joint',
            f'{robot_name}_wrist_3_joint',
        ]

        for i, name in enumerate(joint_names):
            robot_state.set_joint_positions(name, [request.joint_positions[i]])

        planning_component.set_goal_state(robot_state=robot_state)

        self.get_logger().info(f'Planning MoveJ for {robot_name} to joints: {list(request.joint_positions)}')

        # Plan
        plan_result = planning_component.plan()

        if not plan_result:
            self.get_logger().error('Planning failed')
            goal_handle.abort()
            result = MoveJ.Result()
            result.success = False
            result.message = 'Motion planning failed'
            return result

        self.get_logger().info('Plan successful, executing...')

        # Send feedback during execution
        feedback_msg = MoveJ.Feedback()
        feedback_msg.progress = 0.0
        feedback_msg.current_joint_positions = self.get_robot_joint_positions(robot_name)
        goal_handle.publish_feedback(feedback_msg)

        # Execute the trajectory
        robot = self.moveit.get_planning_component(self.robot_configs[robot_name]['planning_group'])

        # Execute with the trajectory
        trajectory = plan_result.trajectory
        execute_result = self.moveit.execute(
            trajectory,
            controllers=[]
        )

        # Final feedback
        feedback_msg.progress = 100.0
        feedback_msg.current_joint_positions = self.get_robot_joint_positions(robot_name)
        goal_handle.publish_feedback(feedback_msg)

        if execute_result:
            self.get_logger().info('MoveJ execution completed successfully')
            goal_handle.succeed()
            result = MoveJ.Result()
            result.success = True
            result.message = 'Motion completed successfully'
        else:
            self.get_logger().error('Execution failed')
            goal_handle.abort()
            result = MoveJ.Result()
            result.success = False
            result.message = 'Motion execution failed'

        return result

    async def execute_movel_callback(self, goal_handle):
        """Execute MoveL action - linear (Cartesian) motion."""
        self.get_logger().info('Executing MoveL goal...')

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

        # Get planning component
        planning_component = self.planning_components[robot_name]
        config = self.robot_configs[robot_name]

        # Set velocity and acceleration scaling
        velocity_scaling = request.velocity_scaling if request.velocity_scaling > 0 else 0.1
        acceleration_scaling = request.acceleration_scaling if request.acceleration_scaling > 0 else 0.1

        # Set start state to current
        planning_component.set_start_state_to_current_state()

        # Create target pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = request.frame_id if request.frame_id else 'world'
        target_pose.pose = request.target_pose

        self.get_logger().info(
            f'Planning MoveL for {robot_name} to pose: '
            f'pos=({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f}), '
            f'frame={target_pose.header.frame_id}'
        )

        # Set goal state from pose
        planning_component.set_goal_state(
            pose_stamped_msg=target_pose,
            pose_link=config['ee_frame']
        )

        # Plan using Pilz LIN planner for linear motion
        plan_result = planning_component.plan(
            planner_id='LIN',
            planning_pipeline='pilz_industrial_motion_planner'
        )

        if not plan_result:
            self.get_logger().warn('Pilz LIN planning failed, trying OMPL...')
            # Fallback to OMPL if Pilz fails
            plan_result = planning_component.plan()

        if not plan_result:
            self.get_logger().error('Planning failed')
            goal_handle.abort()
            result = MoveL.Result()
            result.success = False
            result.message = 'Motion planning failed'
            return result

        self.get_logger().info('Plan successful, executing...')

        # Send feedback
        feedback_msg = MoveL.Feedback()
        feedback_msg.progress = 0.0
        feedback_msg.current_pose = Pose()
        goal_handle.publish_feedback(feedback_msg)

        # Execute the trajectory
        trajectory = plan_result.trajectory
        execute_result = self.moveit.execute(
            trajectory,
            controllers=[]
        )

        # Final feedback
        feedback_msg.progress = 100.0
        goal_handle.publish_feedback(feedback_msg)

        if execute_result:
            self.get_logger().info('MoveL execution completed successfully')
            goal_handle.succeed()
            result = MoveL.Result()
            result.success = True
            result.message = 'Linear motion completed successfully'
        else:
            self.get_logger().error('Execution failed')
            goal_handle.abort()
            result = MoveL.Result()
            result.success = False
            result.message = 'Motion execution failed'

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
