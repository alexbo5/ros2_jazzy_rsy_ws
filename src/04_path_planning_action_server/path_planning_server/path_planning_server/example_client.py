#!/usr/bin/env python3
"""
Example client for the Path Planning Action Server.

Demonstrates how to send MoveL and MoveJ goals to the action server
for both robots with poses relative to the global world frame.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from path_planning_interfaces.action import MoveL, MoveJ

import math
from typing import Optional


class PathPlanningClient(Node):
    """Example client for the path planning action server."""

    def __init__(self):
        super().__init__('path_planning_example_client')

        self.callback_group = ReentrantCallbackGroup()

        # Create action clients for MoveL and MoveJ
        self._movel_client = ActionClient(
            self,
            MoveL,
            'move_l',
            callback_group=self.callback_group
        )

        self._movej_client = ActionClient(
            self,
            MoveJ,
            'move_j',
            callback_group=self.callback_group
        )

        self.get_logger().info('Path Planning Example Client initialized')

    def create_pose_stamped(
        self,
        x: float, y: float, z: float,
        qx: float = 0.0, qy: float = 0.0, qz: float = 0.0, qw: float = 1.0,
        frame_id: str = 'world'
    ) -> PoseStamped:
        """Create a PoseStamped message."""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = Point(x=x, y=y, z=z)
        pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return pose

    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles (radians) to Quaternion."""
        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)
        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)
        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q

    async def send_movel_goal(
        self,
        robot_name: str,
        target_pose: PoseStamped,
        velocity_scaling: float = 0.5,
        acceleration_scaling: float = 0.5,
        execute: bool = True
    ) -> Optional[MoveL.Result]:
        """
        Send a MoveL (linear/Cartesian) goal to the action server.

        Args:
            robot_name: Name of the robot ('robot1' or 'robot2')
            target_pose: Target pose in world frame
            velocity_scaling: Velocity scaling factor (0.0-1.0)
            acceleration_scaling: Acceleration scaling factor (0.0-1.0)
            execute: If True, execute the trajectory; if False, plan only

        Returns:
            Action result or None if failed
        """
        self.get_logger().info(f'Waiting for MoveL action server...')
        if not self._movel_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('MoveL action server not available')
            return None

        goal = MoveL.Goal()
        goal.robot_name = robot_name
        goal.target_pose = target_pose
        goal.velocity_scaling = velocity_scaling
        goal.acceleration_scaling = acceleration_scaling
        goal.execute = execute

        self.get_logger().info(
            f'Sending MoveL goal to {robot_name}: '
            f'pos=({target_pose.pose.position.x:.3f}, '
            f'{target_pose.pose.position.y:.3f}, '
            f'{target_pose.pose.position.z:.3f})'
        )

        send_goal_future = self._movel_client.send_goal_async(
            goal,
            feedback_callback=self._movel_feedback_callback
        )

        goal_handle = await send_goal_future

        if not goal_handle.accepted:
            self.get_logger().error('MoveL goal rejected')
            return None

        self.get_logger().info('MoveL goal accepted, waiting for result...')

        result_future = goal_handle.get_result_async()
        result = await result_future

        return result.result

    async def send_movej_goal(
        self,
        robot_name: str,
        target_pose: PoseStamped,
        velocity_scaling: float = 0.5,
        acceleration_scaling: float = 0.5,
        execute: bool = True
    ) -> Optional[MoveJ.Result]:
        """
        Send a MoveJ (joint space) goal to the action server.

        Args:
            robot_name: Name of the robot ('robot1' or 'robot2')
            target_pose: Target pose in world frame
            velocity_scaling: Velocity scaling factor (0.0-1.0)
            acceleration_scaling: Acceleration scaling factor (0.0-1.0)
            execute: If True, execute the trajectory; if False, plan only

        Returns:
            Action result or None if failed
        """
        self.get_logger().info(f'Waiting for MoveJ action server...')
        if not self._movej_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('MoveJ action server not available')
            return None

        goal = MoveJ.Goal()
        goal.robot_name = robot_name
        goal.target_pose = target_pose
        goal.velocity_scaling = velocity_scaling
        goal.acceleration_scaling = acceleration_scaling
        goal.execute = execute

        self.get_logger().info(
            f'Sending MoveJ goal to {robot_name}: '
            f'pos=({target_pose.pose.position.x:.3f}, '
            f'{target_pose.pose.position.y:.3f}, '
            f'{target_pose.pose.position.z:.3f})'
        )

        send_goal_future = self._movej_client.send_goal_async(
            goal,
            feedback_callback=self._movej_feedback_callback
        )

        goal_handle = await send_goal_future

        if not goal_handle.accepted:
            self.get_logger().error('MoveJ goal rejected')
            return None

        self.get_logger().info('MoveJ goal accepted, waiting for result...')

        result_future = goal_handle.get_result_async()
        result = await result_future

        return result.result

    def _movel_feedback_callback(self, feedback_msg):
        """Handle MoveL feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'MoveL progress: {feedback.progress_percentage:.1f}% - {feedback.status}'
        )

    def _movej_feedback_callback(self, feedback_msg):
        """Handle MoveJ feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'MoveJ progress: {feedback.progress_percentage:.1f}% - {feedback.status}'
        )


async def run_example(client: PathPlanningClient):
    """Run example motions for both robots."""

    # Example poses in world frame
    # These are example positions - adjust based on your robot setup

    # Orientation: pointing down (tool pointing in -Z direction)
    down_orientation = client.euler_to_quaternion(math.pi, 0, 0)

    # =========================================================================
    # Example 1: MoveJ for robot1 - Joint space motion
    # =========================================================================
    client.get_logger().info('=' * 60)
    client.get_logger().info('Example 1: MoveJ for robot1')
    client.get_logger().info('=' * 60)

    target1 = client.create_pose_stamped(
        x=0.3, y=0.2, z=0.3,
        qx=down_orientation.x,
        qy=down_orientation.y,
        qz=down_orientation.z,
        qw=down_orientation.w
    )

    result = await client.send_movej_goal(
        robot_name='robot1',
        target_pose=target1,
        velocity_scaling=0.3,
        acceleration_scaling=0.3,
        execute=False  # Plan only, don't execute
    )

    if result:
        client.get_logger().info(f'MoveJ Result: success={result.success}, message={result.message}')
        client.get_logger().info(f'  Planning time: {result.planning_time:.3f}s')
        client.get_logger().info(f'  Joint distance: {result.joint_distance:.3f} rad')
        client.get_logger().info(f'  Trajectory points: {len(result.planned_trajectory.points)}')

    # =========================================================================
    # Example 2: MoveL for robot1 - Linear/Cartesian motion
    # =========================================================================
    client.get_logger().info('=' * 60)
    client.get_logger().info('Example 2: MoveL for robot1')
    client.get_logger().info('=' * 60)

    target2 = client.create_pose_stamped(
        x=0.3, y=-0.1, z=0.25,
        qx=down_orientation.x,
        qy=down_orientation.y,
        qz=down_orientation.z,
        qw=down_orientation.w
    )

    result = await client.send_movel_goal(
        robot_name='robot1',
        target_pose=target2,
        velocity_scaling=0.2,
        acceleration_scaling=0.2,
        execute=False  # Plan only
    )

    if result:
        client.get_logger().info(f'MoveL Result: success={result.success}, message={result.message}')
        client.get_logger().info(f'  Planning time: {result.planning_time:.3f}s')
        client.get_logger().info(f'  Path length: {result.path_length:.3f} m')
        client.get_logger().info(f'  Trajectory points: {len(result.planned_trajectory.points)}')

    # =========================================================================
    # Example 3: MoveJ for robot2
    # =========================================================================
    client.get_logger().info('=' * 60)
    client.get_logger().info('Example 3: MoveJ for robot2')
    client.get_logger().info('=' * 60)

    target3 = client.create_pose_stamped(
        x=-0.3, y=0.2, z=0.3,
        qx=down_orientation.x,
        qy=down_orientation.y,
        qz=down_orientation.z,
        qw=down_orientation.w
    )

    result = await client.send_movej_goal(
        robot_name='robot2',
        target_pose=target3,
        velocity_scaling=0.3,
        acceleration_scaling=0.3,
        execute=False
    )

    if result:
        client.get_logger().info(f'MoveJ Result: success={result.success}, message={result.message}')
        client.get_logger().info(f'  Planning time: {result.planning_time:.3f}s')
        client.get_logger().info(f'  Joint distance: {result.joint_distance:.3f} rad')

    # =========================================================================
    # Example 4: MoveL for robot2
    # =========================================================================
    client.get_logger().info('=' * 60)
    client.get_logger().info('Example 4: MoveL for robot2')
    client.get_logger().info('=' * 60)

    target4 = client.create_pose_stamped(
        x=-0.3, y=-0.1, z=0.25,
        qx=down_orientation.x,
        qy=down_orientation.y,
        qz=down_orientation.z,
        qw=down_orientation.w
    )

    result = await client.send_movel_goal(
        robot_name='robot2',
        target_pose=target4,
        velocity_scaling=0.2,
        acceleration_scaling=0.2,
        execute=False
    )

    if result:
        client.get_logger().info(f'MoveL Result: success={result.success}, message={result.message}')
        client.get_logger().info(f'  Planning time: {result.planning_time:.3f}s')
        client.get_logger().info(f'  Path length: {result.path_length:.3f} m')

    client.get_logger().info('=' * 60)
    client.get_logger().info('All examples completed!')
    client.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)

    client = PathPlanningClient()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(client)

    try:
        # Run the example using asyncio
        import asyncio

        future = asyncio.ensure_future(run_example(client))

        while not future.done():
            executor.spin_once(timeout_sec=0.1)

    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
