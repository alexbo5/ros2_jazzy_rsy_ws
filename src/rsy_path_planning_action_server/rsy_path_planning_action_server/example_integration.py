#!/usr/bin/env python3
"""
Simple integration example for the path planning action server.
Copy this into your own ROS2 package to control the robots.

Dependencies:
    - rsy_path_planning_action_server (for action definitions)
    - numpy (for transformation matrix helpers)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rsy_path_planning_action_server.action import MoveJ, MoveL
import numpy as np


class RobotController(Node):
    """Simple robot controller using the path planning action server."""

    def __init__(self):
        super().__init__('robot_controller')

        # Create action clients for both motion types
        self.movej_client = ActionClient(self, MoveJ, 'move_j')
        self.movel_client = ActionClient(self, MoveL, 'move_l')

        self.get_logger().info('Waiting for action servers...')
        self.movej_client.wait_for_server()
        self.movel_client.wait_for_server()
        self.get_logger().info('Action servers connected!')

    # =========================================================================
    # Helper functions to create transformation matrices
    # =========================================================================

    def create_matrix(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """
        Create a 4x4 transformation matrix from position and RPY angles.

        Args:
            x, y, z: Position in meters
            roll, pitch, yaw: Orientation in radians (default: 0)

        Returns:
            List of 16 floats (row-major 4x4 matrix)
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        return matrix.flatten().tolist()

    def create_matrix_from_array(self, matrix_4x4):
        """
        Convert a 4x4 numpy array to flat list.

        Args:
            matrix_4x4: 4x4 numpy array

        Returns:
            List of 16 floats
        """
        return matrix_4x4.flatten().tolist()

    # =========================================================================
    # Motion commands
    # =========================================================================

    def move_ptp(self, robot: str, matrix: list, frame: str = 'world',
                 velocity: float = 0.1, acceleration: float = 0.1) -> bool:
        """
        Move robot using PTP (point-to-point) motion - fastest path in joint space.

        Args:
            robot: 'robot1' or 'robot2'
            matrix: 4x4 transformation matrix as flat list (16 elements)
            frame: Reference frame (default: 'world')
            velocity: Velocity scaling 0.0-1.0 (default: 0.1)
            acceleration: Acceleration scaling 0.0-1.0 (default: 0.1)

        Returns:
            True if motion succeeded, False otherwise
        """
        goal = MoveJ.Goal()
        goal.robot_name = robot
        goal.transformation_matrix = matrix
        goal.frame_id = frame
        goal.velocity_scaling = velocity
        goal.acceleration_scaling = acceleration

        future = self.movej_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('PTP goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.success:
            self.get_logger().info(f'PTP motion completed: {result.message}')
        else:
            self.get_logger().error(f'PTP motion failed: {result.message}')
        return result.success

    def move_lin(self, robot: str, matrix: list, frame: str = 'world',
                 velocity: float = 0.1, acceleration: float = 0.1) -> bool:
        """
        Move robot using LIN (linear) motion - straight line in Cartesian space.

        Args:
            robot: 'robot1' or 'robot2'
            matrix: 4x4 transformation matrix as flat list (16 elements)
            frame: Reference frame (default: 'world')
            velocity: Velocity scaling 0.0-1.0 (default: 0.1)
            acceleration: Acceleration scaling 0.0-1.0 (default: 0.1)

        Returns:
            True if motion succeeded, False otherwise
        """
        goal = MoveL.Goal()
        goal.robot_name = robot
        goal.transformation_matrix = matrix
        goal.frame_id = frame
        goal.velocity_scaling = velocity
        goal.acceleration_scaling = acceleration

        future = self.movel_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('LIN goal rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.success:
            self.get_logger().info(f'LIN motion completed: {result.message}')
        else:
            self.get_logger().error(f'LIN motion failed: {result.message}')
        return result.success


# =============================================================================
# Example usage
# =============================================================================

def main():
    rclpy.init()
    controller = RobotController()

    # Example 1: Move robot1 to position using PTP (fast joint motion)
    pose1 = controller.create_matrix(x=0.3, y=0.0, z=0.5)
    controller.move_ptp('robot1', pose1, frame='world', velocity=0.2)

    # Example 2: Move robot1 with orientation (90° rotation around Z)
    pose2 = controller.create_matrix(x=0.4, y=0.1, z=0.4, yaw=1.57)
    controller.move_ptp('robot1', pose2, frame='world')

    # Example 3: Linear motion (straight line) for robot2
    pose3 = controller.create_matrix(x=0.3, y=-0.2, z=0.5)
    controller.move_lin('robot2', pose3, frame='world', velocity=0.1)

    # Example 4: Using a numpy matrix directly
    T = np.array([
        [1, 0, 0, 0.35],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.45],
        [0, 0, 0, 1]
    ])
    controller.move_ptp('robot1', controller.create_matrix_from_array(T))

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
