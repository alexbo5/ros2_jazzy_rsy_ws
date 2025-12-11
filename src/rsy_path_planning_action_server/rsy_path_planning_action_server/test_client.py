#!/usr/bin/env python3
"""
Test client for the path planning action server.
Supports transformation matrices for pose specification.

Usage examples:
    # MoveJ (PTP) with transformation matrix - position only (identity rotation)
    ros2 run rsy_path_planning_action_server test_client.py --movej robot1 --pos 0.3 0.0 0.5 --frame world

    # MoveJ with rotation (90 degrees around Z-axis)
    ros2 run rsy_path_planning_action_server test_client.py --movej robot1 --pos 0.3 0.0 0.5 --rpy 0 0 1.57 --frame world

    # MoveL (LIN) with transformation matrix
    ros2 run rsy_path_planning_action_server test_client.py --movel robot2 --pos 0.4 0.1 0.3 --frame world

    # With full transformation matrix (16 values, row-major)
    ros2 run rsy_path_planning_action_server test_client.py --movej robot1 --matrix 1 0 0 0.3 0 1 0 0 0 0 1 0.5 0 0 0 1
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from rsy_path_planning_action_server.action import MoveJ, MoveL

import argparse
import numpy as np


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll-pitch-yaw angles to a 3x3 rotation matrix.
    Angles are in radians. Convention: Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R


def create_transformation_matrix(x: float, y: float, z: float,
                                  roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> list:
    """
    Create a 4x4 transformation matrix from position and RPY angles.

    Returns:
        List of 16 floats (row-major 4x4 matrix)
    """
    R = rpy_to_rotation_matrix(roll, pitch, yaw)
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix.flatten().tolist()


def format_matrix(matrix_flat: list) -> str:
    """Format a flat 4x4 matrix for display."""
    matrix = np.array(matrix_flat).reshape(4, 4)
    lines = []
    for row in matrix:
        lines.append('  [' + ', '.join(f'{v:8.4f}' for v in row) + ']')
    return '\n'.join(lines)


class TestClient(Node):
    def __init__(self):
        super().__init__('path_planning_test_client')
        self.movej_client = ActionClient(self, MoveJ, 'move_j')
        self.movel_client = ActionClient(self, MoveL, 'move_l')

    def send_movej_goal(self, robot_name: str, transformation_matrix: list,
                        frame_id: str = 'world',
                        velocity_scaling: float = 0.1, acceleration_scaling: float = 0.1):
        """Send a MoveJ (PTP) goal."""
        self.get_logger().info('Waiting for MoveJ action server...')
        self.movej_client.wait_for_server()

        goal_msg = MoveJ.Goal()
        goal_msg.robot_name = robot_name
        goal_msg.transformation_matrix = transformation_matrix
        goal_msg.frame_id = frame_id
        goal_msg.velocity_scaling = velocity_scaling
        goal_msg.acceleration_scaling = acceleration_scaling

        self.get_logger().info(
            f'Sending MoveJ (PTP) goal:\n'
            f'  Robot: {robot_name}\n'
            f'  Frame: {frame_id}\n'
            f'  Transformation matrix:\n{format_matrix(transformation_matrix)}'
        )

        future = self.movej_client.send_goal_async(
            goal_msg,
            feedback_callback=self.movej_feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)
        return future

    def send_movel_goal(self, robot_name: str, transformation_matrix: list,
                        frame_id: str = 'world',
                        velocity_scaling: float = 0.1, acceleration_scaling: float = 0.1):
        """Send a MoveL (LIN) goal."""
        self.get_logger().info('Waiting for MoveL action server...')
        self.movel_client.wait_for_server()

        goal_msg = MoveL.Goal()
        goal_msg.robot_name = robot_name
        goal_msg.transformation_matrix = transformation_matrix
        goal_msg.frame_id = frame_id
        goal_msg.velocity_scaling = velocity_scaling
        goal_msg.acceleration_scaling = acceleration_scaling

        self.get_logger().info(
            f'Sending MoveL (LIN) goal:\n'
            f'  Robot: {robot_name}\n'
            f'  Frame: {frame_id}\n'
            f'  Transformation matrix:\n{format_matrix(transformation_matrix)}'
        )

        future = self.movel_client.send_goal_async(
            goal_msg,
            feedback_callback=self.movel_feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)
        return future

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info(f'Success: {result.message}')
        else:
            self.get_logger().error(f'Failed: {result.message}')
        rclpy.shutdown()

    def movej_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'MoveJ Progress: {feedback.progress:.1f}%')

    def movel_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'MoveL Progress: {feedback.progress:.1f}%')


def main():
    parser = argparse.ArgumentParser(
        description='Test client for path planning action server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MoveJ (PTP) to position with default orientation
  ros2 run rsy_path_planning_action_server test_client.py --movej robot1 --pos 0.3 0.0 0.5

  # MoveL (LIN) to position with rotation
  ros2 run rsy_path_planning_action_server test_client.py --movel robot1 --pos 0.3 0.0 0.5 --rpy 0 0 1.57

  # Using full 4x4 matrix (row-major, 16 values)
  ros2 run rsy_path_planning_action_server test_client.py --movej robot1 --matrix 1 0 0 0.3 0 1 0 0 0 0 1 0.5 0 0 0 1

  # Specify reference frame
  ros2 run rsy_path_planning_action_server test_client.py --movej robot1 --pos 0.3 0.0 0.5 --frame world
        """
    )

    # Motion type
    motion_group = parser.add_mutually_exclusive_group(required=True)
    motion_group.add_argument('--movej', metavar='ROBOT',
                              help='Send MoveJ (PTP) command to specified robot (robot1 or robot2)')
    motion_group.add_argument('--movel', metavar='ROBOT',
                              help='Send MoveL (LIN) command to specified robot (robot1 or robot2)')

    # Pose specification
    pose_group = parser.add_mutually_exclusive_group(required=True)
    pose_group.add_argument('--pos', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                            help='Position (x, y, z) in meters')
    pose_group.add_argument('--matrix', nargs=16, type=float,
                            metavar=('M00', 'M01', 'M02', 'M03', 'M10', 'M11', 'M12', 'M13',
                                    'M20', 'M21', 'M22', 'M23', 'M30', 'M31', 'M32', 'M33'),
                            help='Full 4x4 transformation matrix (16 values, row-major)')

    # Orientation (only with --pos)
    parser.add_argument('--rpy', nargs=3, type=float, metavar=('R', 'P', 'Y'),
                        default=[0.0, 0.0, 0.0],
                        help='Orientation as roll, pitch, yaw in radians (default: 0 0 0)')

    # Reference frame
    parser.add_argument('--frame', type=str, default='world',
                        help='Reference frame for the transformation (default: world)')

    # Motion parameters
    parser.add_argument('--velocity', type=float, default=0.1,
                        help='Velocity scaling (0.0-1.0, default: 0.1)')
    parser.add_argument('--acceleration', type=float, default=0.1,
                        help='Acceleration scaling (0.0-1.0, default: 0.1)')

    args = parser.parse_args()

    # Build transformation matrix
    if args.matrix:
        transformation_matrix = args.matrix
    else:
        x, y, z = args.pos
        roll, pitch, yaw = args.rpy
        transformation_matrix = create_transformation_matrix(x, y, z, roll, pitch, yaw)

    # Get robot name
    robot_name = args.movej if args.movej else args.movel

    rclpy.init()
    client = TestClient()

    if args.movej:
        client.send_movej_goal(
            robot_name,
            transformation_matrix,
            frame_id=args.frame,
            velocity_scaling=args.velocity,
            acceleration_scaling=args.acceleration
        )
    else:
        client.send_movel_goal(
            robot_name,
            transformation_matrix,
            frame_id=args.frame,
            velocity_scaling=args.velocity,
            acceleration_scaling=args.acceleration
        )

    rclpy.spin(client)


if __name__ == '__main__':
    main()
