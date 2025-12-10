#!/usr/bin/env python3
"""
Simple test client for the path planning action server.
Usage examples:
    ros2 run rsy_path_planning_action_server test_client.py --movej robot1 0 -1.57 1.57 -1.57 -1.57 0
    ros2 run rsy_path_planning_action_server test_client.py --movel robot1 0.3 0.0 0.4
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from rsy_path_planning_action_server.action import MoveJ, MoveL
from geometry_msgs.msg import Pose

import sys
import argparse


class TestClient(Node):
    def __init__(self):
        super().__init__('path_planning_test_client')
        self.movej_client = ActionClient(self, MoveJ, 'move_j')
        self.movel_client = ActionClient(self, MoveL, 'move_l')

    def send_movej_goal(self, robot_name: str, joint_positions: list,
                        velocity_scaling: float = 0.1, acceleration_scaling: float = 0.1):
        """Send a MoveJ goal."""
        self.get_logger().info(f'Waiting for MoveJ action server...')
        self.movej_client.wait_for_server()

        goal_msg = MoveJ.Goal()
        goal_msg.robot_name = robot_name
        goal_msg.joint_positions = joint_positions
        goal_msg.velocity_scaling = velocity_scaling
        goal_msg.acceleration_scaling = acceleration_scaling

        self.get_logger().info(f'Sending MoveJ goal: {robot_name} -> {joint_positions}')

        future = self.movej_client.send_goal_async(
            goal_msg,
            feedback_callback=self.movej_feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)
        return future

    def send_movel_goal(self, robot_name: str, x: float, y: float, z: float,
                        qx: float = 0.0, qy: float = 0.707, qz: float = 0.0, qw: float = 0.707,
                        frame_id: str = 'world',
                        velocity_scaling: float = 0.1, acceleration_scaling: float = 0.1):
        """Send a MoveL goal."""
        self.get_logger().info(f'Waiting for MoveL action server...')
        self.movel_client.wait_for_server()

        goal_msg = MoveL.Goal()
        goal_msg.robot_name = robot_name
        goal_msg.target_pose = Pose()
        goal_msg.target_pose.position.x = x
        goal_msg.target_pose.position.y = y
        goal_msg.target_pose.position.z = z
        goal_msg.target_pose.orientation.x = qx
        goal_msg.target_pose.orientation.y = qy
        goal_msg.target_pose.orientation.z = qz
        goal_msg.target_pose.orientation.w = qw
        goal_msg.frame_id = frame_id
        goal_msg.velocity_scaling = velocity_scaling
        goal_msg.acceleration_scaling = acceleration_scaling

        self.get_logger().info(f'Sending MoveL goal: {robot_name} -> pos=({x}, {y}, {z})')

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
    parser = argparse.ArgumentParser(description='Test client for path planning action server')
    parser.add_argument('--movej', nargs=7, metavar=('ROBOT', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='Send MoveJ command: robot_name j1 j2 j3 j4 j5 j6')
    parser.add_argument('--movel', nargs=4, metavar=('ROBOT', 'X', 'Y', 'Z'),
                        help='Send MoveL command: robot_name x y z')
    parser.add_argument('--velocity', type=float, default=0.1,
                        help='Velocity scaling (0.0-1.0, default: 0.1)')
    parser.add_argument('--acceleration', type=float, default=0.1,
                        help='Acceleration scaling (0.0-1.0, default: 0.1)')
    parser.add_argument('--frame', type=str, default='world',
                        help='Reference frame for MoveL (default: world)')

    args = parser.parse_args()

    if not args.movej and not args.movel:
        parser.print_help()
        print('\nExamples:')
        print('  MoveJ: ros2 run rsy_path_planning_action_server test_client.py --movej robot1 0 -1.57 1.57 -1.57 -1.57 0')
        print('  MoveL: ros2 run rsy_path_planning_action_server test_client.py --movel robot1 0.3 0.0 0.4')
        return

    rclpy.init()
    client = TestClient()

    if args.movej:
        robot_name = args.movej[0]
        joint_positions = [float(j) for j in args.movej[1:]]
        client.send_movej_goal(robot_name, joint_positions,
                               args.velocity, args.acceleration)
    elif args.movel:
        robot_name = args.movel[0]
        x, y, z = [float(v) for v in args.movel[1:]]
        client.send_movel_goal(robot_name, x, y, z,
                               frame_id=args.frame,
                               velocity_scaling=args.velocity,
                               acceleration_scaling=args.acceleration)

    rclpy.spin(client)


if __name__ == '__main__':
    main()
