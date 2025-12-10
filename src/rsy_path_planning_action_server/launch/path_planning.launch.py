"""
Launch file for the path planning action server.
This should be launched after the robot startup (rsy_robot_startup).
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Path Planning Action Server Node
    path_planning_action_server = Node(
        package='rsy_path_planning_action_server',
        executable='path_planning_action_server.py',
        name='path_planning_action_server',
        output='screen',
        parameters=[],
    )

    return LaunchDescription([
        path_planning_action_server,
    ])
