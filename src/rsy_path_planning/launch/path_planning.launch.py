"""
Launch file for the path planning action server.
This should be launched after the robot startup (rsy_robot_startup).

The action server connects to the existing move_group node via the MoveGroup action interface,
so no MoveIt configuration is needed here.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Path Planning Action Server
    # Connects to existing move_group via /move_action
    path_planning_action_server = Node(
        package='rsy_path_planning',
        executable='path_planning_action_server.py',
        name='path_planning_action_server',
        output='screen',
    )

    return LaunchDescription([
        path_planning_action_server,
    ])
