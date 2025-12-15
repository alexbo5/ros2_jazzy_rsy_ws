"""
Launch file for the path planning action server.
Loads configuration from config/planning.yaml.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('rsy_path_planning')
    config_file = os.path.join(pkg_share, 'config', 'planning.yaml')

    path_planning_node = Node(
        package='rsy_path_planning',
        executable='path_planning_action_server.py',
        name='path_planning_action_server',
        output='screen',
        parameters=[config_file],
    )

    return LaunchDescription([path_planning_node])
