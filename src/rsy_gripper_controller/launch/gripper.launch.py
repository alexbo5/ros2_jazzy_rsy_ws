"""
Launch file for dual robot gripper action servers.

Starts gripper action servers for both robot1 and robot2.
Each robot can have independent mock hardware settings.
"""

import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # Load gripper configuration
    pkg_share = get_package_share_directory("rsy_gripper_controller")
    config_path = os.path.join(pkg_share, "config", "gripper_config.yaml")
    config = load_config(config_path)

    robot1_config = config.get('robot1', {})
    robot2_config = config.get('robot2', {})

    # Launch arguments for mock hardware (passed from bringup)
    robot1_use_mock_hardware_arg = DeclareLaunchArgument(
        'robot1_use_mock_hardware',
        default_value='false',
        description='Use mock hardware for robot1 gripper'
    )
    robot2_use_mock_hardware_arg = DeclareLaunchArgument(
        'robot2_use_mock_hardware',
        default_value='false',
        description='Use mock hardware for robot2 gripper'
    )

    # Gripper action server for robot1
    robot1_gripper_server = Node(
        package='rsy_gripper_controller',
        executable='gripper_action_server.py',
        name='robot1_gripper_action_server',
        parameters=[{
            'ip_address': robot1_config.get('ip', '192.168.0.51'),
            'port': robot1_config.get('port', 63352),
            'robot_prefix': 'robot1',
            'timeout': 3.0,
            'use_mock_hardware': LaunchConfiguration('robot1_use_mock_hardware'),
        }],
        output='screen',
    )

    # Gripper action server for robot2
    robot2_gripper_server = Node(
        package='rsy_gripper_controller',
        executable='gripper_action_server.py',
        name='robot2_gripper_action_server',
        parameters=[{
            'ip_address': robot2_config.get('ip', '192.168.0.11'),
            'port': robot2_config.get('port', 63352),
            'robot_prefix': 'robot2',
            'timeout': 3.0,
            'use_mock_hardware': LaunchConfiguration('robot2_use_mock_hardware'),
        }],
        output='screen',
    )

    return LaunchDescription([
        robot1_use_mock_hardware_arg,
        robot2_use_mock_hardware_arg,
        robot1_gripper_server,
        robot2_gripper_server,
    ])
