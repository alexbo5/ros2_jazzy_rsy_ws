"""
Launch file for cube perception action server.

Supports mock hardware mode for testing without a camera.
Configuration loaded from config/cube_perception_config.yaml.
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
    # Load configuration
    pkg_share = get_package_share_directory("rsy_cube_perception")
    config_path = os.path.join(pkg_share, "config", "cube_perception_config.yaml")
    config = load_config(config_path)

    camera_config = config.get('camera', {})
    mock_config = config.get('mock', {})

    # Launch argument for mock hardware (passed from bringup)
    use_mock_hardware_arg = DeclareLaunchArgument(
        'use_mock_hardware',
        default_value='false',
        description='Use mock hardware instead of real camera'
    )

    action_server_node = Node(
        package='rsy_cube_perception',
        executable='scan_cube_action_server',
        name='cube_perception',
        output='screen',
        parameters=[{
            'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
            'camera_index': camera_config.get('index', 0),
            'show_preview': camera_config.get('show_preview', True),
            'mock_cube_solution': mock_config.get('cube_solution', "R U R' U'"),
            'mock_cube_description': mock_config.get('cube_description', 'Mock cube solution'),
        }],
    )

    return LaunchDescription([
        use_mock_hardware_arg,
        action_server_node,
    ])
