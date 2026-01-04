"""
Launch file for cube perception action server.

Supports mock hardware mode for testing without a camera.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_mock_hardware_arg = DeclareLaunchArgument(
        'use_mock_hardware',
        default_value='false',
        description='Use mock hardware instead of real camera'
    )

    mock_cube_solution_arg = DeclareLaunchArgument(
        'mock_cube_solution',
        default_value="R U R' U'",
        description='Mock Kociemba solution to return in mock mode'
    )

    mock_cube_description_arg = DeclareLaunchArgument(
        'mock_cube_description',
        default_value='Mock cube solution for testing',
        description='Description of mock solution'
    )

    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='Camera device index'
    )

    show_preview_arg = DeclareLaunchArgument(
        'show_preview',
        default_value='true',
        description='Show camera preview window with grid overlay'
    )

    action_server_node = Node(
        package='rsy_cube_perception',
        executable='scan_cube_action_server',
        name='cube_perception',
        output='screen',
        parameters=[{
            'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
            'mock_cube_solution': LaunchConfiguration('mock_cube_solution'),
            'mock_cube_description': LaunchConfiguration('mock_cube_description'),
            'show_preview': LaunchConfiguration('show_preview'),
        }],
    )

    return LaunchDescription([
        use_mock_hardware_arg,
        mock_cube_solution_arg,
        mock_cube_description_arg,
        camera_index_arg,
        show_preview_arg,
        action_server_node,
    ])


if __name__ == "__main__":
    generate_launch_description()
