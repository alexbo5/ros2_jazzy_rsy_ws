"""
Launch file for dual robot gripper action servers.

This launch file starts gripper action servers for both robot1 and robot2.
Each server uses prefix-based naming (e.g., robot1_robotiq_gripper, robot2_robotiq_gripper).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments for robot1
    robot1_ip_arg = DeclareLaunchArgument(
        'robot1_gripper_ip',
        default_value='192.168.1.10',
        description='IP address of robot1 gripper'
    )

    robot1_port_arg = DeclareLaunchArgument(
        'robot1_gripper_port',
        default_value='63352',
        description='Port of robot1 gripper'
    )

    # Declare launch arguments for robot2
    robot2_ip_arg = DeclareLaunchArgument(
        'robot2_gripper_ip',
        default_value='192.168.1.11',
        description='IP address of robot2 gripper'
    )

    robot2_port_arg = DeclareLaunchArgument(
        'robot2_gripper_port',
        default_value='63352',
        description='Port of robot2 gripper'
    )

    # Gripper action server for robot1
    robot1_gripper_server = Node(
        package='rsy_gripper_controller',
        executable='gripper_action_server.py',
        name='robot1_gripper_action_server',
        parameters=[{
            'ip_address': LaunchConfiguration('robot1_gripper_ip'),
            'port': LaunchConfiguration('robot1_gripper_port'),
            'robot_prefix': 'robot1',
            'timeout': 3.0,
        }],
        output='screen',
    )

    # Gripper action server for robot2
    robot2_gripper_server = Node(
        package='rsy_gripper_controller',
        executable='gripper_action_server.py',
        name='robot2_gripper_action_server',
        parameters=[{
            'ip_address': LaunchConfiguration('robot2_gripper_ip'),
            'port': LaunchConfiguration('robot2_gripper_port'),
            'robot_prefix': 'robot2',
            'timeout': 3.0,
        }],
        output='screen',
    )

    return LaunchDescription([
        # Launch arguments
        robot1_ip_arg,
        robot1_port_arg,
        robot2_ip_arg,
        robot2_port_arg,
        # Nodes
        robot1_gripper_server,
        robot2_gripper_server,
    ])
