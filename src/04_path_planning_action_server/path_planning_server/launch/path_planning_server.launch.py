#!/usr/bin/env python3
"""
Launch file for the Path Planning Action Server.

This launches the action server that provides MoveL and MoveJ actions
for two robots with poses relative to the global world frame.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    world_frame_arg = DeclareLaunchArgument(
        'world_frame',
        default_value='world',
        description='Global reference frame for all poses'
    )

    ground_height_arg = DeclareLaunchArgument(
        'ground_height',
        default_value='0.0',
        description='Height of the ground plane in world frame (meters)'
    )

    ground_safety_margin_arg = DeclareLaunchArgument(
        'ground_safety_margin',
        default_value='0.02',
        description='Safety margin above ground for collision avoidance (meters)'
    )

    # Robot 1 arguments
    robot1_name_arg = DeclareLaunchArgument(
        'robot1_name',
        default_value='robot1',
        description='Name of robot 1'
    )

    robot1_namespace_arg = DeclareLaunchArgument(
        'robot1_namespace',
        default_value='robot1',
        description='Namespace for robot 1'
    )

    robot1_ur_type_arg = DeclareLaunchArgument(
        'robot1_ur_type',
        default_value='ur3e',
        description='UR robot type for robot 1 (ur3, ur3e, ur5, ur5e, ur10, ur10e, etc.)'
    )

    # Robot 2 arguments
    robot2_name_arg = DeclareLaunchArgument(
        'robot2_name',
        default_value='robot2',
        description='Name of robot 2'
    )

    robot2_namespace_arg = DeclareLaunchArgument(
        'robot2_namespace',
        default_value='robot2',
        description='Namespace for robot 2'
    )

    robot2_ur_type_arg = DeclareLaunchArgument(
        'robot2_ur_type',
        default_value='ur3e',
        description='UR robot type for robot 2'
    )

    # Planning parameters
    planning_time_arg = DeclareLaunchArgument(
        'planning_time',
        default_value='5.0',
        description='Maximum planning time in seconds'
    )

    num_planning_attempts_arg = DeclareLaunchArgument(
        'num_planning_attempts',
        default_value='10',
        description='Number of planning attempts'
    )

    # Get config file path
    config_file = PathJoinSubstitution([
        FindPackageShare('path_planning_server'),
        'config',
        'path_planning_params.yaml'
    ])

    # Path Planning Action Server Node
    path_planning_server_node = Node(
        package='path_planning_server',
        executable='path_planning_action_server.py',
        name='path_planning_action_server',
        output='screen',
        parameters=[
            config_file,
            {
                'world_frame': LaunchConfiguration('world_frame'),
                'ground_height': LaunchConfiguration('ground_height'),
                'ground_safety_margin': LaunchConfiguration('ground_safety_margin'),
                'robot1.name': LaunchConfiguration('robot1_name'),
                'robot1.namespace': LaunchConfiguration('robot1_namespace'),
                'robot1.base_frame': [LaunchConfiguration('robot1_namespace'), '/base_link'],
                'robot1.ee_frame': [LaunchConfiguration('robot1_namespace'), '/tool0'],
                'robot1.planning_group': [LaunchConfiguration('robot1_namespace'), '_manipulator'],
                'robot1.ur_type': LaunchConfiguration('robot1_ur_type'),
                'robot2.name': LaunchConfiguration('robot2_name'),
                'robot2.namespace': LaunchConfiguration('robot2_namespace'),
                'robot2.base_frame': [LaunchConfiguration('robot2_namespace'), '/base_link'],
                'robot2.ee_frame': [LaunchConfiguration('robot2_namespace'), '/tool0'],
                'robot2.planning_group': [LaunchConfiguration('robot2_namespace'), '_manipulator'],
                'robot2.ur_type': LaunchConfiguration('robot2_ur_type'),
                'planning_time': LaunchConfiguration('planning_time'),
                'num_planning_attempts': LaunchConfiguration('num_planning_attempts'),
            }
        ],
    )

    return LaunchDescription([
        # Launch arguments
        world_frame_arg,
        ground_height_arg,
        ground_safety_margin_arg,
        robot1_name_arg,
        robot1_namespace_arg,
        robot1_ur_type_arg,
        robot2_name_arg,
        robot2_namespace_arg,
        robot2_ur_type_arg,
        planning_time_arg,
        num_planning_attempts_arg,
        # Nodes
        path_planning_server_node,
    ])
