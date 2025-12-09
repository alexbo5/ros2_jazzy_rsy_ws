#!/usr/bin/env python3
"""
Complete launch file for dual robot path planning.

This launch file starts:
1. MoveIt2 move_group nodes for both robots (provides planning services)
2. Path planning action server (provides MoveL and MoveJ actions)

Prerequisites:
- Robot hardware/simulation must already be running (ur_control.launch.py)
- Robot state publishers must be active
- Joint state broadcasters must be active
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package share directory
    pkg_share = FindPackageShare('path_planning_server')

    # Declare launch arguments
    world_frame_arg = DeclareLaunchArgument(
        'world_frame',
        default_value='world',
        description='Global reference frame'
    )

    robot1_namespace_arg = DeclareLaunchArgument(
        'robot1_namespace',
        default_value='robot1',
        description='Namespace for robot 1'
    )

    robot2_namespace_arg = DeclareLaunchArgument(
        'robot2_namespace',
        default_value='robot2',
        description='Namespace for robot 2'
    )

    robot1_ur_type_arg = DeclareLaunchArgument(
        'robot1_ur_type',
        default_value='ur3e',
        description='UR robot type for robot 1'
    )

    robot2_ur_type_arg = DeclareLaunchArgument(
        'robot2_ur_type',
        default_value='ur3e',
        description='UR robot type for robot 2'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # Include MoveIt launch
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_share, 'launch', 'moveit.launch.py'])
        ]),
        launch_arguments={
            'robot1_namespace': LaunchConfiguration('robot1_namespace'),
            'robot2_namespace': LaunchConfiguration('robot2_namespace'),
        }.items()
    )

    # Path planning action server (delayed to allow MoveIt to start)
    path_planning_server_launch = TimerAction(
        period=3.0,  # Wait 3 seconds for MoveIt to initialize
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([pkg_share, 'launch', 'path_planning_server.launch.py'])
                ]),
                launch_arguments={
                    'world_frame': LaunchConfiguration('world_frame'),
                    'robot1_namespace': LaunchConfiguration('robot1_namespace'),
                    'robot2_namespace': LaunchConfiguration('robot2_namespace'),
                    'robot1_ur_type': LaunchConfiguration('robot1_ur_type'),
                    'robot2_ur_type': LaunchConfiguration('robot2_ur_type'),
                }.items()
            )
        ]
    )

    return LaunchDescription([
        # Arguments
        world_frame_arg,
        robot1_namespace_arg,
        robot2_namespace_arg,
        robot1_ur_type_arg,
        robot2_ur_type_arg,
        use_sim_time_arg,
        # Launch files
        moveit_launch,
        path_planning_server_launch,
    ])
