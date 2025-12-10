#!/usr/bin/env python3
"""
Bringup Launch File for Single UR Robot with Path Planning

Launches:
1. UR robot (mock hardware by default)
2. MoveIt2 move_group for path planning
3. Path planning action server
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Arguments
    robot_ns_arg = DeclareLaunchArgument('robot_namespace', default_value='robot1')
    ur_type_arg = DeclareLaunchArgument('ur_type', default_value='ur3e')
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value='')
    use_mock_arg = DeclareLaunchArgument('use_mock_hardware', default_value='true')
    launch_rviz_arg = DeclareLaunchArgument('launch_rviz', default_value='true')
    launch_planning_arg = DeclareLaunchArgument('launch_planning', default_value='true')

    robot_ns = LaunchConfiguration('robot_namespace')
    ur_type = LaunchConfiguration('ur_type')
    use_mock = LaunchConfiguration('use_mock_hardware')
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_planning = LaunchConfiguration('launch_planning')

    # Robot launch
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'),
                'launch',
                'ur_control.launch.py',
            ])
        ),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': LaunchConfiguration('robot_ip'),
            'tf_prefix': [robot_ns, '/'],
            'use_mock_hardware': use_mock,
            'launch_rviz': 'false',
        }.items(),
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription([
        robot_ns_arg,
        ur_type_arg,
        robot_ip_arg,
        use_mock_arg,
        launch_rviz_arg,
        launch_planning_arg,
        LogInfo(msg='Starting UR Robot...'),
        robot_launch,
        rviz_node,
    ])
