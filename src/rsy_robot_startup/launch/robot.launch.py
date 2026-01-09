"""
Robot startup launch file.
Launches robot_state_publisher with URDF and ros2_control with controllers.

This is the primary source of robot_description in the system. The MTC server
subscribes to /robot_description topic to get the same URDF.
"""

import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    ExecuteProcess,
    RegisterEventHandler,
    DeclareLaunchArgument,
    GroupAction,
)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.descriptions import ParameterValue
from launch.substitutions import Command
from launch.conditions import UnlessCondition


def generate_launch_description():
    pkg_share = get_package_share_directory("rsy_robot_startup")

    # Load robot configuration
    config_path = os.path.join(pkg_share, "config", "robot_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    robot1_config = config.get('robot1', {})
    robot2_config = config.get('robot2', {})
    robot2_pos = robot2_config.get('position', {})
    robot2_ori = robot2_config.get('orientation', {})

    # Launch arguments for mock hardware (passed from bringup)
    robot1_use_mock_hardware_arg = DeclareLaunchArgument(
        'robot1_use_mock_hardware',
        default_value='true',
        description='Use mock hardware for robot1'
    )
    robot2_use_mock_hardware_arg = DeclareLaunchArgument(
        'robot2_use_mock_hardware',
        default_value='true',
        description='Use mock hardware for robot2'
    )

    # Process xacro to get robot description
    urdf_xacro_path = os.path.join(pkg_share, "config", "ur.urdf.xacro")
    robot_description = ParameterValue(
        Command([
            "xacro ", urdf_xacro_path,
            " robot1_use_mock_hardware:=", LaunchConfiguration('robot1_use_mock_hardware'),
            " robot2_use_mock_hardware:=", LaunchConfiguration('robot2_use_mock_hardware'),
            " robot1_robot_ip:=", robot1_config.get('robot_ip', '192.168.0.51'),
            " robot2_robot_ip:=", robot2_config.get('robot_ip', '192.168.0.11'),
            " robot2_x:=", str(robot2_pos.get('x', -0.77)),
            " robot2_y:=", str(robot2_pos.get('y', 0.655)),
            " robot2_z:=", str(robot2_pos.get('z', 0.0)),
            " robot2_roll:=", str(robot2_ori.get('roll', 0.0)),
            " robot2_pitch:=", str(robot2_ori.get('pitch', 0.0)),
            " robot2_yaw:=", str(robot2_ori.get('yaw', 3.141592)),
        ]),
        value_type=str
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": robot_description}],
    )

    # ROS2 control node
    ros2_controllers_path = os.path.join(pkg_share, "config", "ros2_controllers.yaml")
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[("/controller_manager/robot_description", "/robot_description")],
        output="screen",
    )

    # Controller spawner
    controllers = [
        "joint_state_broadcaster",
        "robot1_scaled_joint_trajectory_controller",
        "robot1_io_and_status_controller",
        "robot1_speed_scaling_state_broadcaster",
        "robot1_force_torque_sensor_broadcaster",
        "robot1_ur_configuration_controller",
        "robot2_scaled_joint_trajectory_controller",
        "robot2_io_and_status_controller",
        "robot2_speed_scaling_state_broadcaster",
        "robot2_force_torque_sensor_broadcaster",
        "robot2_ur_configuration_controller",
    ]
    load_controllers = ExecuteProcess(
        cmd=["ros2", "run", "controller_manager", "spawner"] + controllers,
        output="screen",
    )

    # Event-based startup chain:
    # robot_state_publisher -> ros2_control_node -> controllers + urscript interfaces
    start_ros2_control = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[ros2_control_node],
        )
    )

    start_controllers_and_interfaces = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                GroupAction([
                    load_controllers
                ])
            ],
        )
    )

    return LaunchDescription([
        # Launch arguments
        robot1_use_mock_hardware_arg,
        robot2_use_mock_hardware_arg,
        # Nodes and event handlers
        robot_state_publisher,
        start_ros2_control,
        start_controllers_and_interfaces,
    ])
