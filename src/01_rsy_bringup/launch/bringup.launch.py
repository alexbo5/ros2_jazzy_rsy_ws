#!/usr/bin/env python3
"""
Complete Bringup Launch File for Dual UR Robot System

This launch file starts everything needed for dual robot operation:
1. Two UR robots (with mock hardware by default)
2. Static TF publishers for robot positions in world frame
3. MoveIt2 move_group nodes for path planning
4. Path planning action server (MoveL and MoveJ actions)
5. RViz visualization (optional)

Usage:
    # Simulation mode (default)
    ros2 launch 01_rsy_bringup bringup.launch.py

    # With custom robot positions
    ros2 launch 01_rsy_bringup bringup.launch.py robot1_x:=0.0 robot2_x:=0.8

    # Real robots
    ros2 launch 01_rsy_bringup bringup.launch.py \
        use_mock_hardware:=false \
        robot1_ip:=192.168.1.101 \
        robot2_ip:=192.168.1.102
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    """Setup function to spawn robots and planning components."""

    # Get launch configurations
    robot1_name = LaunchConfiguration("robot1_name").perform(context)
    robot1_ip = LaunchConfiguration("robot1_ip").perform(context)
    robot1_type = LaunchConfiguration("robot1_type").perform(context)
    robot1_x = LaunchConfiguration("robot1_x").perform(context)
    robot1_y = LaunchConfiguration("robot1_y").perform(context)
    robot1_z = LaunchConfiguration("robot1_z").perform(context)
    robot1_roll = LaunchConfiguration("robot1_roll").perform(context)
    robot1_pitch = LaunchConfiguration("robot1_pitch").perform(context)
    robot1_yaw = LaunchConfiguration("robot1_yaw").perform(context)

    robot2_name = LaunchConfiguration("robot2_name").perform(context)
    robot2_ip = LaunchConfiguration("robot2_ip").perform(context)
    robot2_type = LaunchConfiguration("robot2_type").perform(context)
    robot2_x = LaunchConfiguration("robot2_x").perform(context)
    robot2_y = LaunchConfiguration("robot2_y").perform(context)
    robot2_z = LaunchConfiguration("robot2_z").perform(context)
    robot2_roll = LaunchConfiguration("robot2_roll").perform(context)
    robot2_pitch = LaunchConfiguration("robot2_pitch").perform(context)
    robot2_yaw = LaunchConfiguration("robot2_yaw").perform(context)

    use_mock_hardware = LaunchConfiguration("use_mock_hardware").perform(context)
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_planning = LaunchConfiguration("launch_planning").perform(context)

    nodes_to_start = []

    # =========================================================================
    # Log startup info
    # =========================================================================
    nodes_to_start.append(
        LogInfo(msg="\n" + "=" * 60)
    )
    nodes_to_start.append(
        LogInfo(msg="Starting Dual UR Robot System")
    )
    nodes_to_start.append(
        LogInfo(msg=f"  Robot 1: {robot1_name} ({robot1_type}) at ({robot1_x}, {robot1_y}, {robot1_z})")
    )
    nodes_to_start.append(
        LogInfo(msg=f"  Robot 2: {robot2_name} ({robot2_type}) at ({robot2_x}, {robot2_y}, {robot2_z})")
    )
    nodes_to_start.append(
        LogInfo(msg=f"  Mock hardware: {use_mock_hardware}")
    )
    nodes_to_start.append(
        LogInfo(msg=f"  Path planning: {launch_planning}")
    )
    nodes_to_start.append(
        LogInfo(msg="=" * 60 + "\n")
    )

    # =========================================================================
    # Static TF Publishers (world -> robot bases)
    # =========================================================================
    robot1_world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"{robot1_name}_world_tf",
        arguments=[
            "--x", robot1_x,
            "--y", robot1_y,
            "--z", robot1_z,
            "--roll", robot1_roll,
            "--pitch", robot1_pitch,
            "--yaw", robot1_yaw,
            "--frame-id", "world",
            "--child-frame-id", f"{robot1_name}/world",
        ],
    )
    nodes_to_start.append(robot1_world_tf)

    robot2_world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"{robot2_name}_world_tf",
        arguments=[
            "--x", robot2_x,
            "--y", robot2_y,
            "--z", robot2_z,
            "--roll", robot2_roll,
            "--pitch", robot2_pitch,
            "--yaw", robot2_yaw,
            "--frame-id", "world",
            "--child-frame-id", f"{robot2_name}/world",
        ],
    )
    nodes_to_start.append(robot2_world_tf)

    # =========================================================================
    # Robot 1 Hardware/Control
    # =========================================================================
    if robot1_ip or use_mock_hardware == "true":
        robot1_launch = GroupAction(
            actions=[
                PushRosNamespace(robot1_name),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([
                            FindPackageShare("02_robot_startup"),
                            "launch",
                            "ur_control.launch.py",
                        ])
                    ),
                    launch_arguments={
                        "ur_type": robot1_type,
                        "robot_ip": robot1_ip if robot1_ip else "0.0.0.0",
                        "tf_prefix": f"{robot1_name}/",
                        "use_mock_hardware": use_mock_hardware,
                        "launch_rviz": "false",
                        "launch_dashboard_client": "true" if use_mock_hardware == "false" else "false",
                        "controllers_file": PathJoinSubstitution([
                            FindPackageShare("02_robot_startup"),
                            "config",
                            "ur_controllers.yaml",
                        ]).perform(context),
                        "description_launchfile": PathJoinSubstitution([
                            FindPackageShare("02_robot_startup"),
                            "launch",
                            "ur_rsp.launch.py",
                        ]).perform(context),
                    }.items(),
                ),
            ]
        )
        nodes_to_start.append(robot1_launch)

    # =========================================================================
    # Robot 2 Hardware/Control
    # =========================================================================
    if robot2_ip or use_mock_hardware == "true":
        robot2_launch = GroupAction(
            actions=[
                PushRosNamespace(robot2_name),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([
                            FindPackageShare("02_robot_startup"),
                            "launch",
                            "ur_control.launch.py",
                        ])
                    ),
                    launch_arguments={
                        "ur_type": robot2_type,
                        "robot_ip": robot2_ip if robot2_ip else "0.0.0.0",
                        "tf_prefix": f"{robot2_name}/",
                        "use_mock_hardware": use_mock_hardware,
                        "launch_rviz": "false",
                        "launch_dashboard_client": "true" if use_mock_hardware == "false" else "false",
                        "controllers_file": PathJoinSubstitution([
                            FindPackageShare("02_robot_startup"),
                            "config",
                            "ur_controllers.yaml",
                        ]).perform(context),
                        "description_launchfile": PathJoinSubstitution([
                            FindPackageShare("02_robot_startup"),
                            "launch",
                            "ur_rsp.launch.py",
                        ]).perform(context),
                    }.items(),
                ),
            ]
        )
        nodes_to_start.append(robot2_launch)

    # =========================================================================
    # Path Planning System (delayed start to wait for robots)
    # =========================================================================
    if launch_planning == "true":
        planning_launch = TimerAction(
            period=5.0,  # Wait 5 seconds for robot controllers to initialize
            actions=[
                LogInfo(msg="Starting path planning system..."),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([
                            FindPackageShare("path_planning_server"),
                            "launch",
                            "path_planning_server.launch.py",
                        ])
                    ),
                    launch_arguments={
                        "world_frame": "world",
                        "robot1_name": robot1_name,
                        "robot1_namespace": robot1_name,
                        "robot1_ur_type": robot1_type,
                        "robot2_name": robot2_name,
                        "robot2_namespace": robot2_name,
                        "robot2_ur_type": robot2_type,
                    }.items(),
                ),
            ]
        )
        nodes_to_start.append(planning_launch)

    # =========================================================================
    # RViz Visualization
    # =========================================================================
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=[
            "-d",
            PathJoinSubstitution([
                FindPackageShare("01_rsy_bringup"),
                "rviz",
                "multi_robot.rviz",
            ]),
        ],
        condition=IfCondition(launch_rviz),
    )
    nodes_to_start.append(rviz_node)

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    # =========================================================================
    # Robot 1 Arguments
    # =========================================================================
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot1_name",
            default_value="robot1",
            description="Name/namespace for robot 1.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot1_type",
            default_value="ur3e",
            description="Type of robot 1.",
            choices=[
                "ur3", "ur5", "ur10", "ur3e", "ur5e", "ur7e",
                "ur10e", "ur12e", "ur16e", "ur8long", "ur15", "ur18", "ur20", "ur30",
            ],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot1_ip",
            default_value="",
            description="IP address of robot 1 (leave empty for mock hardware).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_x", default_value="0.0",
                              description="X position of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_y", default_value="0.3",
                              description="Y position of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_z", default_value="0.0",
                              description="Z position of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_roll", default_value="0.0",
                              description="Roll of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_pitch", default_value="0.0",
                              description="Pitch of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_yaw", default_value="0.0",
                              description="Yaw of robot 1 in world frame.")
    )

    # =========================================================================
    # Robot 2 Arguments
    # =========================================================================
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot2_name",
            default_value="robot2",
            description="Name/namespace for robot 2.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot2_type",
            default_value="ur3e",
            description="Type of robot 2.",
            choices=[
                "ur3", "ur5", "ur10", "ur3e", "ur5e", "ur7e",
                "ur10e", "ur12e", "ur16e", "ur8long", "ur15", "ur18", "ur20", "ur30",
            ],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot2_ip",
            default_value="",
            description="IP address of robot 2 (leave empty for mock hardware).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_x", default_value="0.0",
                              description="X position of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_y", default_value="-0.3",
                              description="Y position of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_z", default_value="0.0",
                              description="Z position of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_roll", default_value="0.0",
                              description="Roll of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_pitch", default_value="0.0",
                              description="Pitch of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_yaw", default_value="3.14159",
                              description="Yaw of robot 2 in world frame (facing robot 1).")
    )

    # =========================================================================
    # Common Arguments
    # =========================================================================
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="true",
            description="Use mock hardware for simulation (default: true).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz visualization.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_planning",
            default_value="true",
            description="Launch path planning action server.",
        )
    )

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
