"""
Robot startup launch file.
Launches robot_state_publisher with URDF and ros2_control with controllers.
Accepts configuration for mock hardware mode and robot IPs.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction, DeclareLaunchArgument
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory
from launch_ros.descriptions import ParameterValue
from launch.substitutions import Command
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory("rsy_robot_startup")

    # Declare launch arguments
    robot1_use_mock_hardware_arg = DeclareLaunchArgument(
        'robot1_use_mock_hardware',
        default_value='false',
        description='Use mock hardware for robot1'
    )

    robot2_use_mock_hardware_arg = DeclareLaunchArgument(
        'robot2_use_mock_hardware',
        default_value='true',  # Changed from 'false' to 'true' for safe default
        description='Use mock hardware for robot2'
    )

    robot1_robot_ip_arg = DeclareLaunchArgument(
        'robot1_robot_ip',
        default_value='192.168.0.51',
        description='IP address of robot1'
    )

    robot2_robot_ip_arg = DeclareLaunchArgument(
        'robot2_robot_ip',
        default_value='192.168.0.11',
        description='IP address of robot2'
    )

    # Robot2 position arguments
    robot2_x_arg = DeclareLaunchArgument(
        'robot2_x', default_value='-0.77', description='Robot2 X position'
    )
    robot2_y_arg = DeclareLaunchArgument(
        'robot2_y', default_value='0.655', description='Robot2 Y position'
    )
    robot2_z_arg = DeclareLaunchArgument(
        'robot2_z', default_value='0', description='Robot2 Z position'
    )
    robot2_roll_arg = DeclareLaunchArgument(
        'robot2_roll', default_value='0', description='Robot2 roll orientation'
    )
    robot2_pitch_arg = DeclareLaunchArgument(
        'robot2_pitch', default_value='0', description='Robot2 pitch orientation'
    )
    robot2_yaw_arg = DeclareLaunchArgument(
        'robot2_yaw', default_value='3.141592', description='Robot2 yaw orientation'
    )

    # URDF file path
    urdf_xacro_path = os.path.join(pkg_share, "config", "ur.urdf.xacro")

    # Process xacro with arguments to get robot description
    robot_description = ParameterValue(
        Command([
            "xacro ", urdf_xacro_path,
            " robot1_use_mock_hardware:=", LaunchConfiguration('robot1_use_mock_hardware'),
            " robot2_use_mock_hardware:=", LaunchConfiguration('robot2_use_mock_hardware'),
            " robot1_robot_ip:=", LaunchConfiguration('robot1_robot_ip'),
            " robot2_robot_ip:=", LaunchConfiguration('robot2_robot_ip'),
            " robot2_x:=", LaunchConfiguration('robot2_x'),
            " robot2_y:=", LaunchConfiguration('robot2_y'),
            " robot2_z:=", LaunchConfiguration('robot2_z'),
            " robot2_roll:=", LaunchConfiguration('robot2_roll'),
            " robot2_pitch:=", LaunchConfiguration('robot2_pitch'),
            " robot2_yaw:=", LaunchConfiguration('robot2_yaw'),
        ]),
        value_type=str
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": robot_description}],
    )

    ros2_controllers_path = os.path.join(
        pkg_share,
        "config",
        "ros2_controllers.yaml",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    robot_1_urscript_interface = Node(
        package="ur_robot_driver",
        executable="urscript_interface",
        parameters=[{"robot_ip": LaunchConfiguration('robot1_robot_ip')}],
        output="screen",
        condition=UnlessCondition(LaunchConfiguration('robot1_use_mock_hardware')),
    )

    robot_2_urscript_interface = Node(
        package="ur_robot_driver",
        executable="urscript_interface",
        parameters=[{"robot_ip": LaunchConfiguration('robot2_robot_ip')}],
        output="screen",
        condition=UnlessCondition(LaunchConfiguration('robot2_use_mock_hardware')),
    )

    # Controller spawners (will be started after ros2_control_node)
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

    # Start ros2_control_node after robot_state_publisher has started
    start_ros2_control = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[
                TimerAction(
                    period=2.0,
                    actions=[ros2_control_node],
                )
            ],
        )
    )

    # Start controller spawners after ros2_control_node has started
    start_controllers = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                TimerAction(
                    period=3.0,
                    actions=[load_controllers],
                )
            ],
        )
    )

    start_robot1_urscript_interface = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                TimerAction(
                    period=4.0,
                    actions=[robot_1_urscript_interface],
                )
            ],
        )
    )

    start_robot2_urscript_interface = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                TimerAction(
                    period=4.0,
                    actions=[robot_2_urscript_interface],
                )
            ],
        )
    )

    return LaunchDescription(
        [
            # Launch arguments
            robot1_use_mock_hardware_arg,
            robot2_use_mock_hardware_arg,
            robot1_robot_ip_arg,
            robot2_robot_ip_arg,
            robot2_x_arg,
            robot2_y_arg,
            robot2_z_arg,
            robot2_roll_arg,
            robot2_pitch_arg,
            robot2_yaw_arg,
            # Nodes and actions
            robot_state_publisher,
            start_ros2_control,
            start_controllers,
            start_robot1_urscript_interface,
            start_robot2_urscript_interface,
        ]
    )
