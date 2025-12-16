import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
from launch_ros.descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory("rsy_robot_startup")

    # URDF file path
    urdf_xacro_path = os.path.join(pkg_share, "config", "ur.urdf.xacro")

    # Process xacro to get robot description
    robot_description = ParameterValue(
        Command(["xacro ", urdf_xacro_path]),
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

    return LaunchDescription(
        [
            robot_state_publisher,
            start_ros2_control,
            start_controllers,
        ]
    )
