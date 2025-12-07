from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    """Setup function to spawn multiple robots based on configuration."""
    # Get launch configurations
    robot1_name = LaunchConfiguration("robot1_name").perform(context)
    robot1_ip = LaunchConfiguration("robot1_ip").perform(context)
    robot1_type = LaunchConfiguration("robot1_type").perform(context)

    robot2_name = LaunchConfiguration("robot2_name").perform(context)
    robot2_ip = LaunchConfiguration("robot2_ip").perform(context)
    robot2_type = LaunchConfiguration("robot2_type").perform(context)

    use_mock_hardware = LaunchConfiguration("use_mock_hardware").perform(context)
    launch_rviz = LaunchConfiguration("launch_rviz").perform(context)

    nodes_to_start = []

    # Spawn Robot 1
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
                        "tf_prefix": f"{robot1_name}_",
                        "use_mock_hardware": use_mock_hardware,
                        "launch_rviz": launch_rviz,
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

    # Spawn Robot 2
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
                        "tf_prefix": f"{robot2_name}_",
                        "use_mock_hardware": use_mock_hardware,
                        "launch_rviz": "false",  # Only one RViz
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

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    # Robot 1 arguments
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
            default_value="ur5e",
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
            description="IP address of robot 1.",
        )
    )

    # Robot 2 arguments
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
            default_value="ur5e",
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
            description="IP address of robot 2.",
        )
    )

    # Common arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="false",
            description="Use mock hardware for simulation.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz?",
        )
    )

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
