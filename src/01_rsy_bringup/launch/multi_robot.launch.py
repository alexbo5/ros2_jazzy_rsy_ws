from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    """Setup function to spawn multiple robots based on configuration."""
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

    nodes_to_start = []

    # Static TF: world -> robot1_world
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
            "--child-frame-id", f"{robot1_name}_world",
        ],
    )
    nodes_to_start.append(robot1_world_tf)

    # Static TF: world -> robot2_world
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
            "--child-frame-id", f"{robot2_name}_world",
        ],
    )
    nodes_to_start.append(robot2_world_tf)

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

    # Launch centralized RViz
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
    # Robot 1 position arguments
    declared_arguments.append(
        DeclareLaunchArgument("robot1_x", default_value="0.0", description="X position of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_y", default_value="0.0", description="Y position of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_z", default_value="0.0", description="Z position of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_roll", default_value="0.0", description="Roll of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_pitch", default_value="0.0", description="Pitch of robot 1 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot1_yaw", default_value="0.0", description="Yaw of robot 1 in world frame.")
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
    # Robot 2 position arguments
    declared_arguments.append(
        DeclareLaunchArgument("robot2_x", default_value="1.0", description="X position of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_y", default_value="0.0", description="Y position of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_z", default_value="0.0", description="Z position of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_roll", default_value="0.0", description="Roll of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_pitch", default_value="0.0", description="Pitch of robot 2 in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot2_yaw", default_value="0.0", description="Yaw of robot 2 in world frame.")
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
