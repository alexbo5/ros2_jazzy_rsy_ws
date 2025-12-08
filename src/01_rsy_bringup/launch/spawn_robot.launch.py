from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    # Get launch configurations
    robot_name = LaunchConfiguration("robot_name").perform(context)
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")

    # Position arguments
    robot_x = LaunchConfiguration("robot_x").perform(context)
    robot_y = LaunchConfiguration("robot_y").perform(context)
    robot_z = LaunchConfiguration("robot_z").perform(context)
    robot_roll = LaunchConfiguration("robot_roll").perform(context)
    robot_pitch = LaunchConfiguration("robot_pitch").perform(context)
    robot_yaw = LaunchConfiguration("robot_yaw").perform(context)

    # tf_prefix should end with '_' for proper namespacing
    tf_prefix = f"{robot_name}_"

    nodes_to_start = []

    # Static TF: world -> robot_world
    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"{robot_name}_world_tf",
        arguments=[
            "--x", robot_x,
            "--y", robot_y,
            "--z", robot_z,
            "--roll", robot_roll,
            "--pitch", robot_pitch,
            "--yaw", robot_yaw,
            "--frame-id", "world",
            "--child-frame-id", f"{robot_name}_world",
        ],
    )
    nodes_to_start.append(world_tf)

    # Include the robot control launch file within a namespace
    robot_launch = GroupAction(
        actions=[
            PushRosNamespace(robot_name),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("02_robot_startup"),
                        "launch",
                        "ur_control.launch.py",
                    ])
                ),
                launch_arguments={
                    "ur_type": ur_type,
                    "robot_ip": robot_ip,
                    "tf_prefix": tf_prefix,
                    "use_mock_hardware": use_mock_hardware,
                    "launch_rviz": "false",
                    "initial_joint_controller": initial_joint_controller,
                    "launch_dashboard_client": "true",
                    "controllers_file": PathJoinSubstitution([
                        FindPackageShare("02_robot_startup"),
                        "config",
                        "ur_controllers.yaml",
                    ]),
                    "description_launchfile": PathJoinSubstitution([
                        FindPackageShare("02_robot_startup"),
                        "launch",
                        "ur_rsp.launch.py",
                    ]),
                }.items(),
            ),
        ]
    )
    nodes_to_start.append(robot_launch)

    return nodes_to_start


def generate_launch_description():
    # Declare arguments
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_name",
            default_value="robot1",
            description="Unique name for the robot. Used as namespace and tf_prefix.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            description="Type/series of used UR robot.",
            choices=[
                "ur3",
                "ur5",
                "ur10",
                "ur3e",
                "ur5e",
                "ur7e",
                "ur10e",
                "ur12e",
                "ur16e",
                "ur8long",
                "ur15",
                "ur18",
                "ur20",
                "ur30",
            ],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            description="IP address by which the robot can be reached.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="false",
            description="Start robot with mock hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="Initially loaded robot controller.",
        )
    )
    # Position arguments
    declared_arguments.append(
        DeclareLaunchArgument("robot_x", default_value="0.0", description="X position of robot in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot_y", default_value="0.0", description="Y position of robot in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot_z", default_value="0.0", description="Z position of robot in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot_roll", default_value="0.0", description="Roll of robot in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot_pitch", default_value="0.0", description="Pitch of robot in world frame.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot_yaw", default_value="0.0", description="Yaw of robot in world frame.")
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
