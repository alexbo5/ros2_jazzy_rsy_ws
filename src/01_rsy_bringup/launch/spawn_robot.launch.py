from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare


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
            "launch_rviz",
            default_value="false",
            description="Launch RViz?",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="Initially loaded robot controller.",
        )
    )

    # Get launch configurations
    robot_name = LaunchConfiguration("robot_name")
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    launch_rviz = LaunchConfiguration("launch_rviz")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")

    # tf_prefix should end with '/' for proper namespacing
    tf_prefix = PythonExpression(["'", robot_name, "_'"])

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
                    "launch_rviz": launch_rviz,
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

    return LaunchDescription(declared_arguments + [robot_launch])
