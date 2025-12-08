from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config",
            default_value="multi_robot.rviz",
            description="Name of the RViz config file to use.",
        )
    )

    rviz_config = LaunchConfiguration("rviz_config")

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
                rviz_config,
            ]),
        ],
    )

    return LaunchDescription(declared_arguments + [rviz_node])
