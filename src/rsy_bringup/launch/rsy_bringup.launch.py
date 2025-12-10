import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Get package share directories
    bringup_pkg_share = get_package_share_directory("rsy_bringup")
    robot_startup_pkg_share = get_package_share_directory("rsy_robot_startup")

    # RViz config path
    rviz_config_file = os.path.join(bringup_pkg_share, "config", "moveit.rviz")

    # Include the robot startup launch file
    robot_startup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_startup_pkg_share, "launch", "robot.launch.py")
        )
    )

    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    return LaunchDescription(
        [
            robot_startup_launch,
            rviz_node,
        ]
    )
