import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Get package share directories
    bringup_pkg_share = get_package_share_directory("rsy_bringup")
    robot_startup_pkg_share = get_package_share_directory("rsy_robot_startup")
    path_planning_pkg_share = get_package_share_directory("rsy_path_planning")
    cube_motion_pkg_share = get_package_share_directory("rsy_cube_motion")
    cube_perception_pkg_share = get_package_share_directory("rsy_cube_perception")

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

    # Path Planning Action Server - delayed to ensure MoveIt is ready
    path_planning_action_server = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(path_planning_pkg_share, "launch", "path_planning.launch.py")
                )
            )
        ]
    )

    # Cube Motion Action Server - delayed to ensure MoveIt is ready
    cube_motion_action_server = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(cube_motion_pkg_share, "launch", "cube_motion_server.launch.py")
                )
            )
        ]
    )

    # Cube Perception Action Server - delayed to ensure MoveIt is ready
    cube_perception_action_server = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(cube_perception_pkg_share, "launch", "cube_perception.launch.py")
                )
            )
        ]
    )

    return LaunchDescription(
        [
            robot_startup_launch,
            rviz_node,
            path_planning_action_server,
            cube_motion_action_server,
            cube_perception_action_server,
        ]
    )
