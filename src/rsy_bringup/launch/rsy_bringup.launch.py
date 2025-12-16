"""
RSY Bringup Launch File
Launches all components: robot, MoveIt, and application nodes.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    bringup_pkg = get_package_share_directory("rsy_bringup")
    robot_pkg = get_package_share_directory("rsy_robot_startup")
    moveit_pkg = get_package_share_directory("rsy_moveit_startup")
    path_planning_pkg = get_package_share_directory("rsy_path_planning")
    cube_motion_pkg = get_package_share_directory("rsy_cube_motion")
    cube_perception_pkg = get_package_share_directory("rsy_cube_perception")

    # 1. Robot startup (robot_state_publisher + controllers)
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(robot_pkg, "launch", "robot.launch.py"))
    )

    # 2. MoveIt startup (move_group)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_pkg, "launch", "moveit.launch.py"))
    )

    # 3. RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(bringup_pkg, "config", "moveit.rviz")],
    )

    # 4. Application nodes (delayed to ensure MoveIt is ready)
    app_delay = 5.0

    path_planning_launch = TimerAction(
        period=app_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(path_planning_pkg, "launch", "path_planning.launch.py"))
        )]
    )

    cube_motion_launch = TimerAction(
        period=app_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(cube_motion_pkg, "launch", "cube_motion_server.launch.py"))
        )]
    )

    cube_perception_launch = TimerAction(
        period=app_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(cube_perception_pkg, "launch", "cube_perception.launch.py"))
        )]
    )

    return LaunchDescription([
        robot_launch,
        moveit_launch,
        rviz_node,
        path_planning_launch,
        cube_motion_launch,
        cube_perception_launch,
    ])
