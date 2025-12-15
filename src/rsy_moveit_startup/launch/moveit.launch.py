"""
MoveIt2 Launch File
Launches move_group with Pilz Industrial Motion Planner for Cartesian path planning.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
import yaml


def generate_launch_description():
    # Package directories
    moveit_pkg = get_package_share_directory("rsy_moveit_startup")
    robot_pkg = get_package_share_directory("rsy_robot_startup")

    # Launch arguments
    launch_robot_arg = DeclareLaunchArgument(
        "launch_robot",
        default_value="false",
        description="Launch robot_startup (robot_state_publisher + controllers)",
    )

    # Build MoveIt config
    moveit_config = (
        MoveItConfigsBuilder("ur", package_name="rsy_moveit_startup")
        .robot_description(file_path=os.path.join(robot_pkg, "config", "ur.urdf.xacro"))
        .robot_description_semantic(file_path="config/ur.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["pilz_industrial_motion_planner", "ompl"])
        .to_moveit_configs()
    )

    # Load joint limits for robot_description_planning namespace
    joint_limits_path = os.path.join(moveit_pkg, "config", "joint_limits.yaml")
    with open(joint_limits_path, 'r') as f:
        joint_limits = yaml.safe_load(f)

    # Include robot launch
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(robot_pkg, "launch", "robot.launch.py")),
        condition=IfCondition(LaunchConfiguration("launch_robot")),
    )

    # MoveIt move_group node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"robot_description_planning": {"joint_limits": joint_limits}},
        ],
    )

    # Delay move_group to allow robot to initialize
    delayed_move_group = TimerAction(period=3.0, actions=[move_group_node])

    return LaunchDescription([
        launch_robot_arg,
        robot_launch,
        delayed_move_group,
    ])
