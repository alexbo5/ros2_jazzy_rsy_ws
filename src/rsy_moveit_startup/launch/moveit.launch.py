"""MoveIt2 launch file for dual UR robot setup."""

import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(package, path):
    """Load a yaml file from a package."""
    file_path = os.path.join(get_package_share_directory(package), path)
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    robot_pkg = "rsy_robot_startup"
    moveit_pkg = "rsy_moveit_startup"

    # Build MoveIt configuration
    moveit_config = (
        MoveItConfigsBuilder("ur", package_name=moveit_pkg)
        .robot_description(file_path=os.path.join(
            get_package_share_directory(robot_pkg), "config", "ur.urdf.xacro"))
        .robot_description_semantic(file_path="config/ur.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["pilz_industrial_motion_planner", "ompl"])
        .to_moveit_configs()
    )

    # Load additional planning limits
    joint_limits = load_yaml(moveit_pkg, "config/joint_limits.yaml")
    pilz_config = load_yaml(moveit_pkg, "config/pilz_industrial_motion_planner_planning.yaml")

    # MoveIt move_group node
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"robot_description_planning": {
                "joint_limits": joint_limits,
                "cartesian_limits": pilz_config.get("cartesian_limits", {}),
            }},
        ],
    )

    return LaunchDescription([move_group])
