"""
MoveIt2 Launch File

Launches move_group node with:
- Pilz Industrial Motion Planner (primary - PTP, LIN)
- OMPL (fallback - obstacle avoidance)
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
import yaml


def generate_launch_description():
    # Package directories
    moveit_pkg = get_package_share_directory("rsy_moveit_startup")
    robot_pkg = get_package_share_directory("rsy_robot_startup")

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

    # Load joint limits
    with open(os.path.join(moveit_pkg, "config", "joint_limits.yaml"), 'r') as f:
        joint_limits = yaml.safe_load(f)

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

    return LaunchDescription([move_group_node])
