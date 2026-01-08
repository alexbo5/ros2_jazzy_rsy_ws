"""Launch file for MTC Motion Sequence Server."""

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
    pkg_share = get_package_share_directory('rsy_mtc_planning')
    robot_pkg = get_package_share_directory('rsy_robot_startup')
    moveit_pkg = get_package_share_directory('rsy_moveit_startup')
    config_file = os.path.join(pkg_share, 'config', 'mtc_planning.yaml')

    # Build MoveIt configuration to get robot_description and robot_description_semantic
    moveit_config = (
        MoveItConfigsBuilder("ur", package_name="rsy_moveit_startup")
        .robot_description(file_path=os.path.join(robot_pkg, "config", "ur.urdf.xacro"))
        .robot_description_semantic(file_path=os.path.join(moveit_pkg, "config", "ur.srdf"))
        .robot_description_kinematics(file_path=os.path.join(moveit_pkg, "config", "kinematics.yaml"))
        .joint_limits(file_path=os.path.join(moveit_pkg, "config", "joint_limits.yaml"))
        .planning_pipelines(pipelines=["pilz_industrial_motion_planner", "ompl"])
        .to_moveit_configs()
    )

    # Load joint limits explicitly for robot_description_planning parameter
    joint_limits = load_yaml('rsy_moveit_startup', 'config/joint_limits.yaml')

    mtc_server = Node(
        package='rsy_mtc_planning',
        executable='mtc_motion_sequence_server',
        name='mtc_motion_sequence_server',
        output='screen',
        parameters=[
            config_file,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.planning_pipelines,
            # Joint limits must be under robot_description_planning.joint_limits for
            # TimeOptimalTrajectoryGeneration to find acceleration limits
            {"robot_description_planning": {"joint_limits": joint_limits}},
            # Increase timeout for waiting on robot_description topics
            {'robot_description_timeout': 30.0},
        ],
    )

    return LaunchDescription([mtc_server])
