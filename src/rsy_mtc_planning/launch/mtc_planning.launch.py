"""Launch file for MTC Motion Sequence Server.

This launch file does NOT generate its own robot_description.
The node subscribes to the /robot_description topic published by robot_state_publisher
(launched by robot.launch.py). This ensures all components use the same URDF
with consistent robot positions from robot_config.yaml.

Architecture:
  robot.launch.py -> robot_state_publisher -> publishes /robot_description topic
  mtc_planning.launch.py -> subscribes to /robot_description topic

This provides a single source of truth for the robot URDF across all packages.
"""

import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def load_yaml(package, path):
    """Load a yaml file from a package."""
    file_path = os.path.join(get_package_share_directory(package), path)
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    pkg_share = get_package_share_directory('rsy_mtc_planning')
    moveit_pkg = get_package_share_directory('rsy_moveit_startup')
    config_file = os.path.join(pkg_share, 'config', 'mtc_planning.yaml')

    # Load MoveIt configuration files
    kinematics = load_yaml('rsy_moveit_startup', 'config/kinematics.yaml')
    joint_limits = load_yaml('rsy_moveit_startup', 'config/joint_limits.yaml')
    cartesian_limits = load_yaml('rsy_moveit_startup', 'config/pilz_cartesian_limits.yaml')
    ompl_config = load_yaml('rsy_moveit_startup', 'config/ompl_planning.yaml')
    pilz_config = load_yaml('rsy_moveit_startup', 'config/pilz_industrial_motion_planner_planning.yaml')

    # Load SRDF for semantic description
    srdf_path = os.path.join(moveit_pkg, "config", "ur.srdf")
    with open(srdf_path, 'r') as f:
        robot_description_semantic = f.read()

    mtc_server = Node(
        package='rsy_mtc_planning',
        executable='mtc_motion_sequence_server',
        name='mtc_motion_sequence_server',
        output='screen',
        parameters=[
            config_file,
            # robot_description is NOT passed here - the node subscribes to /robot_description topic
            {'robot_description_semantic': robot_description_semantic},
            {'robot_description_kinematics': kinematics},
            # Planning pipelines
            {'planning_pipelines': ['pilz_industrial_motion_planner', 'ompl']},
            {'default_planning_pipeline': 'pilz_industrial_motion_planner'},
            {'ompl': ompl_config},
            {'pilz_industrial_motion_planner': pilz_config},
            # Joint limits and cartesian limits for Pilz planner
            {'robot_description_planning': {
                'joint_limits': joint_limits,
                'cartesian_limits': cartesian_limits.get('cartesian_limits', {})
            }},
            # Increase timeout for waiting on robot_description
            {'robot_description_timeout': 30.0},
        ],
    )

    return LaunchDescription([mtc_server])
