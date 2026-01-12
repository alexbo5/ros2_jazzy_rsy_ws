"""Launch file for MTC Motion Sequence Server.

All planner configuration is hardcoded in C++ - no YAML config files needed.
The node subscribes to /robot_description topic from robot_state_publisher.
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
    moveit_pkg = get_package_share_directory('rsy_moveit_startup')

    # Load only essential MoveIt config files (kinematics, SRDF, Pilz)
    kinematics = load_yaml('rsy_moveit_startup', 'config/kinematics.yaml')
    joint_limits = load_yaml('rsy_moveit_startup', 'config/joint_limits.yaml')
    cartesian_limits = load_yaml('rsy_moveit_startup', 'config/pilz_cartesian_limits.yaml')
    pilz_config = load_yaml('rsy_moveit_startup', 'config/pilz_industrial_motion_planner_planning.yaml')

    # Load SRDF for semantic description
    srdf_path = os.path.join(moveit_pkg, "config", "ur.srdf")
    with open(srdf_path, 'r') as f:
        robot_description_semantic = f.read()

    # MTC Motion Sequence Server Node
    # All planner config (OMPL, timeouts, etc.) is hardcoded in C++
    mtc_server = Node(
        package='rsy_mtc_planning',
        executable='mtc_motion_sequence_server',
        name='motion_sequence_server',
        output='screen',
        arguments=[
            '--ros-args',
            '--log-level', 'info',
            '--log-level', 'moveit_ros.planning_scene_monitor:=warn',
            '--log-level', 'moveit_ros.robot_model_loader:=warn',
        ],
        parameters=[
            # MoveIt semantic description (SRDF) - required
            {'robot_description_semantic': robot_description_semantic},

            # Kinematics - required for IK
            {'robot_description_kinematics': kinematics},

            # Planning pipelines - OMPL config is hardcoded in C++
            {'planning_pipelines': ['pilz_industrial_motion_planner', 'ompl']},
            {'default_planning_pipeline': 'pilz_industrial_motion_planner'},

            # Pilz planner configuration - required for LIN motions
            {'pilz_industrial_motion_planner': pilz_config},

            # Joint limits for planning
            {'robot_description_planning': {
                'joint_limits': joint_limits,
                'cartesian_limits': cartesian_limits.get('cartesian_limits', {})
            }},
        ],
    )

    return LaunchDescription([mtc_server])
