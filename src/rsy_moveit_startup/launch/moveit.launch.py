"""MoveIt2 launch file for dual UR robot setup.

Loads robot configuration from robot_config.yaml to ensure consistency
with robot.launch.py. Both use the same URDF xacro with identical mappings.
"""

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

    # Load robot configuration (same source as robot.launch.py)
    config_path = os.path.join(get_package_share_directory(robot_pkg), "config", "robot_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    robot1_config = config.get('robot1', {})
    robot2_config = config.get('robot2', {})
    robot2_pos = robot2_config.get('position', {})
    robot2_ori = robot2_config.get('orientation', {})

    # Build xacro mappings from robot_config.yaml
    xacro_mappings = {
        "robot1_robot_ip": robot1_config.get('robot_ip', '192.168.0.51'),
        "robot2_robot_ip": robot2_config.get('robot_ip', '192.168.0.11'),
        "robot2_x": str(robot2_pos.get('x', -0.77)),
        "robot2_y": str(robot2_pos.get('y', 0.655)),
        "robot2_z": str(robot2_pos.get('z', 0.0)),
        "robot2_roll": str(robot2_ori.get('roll', 0.0)),
        "robot2_pitch": str(robot2_ori.get('pitch', 0.0)),
        "robot2_yaw": str(robot2_ori.get('yaw', 3.141592)),
        "robot1_use_mock_hardware": "true",
        "robot2_use_mock_hardware": "true",
    }

    urdf_path = os.path.join(get_package_share_directory(robot_pkg), "config", "ur.urdf.xacro")

    # Build MoveIt configuration with robot_config.yaml values
    moveit_config = (
        MoveItConfigsBuilder("ur", package_name=moveit_pkg)
        .robot_description(
            file_path=urdf_path,
            mappings=xacro_mappings
        )
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
            # Enable MTC execution capability for execute_task_solution action
            {"capabilities": "move_group/ExecuteTaskSolutionCapability"},
        ],
    )

    return LaunchDescription([move_group])
