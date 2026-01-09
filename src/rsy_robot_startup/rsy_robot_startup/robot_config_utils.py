"""Shared utilities for loading robot configuration."""

import os
import yaml
from ament_index_python.packages import get_package_share_directory


def get_robot_config():
    """Load robot configuration from robot_config.yaml."""
    robot_pkg = get_package_share_directory("rsy_robot_startup")
    config_path = os.path.join(robot_pkg, "config", "robot_config.yaml")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_xacro_mappings(use_mock_hardware=True):
    """
    Get xacro argument mappings for URDF processing.

    These mappings ensure all components (robot, MoveIt, MTC) use
    the same robot configuration from robot_config.yaml.

    Args:
        use_mock_hardware: Whether to use mock hardware (default True for planning)

    Returns:
        Dictionary of xacro argument mappings
    """
    config = get_robot_config()

    robot1_config = config.get('robot1', {})
    robot2_config = config.get('robot2', {})
    robot2_pos = robot2_config.get('position', {})
    robot2_ori = robot2_config.get('orientation', {})

    mock_hw = "true" if use_mock_hardware else "false"

    return {
        "robot1_robot_ip": robot1_config.get('robot_ip', '192.168.0.51'),
        "robot2_robot_ip": robot2_config.get('robot_ip', '192.168.0.11'),
        "robot2_x": str(robot2_pos.get('x', -0.77)),
        "robot2_y": str(robot2_pos.get('y', 0.655)),
        "robot2_z": str(robot2_pos.get('z', 0.0)),
        "robot2_roll": str(robot2_ori.get('roll', 0.0)),
        "robot2_pitch": str(robot2_ori.get('pitch', 0.0)),
        "robot2_yaw": str(robot2_ori.get('yaw', 3.141592)),
        "robot1_use_mock_hardware": mock_hw,
        "robot2_use_mock_hardware": mock_hw,
    }


def get_urdf_path():
    """Get the path to the URDF xacro file."""
    robot_pkg = get_package_share_directory("rsy_robot_startup")
    return os.path.join(robot_pkg, "config", "ur.urdf.xacro")
