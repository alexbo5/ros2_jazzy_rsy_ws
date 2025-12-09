#!/usr/bin/env python3
"""
Launch file for MoveIt2 move_group nodes for dual UR robots.

This launches the MoveIt2 planning pipeline for both robots,
enabling path planning services (compute_ik, plan_kinematic_path, compute_cartesian_path).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import os
import yaml


def load_yaml(package_name, file_path):
    """Load a yaml file from a package."""
    package_share = FindPackageShare(package_name)
    absolute_file_path = os.path.join(
        package_share.find(package_name),
        file_path
    )
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        return {}


def generate_move_group_node(context, robot_name, robot_namespace):
    """Generate a move_group node for a specific robot."""

    # Get package share directory
    pkg_share = FindPackageShare('path_planning_server').find('path_planning_server')

    # Load configuration files
    kinematics_yaml = load_yaml('path_planning_server', 'config/moveit/kinematics.yaml')
    ompl_planning_yaml = load_yaml('path_planning_server', 'config/moveit/ompl_planning.yaml')
    joint_limits_yaml = load_yaml('path_planning_server', 'config/moveit/joint_limits.yaml')
    moveit_controllers_yaml = load_yaml('path_planning_server', 'config/moveit/moveit_controllers.yaml')

    # Read SRDF
    srdf_path = os.path.join(pkg_share, 'config/moveit/dual_ur.srdf')
    with open(srdf_path, 'r') as f:
        robot_description_semantic = f.read()

    # MoveIt configuration
    move_group_params = {
        'robot_description_semantic': robot_description_semantic,
        'robot_description_kinematics': kinematics_yaml,
        'planning_pipelines': ['ompl'],
        'ompl': ompl_planning_yaml,
        'robot_description_planning': {'joint_limits': joint_limits_yaml},
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_simple_controller_manager': moveit_controllers_yaml.get('moveit_simple_controller_manager', {}),
        'capabilities': '',
        'disable_capabilities': '',
        'publish_robot_description': False,
        'publish_robot_description_semantic': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'monitor_dynamics': False,
        'use_sim_time': False,
        'planning_scene_monitor_options': {
            'robot_description': 'robot_description',
            'joint_state_topic': f'/{robot_namespace}/joint_states',
        },
    }

    return Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        namespace=robot_namespace,
        output='screen',
        parameters=[move_group_params],
        remappings=[
            ('joint_states', f'/{robot_namespace}/joint_states'),
        ],
    )


def launch_setup(context):
    """Setup function called at launch time."""

    robot1_namespace = LaunchConfiguration('robot1_namespace').perform(context)
    robot2_namespace = LaunchConfiguration('robot2_namespace').perform(context)

    nodes = []

    # Create move_group for robot1
    nodes.append(generate_move_group_node(context, 'robot1', robot1_namespace))

    # Create move_group for robot2
    nodes.append(generate_move_group_node(context, 'robot2', robot2_namespace))

    return nodes


def generate_launch_description():
    # Declare launch arguments
    robot1_namespace_arg = DeclareLaunchArgument(
        'robot1_namespace',
        default_value='robot1',
        description='Namespace for robot 1'
    )

    robot2_namespace_arg = DeclareLaunchArgument(
        'robot2_namespace',
        default_value='robot2',
        description='Namespace for robot 2'
    )

    return LaunchDescription([
        robot1_namespace_arg,
        robot2_namespace_arg,
        OpaqueFunction(function=launch_setup),
    ])
