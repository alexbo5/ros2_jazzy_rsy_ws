"""
MoveIt2 Launch File
Launches move_group and MoveIt Servo for both robots.
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

    # Load servo configs
    with open(os.path.join(moveit_pkg, "config", "servo_robot1.yaml"), 'r') as f:
        servo_params_robot1 = yaml.safe_load(f)

    with open(os.path.join(moveit_pkg, "config", "servo_robot2.yaml"), 'r') as f:
        servo_params_robot2 = yaml.safe_load(f)

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

    # MoveIt Servo for Robot 1
    servo_robot1 = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_robot1",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            servo_params_robot1,
        ],
    )

    # MoveIt Servo for Robot 2
    servo_robot2 = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_robot2",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            servo_params_robot2,
        ],
    )

    return LaunchDescription([
        move_group_node,
        servo_robot1,
        servo_robot2,
    ])
