"""
RSY Bringup Launch File
Launches all components: robot, MoveIt, and application nodes.
"""

import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    GroupAction,
    RegisterEventHandler,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # Package directories
    bringup_pkg = get_package_share_directory("rsy_bringup")
    robot_pkg = get_package_share_directory("rsy_robot_startup")
    moveit_pkg = get_package_share_directory("rsy_moveit_startup")
    mtc_planning_pkg = get_package_share_directory("rsy_mtc_planning")
    cube_motion_pkg = get_package_share_directory("rsy_cube_motion")
    cube_perception_pkg = get_package_share_directory("rsy_cube_perception")
    gripper_pkg = get_package_share_directory("rsy_gripper_controller")

    # Load system configuration (mock hardware flags)
    config_path = os.path.join(bringup_pkg, "config", "system_config.yaml")
    config = load_config(config_path)

    robot1_use_mock = config.get('robot1', {}).get('use_mock_hardware', False)
    robot2_use_mock = config.get('robot2', {}).get('use_mock_hardware', False)
    camera_use_mock = config.get('camera', {}).get('use_mock_hardware', False)

    # Launch arguments for command line override
    robot1_mock_arg = DeclareLaunchArgument(
        'robot1_use_mock_hardware',
        default_value=str(robot1_use_mock).lower(),
        description='Use mock hardware for robot1'
    )
    robot2_mock_arg = DeclareLaunchArgument(
        'robot2_use_mock_hardware',
        default_value=str(robot2_use_mock).lower(),
        description='Use mock hardware for robot2'
    )
    camera_mock_arg = DeclareLaunchArgument(
        'camera_use_mock_hardware',
        default_value=str(camera_use_mock).lower(),
        description='Use mock hardware for camera'
    )

    # 1. Robot startup (robot_state_publisher + controllers)
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(robot_pkg, "launch", "robot.launch.py")),
        launch_arguments={
            'robot1_use_mock_hardware': LaunchConfiguration('robot1_use_mock_hardware'),
            'robot2_use_mock_hardware': LaunchConfiguration('robot2_use_mock_hardware'),
        }.items()
    )

    # 2. MoveIt startup (move_group)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_pkg, "launch", "moveit.launch.py")),
    )

    # 3. Gripper servers for both robots
    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gripper_pkg, "launch", "gripper.launch.py")),
        launch_arguments={
            'robot1_use_mock_hardware': LaunchConfiguration('robot1_use_mock_hardware'),
            'robot2_use_mock_hardware': LaunchConfiguration('robot2_use_mock_hardware'),
        }.items()
    )

    # 4. RViz visualization
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(bringup_pkg, "config", "moveit.rviz")],
    )

    # 5. Application nodes - grouped together, started after MoveIt is ready
    # This ensures MoveIt has had time to initialize and publish robot_description_semantic
    application_nodes = GroupAction([
        # MTC Planning (must start before cube_motion_server)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(mtc_planning_pkg, "launch", "mtc_planning.launch.py")
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(cube_motion_pkg, "launch", "cube_motion_server.launch.py")
            ),
            launch_arguments={
                'use_mock_hardware': LaunchConfiguration('robot1_use_mock_hardware'),
            }.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(cube_perception_pkg, "launch", "cube_perception.launch.py")
            ),
            launch_arguments={
                'use_mock_hardware': LaunchConfiguration('camera_use_mock_hardware'),
            }.items()
        ),
    ])

    # Start application nodes after a delay to ensure MoveIt is fully initialized
    # and publishing robot_description_semantic topic
    delayed_applications = TimerAction(
        period=5.0,  # Wait 5 seconds after RViz starts
        actions=[application_nodes],
    )

    # Start delayed applications after RViz process starts
    start_applications = RegisterEventHandler(
        OnProcessStart(
            target_action=rviz_node,
            on_start=[delayed_applications],
        )
    )

    return LaunchDescription([
        # Launch arguments
        robot1_mock_arg,
        robot2_mock_arg,
        camera_mock_arg,
        # Core system
        robot_launch,
        moveit_launch,
        gripper_launch,
        rviz_node,
        # Applications (event-triggered with delay)
        start_applications,
    ])
