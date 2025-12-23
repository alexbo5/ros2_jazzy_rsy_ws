"""
RSY Bringup Launch File
Launches all components: robot, MoveIt, and application nodes.
Reads configuration from config/system_config.yaml.
"""

import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
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
    path_planning_pkg = get_package_share_directory("rsy_path_planning")
    cube_motion_pkg = get_package_share_directory("rsy_cube_motion")
    cube_perception_pkg = get_package_share_directory("rsy_cube_perception")
    gripper_pkg = get_package_share_directory("rsy_gripper_controller")

    # Load system configuration
    config_path = os.path.join(bringup_pkg, "config", "system_config.yaml")
    config = load_config(config_path)

    # Extract configuration values
    use_mock_hardware = config.get('use_mock_hardware', False)
    robot1_config = config.get('robot1', {})
    robot2_config = config.get('robot2', {})
    camera_config = config.get('camera', {})
    mock_config = config.get('mock', {})

    # Declare launch argument to override mock hardware from command line
    use_mock_hardware_arg = DeclareLaunchArgument(
        'use_mock_hardware',
        default_value=str(use_mock_hardware).lower(),
        description='Use mock hardware mode (overrides config file)'
    )

    # Extract robot2 position from config
    robot2_position = robot2_config.get('position', {})
    robot2_orientation = robot2_config.get('orientation', {})

    # 1. Robot startup (robot_state_publisher + controllers)
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(robot_pkg, "launch", "robot.launch.py")),
        launch_arguments={
            'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
            'robot1_robot_ip': robot1_config.get('robot_ip', '192.168.0.51'),
            'robot2_robot_ip': robot2_config.get('robot_ip', '192.168.0.11'),
            'robot2_x': str(robot2_position.get('x', -0.77)),
            'robot2_y': str(robot2_position.get('y', 0.655)),
            'robot2_z': str(robot2_position.get('z', 0.0)),
            'robot2_roll': str(robot2_orientation.get('roll', 0.0)),
            'robot2_pitch': str(robot2_orientation.get('pitch', 0.0)),
            'robot2_yaw': str(robot2_orientation.get('yaw', 3.141592)),
        }.items()
    )

    # 2. MoveIt startup (move_group)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_pkg, "launch", "moveit.launch.py")),
        launch_arguments={
            'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
        }.items()
    )

    # 3. Gripper servers for both robots
    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gripper_pkg, "launch", "gripper.launch.py")),
        launch_arguments={
            'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
            'robot1_gripper_ip': robot1_config.get('gripper_ip', '192.168.1.10'),
            'robot1_gripper_port': str(robot1_config.get('gripper_port', 63352)),
            'robot2_gripper_ip': robot2_config.get('gripper_ip', '192.168.1.11'),
            'robot2_gripper_port': str(robot2_config.get('gripper_port', 63352)),
        }.items()
    )

    # 4. RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(bringup_pkg, "config", "moveit.rviz")],
    )

    # 5. Application nodes (delayed to ensure MoveIt is ready)
    app_delay = 5.0

    path_planning_launch = TimerAction(
        period=app_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(path_planning_pkg, "launch", "path_planning.launch.py"))
        )]
    )

    cube_motion_launch = TimerAction(
        period=app_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(cube_motion_pkg, "launch", "cube_motion_server.launch.py"))
        )]
    )

    cube_perception_launch = TimerAction(
        period=app_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(cube_perception_pkg, "launch", "cube_perception.launch.py")),
            launch_arguments={
                'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
                'mock_cube_solution': mock_config.get('cube_solution', "R U R' U'"),
                'mock_cube_description': mock_config.get('cube_description', 'Mock cube solution for testing'),
                'camera_index': str(camera_config.get('index', 0)),
                'show_preview': str(camera_config.get('show_preview', True)).lower(),
            }.items()
        )]
    )

    return LaunchDescription([
        use_mock_hardware_arg,
        robot_launch,
        moveit_launch,
        gripper_launch,
        rviz_node,
        path_planning_launch,
        cube_motion_launch,
        cube_perception_launch,
    ])
