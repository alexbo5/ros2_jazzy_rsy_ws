import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    # Get package share directory
    pkg_share = get_package_share_directory("rsy_robot_startup")

    # Custom joint limits for URDF (expanded limits)
    robot1_joint_limits = os.path.join(pkg_share, "config", "ur3_urdf_joint_limits.yaml")
    robot2_joint_limits = os.path.join(pkg_share, "config", "ur3e_urdf_joint_limits.yaml")

    moveit_config = (
        MoveItConfigsBuilder("ur", package_name="rsy_robot_startup")
        .robot_description(
            file_path="config/ur.urdf.xacro",
            mappings={
                "robot1_joint_limit_params": robot1_joint_limits,
                "robot2_joint_limit_params": robot2_joint_limits,
            },
        )
        .robot_description_semantic(file_path="config/ur.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    # Load joint limits manually and wrap in robot_description_planning namespace
    import yaml
    joint_limits_path = os.path.join(pkg_share, "config", "joint_limits.yaml")
    with open(joint_limits_path, 'r') as f:
        joint_limits_yaml = yaml.safe_load(f)

    # Wrap joint limits in the correct namespace for MoveIt
    joint_limits_params = {"robot_description_planning": {"joint_limits": joint_limits_yaml}}

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            joint_limits_params,  # Add joint limits explicitly
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    ros2_controllers_path = os.path.join(
        pkg_share,
        "config",
        "ros2_controllers.yaml",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="both",
        arguments=["--ros-args", "--log-level", "controller_manager:=error"],
    )

    # Controller spawners (will be started after ros2_control_node)
    controllers = [
        "joint_state_broadcaster",
        "robot1_joint_trajectory_controller",
        "robot1_io_and_status_controller",
        "robot1_speed_scaling_state_broadcaster",
        "robot1_force_torque_sensor_broadcaster",
        #"robot1_tcp_pose_broadcaster",
        "robot1_ur_configuration_controller",
        "robot2_joint_trajectory_controller",
        "robot2_io_and_status_controller",
        "robot2_speed_scaling_state_broadcaster",
        "robot2_force_torque_sensor_broadcaster",
        #"robot2_tcp_pose_broadcaster",
        "robot2_ur_configuration_controller",
    ]

    load_controllers = ExecuteProcess(
        cmd=["ros2", "run", "controller_manager", "spawner"] + controllers,
        output="screen",
    )

    # Start ros2_control_node after robot_state_publisher has started
    start_ros2_control = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[
                TimerAction(
                    period=2.0,
                    actions=[ros2_control_node],
                )
            ],
        )
    )

    # Start controller spawners after ros2_control_node has started
    start_controllers = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                TimerAction(
                    period=3.0,
                    actions=[load_controllers],
                )
            ],
        )
    )

    # Start move_group after robot_state_publisher has started
    start_move_group = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[
                TimerAction(
                    period=1.0,
                    actions=[move_group_node],
                )
            ],
        )
    )

    return LaunchDescription(
        [
            robot_state_publisher,
            start_ros2_control,
            start_controllers,
            start_move_group,
        ]
    )