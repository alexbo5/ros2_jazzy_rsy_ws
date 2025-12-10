import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    # Get package share directory
    pkg_share = get_package_share_directory("05_new_robot_startup")

    moveit_config = (
        MoveItConfigsBuilder("dual_ur", package_name="05_new_robot_startup")
        .robot_description(file_path="config/ur.urdf.xacro")
        .robot_description_semantic(file_path="config/ur.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
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

    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="both",
    )

    # Controller spawners (will be started after ros2_control_node)
    controllers = [
        "robot1_joint_trajectory_controller",
        "robot1_joint_state_broadcaster",
        "robot1_io_and_status_controller",
        "robot1_speed_scaling_state_broadcaster",
        "robot1_force_torque_sensor_broadcaster",
        "robot1_tcp_pose_broadcaster",
        "robot1_ur_configuration_controller",
        "robot2_joint_trajectory_controller",
        "robot2_joint_state_broadcaster",
        "robot2_io_and_status_controller",
        "robot2_speed_scaling_state_broadcaster",
        "robot2_force_torque_sensor_broadcaster",
        "robot2_tcp_pose_broadcaster",
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

    # Start rviz after move_group has started
    start_rviz = RegisterEventHandler(
        OnProcessStart(
            target_action=move_group_node,
            on_start=[
                TimerAction(
                    period=2.0,
                    actions=[rviz_node],
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
            start_rviz,
        ]
    )