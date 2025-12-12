from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rsy_cube_solver',
            executable='cube_solver_node',
            name='cube_solver',
            output='screen'
        )
    ])
