from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rsy_cube_motion',
            executable='cube_motion_server',
            name='cube_motion',
            output='screen'
        )
    ])
