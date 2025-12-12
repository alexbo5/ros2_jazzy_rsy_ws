from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
        
    action_server_node = Node(
        package='rsy_cube_perception',
        executable='scan_cube_action_server',
        name='cube_perception',
        output='screen',
    )
    
    return LaunchDescription([
        action_server_node,
    ])

if __name__ == "__main__":
    generate_launch_description()