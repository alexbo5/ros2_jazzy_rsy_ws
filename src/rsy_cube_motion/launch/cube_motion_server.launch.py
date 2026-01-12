from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_mock_hardware_arg = DeclareLaunchArgument(
        'use_mock_hardware',
        default_value='false',
        description='Use mock hardware for cube motion'
    )

    return LaunchDescription([
        use_mock_hardware_arg,
        Node(
            package='rsy_cube_motion',
            executable='cube_motion_server',
            name='cube_motion',
            output='screen',
            parameters=[{
                'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
            }]
        )
    ])
