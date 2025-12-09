from setuptools import setup, find_packages

package_name = 'path_planning_server'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alexander',
    maintainer_email='alexander@example.com',
    description='Path planning action server for dual UR robots using MoveIt2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_planning_action_server = path_planning_server.path_planning_action_server:main',
            'example_client = path_planning_server.example_client:main',
        ],
    },
)
