FROM osrf/ros:jazzy-desktop-full

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    # ROS2 Control
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    # UR Robot (includes ur_description, ur_robot_driver, ur_controllers, etc.)
    ros-jazzy-ur \
    # Additional controllers used in config
    ros-jazzy-velocity-controllers \
    ros-jazzy-effort-controllers \
    ros-jazzy-position-controllers \
    ros-jazzy-force-torque-sensor-broadcaster \
    ros-jazzy-pose-broadcaster \
    # URDF/Xacro processing
    ros-jazzy-xacro \
    # MoveIt2 for path planning
    ros-jazzy-moveit \
    ros-jazzy-moveit-ros-planning \
    ros-jazzy-moveit-ros-planning-interface \
    ros-jazzy-moveit-ros-move-group \
    ros-jazzy-moveit-planners \
    ros-jazzy-moveit-planners-ompl \
    ros-jazzy-moveit-kinematics \
    # TRAC-IK for robust inverse kinematics
    ros-jazzy-trac-ik-kinematics-plugin \
    ros-jazzy-moveit-simple-controller-manager \
    ros-jazzy-moveit-configs-utils \
    ros-jazzy-robotiq* \
    # Development tools
    git \
    python3-rosdep \
    python3-scipy \
    python3-numpy \
    python3-colcon-common-extensions \
    python3-pip \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --break-system-packages kociemba

# Workspace (will be mounted)
WORKDIR /root/ros2_ws

# Setup bashrc with conditional sourcing for custom packages
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "[ -f /root/ros2_ws/install/setup.bash ] && source /root/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /root/.bashrc && \
    echo "alias startup='cd /root/ros2_ws && colcon build && source install/setup.bash && ros2 launch rsy_bringup rsy_bringup.launch.py'" >> /root/.bashrc
