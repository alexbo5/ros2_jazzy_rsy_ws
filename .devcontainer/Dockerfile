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
    # Development tools
    git \
    python3-rosdep \
    python3-scipy \
    python3-colcon-common-extensions \
    python3-pip \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Workspace (will be mounted)
WORKDIR /root/ros2_ws

# Setup bashrc with conditional sourcing for custom packages
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "[ -f /root/ros2_ws/install/setup.bash ] && source /root/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /root/.bashrc
