cmake_minimum_required(VERSION 3.8)
project(rsy_mtc_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(moveit_task_constructor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rsy_gripper_controller REQUIRED)
find_package(tf2_eigen REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotionStep.msg"
  "action/ExecuteMotionSequence.action"
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

# Get typesupport target for generated interfaces
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

# Include directories
include_directories(include)

# MTC Motion Sequence Server executable
add_executable(mtc_motion_sequence_server
  src/main.cpp
  src/motion_sequence_server.cpp
  src/mtc_task_builder.cpp
)

ament_target_dependencies(mtc_motion_sequence_server
  rclcpp
  rclcpp_action
  std_msgs
  geometry_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
  rsy_gripper_controller
  tf2_eigen
)

target_link_libraries(mtc_motion_sequence_server "${cpp_typesupport_target}")

# Install executables
install(TARGETS
  mtc_motion_sequence_server
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

ament_package()
