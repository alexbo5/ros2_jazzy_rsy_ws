cmake_minimum_required(VERSION 3.8)
project(path_planning_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Generate action interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveL.action"
  "action/MoveJ.action"
  DEPENDENCIES geometry_msgs trajectory_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
