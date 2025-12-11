cmake_minimum_required(VERSION 3.8)
project(rsy_path_planning_action_server)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveJ.action"
  "action/MoveL.action"
  DEPENDENCIES std_msgs action_msgs
)

install(PROGRAMS
  rsy_path_planning_action_server/path_planning_action_server.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
