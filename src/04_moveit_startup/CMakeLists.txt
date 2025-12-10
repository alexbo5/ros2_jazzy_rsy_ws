cmake_minimum_required(VERSION 3.22)
project(04_moveit_startup)

find_package(ament_cmake REQUIRED)

ament_package()

install(
  DIRECTORY config launch srdf
  DESTINATION share/${PROJECT_NAME}
)
