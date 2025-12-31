cmake_minimum_required(VERSION 3.8)
project(rsy_cube_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ScanCubeFace.action"
  "action/CalibrateCamera.action"
  "srv/SolveCube.srv"
)

# Install the Python action server script so ros2 launch can run it by package/executable name.
# The script lives at src/rsy_cube_perception/rsy_cube_perception/scan_cube_action_server.py in this workspace.
# We install it into lib/<package>/scan_cube_action_server (no .py suffix) so your launch file can use
# executable='scan_cube_action_server'.
install(PROGRAMS
  rsy_cube_perception/scan_cube_action_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME scan_cube_action_server
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
