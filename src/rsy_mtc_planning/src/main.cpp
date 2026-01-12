#include <rclcpp/rclcpp.hpp>
#include "rsy_mtc_planning/motion_sequence_server.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  // Allow undeclared parameters so MoveIt2's OMPL interface can access
  // planner configurations like ompl.planner_configs.RRTstarkConfigDefault.*
  // Note: Don't use automatically_declare_parameters_from_overrides as it conflicts
  // with explicit declare_parameter() calls in MotionSequenceServer
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);

  auto node = std::make_shared<rsy_mtc_planning::MotionSequenceServer>(options);

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
