#include <rclcpp/rclcpp.hpp>
#include "rsy_mtc_planning/motion_sequence_server.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rsy_mtc_planning::MotionSequenceServer>();

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
