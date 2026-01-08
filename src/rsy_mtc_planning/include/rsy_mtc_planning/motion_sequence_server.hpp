#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rsy_gripper_controller/srv/robotiq_gripper.hpp>

#include "rsy_mtc_planning/action/execute_motion_sequence.hpp"
#include "rsy_mtc_planning/msg/motion_step.hpp"
#include "rsy_mtc_planning/mtc_task_builder.hpp"
#include "rsy_mtc_planning/planning_logger.hpp"

namespace rsy_mtc_planning
{

class MotionSequenceServer : public rclcpp::Node
{
public:
  using ExecuteMotionSequence = rsy_mtc_planning::action::ExecuteMotionSequence;
  using GoalHandleExecuteMotionSequence = rclcpp_action::ServerGoalHandle<ExecuteMotionSequence>;
  using MotionStep = rsy_mtc_planning::msg::MotionStep;
  using RobotiqGripper = rsy_gripper_controller::srv::RobotiqGripper;

  explicit MotionSequenceServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~MotionSequenceServer();

private:
  // Action server callbacks
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID& uuid,
    std::shared_ptr<const ExecuteMotionSequence::Goal> goal);

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleExecuteMotionSequence> goal_handle);

  void handle_accepted(const std::shared_ptr<GoalHandleExecuteMotionSequence> goal_handle);

  // Process motion sequence in separate thread
  void process_motion_sequence(const std::shared_ptr<GoalHandleExecuteMotionSequence> goal_handle);

  // Execute individual motion step
  bool execute_motion_step(const MotionStep& step);

  // Execute gripper action
  bool execute_gripper_action(const std::string& robot_name, bool open);

  // Action server
  rclcpp_action::Server<ExecuteMotionSequence>::SharedPtr action_server_;

  // MTC Task Builder
  std::shared_ptr<MTCTaskBuilder> task_builder_;

  // Gripper service clients
  rclcpp::Client<RobotiqGripper>::SharedPtr robot1_gripper_client_;
  rclcpp::Client<RobotiqGripper>::SharedPtr robot2_gripper_client_;

  // Execution thread (joinable instead of detached for proper cleanup)
  std::thread execution_thread_;
  std::mutex thread_mutex_;

  // Planning logger for file-based detailed logging
  std::unique_ptr<PlanningLogger> logger_;
};

}  // namespace rsy_mtc_planning
