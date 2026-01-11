#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <atomic>
#include <condition_variable>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>
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

  // Robot description subscription (subscribes to /robot_description topic from robot_state_publisher)
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_sub_;
  std::atomic<bool> robot_description_received_{false};
  std::mutex robot_description_mutex_;
  std::condition_variable robot_description_cv_;

  // Callback for robot_description topic
  void robot_description_callback(const std_msgs::msg::String::SharedPtr msg);

  // Wait for robot_description to be received
  bool wait_for_robot_description(double timeout_sec = 30.0);

  // Declare OMPL parameters on the node for MoveIt to find
  void declareOmplParameters();

  // Planner configuration (loaded once at startup)
  PlannerConfig planner_config_;

  // Configurable parameters
  int max_ik_per_step_{16};
  size_t max_pilz_combinations_{1};
  size_t max_ompl_combinations_{256};
  double robot_description_timeout_{30.0};

  // Planning logger for performance analysis
  std::unique_ptr<PlanningLogger> planning_logger_;
};

}  // namespace rsy_mtc_planning
