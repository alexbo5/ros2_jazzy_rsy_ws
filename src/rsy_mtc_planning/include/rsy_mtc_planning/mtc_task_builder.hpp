#pragma once

#include <rclcpp/rclcpp.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/stages.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>

#include "rsy_mtc_planning/msg/motion_step.hpp"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace rsy_mtc_planning
{

/**
 * @brief Builder class for constructing MoveIt Task Constructor tasks
 * 
 * This class provides methods to build motion sequences from MotionStep messages,
 * supporting MoveJ (PTP), MoveL (Linear), and gripper operations.
 */
class MTCTaskBuilder
{
public:
  /**
   * @brief Construct a new MTCTaskBuilder
   * @param node Shared pointer to the ROS node for parameter access and logging
   */
  explicit MTCTaskBuilder(const rclcpp::Node::SharedPtr& node);

  /**
   * @brief Build a task from a sequence of motion steps
   * @param steps Vector of motion steps to execute
   * @param task_name Name for the task
   * @return Configured Task object
   */
  moveit::task_constructor::Task buildTask(
    const std::vector<rsy_mtc_planning::msg::MotionStep>& steps,
    const std::string& task_name = "motion_sequence");

  /**
   * @brief Plan the task
   * @param task The task to plan
   * @param max_solutions Maximum number of solutions to find
   * @return true if planning succeeded
   */
  bool planTask(moveit::task_constructor::Task& task, int max_solutions = 1);

  /**
   * @brief Execute a planned task
   * @param task The planned task to execute
   * @return true if execution succeeded
   */
  bool executeTask(moveit::task_constructor::Task& task);

  /**
   * @brief Execute a planned task using trajectory execution manager (bypasses buggy MTC execute)
   * @param task The planned task to execute
   * @return true if execution succeeded
   */
  bool executeTaskDirect(moveit::task_constructor::Task& task);

  /**
   * @brief Get the number of sub-trajectories in a planned task solution
   * @param task The planned task
   * @return Number of non-empty sub-trajectories
   */
  size_t getNumSubTrajectories(moveit::task_constructor::Task& task);

  /**
   * @brief Execute a single sub-trajectory by index
   * @param task The planned task
   * @param index The sub-trajectory index to execute
   * @return true if execution succeeded
   */
  bool executeSubTrajectory(moveit::task_constructor::Task& task, size_t index);

private:
  // Add a MoveJ (PTP) stage to the task
  void addMoveJStage(
    moveit::task_constructor::Task& task,
    const rsy_mtc_planning::msg::MotionStep& step,
    int step_index);

  // Add a MoveL (Linear) stage to the task
  void addMoveLStage(
    moveit::task_constructor::Task& task,
    const rsy_mtc_planning::msg::MotionStep& step,
    int step_index);

  // Add a gripper stage to the task
  void addGripperStage(
    moveit::task_constructor::Task& task,
    const rsy_mtc_planning::msg::MotionStep& step,
    int step_index);

  // Get the planning group for a robot
  std::string getPlanningGroup(const std::string& robot_name) const;

  // Get the end effector link for a robot
  std::string getEndEffectorLink(const std::string& robot_name) const;

  // Get the gripper group for a robot
  std::string getGripperGroup(const std::string& robot_name) const;

  // Node reference for logging and parameters
  rclcpp::Node::SharedPtr node_;

  // Robot model loader
  robot_model_loader::RobotModelLoaderPtr robot_model_loader_;

  // Robot model
  moveit::core::RobotModelPtr robot_model_;

  // Planning scene
  planning_scene::PlanningScenePtr planning_scene_;

  // Planners for different motion types
  std::shared_ptr<moveit::task_constructor::solvers::PipelinePlanner> sampling_planner_;
  std::shared_ptr<moveit::task_constructor::solvers::CartesianPath> cartesian_planner_;
  std::shared_ptr<moveit::task_constructor::solvers::JointInterpolationPlanner> joint_interpolation_planner_;

  // Configuration
  double velocity_scaling_ptp_ = 0.5;
  double velocity_scaling_lin_ = 0.3;
  double acceleration_scaling_ptp_ = 0.5;
  double acceleration_scaling_lin_ = 0.3;

  // Robot configurations
  std::unordered_map<std::string, std::string> planning_groups_;
  std::unordered_map<std::string, std::string> ee_links_;
  std::unordered_map<std::string, std::string> gripper_groups_;

  // Planning scene monitor for trajectory execution
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
};

}  // namespace rsy_mtc_planning
