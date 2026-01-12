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
#include <map>
#include <unordered_map>
#include <optional>
#include <moveit_task_constructor_msgs/msg/solution.hpp>

namespace rsy_mtc_planning
{

/**
 * @brief Configuration for planner parameters
 *
 * Values are set in motion_sequence_server.cpp constructor.
 * Defaults here are fallbacks only.
 */
struct PlannerConfig
{
  double velocity_scaling_ptp = 1.0;
  double velocity_scaling_lin = 1.0;
  double acceleration_scaling_ptp = 1.0;
  double acceleration_scaling_lin = 1.0;
  double timeout_ompl = 5.0;
  double timeout_pilz_ptp = 10.0;
  double timeout_pilz_lin = 10.0;
  std::string ompl_planner_id = "RRTConnect";
};

/**
 * @brief Represents an IK solution for a MoveJ step
 */
struct IKSolution
{
  std::vector<double> joint_values;
  std::string planning_group;
};

/**
 * @brief Represents IK solutions for a MoveJ step with metadata
 */
struct MoveJIKData
{
  size_t step_index;                      // Index in original motion steps
  std::string robot_name;
  geometry_msgs::msg::PoseStamped target_pose;
  std::vector<IKSolution> ik_solutions;   // All valid IK solutions for this pose
};

/**
 * @brief Result of a planning attempt with failure information
 */
struct PlanningResult
{
  bool success = false;
  std::string failed_stage_name;          // Name of the stage that failed (e.g., "move_j_0", "move_l_2")
  int failed_stage_index = -1;            // Index in the motion sequence (-1 if unknown)
  std::string robot_name;                 // Robot involved in the failed stage
  bool is_movel = false;                  // True if the failed stage was a MoveL
  std::string error_message;
};

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
   * @param config Planner configuration (optional, uses defaults if not provided)
   */
  explicit MTCTaskBuilder(const rclcpp::Node::SharedPtr& node, const PlannerConfig& config = PlannerConfig());

  /**
   * @brief Build a task from a sequence of motion steps
   * @param steps Vector of motion steps to execute
   * @param task_name Name for the task
   * @param use_ompl_fallback If true, use OMPL for MoveJ instead of Pilz PTP
   * @return Configured Task object
   */
  moveit::task_constructor::Task buildTask(
    const std::vector<rsy_mtc_planning::msg::MotionStep>& steps,
    const std::string& task_name = "motion_sequence",
    bool use_ompl_fallback = false);

  /**
   * @brief Plan the task
   * @param task The task to plan
   * @param max_solutions Maximum number of solutions to find
   * @return true if planning succeeded
   */
  bool planTask(moveit::task_constructor::Task& task, int max_solutions = 1);

  /**
   * @brief Plan the task with detailed failure information
   * @param task The task to plan
   * @param max_solutions Maximum number of solutions to find
   * @return PlanningResult with success status and failure details
   */
  PlanningResult planTaskWithResult(moveit::task_constructor::Task& task, int max_solutions = 1);

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

  /**
   * @brief Compute multiple IK solutions for a target pose
   * @param robot_name The robot name ("robot1" or "robot2")
   * @param target_pose Target pose for the end effector
   * @param max_solutions Maximum number of IK solutions to compute
   * @return Vector of IK solutions (may be empty if no valid IK found)
   */
  std::vector<IKSolution> computeIKSolutions(
    const std::string& robot_name,
    const geometry_msgs::msg::PoseStamped& target_pose,
    int max_solutions = 8);

  /**
   * @brief Build a task with specific IK configurations for MoveJ steps
   * @param steps Vector of motion steps
   * @param ik_indices Map from step index to IK solution index (for MoveJ steps)
   * @param movej_ik_data Pre-computed IK data for MoveJ steps
   * @param task_name Name for the task
   * @param use_sampling_planner If true, use OMPL for MoveJ (obstacle avoidance); false uses Pilz PTP (minimal motion)
   * @return Configured Task object
   */
  moveit::task_constructor::Task buildTaskWithIK(
    const std::vector<rsy_mtc_planning::msg::MotionStep>& steps,
    const std::map<size_t, size_t>& ik_indices,
    const std::vector<MoveJIKData>& movej_ik_data,
    const std::string& task_name = "motion_sequence",
    bool use_sampling_planner = false);

  /**
   * @brief Validate that an IK combination is collision-free for the entire sequence
   *
   * This function simulates the motion sequence step by step, checking for collisions
   * at each endpoint. It considers the positions of ALL robots at each step, not just
   * the current robot's position. This is critical for dual-arm setups where one robot's
   * movement might collide with the other robot's planned position.
   *
   * @param steps Vector of motion steps
   * @param ik_indices Map from step index to IK solution index (for MoveJ steps)
   * @param movej_ik_data Pre-computed IK data for MoveJ steps
   * @param collision_step_index Output: step index where collision was detected (-1 if none)
   * @return true if the combination is collision-free, false if collision detected
   */
  bool validateIKCombinationCollisions(
    const std::vector<rsy_mtc_planning::msg::MotionStep>& steps,
    const std::map<size_t, size_t>& ik_indices,
    const std::vector<MoveJIKData>& movej_ik_data,
    int& collision_step_index);

  /**
   * @brief Get the robot model for IK computation
   */
  moveit::core::RobotModelPtr getRobotModel() const { return robot_model_; }

  /**
   * @brief Update planning scene from current state
   */
  void updatePlanningScene();

  /**
   * @brief Cache solution message for a task (call before executeSubTrajectory loop)
   * @param task The planned task
   * @return true if caching succeeded
   */
  bool cacheSolutionMessage(moveit::task_constructor::Task& task);

  /**
   * @brief Clear cached solution message
   */
  void clearSolutionCache();

  /**
   * @brief Get number of sub-trajectories from cached solution (faster than getNumSubTrajectories)
   * @return Number of non-empty sub-trajectories, or 0 if not cached
   */
  size_t getCachedNumSubTrajectories() const;

  /**
   * @brief Execute a sub-trajectory using cached solution (faster than executeSubTrajectory)
   * @param index The sub-trajectory index to execute
   * @return true if execution succeeded
   */
  bool executeCachedSubTrajectory(size_t index);

private:
  // Add a MoveJ (PTP) stage to the task with pose goal
  void addMoveJStage(
    moveit::task_constructor::Task& task,
    const rsy_mtc_planning::msg::MotionStep& step,
    int step_index,
    bool use_ompl = false);

  // Add a MoveJ (PTP) stage with joint-space goal (specific IK solution)
  // use_sampling_planner: if true, use OMPL (for obstacle avoidance); if false, use Pilz PTP (minimal motion)
  void addMoveJStageWithJoints(
    moveit::task_constructor::Task& task,
    const std::string& robot_name,
    const std::vector<double>& joint_values,
    int step_index,
    bool use_sampling_planner = false);

  // Add a MoveL (Linear) stage to the task
  void addMoveLStage(
    moveit::task_constructor::Task& task,
    const rsy_mtc_planning::msg::MotionStep& step,
    int step_index);

  // Get the planning group for a robot
  std::string getPlanningGroup(const std::string& robot_name) const;

  // Get the end effector link for a robot
  std::string getEndEffectorLink(const std::string& robot_name) const;

  // Declare hardcoded OMPL parameters on the node
  // This bypasses YAML config files and ensures MoveIt can find the planner configs
  void declareOmplParameters(const std::string& default_planner);

  // Node reference for logging and parameters
  rclcpp::Node::SharedPtr node_;

  // Robot model loader
  robot_model_loader::RobotModelLoaderPtr robot_model_loader_;

  // Robot model
  moveit::core::RobotModelPtr robot_model_;

  // Planning scene
  planning_scene::PlanningScenePtr planning_scene_;

  // Planners for different motion types
  std::shared_ptr<moveit::task_constructor::solvers::PipelinePlanner> sampling_planner_;      // OMPL (configurable)
  std::shared_ptr<moveit::task_constructor::solvers::PipelinePlanner> ptp_planner_;           // Pilz PTP (minimal joint motion)
  std::shared_ptr<moveit::task_constructor::solvers::PipelinePlanner> lin_planner_;           // Pilz LIN (Cartesian linear)

  // Stored planner configuration (for stage timeouts)
  PlannerConfig config_;

  // Robot configurations
  std::unordered_map<std::string, std::string> planning_groups_;
  std::unordered_map<std::string, std::string> ee_links_;

  // Planning scene monitor for trajectory execution
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;

  // ===== CACHED OBJECTS FOR PERFORMANCE =====

  // Cached MoveGroupInterface per planning group (created on first use, reused)
  mutable std::unordered_map<std::string, std::shared_ptr<moveit::planning_interface::MoveGroupInterface>> move_group_cache_;

  // Cached solution message for trajectory execution (avoids repeated toMsg() calls)
  std::optional<moveit_task_constructor_msgs::msg::Solution> cached_solution_msg_;

  // Cached non-empty trajectory indices for fast lookup
  std::vector<size_t> cached_trajectory_indices_;

  // Helper to get or create MoveGroupInterface for a planning group
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> getMoveGroup(const std::string& group_name) const;
};

}  // namespace rsy_mtc_planning
