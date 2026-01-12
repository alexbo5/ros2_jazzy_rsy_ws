// Copyright (c) 2023, ROS2 Contributors
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file mtc_task_builder.cpp
 * @brief Implementation of MTCTaskBuilder for building MoveIt Task Constructor tasks
 */

#include "rsy_mtc_planning/mtc_task_builder.hpp"

#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <set>
#include <algorithm>

namespace rsy_mtc_planning
{

namespace mtc = moveit::task_constructor;

MTCTaskBuilder::MTCTaskBuilder(const rclcpp::Node::SharedPtr& node, const PlannerConfig& config)
  : node_(node), config_(config)
{
  // NOTE: OMPL parameters are declared in MotionSequenceServer::declareOmplParameters()
  // at node startup, before this constructor is called

  // Load robot model
  robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(node_, "robot_description");
  robot_model_ = robot_model_loader_->getModel();

  if (!robot_model_)
  {
    RCLCPP_ERROR(node_->get_logger(), "Failed to load robot model");
    throw std::runtime_error("Failed to load robot model");
  }

  // Create planning scene monitor for execution and MTC access
  planning_scene_monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
    node_, robot_model_loader_, "planning_scene_monitor");

  // Start monitoring robot state (requires /joint_states topic)
  // Limit update frequency to reduce overhead during heavy planning loads
  planning_scene_monitor_->startStateMonitor("/joint_states");
  planning_scene_monitor_->setStateUpdateFrequency(20.0);  // 20 Hz max (default can be much higher)

  // Start publishing the planning scene so MTC's CurrentState stage can access it
  // This publishes to /monitored_planning_scene topic
  planning_scene_monitor_->startPublishingPlanningScene(
    planning_scene_monitor::PlanningSceneMonitor::UPDATE_SCENE,
    "/monitored_planning_scene");

  // Start scene and world geometry monitors
  planning_scene_monitor_->startSceneMonitor("/planning_scene");
  planning_scene_monitor_->startWorldGeometryMonitor();

  // Note: We intentionally do NOT call providePlanningSceneService() here.
  // The service is not needed for MTC planning (it uses the monitor directly),
  // and providing it can cause timeout warnings under heavy load during IK computation.
  // If external tools need /get_planning_scene, enable it in a separate node.

  // Create planning scene
  planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_);

  // Create OMPL planner for sampling-based planning
  sampling_planner_ = std::make_shared<mtc::solvers::PipelinePlanner>(node_, "ompl", config.ompl_planner_id);
  sampling_planner_->setTimeout(config.timeout_ompl);
  sampling_planner_->setMaxVelocityScalingFactor(config.velocity_scaling_ptp);
  sampling_planner_->setMaxAccelerationScalingFactor(config.acceleration_scaling_ptp);
  sampling_planner_->setProperty("num_planning_attempts", config.ompl_planning_attempts);

  RCLCPP_INFO(node_->get_logger(), "OMPL planner created: id=%s, timeout=%.1fs, attempts=%u",
              config.ompl_planner_id.c_str(), config.timeout_ompl, config.ompl_planning_attempts);

  // Create Pilz PTP planner for point-to-point motions
  ptp_planner_ = std::make_shared<mtc::solvers::PipelinePlanner>(node_, "pilz_industrial_motion_planner", "PTP");
  ptp_planner_->setTimeout(config.timeout_pilz_ptp);
  ptp_planner_->setMaxVelocityScalingFactor(config.velocity_scaling_ptp);
  ptp_planner_->setMaxAccelerationScalingFactor(config.acceleration_scaling_ptp);

  // Create Pilz LIN planner for linear Cartesian motions
  lin_planner_ = std::make_shared<mtc::solvers::PipelinePlanner>(node_, "pilz_industrial_motion_planner", "LIN");
  lin_planner_->setTimeout(config.timeout_pilz_lin);
  lin_planner_->setMaxVelocityScalingFactor(config.velocity_scaling_lin);
  lin_planner_->setMaxAccelerationScalingFactor(config.acceleration_scaling_lin);

  RCLCPP_INFO(node_->get_logger(), "MTCTaskBuilder: OMPL=%s (%.1fs), Pilz PTP (%.1fs), Pilz LIN (%.1fs)",
              config.ompl_planner_id.c_str(), config.timeout_ompl, config.timeout_pilz_ptp, config.timeout_pilz_lin);

  // Setup robot configurations
  planning_groups_["robot1"] = "robot1_ur_manipulator";
  planning_groups_["robot2"] = "robot2_ur_manipulator";
  ee_links_["robot1"] = "robot1_tcp";
  ee_links_["robot2"] = "robot2_tcp";
}

std::shared_ptr<moveit::planning_interface::MoveGroupInterface> MTCTaskBuilder::getMoveGroup(
  const std::string& group_name) const
{
  // Check cache first
  auto it = move_group_cache_.find(group_name);
  if (it != move_group_cache_.end())
  {
    return it->second;
  }

  // Create and cache new MoveGroupInterface
  RCLCPP_DEBUG(node_->get_logger(), "Creating MoveGroupInterface for group: %s", group_name.c_str());
  auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, group_name);
  move_group_cache_[group_name] = move_group;
  return move_group;
}

mtc::Task MTCTaskBuilder::buildTask(
  const std::vector<rsy_mtc_planning::msg::MotionStep>& steps,
  const std::string& task_name,
  bool use_ompl_fallback)
{
  mtc::Task task;
  task.setName(task_name);  // Set the task name on the task object
  task.stages()->setName(task_name);
  task.setRobotModel(robot_model_);  // Reuse already-loaded robot model to avoid duplicate publishers

  // Enable introspection for better debugging
  task.enableIntrospection();

  // Add current state as the starting point
  auto current_state = std::make_unique<mtc::stages::CurrentState>("current_state");
  task.add(std::move(current_state));

  // Add stages for each motion step
  for (size_t i = 0; i < steps.size(); ++i)
  {
    const auto& step = steps[i];

    switch (step.motion_type)
    {
      case rsy_mtc_planning::msg::MotionStep::MOVE_J:
        addMoveJStage(task, step, static_cast<int>(i), use_ompl_fallback);
        break;

      case rsy_mtc_planning::msg::MotionStep::MOVE_L:
        addMoveLStage(task, step, static_cast<int>(i));
        break;

      default:
        RCLCPP_WARN(node_->get_logger(), "Unknown motion type %d at step %zu", step.motion_type, i);
        break;
    }
  }

  return task;
}

bool MTCTaskBuilder::planTask(mtc::Task& task, int max_solutions)
{
  try
  {
    task.init();

    if (!task.plan(max_solutions))
    {
      return false;
    }

    return true;
  }
  catch (const std::exception& e)
  {
    RCLCPP_DEBUG(node_->get_logger(), "Planning failed: %s", e.what());
    return false;
  }
}

PlanningResult MTCTaskBuilder::planTaskWithResult(mtc::Task& task, int max_solutions)
{
  PlanningResult result;

  try
  {
    task.init();

    if (!task.plan(max_solutions))
    {
      // Planning failed - try to identify which stage failed
      result.success = false;

      // Iterate through stages to find the one that failed
      const auto* stages = task.stages();
      if (stages)
      {
        for (size_t i = 0; i < stages->numChildren(); ++i)
        {
          const auto* stage = (*stages)[i];
          if (stage && stage->solutions().empty() && !stage->failures().empty())
          {
            result.failed_stage_name = stage->name();

            // Parse stage name to extract info (format: "move_j_X" or "move_l_X")
            if (result.failed_stage_name.find("move_l_") == 0)
            {
              result.is_movel = true;
              try {
                result.failed_stage_index = std::stoi(result.failed_stage_name.substr(7));
              } catch (...) {
                result.failed_stage_index = -1;
              }
            }
            else if (result.failed_stage_name.find("move_j_") == 0)
            {
              result.is_movel = false;
              try {
                result.failed_stage_index = std::stoi(result.failed_stage_name.substr(7));
              } catch (...) {
                result.failed_stage_index = -1;
              }
            }

            // Extract robot name from planning group if available
            // Stage names don't include robot, so we need to check the group
            // For now, we'll extract it from the failed stage's properties if possible
            result.error_message = "Stage '" + result.failed_stage_name + "' failed to find solution";

            RCLCPP_DEBUG(node_->get_logger(), "Planning failed at stage '%s' (index %d, is_movel=%d)",
                        result.failed_stage_name.c_str(), result.failed_stage_index, result.is_movel);
            break;
          }
        }
      }

      if (result.failed_stage_name.empty())
      {
        result.error_message = "Planning failed (unknown stage)";
      }

      return result;
    }

    result.success = true;
    return result;
  }
  catch (const std::exception& e)
  {
    result.success = false;
    result.error_message = e.what();
    RCLCPP_DEBUG(node_->get_logger(), "Planning exception: %s", e.what());
    return result;
  }
}

bool MTCTaskBuilder::executeTask(mtc::Task& task)
{
  if (task.numSolutions() == 0)
  {
    RCLCPP_ERROR(node_->get_logger(), "No solutions to execute");
    return false;
  }

  try
  {
    // Get the first solution
    const auto& solution = task.solutions().front();

    RCLCPP_DEBUG(node_->get_logger(), "Executing task '%s'", task.name().c_str());

    // Execute the solution using MTC's execute
    auto result = task.execute(*solution);
    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Task execution failed: code=%d", result.val);
      return false;
    }

    RCLCPP_DEBUG(node_->get_logger(), "Task execution succeeded");
    return true;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "Task execution exception: %s", e.what());
    return false;
  }
}

bool MTCTaskBuilder::executeTaskDirect(mtc::Task& task)
{
  if (task.numSolutions() == 0)
  {
    RCLCPP_ERROR(node_->get_logger(), "No solutions to execute");
    return false;
  }

  try
  {
    // Get the first solution
    const auto& solution = task.solutions().front();

    RCLCPP_DEBUG(node_->get_logger(), "Executing task '%s' direct", task.name().c_str());

    // Get the trajectory from the solution - use toMsg() to get the full trajectory
    moveit_task_constructor_msgs::msg::Solution solution_msg;
    solution->toMsg(solution_msg, nullptr);

    // Execute each sub-trajectory using MoveGroupInterface
    for (const auto& sub_traj : solution_msg.sub_trajectory)
    {
      if (sub_traj.trajectory.joint_trajectory.points.empty() &&
          sub_traj.trajectory.multi_dof_joint_trajectory.points.empty())
      {
        continue;  // Skip empty sub-trajectories silently
      }

      // Determine which planning group this trajectory is for
      std::string group_name;
      if (!sub_traj.trajectory.joint_trajectory.joint_names.empty())
      {
        const auto& first_joint = sub_traj.trajectory.joint_trajectory.joint_names[0];
        if (first_joint.find("robot1") != std::string::npos)
        {
          group_name = "robot1_ur_manipulator";
        }
        else if (first_joint.find("robot2") != std::string::npos)
        {
          group_name = "robot2_ur_manipulator";
        }
        else
        {
          RCLCPP_ERROR(node_->get_logger(), "Unknown planning group for joint: %s", first_joint.c_str());
          return false;
        }
      }
      else
      {
        continue;  // Skip trajectories without joint names
      }

      RCLCPP_DEBUG(node_->get_logger(), "Executing sub-trajectory: %s (%zu pts)",
                  group_name.c_str(), sub_traj.trajectory.joint_trajectory.points.size());

      // Get cached MoveGroupInterface for this group (avoids expensive recreation)
      auto move_group = getMoveGroup(group_name);

      // Execute the trajectory directly using the message
      auto result = move_group->execute(sub_traj.trajectory);

      if (result != moveit::core::MoveItErrorCode::SUCCESS)
      {
        RCLCPP_ERROR(node_->get_logger(), "Sub-trajectory failed: code=%d", result.val);
        return false;
      }
    }

    RCLCPP_DEBUG(node_->get_logger(), "Direct execution succeeded");
    return true;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "Direct execution exception: %s", e.what());
    return false;
  }
}

size_t MTCTaskBuilder::getNumSubTrajectories(mtc::Task& task)
{
  if (task.numSolutions() == 0)
  {
    RCLCPP_WARN(node_->get_logger(), "getNumSubTrajectories: No solutions in task");
    return 0;
  }

  const auto& solution = task.solutions().front();
  moveit_task_constructor_msgs::msg::Solution solution_msg;
  solution->toMsg(solution_msg, nullptr);

  size_t count = 0;
  size_t total_sub_traj = solution_msg.sub_trajectory.size();

  for (const auto& sub_traj : solution_msg.sub_trajectory)
  {
    if (!sub_traj.trajectory.joint_trajectory.points.empty() ||
        !sub_traj.trajectory.multi_dof_joint_trajectory.points.empty())
    {
      count++;
    }
  }

  RCLCPP_INFO(node_->get_logger(), "Found %zu non-empty trajectories out of %zu total sub-trajectories",
              count, total_sub_traj);
  return count;
}

bool MTCTaskBuilder::executeSubTrajectory(mtc::Task& task, size_t index)
{
  if (task.numSolutions() == 0)
  {
    RCLCPP_ERROR(node_->get_logger(), "No solutions to execute");
    return false;
  }

  try
  {
    const auto& solution = task.solutions().front();
    moveit_task_constructor_msgs::msg::Solution solution_msg;
    solution->toMsg(solution_msg, nullptr);

    // Find the index-th non-empty sub-trajectory
    size_t current_idx = 0;
    for (const auto& sub_traj : solution_msg.sub_trajectory)
    {
      if (sub_traj.trajectory.joint_trajectory.points.empty() &&
          sub_traj.trajectory.multi_dof_joint_trajectory.points.empty())
      {
        continue;
      }

      if (current_idx == index)
      {
        // Execute this sub-trajectory
        std::string group_name;
        if (!sub_traj.trajectory.joint_trajectory.joint_names.empty())
        {
          const auto& first_joint = sub_traj.trajectory.joint_trajectory.joint_names[0];
          if (first_joint.find("robot1") != std::string::npos)
          {
            group_name = "robot1_ur_manipulator";
          }
          else if (first_joint.find("robot2") != std::string::npos)
          {
            group_name = "robot2_ur_manipulator";
          }
          else
          {
            RCLCPP_ERROR(node_->get_logger(), "Unknown group for joint: %s", first_joint.c_str());
            return false;
          }
        }
        else
        {
          return true;  // Skip empty trajectory
        }

        RCLCPP_INFO(node_->get_logger(), "Executing trajectory %zu: group=%s, points=%zu",
                    index, group_name.c_str(), sub_traj.trajectory.joint_trajectory.points.size());

        // Get cached MoveGroupInterface (avoids expensive recreation per trajectory)
        auto move_group = getMoveGroup(group_name);

        RCLCPP_INFO(node_->get_logger(), "Sending trajectory to move_group...");
        auto result = move_group->execute(sub_traj.trajectory);

        if (result != moveit::core::MoveItErrorCode::SUCCESS)
        {
          RCLCPP_ERROR(node_->get_logger(), "Trajectory %zu execution failed: code=%d", index, result.val);
          return false;
        }

        RCLCPP_INFO(node_->get_logger(), "Trajectory %zu executed successfully", index);
        return true;
      }
      current_idx++;
    }

    RCLCPP_ERROR(node_->get_logger(), "Trajectory index %zu out of range", index);
    return false;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "Trajectory execution exception: %s", e.what());
    return false;
  }
}

bool MTCTaskBuilder::cacheSolutionMessage(mtc::Task& task)
{
  if (task.numSolutions() == 0)
  {
    RCLCPP_WARN(node_->get_logger(), "cacheSolutionMessage: No solutions in task");
    return false;
  }

  try
  {
    const auto& solution = task.solutions().front();
    cached_solution_msg_ = moveit_task_constructor_msgs::msg::Solution();
    solution->toMsg(*cached_solution_msg_, nullptr);

    // Pre-compute indices of non-empty trajectories for fast lookup
    cached_trajectory_indices_.clear();
    for (size_t i = 0; i < cached_solution_msg_->sub_trajectory.size(); ++i)
    {
      const auto& sub_traj = cached_solution_msg_->sub_trajectory[i];
      if (!sub_traj.trajectory.joint_trajectory.points.empty() ||
          !sub_traj.trajectory.multi_dof_joint_trajectory.points.empty())
      {
        cached_trajectory_indices_.push_back(i);
      }
    }

    RCLCPP_DEBUG(node_->get_logger(), "Cached solution: %zu non-empty trajectories out of %zu total",
                cached_trajectory_indices_.size(), cached_solution_msg_->sub_trajectory.size());
    return true;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "Failed to cache solution: %s", e.what());
    cached_solution_msg_.reset();
    cached_trajectory_indices_.clear();
    return false;
  }
}

void MTCTaskBuilder::clearSolutionCache()
{
  cached_solution_msg_.reset();
  cached_trajectory_indices_.clear();
}

size_t MTCTaskBuilder::getCachedNumSubTrajectories() const
{
  return cached_trajectory_indices_.size();
}

bool MTCTaskBuilder::executeCachedSubTrajectory(size_t index)
{
  if (!cached_solution_msg_)
  {
    RCLCPP_ERROR(node_->get_logger(), "No cached solution - call cacheSolutionMessage first");
    return false;
  }

  if (index >= cached_trajectory_indices_.size())
  {
    RCLCPP_ERROR(node_->get_logger(), "Cached trajectory index %zu out of range (max %zu)",
                index, cached_trajectory_indices_.size());
    return false;
  }

  try
  {
    // Get the actual sub-trajectory using pre-computed index
    size_t actual_idx = cached_trajectory_indices_[index];
    const auto& sub_traj = cached_solution_msg_->sub_trajectory[actual_idx];

    // Determine planning group from joint names
    std::string group_name;
    if (!sub_traj.trajectory.joint_trajectory.joint_names.empty())
    {
      const auto& first_joint = sub_traj.trajectory.joint_trajectory.joint_names[0];
      if (first_joint.find("robot1") != std::string::npos)
      {
        group_name = "robot1_ur_manipulator";
      }
      else if (first_joint.find("robot2") != std::string::npos)
      {
        group_name = "robot2_ur_manipulator";
      }
      else
      {
        RCLCPP_ERROR(node_->get_logger(), "Unknown group for joint: %s", first_joint.c_str());
        return false;
      }
    }
    else
    {
      return true;  // Skip empty trajectory (shouldn't happen due to pre-filtering)
    }

    RCLCPP_INFO(node_->get_logger(), "Executing cached trajectory %zu: group=%s, points=%zu",
                index, group_name.c_str(), sub_traj.trajectory.joint_trajectory.points.size());

    // Get cached MoveGroupInterface
    auto move_group = getMoveGroup(group_name);

    auto result = move_group->execute(sub_traj.trajectory);

    if (result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Cached trajectory %zu execution failed: code=%d", index, result.val);
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Cached trajectory %zu executed successfully", index);
    return true;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(node_->get_logger(), "Cached trajectory execution exception: %s", e.what());
    return false;
  }
}

void MTCTaskBuilder::addMoveJStage(
  mtc::Task& task,
  const rsy_mtc_planning::msg::MotionStep& step,
  int step_index,
  bool use_ompl)
{
  std::string stage_name = "move_j_" + std::to_string(step_index);
  std::string group = getPlanningGroup(step.robot_name);
  std::string ee_link = getEndEffectorLink(step.robot_name);

  geometry_msgs::msg::PoseStamped target = step.target_pose;
  if (target.header.frame_id.empty())
  {
    target.header.frame_id = "world";
  }

  auto& planner = use_ompl ? sampling_planner_ : ptp_planner_;
  double timeout = use_ompl ? config_.timeout_ompl : config_.timeout_pilz_ptp;

  auto stage = std::make_unique<mtc::stages::MoveTo>(stage_name, planner);
  stage->setGroup(group);
  stage->setGoal(target);
  stage->setIKFrame(ee_link);
  stage->setTimeout(timeout);

  task.add(std::move(stage));
}

void MTCTaskBuilder::addMoveLStage(
  mtc::Task& task,
  const rsy_mtc_planning::msg::MotionStep& step,
  int step_index)
{
  std::string stage_name = "move_l_" + std::to_string(step_index);
  std::string group = getPlanningGroup(step.robot_name);
  std::string ee_link = getEndEffectorLink(step.robot_name);

  auto stage = std::make_unique<mtc::stages::MoveTo>(stage_name, lin_planner_);
  stage->setGroup(group);
  stage->setTimeout(config_.timeout_pilz_lin);

  geometry_msgs::msg::PoseStamped target = step.target_pose;
  if (target.header.frame_id.empty())
  {
    target.header.frame_id = "world";
  }

  stage->setGoal(target);
  stage->setIKFrame(ee_link);

  task.add(std::move(stage));
}

std::string MTCTaskBuilder::getPlanningGroup(const std::string& robot_name) const
{
  auto it = planning_groups_.find(robot_name);
  if (it != planning_groups_.end())
  {
    return it->second;
  }
  RCLCPP_WARN(node_->get_logger(), "Unknown robot '%s', using default group", robot_name.c_str());
  return robot_name + "_ur_manipulator";
}

std::string MTCTaskBuilder::getEndEffectorLink(const std::string& robot_name) const
{
  auto it = ee_links_.find(robot_name);
  if (it != ee_links_.end())
  {
    return it->second;
  }
  RCLCPP_WARN(node_->get_logger(), "Unknown robot '%s', using default ee_link", robot_name.c_str());
  return robot_name + "_tcp";
}

void MTCTaskBuilder::updatePlanningScene()
{
  planning_scene_monitor_->requestPlanningSceneState();
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
  planning_scene_ = locked_scene->diff();
}

void MTCTaskBuilder::refreshCollisionScene()
{
  // Request latest state from planning scene service (one synchronous call)
  planning_scene_monitor_->requestPlanningSceneState();

  // Create a modifiable copy for collision checking
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
  cached_collision_scene_ = locked_scene->diff();

  RCLCPP_DEBUG(node_->get_logger(), "Refreshed collision scene snapshot");
}

std::vector<IKSolution> MTCTaskBuilder::computeIKSolutions(
  const std::string& robot_name,
  const geometry_msgs::msg::PoseStamped& target_pose,
  int max_solutions)
{
  std::vector<IKSolution> solutions;

  std::string group_name = getPlanningGroup(robot_name);
  std::string ee_link = getEndEffectorLink(robot_name);

  const moveit::core::JointModelGroup* jmg = robot_model_->getJointModelGroup(group_name);
  if (!jmg)
  {
    RCLCPP_ERROR(node_->get_logger(), "Failed to get joint model group '%s'", group_name.c_str());
    return solutions;
  }

  // Get current state and planning scene snapshot ONCE before the loop
  // This avoids acquiring the lock 240+ times inside the loop
  planning_scene_monitor_->requestPlanningSceneState();
  moveit::core::RobotState robot_state(robot_model_);
  std::vector<double> current_joint_values;
  planning_scene::PlanningScenePtr scene_snapshot;
  {
    planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
    robot_state = locked_scene->getCurrentState();
    robot_state.copyJointGroupPositions(jmg, current_joint_values);
    // Take a snapshot of the planning scene for collision checking
    scene_snapshot = locked_scene->diff();
  }

  // Convert pose to Eigen
  Eigen::Isometry3d target_eigen;
  tf2::fromMsg(target_pose.pose, target_eigen);

  // Try to find multiple IK solutions using random seeds
  std::set<std::vector<int>> seen_configs;  // To avoid duplicates (discretized)

  for (int attempt = 0; attempt < max_solutions * 15 && static_cast<int>(solutions.size()) < max_solutions; ++attempt)
  {
    // Randomize joint positions for this attempt (to get different IK solutions)
    if (attempt > 0)
    {
      robot_state.setToRandomPositions(jmg);
    }

    // Compute IK
    if (robot_state.setFromIK(jmg, target_eigen, ee_link, 0.05))  // 0.05s timeout (faster)
    {
      // Check for collisions using the snapshot (no locking needed)
      if (scene_snapshot->isStateColliding(robot_state, group_name))
      {
        continue;  // Skip collision configurations
      }

      // Get joint values
      std::vector<double> joint_values;
      robot_state.copyJointGroupPositions(jmg, joint_values);

      // Discretize for duplicate detection (0.1 rad resolution)
      std::vector<int> discretized;
      discretized.reserve(joint_values.size());
      for (double v : joint_values)
      {
        discretized.push_back(static_cast<int>(v * 10));
      }

      // Check if we've seen this configuration before
      if (seen_configs.find(discretized) == seen_configs.end())
      {
        seen_configs.insert(discretized);

        IKSolution sol;
        sol.joint_values = std::move(joint_values);
        sol.planning_group = group_name;
        solutions.push_back(std::move(sol));
      }
    }
  }

  // Sort solutions by distance from current configuration (prefer closer solutions)
  std::sort(solutions.begin(), solutions.end(),
    [&current_joint_values](const IKSolution& a, const IKSolution& b) {
      double dist_a = 0.0, dist_b = 0.0;
      for (size_t i = 0; i < current_joint_values.size() && i < a.joint_values.size(); ++i)
      {
        double diff_a = a.joint_values[i] - current_joint_values[i];
        double diff_b = b.joint_values[i] - current_joint_values[i];
        dist_a += diff_a * diff_a;
        dist_b += diff_b * diff_b;
      }
      return dist_a < dist_b;
    });

  return solutions;
}

void MTCTaskBuilder::addMoveJStageWithJoints(
  mtc::Task& task,
  const std::string& robot_name,
  const std::vector<double>& joint_values,
  int step_index,
  bool use_sampling_planner)
{
  std::string stage_name = "move_j_" + std::to_string(step_index);
  std::string group = getPlanningGroup(robot_name);

  // Create joint-space goal
  std::map<std::string, double> joint_goal;
  const moveit::core::JointModelGroup* jmg = robot_model_->getJointModelGroup(group);
  if (!jmg)
  {
    RCLCPP_ERROR(node_->get_logger(), "Failed to get joint model group '%s'", group.c_str());
    return;
  }

  const std::vector<std::string>& joint_names = jmg->getVariableNames();
  if (joint_names.size() != joint_values.size())
  {
    RCLCPP_ERROR(node_->get_logger(), "Joint values size mismatch: expected %zu, got %zu",
                 joint_names.size(), joint_values.size());
    return;
  }

  for (size_t i = 0; i < joint_names.size(); ++i)
  {
    joint_goal[joint_names[i]] = joint_values[i];
  }

  // Select planner
  auto& planner = use_sampling_planner ? sampling_planner_ : ptp_planner_;
  double timeout = use_sampling_planner ? config_.timeout_ompl : config_.timeout_pilz_ptp;

  auto stage = std::make_unique<mtc::stages::MoveTo>(stage_name, planner);
  stage->setGroup(group);
  stage->setGoal(joint_goal);
  stage->setTimeout(timeout);

  task.add(std::move(stage));
}

mtc::Task MTCTaskBuilder::buildTaskWithIK(
  const std::vector<rsy_mtc_planning::msg::MotionStep>& steps,
  const std::map<size_t, size_t>& ik_indices,
  const std::vector<MoveJIKData>& movej_ik_data,
  const std::string& task_name,
  bool use_sampling_planner)
{
  mtc::Task task;
  task.setName(task_name);
  task.stages()->setName(task_name);
  task.setRobotModel(robot_model_);
  // NOTE: Do NOT enable introspection here - this function is called up to 256 times
  // during backtracking search. Enabling introspection registers publishers each time,
  // causing "Publisher already registered" warnings and memory overhead.
  // Introspection can be enabled on the successful task if debugging is needed.

  // Add current state as the starting point
  auto current_state = std::make_unique<mtc::stages::CurrentState>("current_state");
  task.add(std::move(current_state));

  // Create a map from step index to MoveJIKData index for quick lookup
  std::map<size_t, size_t> step_to_movej_data;
  for (size_t i = 0; i < movej_ik_data.size(); ++i)
  {
    step_to_movej_data[movej_ik_data[i].step_index] = i;
  }

  // Add stages for each motion step
  for (size_t i = 0; i < steps.size(); ++i)
  {
    const auto& step = steps[i];

    switch (step.motion_type)
    {
      case rsy_mtc_planning::msg::MotionStep::MOVE_J:
      {
        // Check if we have IK data and a selected index for this step
        auto data_it = step_to_movej_data.find(i);
        auto idx_it = ik_indices.find(i);

        if (data_it != step_to_movej_data.end() && idx_it != ik_indices.end())
        {
          const auto& ik_data = movej_ik_data[data_it->second];
          size_t ik_idx = idx_it->second;

          if (ik_idx < ik_data.ik_solutions.size())
          {
            // Use specific IK solution with selected planner
            addMoveJStageWithJoints(task, step.robot_name,
                                    ik_data.ik_solutions[ik_idx].joint_values,
                                    static_cast<int>(i),
                                    use_sampling_planner);
          }
          else
          {
            // Fallback to pose goal
            addMoveJStage(task, step, static_cast<int>(i), use_sampling_planner);
          }
        }
        else
        {
          // No IK data, use pose goal
          addMoveJStage(task, step, static_cast<int>(i), use_sampling_planner);
        }
        break;
      }

      case rsy_mtc_planning::msg::MotionStep::MOVE_L:
        addMoveLStage(task, step, static_cast<int>(i));
        break;

      default:
        RCLCPP_WARN(node_->get_logger(), "Unknown motion type %d at step %zu", step.motion_type, i);
        break;
    }
  }

  return task;
}

bool MTCTaskBuilder::validateIKCombinationCollisions(
  const std::vector<rsy_mtc_planning::msg::MotionStep>& steps,
  const std::map<size_t, size_t>& ik_indices,
  const std::vector<MoveJIKData>& movej_ik_data,
  int& collision_step_index)
{
  collision_step_index = -1;

  // Number of intermediate points to check along each trajectory
  // Reduced to just endpoints to avoid false positives - actual trajectory planning
  // does more thorough collision checking with proper path planning
  const int NUM_TRAJECTORY_SAMPLES = 1;  // Just start and end

  // Use cached collision scene if available (refreshCollisionScene() should be called beforehand)
  // This avoids expensive requestPlanningSceneState() calls when checking many combinations
  planning_scene::PlanningScenePtr scene;
  if (cached_collision_scene_)
  {
    // Create a diff from cached scene to preserve original state for next check
    scene = cached_collision_scene_->diff();
  }
  else
  {
    // Fallback: fetch scene if not cached (slower path for backward compatibility)
    planning_scene_monitor_->requestPlanningSceneState();
    planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
    scene = locked_scene->diff();
  }

  // Create robot state to simulate the sequence
  // This state is updated after each step, so subsequent steps see previous robots' new positions
  moveit::core::RobotState robot_state = scene->getCurrentState();

  // Create a map from step index to MoveJIKData index
  std::map<size_t, size_t> step_to_movej_data;
  for (size_t i = 0; i < movej_ik_data.size(); ++i)
  {
    step_to_movej_data[movej_ik_data[i].step_index] = i;
  }

  // Helper to get collision details for logging
  auto getCollisionDetails = [&scene, &robot_state]() -> std::string {
    collision_detection::CollisionRequest req;
    req.contacts = true;
    req.max_contacts = 5;
    collision_detection::CollisionResult res;
    scene->checkCollision(req, res, robot_state);

    if (res.contacts.empty()) return "unknown";

    std::string details;
    for (const auto& contact_pair : res.contacts)
    {
      if (!details.empty()) details += ", ";
      details += contact_pair.first.first + "<->" + contact_pair.first.second;
    }
    return details;
  };

  // Simulate each step and check for collisions at endpoints
  for (size_t i = 0; i < steps.size(); ++i)
  {
    const auto& step = steps[i];
    std::string group_name = getPlanningGroup(step.robot_name);
    const moveit::core::JointModelGroup* jmg = robot_model_->getJointModelGroup(group_name);

    if (!jmg)
    {
      RCLCPP_WARN(node_->get_logger(), "Unknown planning group for robot '%s'", step.robot_name.c_str());
      continue;
    }

    if (step.motion_type == rsy_mtc_planning::msg::MotionStep::MOVE_J)
    {
      // Get the IK solution for this step
      auto data_it = step_to_movej_data.find(i);
      auto idx_it = ik_indices.find(i);

      if (data_it != step_to_movej_data.end() && idx_it != ik_indices.end())
      {
        const auto& ik_data = movej_ik_data[data_it->second];
        size_t ik_idx = idx_it->second;

        if (ik_idx < ik_data.ik_solutions.size())
        {
          // Get start joint positions (current state of this robot)
          std::vector<double> start_joints;
          robot_state.copyJointGroupPositions(jmg, start_joints);

          // Get end joint positions (target IK)
          const auto& end_joints = ik_data.ik_solutions[ik_idx].joint_values;

          // Check collision at endpoints only (start and end)
          std::vector<double> interpolated_joints(start_joints.size());
          for (int sample = 0; sample <= NUM_TRAJECTORY_SAMPLES; ++sample)
          {
            double t = static_cast<double>(sample) / NUM_TRAJECTORY_SAMPLES;

            // Linear interpolation in joint space
            for (size_t j = 0; j < start_joints.size() && j < end_joints.size(); ++j)
            {
              interpolated_joints[j] = start_joints[j] + t * (end_joints[j] - start_joints[j]);
            }

            robot_state.setJointGroupPositions(jmg, interpolated_joints);
            robot_state.update();

            // Check for collisions (uses ACM from SRDF to filter allowed collisions)
            if (scene->isStateColliding(robot_state, group_name))
            {
              collision_step_index = static_cast<int>(i);
              std::string collision_info = getCollisionDetails();
              RCLCPP_INFO(node_->get_logger(),
                "Collision at step %zu (MoveJ, %s, IK[%zu], t=%.2f): %s",
                i, step.robot_name.c_str(), ik_idx, t, collision_info.c_str());
              return false;
            }
          }

          // Update robot_state to final position for subsequent steps
          robot_state.setJointGroupPositions(jmg, end_joints);
          robot_state.update();
        }
      }
    }
    else if (step.motion_type == rsy_mtc_planning::msg::MotionStep::MOVE_L)
    {
      // For MoveL, just check the endpoint - intermediate collision checking
      // is done by the actual trajectory planner (Pilz LIN)
      std::string ee_link = getEndEffectorLink(step.robot_name);

      // Get target pose
      Eigen::Isometry3d end_pose;
      tf2::fromMsg(step.target_pose.pose, end_pose);

      // Compute IK for the target pose
      if (robot_state.setFromIK(jmg, end_pose, ee_link, 0.1))
      {
        robot_state.update();

        // Check for collisions at the endpoint (uses ACM from SRDF)
        if (scene->isStateColliding(robot_state, group_name))
        {
          collision_step_index = static_cast<int>(i);
          std::string collision_info = getCollisionDetails();
          RCLCPP_INFO(node_->get_logger(),
            "Collision at step %zu (MoveL endpoint, %s): %s",
            i, step.robot_name.c_str(), collision_info.c_str());
          return false;
        }
      }
      // Note: If IK fails for MoveL endpoint, let the planner catch it
      // This pre-check is just for obvious collisions
    }
  }

  return true;  // No collisions detected at endpoints
}

}  // namespace rsy_mtc_planning