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
#include <moveit/task_constructor/stages/move_relative.h>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <set>
#include <algorithm>

namespace rsy_mtc_planning
{

namespace mtc = moveit::task_constructor;

MTCTaskBuilder::MTCTaskBuilder(const rclcpp::Node::SharedPtr& node, const PlannerConfig& config)
  : node_(node),
    velocity_scaling_ptp_(config.velocity_scaling_ptp),
    velocity_scaling_lin_(config.velocity_scaling_lin),
    acceleration_scaling_ptp_(config.acceleration_scaling_ptp),
    acceleration_scaling_lin_(config.acceleration_scaling_lin)
{
  // Load robot model
  robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(node_, "robot_description");
  robot_model_ = robot_model_loader_->getModel();

  if (!robot_model_)
  {
    RCLCPP_ERROR(node_->get_logger(), "Failed to load robot model");
    throw std::runtime_error("Failed to load robot model");
  }

  // Create planning scene monitor for execution
  planning_scene_monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
    node_, robot_model_loader_, "planning_scene_monitor");
  planning_scene_monitor_->startSceneMonitor();
  planning_scene_monitor_->startStateMonitor();
  planning_scene_monitor_->startWorldGeometryMonitor();

  // Create planning scene
  planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_);

  // Create planners with configurable timeouts
  sampling_planner_ = std::make_shared<mtc::solvers::PipelinePlanner>(node_, "ompl", config.ompl_planner_id);
  sampling_planner_->setTimeout(config.timeout_ompl);

  ptp_planner_ = std::make_shared<mtc::solvers::PipelinePlanner>(node_, "pilz_industrial_motion_planner", "PTP");
  ptp_planner_->setTimeout(config.timeout_pilz_ptp);

  lin_planner_ = std::make_shared<mtc::solvers::PipelinePlanner>(node_, "pilz_industrial_motion_planner", "LIN");
  lin_planner_->setTimeout(config.timeout_pilz_lin);

  RCLCPP_INFO(node_->get_logger(), "Using OMPL planner: %s", config.ompl_planner_id.c_str());

  // Setup robot configurations
  planning_groups_["robot1"] = "robot1_ur_manipulator";
  planning_groups_["robot2"] = "robot2_ur_manipulator";
  ee_links_["robot1"] = "robot1_tcp";
  ee_links_["robot2"] = "robot2_tcp";

  RCLCPP_DEBUG(node_->get_logger(), "MTCTaskBuilder initialized");
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

      // Create MoveGroupInterface for this group
      auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, group_name);

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
    return 0;
  }

  const auto& solution = task.solutions().front();
  moveit_task_constructor_msgs::msg::Solution solution_msg;
  solution->toMsg(solution_msg, nullptr);

  size_t count = 0;
  for (const auto& sub_traj : solution_msg.sub_trajectory)
  {
    if (!sub_traj.trajectory.joint_trajectory.points.empty() ||
        !sub_traj.trajectory.multi_dof_joint_trajectory.points.empty())
    {
      count++;
    }
  }
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

        RCLCPP_DEBUG(node_->get_logger(), "Executing traj %zu: %s (%zu pts)",
                    index, group_name.c_str(), sub_traj.trajectory.joint_trajectory.points.size());

        auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, group_name);
        auto result = move_group->execute(sub_traj.trajectory);

        if (result != moveit::core::MoveItErrorCode::SUCCESS)
        {
          RCLCPP_ERROR(node_->get_logger(), "Trajectory %zu failed: code=%d", index, result.val);
          return false;
        }
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

  auto stage = std::make_unique<mtc::stages::MoveTo>(stage_name, planner);
  stage->setGroup(group);
  stage->setGoal(target);
  stage->setIKFrame(ee_link);

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

  // Get current state from planning scene monitor
  planning_scene_monitor_->requestPlanningSceneState();
  moveit::core::RobotState robot_state(robot_model_);
  std::vector<double> current_joint_values;
  {
    planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
    robot_state = locked_scene->getCurrentState();
    robot_state.copyJointGroupPositions(jmg, current_joint_values);
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
      // Check for collisions in current planning scene
      planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
      if (locked_scene->isStateColliding(robot_state, group_name))
      {
        continue;  // Skip collision configurations
      }

      // Get joint values
      std::vector<double> joint_values;
      robot_state.copyJointGroupPositions(jmg, joint_values);

      // Discretize for duplicate detection (0.1 rad resolution)
      std::vector<int> discretized;
      for (double v : joint_values)
      {
        discretized.push_back(static_cast<int>(v * 10));
      }

      // Check if we've seen this configuration before
      if (seen_configs.find(discretized) == seen_configs.end())
      {
        seen_configs.insert(discretized);

        IKSolution sol;
        sol.joint_values = joint_values;
        sol.planning_group = group_name;
        solutions.push_back(sol);
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

  auto stage = std::make_unique<mtc::stages::MoveTo>(stage_name, planner);
  stage->setGroup(group);
  stage->setGoal(joint_goal);

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
  task.enableIntrospection();

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

}  // namespace rsy_mtc_planning