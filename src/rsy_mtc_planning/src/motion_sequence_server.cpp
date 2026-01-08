#include "rsy_mtc_planning/motion_sequence_server.hpp"

namespace rsy_mtc_planning
{

MotionSequenceServer::MotionSequenceServer(const rclcpp::NodeOptions& options)
  : Node("motion_sequence_server", options)
{
  // Create action server
  action_server_ = rclcpp_action::create_server<ExecuteMotionSequence>(
    this,
    "execute_motion_sequence",
    std::bind(&MotionSequenceServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&MotionSequenceServer::handle_cancel, this, std::placeholders::_1),
    std::bind(&MotionSequenceServer::handle_accepted, this, std::placeholders::_1));

  // Initialize gripper service clients
  robot1_gripper_client_ = this->create_client<RobotiqGripper>("robot1_robotiq_gripper");
  robot2_gripper_client_ = this->create_client<RobotiqGripper>("robot2_robotiq_gripper");

  RCLCPP_INFO(get_logger(), "Motion Sequence Server initialized");
}

MotionSequenceServer::~MotionSequenceServer()
{
  // Wait for any running execution thread to complete
  std::lock_guard<std::mutex> lock(thread_mutex_);
  if (execution_thread_.joinable())
  {
    execution_thread_.join();
  }
}

rclcpp_action::GoalResponse MotionSequenceServer::handle_goal(
  const rclcpp_action::GoalUUID& uuid,
  std::shared_ptr<const ExecuteMotionSequence::Goal> goal)
{
  (void)uuid;
  RCLCPP_INFO(get_logger(), "Received motion sequence with %zu steps",
              goal->motion_steps.size());
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse MotionSequenceServer::handle_cancel(
  const std::shared_ptr<GoalHandleExecuteMotionSequence> goal_handle)
{
  (void)goal_handle;
  RCLCPP_INFO(get_logger(), "Received cancel request");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void MotionSequenceServer::handle_accepted(
  const std::shared_ptr<GoalHandleExecuteMotionSequence> goal_handle)
{
  // Execute in a separate thread (joinable for proper cleanup)
  std::lock_guard<std::mutex> lock(thread_mutex_);

  // Join previous thread if it's still around
  if (execution_thread_.joinable())
  {
    execution_thread_.join();
  }

  execution_thread_ = std::thread{std::bind(&MotionSequenceServer::process_motion_sequence, this, goal_handle)};
}

void MotionSequenceServer::process_motion_sequence(
  const std::shared_ptr<GoalHandleExecuteMotionSequence> goal_handle)
{
  const auto goal = goal_handle->get_goal();
  auto result = std::make_shared<ExecuteMotionSequence::Result>();
  auto feedback = std::make_shared<ExecuteMotionSequence::Feedback>();

  // Initialize task builder if not done yet
  if (!task_builder_)
  {
    try
    {
      task_builder_ = std::make_shared<MTCTaskBuilder>(shared_from_this());
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(get_logger(), "Failed to initialize task builder: %s", e.what());
      result->success = false;
      result->message = "Failed to initialize task builder";
      goal_handle->abort(result);
      return;
    }
  }

  // Collect ALL motion steps (MOVE_J, MOVE_L) into one group for unified planning
  // This allows MTC to find IK solutions that work across the ENTIRE sequence
  // Gripper operations are tracked separately and executed between motion segments
  std::vector<MotionStep> all_motion_steps;
  std::vector<std::pair<size_t, MotionStep>> gripper_operations;  // (after_motion_index, gripper_step)

  for (size_t i = 0; i < goal->motion_steps.size(); ++i)
  {
    const auto& step = goal->motion_steps[i];

    if (step.motion_type == MotionStep::GRIPPER_OPEN ||
        step.motion_type == MotionStep::GRIPPER_CLOSE)
    {
      // Record gripper operation to execute after current motion index
      gripper_operations.push_back({all_motion_steps.size(), step});
    }
    else
    {
      all_motion_steps.push_back(step);
    }
  }

  if (all_motion_steps.empty())
  {
    // Only gripper operations, execute them directly
    for (const auto& [idx, gripper_step] : gripper_operations)
    {
      if (!execute_gripper_action(gripper_step.robot_name,
                                  gripper_step.motion_type == MotionStep::GRIPPER_OPEN))
      {
        result->success = false;
        result->message = "Gripper operation failed";
        result->failed_step_index = 0;
        goal_handle->abort(result);
        return;
      }
    }
    result->success = true;
    result->message = "Completed all steps";
    result->failed_step_index = -1;
    goal_handle->succeed(result);
    return;
  }

  RCLCPP_INFO(get_logger(), "Planning ALL %zu motion steps together, with %zu gripper operations to interleave",
              all_motion_steps.size(), gripper_operations.size());

  // Update feedback
  feedback->current_step = 0;
  feedback->total_steps = static_cast<int32_t>(goal->motion_steps.size());
  feedback->status = "Computing IK solutions for backtracking";
  goal_handle->publish_feedback(feedback);

  // ===== TRUE BACKTRACKING WITH IK EXPLORATION =====
  // 1. Identify all MoveJ steps and compute multiple IK solutions for each
  // 2. Systematically try different combinations of IK solutions
  // 3. Use backtracking to explore the search space

  // Collect MoveJ steps and compute IK solutions
  std::vector<MoveJIKData> movej_ik_data;
  const int max_ik_per_step = 16;  // Maximum IK solutions to compute per MoveJ step

  for (size_t i = 0; i < all_motion_steps.size(); ++i)
  {
    const auto& step = all_motion_steps[i];
    if (step.motion_type == MotionStep::MOVE_J)
    {
      MoveJIKData ik_data;
      ik_data.step_index = i;
      ik_data.robot_name = step.robot_name;
      ik_data.target_pose = step.target_pose;
      ik_data.ik_solutions = task_builder_->computeIKSolutions(step.robot_name, step.target_pose, max_ik_per_step);

      if (ik_data.ik_solutions.empty())
      {
        RCLCPP_WARN(get_logger(), "No IK solutions found for MoveJ step %zu (%s), will use pose goal",
                    i, step.robot_name.c_str());
      }

      movej_ik_data.push_back(ik_data);
    }
  }

  // Calculate total combinations
  size_t total_combinations = 1;
  for (const auto& ik_data : movej_ik_data)
  {
    size_t num_solutions = ik_data.ik_solutions.empty() ? 1 : ik_data.ik_solutions.size();
    total_combinations *= num_solutions;
  }

  RCLCPP_INFO(get_logger(), "Found %zu MoveJ steps with %zu total IK combinations to explore",
              movej_ik_data.size(), total_combinations);

  // Update feedback
  feedback->status = "Exploring IK combinations with backtracking";
  goal_handle->publish_feedback(feedback);

  // ===== TWO-PHASE BACKTRACKING SEARCH =====
  // Phase 1: Try with Pilz PTP (fast, minimal joint motion)
  // Phase 2: If phase 1 fails, try with OMPL (can avoid obstacles, but more joint motion)

  bool planning_succeeded = false;
  moveit::task_constructor::Task successful_task;
  size_t total_combinations_tried = 0;

  for (int phase = 1; phase <= 2 && !planning_succeeded; ++phase)
  {
    bool use_ompl = (phase == 2);
    const char* planner_name = use_ompl ? "OMPL" : "Pilz-PTP";

    // Phase 1 (Pilz): Try fewer combinations since they're sorted by distance
    // Phase 2 (OMPL): Try more combinations since OMPL can find paths around obstacles
    const size_t phase_max = use_ompl ?
      std::min(total_combinations, static_cast<size_t>(1000)) :
      std::min(total_combinations, static_cast<size_t>(500));

    RCLCPP_INFO(get_logger(), "=== Phase %d: Using %s planner (up to %zu combinations) ===",
                phase, planner_name, phase_max);

    // Reset IK indices for this phase
    std::vector<size_t> current_ik_indices(movej_ik_data.size(), 0);
    size_t combinations_tried = 0;

    while (!planning_succeeded && combinations_tried < phase_max)
    {
      if (goal_handle->is_canceling())
      {
        result->success = false;
        result->message = "Canceled";
        goal_handle->canceled(result);
        return;
      }

      combinations_tried++;
      total_combinations_tried++;

      // Build ik_indices map for buildTaskWithIK
      std::map<size_t, size_t> ik_indices;
      for (size_t i = 0; i < movej_ik_data.size(); ++i)
      {
        if (!movej_ik_data[i].ik_solutions.empty())
        {
          ik_indices[movej_ik_data[i].step_index] = current_ik_indices[i];
        }
      }

      // Log progress (less frequently for speed)
      if (combinations_tried == 1 || combinations_tried % 100 == 0)
      {
        std::string combo_str;
        for (size_t i = 0; i < current_ik_indices.size(); ++i)
        {
          combo_str += std::to_string(current_ik_indices[i]);
          if (i < current_ik_indices.size() - 1) combo_str += ",";
        }
        RCLCPP_INFO(get_logger(), "[%s] Trying IK combination %zu/%zu [%s]",
                    planner_name, combinations_tried, phase_max, combo_str.c_str());
      }

      try
      {
        // Build task with specific IK solutions and selected planner
        auto task = task_builder_->buildTaskWithIK(all_motion_steps, ik_indices, movej_ik_data,
                                                    "full_motion_sequence", use_ompl);

        // Try to plan
        if (task_builder_->planTask(task, 1))
        {
          planning_succeeded = true;
          successful_task = std::move(task);
          RCLCPP_INFO(get_logger(), "Planning succeeded with %s at combination %zu (total: %zu)!",
                      planner_name, combinations_tried, total_combinations_tried);
        }
      }
      catch (const std::exception& e)
      {
        RCLCPP_DEBUG(get_logger(), "[%s] IK combination %zu failed: %s", planner_name, combinations_tried, e.what());
      }

      if (!planning_succeeded)
      {
        // Advance to next combination (like counting in mixed-radix)
        bool carry = true;
        for (size_t i = movej_ik_data.size(); i > 0 && carry; --i)
        {
          size_t idx = i - 1;
          size_t max_idx = movej_ik_data[idx].ik_solutions.empty() ? 1 : movej_ik_data[idx].ik_solutions.size();

          current_ik_indices[idx]++;
          if (current_ik_indices[idx] >= max_idx)
          {
            current_ik_indices[idx] = 0;
          }
          else
          {
            carry = false;
          }
        }

        // If we've wrapped around completely, we're done with this phase
        if (carry)
        {
          RCLCPP_INFO(get_logger(), "[%s] Exhausted all %zu IK combinations", planner_name, combinations_tried);
          break;
        }
      }
    }
  }

  if (!planning_succeeded)
  {
    RCLCPP_ERROR(get_logger(), "Planning failed after trying %zu IK combinations in both phases",
                 total_combinations_tried);
    result->success = false;
    result->message = "Planning failed: no IK combination allows feasible paths";
    result->failed_step_index = 0;
    goal_handle->abort(result);
    return;
  }

  RCLCPP_INFO(get_logger(), "Planning succeeded, now executing with interleaved gripper operations");

  // Execute sub-trajectories one at a time, interleaving gripper operations
  size_t num_trajectories = task_builder_->getNumSubTrajectories(successful_task);
  size_t gripper_op_idx = 0;

  RCLCPP_INFO(get_logger(), "Executing %zu sub-trajectories with %zu gripper operations",
              num_trajectories, gripper_operations.size());

  for (size_t traj_idx = 0; traj_idx < num_trajectories; ++traj_idx)
  {
    if (goal_handle->is_canceling())
    {
      result->success = false;
      result->message = "Canceled";
      goal_handle->canceled(result);
      return;
    }

    // Execute any gripper operations that should happen before this trajectory
    // gripper_operations[i].first is the motion index AFTER which the gripper should execute
    // So we execute grippers where first == traj_idx (meaning after motion traj_idx-1, before traj_idx)
    while (gripper_op_idx < gripper_operations.size() &&
           gripper_operations[gripper_op_idx].first == traj_idx)
    {
      const auto& gripper_step = gripper_operations[gripper_op_idx].second;
      RCLCPP_INFO(get_logger(), "Executing gripper operation before trajectory %zu", traj_idx);
      if (!execute_gripper_action(gripper_step.robot_name,
                                  gripper_step.motion_type == MotionStep::GRIPPER_OPEN))
      {
        result->success = false;
        result->message = "Gripper operation failed before trajectory " + std::to_string(traj_idx);
        result->failed_step_index = static_cast<int32_t>(traj_idx);
        goal_handle->abort(result);
        return;
      }
      gripper_op_idx++;
    }

    // Execute this sub-trajectory
    RCLCPP_INFO(get_logger(), "Executing trajectory %zu/%zu", traj_idx + 1, num_trajectories);
    if (!task_builder_->executeSubTrajectory(successful_task, traj_idx))
    {
      RCLCPP_ERROR(get_logger(), "Trajectory %zu execution failed", traj_idx);
      result->success = false;
      result->message = "Trajectory execution failed at index " + std::to_string(traj_idx);
      result->failed_step_index = static_cast<int32_t>(traj_idx);
      goal_handle->abort(result);
      return;
    }
  }

  // Execute any remaining gripper operations (those that come after all motions)
  while (gripper_op_idx < gripper_operations.size())
  {
    const auto& gripper_step = gripper_operations[gripper_op_idx].second;
    RCLCPP_INFO(get_logger(), "Executing gripper operation after all trajectories");
    if (!execute_gripper_action(gripper_step.robot_name,
                                gripper_step.motion_type == MotionStep::GRIPPER_OPEN))
    {
      result->success = false;
      result->message = "Gripper operation failed after trajectories";
      result->failed_step_index = static_cast<int32_t>(goal->motion_steps.size() - 1);
      goal_handle->abort(result);
      return;
    }
    gripper_op_idx++;
  }

  result->success = true;
  result->message = "Completed all steps";
  result->failed_step_index = -1;
  goal_handle->succeed(result);
}

bool MotionSequenceServer::execute_motion_step(const MotionStep& step)
{
  RCLCPP_INFO(get_logger(), "Executing step: type=%d, robot=%s",
              step.motion_type, step.robot_name.c_str());

  // Handle gripper operations directly
  if (step.motion_type == MotionStep::GRIPPER_OPEN)
  {
    return execute_gripper_action(step.robot_name, true);
  }
  else if (step.motion_type == MotionStep::GRIPPER_CLOSE)
  {
    return execute_gripper_action(step.robot_name, false);
  }

  // For motion steps, build and execute a single-step MTC task
  std::vector<MotionStep> single_step = {step};

  try
  {
    auto task = task_builder_->buildTask(single_step, "single_motion");

    if (!task_builder_->planTask(task, 1))
    {
      RCLCPP_ERROR(get_logger(), "Planning failed for motion step");
      return false;
    }

    // Use direct execution to bypass buggy MTC execute_task_solution action
    if (!task_builder_->executeTaskDirect(task))
    {
      RCLCPP_ERROR(get_logger(), "Execution failed for motion step");
      return false;
    }

    return true;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Exception during motion step: %s", e.what());
    return false;
  }
}

bool MotionSequenceServer::execute_gripper_action(const std::string& robot_name, bool open)
{
  auto client = (robot_name == "robot1") ? robot1_gripper_client_ : robot2_gripper_client_;

  if (!client->wait_for_service(std::chrono::seconds(5)))
  {
    RCLCPP_ERROR(get_logger(), "Gripper service not available for %s", robot_name.c_str());
    return false;
  }

  auto request = std::make_shared<RobotiqGripper::Request>();
  request->action = open ? "open" : "close";

  RCLCPP_INFO(get_logger(), "Sending gripper command '%s' to %s", request->action.c_str(), robot_name.c_str());

  auto future = client->async_send_request(request);

  if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready)
  {
    RCLCPP_ERROR(get_logger(), "Gripper service call timeout for %s", robot_name.c_str());
    return false;
  }

  auto result = future.get();
  if (!result->success)
  {
    RCLCPP_ERROR(get_logger(), "Gripper action failed for %s: %s", robot_name.c_str(), result->message.c_str());
    return false;
  }

  RCLCPP_INFO(get_logger(), "Gripper action succeeded for %s", robot_name.c_str());
  return true;
}

}  // namespace rsy_mtc_planning

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rsy_mtc_planning::MotionSequenceServer)