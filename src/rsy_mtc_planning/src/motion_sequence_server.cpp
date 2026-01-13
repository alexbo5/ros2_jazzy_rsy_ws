#include "rsy_mtc_planning/motion_sequence_server.hpp"

#include <chrono>
#include <iomanip>
#include <set>
#include <sstream>

namespace rsy_mtc_planning
{

MotionSequenceServer::MotionSequenceServer(const rclcpp::NodeOptions& options)
  : Node("motion_sequence_server", options)
{
  // ========== HARDCODED CONFIGURATION ==========
  // All values are hardcoded here - no YAML config files needed

  // IK and backtracking settings
  max_ik_per_step_ = 32;               // Max IK solutions per MoveJ step
  max_pilz_combinations_ = 16;         // Phase 1: Pilz PTP (0 = skip)
  max_ompl_combinations_ = 1024;       // Phase 2: OMPL combinations to try (32x32=1024 max)
  robot_description_timeout_ = 30.0;   // Seconds to wait for robot_description

  // Planner configuration - all hardcoded
  planner_config_.timeout_ompl = 3.0;            // OMPL planner timeout per attempt (seconds)
  planner_config_.timeout_pilz_ptp = 10.0;       // Pilz PTP timeout
  planner_config_.timeout_pilz_lin = 10.0;       // Pilz LIN timeout
  planner_config_.ompl_planning_attempts = 3;    // OMPL attempts per IK combination
  planner_config_.velocity_scaling_ptp = 0.3;   // PTP velocity (0.0-1.0)
  planner_config_.velocity_scaling_lin = 0.3;   // LIN velocity
  planner_config_.acceleration_scaling_ptp = 0.3; // PTP acceleration (0.0-1.0)
  planner_config_.acceleration_scaling_lin = 0.3; // LIN acceleration (0.0-1.0)
  // OMPL planner name (short name, looked up in planner_configs)
  // Options: RRTConnect, RRTstar, PRMstar
  planner_config_.ompl_planner_id = "RRTConnect";

  RCLCPP_INFO(get_logger(),
    "Hardcoded config: ik=%d, backtrack(pilz=%zu, ompl=%zu), "
    "timeout(ompl=%.1fs, ptp=%.1fs, lin=%.1fs), planner=%s",
    max_ik_per_step_, max_pilz_combinations_, max_ompl_combinations_,
    planner_config_.timeout_ompl, planner_config_.timeout_pilz_ptp, planner_config_.timeout_pilz_lin,
    planner_config_.ompl_planner_id.c_str());

  // ========== DECLARE OMPL PARAMETERS AT STARTUP ==========
  // MoveIt needs these parameters to configure the OMPL planner
  declareOmplParameters();

  // Initialize planning logger with timestamped filename
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::ostringstream log_filename;
  log_filename << "/root/ros2_ws/src/rsy_mtc_planning/logs/mtc_planning_"
               << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S")
               << ".log";
  planning_logger_ = std::make_unique<PlanningLogger>(log_filename.str());
  RCLCPP_INFO(get_logger(), "Planning log: %s", log_filename.str().c_str());

  // Check if robot_description is already available as a parameter
  if (this->has_parameter("robot_description"))
  {
    RCLCPP_INFO(get_logger(), "Using robot_description from parameter");
    robot_description_received_ = true;
  }
  else
  {
    // Subscribe to /robot_description topic from robot_state_publisher
    RCLCPP_INFO(get_logger(), "Subscribing to /robot_description topic...");

    auto qos = rclcpp::QoS(1).transient_local();

    robot_description_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/robot_description", qos,
      std::bind(&MotionSequenceServer::robot_description_callback, this, std::placeholders::_1));
  }

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

  RCLCPP_INFO(get_logger(), "Motion Sequence Server ready");
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
  RCLCPP_DEBUG(get_logger(), "Received motion sequence with %zu steps", goal->motion_steps.size());
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse MotionSequenceServer::handle_cancel(
  const std::shared_ptr<GoalHandleExecuteMotionSequence> goal_handle)
{
  (void)goal_handle;
  RCLCPP_WARN(get_logger(), "Cancel request received");
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
    // Wait for robot_description if not received yet
    if (!robot_description_received_)
    {
      RCLCPP_INFO(get_logger(), "Waiting for robot_description from topic...");

      if (!wait_for_robot_description(robot_description_timeout_))
      {
        RCLCPP_ERROR(get_logger(), "Timeout waiting for /robot_description topic");
        result->success = false;
        result->message = "Failed to receive robot_description from topic";
        goal_handle->abort(result);
        return;
      }
    }

    try
    {
      // Use the planner_config_ loaded at startup
      task_builder_ = std::make_shared<MTCTaskBuilder>(shared_from_this(), planner_config_);
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
    RCLCPP_DEBUG(get_logger(), "No motion steps, executing %zu gripper ops only", gripper_operations.size());
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

  // Start planning session logging
  std::string session_id = planning_logger_->startSession();
  RCLCPP_INFO(get_logger(), "Planning session %s: %zu motion steps...", session_id.c_str(), all_motion_steps.size());

  // Update feedback
  feedback->current_step = 0;
  feedback->total_steps = static_cast<int32_t>(goal->motion_steps.size());
  feedback->status = "Computing IK solutions for backtracking";
  goal_handle->publish_feedback(feedback);

  // ===== IMPROVED BACKTRACKING WITH MINIMAL JOINT MOVEMENT =====
  // 1. Identify all MoveJ steps and compute multiple IK solutions for each
  // 2. Re-sort IK solutions to minimize total joint movement across the sequence
  // 3. Smart backtracking: when MoveL fails, prioritize changing the preceding MoveJ's IK

  // Collect MoveJ steps and compute IK solutions
  std::vector<MoveJIKData> movej_ik_data;

  for (size_t i = 0; i < all_motion_steps.size(); ++i)
  {
    const auto& step = all_motion_steps[i];

    if (step.motion_type == MotionStep::MOVE_J)
    {
      MoveJIKData ik_data;
      ik_data.step_index = i;
      ik_data.robot_name = step.robot_name;
      ik_data.target_pose = step.target_pose;
      ik_data.ik_solutions = task_builder_->computeIKSolutions(step.robot_name, step.target_pose, max_ik_per_step_);

      if (ik_data.ik_solutions.empty())
      {
        RCLCPP_WARN(get_logger(), "No IK for step %zu (%s), using pose goal", i, step.robot_name.c_str());
      }

      movej_ik_data.push_back(ik_data);
    }
  }

  // Re-sort IK solutions to minimize sequential joint movement
  // For each step after the first (per robot), sort by distance from previous step's best IK
  std::map<std::string, std::vector<double>> last_ik_per_robot;
  for (size_t i = 0; i < movej_ik_data.size(); ++i)
  {
    auto& ik_data = movej_ik_data[i];
    if (ik_data.ik_solutions.empty()) continue;

    auto it = last_ik_per_robot.find(ik_data.robot_name);
    if (it != last_ik_per_robot.end())
    {
      // Sort by distance from previous step's best IK (not current pose)
      const auto& prev_joints = it->second;
      std::sort(ik_data.ik_solutions.begin(), ik_data.ik_solutions.end(),
        [&prev_joints](const IKSolution& a, const IKSolution& b) {
          double dist_a = 0.0, dist_b = 0.0;
          for (size_t j = 0; j < prev_joints.size() && j < a.joint_values.size(); ++j)
          {
            double diff_a = a.joint_values[j] - prev_joints[j];
            double diff_b = b.joint_values[j] - prev_joints[j];
            dist_a += diff_a * diff_a;
            dist_b += diff_b * diff_b;
          }
          return dist_a < dist_b;
        });
    }
    // Update last IK for this robot (use the best one after sorting)
    last_ik_per_robot[ik_data.robot_name] = ik_data.ik_solutions[0].joint_values;
  }

  // Build map from step_index to movej_ik_data index for smart backtracking
  std::map<size_t, size_t> step_to_movej_idx;
  for (size_t i = 0; i < movej_ik_data.size(); ++i)
  {
    step_to_movej_idx[movej_ik_data[i].step_index] = i;
  }

  // Build map: for each step, find the most recent MoveJ for the same robot
  // This is used for smart backtracking when MoveL fails
  std::map<size_t, size_t> step_to_preceding_movej;  // step_index -> movej_ik_data index
  std::map<std::string, size_t> last_movej_per_robot;  // robot_name -> movej_ik_data index
  for (size_t i = 0; i < all_motion_steps.size(); ++i)
  {
    const auto& step = all_motion_steps[i];
    auto it = last_movej_per_robot.find(step.robot_name);
    if (it != last_movej_per_robot.end())
    {
      step_to_preceding_movej[i] = it->second;
    }
    // Update last MoveJ for this robot if current step is MoveJ
    if (step.motion_type == MotionStep::MOVE_J)
    {
      auto movej_it = step_to_movej_idx.find(i);
      if (movej_it != step_to_movej_idx.end())
      {
        last_movej_per_robot[step.robot_name] = movej_it->second;
      }
    }
  }

  // Calculate total combinations and collect IK counts
  size_t total_combinations = 1;
  std::vector<size_t> ik_counts;
  for (const auto& ik_data : movej_ik_data)
  {
    size_t num_solutions = ik_data.ik_solutions.empty() ? 1 : ik_data.ik_solutions.size();
    total_combinations *= num_solutions;
    ik_counts.push_back(num_solutions);
  }

  // Log IK computation summary
  RCLCPP_DEBUG(get_logger(), "IK: %zu MoveJ steps, %zu combinations", movej_ik_data.size(), total_combinations);

  // Record detailed MoveJ IK info for logging
  std::vector<MoveJIKInfo> movej_ik_info;
  for (const auto& ik_data : movej_ik_data)
  {
    MoveJIKInfo info;
    info.step_index = ik_data.step_index;
    info.robot_name = ik_data.robot_name;
    info.total_ik_found = ik_data.ik_solutions.size();
    info.collision_free_ik = ik_data.ik_solutions.size();  // Will be updated after collision checking
    movej_ik_info.push_back(info);
  }
  planning_logger_->recordMoveJIKInfo(movej_ik_info);

  // Record input metrics for logging
  size_t movel_count = all_motion_steps.size() - movej_ik_data.size();
  planning_logger_->recordInput(
    all_motion_steps.size(),
    movej_ik_data.size(),
    movel_count,
    gripper_operations.size(),
    ik_counts,
    total_combinations);

  // Record planner configuration
  planning_logger_->recordPlannerConfig(
    max_pilz_combinations_,
    max_ompl_combinations_,
    planner_config_.timeout_pilz_ptp,
    planner_config_.timeout_ompl,
    planner_config_.ompl_planner_id);

  // Update feedback
  feedback->status = "Pre-filtering IK combinations for collisions";
  goal_handle->publish_feedback(feedback);

  // ===== PRE-FILTER ALL IK COMBINATIONS FOR COLLISION =====
  // Check all combinations upfront and only keep collision-free ones
  // This avoids wasting planning time on combinations that will definitely fail

  // Refresh collision scene ONCE before checking all combinations
  // This avoids 1024 synchronous requestPlanningSceneState() calls
  task_builder_->refreshCollisionScene();

  std::vector<std::vector<size_t>> valid_combinations;  // Collision-free IK combinations
  size_t collision_filtered_count = 0;

  {
    std::vector<size_t> current_indices(movej_ik_data.size(), 0);
    bool has_more = true;

    while (has_more)
    {
      // Build ik_indices map for collision check
      std::map<size_t, size_t> ik_indices;
      for (size_t i = 0; i < movej_ik_data.size(); ++i)
      {
        if (!movej_ik_data[i].ik_solutions.empty())
        {
          ik_indices[movej_ik_data[i].step_index] = current_indices[i];
        }
      }

      // Check collision for this combination
      int collision_step = -1;
      if (task_builder_->validateIKCombinationCollisions(all_motion_steps, ik_indices, movej_ik_data, collision_step))
      {
        // No collision - add to valid combinations
        valid_combinations.push_back(current_indices);
      }
      else
      {
        collision_filtered_count++;
      }

      // Advance to next combination (mixed-radix counting)
      bool carry = true;
      for (size_t i = movej_ik_data.size(); i > 0 && carry; --i)
      {
        size_t idx = i - 1;
        size_t max_idx = movej_ik_data[idx].ik_solutions.empty() ? 1 : movej_ik_data[idx].ik_solutions.size();
        current_indices[idx]++;
        if (current_indices[idx] >= max_idx)
        {
          current_indices[idx] = 0;
        }
        else
        {
          carry = false;
        }
      }
      has_more = !carry;
    }
  }

  RCLCPP_INFO(get_logger(), "IK pre-filter: %zu valid, %zu filtered (collision), %zu total",
              valid_combinations.size(), collision_filtered_count, total_combinations);

  if (valid_combinations.empty())
  {
    RCLCPP_ERROR(get_logger(), "No collision-free IK combinations found");
    planning_logger_->recordPlanningFailure("All IK combinations have collisions");
    planning_logger_->finalizeSession();

    result->success = false;
    result->message = "Planning failed: all IK combinations have collisions";
    result->failed_step_index = 0;
    goal_handle->abort(result);
    return;
  }

  // Update feedback
  feedback->status = "Exploring IK combinations with backtracking";
  goal_handle->publish_feedback(feedback);

  // ===== SMART TWO-PHASE BACKTRACKING SEARCH =====
  // Phase 1: Try with Pilz PTP (fast, minimal joint motion)
  // Phase 2: If phase 1 fails, try with OMPL (can avoid obstacles, but more joint motion)
  // Smart backtracking with fallback: ensures every valid combination is tried exactly once

  bool planning_succeeded = false;
  moveit::task_constructor::Task successful_task;
  std::vector<size_t> success_ik_indices;

  // Create index of valid combinations for fast lookup (combination -> index in valid_combinations)
  std::map<std::vector<size_t>, size_t> valid_combo_index;
  for (size_t i = 0; i < valid_combinations.size(); ++i)
  {
    valid_combo_index[valid_combinations[i]] = i;
  }

  for (int phase = 1; phase <= 2 && !planning_succeeded; ++phase)
  {
    bool use_ompl = (phase == 2);
    std::string planner_name = use_ompl ? "OMPL" : "Pilz-PTP";

    // Max attempts is limited by configured limit per phase
    const size_t phase_limit = use_ompl ? max_ompl_combinations_ : max_pilz_combinations_;
    const size_t phase_max = std::min(valid_combinations.size(), phase_limit);

    RCLCPP_INFO(get_logger(), "Phase %d: %s (max %zu of %zu valid combinations)",
                phase, planner_name.c_str(), phase_max, valid_combinations.size());

    planning_logger_->startPhase(planner_name, phase_max);

    // Track which combinations have been tried in this phase (by index in valid_combinations)
    std::set<size_t> tried_combo_indices;

    std::vector<size_t> current_ik_indices = valid_combinations[0];  // Start with first valid combination
    size_t combinations_tried = 0;
    bool phase_ended = false;

    // Track smart backtracking state
    int last_failed_step = -1;
    std::set<size_t> exhausted_movej_indices;
    size_t current_backtrack_movej = SIZE_MAX;

    // Helper to check if current combination is valid and not yet tried
    auto is_valid_untried = [&](const std::vector<size_t>& indices) -> bool {
      auto it = valid_combo_index.find(indices);
      if (it == valid_combo_index.end()) return false;  // Not collision-free
      return tried_combo_indices.find(it->second) == tried_combo_indices.end();  // Not yet tried
    };

    // Helper to find next valid untried combination via smart backtracking
    auto find_next_valid_smart = [&](size_t movej_idx, std::vector<size_t>& indices) -> bool {
      size_t max_idx = movej_ik_data[movej_idx].ik_solutions.empty() ? 1 : movej_ik_data[movej_idx].ik_solutions.size();
      size_t original = indices[movej_idx];

      for (size_t attempt = 1; attempt < max_idx; ++attempt)
      {
        indices[movej_idx] = (original + attempt) % max_idx;
        if (is_valid_untried(indices))
        {
          return true;
        }
      }
      indices[movej_idx] = original;  // Restore
      return false;
    };

    // ===== SMART BACKTRACKING PHASE =====
    while (!planning_succeeded && combinations_tried < phase_max)
    {
      if (goal_handle->is_canceling())
      {
        RCLCPP_WARN(get_logger(), "Planning canceled by user");
        result->success = false;
        result->message = "Canceled";
        goal_handle->canceled(result);
        return;
      }

      // Skip if already tried or not valid
      auto combo_it = valid_combo_index.find(current_ik_indices);
      if (combo_it == valid_combo_index.end() || tried_combo_indices.count(combo_it->second))
      {
        // Find next untried valid combination (simple linear search as fallback)
        bool found = false;
        for (size_t i = 0; i < valid_combinations.size(); ++i)
        {
          if (tried_combo_indices.find(i) == tried_combo_indices.end())
          {
            current_ik_indices = valid_combinations[i];
            found = true;
            break;
          }
        }
        if (!found) break;  // All tried
        continue;
      }

      // Mark as tried
      tried_combo_indices.insert(combo_it->second);
      combinations_tried++;

      // Build ik_indices map for buildTaskWithIK
      std::map<size_t, size_t> ik_indices;
      for (size_t i = 0; i < movej_ik_data.size(); ++i)
      {
        if (!movej_ik_data[i].ik_solutions.empty())
        {
          ik_indices[movej_ik_data[i].step_index] = current_ik_indices[i];
        }
      }

      if (combinations_tried == 1 || combinations_tried % 50 == 0)
      {
        RCLCPP_DEBUG(get_logger(), "[%s] %zu/%zu", planner_name.c_str(), combinations_tried, phase_max);
      }

      try
      {
        // Create attempt record for logging
        AttemptRecord attempt_record;
        attempt_record.attempt_number = combinations_tried;
        attempt_record.phase_name = planner_name;
        attempt_record.ik_indices = current_ik_indices;
        attempt_record.collision_precheck_passed = true;  // Pre-filtered
        attempt_record.planning_attempted = true;

        // Build and plan task (collision already pre-checked)
        auto task = task_builder_->buildTaskWithIK(all_motion_steps, ik_indices, movej_ik_data,
                                                    "full_motion_sequence", use_ompl);

        PlanningResult plan_result = task_builder_->planTaskWithResult(task, 1);

        if (plan_result.success)
        {
          planning_succeeded = true;
          successful_task = std::move(task);
          success_ik_indices = current_ik_indices;
          RCLCPP_INFO(get_logger(), "SUCCESS: %s @ #%zu", planner_name.c_str(), combinations_tried);

          attempt_record.planning_succeeded = true;
          planning_logger_->recordAttempt(attempt_record);

          phase_ended = true;
          planning_logger_->endPhase(combinations_tried, true);
          planning_logger_->recordPlanningSuccess(planner_name, combinations_tried, current_ik_indices);
        }
        else
        {
          // Record planning failure
          attempt_record.planning_succeeded = false;
          attempt_record.failed_step_index = plan_result.failed_stage_index;
          attempt_record.failed_step_name = plan_result.failed_stage_name;
          attempt_record.failed_step_type = plan_result.is_movel ? "MoveL" : "MoveJ";
          attempt_record.failure_reason = plan_result.error_message;
          planning_logger_->recordAttempt(attempt_record);

          // Record stage failure for statistics
          planning_logger_->recordStageFailure(
            plan_result.failed_stage_index,
            plan_result.failed_stage_name,
            plan_result.is_movel);

          // ===== SMART BACKTRACKING =====
          bool used_smart_backtrack = false;

          if (plan_result.failed_stage_index >= 0)
          {
            size_t failed_step = static_cast<size_t>(plan_result.failed_stage_index);

            // Reset state if new failure location
            if (last_failed_step != plan_result.failed_stage_index)
            {
              last_failed_step = plan_result.failed_stage_index;
              exhausted_movej_indices.clear();
              current_backtrack_movej = SIZE_MAX;
            }

            // Determine primary MoveJ to try
            size_t primary_movej = SIZE_MAX;
            std::string failed_robot_name;

            auto failed_movej_it = step_to_movej_idx.find(failed_step);
            if (!plan_result.is_movel && failed_movej_it != step_to_movej_idx.end())
            {
              primary_movej = failed_movej_it->second;
              failed_robot_name = movej_ik_data[primary_movej].robot_name;
            }
            else if (plan_result.is_movel)
            {
              auto prec_it = step_to_preceding_movej.find(failed_step);
              if (prec_it != step_to_preceding_movej.end())
              {
                primary_movej = prec_it->second;
                failed_robot_name = movej_ik_data[primary_movej].robot_name;
              }
            }

            // Collect candidate MoveJs
            std::vector<size_t> candidate_movejs;
            for (size_t i = 0; i < movej_ik_data.size(); ++i)
            {
              bool is_candidate = plan_result.is_movel ?
                (movej_ik_data[i].step_index < failed_step) :
                (movej_ik_data[i].step_index <= failed_step);

              if (is_candidate &&
                  !movej_ik_data[i].ik_solutions.empty() &&
                  exhausted_movej_indices.find(i) == exhausted_movej_indices.end())
              {
                candidate_movejs.push_back(i);
              }
            }

            // Sort: primary first, then same robot, then closer to failure
            std::sort(candidate_movejs.begin(), candidate_movejs.end(),
              [&](size_t a, size_t b) {
                if ((a == primary_movej) != (b == primary_movej)) return a == primary_movej;
                bool a_same = (movej_ik_data[a].robot_name == failed_robot_name);
                bool b_same = (movej_ik_data[b].robot_name == failed_robot_name);
                if (a_same != b_same) return a_same;
                return movej_ik_data[a].step_index > movej_ik_data[b].step_index;
              });

            // Pick current backtrack target if not set
            if (current_backtrack_movej == SIZE_MAX && !candidate_movejs.empty())
            {
              current_backtrack_movej = candidate_movejs[0];
            }

            // Try to find next valid untried combination via smart backtracking
            while (current_backtrack_movej != SIZE_MAX && !used_smart_backtrack)
            {
              size_t movej_idx = current_backtrack_movej;
              std::vector<size_t> indices_before = current_ik_indices;

              if (find_next_valid_smart(movej_idx, current_ik_indices))
              {
                used_smart_backtrack = true;
                planning_logger_->recordSmartBacktrack(
                  combinations_tried, plan_result.failed_stage_index, plan_result.failed_stage_name,
                  movej_idx, movej_ik_data[movej_idx].step_index,
                  indices_before[movej_idx], current_ik_indices[movej_idx],
                  indices_before, current_ik_indices);
              }
              else
              {
                // This MoveJ exhausted, try next candidate
                exhausted_movej_indices.insert(movej_idx);
                current_backtrack_movej = SIZE_MAX;
                for (size_t cand : candidate_movejs)
                {
                  if (exhausted_movej_indices.find(cand) == exhausted_movej_indices.end())
                  {
                    current_backtrack_movej = cand;
                    break;
                  }
                }
              }
            }
          }

          // If smart backtracking didn't find anything, we'll pick next untried in next iteration
          if (!used_smart_backtrack)
          {
            last_failed_step = -1;
            exhausted_movej_indices.clear();
            current_backtrack_movej = SIZE_MAX;
          }
        }
      }
      catch (const std::exception& e)
      {
        RCLCPP_DEBUG(get_logger(), "Attempt %zu exception: %s", combinations_tried, e.what());
        last_failed_step = -1;
        exhausted_movej_indices.clear();
        current_backtrack_movej = SIZE_MAX;
      }
    }

    // End phase if not already ended
    if (!phase_ended && !planning_succeeded)
    {
      RCLCPP_DEBUG(get_logger(), "Phase %d completed: %zu/%zu tried", phase, combinations_tried, phase_max);
      planning_logger_->endPhase(combinations_tried, false);
    }
  }

  if (!planning_succeeded)
  {
    RCLCPP_ERROR(get_logger(), "Planning failed after exhausting all combinations");
    planning_logger_->recordPlanningFailure("No IK combination allows feasible paths");
    planning_logger_->finalizeSession();

    result->success = false;
    result->message = "Planning failed: no IK combination allows feasible paths";
    result->failed_step_index = 0;
    goal_handle->abort(result);
    return;
  }

  // Cache solution message ONCE for efficient trajectory execution
  // This avoids repeated toMsg() conversions inside the execution loop
  if (!task_builder_->cacheSolutionMessage(successful_task))
  {
    RCLCPP_ERROR(get_logger(), "Failed to cache solution message");
    planning_logger_->recordPlanningFailure("Failed to cache solution for execution");
    planning_logger_->finalizeSession();

    result->success = false;
    result->message = "Failed to cache solution for execution";
    result->failed_step_index = 0;
    goal_handle->abort(result);
    return;
  }

  // Execute sub-trajectories one at a time, interleaving gripper operations
  size_t num_trajectories = task_builder_->getCachedNumSubTrajectories();
  size_t gripper_op_idx = 0;

  RCLCPP_INFO(get_logger(), "Executing: %zu trajectories, %zu gripper ops", num_trajectories, gripper_operations.size());

  // Start execution timing
  planning_logger_->startExecution(num_trajectories);

  for (size_t traj_idx = 0; traj_idx < num_trajectories; ++traj_idx)
  {
    if (goal_handle->is_canceling())
    {
      RCLCPP_WARN(get_logger(), "Execution canceled by user");
      planning_logger_->endExecution(false, "Canceled by user");
      planning_logger_->finalizeSession();

      result->success = false;
      result->message = "Canceled";
      goal_handle->canceled(result);
      return;
    }

    // Execute any gripper operations that should happen before this trajectory
    while (gripper_op_idx < gripper_operations.size() &&
           gripper_operations[gripper_op_idx].first == traj_idx)
    {
      const auto& gripper_step = gripper_operations[gripper_op_idx].second;
      bool is_open = gripper_step.motion_type == MotionStep::GRIPPER_OPEN;
      bool success = execute_gripper_action(gripper_step.robot_name, is_open);
      RCLCPP_DEBUG(get_logger(), "Gripper %s %s: %s", gripper_step.robot_name.c_str(),
                   is_open ? "open" : "close", success ? "OK" : "FAILED");

      if (!success)
      {
        RCLCPP_ERROR(get_logger(), "Gripper failed before trajectory %zu", traj_idx);
        planning_logger_->endExecution(false, "Gripper operation failed before trajectory " + std::to_string(traj_idx));
        planning_logger_->finalizeSession();

        result->success = false;
        result->message = "Gripper operation failed before trajectory " + std::to_string(traj_idx);
        result->failed_step_index = static_cast<int32_t>(traj_idx);
        goal_handle->abort(result);
        return;
      }
      gripper_op_idx++;
    }

    // Execute this sub-trajectory using cached solution (faster)
    bool traj_success = task_builder_->executeCachedSubTrajectory(traj_idx);
    RCLCPP_DEBUG(get_logger(), "Trajectory %zu/%zu: %s", traj_idx + 1, num_trajectories, traj_success ? "OK" : "FAILED");

    if (!traj_success)
    {
      RCLCPP_ERROR(get_logger(), "Trajectory %zu execution failed", traj_idx);
      planning_logger_->endExecution(false, "Trajectory execution failed at index " + std::to_string(traj_idx));
      planning_logger_->finalizeSession();

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
    bool is_open = gripper_step.motion_type == MotionStep::GRIPPER_OPEN;
    bool success = execute_gripper_action(gripper_step.robot_name, is_open);
    RCLCPP_DEBUG(get_logger(), "Gripper %s %s: %s", gripper_step.robot_name.c_str(),
                 is_open ? "open" : "close", success ? "OK" : "FAILED");

    if (!success)
    {
      RCLCPP_ERROR(get_logger(), "Gripper failed after all trajectories");
      planning_logger_->endExecution(false, "Gripper operation failed after trajectories");
      planning_logger_->finalizeSession();

      result->success = false;
      result->message = "Gripper operation failed after trajectories";
      result->failed_step_index = static_cast<int32_t>(goal->motion_steps.size() - 1);
      goal_handle->abort(result);
      return;
    }
    gripper_op_idx++;
  }

  // Clear cached solution to free memory
  task_builder_->clearSolutionCache();

  // Log successful execution
  planning_logger_->endExecution(true);
  planning_logger_->finalizeSession();
  RCLCPP_INFO(get_logger(), "Session %s completed successfully", session_id.c_str());

  result->success = true;
  result->message = "Completed all steps";
  result->failed_step_index = -1;
  goal_handle->succeed(result);
}

bool MotionSequenceServer::execute_gripper_action(const std::string& robot_name, bool open)
{
  auto client = (robot_name == "robot1") ? robot1_gripper_client_ : robot2_gripper_client_;
  std::string action = open ? "open" : "close";

  if (!client->wait_for_service(std::chrono::seconds(5)))
  {
    RCLCPP_ERROR(get_logger(), "Gripper service not available: %s", robot_name.c_str());
    return false;
  }

  auto request = std::make_shared<RobotiqGripper::Request>();
  request->action = action;

  RCLCPP_DEBUG(get_logger(), "Gripper %s %s", robot_name.c_str(), action.c_str());

  auto future = client->async_send_request(request);

  if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready)
  {
    RCLCPP_ERROR(get_logger(), "Gripper timeout: %s", robot_name.c_str());
    return false;
  }

  auto result = future.get();
  if (!result->success)
  {
    RCLCPP_ERROR(get_logger(), "Gripper failed %s: %s", robot_name.c_str(), result->message.c_str());
    return false;
  }

  return true;
}

void MotionSequenceServer::robot_description_callback(const std_msgs::msg::String::SharedPtr msg)
{
  if (robot_description_received_)
  {
    return;  // Already received, ignore duplicate messages
  }

  RCLCPP_INFO(get_logger(), "Received robot_description from topic (%zu bytes)", msg->data.size());

  // Set the robot_description as a parameter on this node
  // This makes it available to RobotModelLoader
  this->declare_parameter<std::string>("robot_description", msg->data);

  {
    std::lock_guard<std::mutex> lock(robot_description_mutex_);
    robot_description_received_ = true;
  }
  robot_description_cv_.notify_all();
}

bool MotionSequenceServer::wait_for_robot_description(double timeout_sec)
{
  // Wait using condition variable - the executor will call our callback
  std::unique_lock<std::mutex> lock(robot_description_mutex_);
  return robot_description_cv_.wait_for(lock, std::chrono::duration<double>(timeout_sec), [this]() {
    return robot_description_received_.load();
  });
}

void MotionSequenceServer::declareOmplParameters()
{
  // Declare OMPL parameters so MoveIt can find them
  // This must be done at node startup, before the planning pipeline is created
  //
  // MoveIt's OMPL planning context manager expects:
  // - ompl.planner_configs.<name>.type = "geometric::<PlannerType>"
  // - ompl.<group>.planner_configs = ["<name1>", "<name2>", ...]
  // - ompl.<group>.default_planner_config = "<name>"

  auto declare_param = [this](const std::string& name, const rclcpp::ParameterValue& value) {
    if (!this->has_parameter(name))
    {
      this->declare_parameter(name, value);
    }
  };

  RCLCPP_INFO(get_logger(), "Declaring OMPL parameters for planner: %s", planner_config_.ompl_planner_id.c_str());

  // OMPL pipeline configuration
  declare_param("ompl.planning_plugins", rclcpp::ParameterValue(std::vector<std::string>{"ompl_interface/OMPLPlanner"}));
  declare_param("ompl.start_state_max_bounds_error", rclcpp::ParameterValue(0.1));

  // Request/response adapters
  declare_param("ompl.request_adapters", rclcpp::ParameterValue(std::vector<std::string>{
    "default_planning_request_adapters/ResolveConstraintFrames",
    "default_planning_request_adapters/ValidateWorkspaceBounds",
    "default_planning_request_adapters/CheckStartStateBounds",
    "default_planning_request_adapters/CheckStartStateCollision"
  }));
  declare_param("ompl.response_adapters", rclcpp::ParameterValue(std::vector<std::string>{
    "default_planning_response_adapters/AddTimeOptimalParameterization",
    "default_planning_response_adapters/ValidateSolution"
  }));

  // Planner configurations under planner_configs.<name>.*
  // MoveIt looks up planners by name in this structure
  declare_param("ompl.planner_configs.RRTConnect.type", rclcpp::ParameterValue(std::string("geometric::RRTConnect")));
  declare_param("ompl.planner_configs.RRTConnect.range", rclcpp::ParameterValue(0.2));

  declare_param("ompl.planner_configs.RRTstar.type", rclcpp::ParameterValue(std::string("geometric::RRTstar")));
  declare_param("ompl.planner_configs.RRTstar.goal_bias", rclcpp::ParameterValue(0.05));
  declare_param("ompl.planner_configs.RRTstar.range", rclcpp::ParameterValue(0.0));
  declare_param("ompl.planner_configs.RRTstar.delay_collision_checking", rclcpp::ParameterValue(true));

  // Planning timeout - set per group and globally
  declare_param("ompl.planning_time", rclcpp::ParameterValue(planner_config_.timeout_ompl));
  declare_param("ompl.default_planning_time", rclcpp::ParameterValue(planner_config_.timeout_ompl));
  declare_param("ompl.robot1_ur_manipulator.planning_time", rclcpp::ParameterValue(planner_config_.timeout_ompl));
  declare_param("ompl.robot2_ur_manipulator.planning_time", rclcpp::ParameterValue(planner_config_.timeout_ompl));
  declare_param("ompl.robot1_ur_manipulator.longest_valid_segment_fraction", rclcpp::ParameterValue(0.02));
  declare_param("ompl.robot2_ur_manipulator.longest_valid_segment_fraction", rclcpp::ParameterValue(0.02));

  declare_param("ompl.planner_configs.PRMstar.type", rclcpp::ParameterValue(std::string("geometric::PRMstar")));

  // Group configurations - use short names that match planner_configs keys
  std::vector<std::string> planners = {"RRTConnect", "RRTstar", "PRMstar"};

  declare_param("ompl.robot1_ur_manipulator.default_planner_config", rclcpp::ParameterValue(planner_config_.ompl_planner_id));
  declare_param("ompl.robot1_ur_manipulator.planner_configs", rclcpp::ParameterValue(planners));
  declare_param("ompl.robot1_ur_manipulator.projection_evaluator",
    rclcpp::ParameterValue(std::string("joints(robot1_shoulder_pan_joint,robot1_shoulder_lift_joint)")));

  declare_param("ompl.robot2_ur_manipulator.default_planner_config", rclcpp::ParameterValue(planner_config_.ompl_planner_id));
  declare_param("ompl.robot2_ur_manipulator.planner_configs", rclcpp::ParameterValue(planners));
  declare_param("ompl.robot2_ur_manipulator.projection_evaluator",
    rclcpp::ParameterValue(std::string("joints(robot2_shoulder_pan_joint,robot2_shoulder_lift_joint)")));

  auto params = this->list_parameters({"ompl"}, 10);
  RCLCPP_INFO(get_logger(), "Declared %zu OMPL parameters", params.names.size());
}

}  // namespace rsy_mtc_planning

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rsy_mtc_planning::MotionSequenceServer)
