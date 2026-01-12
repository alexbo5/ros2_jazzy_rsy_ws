#include "rsy_mtc_planning/planning_logger.hpp"

#include <iomanip>
#include <random>
#include <sstream>
#include <ctime>
#include <iostream>
#include <filesystem>

namespace rsy_mtc_planning
{

PlanningLogger::PlanningLogger(const std::string& log_file_path)
  : log_file_path_(log_file_path)
{
  // Ensure the directory exists
  std::filesystem::path path(log_file_path_);
  std::filesystem::path dir = path.parent_path();
  if (!dir.empty() && !std::filesystem::exists(dir))
  {
    std::filesystem::create_directories(dir);
  }
}

PlanningLogger::~PlanningLogger() = default;

std::string PlanningLogger::startSession()
{
  current_metrics_ = PlanningMetrics{};
  current_metrics_.session_id = generateSessionId();
  current_metrics_.timestamp = getCurrentTimestamp();
  planning_start_time_ = std::chrono::steady_clock::now();
  return current_metrics_.session_id;
}

void PlanningLogger::recordInput(
  size_t total_motion_steps,
  size_t movej_steps,
  size_t movel_steps,
  size_t gripper_operations,
  const std::vector<size_t>& ik_solutions_per_step,
  size_t total_ik_combinations)
{
  current_metrics_.total_motion_steps = total_motion_steps;
  current_metrics_.movej_steps = movej_steps;
  current_metrics_.movel_steps = movel_steps;
  current_metrics_.gripper_operations = gripper_operations;
  current_metrics_.ik_solutions_per_step = ik_solutions_per_step;
  current_metrics_.total_ik_combinations = total_ik_combinations;
}

void PlanningLogger::recordMoveJIKInfo(const std::vector<MoveJIKInfo>& movej_ik_info)
{
  current_metrics_.movej_ik_info = movej_ik_info;
}

void PlanningLogger::recordAttempt(const AttemptRecord& attempt)
{
  // Limit history size for memory efficiency
  if (current_metrics_.attempt_history.size() >= PlanningMetrics::MAX_ATTEMPT_HISTORY)
  {
    // Remove oldest attempt
    current_metrics_.attempt_history.erase(current_metrics_.attempt_history.begin());
  }
  current_metrics_.attempt_history.push_back(attempt);
}

void PlanningLogger::recordPlannerConfig(
  size_t max_pilz_combinations,
  size_t max_ompl_combinations,
  double timeout_pilz_ptp,
  double timeout_ompl,
  const std::string& ompl_planner_id)
{
  current_metrics_.max_pilz_combinations = max_pilz_combinations;
  current_metrics_.max_ompl_combinations = max_ompl_combinations;
  current_metrics_.timeout_pilz_ptp = timeout_pilz_ptp;
  current_metrics_.timeout_ompl = timeout_ompl;
  current_metrics_.ompl_planner_id = ompl_planner_id;
}

void PlanningLogger::startPhase(const std::string& planner_name, size_t max_attempts)
{
  phase_start_time_ = std::chrono::steady_clock::now();
  current_phase_planner_ = planner_name;
  current_phase_max_attempts_ = max_attempts;
  current_phase_backtracking_ = BacktrackingStats{};  // Reset phase stats
}

void PlanningLogger::endPhase(size_t attempts, bool succeeded)
{
  auto phase_end_time = std::chrono::steady_clock::now();
  double duration_ms = std::chrono::duration<double, std::milli>(
    phase_end_time - phase_start_time_).count();

  PlanningMetrics::PhaseResult result;
  result.planner_name = current_phase_planner_;
  result.attempts = attempts;
  result.max_attempts = current_phase_max_attempts_;
  result.duration_ms = duration_ms;
  result.succeeded = succeeded;
  result.backtracking = current_phase_backtracking_;  // Include backtracking stats

  current_metrics_.phase_results.push_back(result);
}

void PlanningLogger::addBacktrackEvent(const BacktrackEvent& event)
{
  // Limit history size for memory efficiency
  if (current_metrics_.backtrack_history.size() >= PlanningMetrics::MAX_BACKTRACK_HISTORY)
  {
    // Remove oldest event
    current_metrics_.backtrack_history.erase(current_metrics_.backtrack_history.begin());
  }
  current_metrics_.backtrack_history.push_back(event);
}

void PlanningLogger::recordSmartBacktrack(
  size_t attempt_number,
  int failed_stage_index,
  const std::string& failed_stage_name,
  size_t movej_data_index,
  size_t movej_step_index,
  size_t old_ik_index,
  size_t new_ik_index,
  const std::vector<size_t>& ik_indices_before,
  const std::vector<size_t>& ik_indices_after)
{
  BacktrackEvent event;
  event.attempt_number = attempt_number;
  event.event_type = "smart_backtrack";
  event.failed_stage_index = failed_stage_index;
  event.failed_stage_name = failed_stage_name;
  event.is_movel_failure = true;
  event.changed_movej_data_index = movej_data_index;
  event.changed_movej_step_index = movej_step_index;
  event.old_ik_index = old_ik_index;
  event.new_ik_index = new_ik_index;
  event.ik_indices_before = ik_indices_before;
  event.ik_indices_after = ik_indices_after;

  addBacktrackEvent(event);

  // Update phase statistics
  current_phase_backtracking_.total_attempts++;
  current_phase_backtracking_.smart_backtracks++;
  current_phase_backtracking_.movej_ik_changes[movej_step_index]++;
}

void PlanningLogger::recordStandardAdvance(
  size_t attempt_number,
  const std::vector<size_t>& ik_indices_before,
  const std::vector<size_t>& ik_indices_after)
{
  BacktrackEvent event;
  event.attempt_number = attempt_number;
  event.event_type = "standard_advance";
  event.ik_indices_before = ik_indices_before;
  event.ik_indices_after = ik_indices_after;

  addBacktrackEvent(event);

  // Update phase statistics
  current_phase_backtracking_.total_attempts++;
  current_phase_backtracking_.standard_advances++;
}

void PlanningLogger::recordStageFailure(int stage_index, const std::string& stage_name, bool is_movel)
{
  (void)stage_name;  // Used for debugging if needed
  (void)is_movel;
  current_phase_backtracking_.stage_failure_counts[stage_index]++;
}

void PlanningLogger::recordPlanningSuccess(
  const std::string& planner_name,
  size_t attempt_number,
  const std::vector<size_t>& success_ik_indices)
{
  auto planning_end_time = std::chrono::steady_clock::now();
  current_metrics_.total_planning_duration_ms = std::chrono::duration<double, std::milli>(
    planning_end_time - planning_start_time_).count();

  current_metrics_.planning_succeeded = true;
  current_metrics_.success_planner = planner_name;
  current_metrics_.success_attempt = attempt_number;
  current_metrics_.success_ik_indices = success_ik_indices;
}

void PlanningLogger::recordPlanningFailure(const std::string& reason)
{
  auto planning_end_time = std::chrono::steady_clock::now();
  current_metrics_.total_planning_duration_ms = std::chrono::duration<double, std::milli>(
    planning_end_time - planning_start_time_).count();

  current_metrics_.planning_succeeded = false;
  current_metrics_.failure_reason = reason;
}

void PlanningLogger::startExecution(size_t num_trajectories)
{
  execution_start_time_ = std::chrono::steady_clock::now();
  current_metrics_.num_trajectories = num_trajectories;
}

void PlanningLogger::endExecution(bool succeeded, const std::string& failure_reason)
{
  auto execution_end_time = std::chrono::steady_clock::now();
  current_metrics_.total_execution_duration_ms = std::chrono::duration<double, std::milli>(
    execution_end_time - execution_start_time_).count();

  current_metrics_.execution_succeeded = succeeded;
  if (!succeeded && !failure_reason.empty())
  {
    current_metrics_.failure_reason = failure_reason;
  }
}

void PlanningLogger::finalizeSession()
{
  writeToFile(current_metrics_);
}

std::string PlanningLogger::generateSessionId()
{
  // Generate a random 8-character hex ID
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);

  std::stringstream ss;
  for (int i = 0; i < 8; ++i)
  {
    ss << std::hex << dis(gen);
  }
  return ss.str();
}

std::string PlanningLogger::getCurrentTimestamp()
{
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    now.time_since_epoch()) % 1000;

  std::tm tm_buf;
  localtime_r(&time_t_now, &tm_buf);

  std::stringstream ss;
  ss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
  ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return ss.str();
}

void PlanningLogger::writeToFile(const PlanningMetrics& metrics)
{
  std::lock_guard<std::mutex> lock(file_mutex_);

  // Open file in append mode
  std::ofstream file(log_file_path_, std::ios::app);
  if (!file.is_open())
  {
    std::cerr << "PlanningLogger: Failed to open log file: " << log_file_path_ << std::endl;
    return;
  }

  file << metricsToString(metrics) << std::endl;
  file.flush();
  file.close();
}

std::string PlanningLogger::metricsToString(const PlanningMetrics& metrics)
{
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2);

  // Header with separator
  ss << "\n";
  ss << "================================================================================\n";
  ss << "PLANNING SESSION: " << metrics.session_id << "\n";
  ss << "Timestamp: " << metrics.timestamp << "\n";
  ss << "================================================================================\n";

  // Input summary
  ss << "\n[INPUT]\n";
  ss << "  Total motion steps:    " << metrics.total_motion_steps << "\n";
  ss << "  MoveJ steps:           " << metrics.movej_steps << "\n";
  ss << "  MoveL steps:           " << metrics.movel_steps << "\n";
  ss << "  Gripper operations:    " << metrics.gripper_operations << "\n";
  ss << "  IK solutions/step:     [";
  for (size_t i = 0; i < metrics.ik_solutions_per_step.size(); ++i)
  {
    if (i > 0) ss << ", ";
    ss << metrics.ik_solutions_per_step[i];
  }
  ss << "]\n";
  ss << "  Total IK combinations: " << metrics.total_ik_combinations << "\n";

  // Detailed MoveJ IK information
  if (!metrics.movej_ik_info.empty())
  {
    ss << "\n[MOVEJ IK DETAILS]\n";
    for (const auto& ik_info : metrics.movej_ik_info)
    {
      ss << "  Step " << ik_info.step_index << " (" << ik_info.robot_name << "):\n";
      ss << "    Total IK found:       " << ik_info.total_ik_found << "\n";
      ss << "    Collision-free IK:    " << ik_info.collision_free_ik << "\n";
    }
  }

  // Planner configuration
  ss << "\n[CONFIG]\n";
  ss << "  Max Pilz combinations: " << metrics.max_pilz_combinations << "\n";
  ss << "  Max OMPL combinations: " << metrics.max_ompl_combinations << "\n";
  ss << "  Pilz PTP timeout:      " << metrics.timeout_pilz_ptp << " s\n";
  ss << "  OMPL timeout:          " << metrics.timeout_ompl << " s\n";
  ss << "  OMPL planner:          " << metrics.ompl_planner_id << "\n";

  // Phase results with backtracking stats
  ss << "\n[PLANNING PHASES]\n";
  for (size_t i = 0; i < metrics.phase_results.size(); ++i)
  {
    const auto& phase = metrics.phase_results[i];
    ss << "  Phase " << (i + 1) << ": " << phase.planner_name << "\n";
    ss << "    Attempts:         " << phase.attempts << " / " << phase.max_attempts << "\n";
    ss << "    Duration:         " << phase.duration_ms << " ms\n";
    ss << "    Result:           " << (phase.succeeded ? "SUCCESS" : "FAILED") << "\n";

    // Backtracking statistics for this phase
    const auto& bt = phase.backtracking;
    if (bt.total_attempts > 0 || bt.smart_backtracks > 0 || bt.standard_advances > 0)
    {
      ss << "    [Backtracking]\n";
      ss << "      Smart backtracks:   " << bt.smart_backtracks << "\n";
      ss << "      Standard advances:  " << bt.standard_advances << "\n";

      // Stage failure distribution
      if (!bt.stage_failure_counts.empty())
      {
        ss << "      Stage failures:     {";
        bool first = true;
        for (const auto& [stage_idx, count] : bt.stage_failure_counts)
        {
          if (!first) ss << ", ";
          ss << "step" << stage_idx << ":" << count;
          first = false;
        }
        ss << "}\n";
      }

      // MoveJ IK changes distribution
      if (!bt.movej_ik_changes.empty())
      {
        ss << "      MoveJ IK changes:   {";
        bool first = true;
        for (const auto& [step_idx, count] : bt.movej_ik_changes)
        {
          if (!first) ss << ", ";
          ss << "step" << step_idx << ":" << count;
          first = false;
        }
        ss << "}\n";
      }
    }
  }

  // Planning result
  ss << "\n[PLANNING RESULT]\n";
  ss << "  Status:          " << (metrics.planning_succeeded ? "SUCCESS" : "FAILED") << "\n";
  if (metrics.planning_succeeded)
  {
    ss << "  Success planner: " << metrics.success_planner << "\n";
    ss << "  Success attempt: " << metrics.success_attempt << "\n";
    if (!metrics.success_ik_indices.empty())
    {
      ss << "  Success IK idx:  [";
      for (size_t i = 0; i < metrics.success_ik_indices.size(); ++i)
      {
        if (i > 0) ss << ", ";
        ss << metrics.success_ik_indices[i];
      }
      ss << "]\n";
    }
  }
  ss << "  Total duration:  " << metrics.total_planning_duration_ms << " ms\n";

  // Backtracking history (last N events for debugging)
  if (!metrics.backtrack_history.empty())
  {
    ss << "\n[BACKTRACK HISTORY] (last " << metrics.backtrack_history.size() << " events)\n";

    // Show last 10 events in detail
    size_t start_idx = metrics.backtrack_history.size() > 10 ?
                       metrics.backtrack_history.size() - 10 : 0;

    for (size_t i = start_idx; i < metrics.backtrack_history.size(); ++i)
    {
      const auto& evt = metrics.backtrack_history[i];
      ss << "  #" << evt.attempt_number << " " << evt.event_type;
      if (evt.event_type == "smart_backtrack")
      {
        ss << ": MoveL@" << evt.failed_stage_index << " failed"
           << " -> changed MoveJ@" << evt.changed_movej_step_index
           << " IK " << evt.old_ik_index << "->" << evt.new_ik_index;
      }
      ss << "\n";
    }
  }

  // Failure analysis summary (aggregated statistics from attempt history)
  if (!metrics.attempt_history.empty() && !metrics.planning_succeeded)
  {
    ss << "\n[FAILURE ANALYSIS]\n";

    // Count failures by type
    size_t collision_failures = 0;
    size_t planning_failures = 0;
    std::map<int, size_t> failures_by_step;  // step_index -> count
    std::map<std::string, size_t> failures_by_type;  // "MoveJ"/"MoveL" -> count

    for (const auto& attempt : metrics.attempt_history)
    {
      if (attempt.planning_succeeded) continue;

      if (!attempt.collision_precheck_passed)
      {
        collision_failures++;
        if (attempt.collision_step >= 0)
        {
          failures_by_step[attempt.collision_step]++;
        }
      }
      else if (attempt.planning_attempted)
      {
        planning_failures++;
        if (attempt.failed_step_index >= 0)
        {
          failures_by_step[attempt.failed_step_index]++;
        }
        if (!attempt.failed_step_type.empty())
        {
          failures_by_type[attempt.failed_step_type]++;
        }
      }
    }

    ss << "  Collision pre-check failures: " << collision_failures << "\n";
    ss << "  Planning failures:            " << planning_failures << "\n";

    if (!failures_by_step.empty())
    {
      ss << "  Failures by step:\n";
      for (const auto& [step, count] : failures_by_step)
      {
        ss << "    Step " << step << ": " << count << " failures\n";
      }
    }

    if (!failures_by_type.empty())
    {
      ss << "  Failures by motion type:\n";
      for (const auto& [type, count] : failures_by_type)
      {
        ss << "    " << type << ": " << count << " failures\n";
      }
    }
  }

  // Attempt history (detailed log of every tried solution)
  if (!metrics.attempt_history.empty())
  {
    ss << "\n[ATTEMPT HISTORY] (" << metrics.attempt_history.size() << " attempts)\n";
    ss << "  Format: #attempt [phase] IK=[indices] -> result\n";
    ss << "  ---------------------------------------------------------------\n";

    for (const auto& attempt : metrics.attempt_history)
    {
      ss << "  #" << std::setw(3) << attempt.attempt_number << " ";
      ss << "[" << std::setw(8) << attempt.phase_name << "] ";

      // IK indices used
      ss << "IK=[";
      for (size_t i = 0; i < attempt.ik_indices.size(); ++i)
      {
        if (i > 0) ss << ",";
        ss << attempt.ik_indices[i];
      }
      ss << "] -> ";

      // Result
      if (attempt.planning_succeeded)
      {
        ss << "SUCCESS\n";
      }
      else
      {
        // Show failure details
        if (!attempt.collision_precheck_passed)
        {
          ss << "COLLISION at step " << attempt.collision_step << "\n";
        }
        else if (!attempt.planning_attempted)
        {
          ss << "SKIPPED (precheck failed)\n";
        }
        else
        {
          ss << "FAILED: " << attempt.failed_step_type;
          if (attempt.failed_step_index >= 0)
          {
            ss << " at step " << attempt.failed_step_index;
          }
          if (!attempt.failed_step_name.empty())
          {
            ss << " (" << attempt.failed_step_name << ")";
          }
          if (!attempt.failure_reason.empty())
          {
            ss << " - " << attempt.failure_reason;
          }
          ss << "\n";
        }
      }
    }
  }

  // Execution result
  ss << "\n[EXECUTION]\n";
  ss << "  Trajectories:    " << metrics.num_trajectories << "\n";
  ss << "  Duration:        " << metrics.total_execution_duration_ms << " ms\n";
  ss << "  Status:          " << (metrics.execution_succeeded ? "SUCCESS" : "FAILED") << "\n";

  // Failure reason (if any)
  if (!metrics.failure_reason.empty())
  {
    ss << "\n[FAILURE]\n";
    ss << "  Reason: " << metrics.failure_reason << "\n";
  }

  // Summary line for quick analysis
  ss << "\n[SUMMARY]\n";
  ss << "  " << metrics.timestamp << " | ";
  ss << (metrics.planning_succeeded ? "PLAN_OK" : "PLAN_FAIL") << " | ";
  ss << (metrics.execution_succeeded ? "EXEC_OK" : "EXEC_FAIL") << " | ";
  ss << "Steps:" << metrics.total_motion_steps << " | ";
  ss << "IK:" << metrics.total_ik_combinations << " | ";
  if (metrics.planning_succeeded)
  {
    ss << metrics.success_planner << "@" << metrics.success_attempt << " | ";
  }

  // Aggregate backtracking stats across phases
  size_t total_smart_bt = 0, total_std_adv = 0;
  for (const auto& phase : metrics.phase_results)
  {
    total_smart_bt += phase.backtracking.smart_backtracks;
    total_std_adv += phase.backtracking.standard_advances;
  }
  if (total_smart_bt > 0 || total_std_adv > 0)
  {
    ss << "BT:" << total_smart_bt << "/" << total_std_adv << " | ";
  }

  ss << "Plan:" << metrics.total_planning_duration_ms << "ms | ";
  ss << "Exec:" << metrics.total_execution_duration_ms << "ms\n";

  ss << "================================================================================\n";

  return ss.str();
}

}  // namespace rsy_mtc_planning
