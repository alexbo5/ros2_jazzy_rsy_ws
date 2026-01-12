#pragma once

#include <chrono>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <optional>

namespace rsy_mtc_planning
{

/**
 * @brief Record of a single backtrack event during planning
 */
struct BacktrackEvent
{
  size_t attempt_number = 0;
  std::string event_type;  // "smart_backtrack", "standard_advance", "exhausted"
  int failed_stage_index = -1;
  std::string failed_stage_name;
  bool is_movel_failure = false;
  size_t changed_movej_data_index = 0;     // Which MoveJ's IK was changed
  size_t changed_movej_step_index = 0;     // The step index of that MoveJ
  size_t old_ik_index = 0;
  size_t new_ik_index = 0;
  std::vector<size_t> ik_indices_before;   // IK combination before change
  std::vector<size_t> ik_indices_after;    // IK combination after change
};

/**
 * @brief Aggregated backtracking statistics
 */
struct BacktrackingStats
{
  size_t total_attempts = 0;
  size_t smart_backtracks = 0;
  size_t standard_advances = 0;
  std::map<int, size_t> stage_failure_counts;  // stage_index -> failure count
  std::map<size_t, size_t> movej_ik_changes;   // movej_step_index -> times IK was changed
};

/**
 * @brief Metrics collected during a single planning session
 */
struct PlanningMetrics
{
  // Session identification
  std::string session_id;
  std::string timestamp;

  // Input configuration
  size_t total_motion_steps = 0;
  size_t movej_steps = 0;
  size_t movel_steps = 0;
  size_t gripper_operations = 0;
  std::vector<size_t> ik_solutions_per_step;
  size_t total_ik_combinations = 0;

  // Planner configuration
  size_t max_pilz_combinations = 0;
  size_t max_ompl_combinations = 0;
  double timeout_pilz_ptp = 0.0;
  double timeout_ompl = 0.0;
  std::string ompl_planner_id;

  // Planning phase results
  struct PhaseResult
  {
    std::string planner_name;
    size_t attempts = 0;
    size_t max_attempts = 0;
    double duration_ms = 0.0;
    bool succeeded = false;

    // Backtracking statistics for this phase
    BacktrackingStats backtracking;
  };
  std::vector<PhaseResult> phase_results;

  // Backtracking event history (limited to most recent events for memory efficiency)
  std::vector<BacktrackEvent> backtrack_history;
  static constexpr size_t MAX_BACKTRACK_HISTORY = 100;

  // Overall planning result
  bool planning_succeeded = false;
  std::string success_planner;
  size_t success_attempt = 0;
  std::vector<size_t> success_ik_indices;  // The winning IK combination
  double total_planning_duration_ms = 0.0;

  // Execution results
  size_t num_trajectories = 0;
  double total_execution_duration_ms = 0.0;
  bool execution_succeeded = false;
  std::string failure_reason;
};

/**
 * @brief Logger for MTC planning sessions
 *
 * Writes detailed metrics to a log file for analysis.
 * Each planning session creates a new entry in the log.
 * Supports detailed backtracking event logging.
 */
class PlanningLogger
{
public:
  /**
   * @brief Construct a new PlanningLogger
   * @param log_file_path Path to the log file
   */
  explicit PlanningLogger(const std::string& log_file_path);

  ~PlanningLogger();

  /**
   * @brief Start a new planning session
   * @return Session ID for this planning session
   */
  std::string startSession();

  /**
   * @brief Record input configuration
   */
  void recordInput(
    size_t total_motion_steps,
    size_t movej_steps,
    size_t movel_steps,
    size_t gripper_operations,
    const std::vector<size_t>& ik_solutions_per_step,
    size_t total_ik_combinations);

  /**
   * @brief Record planner configuration
   */
  void recordPlannerConfig(
    size_t max_pilz_combinations,
    size_t max_ompl_combinations,
    double timeout_pilz_ptp,
    double timeout_ompl,
    const std::string& ompl_planner_id);

  /**
   * @brief Start timing a planning phase
   * @param planner_name Name of the planner (e.g., "Pilz-PTP", "OMPL")
   * @param max_attempts Maximum attempts for this phase
   */
  void startPhase(const std::string& planner_name, size_t max_attempts);

  /**
   * @brief End the current planning phase
   * @param attempts Number of attempts made
   * @param succeeded Whether the phase succeeded
   */
  void endPhase(size_t attempts, bool succeeded);

  /**
   * @brief Record a smart backtrack event (MoveL failed, changing preceding MoveJ's IK)
   * @param attempt_number Current attempt number
   * @param failed_stage_index Index of the failed MoveL stage
   * @param failed_stage_name Name of the failed stage
   * @param movej_data_index Index in movej_ik_data that was changed
   * @param movej_step_index The step index of the MoveJ whose IK was changed
   * @param old_ik_index Previous IK index
   * @param new_ik_index New IK index
   * @param ik_indices_before IK indices before change
   * @param ik_indices_after IK indices after change
   */
  void recordSmartBacktrack(
    size_t attempt_number,
    int failed_stage_index,
    const std::string& failed_stage_name,
    size_t movej_data_index,
    size_t movej_step_index,
    size_t old_ik_index,
    size_t new_ik_index,
    const std::vector<size_t>& ik_indices_before,
    const std::vector<size_t>& ik_indices_after);

  /**
   * @brief Record a standard IK combination advance
   * @param attempt_number Current attempt number
   * @param ik_indices_before IK indices before advance
   * @param ik_indices_after IK indices after advance
   */
  void recordStandardAdvance(
    size_t attempt_number,
    const std::vector<size_t>& ik_indices_before,
    const std::vector<size_t>& ik_indices_after);

  /**
   * @brief Record a stage failure (for statistics)
   * @param stage_index Index of the failed stage
   * @param stage_name Name of the failed stage
   * @param is_movel Whether the failed stage is a MoveL
   */
  void recordStageFailure(int stage_index, const std::string& stage_name, bool is_movel);

  /**
   * @brief Record planning success
   * @param planner_name The planner that succeeded
   * @param attempt_number The attempt number that succeeded
   * @param success_ik_indices The IK combination that succeeded
   */
  void recordPlanningSuccess(
    const std::string& planner_name,
    size_t attempt_number,
    const std::vector<size_t>& success_ik_indices = {});

  /**
   * @brief Record planning failure
   * @param reason Reason for failure
   */
  void recordPlanningFailure(const std::string& reason);

  /**
   * @brief Start timing execution phase
   * @param num_trajectories Number of trajectories to execute
   */
  void startExecution(size_t num_trajectories);

  /**
   * @brief End execution and record results
   * @param succeeded Whether execution succeeded
   * @param failure_reason Reason for failure if not succeeded
   */
  void endExecution(bool succeeded, const std::string& failure_reason = "");

  /**
   * @brief Finalize and write the session to the log file
   */
  void finalizeSession();

  /**
   * @brief Get the current metrics (for testing/debugging)
   */
  const PlanningMetrics& getMetrics() const { return current_metrics_; }

private:
  std::string log_file_path_;
  std::mutex file_mutex_;

  PlanningMetrics current_metrics_;

  // Timing helpers
  std::chrono::steady_clock::time_point planning_start_time_;
  std::chrono::steady_clock::time_point phase_start_time_;
  std::chrono::steady_clock::time_point execution_start_time_;

  std::string current_phase_planner_;
  size_t current_phase_max_attempts_ = 0;
  BacktrackingStats current_phase_backtracking_;

  // Helper to add backtrack event (with size limiting)
  void addBacktrackEvent(const BacktrackEvent& event);

  // Generate unique session ID
  std::string generateSessionId();

  // Get ISO 8601 timestamp
  std::string getCurrentTimestamp();

  // Write metrics to file
  void writeToFile(const PlanningMetrics& metrics);

  // Convert metrics to formatted string
  std::string metricsToString(const PlanningMetrics& metrics);
};

}  // namespace rsy_mtc_planning
