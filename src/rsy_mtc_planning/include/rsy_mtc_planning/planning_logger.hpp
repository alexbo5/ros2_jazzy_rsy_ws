#pragma once

#include <rclcpp/rclcpp.hpp>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <mutex>

namespace rsy_mtc_planning
{

/**
 * @brief Planning statistics for a single planning session
 */
struct PlanningStats
{
  // Phase info
  int current_phase = 0;           // 1 = Pilz, 2 = OMPL
  std::string planner_name;        // "Pilz-PTP" or "OMPL"

  // IK exploration
  size_t num_movej_steps = 0;      // Number of MoveJ steps
  size_t total_ik_combinations = 0; // Total IK combinations possible

  // Progress
  size_t phase1_attempts = 0;      // Combinations tried in phase 1
  size_t phase2_attempts = 0;      // Combinations tried in phase 2
  size_t total_attempts = 0;       // Total combinations tried

  // Timing
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;

  // Result
  bool success = false;
  size_t successful_combination = 0;
  std::string success_planner;     // Which planner succeeded
  std::vector<size_t> success_ik_indices; // IK indices that worked

  // Per-step IK counts
  std::vector<size_t> ik_counts_per_step;

  double elapsed_seconds() const
  {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    return duration.count() / 1000.0;
  }
};

/**
 * @brief Logger for MTC planning that provides:
 *   - Minimal terminal output (INFO level only for critical events)
 *   - Detailed file logging with planning statistics
 *   - ROS2 conformant logging
 */
class PlanningLogger
{
public:
  explicit PlanningLogger(rclcpp::Logger ros_logger, const std::string& log_dir = "")
    : ros_logger_(ros_logger)
  {
    // Determine log directory
    if (log_dir.empty())
    {
      // Use ros_logging in current working directory
      log_dir_ = std::filesystem::current_path().string() + "/ros_logging/mtc_planning";
    }
    else
    {
      log_dir_ = log_dir;
    }

    // Create directory if it doesn't exist
    std::filesystem::create_directories(log_dir_);

    // Create log file with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << log_dir_ << "/mtc_planning_"
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
    log_file_path_ = ss.str();

    log_file_.open(log_file_path_, std::ios::out | std::ios::app);
    if (log_file_.is_open())
    {
      logToFile("========================================");
      logToFile("MTC Planning Log Started");
      logToFile("========================================");
    }
  }

  ~PlanningLogger()
  {
    if (log_file_.is_open())
    {
      logToFile("========================================");
      logToFile("MTC Planning Log Ended");
      logToFile("========================================");
      log_file_.close();
    }
  }

  /**
   * @brief Start a new planning session
   */
  void startPlanningSession(size_t num_motion_steps, size_t num_gripper_ops)
  {
    stats_ = PlanningStats();
    stats_.start_time = std::chrono::steady_clock::now();

    // Terminal: Single line start message
    RCLCPP_INFO(ros_logger_, "Planning %zu motion steps...", num_motion_steps);

    // File: Detailed info
    logToFile("----------------------------------------");
    logToFile("NEW PLANNING SESSION");
    logToFile("Motion steps: " + std::to_string(num_motion_steps));
    logToFile("Gripper operations: " + std::to_string(num_gripper_ops));
  }

  /**
   * @brief Log IK computation results
   */
  void logIKComputation(size_t num_movej_steps, size_t total_combinations,
                        const std::vector<size_t>& ik_counts)
  {
    stats_.num_movej_steps = num_movej_steps;
    stats_.total_ik_combinations = total_combinations;
    stats_.ik_counts_per_step = ik_counts;

    // Terminal: Compact summary
    RCLCPP_INFO(ros_logger_, "IK: %zu MoveJ steps, %zu combinations",
                num_movej_steps, total_combinations);

    // File: Detailed per-step IK counts
    logToFile("IK Computation:");
    logToFile("  MoveJ steps: " + std::to_string(num_movej_steps));
    logToFile("  Total combinations: " + std::to_string(total_combinations));
    std::string ik_detail = "  IK per step: [";
    for (size_t i = 0; i < ik_counts.size(); ++i)
    {
      ik_detail += std::to_string(ik_counts[i]);
      if (i < ik_counts.size() - 1) ik_detail += ", ";
    }
    ik_detail += "]";
    logToFile(ik_detail);
  }

  /**
   * @brief Log phase start
   */
  void logPhaseStart(int phase, const std::string& planner_name, size_t max_attempts)
  {
    stats_.current_phase = phase;
    stats_.planner_name = planner_name;

    // Terminal: Single line
    RCLCPP_INFO(ros_logger_, "Phase %d: %s (max %zu)", phase, planner_name.c_str(), max_attempts);

    // File: More detail
    logToFile("--- Phase " + std::to_string(phase) + " ---");
    logToFile("  Planner: " + planner_name);
    logToFile("  Max attempts: " + std::to_string(max_attempts));
  }

  /**
   * @brief Log planning attempt (file only, called frequently)
   */
  void logAttempt(size_t attempt_num, const std::vector<size_t>& ik_indices, bool log_to_terminal = false)
  {
    stats_.total_attempts++;
    if (stats_.current_phase == 1)
    {
      stats_.phase1_attempts++;
    }
    else
    {
      stats_.phase2_attempts++;
    }

    // Build IK indices string
    std::string indices_str = "[";
    for (size_t i = 0; i < ik_indices.size(); ++i)
    {
      indices_str += std::to_string(ik_indices[i]);
      if (i < ik_indices.size() - 1) indices_str += ",";
    }
    indices_str += "]";

    // Terminal: Only every 100 attempts or if requested
    if (log_to_terminal || attempt_num % 100 == 0)
    {
      RCLCPP_INFO(ros_logger_, "[%s] %zu/%zu %s",
                  stats_.planner_name.c_str(), attempt_num,
                  stats_.total_ik_combinations, indices_str.c_str());
    }

    // File: Every attempt
    logToFile("  Attempt " + std::to_string(attempt_num) + ": " + indices_str);
  }

  /**
   * @brief Log planning failure for an attempt (file only)
   */
  void logAttemptFailed(size_t /*attempt_num*/, const std::string& reason = "")
  {
    if (!reason.empty())
    {
      logToFile("    FAILED: " + reason);
    }
  }

  /**
   * @brief Log phase exhausted
   */
  void logPhaseExhausted(int phase, size_t attempts)
  {
    // Terminal: Single line
    RCLCPP_INFO(ros_logger_, "Phase %d exhausted after %zu attempts", phase, attempts);

    // File
    logToFile("  Phase " + std::to_string(phase) + " exhausted: " + std::to_string(attempts) + " attempts");
  }

  /**
   * @brief Log planning success
   */
  void logSuccess(const std::string& planner_name, size_t combination_num,
                  const std::vector<size_t>& ik_indices)
  {
    stats_.success = true;
    stats_.success_planner = planner_name;
    stats_.successful_combination = combination_num;
    stats_.success_ik_indices = ik_indices;
    stats_.end_time = std::chrono::steady_clock::now();

    // Build summary
    std::string indices_str = "[";
    for (size_t i = 0; i < ik_indices.size(); ++i)
    {
      indices_str += std::to_string(ik_indices[i]);
      if (i < ik_indices.size() - 1) indices_str += ",";
    }
    indices_str += "]";

    // Terminal: Success message with key info
    RCLCPP_INFO(ros_logger_, "SUCCESS: %s @ #%zu %s (%.2fs, %zu total attempts)",
                planner_name.c_str(), combination_num, indices_str.c_str(),
                stats_.elapsed_seconds(), stats_.total_attempts);

    // File: Full summary
    logToFile("========================================");
    logToFile("PLANNING SUCCEEDED");
    logToFile("  Planner: " + planner_name);
    logToFile("  Combination: " + std::to_string(combination_num));
    logToFile("  IK indices: " + indices_str);
    logToFile("  Phase 1 attempts: " + std::to_string(stats_.phase1_attempts));
    logToFile("  Phase 2 attempts: " + std::to_string(stats_.phase2_attempts));
    logToFile("  Total attempts: " + std::to_string(stats_.total_attempts));
    logToFile("  Duration: " + std::to_string(stats_.elapsed_seconds()) + "s");
    logToFile("========================================");
  }

  /**
   * @brief Log planning failure
   */
  void logFailure()
  {
    stats_.success = false;
    stats_.end_time = std::chrono::steady_clock::now();

    // Terminal: Error message
    RCLCPP_ERROR(ros_logger_, "FAILED: %zu attempts in %.2fs (P1: %zu, P2: %zu)",
                 stats_.total_attempts, stats_.elapsed_seconds(),
                 stats_.phase1_attempts, stats_.phase2_attempts);

    // File: Full summary
    logToFile("========================================");
    logToFile("PLANNING FAILED");
    logToFile("  Phase 1 attempts: " + std::to_string(stats_.phase1_attempts));
    logToFile("  Phase 2 attempts: " + std::to_string(stats_.phase2_attempts));
    logToFile("  Total attempts: " + std::to_string(stats_.total_attempts));
    logToFile("  Duration: " + std::to_string(stats_.elapsed_seconds()) + "s");
    logToFile("========================================");
  }

  /**
   * @brief Log execution start
   */
  void logExecutionStart(size_t num_trajectories, size_t num_gripper_ops)
  {
    RCLCPP_INFO(ros_logger_, "Executing: %zu trajectories, %zu gripper ops",
                num_trajectories, num_gripper_ops);
    logToFile("Execution: " + std::to_string(num_trajectories) + " trajectories, " +
              std::to_string(num_gripper_ops) + " gripper ops");
  }

  /**
   * @brief Log trajectory execution (file only)
   */
  void logTrajectoryExecution(size_t idx, size_t total, bool success)
  {
    logToFile("  Trajectory " + std::to_string(idx + 1) + "/" + std::to_string(total) +
              ": " + (success ? "OK" : "FAILED"));
  }

  /**
   * @brief Log gripper operation (file only)
   */
  void logGripperOp(const std::string& robot, const std::string& action, bool success)
  {
    logToFile("  Gripper " + robot + " " + action + ": " + (success ? "OK" : "FAILED"));
  }

  /**
   * @brief Log general debug info (file only)
   */
  void debug(const std::string& msg)
  {
    logToFile("[DEBUG] " + msg);
  }

  /**
   * @brief Log warning (both terminal and file)
   */
  void warn(const std::string& msg)
  {
    RCLCPP_WARN(ros_logger_, "%s", msg.c_str());
    logToFile("[WARN] " + msg);
  }

  /**
   * @brief Log error (both terminal and file)
   */
  void error(const std::string& msg)
  {
    RCLCPP_ERROR(ros_logger_, "%s", msg.c_str());
    logToFile("[ERROR] " + msg);
  }

  /**
   * @brief Get log file path
   */
  std::string getLogFilePath() const { return log_file_path_; }

  /**
   * @brief Get current statistics
   */
  const PlanningStats& getStats() const { return stats_; }

private:
  void logToFile(const std::string& msg)
  {
    if (!log_file_.is_open()) return;

    std::lock_guard<std::mutex> lock(file_mutex_);

    // Add timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;

    log_file_ << std::put_time(std::localtime(&time_t), "%H:%M:%S")
              << "." << std::setfill('0') << std::setw(3) << ms.count()
              << " " << msg << std::endl;
  }

  rclcpp::Logger ros_logger_;
  std::string log_dir_;
  std::string log_file_path_;
  std::ofstream log_file_;
  std::mutex file_mutex_;
  PlanningStats stats_;
};

}  // namespace rsy_mtc_planning
