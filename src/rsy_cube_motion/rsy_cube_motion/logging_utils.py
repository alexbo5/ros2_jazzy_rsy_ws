"""
Logging utilities for cube_motion package.
Provides file-based detailed logging with minimal terminal output.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


class CubeMotionLogger:
    """
    Logger for cube motion actions.
    - Terminal: Minimal INFO level output via ROS2 logger
    - File: Detailed DEBUG level logs with timestamps
    """

    def __init__(self, ros_logger, log_dir: str = None):
        """
        Initialize the logger.

        Args:
            ros_logger: ROS2 node logger (from node.get_logger())
            log_dir: Optional custom log directory
        """
        self.ros_logger = ros_logger

        # Determine log directory
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "ros_logging", "cube_motion")

        # Create directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"cube_motion_{timestamp}.log")

        # Setup Python file logger
        self.file_logger = logging.getLogger(f"cube_motion_{timestamp}")
        self.file_logger.setLevel(logging.DEBUG)

        # File handler with detailed format
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(message)s',
                                       datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)

        # Write header
        self.file_logger.info("========================================")
        self.file_logger.info("Cube Motion Log Started")
        self.file_logger.info("========================================")

    def action_start(self, action_name: str, details: str = ""):
        """Log action start (terminal and file)."""
        msg = f"[{action_name}] Started"
        if details:
            msg += f": {details}"
        self.ros_logger.info(msg)
        self.file_logger.info(f">>> {msg}")

    def action_progress(self, action_name: str, step: str):
        """Log action progress (file only)."""
        self.file_logger.info(f"  [{action_name}] {step}")

    def action_complete(self, action_name: str, success: bool, message: str = ""):
        """Log action completion (terminal and file)."""
        status = "OK" if success else "FAILED"
        msg = f"[{action_name}] {status}"
        if message:
            msg += f": {message}"

        if success:
            self.ros_logger.info(msg)
        else:
            self.ros_logger.error(msg)

        self.file_logger.info(f"<<< {msg}")

    def mtc_start(self, num_steps: int):
        """Log MTC sequence start (file only)."""
        self.file_logger.info(f"  MTC sequence: {num_steps} steps")

    def mtc_feedback(self, feedback: str):
        """Log MTC feedback (file only)."""
        self.file_logger.debug(f"    MTC: {feedback}")

    def mtc_result(self, success: bool, message: str = ""):
        """Log MTC result (file only for success, terminal for failure)."""
        if success:
            self.file_logger.info(f"    MTC result: OK - {message}")
        else:
            self.ros_logger.error(f"MTC failed: {message}")
            self.file_logger.error(f"    MTC result: FAILED - {message}")

    def debug(self, message: str):
        """Log debug info (file only)."""
        self.file_logger.debug(f"  [DEBUG] {message}")

    def warn(self, message: str):
        """Log warning (terminal and file)."""
        self.ros_logger.warn(message)
        self.file_logger.warning(f"[WARN] {message}")

    def error(self, message: str):
        """Log error (terminal and file)."""
        self.ros_logger.error(message)
        self.file_logger.error(f"[ERROR] {message}")

    def get_log_file_path(self) -> str:
        """Get the log file path."""
        return self.log_file_path
