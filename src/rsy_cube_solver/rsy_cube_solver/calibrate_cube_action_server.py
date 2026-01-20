#!/usr/bin/env python3
"""
CalibrateCube Action Server for rsy_cube_solver package.

This action server orchestrates cube color calibration by:
1. Taking up the cube
2. Presenting each face to the camera
3. Allowing user to define ROI and sample colors via GUI
4. Computing HSV ranges from samples
5. Saving calibration data persistently
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.executors import MultiThreadedExecutor

import cv2
import numpy as np
import json
import threading
import time
from typing import Dict, List, Optional
from pathlib import Path

from rsy_cube_solver.action import CalibrateCube
from rsy_cube_motion.action import TakeUpCube, PutDownCube, PresentCubeFace

# Face order for calibration (matches standard cube notation)
FACE_ORDER = ["U", "D", "F", "B", "L", "R"]

# Color names corresponding to each face's center on a solved cube
FACE_CENTER_COLORS = {
    "U": "white",
    "D": "yellow",
    "F": "green",
    "B": "blue",
    "L": "orange",
    "R": "red",
}

def _try_create_dir(path: Path) -> bool:
    """Try to create directory, return True if successful."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_calibration_file_path() -> Path:
    """Get calibration file path (shared with cube_perception)."""
    package_config = Path("/root/ros2_ws/src/rsy_cube_perception/config/cube_calibration.json")
    if package_config.parent.exists() or _try_create_dir(package_config.parent):
        return package_config
    return Path.home() / ".ros" / "cube_calibration.json"


class CalibrateCubeServer(Node):
    def __init__(self):
        super().__init__("calibrate_cube_server")

        # Declare parameters
        self.declare_parameter("camera_index", 4)
        self.declare_parameter("use_mock_hardware", False)

        self.camera_index = self.get_parameter("camera_index").get_parameter_value().integer_value
        self.use_mock_hardware = self.get_parameter("use_mock_hardware").get_parameter_value().bool_value

        if self.use_mock_hardware:
            self.get_logger().info("Running in MOCK HARDWARE mode")
        else:
            self.get_logger().info(f"Using camera index: {self.camera_index}")

        # Action clients for cube motion
        self.take_up_client = ActionClient(self, TakeUpCube, "take_up_cube")
        self.put_down_client = ActionClient(self, PutDownCube, "put_down_cube")
        self.present_face_client = ActionClient(self, PresentCubeFace, "present_cube_face")

        # Action server
        self._action_server = ActionServer(
            self,
            CalibrateCube,
            "calibrate_cube",
            execute_callback=self.execute_calibrate_cube,
        )

        # Camera
        self._camera = None
        self._camera_lock = threading.Lock()

        self.get_logger().info("CalibrateCube Action Server started")

    def _open_camera(self) -> bool:
        """Open the camera if not already open."""
        with self._camera_lock:
            if self._camera is not None and self._camera.isOpened():
                return True
            self._camera = cv2.VideoCapture(self.camera_index)
            if not self._camera.isOpened():
                self._camera = None
                return False
            return True

    def _close_camera(self):
        """Close the camera."""
        with self._camera_lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None

    def _read_frame(self):
        """Read a frame from the camera."""
        with self._camera_lock:
            if self._camera is None or not self._camera.isOpened():
                return False, None
            return self._camera.read()

    def _call_action_sync(self, client, goal, timeout=60.0) -> bool:
        """Call an action synchronously and return success status."""
        if not client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"Action server not available")
            return False

        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

        if not future.done():
            self.get_logger().error("Failed to send goal")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)

        if not result_future.done():
            self.get_logger().error("Failed to get result")
            return False

        result = result_future.result()
        return result.result.success

    def _take_up_cube(self) -> bool:
        """Call TakeUpCube action."""
        if self.use_mock_hardware:
            self.get_logger().info("[MOCK] TakeUpCube")
            time.sleep(0.5)
            return True
        goal = TakeUpCube.Goal()
        return self._call_action_sync(self.take_up_client, goal)

    def _put_down_cube(self) -> bool:
        """Call PutDownCube action."""
        if self.use_mock_hardware:
            self.get_logger().info("[MOCK] PutDownCube")
            time.sleep(0.5)
            return True
        goal = PutDownCube.Goal()
        return self._call_action_sync(self.put_down_client, goal)

    def _present_face(self, face: str) -> bool:
        """Call PresentCubeFace action."""
        if self.use_mock_hardware:
            self.get_logger().info(f"[MOCK] PresentCubeFace: {face}")
            time.sleep(0.5)
            return True
        goal = PresentCubeFace.Goal()
        goal.face = face
        return self._call_action_sync(self.present_face_client, goal)

    def _show_roi_selection_gui(self) -> Optional[Dict]:
        """Show GUI for ROI selection on first face."""
        window_title = "CalibrateCube - Draw ROI around cube face"

        if not self._open_camera():
            self.get_logger().error("Failed to open camera for ROI selection")
            return None

        selection = {
            "start": None,
            "end": None,
            "drawing": False,
            "complete": False,
            "confirmed": False,
        }

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selection["start"] = (x, y)
                selection["end"] = None
                selection["drawing"] = True
                selection["complete"] = False
            elif event == cv2.EVENT_MOUSEMOVE and selection["drawing"]:
                selection["end"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                selection["end"] = (x, y)
                selection["drawing"] = False
                selection["complete"] = True

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, mouse_callback)

        instructions_draw = "Draw rectangle around cube face. ESC=Cancel"
        instructions_confirm = "ENTER=Confirm, R=Redraw, ESC=Cancel"

        while not selection["confirmed"]:
            ret, frame = self._read_frame()
            if not ret:
                time.sleep(0.05)
                continue

            display = frame.copy()

            # Draw selection rectangle
            if selection["start"] and selection["end"]:
                cv2.rectangle(display, selection["start"], selection["end"], (0, 255, 0), 2)

                if selection["complete"]:
                    # Draw 3x3 grid preview
                    x1, y1 = selection["start"]
                    x2, y2 = selection["end"]
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    size = max(x_max - x_min, y_max - y_min)
                    cell = size // 3

                    for i in range(1, 3):
                        cv2.line(display, (x_min + i * cell, y_min),
                                 (x_min + i * cell, y_min + size), (0, 255, 0), 1)
                        cv2.line(display, (x_min, y_min + i * cell),
                                 (x_min + size, y_min + i * cell), (0, 255, 0), 1)

                    # Highlight center facelet
                    cx = x_min + size // 2
                    cy = y_min + size // 2
                    cv2.circle(display, (cx, cy), 8, (0, 255, 255), 2)

            # Instructions
            if selection["complete"]:
                text = instructions_confirm
            else:
                text = instructions_draw

            # Draw text with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(display, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1)
            cv2.putText(display, text, (10, 25), font, font_scale, (255, 255, 255), thickness)

            cv2.imshow(window_title, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                cv2.destroyWindow(window_title)
                return None
            elif selection["complete"] and key in [13, ord("c"), ord("C")]:  # Enter/C
                selection["confirmed"] = True
            elif selection["complete"] and key in [ord("r"), ord("R")]:  # R
                selection["start"] = None
                selection["end"] = None
                selection["complete"] = False

        cv2.destroyWindow(window_title)

        x1, y1 = selection["start"]
        x2, y2 = selection["end"]
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        size = max(abs(x2 - x1), abs(y2 - y1))

        return {"x": x_min, "y": y_min, "size": size}

    def _show_color_sampling_gui(self, roi_data: Dict, face: str, expected_color: str) -> Optional[np.ndarray]:
        """Show GUI for user to click center facelet and sample color."""
        window_title = f"CalibrateCube - Face {face} ({expected_color}) - Click center"

        if not self._open_camera():
            self.get_logger().error("Failed to open camera for color sampling")
            return None

        x0, y0, size = roi_data["x"], roi_data["y"], roi_data["size"]
        cell = size // 3

        # Center facelet bounds
        center_x = x0 + cell + cell // 2
        center_y = y0 + cell + cell // 2
        sample_radius = max(cell // 4, 5)

        sampled_hsv = [None]  # Use list to allow modification in callback
        confirmed = [False]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Sample HSV at click location
                ret, frame = self._read_frame()
                if ret:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    h, w = frame.shape[:2]
                    # Sample region around click
                    r = sample_radius
                    x1 = max(0, x - r)
                    x2 = min(w, x + r)
                    y1 = max(0, y - r)
                    y2 = min(h, y + r)
                    sample = hsv[y1:y2, x1:x2]
                    if sample.size > 0:
                        sampled_hsv[0] = np.median(sample.reshape(-1, 3), axis=0)

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, mouse_callback)

        while not confirmed[0]:
            ret, frame = self._read_frame()
            if not ret:
                time.sleep(0.05)
                continue

            display = frame.copy()
            h, w = frame.shape[:2]

            # Draw ROI and grid
            cv2.rectangle(display, (x0, y0), (x0 + size, y0 + size), (0, 255, 0), 2)
            for i in range(1, 3):
                cv2.line(display, (x0 + i * cell, y0), (x0 + i * cell, y0 + size), (0, 255, 0), 1)
                cv2.line(display, (x0, y0 + i * cell), (x0 + size, y0 + i * cell), (0, 255, 0), 1)

            # Highlight center facelet
            cv2.circle(display, (center_x, center_y), sample_radius, (0, 255, 255), 2)

            # Show sampled color swatch if available
            if sampled_hsv[0] is not None:
                hsv_val = sampled_hsv[0]
                h_val, s_val, v_val = int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])
                # Create color swatch
                swatch = np.zeros((50, 100, 3), dtype=np.uint8)
                swatch[:] = [h_val, s_val, v_val]
                swatch_bgr = cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR)
                # Place swatch in top-right corner
                swatch_x = w - 110
                swatch_y = 10
                if swatch_x > 0 and swatch_y + 50 < h:
                    display[swatch_y:swatch_y + 50, swatch_x:swatch_x + 100] = swatch_bgr
                    # Draw border around swatch
                    cv2.rectangle(display, (swatch_x, swatch_y), (swatch_x + 100, swatch_y + 50), (255, 255, 255), 1)

                text = f"H:{h_val} S:{s_val} V:{v_val} - ENTER=Confirm, Click=Resample"
            else:
                text = f"Click center facelet ({expected_color}). ESC=Cancel"

            # Draw instructions
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(display, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1)
            cv2.putText(display, text, (10, 22), font, font_scale, (255, 255, 255), thickness)

            # Draw face label
            face_text = f"Face: {face} ({expected_color})"
            cv2.rectangle(display, (5, 30), (200, 55), (0, 0, 0), -1)
            cv2.putText(display, face_text, (10, 48), font, 0.6, (0, 255, 255), 1)

            cv2.imshow(window_title, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                cv2.destroyWindow(window_title)
                return None
            elif key == 13 and sampled_hsv[0] is not None:  # Enter
                confirmed[0] = True

        cv2.destroyWindow(window_title)
        return sampled_hsv[0]

    def _save_calibration(self, data: Dict):
        """Save calibration data to JSON file."""
        cal_file = get_calibration_file_path()
        cal_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cal_file, "w") as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f"Calibration saved to {cal_file}")

    def execute_calibrate_cube(self, goal_handle):
        """Main calibration procedure."""
        feedback = CalibrateCube.Feedback()
        result = CalibrateCube.Result()

        self.get_logger().info("CalibrateCube action started")

        try:
            # Step 1: Take up cube
            feedback.status = "Taking up cube..."
            feedback.faces_completed = 0
            feedback.total_faces = 6
            goal_handle.publish_feedback(feedback)

            if not self._take_up_cube():
                result.success = False
                result.message = "Failed to take up cube"
                goal_handle.abort()
                return result

            # Step 2: Present first face
            first_face = FACE_ORDER[0]
            feedback.status = f"Presenting face {first_face} for ROI selection..."
            feedback.current_face = first_face
            goal_handle.publish_feedback(feedback)

            if not self._present_face(first_face):
                result.success = False
                result.message = f"Failed to present face {first_face}"
                goal_handle.abort()
                return result

            # Step 3: ROI selection on first face
            feedback.status = "Draw ROI rectangle around cube face"
            feedback.window_title = "CalibrateCube - Draw ROI"
            goal_handle.publish_feedback(feedback)

            roi_data = self._show_roi_selection_gui()
            if not roi_data:
                result.success = False
                result.message = "User cancelled ROI selection"
                goal_handle.abort()
                return result

            self.get_logger().info(f"ROI selected: {roi_data}")

            # Step 4: Collect color samples from all 6 faces
            color_samples = {}

            for i, face in enumerate(FACE_ORDER):
                expected_color = FACE_CENTER_COLORS[face]

                feedback.status = f"Calibrating face {face} ({expected_color})"
                feedback.current_face = face
                feedback.faces_completed = i
                goal_handle.publish_feedback(feedback)

                # Present face (first face already presented)
                if i > 0:
                    if not self._present_face(face):
                        result.success = False
                        result.message = f"Failed to present face {face}"
                        goal_handle.abort()
                        return result

                # Show GUI for color sampling
                feedback.status = f"Click center facelet of face {face}"
                feedback.window_title = f"Face {face} - {expected_color}"
                goal_handle.publish_feedback(feedback)

                hsv_sample = self._show_color_sampling_gui(roi_data, face, expected_color)

                if hsv_sample is None:
                    result.success = False
                    result.message = f"User cancelled on face {face}"
                    goal_handle.abort()
                    return result

                color_samples[expected_color] = hsv_sample
                self.get_logger().info(f"Face {face} ({expected_color}): HSV = {hsv_sample}")

            # Step 5: Save calibration
            feedback.status = "Saving calibration..."
            feedback.faces_completed = 6
            goal_handle.publish_feedback(feedback)
            # Convert color samples to JSON-serializable format for centroid-based classification
            color_centroids = {}
            for color_name, hsv_sample in color_samples.items():
                color_centroids[color_name] = [float(hsv_sample[0]), float(hsv_sample[1]), float(hsv_sample[2])]

            calibration_data = {
                "roi": roi_data,
                "color_centroids": color_centroids,
                "camera_index": self.camera_index,
            }
            self._save_calibration(calibration_data)

            # Step 7: Put down cube
            feedback.status = "Putting down cube..."
            goal_handle.publish_feedback(feedback)

            if not self._put_down_cube():
                self.get_logger().warn("Failed to put down cube, but calibration was saved")

            # Success
            result.success = True
            result.message = "Calibration completed successfully"
            result.x = roi_data["x"]
            result.y = roi_data["y"]
            result.size = roi_data["size"]
            result.colors_calibrated = list(color_samples.keys())

            goal_handle.succeed()
            self.get_logger().info("CalibrateCube action completed successfully")
            return result

        except Exception as e:
            self.get_logger().error(f"CalibrateCube failed: {e}")
            result.success = False
            result.message = f"Calibration failed: {str(e)}"
            goal_handle.abort()
            return result

        finally:
            self._close_camera()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)

    try:
        server = CalibrateCubeServer()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(server)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
