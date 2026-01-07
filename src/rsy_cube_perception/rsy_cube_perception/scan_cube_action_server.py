#!/usr/bin/env python3
"""
ROS2 Action/Service Server for cube perception and solving.
1. ScanCubeFace action: scan a single face, store colors internally
2. SolveCube service: run solver on stored colors
3. CalibrateCamera action: GUI-based camera calibration with persistent storage

Supports mock hardware mode for testing without a camera.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

import cv2
import numpy as np
import time
import json
from typing import List, Dict
from pathlib import Path

try:
    from rsy_cube_perception.action import ScanCubeFace, CalibrateCamera
    from rsy_cube_perception.srv import SolveCube
except Exception as e:
    raise RuntimeError(
        f"Failed to import action/service types: {e}. "
        "Make sure ScanCubeFace.action, CalibrateCamera.action, and SolveCube.srv exist and package is built."
    )

# Calibration data file location - prefer package config directory
def get_calibration_file_path() -> Path:
    """Get calibration file path, preferring package config directory."""
    # Try package config directory first
    package_config = Path("/root/ros2_ws/src/rsy_cube_perception/config/cube_perception_calibration.json")
    if package_config.parent.exists() or _try_create_dir(package_config.parent):
        return package_config
    # Fallback to home .ros directory
    return Path.home() / ".ros" / "cube_perception_calibration.json"


def _try_create_dir(path: Path) -> bool:
    """Try to create directory, return True if successful."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


FACE_ORDER = ["U", "R", "F", "D", "L", "B"]

# Rotation angles for each face to match solver notation
FACE_ROTATION_ANGLES = {
    "U": 0,
    "D": 180,
    "F": 0,
    "B": 270,
    "L": 270,
    "R": 90,
}

# Mock colors for each face (used in mock hardware mode)
MOCK_FACE_COLORS = {
    "U": ["white"] * 9,
    "D": ["yellow"] * 9,
    "F": ["green"] * 9,
    "B": ["blue"] * 9,
    "L": ["orange"] * 9,
    "R": ["red"] * 9,
}

# Color ranges (HSV)
COLOR_RANGES = {
    "white": ((0, 0, 180), (180, 60, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
    "red": ((0, 120, 70), (10, 255, 255)),
    "red2": ((170, 120, 70), (180, 255, 255)),
    "orange": ((10, 120, 100), (20, 255, 255)),
    "green": ((40, 70, 70), (85, 255, 255)),
    "blue": ((90, 70, 70), (130, 255, 255)),
}

# Color to BGR mapping for visualization
COLOR_BGR = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "unknown": (128, 128, 128),
}

MOVE_DESCRIPTIONS = {
    "U": "Oben im Uhrzeigersinn",
    "U'": "Oben gegen den Uhrzeigersinn",
    "U2": "Oben doppelt",
    "R": "Rechts im Uhrzeigersinn",
    "R'": "Rechts gegen den Uhrzeigersinn",
    "R2": "Rechts doppelt",
    "F": "Vorne im Uhrzeigersinn",
    "F'": "Vorne gegen den Uhrzeigersinn",
    "F2": "Vorne doppelt",
    "D": "Unten im Uhrzeigersinn",
    "D'": "Unten gegen den Uhrzeigersinn",
    "D2": "Unten doppelt",
    "L": "Links im Uhrzeigersinn",
    "L'": "Links gegen den Uhrzeigersinn",
    "L2": "Links doppelt",
    "B": "Hinten im Uhrzeigersinn",
    "B'": "Hinten gegen den Uhrzeigersinn",
    "B2": "Hinten doppelt",
}


def describe_solution(solution: str) -> str:
    """Convert solution string to human-readable description."""
    if not solution:
        return "Keine Züge erforderlich."
    parts = []
    for move in solution.split():
        parts.append(MOVE_DESCRIPTIONS.get(move, move))
    return " · ".join(parts)


def classify_color(hsv_value: np.ndarray) -> str:
    """Classify color based on HSV value."""
    h, s, v = int(hsv_value[0]), int(hsv_value[1]), int(hsv_value[2])
    for name, (lower, upper) in COLOR_RANGES.items():
        low = np.array(lower, dtype=np.uint8)
        up = np.array(upper, dtype=np.uint8)
        px = np.uint8([[[h, s, v]]])
        mask = cv2.inRange(px, low, up)
        if mask[0, 0] != 0:
            return "red" if name == "red2" else name
    return "unknown"


def rotate_facelets_counterclockwise(colors: List[str], angle_degrees: int) -> List[str]:
    """Rotate a 3x3 grid of colors counterclockwise by the specified angle."""
    if angle_degrees % 90 != 0:
        raise ValueError(f"Angle must be multiple of 90, got {angle_degrees}")

    angle = angle_degrees % 360
    if angle == 0:
        return colors

    matrix = [colors[i:i+3] for i in range(0, 9, 3)]
    num_rotations = angle // 90

    for _ in range(num_rotations):
        new_matrix = [
            [matrix[2][0], matrix[1][0], matrix[0][0]],
            [matrix[2][1], matrix[1][1], matrix[0][1]],
            [matrix[2][2], matrix[1][2], matrix[0][2]]
        ]
        matrix = new_matrix

    return [matrix[i][j] for i in range(3) for j in range(3)]


def ensure_calibration_dir():
    """Ensure calibration directory exists."""
    cal_file = get_calibration_file_path()
    cal_file.parent.mkdir(parents=True, exist_ok=True)


def load_calibration() -> Dict:
    """Load calibration data from file."""
    ensure_calibration_dir()
    cal_file = get_calibration_file_path()
    if cal_file.exists():
        try:
            with open(cal_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_calibration(data: Dict):
    """Save calibration data to file."""
    ensure_calibration_dir()
    cal_file = get_calibration_file_path()
    with open(cal_file, 'w') as f:
        json.dump(data, f, indent=2)


class CubePerceptionServer(Node):
    def __init__(self):
        super().__init__("cube_perception_server")

        # Declare parameters
        self.declare_parameter("use_mock_hardware", False)
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("show_preview", True)
        self.declare_parameter("mock_cube_solution", "R U R' U'")
        self.declare_parameter("mock_cube_description", "Mock cube solution for testing")

        # Get parameters
        self.use_mock_hardware = self.get_parameter("use_mock_hardware").get_parameter_value().bool_value
        self.camera_index = self.get_parameter("camera_index").get_parameter_value().integer_value
        self.show_preview = self.get_parameter("show_preview").get_parameter_value().bool_value
        self.mock_cube_solution = self.get_parameter("mock_cube_solution").get_parameter_value().string_value
        self.mock_cube_description = self.get_parameter("mock_cube_description").get_parameter_value().string_value

        if self.use_mock_hardware:
            self.get_logger().info("Running in MOCK HARDWARE mode - no camera connection")
        else:
            self.get_logger().info(f"Using camera index: {self.camera_index}")

        # Internal storage for scanned colors
        self.scanned_colors = {}  # face -> [9 color names]
        self.color_to_face = {}   # center_color -> face

        # Create action servers
        self._scan_face_action_server = ActionServer(
            self,
            ScanCubeFace,
            'scan_cube_face',
            execute_callback=self.execute_scan_cube_face
        )

        self._calibrate_camera_action_server = ActionServer(
            self,
            CalibrateCamera,
            'calibrate_camera',
            execute_callback=self.execute_calibrate_camera
        )

        # Create service
        self._solve_cube_service = self.create_service(
            SolveCube,
            'solve_cube',
            self.execute_solve_cube
        )

        self.get_logger().info("Cube Perception Server started.")

    def execute_scan_cube_face(self, goal_handle):
        """Execute ScanCubeFace action."""
        face = goal_handle.request.cube_face

        self.get_logger().info(f"ScanCubeFace goal received: face={face}")

        feedback_msg = ScanCubeFace.Feedback()
        result = ScanCubeFace.Result()

        # Validate face
        if face not in FACE_ORDER:
            result.success = False
            result.message = f"Invalid face: {face}. Must be one of {FACE_ORDER}"
            goal_handle.abort()
            return result

        # Mock hardware mode - return fake colors without camera
        if self.use_mock_hardware:
            return self._execute_mock_scan_face(goal_handle, face, feedback_msg, result)

        # Real hardware - use camera
        return self._execute_real_scan_face(goal_handle, face, self.camera_index, feedback_msg, result)

    def _execute_mock_scan_face(self, goal_handle, face, feedback_msg, result):
        """Execute mock face scan without camera."""
        self.get_logger().info(f"[MOCK] Simulating scan of face {face}...")

        feedback_msg.status = f"[MOCK] Scanning face {face}..."
        feedback_msg.frames_sampled = 0
        goal_handle.publish_feedback(feedback_msg)

        # Simulate processing time
        time.sleep(0.5)

        # Get mock colors for this face
        colors = MOCK_FACE_COLORS.get(face, ["unknown"] * 9)

        # Apply rotation to match solver notation
        rotation_angle = FACE_ROTATION_ANGLES.get(face, 0)
        if rotation_angle != 0:
            colors = rotate_facelets_counterclockwise(colors, rotation_angle)

        # Store colors internally
        self.scanned_colors[face] = colors
        center_color = colors[4]
        self.color_to_face[center_color] = face

        result.success = True
        result.message = f"[MOCK] Face {face} scanned successfully"
        result.colors = colors

        feedback_msg.status = f"[MOCK] Face {face} captured"
        feedback_msg.frames_sampled = 4
        goal_handle.publish_feedback(feedback_msg)
        goal_handle.succeed()

        self.get_logger().info(f"[MOCK] Face {face} scanned: {colors}")
        return result

    def _execute_real_scan_face(self, goal_handle, face, camera_index, feedback_msg, result):
        """Execute real face scan with camera."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            result.success = False
            result.message = f"Camera index {camera_index} not available"
            goal_handle.abort()
            return result

        show_preview = self.show_preview
        preview_window = f"Scanning face {face}"

        if show_preview:
            try:
                cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
            except Exception:
                self.get_logger().warn("Preview window could not be created")
                show_preview = False

        try:
            max_wait_s = 200.0
            start_t = time.time()
            colors = None

            while time.time() - start_t < max_wait_s:
                if goal_handle.is_cancel_requested:
                    result.success = False
                    result.message = "Goal cancelled by client"
                    goal_handle.canceled()
                    return result

                feedback_msg.status = f"Scanning face {face}..."
                feedback_msg.frames_sampled = int((time.time() - start_t) * 30)
                goal_handle.publish_feedback(feedback_msg)

                colors = self.capture_face_colors(cap, sample_frames=4, show_preview=show_preview, preview_window=preview_window)

                if colors and len(colors) == 9 and all(c != "unknown" for c in colors):
                    break

                time.sleep(0.5)

            if not colors or len(colors) != 9 or any(c == "unknown" for c in colors):
                result.success = False
                result.message = f"Failed to capture complete face {face} within {max_wait_s}s"
                goal_handle.abort()
                return result

            # Apply rotation to match solver notation
            rotation_angle = FACE_ROTATION_ANGLES.get(face, 0)
            if rotation_angle != 0:
                self.get_logger().info(f"Rotating face {face} by {rotation_angle}° counterclockwise")
                colors = rotate_facelets_counterclockwise(colors, rotation_angle)

            # Store colors internally
            self.scanned_colors[face] = colors
            center_color = colors[4]
            self.color_to_face[center_color] = face

            result.success = True
            result.message = f"Face {face} scanned successfully"
            result.colors = colors

            feedback_msg.status = f"Face {face} captured successfully"
            goal_handle.publish_feedback(feedback_msg)
            goal_handle.succeed()

            self.get_logger().info(f"Face {face} scanned: {colors}")
            return result

        finally:
            cap.release()
            try:
                if show_preview:
                    cv2.destroyAllWindows()
            except Exception:
                pass

    def execute_calibrate_camera(self, goal_handle):
        """Execute CalibrateCamera action with GUI."""
        self.get_logger().info("CalibrateCamera action received")

        feedback_msg = CalibrateCamera.Feedback()
        result = CalibrateCamera.Result()

        if self.use_mock_hardware:
            result.success = True
            result.message = "[MOCK] Calibration skipped in mock mode"
            result.x = 0
            result.y = 0
            result.size = 100
            goal_handle.succeed()
            return result

        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            result.success = False
            result.message = f"Camera index {self.camera_index} not available"
            goal_handle.abort()
            return result

        window_title = "Calibrate Camera - Select Cube Region"

        try:
            ret, frame = cap.read()
            if not ret:
                result.success = False
                result.message = "Failed to capture frame"
                goal_handle.abort()
                return result

            feedback_msg.status = "Waiting for user to select cube region..."
            feedback_msg.window_title = window_title
            goal_handle.publish_feedback(feedback_msg)

            calibration_data = self.show_calibration_gui(frame, window_title)

            if not calibration_data:
                result.success = False
                result.message = "User cancelled calibration"
                goal_handle.abort()
                return result

            cal = load_calibration()
            cal[str(self.camera_index)] = calibration_data
            save_calibration(cal)

            result.success = True
            result.message = "Calibration saved successfully"
            result.x = calibration_data['x']
            result.y = calibration_data['y']
            result.size = calibration_data['size']

            feedback_msg.status = "Calibration complete"
            goal_handle.publish_feedback(feedback_msg)
            goal_handle.succeed()

            self.get_logger().info(f"Calibration saved: {calibration_data}")
            return result

        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def show_calibration_gui(self, frame, window_title) -> Dict:
        """Show interactive GUI for user to select cube region."""
        selection = {
            'start': None,
            'end': None,
            'drawing': False,
            'complete': False,
            'confirmed': False
        }

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selection['start'] = (x, y)
                selection['end'] = None
                selection['drawing'] = True
                selection['complete'] = False
            elif event == cv2.EVENT_MOUSEMOVE and selection['drawing']:
                selection['end'] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                selection['end'] = (x, y)
                selection['drawing'] = False
                selection['complete'] = True

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, mouse_callback)

        instructions_select = "Draw rectangle around cube. ESC=Cancel"
        instructions_confirm = "Press ENTER/C=Confirm, R=Redraw, ESC=Cancel"

        while not selection['confirmed']:
            display = frame.copy()
            
            # Draw selection rectangle if exists
            if selection['start'] and selection['end']:
                cv2.rectangle(display, selection['start'], selection['end'], (0, 255, 0), 2)
            elif selection['start']:
                cv2.circle(display, selection['start'], 5, (0, 255, 0), -1)

            # Show appropriate instructions
            if selection['complete']:
                instructions = instructions_confirm
                # Draw 3x3 grid preview on selected region
                x1, y1 = selection['start']
                x2, y2 = selection['end']
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                size = max(x_max - x_min, y_max - y_min)
                cell = size // 3
                
                # Draw grid lines
                for i in range(1, 3):
                    x = x_min + i * cell
                    cv2.line(display, (x, y_min), (x, y_min + size), (0, 255, 0), 1)
                    y = y_min + i * cell
                    cv2.line(display, (x_min, y), (x_min + size, y), (0, 255, 0), 1)
                
                # Draw sample points
                for row in range(3):
                    for col in range(3):
                        cx = x_min + col * cell + cell // 2
                        cy = y_min + row * cell + cell // 2
                        cv2.circle(display, (cx, cy), 4, (0, 255, 255), -1)
            else:
                instructions = instructions_select

            # Draw instruction text with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            text_size = cv2.getTextSize(instructions, font, font_scale, thickness)[0]
            text_x, text_y = 10, 25
            cv2.rectangle(display, (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(display, instructions, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

            cv2.imshow(window_title, display)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC to cancel
                cv2.destroyWindow(window_title)
                return None
            elif selection['complete'] and key in [13, ord('c'), ord('C')]:  # Enter or C to confirm
                selection['confirmed'] = True
            elif selection['complete'] and key in [ord('r'), ord('R')]:  # R to redraw
                selection['start'] = None
                selection['end'] = None
                selection['complete'] = False

        cv2.destroyWindow(window_title)

        x1, y1 = selection['start']
        x2, y2 = selection['end']

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        size = max(x_max - x_min, y_max - y_min)

        return {
            'x': x_min,
            'y': y_min,
            'size': size
        }

    def execute_solve_cube(self, request, response):
        """Execute SolveCube service."""
        self.get_logger().info("SolveCube service called")

        # Mock hardware mode - return mock solution
        if self.use_mock_hardware:
            self.get_logger().info(f"[MOCK] Returning mock solution: {self.mock_cube_solution}")
            response.success = True
            response.message = "[MOCK] Returning configured mock solution"
            response.solution = self.mock_cube_solution
            response.description = describe_solution(self.mock_cube_solution)
            response.missing_faces = []
            return response

        # Real mode - check which faces are missing
        missing_faces = [f for f in FACE_ORDER if f not in self.scanned_colors]

        if missing_faces:
            response.success = False
            response.message = f"Missing scanned faces: {missing_faces}"
            response.solution = ""
            response.description = ""
            response.missing_faces = missing_faces
            return response

        # Build facelet string for kociemba
        try:
            facelet_order = []
            for face in FACE_ORDER:
                colors = self.scanned_colors[face]
                if len(colors) != 9:
                    response.success = False
                    response.message = f"Face {face} incomplete"
                    response.solution = ""
                    response.description = ""
                    response.missing_faces = [face]
                    return response

                for c in colors:
                    mapped = self.color_to_face.get(c, "?")
                    facelet_order.append(mapped)

            if "?" in facelet_order:
                response.success = False
                response.message = "Unable to map all colors to faces"
                response.solution = ""
                response.description = ""
                response.missing_faces = []
                return response

            if len(facelet_order) != 54:
                response.success = False
                response.message = "Incorrect number of facelets"
                response.solution = ""
                response.description = ""
                response.missing_faces = []
                return response

            facelet_string = "".join(facelet_order)
            self.get_logger().info(f"Facelet string: {facelet_string}")

            # Run solver
            try:
                import kociemba
                solution = kociemba.solve(facelet_string)
                description = describe_solution(solution)

                response.success = True
                response.message = "Cube solved successfully"
                response.solution = solution
                response.description = description
                response.missing_faces = []

                self.get_logger().info(f"Solution: {solution}")
                return response

            except ImportError:
                response.success = False
                response.message = "kociemba not installed (pip install kociemba)"
                response.solution = ""
                response.description = ""
                response.missing_faces = []
                return response

            except Exception as e:
                response.success = False
                response.message = f"Solver error: {str(e)}"
                response.solution = ""
                response.description = ""
                response.missing_faces = []
                return response

        except Exception as e:
            response.success = False
            response.message = f"Error building facelet string: {str(e)}"
            response.solution = ""
            response.description = ""
            response.missing_faces = []
            return response

    def capture_face_colors(self, cap: cv2.VideoCapture, sample_frames: int = 6,
                           show_preview: bool = False, preview_window: str = "Preview") -> List[str]:
        """Capture 9 facelet colors for current face."""
        calibration = load_calibration()
        cal_data = calibration.get(str(self.camera_index))

        collected = []
        attempts = 0
        max_attempts = sample_frames * 2

        while len(collected) < sample_frames and attempts < max_attempts:
            ret, frame = cap.read()
            attempts += 1
            if not ret:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            if cal_data:
                x0, y0, face_size = cal_data['x'], cal_data['y'], cal_data['size']
                x0 = max(0, min(x0, w))
                y0 = max(0, min(y0, h))
                face_size = min(face_size, w - x0, h - y0)
            else:
                face_size = min(h, w) // 2
                x0, y0 = (w - face_size) // 2, (h - face_size) // 2

            cell = face_size // 3
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            colors = []
            color_positions = []  # Store positions for preview
            ok = True

            for row in range(3):
                for col in range(3):
                    cx = x0 + col * cell + cell // 2
                    cy = y0 + row * cell + cell // 2

                    x1 = max(cx - 5, 0)
                    x2 = min(cx + 5, w - 1)
                    y1 = max(cy - 5, 0)
                    y2 = min(cy + 5, h - 1)

                    sample = hsv[y1:y2+1, x1:x2+1]
                    if sample.size == 0:
                        ok = False
                        break

                    mean = sample.reshape(-1, 3).mean(axis=0)
                    color_name = classify_color(mean.astype(np.uint8))
                    colors.append(color_name)
                    color_positions.append((cx, cy, color_name))

                if not ok:
                    break

            if ok and len(colors) == 9:
                collected.append(colors)
            else:
                time.sleep(0.05)

            if show_preview:
                try:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x0, y0), (x0 + face_size, y0 + face_size), (0, 255, 0), 2)
                    for i in range(1, 3):
                        x = x0 + i * cell
                        cv2.line(overlay, (x, y0), (x, y0 + face_size), (0, 255, 0), 1)
                        y = y0 + i * cell
                        cv2.line(overlay, (x0, y), (x0 + face_size, y), (0, 255, 0), 1)
                    
                    # Draw sample points with detected colors
                    for cx, cy, color_name in color_positions:
                        # Draw colored circle for the detected color
                        bgr_color = COLOR_BGR.get(color_name, (128, 128, 128))
                        cv2.circle(overlay, (cx, cy), 12, bgr_color, -1)
                        cv2.circle(overlay, (cx, cy), 12, (0, 0, 0), 2)  # Black border
                        
                        # Draw color name label
                        label = color_name[:3].upper()  # Shortened: WHI, YEL, RED, etc.
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4
                        thickness = 1
                        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                        text_x = cx - text_size[0] // 2
                        text_y = cy + cell // 2 - 5
                        
                        # Background for text
                        cv2.rectangle(overlay, 
                                     (text_x - 2, text_y - text_size[1] - 2),
                                     (text_x + text_size[0] + 2, text_y + 2),
                                     (0, 0, 0), -1)
                        cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                    
                    # Show progress info
                    progress_text = f"Frames: {len(collected)}/{sample_frames}"
                    cv2.rectangle(overlay, (5, 5), (150, 30), (0, 0, 0), -1)
                    cv2.putText(overlay, progress_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow(preview_window, overlay)
                    cv2.waitKey(1)
                except Exception:
                    pass

        if not collected:
            return None

        # Vote for best color across frames
        final = []
        for idx in range(9):
            votes = {}
            for frame_colors in collected:
                c = frame_colors[idx]
                votes[c] = votes.get(c, 0) + 1
            best = max(votes.items(), key=lambda kv: kv[1])[0]
            final.append(best)

        return final


def main(args=None):
    rclpy.init(args=args)

    try:
        server = CubePerceptionServer()
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
