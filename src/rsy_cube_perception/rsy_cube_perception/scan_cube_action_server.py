#!/usr/bin/env python3
"""
ROS2 Action/Service Server for cube perception and solving.
1. ScanCubeFace action: scan a single face, store colors internally
2. SolveCube service: run solver on stored colors
3. CalibrateCamera action: GUI-based camera calibration with persistent storage
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.service import Service
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile

import cv2
import numpy as np
import time
import json
import os
from typing import List, Tuple, Dict
from pathlib import Path

try:
    from rsy_cube_perception.action import ScanCubeFace, CalibrateCamera
    from rsy_cube_perception.srv import SolveCube
except Exception as e:
    raise RuntimeError(
        f"Failed to import action/service types: {e}. "
        "Make sure ScanCubeFace.action, CalibrateCamera.action, and SolveCube.srv exist and package is built."
    )

# Calibration data file location
CALIBRATION_FILE = Path.home() / ".ros" / "cube_perception_calibration.json"

# (Optional) Typ für Drive action des Roboters, als Platzhalter:
# from robot_interfaces.action import DriveFace
# Wir verwenden hier keinen konkreten Typ, machen nur ein optionales Client-Call-Pattern.

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]

# Rotation angles for each face to match solver notation
# The scanned 3x3 matrix needs to be rotated counterclockwise by this angle
# to match kociemba solver's expected orientation
FACE_ROTATION_ANGLES = {
    "U": 0,      # No rotation needed for top face
    "D": 180,    # Bottom face needs 180° rotation
    "F": 0,      # No rotation needed for front face
    "B": 270,    # Back face needs 270° counterclockwise (-90°)
    "L": 270,    # Left face needs 270° counterclockwise (-90°)
    "R": 90,     # Right face needs 90° counterclockwise
}

# Farbgrenzen (HSV)
COLOR_RANGES = {
    "white": ((0, 0, 180), (180, 60, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
    "red": ((0, 120, 70), (10, 255, 255)),
    "red2": ((170, 120, 70), (180, 255, 255)),
    "orange": ((10, 120, 100), (20, 255, 255)),
    "green": ((40, 70, 70), (85, 255, 255)),
    "blue": ((90, 70, 70), (130, 255, 255)),
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
    """
    Klassifiziert anhand der COLOR_RANGES. hsv_value ist (H,S,V).
    Gibt einen Namen zurück oder 'unknown'.
    """
    h, s, v = int(hsv_value[0]), int(hsv_value[1]), int(hsv_value[2])
    for name, (lower, upper) in COLOR_RANGES.items():
        low = np.array(lower, dtype=np.uint8)
        up = np.array(upper, dtype=np.uint8)
        # cv2.inRange erwartet ein Bild, hier bauen wir eine 1x1 px
        px = np.uint8([[[h, s, v]]])
        mask = cv2.inRange(px, low, up)
        if mask[0, 0] != 0:
            return "red" if name == "red2" else name
    return "unknown"


def rotate_facelets_counterclockwise(colors: List[str], angle_degrees: int) -> List[str]:
    """
    Rotate a 3x3 grid of colors counterclockwise by the specified angle.
    
    Input: list of 9 colors in row-major order (indices 0-8):
        0 1 2
        3 4 5
        6 7 8
    
    Args:
        colors: list of 9 color strings
        angle_degrees: rotation angle (0, 90, 180, 270, -90)
    
    Returns:
        rotated list of 9 colors in row-major order
    """
    if angle_degrees % 90 != 0:
        raise ValueError(f"Angle must be multiple of 90, got {angle_degrees}")
    
    # Normalize angle to 0-270 range
    angle = angle_degrees % 360
    
    if angle == 0:
        return colors
    
    # Convert to 3x3 matrix
    matrix = [colors[i:i+3] for i in range(0, 9, 3)]
    
    # Apply rotation counterclockwise (angle / 90) times
    num_rotations = angle // 90
    
    for _ in range(num_rotations):
        # One 90° counterclockwise rotation:
        # [0 1 2]      [2 5 8]
        # [3 4 5]  =>  [1 4 7]
        # [6 7 8]      [0 3 6]
        new_matrix = [
            [matrix[2][0], matrix[1][0], matrix[0][0]],
            [matrix[2][1], matrix[1][1], matrix[0][1]],
            [matrix[2][2], matrix[1][2], matrix[0][2]]
        ]
        matrix = new_matrix
    
    # Convert back to flat list
    return [matrix[i][j] for i in range(3) for j in range(3)]


def ensure_calibration_dir():
    """Ensure calibration directory exists."""
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_calibration() -> Dict:
    """Load calibration data from file."""
    ensure_calibration_dir()
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_calibration(data: Dict):
    """Save calibration data to file."""
    ensure_calibration_dir()
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)


class CubePerceptionServer(Node):
    def __init__(self):
        super().__init__("cube_perception_server")
        
        # Declare parameters
        self.declare_parameter("show_preview", True)
        
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
        
        self.get_logger().info("Cube Perception Server started with ScanCubeFace, CalibrateCamera, and SolveCube service.")
    
    def execute_scan_cube_face(self, goal_handle):
        """Execute ScanCubeFace action."""
        self.get_logger().info(f"ScanCubeFace goal received: face={goal_handle.request.cube_face}")
        
        feedback_msg = ScanCubeFace.Feedback()
        result = ScanCubeFace.Result()
        
        face = goal_handle.request.cube_face
        camera_index = goal_handle.request.camera_index
        
        # Validate face
        if face not in FACE_ORDER:
            result.success = False
            result.message = f"Invalid face: {face}. Must be one of {FACE_ORDER}"
            goal_handle.abort()
            return result
        
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            result.success = True
            result.message = f"Camera index {camera_index} not available"
            # TODO: Remove dummy
            result.colors = "Dummy"
            goal_handle.abort()
            return result

            result.success = False
            result.message = f"Camera index {camera_index} not available"
            goal_handle.abort()
            return result
        
        show_preview = bool(self.get_parameter("show_preview").value)
        preview_window = f"Scanning face {face}"
        
        if show_preview:
            try:
                cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
            except Exception:
                self.get_logger().warn("Preview window could not be created")
                show_preview = False
        
        try:
            # Capture colors for this face with polling
            max_wait_s = 20.0
            start_t = time.time()
            colors = None
            
            while time.time() - start_t < max_wait_s:
                if goal_handle.is_cancel_requested:
                    result.success = False
                    result.message = "Goal cancelled by client"
                    goal_handle.canceled()
                    return result
                
                feedback_msg.status = f"Scanning face {face}..."
                feedback_msg.frames_sampled = int((time.time() - start_t) * 30)  # approx FPS
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
        
        camera_index = goal_handle.request.camera_index
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            result.success = False
            result.message = f"Camera index {camera_index} not available"
            goal_handle.abort()
            return result
        
        window_title = "Calibrate Camera - Select Cube Region"
        calibration_data = None
        
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
            
            # Create interactive window for user to select cube region
            calibration_data = self.show_calibration_gui(frame, window_title)
            
            if not calibration_data:
                result.success = False
                result.message = "User cancelled calibration"
                goal_handle.abort()
                return result
            
            # Save calibration
            cal = load_calibration()
            cal[str(camera_index)] = calibration_data
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
        """
        Show interactive GUI for user to select cube region.
        User clicks and drags to define the cube area.
        """
        h, w = frame.shape[:2]
        selection = {
            'start': None,
            'end': None,
            'complete': False
        }
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selection['start'] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                selection['end'] = (x, y)
                selection['complete'] = True
        
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, mouse_callback)
        
        while not selection['complete']:
            display = frame.copy()
            if selection['start']:
                cv2.circle(display, selection['start'], 5, (0, 255, 0), -1)
                if selection['end']:
                    cv2.rectangle(display, selection['start'], selection['end'], (0, 255, 0), 2)
            
            cv2.imshow(window_title, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to cancel
                return None
        
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
        
        # Check which faces are missing
        missing_faces = [f for f in FACE_ORDER if f not in self.scanned_colors]
        
        if missing_faces:
            response.message = f"Missing scanned faces: {missing_faces}"
            response.success = True
            response.solution = "U F D R B L"  # Dummy solution
            # TODO: Remove dummy solution
            # response.success = False
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
                
                # Map colors to face letters via center color
                for c in colors:
                    mapped = self.color_to_face.get(c, "?")
                    facelet_order.append(mapped)
            
            if "?" in facelet_order:
                response.success = False
                response.message = "Unable to map all colors to faces"
                response.solution = ""
                response.description = ""
                response.missing_faces = missing_faces
                return response
            
            if len(facelet_order) != 54:
                response.success = False
                response.message = "Incorrect number of facelets"
                response.solution = ""
                response.description = ""
                response.missing_faces = missing_faces
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
                response.missing_faces = missing_faces
                return response
            
            except Exception as e:
                response.success = False
                response.message = f"Solver error: {str(e)}"
                response.solution = ""
                response.description = ""
                response.missing_faces = missing_faces
                return response
        
        except Exception as e:
            response.success = False
            response.message = f"Error building facelet string: {str(e)}"
            response.solution = ""
            response.description = ""
            response.missing_faces = missing_faces
            return response
    
    def capture_face_colors(self, cap: cv2.VideoCapture, sample_frames: int = 6, 
                           show_preview: bool = False, preview_window: str = "Preview") -> List[str]:
        """
        Capture 9 facelet colors for current face.
        Uses calibration data if available, otherwise uses center of frame.
        """
        calibration = load_calibration()
        camera_index = 0  # TODO: get from cap if possible
        cal_data = calibration.get(str(camera_index))
        
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
            
            # Use calibration data if available
            if cal_data:
                x0, y0, face_size = cal_data['x'], cal_data['y'], cal_data['size']
                # Ensure bounds
                x0 = max(0, min(x0, w))
                y0 = max(0, min(y0, h))
                face_size = min(face_size, w - x0, h - y0)
            else:
                # Default: center of frame
                face_size = min(h, w) // 2
                x0, y0 = (w - face_size) // 2, (h - face_size) // 2
            
            cell = face_size // 3
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            colors = []
            ok = True
            
            for row in range(3):
                for col in range(3):
                    cx = x0 + col * cell + cell // 2
                    cy = y0 + row * cell + cell // 2
                    
                    # Clamp coordinates
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
                
                if not ok:
                    break
            
            if ok and len(colors) == 9:
                collected.append(colors)
            else:
                time.sleep(0.05)
            
            # Draw preview
            if show_preview:
                try:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x0, y0), (x0 + face_size, y0 + face_size), (0, 255, 0), 2)
                    for i in range(1, 3):
                        x = x0 + i * cell
                        cv2.line(overlay, (x, y0), (x, y0 + face_size), (0, 255, 0), 1)
                        y = y0 + i * cell
                        cv2.line(overlay, (x0, y), (x0 + face_size, y), (0, 255, 0), 1)
                    for row in range(3):
                        for col in range(3):
                            cx = x0 + col * cell + cell // 2
                            cy = y0 + row * cell + cell // 2
                            cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)
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

    def _execute_mock_scan(self, goal_handle, feedback_msg, result):
        """Execute mock cube scan without camera connection."""
        self.get_logger().info("[MOCK] Simulating cube scan...")

        # Get mock solution from parameters
        mock_solution = str(self.get_parameter("mock_cube_solution").value)
        mock_description = str(self.get_parameter("mock_cube_description").value)

        # Simulate scanning each face with feedback
        total_faces = len(FACE_ORDER)
        for i, face in enumerate(CAPTURE_ORDER):
            feedback_msg.current_face = face
            feedback_msg.faces_captured = i
            feedback_msg.total_faces = total_faces
            feedback_msg.status = f"[MOCK] Simulating scan of face {face}..."
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"[MOCK] Simulating face {face} ({i+1}/{total_faces})")

            # Check for cancellation
            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = "Goal cancelled by client."
                goal_handle.canceled()
                return result

            # Simulate processing time
            time.sleep(0.5)

        # Return mock result
        result.solution = mock_solution
        result.description = describe_solution(mock_solution) if not mock_description else mock_description
        result.success = True
        result.message = "[MOCK] Cube scan simulation completed successfully."

        feedback_msg.faces_captured = total_faces
        feedback_msg.status = "[MOCK] Fertig."
        goal_handle.publish_feedback(feedback_msg)

        self.get_logger().info(f"[MOCK] Returning solution: {mock_solution}")
        goal_handle.succeed()
        return result


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
