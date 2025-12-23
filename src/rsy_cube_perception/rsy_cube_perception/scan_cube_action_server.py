#!/usr/bin/env python3
"""
ROS2 Action Server zur Erfassung aller 6 Rubik-Seiten und Berechnung der Lösung.
Action: rsy_cube_perception.action.ScanCube  (aus action/ScanCube.action)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile

import cv2
import numpy as np
import time
from typing import List, Tuple

try:
    from rsy_cube_perception.action import ScanCube
except Exception as e:
    raise RuntimeError(
        "Konnte Action-Type rsy_cube_perception.action.ScanCube nicht importieren. "
        "Stelle sicher, dass du die Action 'ScanCube.action' im Paket hattest und das Paket gebaut wurde."
    )

# (Optional) Typ für Drive action des Roboters, als Platzhalter:
# from robot_interfaces.action import DriveFace
# Wir verwenden hier keinen konkreten Typ, machen nur ein optionales Client-Call-Pattern.

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
CAPTURE_ORDER = ["U", "F", "D", "R", "B", "L"]
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


class ScanCubeActionServer(Node):
    def __init__(self):
        super().__init__("scan_cube_action_server")
        # declare parameter to enable preview window with grid
        self.declare_parameter("show_preview", True)
        self.declare_parameter("use_mock_hardware", False)
        self.declare_parameter("mock_cube_solution", "R U R' U'")
        self.declare_parameter("mock_cube_description", "Mock cube solution for testing")
        self.preview_window = "Cube Preview"

        self._action_server = ActionServer(
            self,
            ScanCube,
            'scan_cube',
            execute_callback=self.execute_callback
        )

        # Log mock hardware status
        use_mock = self.get_parameter("use_mock_hardware").value
        if use_mock:
            self.get_logger().info("ScanCube Action Server gestartet (MOCK HARDWARE MODE).")
        else:
            self.get_logger().info("ScanCube Action Server gestartet.")
        # Optionaler ActionClient-Name/Typ für Fahren (Platzhalter)
        # self.drive_client = ActionClient(self, DriveFace, 'drive_to_face')
        # Wir verwenden keine Typen, nur optionales Warten auf Server.
        self.camera = None

    def execute_callback(self, goal_handle):
        self.get_logger().info("Goal erhalten: Starte Erfassung aller Seiten...")
        feedback_msg = ScanCube.Feedback()
        result = ScanCube.Result()

        # Check for mock hardware mode
        use_mock_hardware = bool(self.get_parameter("use_mock_hardware").value)
        if use_mock_hardware:
            return self._execute_mock_scan(goal_handle, feedback_msg, result)

        # read parameter for preview display (can change between goals)
        show_preview = bool(self.get_parameter("show_preview").value)

        # Parameter aus Goal lesen
        camera_index = 0
        try:
            camera_index = int(goal_handle.request.camera_index)
        except Exception:
            camera_index = 0

        # Versuche Kamera zu öffnen
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            msg = f"Kamera index={camera_index} nicht verfügbar."
            self.get_logger().error(msg)

            #return sample result for development
            result.solution = "F"
            result.success = True
            goal_handle.abort()
            return result

            result.success = False
            result.message = msg
            goal_handle.abort()
            return result

        # If preview requested, prepare window
        if show_preview:
            try:
                cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
            except Exception:
                # some environments may not support GUI; log and disable preview
                self.get_logger().warn("Preview window konnte nicht erstellt werden. Deaktiviere Preview.")
                show_preview = False

        captured_faces = {}  # face -> list[str] 9 Farben
        color_to_face = {}   # center_color -> face

        total_faces = len(FACE_ORDER)
        faces_captured = 0

        try:
            for face in CAPTURE_ORDER:
                # 1) Feedback: wir starten diese Seite
                feedback_msg.current_face = face
                feedback_msg.faces_captured = faces_captured
                feedback_msg.total_faces = total_faces
                feedback_msg.status = f"Anfordern: Roboter zeigt Seite {face}."
                goal_handle.publish_feedback(feedback_msg)
                self.get_logger().info(f"Fordere Roboter auf, Seite {face} zu zeigen (platzhalter).")

                # --- Platzhalter: Action an Roboter senden, damit richtige Seite gezeigt wird ---
                # Wenn ein echter ActionServer existiert, hier einen ActionClient benutzen, z.B.:
                #   drive_goal = DriveFace.Goal()
                #   drive_goal.face = face
                #   self.drive_client.wait_for_server(timeout_sec=2.0)
                #   send_goal_future = self.drive_client.send_goal_async(drive_goal)
                #   ...
                # Wir implementieren hier ein kurzes Initial-Warten, dann Polling der Kamera,
                # bis alle 9 Facelets erkannt sind (keine 'unknown' mehr) oder bis Timeout.
                time.sleep(5)  # kurze Wartezeit, damit der Roboter sich bewegen kann

                # Versuche wiederholt, die Seite zu erfassen, bis alle Facelets erkannt sind
                max_wait_s = 20.0
                start_t = time.time()
                captured_ok = False
                colors = None
                while time.time() - start_t < max_wait_s:
                    # Allow client to cancel
                    if goal_handle.is_cancel_requested:
                        msg = "Goal vom Client abgebrochen."
                        self.get_logger().info(msg)
                        result.success = False
                        result.message = msg
                        goal_handle.canceled()
                        return result

                    feedback_msg.status = f"Erfasse Seite {face} per Kamera..."
                    goal_handle.publish_feedback(feedback_msg)

                    # sample_frames kleiner während polling, da wir ggf. mehrfach versuchen
                    colors = self.capture_face_colors(cap, sample_frames=4, show_preview=show_preview)
                    if colors and len(colors) == 9 and all(c != "unknown" for c in colors):
                        captured_ok = True
                        break

                    # noch nicht vollständig erkannt -> kurz warten und erneut versuchen
                    self.get_logger().debug(f"Seite {face} noch nicht vollständig erkannt, erneuter Versuch...")
                    time.sleep(0.5)

                if not captured_ok:
                    msg = f"Fehler beim Erfassen der Seite {face} innerhalb {max_wait_s}s."
                    self.get_logger().warn(msg)
                    # wir fahren trotzdem fort — aber markieren result
                    result.success = False
                    result.message = msg
                    # optional: continue to next face to gather what we can
                else:
                    # Center-Farbe der Seite = colors[4]
                    center = colors[4]
                    # Wenn center noch unbekannt, ordne diese Seite zu
                    color_to_face[center] = face
                    captured_faces[face] = colors
                    faces_captured += 1
                    feedback_msg.faces_captured = faces_captured
                    feedback_msg.status = f"Seite {face} gespeichert ({faces_captured}/{total_faces})."
                    goal_handle.publish_feedback(feedback_msg)
                    self.get_logger().info(f"Seite {face} gespeichert: {colors}")

            # Ende Schleife über alle Seiten
            # Prüfe Vollständigkeit
            if len(captured_faces) != 6:
                msg = f"Nur {len(captured_faces)}/6 Seiten erfasst."
                self.get_logger().warn(msg)
                result.success = False
                result.message = msg
                # Wir geben dennoch das Ergebnis zurück mit Fehlerhinweis.
            else:
                # Baue facelet_string für kociemba
                facelet_order = []
                for face in FACE_ORDER:
                    colors = captured_faces[face]
                    if len(colors) != 9:
                        msg = f"Seite {face} unvollständig."
                        self.get_logger().error(msg)
                        result.success = False
                        result.message = msg
                        break
                    # map colors to face letters via center mapping color_to_face
                    for c in colors:
                        mapped = color_to_face.get(c, "?")
                        if mapped == "?":
                            # unknown color — unsichere Zuordnung
                            pass
                        facelet_order.append(mapped)
                # Wenn Fragezeichen in facelet_order -> Fehler
                if "?" in facelet_order or len(facelet_order) != 54:
                    msg = "Unvollständige oder unklare Farberkennung: facelet string ungültig."
                    self.get_logger().error(msg)
                    result.success = False
                    result.message = msg
                else:
                    facelet_string = "".join(facelet_order)
                    self.get_logger().info(f"Facelet string: {facelet_string}")
                    # Versuche kociemba zu verwenden
                    try:
                        import kociemba
                        solution = kociemba.solve(facelet_string)
                        description = describe_solution(solution)
                        result.solution = solution
                        result.description = description
                        result.success = True
                        result.message = "Erfolgreich gelöst."
                        self.get_logger().info(f"Solver Ergebnis: {solution}")
                    except ImportError:
                        msg = "kociemba nicht installiert (pip install kociemba)."
                        self.get_logger().error(msg)
                        result.success = False
                        result.message = msg
                    except Exception as e:
                        msg = f"Solver-Fehler: {e}"
                        self.get_logger().error(msg)
                        result.success = False
                        result.message = msg

            # Abschließendes Feedback
            feedback_msg.status = "Fertig."
            goal_handle.publish_feedback(feedback_msg)
            # Sende Result und setze Goal auf succeeded (auch bei Fehlern, so dass der Client das Ergebnis lesen kann)
            if (result.success):
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result

        finally:
            cap.release()
            # destroy preview window if used
            try:
                if bool(self.get_parameter("show_preview").value):
                    cv2.destroyAllWindows()
            except Exception:
                pass

    def capture_face_colors(self, cap: cv2.VideoCapture, sample_frames: int = 6, show_preview: bool = False) -> List[str]:
        """
        Erfasse die 9 Facelet-Farben für die aktuell sichtbare Seite.
        Wir nehmen mehrere Frames, extrahieren für jeden Frame die 3x3-Zellen,
        mitteln den HSV und klassifizieren die Farbe.

        Wenn show_preview True, wird ein Fenster mit Kamera-Bild und einer 3x3 Grid-Überlagerung angezeigt.
        """
        collected = []  # wird Liste von lists (frames x 9) sein
        attempts = 0
        max_attempts = sample_frames * 2
        while len(collected) < sample_frames and attempts < max_attempts:
            ret, frame = cap.read()
            attempts += 1
            if not ret:
                time.sleep(0.05)
                continue
            h, w, _ = frame.shape
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
                    # clamp coords
                    x1 = max(cx - 5, 0); x2 = min(cx + 5, w-1)
                    y1 = max(cy - 5, 0); y2 = min(cy + 5, h-1)
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

            # draw preview overlay if requested
            if show_preview:
                try:
                    overlay = frame.copy()
                    # draw outer box
                    cv2.rectangle(overlay, (x0, y0), (x0 + face_size, y0 + face_size), (0, 255, 0), 2)
                    # draw grid lines
                    for i in range(1, 3):
                        # vertical
                        x = x0 + i * cell
                        cv2.line(overlay, (x, y0), (x, y0 + face_size), (0, 255, 0), 1)
                        # horizontal
                        y = y0 + i * cell
                        cv2.line(overlay, (x0, y), (x0 + face_size, y), (0, 255, 0), 1)
                    # draw small centers
                    for row in range(3):
                        for col in range(3):
                            cx = x0 + col * cell + cell // 2
                            cy = y0 + row * cell + cell // 2
                            cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.imshow(self.preview_window, overlay)
                    cv2.waitKey(1)
                except Exception:
                    # GUI might not be available in the environment; ignore errors
                    pass

        if not collected:
            return None

        # Nun mitteln: für jedes facelet wähle die am häufigsten vorkommende Klassifikation über Frames
        final = []
        for idx in range(9):
            votes = {}
            for frame_colors in collected:
                c = frame_colors[idx]
                votes[c] = votes.get(c, 0) + 1
            # Wähle größtes Vote
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
        server = ScanCubeActionServer()
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
