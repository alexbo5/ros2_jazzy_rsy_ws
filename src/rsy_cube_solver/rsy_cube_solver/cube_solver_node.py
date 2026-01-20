#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time

from rsy_cube_perception.action import ScanCubeFace, CalibrateCamera
from rsy_cube_perception.srv import SolveCube, ShowPreviewWindow
from rsy_cube_motion.action import RotateFace, TakeUpCube, PutDownCube, PresentCubeFace, StartupPosition

class CubeSolver(Node):
    MAX_SCAN_RETRIES = 2  # Number of times to retry scanning if solving fails

    def __init__(self):
        super().__init__('cube_solver')
        self.scan_cube_face_client = ActionClient(self, ScanCubeFace, 'scan_cube_face')
        self.solve_cube_client = self.create_client(SolveCube, 'solve_cube')
        self.show_preview_window_client = self.create_client(ShowPreviewWindow, 'show_preview_window')
        self.face_rotation_client = ActionClient(self, RotateFace, 'rotate_face')
        self.take_up_cube_client = ActionClient(self, TakeUpCube, 'take_up_cube')
        self.put_down_cube_client = ActionClient(self, PutDownCube, 'put_down_cube')
        self.present_cube_face_client = ActionClient(self, PresentCubeFace, 'present_cube_face')
        self.startup_position_client = ActionClient(self, StartupPosition, 'startup_position')

    def run(self):
        self.get_logger().info("Main started...")

        # Show preview window at startup
        self.get_logger().info("Starting preview window...")
        if self.show_preview_window_client.wait_for_service(timeout_sec=10.0):
            request = ShowPreviewWindow.Request()
            request.enable = True
            response_future = self.show_preview_window_client.call_async(request)
            rclpy.spin_until_future_complete(self, response_future)
            response = response_future.result()
            if response and response.success:
                self.get_logger().info("Preview window started successfully")
            else:
                self.get_logger().warn(f"Preview window start failed: {response.message if response else 'no response'}")
        else:
            self.get_logger().warn("Preview window service not available")

        # Move to startup position (open grippers, move robot 1 to pretarget)
        self.get_logger().info("Moving to startup position...")
        if not self.startup_position_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Startup position action server not available")
            return

        goal = StartupPosition.Goal()
        send_goal_future = self.startup_position_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            self.get_logger().error("Startup position goal was rejected or no goal handle received")
            return

        self.get_logger().info("Startup position goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        res = result_future.result()
        if res is None or not res.result.success:
            self.get_logger().error("Failed to move to startup position")
            return

        self.get_logger().info("Startup position reached successfully!")

        # Take up the cube before scanning
        self.get_logger().info("Taking up the cube...")
        if not self.take_up_cube_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Take up cube action server not available")
            return
        
        goal = TakeUpCube.Goal()
        send_goal_future = self.take_up_cube_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            self.get_logger().error("Take up cube goal was rejected or no goal handle received")
            return

        self.get_logger().info("Take up cube goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        res = result_future.result()
        if res is None or not res.result.success:
            self.get_logger().error("Failed to take up the cube")
            return
        
        self.get_logger().info("Cube taken up successfully!")

        # Scan and solve with retry logic
        solution = None
        for attempt in range(1, self.MAX_SCAN_RETRIES + 2):  # +2 for initial attempt + retries
            self.get_logger().info(f"Scan attempt {attempt}/{self.MAX_SCAN_RETRIES + 1}...")

            # Present and scan all faces
            if not self._scan_all_faces():
                self.get_logger().error("Failed to scan cube faces")
                return

            # Try to solve
            solution = self._solve_cube()
            if solution is not None:
                break  # Success!

            if attempt <= self.MAX_SCAN_RETRIES:
                self.get_logger().warn(f"Solving failed, retrying scan ({attempt}/{self.MAX_SCAN_RETRIES} retries used)...")
            else:
                self.get_logger().error(f"Solving failed after {self.MAX_SCAN_RETRIES + 1} attempts")
                self._put_down_cube()
                return

        # Execute the solution moves
        for move in solution:
            if not self.face_rotation_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("Cube motion action server not available")
                return
            
            self.get_logger().info("Next move: " + move)

            goal = RotateFace.Goal()
            goal.move = move
            send_goal_future = self.face_rotation_client.send_goal_async(goal)
            # block until send_goal completes
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                self.get_logger().error("Face rotation goal was rejected or no goal handle received")
                return

            self.get_logger().info("Face rotation goal accepted, waiting for result...")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            res = result_future.result()
            if res is None:
                self.get_logger().error("Failed to get result from cube motion server")
                return
            if (not res.result.success):
                self.get_logger().error("Face rotation failed")
                return    # abort entire solving when one movement failed 

        self.get_logger().info("Cube solved!")

        # Put down the cube after solving
        self._put_down_cube()

    def _scan_all_faces(self):
        """Present and scan all cube faces.

        Returns:
            bool: True if all faces were scanned successfully, False otherwise.
        """
        self.get_logger().info("Presenting and scanning all cube faces...")
        faces_to_scan = ["U", "D", "F", "B", "L", "R"]

        for face in faces_to_scan:
            # Present the face
            self.get_logger().info(f"Presenting cube face {face}...")
            if not self.present_cube_face_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("Present cube face action server not available")
                return False

            goal = PresentCubeFace.Goal()
            goal.face = face
            send_goal_future = self.present_cube_face_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                self.get_logger().error(f"Present cube face goal for {face} was rejected")
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            res = result_future.result()
            if res is None or not res.result.success:
                self.get_logger().error(f"Failed to present cube face {face}")
                return False

            self.get_logger().info(f"Face {face} presented successfully!")

            # Scan the presented face
            self.get_logger().info(f"Scanning face {face}...")
            if not self.scan_cube_face_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("Scan cube face action server not available")
                return False

            goal = ScanCubeFace.Goal()
            goal.cube_face = face
            send_goal_future = self.scan_cube_face_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                self.get_logger().error(f"Scan cube face goal for {face} was rejected")
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            res = result_future.result()
            if res is None or not res.result.success:
                self.get_logger().error(f"Failed to scan face {face}")
                return False

            self.get_logger().info(f"Face {face} scanned: {res.result.colors}")

        self.get_logger().info("All cube faces presented and scanned successfully!")
        return True

    def _solve_cube(self):
        """Call the solve_cube service.

        Returns:
            list: List of move strings if successful, None otherwise.
        """
        self.get_logger().info("Calling solve_cube service...")
        if not self.solve_cube_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Solve cube service not available")
            return None

        request = SolveCube.Request()
        response_future = self.solve_cube_client.call_async(request)
        rclpy.spin_until_future_complete(self, response_future)
        response = response_future.result()

        if response is None or not response.success:
            self.get_logger().error(f"Solve cube service failed: {response.message if response else 'no response'}")
            return None

        self.get_logger().info(f"Solution found: {response.solution}")
        self.get_logger().info(f"Description: {response.description}")

        return response.solution.split()

    def _put_down_cube(self):
        """Put down the cube. Used for cleanup after errors."""
        self.get_logger().info("Putting down the cube...")
        if not self.put_down_cube_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Put down cube action server not available")
            return False

        goal = PutDownCube.Goal()
        send_goal_future = self.put_down_cube_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            self.get_logger().error("Put down cube goal was rejected or no goal handle received")
            return False

        self.get_logger().info("Put down cube goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        res = result_future.result()
        if res is None or not res.result.success:
            self.get_logger().error("Failed to put down the cube")
            return False

        self.get_logger().info("Cube put down successfully!")
        return True

def main(args=None):
    rclpy.init(args=args)
    node = CubeSolver()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()