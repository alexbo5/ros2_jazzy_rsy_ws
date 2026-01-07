#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time

from rsy_cube_perception.action import ScanCubeFace, CalibrateCamera
from rsy_cube_perception.srv import SolveCube
from rsy_cube_motion.action import RotateFace, TakeUpCube, PutDownCube, PresentCubeFace

class CubeSolver(Node):
    def __init__(self):
        super().__init__('cube_solver')        
        self.scan_cube_face_client = ActionClient(self, ScanCubeFace, 'scan_cube_face')
        self.solve_cube_client = self.create_client(SolveCube, 'solve_cube')
        self.face_rotation_client = ActionClient(self, RotateFace, 'rotate_face')
        self.take_up_cube_client = ActionClient(self, TakeUpCube, 'take_up_cube')
        self.put_down_cube_client = ActionClient(self, PutDownCube, 'put_down_cube')
        self.present_cube_face_client = ActionClient(self, PresentCubeFace, 'present_cube_face')

    def run(self):
        self.get_logger().info("Main started...")
        
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
        
        # Present each cube face and scan it
        self.get_logger().info("Presenting and scanning all cube faces...")
        faces_to_scan = ["U", "D", "F", "B", "L", "R"]
        
        for face in faces_to_scan:
            # Present the face
            self.get_logger().info(f"Presenting cube face {face}...")
            if not self.present_cube_face_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("Present cube face action server not available")
                return
            
            goal = PresentCubeFace.Goal()
            goal.face = face
            send_goal_future = self.present_cube_face_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                self.get_logger().error(f"Present cube face goal for {face} was rejected")
                return
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            res = result_future.result()
            if res is None or not res.result.success:
                self.get_logger().error(f"Failed to present cube face {face}")
                return
            
            self.get_logger().info(f"Face {face} presented successfully!")
            
            # Delay before scanning (for development - allows robot to stabilize position)
            # delay_seconds = 10.0
            # if delay_seconds > 0:
            #     self.get_logger().info(f"Waiting {delay_seconds} seconds before scanning...")
            #     time.sleep(delay_seconds)
            
            # Scan the presented face
            self.get_logger().info(f"Scanning face {face}...")
            if not self.scan_cube_face_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("Scan cube face action server not available")
                return
            
            goal = ScanCubeFace.Goal()
            goal.cube_face = face
            send_goal_future = self.scan_cube_face_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                self.get_logger().error(f"Scan cube face goal for {face} was rejected")
                return
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            res = result_future.result()
            if res is None or not res.result.success:
                self.get_logger().error(f"Failed to scan face {face}")
                return
            
            self.get_logger().info(f"Face {face} scanned: {res.result.colors}")
        
        self.get_logger().info("All cube faces presented and scanned successfully!")
        
        # Call solve_cube service
        self.get_logger().info("Calling solve_cube service...")
        if not self.solve_cube_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Solve cube service not available")
            return
        
        request = SolveCube.Request()
        response_future = self.solve_cube_client.call_async(request)
        rclpy.spin_until_future_complete(self, response_future)
        response = response_future.result()
        
        if response is None or not response.success:
            self.get_logger().error(f"Solve cube service failed: {response.message if response else 'no response'}")
            return
        
        self.get_logger().info(f"Cube solved! Solution: {response.solution}")
        self.get_logger().info(f"Description: {response.description}")
        
        solution = response.solution.split()
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
        self.get_logger().info("Putting down the cube...")
        if not self.put_down_cube_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Put down cube action server not available")
            return
        
        goal = PutDownCube.Goal()
        send_goal_future = self.put_down_cube_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            self.get_logger().error("Put down cube goal was rejected or no goal handle received")
            return

        self.get_logger().info("Put down cube goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        res = result_future.result()
        if res is None or not res.result.success:
            self.get_logger().error("Failed to put down the cube")
            return
        
        self.get_logger().info("Cube put down successfully!")

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