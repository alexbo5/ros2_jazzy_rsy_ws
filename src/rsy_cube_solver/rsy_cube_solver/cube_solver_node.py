#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from rsy_cube_perception.action import ScanCube
from rsy_cube_motion.action import RotateFace, HandOver, TakeUpCube, PutDownCube

class CubeSolver(Node):
    def __init__(self):
        super().__init__('cube_solver')
        self.perception_client = ActionClient(self, ScanCube, 'scan_cube')
        self.face_rotation_client = ActionClient(self, RotateFace, 'rotate_face')
        self.take_up_cube_client = ActionClient(self, TakeUpCube, 'take_up_cube')
        self.put_down_cube_client = ActionClient(self, PutDownCube, 'put_down_cube')

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
        
        # wait for action server (blocking)
        if not self.perception_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Perception action server not available")
            return

        goal = ScanCube.Goal()
        #goal.camera_index = 1
        send_goal_future = self.perception_client.send_goal_async(goal)
        # block until send_goal completes
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            self.get_logger().error("Perception goal was rejected or no goal handle received")
            return

        self.get_logger().info("Perception goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        res = result_future.result()
        if res is None:
            self.get_logger().error("Failed to get result from perception server")
            return

        solution = res.result.solution.split()
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