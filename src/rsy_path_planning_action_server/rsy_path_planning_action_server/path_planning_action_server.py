#!/usr/bin/env python3
"""
Path Planning Action Server - MoveJ (PTP) and MoveL (LIN) via Pilz planner.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from moveit_msgs.action import ExecuteTrajectory
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose

from rsy_path_planning_action_server.action import MoveJ, MoveL


class PathPlanningActionServer(Node):

    ROBOTS = {
        'robot1': ('robot1_ur_manipulator', 'robot1_tool0'),
        'robot2': ('robot2_ur_manipulator', 'robot2_tool0'),
    }

    def __init__(self):
        super().__init__('path_planning_action_server')

        cb = ReentrantCallbackGroup()

        self._plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path', callback_group=cb)
        self._exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory', callback_group=cb)

        self._plan_client.wait_for_service(timeout_sec=30.0)
        self._exec_client.wait_for_server(timeout_sec=30.0)

        ActionServer(self, MoveJ, 'move_j', execute_callback=self._exec_movej, callback_group=cb)
        ActionServer(self, MoveL, 'move_l', execute_callback=self._exec_movel, callback_group=cb)

        self.get_logger().info('Ready: /move_j (PTP), /move_l (LIN)')

    async def _exec_movej(self, goal_handle):
        return await self._execute(goal_handle, 'PTP', MoveJ)

    async def _exec_movel(self, goal_handle):
        return await self._execute(goal_handle, 'LIN', MoveL)

    async def _execute(self, goal_handle, planner_id, action_type):
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.ROBOTS:
            result.success = False
            result.message = f'Unknown robot: {req.robot_name}'
            goal_handle.abort()
            return result

        group, ee_link = self.ROBOTS[req.robot_name]
        frame = req.frame_id or 'world'

        # Build pose
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = req.x
        pose.pose.position.y = req.y
        pose.pose.position.z = req.z
        pose.pose.orientation.x = req.qx
        pose.pose.orientation.y = req.qy
        pose.pose.orientation.z = req.qz
        pose.pose.orientation.w = req.qw

        self.get_logger().info(f'{planner_id} {req.robot_name}: ({req.x:.3f}, {req.y:.3f}, {req.z:.3f})')

        # Plan
        feedback.status = 'Planning...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_plan_request(group, ee_link, pose, planner_id)
        plan_resp = await self._plan_client.call_async(plan_req)

        if plan_resp.motion_plan_response.error_code.val != 1:
            result.success = False
            result.message = f'Planning failed: {plan_resp.motion_plan_response.error_code.val}'
            goal_handle.abort()
            return result

        # Execute
        feedback.status = 'Executing...'
        goal_handle.publish_feedback(feedback)

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = plan_resp.motion_plan_response.trajectory

        exec_handle = await self._exec_client.send_goal_async(exec_goal)
        if not exec_handle.accepted:
            result.success = False
            result.message = 'Execution rejected'
            goal_handle.abort()
            return result

        exec_result = await exec_handle.get_result_async()

        if exec_result.result.error_code.val == 1:
            result.success = True
            result.message = 'Done'
            goal_handle.succeed()
        else:
            result.success = False
            result.message = f'Execution failed: {exec_result.result.error_code.val}'
            goal_handle.abort()

        return result

    def _build_plan_request(self, group, ee_link, pose, planner_id):
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = group
        mp.pipeline_id = 'pilz_industrial_motion_planner'
        mp.planner_id = planner_id
        mp.num_planning_attempts = 10
        mp.allowed_planning_time = 5.0
        mp.max_velocity_scaling_factor = 0.1
        mp.max_acceleration_scaling_factor = 0.1
        mp.start_state.is_diff = True

        # Position constraint
        pos = PositionConstraint()
        pos.header = pose.header
        pos.link_name = ee_link
        pos.weight = 1.0
        bv = BoundingVolume()
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [0.001]
        bv.primitives.append(prim)
        p = Pose()
        p.position = pose.pose.position
        p.orientation.w = 1.0
        bv.primitive_poses.append(p)
        pos.constraint_region = bv

        # Orientation constraint
        ori = OrientationConstraint()
        ori.header = pose.header
        ori.link_name = ee_link
        ori.orientation = pose.pose.orientation
        ori.absolute_x_axis_tolerance = 0.01
        ori.absolute_y_axis_tolerance = 0.01
        ori.absolute_z_axis_tolerance = 0.01
        ori.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos)
        constraints.orientation_constraints.append(ori)
        mp.goal_constraints.append(constraints)

        return req


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
