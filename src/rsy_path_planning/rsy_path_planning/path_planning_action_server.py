#!/usr/bin/env python3
"""
Path Planning Action Server - MoveJ (PTP) and MoveL (LIN) via Pilz planner.
Supports both Cartesian pose and joint value targets.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, JointConstraint
from moveit_msgs.action import ExecuteTrajectory
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose

from rsy_path_planning.action import MoveJ, MoveL, MoveJJoints, MoveLJoints


class PathPlanningActionServer(Node):

    ROBOTS = {
        'robot1': ('robot1_ur_manipulator', 'robot1_tool0'),
        'robot2': ('robot2_ur_manipulator', 'robot2_tool0'),
    }

    # Joint names for each robot (UR robot has 6 joints)
    JOINT_NAMES = {
        'robot1': [
            'robot1_shoulder_pan_joint',
            'robot1_shoulder_lift_joint',
            'robot1_elbow_joint',
            'robot1_wrist_1_joint',
            'robot1_wrist_2_joint',
            'robot1_wrist_3_joint',
        ],
        'robot2': [
            'robot2_shoulder_pan_joint',
            'robot2_shoulder_lift_joint',
            'robot2_elbow_joint',
            'robot2_wrist_1_joint',
            'robot2_wrist_2_joint',
            'robot2_wrist_3_joint',
        ],
    }

    def __init__(self):
        super().__init__('path_planning_action_server')

        cb = ReentrantCallbackGroup()

        self._plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path', callback_group=cb)
        self._exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory', callback_group=cb)

        self._plan_client.wait_for_service(timeout_sec=30.0)
        self._exec_client.wait_for_server(timeout_sec=30.0)

        # Cartesian pose actions
        ActionServer(self, MoveJ, 'move_j', execute_callback=self._exec_movej, callback_group=cb)
        ActionServer(self, MoveL, 'move_l', execute_callback=self._exec_movel, callback_group=cb)

        # Joint value actions
        ActionServer(self, MoveJJoints, 'move_j_joints', execute_callback=self._exec_movej_joints, callback_group=cb)
        ActionServer(self, MoveLJoints, 'move_l_joints', execute_callback=self._exec_movel_joints, callback_group=cb)

        self.get_logger().info('Ready: /move_j (PTP), /move_l (LIN), /move_j_joints (PTP), /move_l_joints (LIN)')

    async def _exec_movej(self, goal_handle):
        return await self._execute_cartesian(goal_handle, 'PTP', MoveJ)

    async def _exec_movel(self, goal_handle):
        return await self._execute_cartesian(goal_handle, 'LIN', MoveL)

    async def _exec_movej_joints(self, goal_handle):
        return await self._execute_joints(goal_handle, 'PTP', MoveJJoints)

    async def _exec_movel_joints(self, goal_handle):
        return await self._execute_joints(goal_handle, 'LIN', MoveLJoints)

    async def _execute_cartesian(self, goal_handle, planner_id, action_type):
        """Execute motion to Cartesian pose target."""
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

        plan_req = self._build_cartesian_plan_request(group, ee_link, pose, planner_id)
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

    async def _execute_joints(self, goal_handle, planner_id, action_type):
        """Execute motion to joint value target."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.ROBOTS:
            result.success = False
            result.message = f'Unknown robot: {req.robot_name}'
            goal_handle.abort()
            return result

        if req.robot_name not in self.JOINT_NAMES:
            result.success = False
            result.message = f'No joint names defined for robot: {req.robot_name}'
            goal_handle.abort()
            return result

        joint_names = self.JOINT_NAMES[req.robot_name]
        joint_values = list(req.joint_values)

        if len(joint_values) != len(joint_names):
            result.success = False
            result.message = f'Expected {len(joint_names)} joint values, got {len(joint_values)}'
            goal_handle.abort()
            return result

        group, _ = self.ROBOTS[req.robot_name]

        self.get_logger().info(f'{planner_id} Joints {req.robot_name}: {[f"{v:.3f}" for v in joint_values]}')

        # Plan
        feedback.status = 'Planning...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_joints_plan_request(group, joint_names, joint_values, planner_id)
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

    def _build_cartesian_plan_request(self, group, ee_link, pose, planner_id):
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = group
        mp.pipeline_id = 'pilz_industrial_motion_planner'
        mp.planner_id = planner_id
        mp.num_planning_attempts = 50
        mp.allowed_planning_time = 15.0
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
        prim.dimensions = [0.01]  # 10mm tolerance
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
        ori.absolute_x_axis_tolerance = 0.1  # ~5.7° tolerance
        ori.absolute_y_axis_tolerance = 0.1
        ori.absolute_z_axis_tolerance = 0.1
        ori.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos)
        constraints.orientation_constraints.append(ori)
        mp.goal_constraints.append(constraints)

        return req

    def _build_joints_plan_request(self, group, joint_names, joint_values, planner_id):
        """Build motion plan request for joint value target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = group
        mp.pipeline_id = 'pilz_industrial_motion_planner'
        mp.planner_id = planner_id
        mp.num_planning_attempts = 50
        mp.allowed_planning_time = 15.0
        mp.max_velocity_scaling_factor = 0.1
        mp.max_acceleration_scaling_factor = 0.1
        mp.start_state.is_diff = True

        # Joint constraints
        constraints = Constraints()
        for name, value in zip(joint_names, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.01  # ~0.57° tolerance
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

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
