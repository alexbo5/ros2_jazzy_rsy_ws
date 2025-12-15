#!/usr/bin/env python3
"""
Path Planning Action Server
Provides MoveJ (PTP) and MoveL (LIN) motion actions using Pilz Industrial Motion Planner.
All planning parameters are loaded from config/planning.yaml.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, JointConstraint
)
from moveit_msgs.action import ExecuteTrajectory
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose

from rsy_path_planning.action import MoveJ, MoveL, MoveJJoints, MoveLJoints


class PathPlanningActionServer(Node):

    def __init__(self):
        super().__init__('path_planning_action_server')

        # Declare and load parameters
        self._declare_parameters()
        self._load_config()

        cb = ReentrantCallbackGroup()

        # Service and action clients
        self._plan_client = self.create_client(
            GetMotionPlan, '/plan_kinematic_path', callback_group=cb
        )
        self._exec_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=cb
        )

        self._plan_client.wait_for_service(timeout_sec=30.0)
        self._exec_client.wait_for_server(timeout_sec=30.0)

        # Action servers
        ActionServer(self, MoveJ, 'move_j', execute_callback=self._exec_movej, callback_group=cb)
        ActionServer(self, MoveL, 'move_l', execute_callback=self._exec_movel, callback_group=cb)
        ActionServer(self, MoveJJoints, 'move_j_joints', execute_callback=self._exec_movej_joints, callback_group=cb)
        ActionServer(self, MoveLJoints, 'move_l_joints', execute_callback=self._exec_movel_joints, callback_group=cb)

        self.get_logger().info('Path Planning Server ready: /move_j, /move_l, /move_j_joints, /move_l_joints')

    def _declare_parameters(self):
        """Declare all parameters with defaults."""
        # Robot config
        self.declare_parameter('robots.robot1.planning_group', 'robot1_ur_manipulator')
        self.declare_parameter('robots.robot1.end_effector_link', 'robot1_tool0')
        self.declare_parameter('robots.robot2.planning_group', 'robot2_ur_manipulator')
        self.declare_parameter('robots.robot2.end_effector_link', 'robot2_tool0')

        # Planning parameters
        self.declare_parameter('planning.num_attempts', 10)
        self.declare_parameter('planning.allowed_time', 5.0)
        self.declare_parameter('planning.default_frame', 'world')

        # Velocity/acceleration scaling
        self.declare_parameter('velocity_scaling.ptp', 0.5)
        self.declare_parameter('velocity_scaling.linear', 0.3)
        self.declare_parameter('acceleration_scaling.ptp', 0.5)
        self.declare_parameter('acceleration_scaling.linear', 0.3)

        # Tolerances
        self.declare_parameter('tolerances.position', 0.001)
        self.declare_parameter('tolerances.orientation', 0.01)
        self.declare_parameter('tolerances.joint', 0.01)

    def _load_config(self):
        """Load configuration from parameters."""
        self.robots = {
            'robot1': {
                'group': self.get_parameter('robots.robot1.planning_group').value,
                'ee_link': self.get_parameter('robots.robot1.end_effector_link').value,
            },
            'robot2': {
                'group': self.get_parameter('robots.robot2.planning_group').value,
                'ee_link': self.get_parameter('robots.robot2.end_effector_link').value,
            },
        }

        # Joint names derived from planning group prefix
        self.joint_names = {
            'robot1': [f'robot1_{j}' for j in [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]],
            'robot2': [f'robot2_{j}' for j in [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]],
        }

        self.planning = {
            'num_attempts': self.get_parameter('planning.num_attempts').value,
            'allowed_time': self.get_parameter('planning.allowed_time').value,
            'default_frame': self.get_parameter('planning.default_frame').value,
        }

        self.vel_scale = {
            'PTP': self.get_parameter('velocity_scaling.ptp').value,
            'LIN': self.get_parameter('velocity_scaling.linear').value,
        }

        self.acc_scale = {
            'PTP': self.get_parameter('acceleration_scaling.ptp').value,
            'LIN': self.get_parameter('acceleration_scaling.linear').value,
        }

        self.tolerances = {
            'position': self.get_parameter('tolerances.position').value,
            'orientation': self.get_parameter('tolerances.orientation').value,
            'joint': self.get_parameter('tolerances.joint').value,
        }

    # === Action Callbacks ===

    async def _exec_movej(self, goal_handle):
        return await self._execute_pose(goal_handle, 'PTP', MoveJ)

    async def _exec_movel(self, goal_handle):
        return await self._execute_pose(goal_handle, 'LIN', MoveL)

    async def _exec_movej_joints(self, goal_handle):
        return await self._execute_joints(goal_handle, 'PTP', MoveJJoints)

    async def _exec_movel_joints(self, goal_handle):
        return await self._execute_joints(goal_handle, 'LIN', MoveLJoints)

    # === Motion Execution ===

    async def _execute_pose(self, goal_handle, planner_id, action_type):
        """Execute motion to Cartesian pose target."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        # Validate robot
        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        robot = self.robots[req.robot_name]
        frame = req.frame_id or self.planning['default_frame']

        # Build target pose
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

        self.get_logger().info(
            f'{planner_id} {req.robot_name}: pos=({req.x:.3f}, {req.y:.3f}, {req.z:.3f})'
        )

        # Plan
        feedback.status = 'Planning...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_pose_request(robot, pose, planner_id)
        plan_resp = await self._plan_client.call_async(plan_req)

        if plan_resp.motion_plan_response.error_code.val != 1:
            return self._abort(
                goal_handle, result,
                f'Planning failed (error {plan_resp.motion_plan_response.error_code.val})'
            )

        # Execute
        return await self._execute_trajectory(
            goal_handle, result, feedback,
            plan_resp.motion_plan_response.trajectory
        )

    async def _execute_joints(self, goal_handle, planner_id, action_type):
        """Execute motion to joint target."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        # Validate robot
        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        joint_names = self.joint_names[req.robot_name]
        joint_values = list(req.joint_values)

        if len(joint_values) != len(joint_names):
            return self._abort(
                goal_handle, result,
                f'Expected {len(joint_names)} joints, got {len(joint_values)}'
            )

        robot = self.robots[req.robot_name]

        self.get_logger().info(
            f'{planner_id} Joints {req.robot_name}: {[f"{v:.2f}" for v in joint_values]}'
        )

        # Plan
        feedback.status = 'Planning...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_joints_request(robot, joint_names, joint_values, planner_id)
        plan_resp = await self._plan_client.call_async(plan_req)

        if plan_resp.motion_plan_response.error_code.val != 1:
            return self._abort(
                goal_handle, result,
                f'Planning failed (error {plan_resp.motion_plan_response.error_code.val})'
            )

        # Execute
        return await self._execute_trajectory(
            goal_handle, result, feedback,
            plan_resp.motion_plan_response.trajectory
        )

    async def _execute_trajectory(self, goal_handle, result, feedback, trajectory):
        """Execute a planned trajectory."""
        feedback.status = 'Executing...'
        goal_handle.publish_feedback(feedback)

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = trajectory

        exec_handle = await self._exec_client.send_goal_async(exec_goal)
        if not exec_handle.accepted:
            return self._abort(goal_handle, result, 'Execution rejected')

        exec_result = await exec_handle.get_result_async()

        if exec_result.result.error_code.val == 1:
            result.success = True
            result.message = 'Done'
            goal_handle.succeed()
        else:
            result.success = False
            result.message = f'Execution failed (error {exec_result.result.error_code.val})'
            goal_handle.abort()

        return result

    # === Request Building ===

    def _build_pose_request(self, robot, pose, planner_id):
        """Build motion plan request for pose target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = robot['group']
        mp.pipeline_id = 'pilz_industrial_motion_planner'
        mp.planner_id = planner_id
        mp.num_planning_attempts = self.planning['num_attempts']
        mp.allowed_planning_time = self.planning['allowed_time']
        mp.max_velocity_scaling_factor = self.vel_scale[planner_id]
        mp.max_acceleration_scaling_factor = self.acc_scale[planner_id]
        mp.start_state.is_diff = True

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header = pose.header
        pos_constraint.link_name = robot['ee_link']
        pos_constraint.weight = 1.0

        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.tolerances['position']]
        bv.primitives.append(sphere)

        sphere_pose = Pose()
        sphere_pose.position = pose.pose.position
        sphere_pose.orientation.w = 1.0
        bv.primitive_poses.append(sphere_pose)
        pos_constraint.constraint_region = bv

        # Orientation constraint
        ori_constraint = OrientationConstraint()
        ori_constraint.header = pose.header
        ori_constraint.link_name = robot['ee_link']
        ori_constraint.orientation = pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = self.tolerances['orientation']
        ori_constraint.absolute_y_axis_tolerance = self.tolerances['orientation']
        ori_constraint.absolute_z_axis_tolerance = self.tolerances['orientation']
        ori_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        mp.goal_constraints.append(constraints)

        return req

    def _build_joints_request(self, robot, joint_names, joint_values, planner_id):
        """Build motion plan request for joint target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = robot['group']
        mp.pipeline_id = 'pilz_industrial_motion_planner'
        mp.planner_id = planner_id
        mp.num_planning_attempts = self.planning['num_attempts']
        mp.allowed_planning_time = self.planning['allowed_time']
        mp.max_velocity_scaling_factor = self.vel_scale[planner_id]
        mp.max_acceleration_scaling_factor = self.acc_scale[planner_id]
        mp.start_state.is_diff = True

        constraints = Constraints()
        for name, value in zip(joint_names, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = self.tolerances['joint']
            jc.tolerance_below = self.tolerances['joint']
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        mp.goal_constraints.append(constraints)
        return req

    # === Helpers ===

    def _abort(self, goal_handle, result, message):
        """Abort action with error message."""
        self.get_logger().error(message)
        result.success = False
        result.message = message
        goal_handle.abort()
        return result


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
