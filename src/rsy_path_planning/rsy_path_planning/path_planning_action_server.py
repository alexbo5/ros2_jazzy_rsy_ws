#!/usr/bin/env python3
"""
Path Planning Action Server with Hybrid Planning Strategy

Actions:
- MoveJ/MoveJJoints: PTP motion (Pilz) with OMPL fallback for obstacles
- MoveL/MoveLJoints: LIN motion (Pilz) with Cartesian path fallback

Strategy:
1. Try Pilz planner first (deterministic, efficient)
2. Fall back to OMPL if Pilz fails (obstacle avoidance)

Features:
- Shortest joint path selection to prevent robot spinning
- Automatic planner fallback for robust obstacle handling
- Collision-checked paths via MoveIt planning scene
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetMotionPlan, GetPositionFK, GetCartesianPath
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, JointConstraint, MoveItErrorCodes, RobotState
)
from moveit_msgs.action import ExecuteTrajectory
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
import tf2_ros

from rsy_path_planning.action import MoveJ, MoveL, MoveJJoints, MoveLJoints


class PathPlanningActionServer(Node):
    """Path planning server with hybrid Pilz/OMPL strategy."""

    # Continuous joints that can rotate ±360° - need shortest path handling
    CONTINUOUS_JOINT_INDICES = [0, 3, 4, 5]  # shoulder_pan, wrist_1, wrist_2, wrist_3

    # MoveIt error codes worth retrying with different planner
    # Includes collision errors and IK failures (IK can fail due to collision during solve)
    RETRYABLE_ERRORS = {
        MoveItErrorCodes.PLANNING_FAILED,
        MoveItErrorCodes.INVALID_MOTION_PLAN,
        MoveItErrorCodes.GOAL_IN_COLLISION,
        MoveItErrorCodes.START_STATE_IN_COLLISION,
        MoveItErrorCodes.NO_IK_SOLUTION,  # IK can fail due to collision during solve
        MoveItErrorCodes.FAILURE,  # Generic failure, worth retrying with OMPL
    }

    def __init__(self):
        super().__init__('path_planning_action_server')

        self._declare_parameters()
        self._load_config()

        cb = ReentrantCallbackGroup()

        # Service clients
        self._plan_client = self.create_client(
            GetMotionPlan, '/plan_kinematic_path', callback_group=cb
        )
        self._fk_client = self.create_client(
            GetPositionFK, '/compute_fk', callback_group=cb
        )
        self._cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=cb
        )

        # Action client for trajectory execution
        self._exec_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=cb
        )

        # Wait for services
        self.get_logger().info('Waiting for planning services...')
        self._plan_client.wait_for_service(timeout_sec=30.0)
        self._fk_client.wait_for_service(timeout_sec=30.0)
        self._cartesian_client.wait_for_service(timeout_sec=30.0)
        self._exec_client.wait_for_server(timeout_sec=30.0)

        # TF buffer
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Joint state subscriber
        self._current_joint_states = {}
        self._joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10
        )

        # Action servers
        ActionServer(self, MoveJ, 'move_j', execute_callback=self._exec_movej, callback_group=cb)
        ActionServer(self, MoveL, 'move_l', execute_callback=self._exec_movel, callback_group=cb)
        ActionServer(self, MoveJJoints, 'move_j_joints', execute_callback=self._exec_movej_joints, callback_group=cb)
        ActionServer(self, MoveLJoints, 'move_l_joints', execute_callback=self._exec_movel_joints, callback_group=cb)

        self.get_logger().info('Path Planning Server ready (Hybrid Pilz/OMPL)')
        self.get_logger().info(f'  MoveJ/MoveJJoints: PTP -> OMPL fallback')
        self.get_logger().info(f'  MoveL/MoveLJoints: LIN -> Cartesian fallback')

    def _declare_parameters(self):
        """Declare all parameters with defaults."""
        # Robot configuration
        self.declare_parameter('robots.robot1.planning_group', 'robot1_ur_manipulator')
        self.declare_parameter('robots.robot1.end_effector_link', 'robot1_tcp')
        self.declare_parameter('robots.robot2.planning_group', 'robot2_ur_manipulator')
        self.declare_parameter('robots.robot2.end_effector_link', 'robot2_tcp')

        # Planning parameters
        self.declare_parameter('planning.default_frame', 'world')
        self.declare_parameter('planning.allowed_time', 5.0)

        # OMPL fallback parameters
        self.declare_parameter('ompl.planner_id', 'RRTConnect')
        self.declare_parameter('ompl.num_attempts', 10)
        self.declare_parameter('ompl.allowed_time', 10.0)

        # Cartesian path parameters (for MoveL fallback)
        self.declare_parameter('cartesian.max_step', 0.01)
        self.declare_parameter('cartesian.min_fraction', 0.95)

        # Velocity and acceleration scaling (0.0 - 1.0)
        self.declare_parameter('velocity_scaling.ptp', 0.5)
        self.declare_parameter('velocity_scaling.lin', 0.3)
        self.declare_parameter('acceleration_scaling.ptp', 0.5)
        self.declare_parameter('acceleration_scaling.lin', 0.3)

        # Goal tolerances
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
            'default_frame': self.get_parameter('planning.default_frame').value,
            'allowed_time': self.get_parameter('planning.allowed_time').value,
        }

        self.ompl = {
            'planner_id': self.get_parameter('ompl.planner_id').value,
            'num_attempts': self.get_parameter('ompl.num_attempts').value,
            'allowed_time': self.get_parameter('ompl.allowed_time').value,
        }

        self.cartesian = {
            'max_step': self.get_parameter('cartesian.max_step').value,
            'min_fraction': self.get_parameter('cartesian.min_fraction').value,
        }

        self.vel_scale = {
            'ptp': self.get_parameter('velocity_scaling.ptp').value,
            'lin': self.get_parameter('velocity_scaling.lin').value,
        }

        self.acc_scale = {
            'ptp': self.get_parameter('acceleration_scaling.ptp').value,
            'lin': self.get_parameter('acceleration_scaling.lin').value,
        }

        self.tolerances = {
            'position': self.get_parameter('tolerances.position').value,
            'orientation': self.get_parameter('tolerances.orientation').value,
            'joint': self.get_parameter('tolerances.joint').value,
        }

    def _joint_state_callback(self, msg):
        """Store current joint states."""
        for i, name in enumerate(msg.name):
            self._current_joint_states[name] = msg.position[i]

    def _get_current_joint_values(self, robot_name):
        """Get current joint values for a robot."""
        joint_names = self.joint_names[robot_name]
        values = []
        for name in joint_names:
            if name in self._current_joint_states:
                values.append(self._current_joint_states[name])
            else:
                self.get_logger().warn(f'Joint {name} not found, using 0.0')
                values.append(0.0)
        return values

    def _get_current_robot_state(self, robot_name):
        """Get current robot state for planning."""
        robot_state = RobotState()
        joint_names = self.joint_names[robot_name]
        robot_state.joint_state.name = joint_names
        robot_state.joint_state.position = self._get_current_joint_values(robot_name)
        return robot_state

    def _normalize_joint_target(self, robot_name, target_values):
        """
        Normalize joint targets to shortest path from current position.
        For continuous joints (±360°), select equivalent angle closest to current.
        """
        current_values = self._get_current_joint_values(robot_name)
        normalized = list(target_values)

        for idx in self.CONTINUOUS_JOINT_INDICES:
            if idx < len(normalized):
                current = current_values[idx]
                target = normalized[idx]

                # Find equivalent angle closest to current
                diff = target - current
                while diff > math.pi:
                    diff -= 2 * math.pi
                while diff < -math.pi:
                    diff += 2 * math.pi

                normalized[idx] = current + diff

        return normalized

    # === Action Callbacks ===

    async def _exec_movej(self, goal_handle):
        """MoveJ: PTP motion to Cartesian target with OMPL fallback."""
        return await self._execute_ptp_pose(goal_handle, MoveJ)

    async def _exec_movel(self, goal_handle):
        """MoveL: LIN motion with Cartesian path fallback."""
        return await self._execute_lin_pose(goal_handle, MoveL)

    async def _exec_movej_joints(self, goal_handle):
        """MoveJJoints: PTP motion to joint target with OMPL fallback."""
        return await self._execute_ptp_joints(goal_handle, MoveJJoints)

    async def _exec_movel_joints(self, goal_handle):
        """MoveLJoints: LIN motion to joint target with Cartesian fallback."""
        return await self._execute_lin_joints(goal_handle, MoveLJoints)

    # === PTP Planning (MoveJ) with OMPL Fallback ===

    async def _execute_ptp_pose(self, goal_handle, action_type):
        """Execute PTP motion to Cartesian pose with OMPL fallback."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        robot = self.robots[req.robot_name]
        frame = req.frame_id or self.planning['default_frame']

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
            f'MoveJ {req.robot_name}: ({req.x:.3f}, {req.y:.3f}, {req.z:.3f})'
        )

        # Try Pilz PTP first
        feedback.status = 'Planning PTP motion (Pilz)...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_pilz_pose_request(req.robot_name, robot, pose, 'PTP')
        plan_resp = await self._plan_client.call_async(plan_req)
        error_code = plan_resp.motion_plan_response.error_code.val

        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('PTP planning succeeded (Pilz)')
            return await self._execute_trajectory(
                goal_handle, result, feedback,
                plan_resp.motion_plan_response.trajectory
            )

        # Check if we should try OMPL fallback
        error_name = self._get_error_name(error_code)
        self.get_logger().warn(f'Pilz PTP failed: {error_name}, trying OMPL...')

        if error_code not in self.RETRYABLE_ERRORS:
            return self._abort(goal_handle, result, f'PTP planning failed: {error_name}')

        # Fall back to OMPL
        feedback.status = 'Planning with OMPL (obstacle avoidance)...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_ompl_pose_request(req.robot_name, robot, pose)
        plan_resp = await self._plan_client.call_async(plan_req)
        error_code = plan_resp.motion_plan_response.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            error_name = self._get_error_name(error_code)
            return self._abort(goal_handle, result, f'OMPL planning failed: {error_name}')

        self.get_logger().info('OMPL planning succeeded')
        return await self._execute_trajectory(
            goal_handle, result, feedback,
            plan_resp.motion_plan_response.trajectory
        )

    async def _execute_ptp_joints(self, goal_handle, action_type):
        """Execute PTP motion to joint target with OMPL fallback."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        joint_names = self.joint_names[req.robot_name]
        joint_values = list(req.joint_values)

        if len(joint_values) != len(joint_names):
            return self._abort(goal_handle, result,
                f'Expected {len(joint_names)} joints, got {len(joint_values)}')

        # Normalize to shortest path
        joint_values = self._normalize_joint_target(req.robot_name, joint_values)
        robot = self.robots[req.robot_name]

        self.get_logger().info(
            f'MoveJJoints {req.robot_name}: {[f"{v:.2f}" for v in joint_values]}'
        )

        # Try Pilz PTP first
        feedback.status = 'Planning PTP motion (Pilz)...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_pilz_joints_request(req.robot_name, robot, joint_names, joint_values, 'PTP')
        plan_resp = await self._plan_client.call_async(plan_req)
        error_code = plan_resp.motion_plan_response.error_code.val

        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('PTP planning succeeded (Pilz)')
            return await self._execute_trajectory(
                goal_handle, result, feedback,
                plan_resp.motion_plan_response.trajectory
            )

        # Check if we should try OMPL fallback
        error_name = self._get_error_name(error_code)
        self.get_logger().warn(f'Pilz PTP failed: {error_name}, trying OMPL...')

        if error_code not in self.RETRYABLE_ERRORS:
            return self._abort(goal_handle, result, f'PTP planning failed: {error_name}')

        # Fall back to OMPL
        feedback.status = 'Planning with OMPL (obstacle avoidance)...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_ompl_joints_request(req.robot_name, robot, joint_names, joint_values)
        plan_resp = await self._plan_client.call_async(plan_req)
        error_code = plan_resp.motion_plan_response.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            error_name = self._get_error_name(error_code)
            return self._abort(goal_handle, result, f'OMPL planning failed: {error_name}')

        self.get_logger().info('OMPL planning succeeded')
        return await self._execute_trajectory(
            goal_handle, result, feedback,
            plan_resp.motion_plan_response.trajectory
        )

    # === LIN Planning (MoveL) with Cartesian Fallback ===

    async def _execute_lin_pose(self, goal_handle, action_type):
        """Execute LIN motion to pose with Cartesian path fallback."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        robot = self.robots[req.robot_name]
        frame = req.frame_id or self.planning['default_frame']

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
            f'MoveL {req.robot_name}: ({req.x:.3f}, {req.y:.3f}, {req.z:.3f})'
        )

        # Try Pilz LIN first
        feedback.status = 'Planning LIN motion (Pilz)...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_pilz_pose_request(req.robot_name, robot, pose, 'LIN')
        plan_resp = await self._plan_client.call_async(plan_req)
        error_code = plan_resp.motion_plan_response.error_code.val

        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('LIN planning succeeded (Pilz)')
            return await self._execute_trajectory(
                goal_handle, result, feedback,
                plan_resp.motion_plan_response.trajectory
            )

        # Fall back to computeCartesianPath
        error_name = self._get_error_name(error_code)
        self.get_logger().warn(f'Pilz LIN failed: {error_name}, trying Cartesian path...')

        feedback.status = 'Planning Cartesian path...'
        goal_handle.publish_feedback(feedback)

        trajectory = await self._plan_cartesian_path(req.robot_name, robot, pose.pose, frame)
        if trajectory is None:
            return self._abort(goal_handle, result, 'Cartesian path planning failed')

        self.get_logger().info('Cartesian path planning succeeded')
        return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

    async def _execute_lin_joints(self, goal_handle, action_type):
        """Execute LIN motion to joint target with Cartesian fallback."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        joint_names = self.joint_names[req.robot_name]
        joint_values = list(req.joint_values)

        if len(joint_values) != len(joint_names):
            return self._abort(goal_handle, result,
                f'Expected {len(joint_names)} joints, got {len(joint_values)}')

        robot = self.robots[req.robot_name]

        # Compute FK to get target pose
        feedback.status = 'Computing FK for target...'
        goal_handle.publish_feedback(feedback)

        target_pose = await self._compute_fk(robot, joint_names, joint_values)
        if target_pose is None:
            return self._abort(goal_handle, result, 'FK computation failed')

        self.get_logger().info(
            f'MoveLJoints {req.robot_name}: ({target_pose.position.x:.3f}, '
            f'{target_pose.position.y:.3f}, {target_pose.position.z:.3f})'
        )

        pose = PoseStamped()
        pose.header.frame_id = self.planning['default_frame']
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose = target_pose

        # Try Pilz LIN first
        feedback.status = 'Planning LIN motion (Pilz)...'
        goal_handle.publish_feedback(feedback)

        plan_req = self._build_pilz_pose_request(req.robot_name, robot, pose, 'LIN')
        plan_resp = await self._plan_client.call_async(plan_req)
        error_code = plan_resp.motion_plan_response.error_code.val

        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('LIN planning succeeded (Pilz)')
            return await self._execute_trajectory(
                goal_handle, result, feedback,
                plan_resp.motion_plan_response.trajectory
            )

        # Fall back to computeCartesianPath
        error_name = self._get_error_name(error_code)
        self.get_logger().warn(f'Pilz LIN failed: {error_name}, trying Cartesian path...')

        feedback.status = 'Planning Cartesian path...'
        goal_handle.publish_feedback(feedback)

        trajectory = await self._plan_cartesian_path(
            req.robot_name, robot, target_pose, self.planning['default_frame']
        )
        if trajectory is None:
            return self._abort(goal_handle, result, 'Cartesian path planning failed')

        self.get_logger().info('Cartesian path planning succeeded')
        return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

    # === Helper Planning Methods ===

    async def _plan_cartesian_path(self, robot_name, robot, target_pose, frame):
        """Plan Cartesian path using computeCartesianPath."""
        cart_req = GetCartesianPath.Request()
        cart_req.header.frame_id = frame
        cart_req.header.stamp = self.get_clock().now().to_msg()
        cart_req.group_name = robot['group']
        cart_req.link_name = robot['ee_link']
        cart_req.waypoints = [target_pose]
        cart_req.max_step = self.cartesian['max_step']
        cart_req.jump_threshold = 0.0
        cart_req.avoid_collisions = True
        cart_req.start_state = self._get_current_robot_state(robot_name)

        cart_resp = await self._cartesian_client.call_async(cart_req)

        fraction = cart_resp.fraction
        error_code = cart_resp.error_code.val

        self.get_logger().info(f'Cartesian path: fraction={fraction:.2%}')

        if error_code == MoveItErrorCodes.SUCCESS and fraction >= self.cartesian['min_fraction']:
            return self._scale_trajectory(cart_resp.solution, 'lin')

        return None

    def _scale_trajectory(self, trajectory, motion_type):
        """Scale trajectory velocities."""
        vel_scale = self.vel_scale.get(motion_type, 0.5)

        for point in trajectory.joint_trajectory.points:
            original_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            scaled_time = original_time / vel_scale
            point.time_from_start.sec = int(scaled_time)
            point.time_from_start.nanosec = int((scaled_time % 1) * 1e9)
            point.velocities = [v * vel_scale for v in point.velocities]
            if point.accelerations:
                point.accelerations = [a * vel_scale * vel_scale for a in point.accelerations]

        return trajectory

    async def _compute_fk(self, robot, joint_names, joint_values):
        """Compute forward kinematics."""
        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = self.planning['default_frame']
        fk_request.header.stamp = self.get_clock().now().to_msg()
        fk_request.fk_link_names = [robot['ee_link']]
        fk_request.robot_state.joint_state.name = joint_names
        fk_request.robot_state.joint_state.position = joint_values

        try:
            fk_response = await self._fk_client.call_async(fk_request)
            if fk_response.error_code.val == MoveItErrorCodes.SUCCESS:
                return fk_response.pose_stamped[0].pose
            else:
                self.get_logger().error(f'FK failed: {fk_response.error_code.val}')
                return None
        except Exception as e:
            self.get_logger().error(f'FK service call failed: {e}')
            return None

    # === Trajectory Execution ===

    async def _execute_trajectory(self, goal_handle, result, feedback, trajectory):
        """Execute a planned trajectory."""
        feedback.status = 'Executing trajectory...'
        goal_handle.publish_feedback(feedback)

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = trajectory

        exec_handle = await self._exec_client.send_goal_async(exec_goal)
        if not exec_handle.accepted:
            return self._abort(goal_handle, result, 'Execution rejected')

        exec_result = await exec_handle.get_result_async()

        if exec_result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            result.success = True
            result.message = 'Done'
            goal_handle.succeed()
        else:
            result.success = False
            error_name = self._get_error_name(exec_result.result.error_code.val)
            result.message = f'Execution failed: {error_name}'
            goal_handle.abort()

        return result

    # === Request Building ===

    def _build_pilz_pose_request(self, robot_name, robot, pose, planner_id):
        """Build Pilz request (PTP or LIN) for pose target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = robot['group']
        mp.pipeline_id = 'pilz_industrial_motion_planner'
        mp.planner_id = planner_id
        mp.num_planning_attempts = 1
        mp.allowed_planning_time = self.planning['allowed_time']

        vel_type = 'ptp' if planner_id == 'PTP' else 'lin'
        mp.max_velocity_scaling_factor = self.vel_scale[vel_type]
        mp.max_acceleration_scaling_factor = self.acc_scale[vel_type]

        mp.start_state = self._get_current_robot_state(robot_name)

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

    def _build_pilz_joints_request(self, robot_name, robot, joint_names, joint_values, planner_id):
        """Build Pilz request (PTP or LIN) for joint target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = robot['group']
        mp.pipeline_id = 'pilz_industrial_motion_planner'
        mp.planner_id = planner_id
        mp.num_planning_attempts = 1
        mp.allowed_planning_time = self.planning['allowed_time']

        vel_type = 'ptp' if planner_id == 'PTP' else 'lin'
        mp.max_velocity_scaling_factor = self.vel_scale[vel_type]
        mp.max_acceleration_scaling_factor = self.acc_scale[vel_type]

        mp.start_state = self._get_current_robot_state(robot_name)

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

    def _build_ompl_pose_request(self, robot_name, robot, pose):
        """Build OMPL request for pose target (obstacle avoidance)."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = robot['group']
        mp.pipeline_id = 'ompl'
        mp.planner_id = self.ompl['planner_id']
        mp.num_planning_attempts = self.ompl['num_attempts']
        mp.allowed_planning_time = self.ompl['allowed_time']
        mp.max_velocity_scaling_factor = self.vel_scale['ptp']
        mp.max_acceleration_scaling_factor = self.acc_scale['ptp']

        mp.start_state = self._get_current_robot_state(robot_name)

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

    def _build_ompl_joints_request(self, robot_name, robot, joint_names, joint_values):
        """Build OMPL request for joint target (obstacle avoidance)."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = robot['group']
        mp.pipeline_id = 'ompl'
        mp.planner_id = self.ompl['planner_id']
        mp.num_planning_attempts = self.ompl['num_attempts']
        mp.allowed_planning_time = self.ompl['allowed_time']
        mp.max_velocity_scaling_factor = self.vel_scale['ptp']
        mp.max_acceleration_scaling_factor = self.acc_scale['ptp']

        mp.start_state = self._get_current_robot_state(robot_name)

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

    def _get_error_name(self, error_code):
        """Convert MoveIt error code to name."""
        error_names = {
            MoveItErrorCodes.SUCCESS: 'SUCCESS',
            MoveItErrorCodes.FAILURE: 'FAILURE',
            MoveItErrorCodes.PLANNING_FAILED: 'PLANNING_FAILED',
            MoveItErrorCodes.INVALID_MOTION_PLAN: 'INVALID_MOTION_PLAN',
            MoveItErrorCodes.CONTROL_FAILED: 'CONTROL_FAILED',
            MoveItErrorCodes.TIMED_OUT: 'TIMED_OUT',
            MoveItErrorCodes.START_STATE_IN_COLLISION: 'START_STATE_IN_COLLISION',
            MoveItErrorCodes.GOAL_IN_COLLISION: 'GOAL_IN_COLLISION',
            MoveItErrorCodes.NO_IK_SOLUTION: 'NO_IK_SOLUTION',
            MoveItErrorCodes.INVALID_GROUP_NAME: 'INVALID_GROUP_NAME',
        }
        return error_names.get(error_code, f'UNKNOWN_{error_code}')


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
