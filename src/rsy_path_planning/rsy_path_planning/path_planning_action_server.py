#!/usr/bin/env python3
"""
Path Planning Action Server
- MoveJ/MoveJJoints: OMPL planner for joint-space motions
- MoveL/MoveLJoints: Cartesian path planning for linear motions

Features:
- Configurable retry logic for robust IK/planning
- Collision-checked Cartesian paths via computeCartesianPath
"""

import time
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetMotionPlan, GetCartesianPath, GetPositionFK
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, JointConstraint, MoveItErrorCodes, RobotState
)
from moveit_msgs.action import ExecuteTrajectory
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
import tf2_ros
from tf2_ros import TransformException

from rsy_path_planning.action import MoveJ, MoveL, MoveJJoints, MoveLJoints


class PathPlanningActionServer(Node):

    # MoveIt error codes that indicate IK/planning failures worth retrying
    RETRYABLE_ERRORS = {
        MoveItErrorCodes.FAILURE,
        MoveItErrorCodes.PLANNING_FAILED,
        MoveItErrorCodes.INVALID_MOTION_PLAN,
        MoveItErrorCodes.NO_IK_SOLUTION,
        MoveItErrorCodes.GOAL_IN_COLLISION,
        MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS,
        MoveItErrorCodes.START_STATE_IN_COLLISION,
        MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS,
        MoveItErrorCodes.TIMED_OUT,
    }

    def __init__(self):
        super().__init__('path_planning_action_server')

        # Declare and load parameters
        self._declare_parameters()
        self._load_config()

        cb = ReentrantCallbackGroup()

        # Service clients
        self._plan_client = self.create_client(
            GetMotionPlan, '/plan_kinematic_path', callback_group=cb
        )
        self._cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=cb
        )
        self._fk_client = self.create_client(
            GetPositionFK, '/compute_fk', callback_group=cb
        )

        # Action client for trajectory execution
        self._exec_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=cb
        )

        # Wait for services
        self.get_logger().info('Waiting for planning services...')
        self._plan_client.wait_for_service(timeout_sec=30.0)
        self._cartesian_client.wait_for_service(timeout_sec=30.0)
        self._fk_client.wait_for_service(timeout_sec=30.0)
        self._exec_client.wait_for_server(timeout_sec=30.0)

        # TF buffer for getting current pose
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

        self.get_logger().info('Path Planning Server ready')
        self.get_logger().info(f'  MoveJ/MoveJJoints: OMPL ({self.planning["planner_id"]})')
        self.get_logger().info(f'  MoveL/MoveLJoints: Cartesian path planning')
        self.get_logger().info(f'  Retry config: max_retries={self.retry["max_retries"]}')

    def _declare_parameters(self):
        """Declare all parameters with defaults."""
        # Robot config
        self.declare_parameter('robots.robot1.planning_group', 'robot1_ur_manipulator')
        self.declare_parameter('robots.robot1.end_effector_link', 'robot1_tcp')
        self.declare_parameter('robots.robot2.planning_group', 'robot2_ur_manipulator')
        self.declare_parameter('robots.robot2.end_effector_link', 'robot2_tcp')

        # OMPL Planning parameters
        self.declare_parameter('planning.num_attempts', 10)
        self.declare_parameter('planning.allowed_time', 5.0)
        self.declare_parameter('planning.default_frame', 'world')
        self.declare_parameter('planning.planner_id', 'RRTConnect')

        # Retry configuration
        self.declare_parameter('planning.retry.max_retries', 3)
        self.declare_parameter('planning.retry.delay_between_retries', 0.1)

        # Cartesian path settings
        self.declare_parameter('cartesian.max_step', 0.01)
        self.declare_parameter('cartesian.jump_threshold', 0.0)
        self.declare_parameter('cartesian.avoid_collisions', True)
        self.declare_parameter('cartesian.min_fraction', 0.95)

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
            'planner_id': self.get_parameter('planning.planner_id').value,
        }

        self.retry = {
            'max_retries': self.get_parameter('planning.retry.max_retries').value,
            'delay_between_retries': self.get_parameter('planning.retry.delay_between_retries').value,
        }

        self.cartesian = {
            'max_step': self.get_parameter('cartesian.max_step').value,
            'jump_threshold': self.get_parameter('cartesian.jump_threshold').value,
            'avoid_collisions': self.get_parameter('cartesian.avoid_collisions').value,
            'min_fraction': self.get_parameter('cartesian.min_fraction').value,
        }

        self.vel_scale = {
            'ptp': self.get_parameter('velocity_scaling.ptp').value,
            'linear': self.get_parameter('velocity_scaling.linear').value,
        }

        self.acc_scale = {
            'ptp': self.get_parameter('acceleration_scaling.ptp').value,
            'linear': self.get_parameter('acceleration_scaling.linear').value,
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

    def _get_current_robot_state(self, robot_name):
        """Get current robot state for a specific robot."""
        robot_state = RobotState()
        joint_names = self.joint_names[robot_name]
        positions = []

        for name in joint_names:
            if name in self._current_joint_states:
                positions.append(self._current_joint_states[name])
            else:
                self.get_logger().warn(f'Joint {name} not found in current state')
                positions.append(0.0)

        robot_state.joint_state.name = joint_names
        robot_state.joint_state.position = positions
        return robot_state

    # === Action Callbacks ===

    async def _exec_movej(self, goal_handle):
        """MoveJ: Joint-space motion to Cartesian target using OMPL."""
        return await self._execute_ompl_pose(goal_handle, MoveJ)

    async def _exec_movel(self, goal_handle):
        """MoveL: Cartesian linear motion using computeCartesianPath."""
        return await self._execute_cartesian_pose(goal_handle, MoveL)

    async def _exec_movej_joints(self, goal_handle):
        """MoveJJoints: Joint-space motion to joint target using OMPL."""
        return await self._execute_ompl_joints(goal_handle, MoveJJoints)

    async def _exec_movel_joints(self, goal_handle):
        """MoveLJoints: Cartesian linear motion to joint target."""
        return await self._execute_cartesian_joints(goal_handle, MoveLJoints)

    # === OMPL Planning (MoveJ) ===

    async def _execute_ompl_pose(self, goal_handle, action_type):
        """Execute joint-space motion to Cartesian pose using OMPL."""
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
            f'MoveJ {req.robot_name}: ({req.x:.3f}, {req.y:.3f}, {req.z:.3f}) via OMPL'
        )

        trajectory = await self._plan_ompl_with_retry(
            goal_handle, feedback, robot, pose, 'pose'
        )

        if trajectory is None:
            return self._abort(goal_handle, result,
                f'OMPL planning failed after {self.retry["max_retries"]} retries')

        return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

    async def _execute_ompl_joints(self, goal_handle, action_type):
        """Execute joint-space motion to joint target using OMPL."""
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

        self.get_logger().info(
            f'MoveJJoints {req.robot_name}: {[f"{v:.2f}" for v in joint_values]} via OMPL'
        )

        trajectory = await self._plan_ompl_with_retry(
            goal_handle, feedback, robot, (joint_names, joint_values), 'joints'
        )

        if trajectory is None:
            return self._abort(goal_handle, result,
                f'OMPL planning failed after {self.retry["max_retries"]} retries')

        return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

    async def _plan_ompl_with_retry(self, goal_handle, feedback, robot, target, target_type):
        """Plan with OMPL with retry logic - progressively increase planning resources."""
        # Different strategies: increase attempts and time on each retry
        strategies = [
            {'num_attempts': self.planning['num_attempts'], 'allowed_time': self.planning['allowed_time'], 'planner': 'RRTstar'},
            {'num_attempts': self.planning['num_attempts'] * 2, 'allowed_time': self.planning['allowed_time'] * 1.5, 'planner': 'RRTstar'},
            {'num_attempts': self.planning['num_attempts'] * 3, 'allowed_time': self.planning['allowed_time'] * 2.0, 'planner': 'RRTstar'},
            {'num_attempts': self.planning['num_attempts'] * 5, 'allowed_time': self.planning['allowed_time'] * 3.0, 'planner': 'BiTRRT'},
            {'num_attempts': self.planning['num_attempts'] * 5, 'allowed_time': self.planning['allowed_time'] * 3.0, 'planner': 'PRM'},
        ]
        
        for attempt in range(1, self.retry['max_retries'] + 1):
            strategy = strategies[min(attempt - 1, len(strategies) - 1)]
            
            feedback.status = f'OMPL Planning (attempt {attempt}/{self.retry["max_retries"]})...'
            goal_handle.publish_feedback(feedback)

            if target_type == 'pose':
                plan_req = self._build_ompl_pose_request(robot, target, strategy)
            else:
                joint_names, joint_values = target
                plan_req = self._build_ompl_joints_request(robot, joint_names, joint_values, strategy)

            self.get_logger().info(
                f'OMPL attempt {attempt}: planner={strategy["planner"]}, '
                f'attempts={strategy["num_attempts"]}, time={strategy["allowed_time"]:.1f}s'
            )

            plan_resp = await self._plan_client.call_async(plan_req)
            error_code = plan_resp.motion_plan_response.error_code.val

            if error_code == MoveItErrorCodes.SUCCESS:
                self.get_logger().info(f'OMPL planning succeeded on attempt {attempt}')
                return plan_resp.motion_plan_response.trajectory

            error_name = self._get_error_name(error_code)
            self.get_logger().warn(f'OMPL attempt {attempt} failed: {error_name}')

            if error_code not in self.RETRYABLE_ERRORS:
                break

            if attempt < self.retry['max_retries']:
                time.sleep(self.retry['delay_between_retries'])

        return None

    # === Cartesian Path Planning (MoveL) ===

    async def _execute_cartesian_pose(self, goal_handle, action_type):
        """Execute Cartesian linear motion to pose target."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        robot = self.robots[req.robot_name]
        frame = req.frame_id or self.planning['default_frame']

        target_pose = Pose()
        target_pose.position.x = req.x
        target_pose.position.y = req.y
        target_pose.position.z = req.z
        target_pose.orientation.x = req.qx
        target_pose.orientation.y = req.qy
        target_pose.orientation.z = req.qz
        target_pose.orientation.w = req.qw

        self.get_logger().info(
            f'MoveL {req.robot_name}: ({req.x:.3f}, {req.y:.3f}, {req.z:.3f}) via Cartesian path'
        )

        trajectory = await self._plan_cartesian_with_retry(
            goal_handle, feedback, req.robot_name, robot, [target_pose], frame
        )

        if trajectory is None:
            return self._abort(goal_handle, result,
                f'Cartesian planning failed after {self.retry["max_retries"]} retries')

        return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

    async def _execute_cartesian_joints(self, goal_handle, action_type):
        """Execute Cartesian linear motion to joint target."""
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

        # Compute FK to get target Cartesian pose
        feedback.status = 'Computing FK for target joints...'
        goal_handle.publish_feedback(feedback)

        target_pose = await self._compute_fk(robot, joint_names, joint_values)
        if target_pose is None:
            return self._abort(goal_handle, result, 'FK computation failed')

        self.get_logger().info(
            f'MoveLJoints {req.robot_name}: ({target_pose.position.x:.3f}, '
            f'{target_pose.position.y:.3f}, {target_pose.position.z:.3f}) via Cartesian path'
        )

        trajectory = await self._plan_cartesian_with_retry(
            goal_handle, feedback, req.robot_name, robot,
            [target_pose], self.planning['default_frame']
        )

        if trajectory is None:
            return self._abort(goal_handle, result,
                f'Cartesian planning failed after {self.retry["max_retries"]} retries')

        return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

    async def _plan_cartesian_with_retry(self, goal_handle, feedback, robot_name, robot, waypoints, frame):
        """Plan Cartesian path with retry logic."""
        for attempt in range(1, self.retry['max_retries'] + 1):
            feedback.status = f'Cartesian planning (attempt {attempt}/{self.retry["max_retries"]})...'
            goal_handle.publish_feedback(feedback)

            # Build Cartesian path request
            cart_req = GetCartesianPath.Request()
            cart_req.header.frame_id = frame
            cart_req.header.stamp = self.get_clock().now().to_msg()
            cart_req.group_name = robot['group']
            cart_req.link_name = robot['ee_link']
            cart_req.waypoints = waypoints
            cart_req.max_step = self.cartesian['max_step']
            cart_req.jump_threshold = self.cartesian['jump_threshold']
            cart_req.avoid_collisions = self.cartesian['avoid_collisions']
            cart_req.start_state = self._get_current_robot_state(robot_name)

            cart_resp = await self._cartesian_client.call_async(cart_req)

            fraction = cart_resp.fraction
            error_code = cart_resp.error_code.val

            self.get_logger().info(
                f'Cartesian attempt {attempt}: fraction={fraction:.2%}, error={error_code}'
            )

            if error_code == MoveItErrorCodes.SUCCESS and fraction >= self.cartesian['min_fraction']:
                self.get_logger().info(
                    f'Cartesian planning succeeded: {fraction:.1%} of path achieved'
                )
                # Apply velocity scaling to the trajectory
                return self._scale_trajectory(cart_resp.solution, 'linear')

            if fraction < self.cartesian['min_fraction']:
                self.get_logger().warn(
                    f'Cartesian attempt {attempt}: only {fraction:.1%} achieved '
                    f'(need {self.cartesian["min_fraction"]:.0%})'
                )

            if error_code != MoveItErrorCodes.SUCCESS:
                error_name = self._get_error_name(error_code)
                self.get_logger().warn(f'Cartesian attempt {attempt} error: {error_name}')

            if attempt < self.retry['max_retries']:
                time.sleep(self.retry['delay_between_retries'])

        return None

    def _scale_trajectory(self, trajectory, motion_type):
        """Scale trajectory velocities and accelerations."""
        vel_scale = self.vel_scale.get(motion_type, 0.5)
        acc_scale = self.acc_scale.get(motion_type, 0.5)

        # Scale time from start for each point
        for point in trajectory.joint_trajectory.points:
            # Scale time
            original_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            scaled_time = original_time / vel_scale
            point.time_from_start.sec = int(scaled_time)
            point.time_from_start.nanosec = int((scaled_time % 1) * 1e9)

            # Scale velocities
            point.velocities = [v * vel_scale for v in point.velocities]

            # Scale accelerations if present
            if point.accelerations:
                point.accelerations = [a * acc_scale for a in point.accelerations]

        return trajectory

    async def _compute_fk(self, robot, joint_names, joint_values):
        """Compute forward kinematics to get Cartesian pose from joint values."""
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

    def _build_ompl_pose_request(self, robot, pose, strategy=None):
        """Build OMPL motion plan request for pose target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        if strategy is None:
            strategy = {
                'planner': self.planning['planner_id'],
                'num_attempts': self.planning['num_attempts'],
                'allowed_time': self.planning['allowed_time']
            }

        mp.group_name = robot['group']
        mp.pipeline_id = 'ompl'
        mp.planner_id = strategy['planner']
        mp.num_planning_attempts = strategy['num_attempts']
        mp.allowed_planning_time = strategy['allowed_time']
        mp.max_velocity_scaling_factor = self.vel_scale['ptp']
        mp.max_acceleration_scaling_factor = self.acc_scale['ptp']
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

    def _build_ompl_joints_request(self, robot, joint_names, joint_values, strategy=None):
        """Build OMPL motion plan request for joint target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        if strategy is None:
            strategy = {
                'planner': self.planning['planner_id'],
                'num_attempts': self.planning['num_attempts'],
                'allowed_time': self.planning['allowed_time']
            }

        mp.group_name = robot['group']
        mp.pipeline_id = 'ompl'
        mp.planner_id = strategy['planner']
        mp.num_planning_attempts = strategy['num_attempts']
        mp.allowed_planning_time = strategy['allowed_time']
        mp.max_velocity_scaling_factor = self.vel_scale['ptp']
        mp.max_acceleration_scaling_factor = self.acc_scale['ptp']
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

    def _get_error_name(self, error_code):
        """Convert MoveIt error code to human-readable name."""
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
