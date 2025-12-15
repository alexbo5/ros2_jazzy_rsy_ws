#!/usr/bin/env python3
"""
Path Planning Action Server
Provides MoveJ (PTP) and MoveL (LIN) motion actions using Pilz Industrial Motion Planner.
Features:
- Configurable retry logic for robust IK/planning
- Fallback planner (OMPL) when primary planner fails
- Optional MoveIt Servo integration for fine corrections
All planning parameters are loaded from config/planning.yaml.
"""

import asyncio
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, JointConstraint, MoveItErrorCodes
)
from moveit_msgs.action import ExecuteTrajectory
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
import tf2_ros

from rsy_path_planning.action import MoveJ, MoveL, MoveJJoints, MoveLJoints


class PathPlanningActionServer(Node):

    # MoveIt error codes that indicate IK/planning failures worth retrying
    RETRYABLE_ERRORS = {
        MoveItErrorCodes.PLANNING_FAILED,
        MoveItErrorCodes.INVALID_MOTION_PLAN,
        MoveItErrorCodes.NO_IK_SOLUTION,
        MoveItErrorCodes.GOAL_IN_COLLISION,
        MoveItErrorCodes.TIMED_OUT,
    }

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

        # Servo publishers and services (per robot)
        self._servo_twist_pubs = {}
        self._servo_start_clients = {}
        self._servo_stop_clients = {}

        if self.servo['enable_fine_correction']:
            self._setup_servo_clients(cb)

        # TF buffer for pose checking
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

        self.get_logger().info('Path Planning Server ready: /move_j, /move_l, /move_j_joints, /move_l_joints')
        self.get_logger().info(f'Retry config: max_retries={self.retry["max_retries"]}, '
                               f'fallback_planner={self.retry["use_fallback_planner"]}')

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

        # Retry configuration
        self.declare_parameter('planning.retry.max_retries', 3)
        self.declare_parameter('planning.retry.delay_between_retries', 0.1)
        self.declare_parameter('planning.retry.use_fallback_planner', True)
        self.declare_parameter('planning.retry.fallback_pipeline', 'ompl')
        self.declare_parameter('planning.retry.fallback_planner_id', 'RRTConnect')

        # Servo configuration
        self.declare_parameter('planning.servo.enable_fine_correction', False)
        self.declare_parameter('planning.servo.position_threshold', 0.05)
        self.declare_parameter('planning.servo.max_correction_time', 5.0)
        self.declare_parameter('planning.servo.correction_linear_velocity', 0.05)
        self.declare_parameter('planning.servo.correction_angular_velocity', 0.1)

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

        self.retry = {
            'max_retries': self.get_parameter('planning.retry.max_retries').value,
            'delay_between_retries': self.get_parameter('planning.retry.delay_between_retries').value,
            'use_fallback_planner': self.get_parameter('planning.retry.use_fallback_planner').value,
            'fallback_pipeline': self.get_parameter('planning.retry.fallback_pipeline').value,
            'fallback_planner_id': self.get_parameter('planning.retry.fallback_planner_id').value,
        }

        self.servo = {
            'enable_fine_correction': self.get_parameter('planning.servo.enable_fine_correction').value,
            'position_threshold': self.get_parameter('planning.servo.position_threshold').value,
            'max_correction_time': self.get_parameter('planning.servo.max_correction_time').value,
            'correction_linear_velocity': self.get_parameter('planning.servo.correction_linear_velocity').value,
            'correction_angular_velocity': self.get_parameter('planning.servo.correction_angular_velocity').value,
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

    def _setup_servo_clients(self, cb):
        """Setup Servo publishers and service clients for each robot."""
        for robot_name in self.robots:
            # Twist command publisher for Servo
            self._servo_twist_pubs[robot_name] = self.create_publisher(
                TwistStamped,
                f'/{robot_name}_servo/delta_twist_cmds',
                10
            )

            # Servo start/stop services
            self._servo_start_clients[robot_name] = self.create_client(
                SetBool,
                f'/{robot_name}_servo/start_servo',
                callback_group=cb
            )
            self._servo_stop_clients[robot_name] = self.create_client(
                SetBool,
                f'/{robot_name}_servo/stop_servo',
                callback_group=cb
            )

        self.get_logger().info('Servo clients initialized for fine corrections')

    def _joint_state_callback(self, msg):
        """Store current joint states."""
        for i, name in enumerate(msg.name):
            self._current_joint_states[name] = msg.position[i]

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
        """Execute motion to Cartesian pose target with retry logic."""
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

        # Plan with retry logic
        trajectory = await self._plan_with_retry(
            goal_handle, feedback, robot, pose, planner_id, 'pose'
        )

        if trajectory is None:
            return self._abort(
                goal_handle, result,
                f'Planning failed after {self.retry["max_retries"]} retries'
            )

        # Execute
        exec_result = await self._execute_trajectory(
            goal_handle, result, feedback, trajectory
        )

        # Optional: Fine correction with Servo
        if (exec_result.success and
            self.servo['enable_fine_correction'] and
            action_type == MoveL):  # Only for linear moves
            await self._servo_fine_correction(
                req.robot_name, pose, feedback, goal_handle
            )

        return exec_result

    async def _execute_joints(self, goal_handle, planner_id, action_type):
        """Execute motion to joint target with retry logic."""
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

        # Plan with retry logic
        trajectory = await self._plan_with_retry(
            goal_handle, feedback, robot, (joint_names, joint_values), planner_id, 'joints'
        )

        if trajectory is None:
            return self._abort(
                goal_handle, result,
                f'Planning failed after {self.retry["max_retries"]} retries'
            )

        # Execute
        return await self._execute_trajectory(
            goal_handle, result, feedback, trajectory
        )

    async def _plan_with_retry(self, goal_handle, feedback, robot, target, planner_id, target_type):
        """
        Plan motion with configurable retry logic.

        Retry strategy:
        1. Try primary planner (Pilz) up to max_retries times
        2. If all retries fail and fallback is enabled, try OMPL planner

        Args:
            goal_handle: Action goal handle
            feedback: Feedback message
            robot: Robot configuration dict
            target: PoseStamped (for pose) or (joint_names, joint_values) tuple (for joints)
            planner_id: 'PTP' or 'LIN'
            target_type: 'pose' or 'joints'

        Returns:
            Trajectory if successful, None otherwise
        """
        last_error_code = None

        # Primary planner attempts
        for attempt in range(1, self.retry['max_retries'] + 1):
            feedback.status = f'Planning (attempt {attempt}/{self.retry["max_retries"]})...'
            goal_handle.publish_feedback(feedback)

            # Build request based on target type
            if target_type == 'pose':
                plan_req = self._build_pose_request(
                    robot, target, planner_id,
                    pipeline='pilz_industrial_motion_planner'
                )
            else:
                joint_names, joint_values = target
                plan_req = self._build_joints_request(
                    robot, joint_names, joint_values, planner_id,
                    pipeline='pilz_industrial_motion_planner'
                )

            plan_resp = await self._plan_client.call_async(plan_req)
            error_code = plan_resp.motion_plan_response.error_code.val
            last_error_code = error_code

            if error_code == MoveItErrorCodes.SUCCESS:
                self.get_logger().info(f'Planning succeeded on attempt {attempt}')
                return plan_resp.motion_plan_response.trajectory

            # Log the failure
            error_name = self._get_error_name(error_code)
            self.get_logger().warn(
                f'Planning attempt {attempt} failed: {error_name} (code {error_code})'
            )

            # Only retry for retryable errors
            if error_code not in self.RETRYABLE_ERRORS:
                self.get_logger().error(f'Non-retryable error, aborting: {error_name}')
                break

            # Delay between retries
            if attempt < self.retry['max_retries']:
                await asyncio.sleep(self.retry['delay_between_retries'])

        # Fallback planner attempt
        if self.retry['use_fallback_planner']:
            feedback.status = f'Trying fallback planner ({self.retry["fallback_planner_id"]})...'
            goal_handle.publish_feedback(feedback)

            self.get_logger().info(
                f'Primary planner failed, trying fallback: {self.retry["fallback_pipeline"]}/{self.retry["fallback_planner_id"]}'
            )

            # Build request with fallback planner
            if target_type == 'pose':
                plan_req = self._build_pose_request(
                    robot, target, self.retry['fallback_planner_id'],
                    pipeline=self.retry['fallback_pipeline']
                )
            else:
                joint_names, joint_values = target
                plan_req = self._build_joints_request(
                    robot, joint_names, joint_values, self.retry['fallback_planner_id'],
                    pipeline=self.retry['fallback_pipeline']
                )

            plan_resp = await self._plan_client.call_async(plan_req)
            error_code = plan_resp.motion_plan_response.error_code.val

            if error_code == MoveItErrorCodes.SUCCESS:
                self.get_logger().info('Fallback planner succeeded')
                return plan_resp.motion_plan_response.trajectory
            else:
                error_name = self._get_error_name(error_code)
                self.get_logger().error(f'Fallback planner also failed: {error_name}')

        return None

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

    async def _servo_fine_correction(self, robot_name, target_pose, feedback, goal_handle):
        """
        Use MoveIt Servo to make fine corrections to reach exact target pose.

        This is useful when the trajectory execution ends slightly off target.
        Servo provides real-time closed-loop control for fine adjustments.
        """
        if robot_name not in self._servo_twist_pubs:
            self.get_logger().warn(f'Servo not configured for {robot_name}')
            return

        feedback.status = 'Fine correction with Servo...'
        goal_handle.publish_feedback(feedback)

        robot = self.robots[robot_name]
        ee_link = robot['ee_link']

        # Start servo
        if robot_name in self._servo_start_clients:
            if self._servo_start_clients[robot_name].service_is_ready():
                start_req = SetBool.Request()
                start_req.data = True
                await self._servo_start_clients[robot_name].call_async(start_req)
                await asyncio.sleep(0.1)  # Give servo time to start

        start_time = self.get_clock().now()
        max_duration = rclpy.duration.Duration(seconds=self.servo['max_correction_time'])

        try:
            while (self.get_clock().now() - start_time) < max_duration:
                # Get current pose
                try:
                    transform = self._tf_buffer.lookup_transform(
                        target_pose.header.frame_id,
                        ee_link,
                        rclpy.time.Time()
                    )
                except tf2_ros.LookupException:
                    self.get_logger().warn('TF lookup failed during Servo correction')
                    break

                # Calculate position error
                dx = target_pose.pose.position.x - transform.transform.translation.x
                dy = target_pose.pose.position.y - transform.transform.translation.y
                dz = target_pose.pose.position.z - transform.transform.translation.z

                pos_error = math.sqrt(dx*dx + dy*dy + dz*dz)

                # Check if within tolerance
                if pos_error < self.tolerances['position']:
                    self.get_logger().info('Servo fine correction complete')
                    break

                # Calculate velocity commands (proportional control)
                vel_scale = min(pos_error / self.servo['position_threshold'], 1.0)

                twist = TwistStamped()
                twist.header.stamp = self.get_clock().now().to_msg()
                twist.header.frame_id = ee_link

                # Normalize and scale velocity
                if pos_error > 0.001:
                    twist.twist.linear.x = (dx / pos_error) * self.servo['correction_linear_velocity'] * vel_scale
                    twist.twist.linear.y = (dy / pos_error) * self.servo['correction_linear_velocity'] * vel_scale
                    twist.twist.linear.z = (dz / pos_error) * self.servo['correction_linear_velocity'] * vel_scale

                self._servo_twist_pubs[robot_name].publish(twist)
                await asyncio.sleep(0.01)  # 100 Hz

        finally:
            # Stop servo - send zero twist
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = ee_link
            self._servo_twist_pubs[robot_name].publish(twist)

            # Stop servo service
            if robot_name in self._servo_stop_clients:
                if self._servo_stop_clients[robot_name].service_is_ready():
                    stop_req = SetBool.Request()
                    stop_req.data = True
                    await self._servo_stop_clients[robot_name].call_async(stop_req)

    # === Request Building ===

    def _build_pose_request(self, robot, pose, planner_id, pipeline='pilz_industrial_motion_planner'):
        """Build motion plan request for pose target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = robot['group']
        mp.pipeline_id = pipeline
        mp.planner_id = planner_id
        mp.num_planning_attempts = self.planning['num_attempts']
        mp.allowed_planning_time = self.planning['allowed_time']
        mp.max_velocity_scaling_factor = self.vel_scale.get(planner_id, 0.5)
        mp.max_acceleration_scaling_factor = self.acc_scale.get(planner_id, 0.5)
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

    def _build_joints_request(self, robot, joint_names, joint_values, planner_id,
                               pipeline='pilz_industrial_motion_planner'):
        """Build motion plan request for joint target."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = robot['group']
        mp.pipeline_id = pipeline
        mp.planner_id = planner_id
        mp.num_planning_attempts = self.planning['num_attempts']
        mp.allowed_planning_time = self.planning['allowed_time']
        mp.max_velocity_scaling_factor = self.vel_scale.get(planner_id, 0.5)
        mp.max_acceleration_scaling_factor = self.acc_scale.get(planner_id, 0.5)
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
            MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
            MoveItErrorCodes.CONTROL_FAILED: 'CONTROL_FAILED',
            MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: 'UNABLE_TO_AQUIRE_SENSOR_DATA',
            MoveItErrorCodes.TIMED_OUT: 'TIMED_OUT',
            MoveItErrorCodes.PREEMPTED: 'PREEMPTED',
            MoveItErrorCodes.START_STATE_IN_COLLISION: 'START_STATE_IN_COLLISION',
            MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
            MoveItErrorCodes.GOAL_IN_COLLISION: 'GOAL_IN_COLLISION',
            MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
            MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: 'GOAL_CONSTRAINTS_VIOLATED',
            MoveItErrorCodes.INVALID_GROUP_NAME: 'INVALID_GROUP_NAME',
            MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: 'INVALID_GOAL_CONSTRAINTS',
            MoveItErrorCodes.INVALID_ROBOT_STATE: 'INVALID_ROBOT_STATE',
            MoveItErrorCodes.INVALID_LINK_NAME: 'INVALID_LINK_NAME',
            MoveItErrorCodes.INVALID_OBJECT_NAME: 'INVALID_OBJECT_NAME',
            MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: 'FRAME_TRANSFORM_FAILURE',
            MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE: 'COLLISION_CHECKING_UNAVAILABLE',
            MoveItErrorCodes.ROBOT_STATE_STALE: 'ROBOT_STATE_STALE',
            MoveItErrorCodes.SENSOR_INFO_STALE: 'SENSOR_INFO_STALE',
            MoveItErrorCodes.COMMUNICATION_FAILURE: 'COMMUNICATION_FAILURE',
            MoveItErrorCodes.NO_IK_SOLUTION: 'NO_IK_SOLUTION',
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
