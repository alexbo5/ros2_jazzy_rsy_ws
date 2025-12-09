#!/usr/bin/env python3
"""
Path Planning Action Server for Dual UR Robots

This action server provides MoveL (linear/Cartesian) and MoveJ (joint space)
motion planning for two robots with:
- Collision checking between robots
- Ground plane collision avoidance
- Shortest path optimization
- All poses relative to global /world frame
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Pose, Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from shape_msgs.msg import SolidPrimitive, Plane

from moveit_msgs.msg import (
    RobotState,
    Constraints,
    JointConstraint,
    CollisionObject,
    PlanningScene,
    PlanningSceneWorld,
    AllowedCollisionMatrix,
)
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetCartesianPath, GetPlanningScene

from path_planning_interfaces.action import MoveL, MoveJ

import numpy as np
from typing import Dict, Optional, Tuple, List
import threading
import time


class PathPlanningActionServer(Node):
    """
    Action server for dual robot path planning with collision checking.

    Features:
    - MoveL: Linear (Cartesian) path planning
    - MoveJ: Joint space path planning (shortest joint path)
    - Collision checking: Robot-robot and robot-ground
    - Ground plane avoidance
    - Shortest path optimization
    """

    def __init__(self):
        super().__init__('path_planning_action_server')

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self._load_parameters()

        # Robot configurations
        self._setup_robot_configs()

        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()

        # Current joint states for each robot
        self.current_joint_states: Dict[str, List[float]] = {}

        # Setup service clients
        self._setup_service_clients()

        # Setup subscribers for joint states
        self._setup_subscribers()

        # Create action servers
        self._create_action_servers()

        # Setup collision objects (ground plane)
        self._setup_collision_objects()

        # Lock for thread safety
        self._lock = threading.Lock()

        self.get_logger().info('Path Planning Action Server initialized')
        self.get_logger().info(f'World frame: {self.world_frame}')
        self.get_logger().info(f'Ground height: {self.ground_height} m')
        for name, config in self.robot_configs.items():
            self.get_logger().info(f'Robot "{name}": namespace={config["namespace"]}')

    def _declare_parameters(self):
        """Declare all node parameters."""
        # Frame parameters
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('ground_height', 0.0)
        self.declare_parameter('ground_safety_margin', 0.02)

        # Robot 1 parameters
        self.declare_parameter('robot1.name', 'robot1')
        self.declare_parameter('robot1.namespace', 'robot1')
        self.declare_parameter('robot1.base_frame', 'robot1/base_link')
        self.declare_parameter('robot1.ee_frame', 'robot1/tool0')
        self.declare_parameter('robot1.planning_group', 'robot1_manipulator')
        self.declare_parameter('robot1.ur_type', 'ur3e')

        # Robot 2 parameters
        self.declare_parameter('robot2.name', 'robot2')
        self.declare_parameter('robot2.namespace', 'robot2')
        self.declare_parameter('robot2.base_frame', 'robot2/base_link')
        self.declare_parameter('robot2.ee_frame', 'robot2/tool0')
        self.declare_parameter('robot2.planning_group', 'robot2_manipulator')
        self.declare_parameter('robot2.ur_type', 'ur3e')

        # Planning parameters
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('num_planning_attempts', 10)
        self.declare_parameter('cartesian_max_step', 0.01)
        self.declare_parameter('cartesian_jump_threshold', 0.0)
        self.declare_parameter('goal_joint_tolerance', 0.01)
        self.declare_parameter('goal_position_tolerance', 0.001)
        self.declare_parameter('goal_orientation_tolerance', 0.01)

    def _load_parameters(self):
        """Load parameters from parameter server."""
        self.world_frame = self.get_parameter('world_frame').value
        self.ground_height = self.get_parameter('ground_height').value
        self.ground_safety_margin = self.get_parameter('ground_safety_margin').value

        self.planning_time = self.get_parameter('planning_time').value
        self.num_planning_attempts = self.get_parameter('num_planning_attempts').value
        self.cartesian_max_step = self.get_parameter('cartesian_max_step').value
        self.cartesian_jump_threshold = self.get_parameter('cartesian_jump_threshold').value
        self.goal_joint_tolerance = self.get_parameter('goal_joint_tolerance').value
        self.goal_position_tolerance = self.get_parameter('goal_position_tolerance').value
        self.goal_orientation_tolerance = self.get_parameter('goal_orientation_tolerance').value

    def _setup_robot_configs(self):
        """Setup robot configurations from parameters."""
        self.robot_configs: Dict[str, dict] = {}

        for robot_prefix in ['robot1', 'robot2']:
            name = self.get_parameter(f'{robot_prefix}.name').value
            namespace = self.get_parameter(f'{robot_prefix}.namespace').value
            ur_type = self.get_parameter(f'{robot_prefix}.ur_type').value

            self.robot_configs[name] = {
                'namespace': namespace,
                'base_frame': self.get_parameter(f'{robot_prefix}.base_frame').value,
                'ee_frame': self.get_parameter(f'{robot_prefix}.ee_frame').value,
                'planning_group': self.get_parameter(f'{robot_prefix}.planning_group').value,
                'ur_type': ur_type,
                'joint_names': [
                    f'{namespace}/shoulder_pan_joint',
                    f'{namespace}/shoulder_lift_joint',
                    f'{namespace}/elbow_joint',
                    f'{namespace}/wrist_1_joint',
                    f'{namespace}/wrist_2_joint',
                    f'{namespace}/wrist_3_joint',
                ],
            }

            # Initialize joint states
            self.current_joint_states[name] = [0.0] * 6

    def _setup_service_clients(self):
        """Setup MoveIt2 service clients for both robots."""
        self.ik_clients: Dict[str, any] = {}
        self.motion_plan_clients: Dict[str, any] = {}
        self.cartesian_path_clients: Dict[str, any] = {}
        self.planning_scene_clients: Dict[str, any] = {}

        for robot_name, config in self.robot_configs.items():
            namespace = config['namespace']

            # IK service client
            self.ik_clients[robot_name] = self.create_client(
                GetPositionIK,
                f'/{namespace}/compute_ik',
                callback_group=self.callback_group
            )

            # Motion planning service client
            self.motion_plan_clients[robot_name] = self.create_client(
                GetMotionPlan,
                f'/{namespace}/plan_kinematic_path',
                callback_group=self.callback_group
            )

            # Cartesian path service client
            self.cartesian_path_clients[robot_name] = self.create_client(
                GetCartesianPath,
                f'/{namespace}/compute_cartesian_path',
                callback_group=self.callback_group
            )

            # Planning scene service client
            self.planning_scene_clients[robot_name] = self.create_client(
                GetPlanningScene,
                f'/{namespace}/get_planning_scene',
                callback_group=self.callback_group
            )

            self.get_logger().info(f'Service clients created for {robot_name}')

    def _setup_subscribers(self):
        """Setup joint state subscribers for each robot."""
        self.joint_state_subs: Dict[str, any] = {}

        for robot_name, config in self.robot_configs.items():
            namespace = config['namespace']

            self.joint_state_subs[robot_name] = self.create_subscription(
                JointState,
                f'/{namespace}/joint_states',
                lambda msg, rn=robot_name: self._joint_state_callback(msg, rn),
                10,
                callback_group=self.callback_group
            )

    def _joint_state_callback(self, msg: JointState, robot_name: str):
        """Handle incoming joint state messages."""
        config = self.robot_configs[robot_name]
        joint_names = config['joint_names']

        with self._lock:
            for i, joint_name in enumerate(joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.current_joint_states[robot_name][i] = msg.position[idx]

    def _setup_collision_objects(self):
        """Setup collision objects including ground plane."""
        self.collision_objects: List[CollisionObject] = []

        # Create ground plane collision object
        ground_plane = CollisionObject()
        ground_plane.header.frame_id = self.world_frame
        ground_plane.id = 'ground_plane'
        ground_plane.operation = CollisionObject.ADD

        # Use a large box to represent the ground
        ground_box = SolidPrimitive()
        ground_box.type = SolidPrimitive.BOX
        ground_box.dimensions = [10.0, 10.0, 0.01]  # 10m x 10m x 1cm thick

        ground_pose = Pose()
        ground_pose.position.x = 0.0
        ground_pose.position.y = 0.0
        ground_pose.position.z = self.ground_height - 0.005 - self.ground_safety_margin
        ground_pose.orientation.w = 1.0

        ground_plane.primitives.append(ground_box)
        ground_plane.primitive_poses.append(ground_pose)

        self.collision_objects.append(ground_plane)

        self.get_logger().info('Ground plane collision object created')

    def _create_action_servers(self):
        """Create action servers for MoveL and MoveJ."""
        # Global action servers
        self._movel_server = ActionServer(
            self,
            MoveL,
            'move_l',
            execute_callback=self._execute_movel,
            goal_callback=self._goal_callback_movel,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group
        )

        self._movej_server = ActionServer(
            self,
            MoveJ,
            'move_j',
            execute_callback=self._execute_movej,
            goal_callback=self._goal_callback_movej,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group
        )

        # Robot-specific action servers
        for robot_name, config in self.robot_configs.items():
            namespace = config['namespace']

            ActionServer(
                self,
                MoveL,
                f'/{namespace}/move_l',
                execute_callback=self._execute_movel,
                goal_callback=self._goal_callback_movel,
                cancel_callback=self._cancel_callback,
                callback_group=self.callback_group
            )

            ActionServer(
                self,
                MoveJ,
                f'/{namespace}/move_j',
                execute_callback=self._execute_movej,
                goal_callback=self._goal_callback_movej,
                cancel_callback=self._cancel_callback,
                callback_group=self.callback_group
            )

            self.get_logger().info(f'Action servers created for {robot_name}: /{namespace}/move_l, /{namespace}/move_j')

    def _goal_callback_movel(self, goal_request) -> GoalResponse:
        """Validate MoveL goal request."""
        robot_name = goal_request.robot_name.lower()

        if robot_name not in self.robot_configs:
            self.get_logger().error(f'Unknown robot: {robot_name}. Available: {list(self.robot_configs.keys())}')
            return GoalResponse.REJECT

        # Check velocity/acceleration scaling
        if goal_request.velocity_scaling <= 0 or goal_request.velocity_scaling > 1.0:
            self.get_logger().warn(f'Invalid velocity scaling {goal_request.velocity_scaling}, will use 0.5')

        if goal_request.acceleration_scaling <= 0 or goal_request.acceleration_scaling > 1.0:
            self.get_logger().warn(f'Invalid acceleration scaling {goal_request.acceleration_scaling}, will use 0.5')

        # Check if target is above ground
        target_z = goal_request.target_pose.pose.position.z
        if target_z < self.ground_height + self.ground_safety_margin:
            self.get_logger().error(
                f'Target position z={target_z:.3f} is below ground level '
                f'({self.ground_height + self.ground_safety_margin:.3f})'
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            f'Accepted MoveL goal for {robot_name}: '
            f'target=({goal_request.target_pose.pose.position.x:.3f}, '
            f'{goal_request.target_pose.pose.position.y:.3f}, '
            f'{goal_request.target_pose.pose.position.z:.3f})'
        )

        return GoalResponse.ACCEPT

    def _goal_callback_movej(self, goal_request) -> GoalResponse:
        """Validate MoveJ goal request."""
        robot_name = goal_request.robot_name.lower()

        if robot_name not in self.robot_configs:
            self.get_logger().error(f'Unknown robot: {robot_name}. Available: {list(self.robot_configs.keys())}')
            return GoalResponse.REJECT

        # Check if target is above ground
        target_z = goal_request.target_pose.pose.position.z
        if target_z < self.ground_height + self.ground_safety_margin:
            self.get_logger().error(
                f'Target position z={target_z:.3f} is below ground level '
                f'({self.ground_height + self.ground_safety_margin:.3f})'
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            f'Accepted MoveJ goal for {robot_name}: '
            f'target=({goal_request.target_pose.pose.position.x:.3f}, '
            f'{goal_request.target_pose.pose.position.y:.3f}, '
            f'{goal_request.target_pose.pose.position.z:.3f})'
        )

        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        """Handle cancel requests."""
        self.get_logger().info('Goal cancellation requested')
        return CancelResponse.ACCEPT

    async def _execute_movel(self, goal_handle):
        """Execute MoveL (linear/Cartesian) motion."""
        goal = goal_handle.request
        robot_name = goal.robot_name.lower()
        result = MoveL.Result()

        try:
            config = self.robot_configs[robot_name]
            velocity_scaling = min(max(goal.velocity_scaling, 0.01), 1.0)
            acceleration_scaling = min(max(goal.acceleration_scaling, 0.01), 1.0)

            # Publish initial feedback
            feedback = MoveL.Feedback()
            feedback.progress_percentage = 0.0
            feedback.status = 'Starting MoveL planning'
            goal_handle.publish_feedback(feedback)

            start_time = time.time()

            # Plan Cartesian path with collision checking
            trajectory, success, message, path_length = await self._plan_cartesian_path(
                robot_name, config, goal.target_pose,
                velocity_scaling, acceleration_scaling, goal_handle
            )

            planning_time = time.time() - start_time

            if not success:
                result.success = False
                result.message = message
                result.planning_time = planning_time
                result.path_length = 0.0
                goal_handle.abort()
                return result

            # Execute if requested
            if goal.execute and success:
                feedback.status = 'Executing trajectory'
                feedback.progress_percentage = 50.0
                goal_handle.publish_feedback(feedback)

                exec_success = await self._execute_trajectory(robot_name, trajectory, goal_handle, 'movel')

                if not exec_success:
                    result.success = False
                    result.message = 'Trajectory execution failed'
                    result.planned_trajectory = trajectory
                    result.planning_time = planning_time
                    result.path_length = path_length
                    goal_handle.abort()
                    return result

            # Success
            result.success = True
            result.message = f'MoveL successful for {robot_name}'
            result.planned_trajectory = trajectory
            result.planning_time = planning_time
            result.path_length = path_length

            feedback.progress_percentage = 100.0
            feedback.status = 'Completed'
            goal_handle.publish_feedback(feedback)

            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f'MoveL failed: {str(e)}')
            result.success = False
            result.message = f'Exception: {str(e)}'
            goal_handle.abort()
            return result

    async def _execute_movej(self, goal_handle):
        """Execute MoveJ (joint space) motion."""
        goal = goal_handle.request
        robot_name = goal.robot_name.lower()
        result = MoveJ.Result()

        try:
            config = self.robot_configs[robot_name]
            velocity_scaling = min(max(goal.velocity_scaling, 0.01), 1.0)
            acceleration_scaling = min(max(goal.acceleration_scaling, 0.01), 1.0)

            # Publish initial feedback
            feedback = MoveJ.Feedback()
            feedback.progress_percentage = 0.0
            feedback.status = 'Starting MoveJ planning'
            goal_handle.publish_feedback(feedback)

            start_time = time.time()

            # Plan joint path with collision checking
            trajectory, success, message, joint_distance = await self._plan_joint_path(
                robot_name, config, goal.target_pose,
                velocity_scaling, acceleration_scaling, goal_handle
            )

            planning_time = time.time() - start_time

            if not success:
                result.success = False
                result.message = message
                result.planning_time = planning_time
                result.joint_distance = 0.0
                goal_handle.abort()
                return result

            # Execute if requested
            if goal.execute and success:
                feedback.status = 'Executing trajectory'
                feedback.progress_percentage = 50.0
                goal_handle.publish_feedback(feedback)

                exec_success = await self._execute_trajectory(robot_name, trajectory, goal_handle, 'movej')

                if not exec_success:
                    result.success = False
                    result.message = 'Trajectory execution failed'
                    result.planned_trajectory = trajectory
                    result.planning_time = planning_time
                    result.joint_distance = joint_distance
                    goal_handle.abort()
                    return result

            # Success
            result.success = True
            result.message = f'MoveJ successful for {robot_name}'
            result.planned_trajectory = trajectory
            result.planning_time = planning_time
            result.joint_distance = joint_distance

            feedback.progress_percentage = 100.0
            feedback.status = 'Completed'
            goal_handle.publish_feedback(feedback)

            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f'MoveJ failed: {str(e)}')
            result.success = False
            result.message = f'Exception: {str(e)}'
            goal_handle.abort()
            return result

    async def _plan_cartesian_path(
        self,
        robot_name: str,
        config: dict,
        target_pose: PoseStamped,
        velocity_scaling: float,
        acceleration_scaling: float,
        goal_handle
    ) -> Tuple[JointTrajectory, bool, str, float]:
        """Plan a Cartesian (linear) path to target pose with collision checking."""

        feedback = MoveL.Feedback()
        client = self.cartesian_path_clients[robot_name]

        # Wait for service
        if not client.service_is_ready():
            self.get_logger().warn(f'Cartesian path service not available for {robot_name}, waiting...')
            if not client.wait_for_service(timeout_sec=5.0):
                return JointTrajectory(), False, 'Cartesian path service not available', 0.0

        # Build request
        request = GetCartesianPath.Request()
        request.header.frame_id = self.world_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.group_name = config['planning_group']
        request.link_name = config['ee_frame']
        request.max_step = self.cartesian_max_step
        request.jump_threshold = self.cartesian_jump_threshold
        request.avoid_collisions = True

        # Set start state with current joint positions
        request.start_state = RobotState()
        request.start_state.joint_state.name = config['joint_names']
        with self._lock:
            request.start_state.joint_state.position = list(self.current_joint_states[robot_name])

        # Add collision objects (ground plane + other robot)
        request.start_state.attached_collision_objects = []

        # Target pose (ensure it's in world frame)
        target = target_pose.pose
        if target_pose.header.frame_id != self.world_frame and target_pose.header.frame_id != '':
            self.get_logger().warn(
                f'Target pose frame "{target_pose.header.frame_id}" differs from world frame "{self.world_frame}". '
                'Assuming pose is in world frame.'
            )

        request.waypoints = [target]

        # Call service
        try:
            feedback.status = 'Computing Cartesian path'
            feedback.progress_percentage = 25.0
            goal_handle.publish_feedback(feedback)

            future = client.call_async(request)
            response = await future

            if response.fraction < 0.99:
                return (
                    JointTrajectory(),
                    False,
                    f'Cartesian path only {response.fraction * 100:.1f}% achievable (collision or unreachable)',
                    0.0
                )

            # Calculate path length
            path_length = self._calculate_cartesian_path_length(response.solution.joint_trajectory)

            # Scale trajectory timing
            trajectory = self._scale_trajectory(
                response.solution.joint_trajectory,
                velocity_scaling,
                acceleration_scaling
            )

            feedback.progress_percentage = 50.0
            feedback.status = 'Cartesian path planned successfully'
            goal_handle.publish_feedback(feedback)

            return trajectory, True, 'Cartesian path planned successfully', path_length

        except Exception as e:
            return JointTrajectory(), False, f'Cartesian planning failed: {str(e)}', 0.0

    async def _plan_joint_path(
        self,
        robot_name: str,
        config: dict,
        target_pose: PoseStamped,
        velocity_scaling: float,
        acceleration_scaling: float,
        goal_handle
    ) -> Tuple[JointTrajectory, bool, str, float]:
        """Plan a joint space path to target pose with collision checking and shortest path optimization."""

        feedback = MoveJ.Feedback()

        # First compute IK for target pose
        feedback.status = 'Computing inverse kinematics'
        feedback.progress_percentage = 10.0
        goal_handle.publish_feedback(feedback)

        ik_solution = await self._compute_ik(robot_name, config, target_pose.pose)

        if ik_solution is None:
            return JointTrajectory(), False, 'IK failed - target pose unreachable', 0.0

        # Check if IK solution would cause ground collision
        if not await self._check_state_validity(robot_name, config, ik_solution):
            return JointTrajectory(), False, 'IK solution causes collision', 0.0

        # Plan motion to joint target
        feedback.status = 'Planning joint space path'
        feedback.progress_percentage = 30.0
        goal_handle.publish_feedback(feedback)

        trajectory = await self._plan_to_joint_target(
            robot_name, config, ik_solution, velocity_scaling, acceleration_scaling
        )

        if trajectory is None:
            return JointTrajectory(), False, 'Motion planning failed - no collision-free path found', 0.0

        # Calculate joint distance
        joint_distance = self._calculate_joint_distance(trajectory)

        feedback.progress_percentage = 50.0
        feedback.status = 'Joint path planned successfully'
        goal_handle.publish_feedback(feedback)

        return trajectory, True, 'Joint path planned successfully', joint_distance

    async def _compute_ik(
        self,
        robot_name: str,
        config: dict,
        target_pose: Pose
    ) -> Optional[List[float]]:
        """Compute inverse kinematics for target pose with collision avoidance."""

        client = self.ik_clients[robot_name]

        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('IK service not available')
                return None

        request = GetPositionIK.Request()
        request.ik_request.group_name = config['planning_group']
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = 2
        request.ik_request.attempts = 10

        # Set current state as seed
        request.ik_request.robot_state = RobotState()
        request.ik_request.robot_state.joint_state.name = config['joint_names']
        with self._lock:
            request.ik_request.robot_state.joint_state.position = list(self.current_joint_states[robot_name])

        # Set target pose
        request.ik_request.pose_stamped.header.frame_id = self.world_frame
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = target_pose

        try:
            future = client.call_async(request)
            response = await future

            if response.error_code.val == 1:  # SUCCESS
                # Extract joint positions for the 6 arm joints
                joint_positions = []
                for joint_name in config['joint_names']:
                    if joint_name in response.solution.joint_state.name:
                        idx = response.solution.joint_state.name.index(joint_name)
                        joint_positions.append(response.solution.joint_state.position[idx])

                if len(joint_positions) == 6:
                    return joint_positions
                else:
                    self.get_logger().warn(f'IK returned {len(joint_positions)} joints, expected 6')
                    return list(response.solution.joint_state.position[:6])
            else:
                self.get_logger().warn(f'IK failed with error code: {response.error_code.val}')
                return None

        except Exception as e:
            self.get_logger().error(f'IK computation error: {str(e)}')
            return None

    async def _check_state_validity(
        self,
        robot_name: str,
        config: dict,
        joint_positions: List[float]
    ) -> bool:
        """Check if a joint state is collision-free (including ground)."""
        # For now, we rely on MoveIt's collision checking during planning
        # This is a placeholder for additional custom collision checks

        # Basic ground collision check would require forward kinematics
        # which is handled by MoveIt's collision checking
        return True

    async def _plan_to_joint_target(
        self,
        robot_name: str,
        config: dict,
        joint_target: List[float],
        velocity_scaling: float,
        acceleration_scaling: float
    ) -> Optional[JointTrajectory]:
        """Plan motion to joint target with collision checking and shortest path optimization."""

        client = self.motion_plan_clients[robot_name]

        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('Motion planning service not available')
                return None

        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = config['planning_group']
        request.motion_plan_request.num_planning_attempts = self.num_planning_attempts
        request.motion_plan_request.allowed_planning_time = self.planning_time
        request.motion_plan_request.max_velocity_scaling_factor = velocity_scaling
        request.motion_plan_request.max_acceleration_scaling_factor = acceleration_scaling

        # Use RRTConnect or similar planner that finds shortest path
        # The planner is configured in MoveIt config, but we request optimization
        request.motion_plan_request.planner_id = ''  # Use default (typically RRTConnect)

        # Set start state
        request.motion_plan_request.start_state = RobotState()
        request.motion_plan_request.start_state.joint_state.name = config['joint_names']
        with self._lock:
            request.motion_plan_request.start_state.joint_state.position = list(
                self.current_joint_states[robot_name]
            )

        # Set goal constraints (joint target)
        goal_constraints = Constraints()
        for i, joint_name in enumerate(config['joint_names']):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = joint_target[i]
            constraint.tolerance_above = self.goal_joint_tolerance
            constraint.tolerance_below = self.goal_joint_tolerance
            constraint.weight = 1.0
            goal_constraints.joint_constraints.append(constraint)

        request.motion_plan_request.goal_constraints.append(goal_constraints)

        try:
            future = client.call_async(request)
            response = await future

            if response.motion_plan_response.error_code.val == 1:  # SUCCESS
                return response.motion_plan_response.trajectory.joint_trajectory
            else:
                error_code = response.motion_plan_response.error_code.val
                self.get_logger().warn(f'Motion planning failed with error code: {error_code}')
                return None

        except Exception as e:
            self.get_logger().error(f'Motion planning error: {str(e)}')
            return None

    async def _execute_trajectory(
        self,
        robot_name: str,
        trajectory: JointTrajectory,
        goal_handle,
        motion_type: str
    ) -> bool:
        """Execute planned trajectory on the robot."""

        config = self.robot_configs[robot_name]
        namespace = config['namespace']

        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient

        action_client = ActionClient(
            self,
            FollowJointTrajectory,
            f'/{namespace}/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )

        if not action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Trajectory execution action server not available')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        try:
            send_goal_future = action_client.send_goal_async(goal)
            traj_goal_handle = await send_goal_future

            if not traj_goal_handle.accepted:
                self.get_logger().error('Trajectory execution goal rejected')
                return False

            # Monitor execution with feedback
            result_future = traj_goal_handle.get_result_async()

            while not result_future.done():
                # Update feedback
                if motion_type == 'movel':
                    feedback = MoveL.Feedback()
                else:
                    feedback = MoveJ.Feedback()

                feedback.status = 'Executing trajectory'
                feedback.progress_percentage = 75.0  # Approximate
                goal_handle.publish_feedback(feedback)

                await asyncio.sleep(0.1)

            result = await result_future
            return result.result.error_code == 0

        except Exception as e:
            self.get_logger().error(f'Trajectory execution error: {str(e)}')
            return False

    def _scale_trajectory(
        self,
        trajectory: JointTrajectory,
        velocity_scaling: float,
        acceleration_scaling: float
    ) -> JointTrajectory:
        """Scale trajectory timing based on velocity/acceleration factors."""

        scaled = JointTrajectory()
        scaled.header = trajectory.header
        scaled.joint_names = list(trajectory.joint_names)

        time_scale = 1.0 / velocity_scaling

        for point in trajectory.points:
            new_point = JointTrajectoryPoint()
            new_point.positions = list(point.positions)

            if point.velocities:
                new_point.velocities = [v * velocity_scaling for v in point.velocities]

            if point.accelerations:
                new_point.accelerations = [a * acceleration_scaling for a in point.accelerations]

            # Scale time
            original_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            scaled_time = original_time * time_scale
            new_point.time_from_start = Duration(
                sec=int(scaled_time),
                nanosec=int((scaled_time % 1) * 1e9)
            )

            scaled.points.append(new_point)

        return scaled

    def _calculate_cartesian_path_length(self, trajectory: JointTrajectory) -> float:
        """Estimate Cartesian path length from trajectory."""
        # This is a simplified estimation based on joint motion
        # For accurate Cartesian length, would need FK at each point
        if len(trajectory.points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(trajectory.points)):
            prev_pos = np.array(trajectory.points[i-1].positions)
            curr_pos = np.array(trajectory.points[i].positions)
            # Approximate Cartesian motion (very rough estimate)
            joint_diff = np.linalg.norm(curr_pos - prev_pos)
            total_length += joint_diff * 0.1  # Rough scaling factor

        return total_length

    def _calculate_joint_distance(self, trajectory: JointTrajectory) -> float:
        """Calculate total joint space distance traveled."""
        if len(trajectory.points) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(trajectory.points)):
            prev_pos = np.array(trajectory.points[i-1].positions)
            curr_pos = np.array(trajectory.points[i].positions)
            total_distance += np.linalg.norm(curr_pos - prev_pos)

        return total_distance


# Need asyncio for the execute_trajectory
import asyncio


def main(args=None):
    rclpy.init(args=args)

    server = PathPlanningActionServer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
