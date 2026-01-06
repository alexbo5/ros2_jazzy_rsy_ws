#!/usr/bin/env python3
"""Path Planning Action Server - Pilz with OMPL/Cartesian fallback."""

import math
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

from rsy_path_planning.action import MoveJ, MoveL, MoveJJoints, MoveLJoints


UR_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]

CONTINUOUS_JOINTS = [0, 3, 4, 5]

ERROR_NAMES = {
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


class PathPlanningActionServer(Node):
    """Path planning server with hybrid Pilz/OMPL strategy."""

    def __init__(self):
        super().__init__('path_planning_action_server')
        self._declare_parameters()
        self._load_config()
        self._setup_clients()
        self._setup_action_servers()
        self.get_logger().info('Path Planning Server ready')

    def _declare_parameters(self):
        """Declare ROS parameters."""
        params = [
            ('robots.robot1.planning_group', 'robot1_ur_manipulator'),
            ('robots.robot1.end_effector_link', 'robot1_tcp'),
            ('robots.robot2.planning_group', 'robot2_ur_manipulator'),
            ('robots.robot2.end_effector_link', 'robot2_tcp'),
            ('planning.default_frame', 'world'),
            ('planning.allowed_time', 5.0),
            ('ompl.planner_id', 'RRTConnect'),
            ('ompl.num_attempts', 10),
            ('ompl.allowed_time', 10.0),
            ('cartesian.max_step', 0.01),
            ('cartesian.min_fraction', 0.95),
            ('velocity_scaling.ptp', 0.5),
            ('velocity_scaling.lin', 0.3),
            ('acceleration_scaling.ptp', 0.5),
            ('acceleration_scaling.lin', 0.3),
            ('tolerances.position', 0.001),
            ('tolerances.orientation', 0.01),
            ('tolerances.joint', 0.01),
        ]
        for name, default in params:
            self.declare_parameter(name, default)

    def _load_config(self):
        """Load configuration from parameters."""
        get = lambda p: self.get_parameter(p).value

        self.robots = {
            name: {
                'group': get(f'robots.{name}.planning_group'),
                'ee_link': get(f'robots.{name}.end_effector_link'),
            }
            for name in ['robot1', 'robot2']
        }

        self.joint_names = {
            name: [f'{name}_{j}' for j in UR_JOINTS]
            for name in ['robot1', 'robot2']
        }

        self.cfg = {
            'frame': get('planning.default_frame'),
            'pilz_time': get('planning.allowed_time'),
            'ompl_planner': get('ompl.planner_id'),
            'ompl_attempts': get('ompl.num_attempts'),
            'ompl_time': get('ompl.allowed_time'),
            'cart_step': get('cartesian.max_step'),
            'cart_min_frac': get('cartesian.min_fraction'),
            'vel_ptp': get('velocity_scaling.ptp'),
            'vel_lin': get('velocity_scaling.lin'),
            'acc_ptp': get('acceleration_scaling.ptp'),
            'acc_lin': get('acceleration_scaling.lin'),
            'tol_pos': get('tolerances.position'),
            'tol_ori': get('tolerances.orientation'),
            'tol_joint': get('tolerances.joint'),
        }

        self._joint_states = {}

    def _setup_clients(self):
        """Setup service and action clients."""
        cb = ReentrantCallbackGroup()

        self._plan_client = self.create_client(
            GetMotionPlan, '/plan_kinematic_path', callback_group=cb)
        self._fk_client = self.create_client(
            GetPositionFK, '/compute_fk', callback_group=cb)
        self._cart_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=cb)
        self._exec_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=cb)

        self.get_logger().info('Waiting for MoveIt services...')
        for client in [self._plan_client, self._fk_client, self._cart_client]:
            client.wait_for_service(timeout_sec=30.0)
        self._exec_client.wait_for_server(timeout_sec=30.0)

        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)

    def _setup_action_servers(self):
        """Setup action servers."""
        cb = ReentrantCallbackGroup()
        actions = [
            (MoveJ, 'move_j', self._exec_movej),
            (MoveL, 'move_l', self._exec_movel),
            (MoveJJoints, 'move_j_joints', self._exec_movej_joints),
            (MoveLJoints, 'move_l_joints', self._exec_movel_joints),
        ]
        for action_type, name, callback in actions:
            ActionServer(self, action_type, name,
                         execute_callback=callback, callback_group=cb)

    def _on_joint_state(self, msg):
        """Store joint states."""
        for name, pos in zip(msg.name, msg.position):
            self._joint_states[name] = pos

    # Action callbacks

    async def _exec_movej(self, goal_handle):
        """MoveJ action."""
        return await self._execute_ptp(goal_handle, MoveJ, target_type='pose')

    async def _exec_movel(self, goal_handle):
        """MoveL action."""
        return await self._execute_lin(goal_handle, MoveL, target_type='pose')

    async def _exec_movej_joints(self, goal_handle):
        """MoveJJoints action."""
        return await self._execute_ptp(goal_handle, MoveJJoints, target_type='joints')

    async def _exec_movel_joints(self, goal_handle):
        """MoveLJoints action."""
        return await self._execute_lin(goal_handle, MoveLJoints, target_type='joints')

    # Planning

    async def _execute_ptp(self, goal_handle, action_type, target_type):
        """Execute PTP motion with OMPL fallback."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        robot = self.robots[req.robot_name]

        if target_type == 'pose':
            pose = self._build_pose(req)
            self.get_logger().info(
                f'MoveJ {req.robot_name}: ({req.x:.3f}, {req.y:.3f}, {req.z:.3f})')
            pilz_req = self._build_request(req.robot_name, robot, 'pilz', 'PTP', pose=pose)
            ompl_req = self._build_request(req.robot_name, robot, 'ompl', None, pose=pose)
        else:
            joints = self._validate_joints(req, result, goal_handle)
            if joints is None:
                return result
            joints = self._normalize_joints(req.robot_name, joints)
            self.get_logger().info(
                f'MoveJJoints {req.robot_name}: {[f"{v:.2f}" for v in joints]}')
            pilz_req = self._build_request(req.robot_name, robot, 'pilz', 'PTP', joints=joints)
            ompl_req = self._build_request(req.robot_name, robot, 'ompl', None, joints=joints)

        for planner, plan_req in [('Pilz', pilz_req), ('OMPL', ompl_req)]:
            feedback.status = f'Planning ({planner})...'
            goal_handle.publish_feedback(feedback)

            trajectory = await self._try_plan(plan_req)
            if trajectory:
                self.get_logger().info(f'{planner} succeeded')
                return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

            self.get_logger().warn(f'{planner} failed: {self._get_error_name(self._last_error)}')

        return self._abort(goal_handle, result, 'PTP planning failed')

    async def _execute_lin(self, goal_handle, action_type, target_type):
        """Execute LIN motion with Cartesian fallback."""
        req = goal_handle.request
        result = action_type.Result()
        feedback = action_type.Feedback()

        if req.robot_name not in self.robots:
            return self._abort(goal_handle, result, f'Unknown robot: {req.robot_name}')

        robot = self.robots[req.robot_name]

        if target_type == 'pose':
            pose = self._build_pose(req)
            frame = req.frame_id or self.cfg['frame']
            self.get_logger().info(
                f'MoveL {req.robot_name}: ({req.x:.3f}, {req.y:.3f}, {req.z:.3f})')
        else:
            joints = self._validate_joints(req, result, goal_handle)
            if joints is None:
                return result
            feedback.status = 'Computing FK...'
            goal_handle.publish_feedback(feedback)
            target_pose = await self._compute_fk(robot, self.joint_names[req.robot_name], joints)
            if target_pose is None:
                return self._abort(goal_handle, result, 'FK computation failed')
            pose = PoseStamped()
            pose.header.frame_id = self.cfg['frame']
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose = target_pose
            frame = self.cfg['frame']
            self.get_logger().info(
                f'MoveLJoints {req.robot_name}: ({target_pose.position.x:.3f}, '
                f'{target_pose.position.y:.3f}, {target_pose.position.z:.3f})')

        feedback.status = 'Planning LIN (Pilz)...'
        goal_handle.publish_feedback(feedback)

        pilz_req = self._build_request(req.robot_name, robot, 'pilz', 'LIN', pose=pose)
        trajectory = await self._try_plan(pilz_req)
        if trajectory:
            self.get_logger().info('Pilz LIN succeeded')
            return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

        self.get_logger().warn(f'Pilz LIN failed: {self._get_error_name(self._last_error)}')

        feedback.status = 'Planning Cartesian path...'
        goal_handle.publish_feedback(feedback)

        trajectory = await self._plan_cartesian(req.robot_name, robot, pose.pose, frame)
        if trajectory:
            self.get_logger().info('Cartesian path succeeded')
            return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

        self.get_logger().warn('Cartesian path failed, trying PTP fallback...')
        feedback.status = 'Planning PTP fallback...'
        goal_handle.publish_feedback(feedback)

        # PTP fallback for LIN - use OMPL when linear path is not possible
        ompl_req = self._build_request(req.robot_name, robot, 'ompl', None, pose=pose)
        trajectory = await self._try_plan(ompl_req)
        if trajectory:
            self.get_logger().info('OMPL PTP fallback succeeded for LIN')
            return await self._execute_trajectory(goal_handle, result, feedback, trajectory)

        return self._abort(goal_handle, result, 'LIN planning failed (all methods)')

    async def _try_plan(self, plan_req):
        """Try planning, return trajectory or None."""
        resp = await self._plan_client.call_async(plan_req)
        self._last_error = resp.motion_plan_response.error_code.val
        if self._last_error == MoveItErrorCodes.SUCCESS:
            return resp.motion_plan_response.trajectory
        return None

    async def _plan_cartesian(self, robot_name, robot, target_pose, frame):
        """Plan Cartesian path with interpolated waypoints for longer distances."""
        # First get current end-effector pose to compute distance
        current_joints = self._get_joint_values(robot_name)
        current_pose = await self._compute_fk(robot, self.joint_names[robot_name], current_joints)
        
        if current_pose is None:
            self.get_logger().warn('Could not compute current FK for cartesian planning')
            return await self._plan_cartesian_direct(robot_name, robot, target_pose, frame)
        
        # Compute distance
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        dz = target_pose.position.z - current_pose.position.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        self.get_logger().info(f'Cartesian distance: {distance:.4f}m')
        
        # For short distances, use direct planning
        if distance < 0.05:  # Less than 5cm
            return await self._plan_cartesian_direct(robot_name, robot, target_pose, frame)
        
        # For longer distances, interpolate waypoints
        num_waypoints = max(2, int(distance / 0.02))  # One waypoint every 2cm
        waypoints = []
        
        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            wp = Pose()
            wp.position.x = current_pose.position.x + t * dx
            wp.position.y = current_pose.position.y + t * dy
            wp.position.z = current_pose.position.z + t * dz
            # SLERP for orientation would be better, but linear interpolation for now
            wp.orientation.x = current_pose.orientation.x + t * (target_pose.orientation.x - current_pose.orientation.x)
            wp.orientation.y = current_pose.orientation.y + t * (target_pose.orientation.y - current_pose.orientation.y)
            wp.orientation.z = current_pose.orientation.z + t * (target_pose.orientation.z - current_pose.orientation.z)
            wp.orientation.w = current_pose.orientation.w + t * (target_pose.orientation.w - current_pose.orientation.w)
            # Normalize quaternion
            qnorm = math.sqrt(wp.orientation.x**2 + wp.orientation.y**2 + wp.orientation.z**2 + wp.orientation.w**2)
            if qnorm > 0:
                wp.orientation.x /= qnorm
                wp.orientation.y /= qnorm
                wp.orientation.z /= qnorm
                wp.orientation.w /= qnorm
            waypoints.append(wp)
        
        self.get_logger().info(f'Planning cartesian path with {len(waypoints)} waypoints')
        
        req = GetCartesianPath.Request()
        req.header.frame_id = frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = robot['group']
        req.link_name = robot['ee_link']
        req.waypoints = waypoints
        req.max_step = self.cfg['cart_step']
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state = self._get_robot_state(robot_name)
        req.max_velocity_scaling_factor = self.cfg['vel_lin']
        req.max_acceleration_scaling_factor = self.cfg['acc_lin']

        resp = await self._cart_client.call_async(req)

        self.get_logger().info(f'Cartesian path result: {resp.fraction:.1%} achieved, error: {resp.error_code.val}')

        if (resp.error_code.val == MoveItErrorCodes.SUCCESS and
                resp.fraction >= self.cfg['cart_min_frac']):
            return resp.solution
        return None

    async def _plan_cartesian_direct(self, robot_name, robot, target_pose, frame):
        """Plan direct Cartesian path (single waypoint)."""
        req = GetCartesianPath.Request()
        req.header.frame_id = frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = robot['group']
        req.link_name = robot['ee_link']
        req.waypoints = [target_pose]
        req.max_step = self.cfg['cart_step']
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state = self._get_robot_state(robot_name)
        req.max_velocity_scaling_factor = self.cfg['vel_lin']
        req.max_acceleration_scaling_factor = self.cfg['acc_lin']

        resp = await self._cart_client.call_async(req)

        self.get_logger().info(f'Direct cartesian path: {resp.fraction:.1%} achieved')

        if (resp.error_code.val == MoveItErrorCodes.SUCCESS and
                resp.fraction >= self.cfg['cart_min_frac']):
            return resp.solution
        return None

    async def _compute_fk(self, robot, joint_names, joint_values):
        """Compute forward kinematics."""
        req = GetPositionFK.Request()
        req.header.frame_id = self.cfg['frame']
        req.header.stamp = self.get_clock().now().to_msg()
        req.fk_link_names = [robot['ee_link']]
        req.robot_state.joint_state.name = joint_names
        req.robot_state.joint_state.position = joint_values

        try:
            resp = await self._fk_client.call_async(req)
            if resp.error_code.val == MoveItErrorCodes.SUCCESS:
                return resp.pose_stamped[0].pose
        except Exception as e:
            self.get_logger().error(f'FK failed: {e}')
        return None

    async def _execute_trajectory(self, goal_handle, result, feedback, trajectory):
        """Execute trajectory."""
        feedback.status = 'Executing...'
        goal_handle.publish_feedback(feedback)

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        handle = await self._exec_client.send_goal_async(goal)
        if not handle.accepted:
            return self._abort(goal_handle, result, 'Execution rejected')

        exec_result = await handle.get_result_async()

        if exec_result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            result.success = True
            result.message = 'Done'
            goal_handle.succeed()
        else:
            result.success = False
            result.message = f'Execution failed: {self._get_error_name(exec_result.result.error_code.val)}'
            goal_handle.abort()

        return result

    # Request building

    def _build_request(self, robot_name, robot, pipeline, planner_id, pose=None, joints=None):
        """Build motion plan request."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request
        mp.group_name = robot['group']
        mp.start_state = self._get_robot_state(robot_name)

        if pipeline == 'pilz':
            mp.pipeline_id = 'pilz_industrial_motion_planner'
            mp.planner_id = planner_id
            mp.num_planning_attempts = 1
            mp.allowed_planning_time = self.cfg['pilz_time']
            vel_type = 'ptp' if planner_id == 'PTP' else 'lin'
        else:
            mp.pipeline_id = 'ompl'
            mp.planner_id = self.cfg['ompl_planner']
            mp.num_planning_attempts = self.cfg['ompl_attempts']
            mp.allowed_planning_time = self.cfg['ompl_time']
            vel_type = 'ptp'

        mp.max_velocity_scaling_factor = self.cfg[f'vel_{vel_type}']
        mp.max_acceleration_scaling_factor = self.cfg[f'acc_{vel_type}']

        if pose:
            mp.goal_constraints.append(self._build_pose_constraints(robot, pose))
        elif joints:
            mp.goal_constraints.append(
                self._build_joint_constraints(self.joint_names[robot_name], joints))

        return req

    def _build_pose_constraints(self, robot, pose):
        """Build pose constraints."""
        constraints = Constraints()

        pos = PositionConstraint()
        pos.header = pose.header
        pos.link_name = robot['ee_link']
        pos.weight = 1.0

        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.cfg['tol_pos']]
        bv.primitives.append(sphere)

        sphere_pose = Pose()
        sphere_pose.position = pose.pose.position
        sphere_pose.orientation.w = 1.0
        bv.primitive_poses.append(sphere_pose)
        pos.constraint_region = bv
        constraints.position_constraints.append(pos)

        ori = OrientationConstraint()
        ori.header = pose.header
        ori.link_name = robot['ee_link']
        ori.orientation = pose.pose.orientation
        ori.absolute_x_axis_tolerance = self.cfg['tol_ori']
        ori.absolute_y_axis_tolerance = self.cfg['tol_ori']
        ori.absolute_z_axis_tolerance = self.cfg['tol_ori']
        ori.weight = 1.0
        constraints.orientation_constraints.append(ori)

        return constraints

    def _build_joint_constraints(self, joint_names, joint_values):
        """Build joint constraints."""
        constraints = Constraints()
        for name, value in zip(joint_names, joint_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = self.cfg['tol_joint']
            jc.tolerance_below = self.cfg['tol_joint']
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        return constraints

    # Utilities

    def _build_pose(self, req):
        """Build PoseStamped from request."""
        pose = PoseStamped()
        pose.header.frame_id = req.frame_id or self.cfg['frame']
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = req.x
        pose.pose.position.y = req.y
        pose.pose.position.z = req.z
        pose.pose.orientation.x = req.qx
        pose.pose.orientation.y = req.qy
        pose.pose.orientation.z = req.qz
        pose.pose.orientation.w = req.qw
        return pose

    def _validate_joints(self, req, result, goal_handle):
        """Validate joint values count."""
        joint_names = self.joint_names[req.robot_name]
        joints = list(req.joint_values)
        if len(joints) != len(joint_names):
            self._abort(goal_handle, result,
                        f'Expected {len(joint_names)} joints, got {len(joints)}')
            return None
        return joints

    def _normalize_joints(self, robot_name, target):
        """Normalize continuous joints to shortest path from current position.

        Uses the raw current state (what the robot reports) as the reference
        for computing the shortest path to the target.
        """
        current = self._get_joint_values(robot_name)  # Use raw values, not normalized
        normalized = list(target)

        for idx in CONTINUOUS_JOINTS:
            if idx < len(normalized):
                # Find equivalent target angle closest to current position
                tgt = target[idx]
                curr = current[idx]
                
                # Bring target to within 2π of current
                while tgt - curr > math.pi:
                    tgt -= 2 * math.pi
                while tgt - curr < -math.pi:
                    tgt += 2 * math.pi
                normalized[idx] = tgt

        return normalized

    def _get_joint_values(self, robot_name):
        """Get current joint values."""
        return [self._joint_states.get(name, 0.0) for name in self.joint_names[robot_name]]

    def _get_robot_state(self, robot_name):
        """Get robot state for planning - use actual joint values."""
        state = RobotState()
        state.joint_state.name = self.joint_names[robot_name]
        state.joint_state.position = self._get_joint_values(robot_name)  # Raw values, no normalization
        return state

    # Remove _normalize_start_state method entirely - it's no longer needed

    def _scale_trajectory(self, trajectory, motion_type):
        """Scale trajectory velocities and accelerations.
        
        Note: This is only used when the planner doesn't support scaling factors directly.
        For most cases, use the scaling factors in the planning request instead.
        """
        vel_scale = self.cfg[f'vel_{motion_type}']
        acc_scale = self.cfg[f'acc_{motion_type}']
        
        for pt in trajectory.joint_trajectory.points:
            # Scale time (slower = longer time)
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            t_scaled = t / vel_scale
            pt.time_from_start.sec = int(t_scaled)
            pt.time_from_start.nanosec = int((t_scaled % 1) * 1e9)
            
            # Scale velocities
            pt.velocities = [v * vel_scale for v in pt.velocities]
            
            # Scale accelerations using acceleration scaling factor
            if pt.accelerations:
                pt.accelerations = [a * acc_scale for a in pt.accelerations]
        return trajectory

    def _abort(self, goal_handle, result, message):
        """Abort action with error."""
        self.get_logger().error(message)
        result.success = False
        result.message = message
        goal_handle.abort()
        return result

    def _get_error_name(self, code):
        """Get error name from code."""
        return ERROR_NAMES.get(code, f'UNKNOWN_{code}')


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
