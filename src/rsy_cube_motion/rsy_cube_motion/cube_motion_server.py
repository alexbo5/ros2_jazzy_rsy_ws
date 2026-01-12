#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
import numpy as np
from dataclasses import dataclass

from rsy_cube_motion.action import RotateFace, HandOver, TakeUpCube, PutDownCube, PresentCubeFace
from rsy_mtc_planning.action import ExecuteMotionSequence
from rsy_mtc_planning.msg import MotionStep
from geometry_msgs.msg import PoseStamped

# Roboter 1 dreht U, F, D; Roboter 2 dreht L, B, R
ROBOT_FACES = ["U", "F", "D"], ["L", "B", "R"]

# distance (mm) from cube center to gripper contact point
OFFSET_DIST_HOLD_CUBE = -42     # distance when holding the cube (grasps 2 rows of cube)
OFFSET_DIST_SPIN_CUBE = -22    # distance when spinning the cube (grasps 1 row of cube)
OFFSET_DIST_PRE_TARGET = 100   # distance when approaching the cube (pre-grasp position)
OFFSET_DIST_TAKE_CUBE = 150    # distance when taking up the cube from rest position

# Compact, editable definition of cube-access poses.
# CUBE_POSE_DEFS format:
#   face: (position_xyz, face_normal_orientation, holding_angle)
# where face_normal_orientation = [nx, ny, nz, rotation_rad] is a unified representation:
#   - [nx, ny, nz]: unit vector normal to the face (direction the face points)
#   - rotation_rad: rotation angle (radians) around the face normal axis
#     This unambiguously defines the cube's 3D orientation in world coordinates.
CUBE_POSE_DEFS = {
    # face: (position_xyz, face_normal_orientation=[nx, ny, nz, rotation_rad], holding_angle)
    "U": ([-0.38313, 0.32432, 0.5], [1.0, 0.0, 0.0, 3.14159], [90]),
    "D": ([-0.38313, 0.32432, 0.5], [1.0, 0.0, 0.0, 0.0], [-90]),
    "F": ([-0.38313, 0.32432, 0.5], [1.0, 0.0, 0.0, 0.0], [180]),
    "B": ([-0.38313, 0.32432, 0.5], [-1.0, 0.0, 0.0, 0.0], [180]),
    "L": ([-0.38313, 0.32432, 0.5], [-1.0, 0.0, 0.0, 3.14159], [-90]),
    "R": ([-0.38313, 0.32432, 0.5], [-1.0, 0.0, 0.0, 0.0], [90]),
}

HAND_OVER_POSE_DEF = {
    "position": [-0.38313, 0.32432, 0.5],
    "orientation_vector": [1.0, 0.0, 0.0]  # approach axis
}

# Position where the cube rests (for taking up and putting down)
# Robot 2 will always take up and put down from/to this position
CUBE_REST_POSE_DEF = {
    "position": [-0.11954 + (OFFSET_DIST_HOLD_CUBE / 1000) , 0.7334, 0.11039],  # Rest position
    "orientation_vector": [-1.0, 0.0, 0.0]  # approach axis
}

# Presentation poses - each face is presented to the camera (facing upward in Z direction)
# These poses are used by PresentCubeFace action to show each face for scanning
# Format: face -> (position, face_normal_orientation, robot_holding_angle)
CUBE_PRESENT_POSES = {
    "U": ([-0.38313, 0.32432, 0.5], [0.0, 0.0, 1.0, -3.14159/2.0], [-90]),
    "D": ([-0.38313, 0.32432, 0.5], [0.0, 0.0, 1.0, 3.14159/2.0], [90]),
    "F": ([-0.38313, 0.32432, 0.5], [0.0, 0.0, 1.0, 0.0], [180]),
    "B": ([-0.38313, 0.32432, 0.5], [0.0, 0.0, 1.0, 0.0], [180]),
    "L": ([-0.38313, 0.32432, 0.5], [0.0, 0.0, 1.0, -3.14159/2.0], [90]),
    "R": ([-0.38313, 0.32432, 0.5], [0.0, 0.0, 1.0, 3.14159/2.0], [-90]),
}

# Define gripper reference forward vector in gripper's local frame (tool Z-axis?)
GRIPPER_FORWARD_DIRECTION = np.array([0.0, 0.0, 1.0])



@dataclass
class CubePose:
    """Represents a cube pose with position and unified face normal orientation.

    The face_normal_orientation is a 4-tuple [nx, ny, nz, rotation_rad] where:
    - [nx, ny, nz] is the unit vector normal to the face (which direction the face points)
    - rotation_rad is the rotation angle (in radians) around that normal axis

    This unified representation unambiguously defines the cube's complete 3D orientation
    in world coordinates. For each face and holding angle, there is exactly one possible
    approach direction for the holding robot and exactly one direction for each spinner.
    """
    position: np.ndarray  # [x, y, z]
    face_normal_orientation: np.ndarray  # [nx, ny, nz, rotation_rad]
    holding_angle: float = 0.0  # degrees, preferred holding angle for this face

@dataclass
class GripperPose:
    """Represents a gripper pose for grasping the cube."""
    position: np.ndarray  # [x, y, z]
    quaternion: np.ndarray  # [x, y, z, w]

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quaternion_from_axis_angle(axis, angle):
    """Create quaternion from axis-angle representation."""
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    return np.array([
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle),
        np.cos(half_angle)
    ])

def decode_kociemba_move(move):
    """
    Decodes a single Kociemba move like 'U', 'R2', "F'", etc.

    Returns:
        face (str): One of U, D, L, R, F, B
        angle_deg (int): +90, -90, or 180
    """

    if len(move) == 0:
        raise ValueError("Move string is empty")

    # First character is always the face
    face = move[0]

    if face not in ["U", "D", "L", "R", "F", "B"]:
        raise ValueError(f"Invalid face in move: {move}")

    # Default angle: 90°
    angle_deg = 90

    # Check for modifier (', 2, nothing)
    if len(move) > 1:
        modifier = move[1]

        if modifier == "'":
            angle_deg = -90
        elif modifier == "2":
            angle_deg = 180
        else:
            raise ValueError(f"Invalid move modifier: {modifier}")

    return face, angle_deg

def quaternion_normalize(q):
    norm = np.linalg.norm(q)
    if norm == 0:
        return q
    return q / norm

def rotate_vector_by_quaternion(v, q):
    """Rotate a vector v by quaternion q.
    Uses the rotation formula: v_rot = q * v * q^-1
    where v is treated as [v.x, v.y, v.z, 0] as a quaternion.
    """
    # Conjugate of q (inverse for unit quaternion)
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])

    # Treat v as quaternion with w=0
    v_quat = np.array([v[0], v[1], v[2], 0.0])

    # q * v
    qv = quaternion_multiply(q, v_quat)
    # (q * v) * q_conj
    result_quat = quaternion_multiply(qv, q_conj)

    # Extract vector part (discard w component)
    return result_quat[:3]

def quaternion_from_two_vectors(v_from, v_to):
    """Return quaternion rotating v_from -> v_to (shortest rotation).
    Both vectors should be 3D and will be normalized internally.
    Handles opposite vectors by choosing an arbitrary orthogonal axis.
    """
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)

    dot = np.dot(a, b)
    if dot > 0.999999:
        # nearly identical
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.999999:
        # opposite vectors: need an orthogonal axis
        # pick axis perpendicular to a
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis = axis / np.linalg.norm(axis)
        return quaternion_from_axis_angle(axis, np.pi)

    axis = np.cross(a, b)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot])
    return quaternion_normalize(q)


class CubeMotionServer(Node):

    def __init__(self):
        super().__init__('cube_motion_server')

        # Roboter, der den Würfel dreht (1 oder 2)
        self.cube_spinning_robot = 1

        # Cube poses for each accessible face (U, D, L, R, F, B)
        self.cube_poses = self._init_cube_poses()

        # Handover pose (neutral position between robots)
        # face_normal_orientation defines both the approach vector and cube orientation
        self.handover_pose = CubePose(
            position=np.array(HAND_OVER_POSE_DEF["position"]),
            face_normal_orientation=np.array([
                HAND_OVER_POSE_DEF["orientation_vector"][0],
                HAND_OVER_POSE_DEF["orientation_vector"][1],
                HAND_OVER_POSE_DEF["orientation_vector"][2],
                0.0  # rotation_rad = 0 (no rotation around normal for handover)
            ], dtype=float)
        )

        # Rest pose (where the cube is placed at the beginning and where it goes after solving)
        # Robot 2 always takes up and puts down from/to this position
        self.rest_pose = CubePose(
            position=np.array(CUBE_REST_POSE_DEF["position"]),
            face_normal_orientation=np.array([
                CUBE_REST_POSE_DEF["orientation_vector"][0],
                CUBE_REST_POSE_DEF["orientation_vector"][1],
                CUBE_REST_POSE_DEF["orientation_vector"][2],
                -3.14159/2
            ], dtype=float)
        )

        # ActionServer: Drehen einer Würfelseite
        self.rotate_server = ActionServer(
            self,
            RotateFace,
            'rotate_face',
            self.execute_rotate_face
        )

        # ActionServer: Griff wechseln
        self.hand_over_server = ActionServer(
            self,
            HandOver,
            'hand_over',
            self.execute_hand_over
        )

        # ActionServer: Cube aufnehmen
        self.take_up_server = ActionServer(
            self,
            TakeUpCube,
            'take_up_cube',
            self.execute_take_up_cube
        )

        # ActionServer: Cube ablegen
        self.put_down_server = ActionServer(
            self,
            PutDownCube,
            'put_down_cube',
            self.execute_put_down_cube
        )

        # ActionServer: Würfel face präsentieren
        self.present_server = ActionServer(
            self,
            PresentCubeFace,
            'present_cube_face',
            self.execute_present_cube_face
        )

        # interner Client: RotateFace kann HandOver aufrufen
        self.hand_over_client = ActionClient(self, HandOver, 'hand_over')

        # MTC motion sequence action client
        self.mtc_client = ActionClient(self, ExecuteMotionSequence, 'execute_motion_sequence')

        self.get_logger().info("Cube Motion Server ready")

    def _init_cube_poses(self):
        """Initialize cube poses for each accessible face.
        Uses the compact CUBE_POSE_DEFS above (position, face_normal_orientation, holding_angle).
        """
        poses = {}
        for face, pose_data in CUBE_POSE_DEFS.items():
            # Unpack the tuple (position, face_normal_orientation, holding_angle)
            pos, face_normal_orientation, holding = pose_data

            # holding in CUBE_POSE_DEFS may be a list like [90] or a scalar
            if isinstance(holding, (list, tuple)):
                holding_angle = float(holding[0]) if len(holding) > 0 else 0.0
            else:
                holding_angle = float(holding)

            # Ensure face_normal_orientation is a numpy array with 4 components
            face_normal_orientation_array = np.array(face_normal_orientation, dtype=float)
            if len(face_normal_orientation_array) != 4:
                raise ValueError(
                    f"Face normal orientation for face {face} must have 4 components [nx, ny, nz, rotation_rad], "
                    f"got {len(face_normal_orientation_array)}"
                )

            # Normalize the face normal vector (first 3 components)
            face_normal = face_normal_orientation_array[:3]
            face_normal_norm = np.linalg.norm(face_normal)
            if face_normal_norm < 1e-8:
                raise ValueError(f"Face normal for face {face} is too small: {face_normal}")
            face_normal = face_normal / face_normal_norm

            # Create the normalized array with normalized normal and original rotation angle
            face_normal_orientation_array[:3] = face_normal

            poses[face] = CubePose(
                position=np.array(pos),
                face_normal_orientation=face_normal_orientation_array,
                holding_angle=holding_angle
            )
        return poses

    def get_gripper_pose(self, cube_pose, approach_direction: float = 0.0, offset_dist = OFFSET_DIST_HOLD_CUBE, twist_angle = 0.0):
        """
        Convert cube pose to gripper pose.

        Params:
        cube_pose: CubePose object with position and face_normal_orientation [nx, ny, nz, rotation_rad].
        approach_direction: angle in degrees specifying a plane relative to the face normal.
            0°  -> plane containing face normal and reference direction
            90° -> plane perpendicular to face normal (infinite possible directions in this plane)
            180° -> opposite plane from 0°
        offset_dist: distance from cube center to gripper contact point (in mm).
        twist_angle: additional twist angle (radians) around the approach axis (face-normal).

        Returns:
        GripperPose object with position and quaternion orientation.

        Logic:
        1. approach_direction defines which PLANE to approach from relative to the face normal
        2. rotation_rad from face_normal_orientation unambiguously selects which DIRECTION
           within that plane should be used

        For example, when approach_direction=90° (perpendicular plane):
        - The infinite possible directions in that plane are parameterized by rotation_rad
        - rotation_rad=0 gives one specific direction
        - rotation_rad=π/2 gives the perpendicular direction within the plane
        - This ensures each cube pose has exactly one defined gripper approach
        """
        # Get the normalized face normal and rotation angle
        face_normal_orientation = cube_pose.face_normal_orientation.astype(float)
        if len(face_normal_orientation) != 4:
            raise ValueError(f"face_normal_orientation must have 4 components, got {len(face_normal_orientation)}")

        n = face_normal_orientation[:3]
        n = n / np.linalg.norm(n)
        rotation_rad = float(face_normal_orientation[3])  # rotation around face normal axis

        # approach_direction defines which plane to approach from (in degrees)
        approach_angle_rad = np.deg2rad(float(approach_direction) % 360.0)

        # Find a stable reference direction in world frame
        # This will be our primary axis for building perpendicular directions
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, n)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])

        # First perpendicular direction to face normal
        p1 = np.cross(n, ref)
        p1_norm = np.linalg.norm(p1)
        if p1_norm < 1e-8:
            ref = np.array([0.0, 1.0, 0.0])
            p1 = np.cross(n, ref)
            p1_norm = np.linalg.norm(p1)
            if p1_norm < 1e-8:
                if abs(n[0]) < 0.9:
                    p1 = np.array([1.0, 0.0, 0.0]) - n * n[0]
                else:
                    p1 = np.array([0.0, 1.0, 0.0]) - n * n[1]
                p1_norm = np.linalg.norm(p1)
        p1 = p1 / p1_norm

        # Second perpendicular direction (orthogonal to both n and p1)
        p2 = np.cross(n, p1)
        p2 = p2 / np.linalg.norm(p2)

        # Now interpret rotation_rad:
        # We want to select a specific direction from all possible directions at approach_angle_rad
        # We do this by rotating around the face normal by rotation_rad

        # Build a reference vector in the plane perpendicular to n
        # This reference will be rotated by rotation_rad to select the exact approach direction
        ref_perp = p1  # Start with first perpendicular basis vector

        # Build the approach direction by:
        # 1. Creating a vector in the plane defined by approach_angle_rad
        # 2. Using rotation_rad to select which direction in that plane

        # When approach_angle_rad = 0: direction is along n
        # When approach_angle_rad = 90: direction is in the perpendicular plane (anywhere)
        # When approach_angle_rad = 180: direction is along -n

        # Compute base direction (before applying rotation_rad selection)
        u_base = np.cos(approach_angle_rad) * n + np.sin(approach_angle_rad) * ref_perp
        u_base = u_base / np.linalg.norm(u_base)

        # Now apply rotation_rad to select the exact direction within the plane
        # Rotate around the face normal by rotation_rad
        q_rotation = quaternion_from_axis_angle(n, rotation_rad)
        u_world = rotate_vector_by_quaternion(u_base, q_rotation)
        u_world = u_world / np.linalg.norm(u_world)

        # offset distance convert mm->m
        offset_m = offset_dist / 1000.0

        # position of gripper contact point (in world frame)
        pos = cube_pose.position + u_world * offset_m

        # desired gripper forward (local z) is opposite to the approach vector
        desired_z = -u_world
        desired_z = desired_z / np.linalg.norm(desired_z)

        # Choose a stable reference not parallel to desired_z for constructing axes
        ref2 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref2, desired_z)) > 0.9:
            ref2 = np.array([0.0, 1.0, 0.0])

        # Build orthonormal basis
        x_axis = np.cross(ref2, desired_z)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            ref2 = np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(ref2, desired_z)
            x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / x_norm
        y_axis = np.cross(desired_z, x_axis)

        # Apply twist around desired_z by rotating x,y in their plane
        # twist_angle is any additional twist requested by the caller
        # PLUS rotation_rad which defines the gripper's orientation when grasping
        combined_twist = twist_angle + rotation_rad
        ct = np.cos(combined_twist)
        st = np.sin(combined_twist)
        x_rot = ct * x_axis + st * y_axis
        y_rot = -st * x_axis + ct * y_axis
        z_rot = desired_z

        # Construct rotation matrix with columns = gripper local axes in world frame
        R = np.vstack((x_rot, y_rot, z_rot)).T  # shape (3,3)

        # Convert rotation matrix to quaternion [x, y, z, w]
        trace = np.trace(R)
        if trace > 0.0:
            S = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        else:
            if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
                qw = (R[2,1] - R[1,2]) / S
                qx = 0.25 * S
                qy = (R[0,1] + R[1,0]) / S
                qz = (R[0,2] + R[2,0]) / S
            elif R[1,1] > R[2,2]:
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
                qw = (R[0,2] - R[2,0]) / S
                qx = (R[0,1] + R[1,0]) / S
                qy = 0.25 * S
                qz = (R[1,2] + R[2,1]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
                qw = (R[1,0] - R[0,1]) / S
                qx = (R[0,2] + R[2,0]) / S
                qy = (R[1,2] + R[2,1]) / S
                qz = 0.25 * S

        q = np.array([qx, qy, qz, qw])
        q = quaternion_normalize(q)

        gripper_pose = GripperPose(position=pos, quaternion=q)

        return gripper_pose

    def get_cube_pose_for_face(self, face):
        """Get the cube pose when accessing a specific face."""
        if face not in self.cube_poses:
            raise ValueError(f"Invalid face: {face}")
        return self.cube_poses[face]

    # -------------------------------
    # MTC HELPER FUNCTIONS
    # -------------------------------
    def _create_motion_step(self, motion_type: int, robot_name: str, gripper_pose: GripperPose = None) -> MotionStep:
        """Create a MotionStep message for MTC execution.

        Args:
            motion_type: MotionStep.MOVE_J, MOVE_L, GRIPPER_OPEN, or GRIPPER_CLOSE
            robot_name: 'robot1' or 'robot2'
            gripper_pose: GripperPose object (required for MOVE_J/MOVE_L)

        Returns:
            MotionStep message
        """
        step = MotionStep()
        step.motion_type = motion_type
        step.robot_name = robot_name

        if gripper_pose is not None:
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position.x = float(gripper_pose.position[0])
            pose.pose.position.y = float(gripper_pose.position[1])
            pose.pose.position.z = float(gripper_pose.position[2])
            pose.pose.orientation.x = float(gripper_pose.quaternion[0])
            pose.pose.orientation.y = float(gripper_pose.quaternion[1])
            pose.pose.orientation.z = float(gripper_pose.quaternion[2])
            pose.pose.orientation.w = float(gripper_pose.quaternion[3])
            step.target_pose = pose

        return step

    def _move_j_step(self, robot_name: str, gripper_pose: GripperPose) -> MotionStep:
        """Create a MoveJ (PTP) motion step."""
        return self._create_motion_step(MotionStep.MOVE_J, robot_name, gripper_pose)

    def _move_l_step(self, robot_name: str, gripper_pose: GripperPose) -> MotionStep:
        """Create a MoveL (Linear) motion step."""
        return self._create_motion_step(MotionStep.MOVE_L, robot_name, gripper_pose)

    def _gripper_open_step(self, robot_name: str) -> MotionStep:
        """Create a gripper open step."""
        return self._create_motion_step(MotionStep.GRIPPER_OPEN, robot_name)

    def _gripper_close_step(self, robot_name: str) -> MotionStep:
        """Create a gripper close step."""
        return self._create_motion_step(MotionStep.GRIPPER_CLOSE, robot_name)

    async def _execute_mtc_sequence(self, steps: list, max_attempts: int = 3) -> bool:
        """Execute a motion sequence using the MTC action server.

        Args:
            steps: List of MotionStep messages
            max_attempts: Maximum planning attempts

        Returns:
            True if execution succeeded, False otherwise
        """
        if not self.mtc_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MTC action server not available")
            return False

        goal = ExecuteMotionSequence.Goal()
        goal.motion_steps = steps
        goal.max_planning_attempts = max_attempts
        goal.validate_only = False

        self.get_logger().debug(f"MTC sequence: {len(steps)} steps")

        goal_handle = await self.mtc_client.send_goal_async(goal)
        if not goal_handle.accepted:
            self.get_logger().error("MTC goal rejected")
            return False

        result = await goal_handle.get_result_async()

        if result.result.success:
            self.get_logger().debug(f"MTC result: OK - {result.result.message}")
            return True
        else:
            msg = result.result.message
            if result.result.failed_step_index >= 0:
                msg += f" (step {result.result.failed_step_index})"
            self.get_logger().error(f"MTC failed: {msg}")
            return False

    # -------------------------------
    # HAND OVER ACTION (MTC-based)
    # -------------------------------
    async def execute_hand_over(self, goal_handle):
        targeted_cube_spinning_robot = goal_handle.request.cube_spinning_robot

        # Wenn kein Roboter angegeben wurde, zum anderen wechseln
        if (targeted_cube_spinning_robot == 0):
            targeted_cube_spinning_robot = (self.cube_spinning_robot) % 2 + 1  # Wechsel zum anderen Roboter

        if (targeted_cube_spinning_robot == self.cube_spinning_robot):
            self.get_logger().debug("HandOver: no change needed")
            result = HandOver.Result()
            result.success = True
            result.cube_spinning_robot = self.cube_spinning_robot
            goal_handle.abort()
            return result

        self.get_logger().info(f"[HandOver] Started: robot{self.cube_spinning_robot} -> robot{targeted_cube_spinning_robot}")

        # Define the handover sequence
        old_spinning_robot = self._get_robot_name(self.cube_spinning_robot)
        new_spinning_robot = self._get_robot_name(targeted_cube_spinning_robot)
        if self.cube_spinning_robot == 1:
            old_spinning_robot_approach_direction = 0.0
            new_spinning_robot_approach_direction = 180.0
        else:
            old_spinning_robot_approach_direction = 180.0
            new_spinning_robot_approach_direction = 0.0

        # Build motion sequence
        steps = []

        # Old robot moves to handover pre-position with cube
        old_pre = self.get_gripper_pose(self.handover_pose, approach_direction=old_spinning_robot_approach_direction, offset_dist=OFFSET_DIST_PRE_TARGET)
        steps.append(self._move_j_step(old_spinning_robot, old_pre))

        # New robot approaches handover position
        new_pre = self.get_gripper_pose(self.handover_pose, approach_direction=new_spinning_robot_approach_direction, offset_dist=OFFSET_DIST_PRE_TARGET, twist_angle=np.pi/2)
        steps.append(self._move_j_step(new_spinning_robot, new_pre))

        # Old robot moves to handover position with cube
        old_target = self.get_gripper_pose(self.handover_pose, approach_direction=old_spinning_robot_approach_direction, offset_dist=OFFSET_DIST_HOLD_CUBE)
        steps.append(self._move_l_step(old_spinning_robot, old_target))

        # New robot moves linearly to grasp cube
        new_target = self.get_gripper_pose(self.handover_pose, approach_direction=new_spinning_robot_approach_direction, offset_dist=OFFSET_DIST_HOLD_CUBE, twist_angle=np.pi/2)
        steps.append(self._move_l_step(new_spinning_robot, new_target))

        # DEBUG: Log the target poses with full precision
        self.get_logger().info(f"[HandOver DEBUG] {old_spinning_robot} target: pos=[{old_target.position[0]:.6f}, {old_target.position[1]:.6f}, {old_target.position[2]:.6f}]")
        self.get_logger().info(f"[HandOver DEBUG] {new_spinning_robot} target: pos=[{new_target.position[0]:.6f}, {new_target.position[1]:.6f}, {new_target.position[2]:.6f}]")

        # Gripper exchange: new robot grabs, old robot releases
        steps.append(self._gripper_close_step(old_spinning_robot))
        steps.append(self._gripper_open_step(new_spinning_robot))

        # Old robot retracts
        old_post = self.get_gripper_pose(self.handover_pose, approach_direction=old_spinning_robot_approach_direction, offset_dist=OFFSET_DIST_PRE_TARGET)
        steps.append(self._move_l_step(old_spinning_robot, old_post))

        # New robot retracts
        new_post = self.get_gripper_pose(self.handover_pose, approach_direction=new_spinning_robot_approach_direction, offset_dist=OFFSET_DIST_PRE_TARGET, twist_angle=np.pi/2)
        steps.append(self._move_l_step(new_spinning_robot, new_post))

        # Execute the sequence
        success = await self._execute_mtc_sequence(steps)

        if success:
            self.cube_spinning_robot = targeted_cube_spinning_robot

        result = HandOver.Result()
        result.success = success
        result.cube_spinning_robot = self.cube_spinning_robot

        self.get_logger().info(f"[HandOver] {'OK' if success else 'FAILED'}")
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # -------------------------------
    # ROTATE FACE ACTION (MTC-based)
    # -------------------------------
    async def execute_rotate_face(self, goal_handle):
        move = goal_handle.request.move
        face, angle = decode_kociemba_move(move)

        self.get_logger().info(f"[RotateFace] Started: {move} ({face} {angle})")

        # Beispiel: Vor dem Drehen muss evtl. umgegriffen werden
        if face not in ROBOT_FACES[self.cube_spinning_robot-1]:
            self.get_logger().debug("[RotateFace] HandOver required")
            await self.call_hand_over()

        # Get cube pose for this face
        cube_pose = self.get_cube_pose_for_face(face)
        self.get_logger().debug(f"[RotateFace] face {face} pos {cube_pose.position}")

        # Determine which robot holds and which spins
        spinning_robot = self._get_robot_name(self.cube_spinning_robot)
        holding_robot = self._get_robot_name(3 - self.cube_spinning_robot)  # other robot

        # Build motion sequence
        steps = []

        # Holding robot moves to hold the cube
        cube_holding_target = self.get_gripper_pose(cube_pose, approach_direction=cube_pose.holding_angle, offset_dist=OFFSET_DIST_HOLD_CUBE)
        steps.append(self._move_j_step(holding_robot, cube_holding_target))

        # Spinning robot approaches the face
        cube_spinning_pre = self.get_gripper_pose(cube_pose, approach_direction=0.0, offset_dist=OFFSET_DIST_PRE_TARGET)
        steps.append(self._move_j_step(spinning_robot, cube_spinning_pre))

        # Spinning robot moves linearly to grasp position
        cube_spinning_start = self.get_gripper_pose(cube_pose, approach_direction=0.0, offset_dist=OFFSET_DIST_SPIN_CUBE)
        steps.append(self._move_l_step(spinning_robot, cube_spinning_start))

        # Gripper close: spinning robot grabs the face to rotate
        steps.append(self._gripper_close_step(spinning_robot))

        # Spinning robot rotates the face (reorient)
        cube_spinning_end = self.get_gripper_pose(cube_pose, approach_direction=0.0, offset_dist=OFFSET_DIST_SPIN_CUBE, twist_angle=np.radians(angle))
        steps.append(self._move_l_step(spinning_robot, cube_spinning_end))

        # Gripper open: spinning robot releases the face after rotation
        steps.append(self._gripper_open_step(spinning_robot))

        # Spinning robot retracts
        cube_spinning_post = self.get_gripper_pose(cube_pose, approach_direction=0.0, offset_dist=OFFSET_DIST_PRE_TARGET, twist_angle=np.radians(angle))
        steps.append(self._move_l_step(spinning_robot, cube_spinning_post))

        # Execute the sequence
        success = await self._execute_mtc_sequence(steps)

        result = RotateFace.Result()
        result.success = success
        self.get_logger().info(f"[RotateFace] {'OK' if success else 'FAILED'}: {move}")
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # -------------------------------
    # TAKE UP CUBE ACTION (MTC-based)
    # -------------------------------
    async def execute_take_up_cube(self, goal_handle):
        """
        Take up the cube from the rest position.
        Robot 2 always takes up the cube.
        After taking up, Robot 1 becomes the spinning robot.
        """
        self.get_logger().info("[TakeUpCube] Started")

        robot2_name = self._get_robot_name(2)

        # Build motion sequence
        steps = []

        # Robot 2 approaches rest position from top
        pre_grasp = self.get_gripper_pose(
            self.rest_pose,
            approach_direction=0.0,
            offset_dist=OFFSET_DIST_PRE_TARGET
        )
        steps.append(self._move_j_step(robot2_name, pre_grasp))

        # Robot 2 moves linearly to grasp the cube
        grasp = self.get_gripper_pose(
            self.rest_pose,
            approach_direction=0.0,
            offset_dist=OFFSET_DIST_HOLD_CUBE
        )
        steps.append(self._move_l_step(robot2_name, grasp))

        # Close gripper
        steps.append(self._gripper_close_step(robot2_name))

        # Create above-rest pose
        above_rest_pose = CubePose(
            position=self.rest_pose.position + np.array([0.0, 0.0, OFFSET_DIST_TAKE_CUBE/1000.0]),
            face_normal_orientation=self.rest_pose.face_normal_orientation
        )

        post_grasp = self.get_gripper_pose(
            above_rest_pose,
            approach_direction=0.0,
            offset_dist=OFFSET_DIST_HOLD_CUBE
        )
        steps.append(self._move_l_step(robot2_name, post_grasp))

        # Execute the sequence
        success = await self._execute_mtc_sequence(steps)

        if success:
            # After taking up, Robot 1 becomes the spinning robot (Robot 2 holds)
            self.cube_spinning_robot = 1

        result = TakeUpCube.Result()
        result.success = success

        self.get_logger().info(f"[TakeUpCube] {'OK' if success else 'FAILED'}")
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    # -------------------------------
    # PUT DOWN CUBE ACTION (MTC-based)
    # -------------------------------
    async def execute_put_down_cube(self, goal_handle):
        """
        Put down the cube to the rest position.
        Robot 2 must be holding the cube (cube_spinning_robot = 1).
        """
        self.get_logger().info("[PutDownCube] Started")

        robot2_name = self._get_robot_name(2)

        # Check if Robot 2 is holding the cube
        if self.cube_spinning_robot != 1:
            self.get_logger().debug("[PutDownCube] HandOver required")
            await self.call_hand_over()

            if self.cube_spinning_robot != 1:
                self.get_logger().info("[PutDownCube] FAILED: HandOver failed")
                result = PutDownCube.Result()
                result.success = False
                goal_handle.abort()
                return result

        # Build motion sequence
        steps = []

        # Create above-rest pose
        above_rest_pose = CubePose(
            position=self.rest_pose.position + np.array([0.0, 0.0, OFFSET_DIST_TAKE_CUBE/1000.0]),
            face_normal_orientation=self.rest_pose.face_normal_orientation
        )

        pre_pos = self.get_gripper_pose(
            above_rest_pose,
            approach_direction=0.0,
            offset_dist=OFFSET_DIST_HOLD_CUBE
        )
        steps.append(self._move_j_step(robot2_name, pre_pos))

        # Robot 2 moves linearly to put down position
        put_down = self.get_gripper_pose(
            self.rest_pose,
            approach_direction=0.0,
            offset_dist=OFFSET_DIST_HOLD_CUBE
        )
        steps.append(self._move_l_step(robot2_name, put_down))

        # Open gripper
        steps.append(self._gripper_open_step(robot2_name))

        # Robot 2 retracts
        post_pos = self.get_gripper_pose(
            self.rest_pose,
            approach_direction=0.0,
            offset_dist=OFFSET_DIST_PRE_TARGET
        )
        steps.append(self._move_l_step(robot2_name, post_pos))

        # Execute the sequence
        success = await self._execute_mtc_sequence(steps)

        result = PutDownCube.Result()
        result.success = success

        self.get_logger().info(f"[PutDownCube] {'OK' if success else 'FAILED'}")
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    # -------------------------------
    # PRESENT CUBE FACE ACTION (MTC-based)
    # -------------------------------
    async def execute_present_cube_face(self, goal_handle):
        """
        Present a specific cube face to the camera for scanning.
        """
        face = goal_handle.request.face
        self.get_logger().info(f"[PresentCubeFace] Started: face {face}")

        success = True

        try:
            # Validate face
            if face not in CUBE_PRESENT_POSES:
                raise ValueError(f"Unknown face: {face}")

            # Get the presentation pose for this face
            present_pose_data = CUBE_PRESENT_POSES[face]
            present_pos, face_normal_orientation, holding_angle = present_pose_data

            present_pose = CubePose(
                position=np.array(present_pos),
                face_normal_orientation=np.array(face_normal_orientation, dtype=float),
                holding_angle=float(holding_angle[0]) if isinstance(holding_angle, (list, tuple)) else float(holding_angle)
            )

            # Determine which robot should hold for this face
            if face in ROBOT_FACES[0]:  # face is in robot 1's spinning list
                holding_robot_id = 2  # robot 2 holds
            elif face in ROBOT_FACES[1]:  # face is in robot 2's spinning list
                holding_robot_id = 1  # robot 1 holds
            else:
                raise ValueError(f"Face {face} not found in ROBOT_FACES")
            holding_robot_name = self._get_robot_name(holding_robot_id)

            self.get_logger().debug(f"[PresentCubeFace] held by {holding_robot_name}")

            # Perform handover if necessary
            if holding_robot_id == 1:
                if self.cube_spinning_robot != 2:
                    self.get_logger().debug("[PresentCubeFace] HandOver required")
                    await self.call_hand_over()
                    if self.cube_spinning_robot != 2:
                        raise RuntimeError("Handover failed")
            else:  # holding_robot_id == 2
                if self.cube_spinning_robot != 1:
                    self.get_logger().debug("[PresentCubeFace] HandOver required")
                    await self.call_hand_over()
                    if self.cube_spinning_robot != 1:
                        raise RuntimeError("Handover failed")

            # Build motion sequence (single move)
            steps = []
            present_target = self.get_gripper_pose(
                present_pose,
                approach_direction=present_pose.holding_angle,
                offset_dist=OFFSET_DIST_HOLD_CUBE
            )
            steps.append(self._move_j_step(holding_robot_name, present_target))

            # Execute the sequence
            success = await self._execute_mtc_sequence(steps)

        except Exception as e:
            self.get_logger().error(f"PresentCubeFace error: {str(e)}")
            success = False

        result = PresentCubeFace.Result()
        result.success = success

        self.get_logger().info(f"[PresentCubeFace] {'OK' if success else 'FAILED'}: {face}")
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    # -------------------------------
    # interne Hilfsfunktionen
    # -------------------------------
    def _get_robot_name(self, robot_num):
        """Convert robot number (1 or 2) to robot name string."""
        return f"robot{robot_num}"

    async def call_hand_over(self):
        """Interner Aufruf des HandOver Actionsservers."""

        server_ready = self.hand_over_client.wait_for_server(timeout_sec=5.0)
        if not server_ready:
            self.get_logger().error("HandOver action server not available")
            return

        goal = HandOver.Goal()

        goal_future = self.hand_over_client.send_goal_async(goal)
        goal_handle = await goal_future

        result_future = goal_handle.get_result_async()
        result = await result_future

        if not result.result.success:
            self.get_logger().error("HandOver internal call failed")
            raise Exception("HandOver failed")


def main(args=None):
    rclpy.init(args=args)
    node = CubeMotionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
