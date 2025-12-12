#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
import numpy as np
from dataclasses import dataclass

from rsy_cube_motion.action import RotateFace, HandOver

# Roboter 1 dreht U, F, D; Roboter 2 dreht L, B, R
ROBOT_FACES = ["U", "F", "D"], ["L", "B", "R"]

# Compact, editable definition of cube-access poses (positions + face normal vector).
# Adjust these values later in real-world testing.
CUBE_POSE_DEFS = {
    # face: (position_xyz, face_normal_vector)
    "U": ([0.2, 0.2, 0.3], [0.0, 0.0, 1.0]),
    "D": ([0.2, 0.2, 0.3], [0.0, 0.0, -1.0]),
    "F": ([0.2, 0.2, 0.3], [0.0, 1.0, 0.0]),
    "B": ([0.2, 0.2, 0.3], [0.0, -1.0, 0.0]),
    "L": ([0.2, 0.2, 0.3], [-1.0, 0.0, 0.0]),
    "R": ([0.2, 0.2, 0.3], [1.0, 0.0, 0.0]),
}

HAND_OVER_POSE_DEF = {
    "position": [0.2, 0.2, 0.3],
    "orientation_vector": [1.0, 0.0, 0.0]  # approach axis
}

# Define gripper reference forward vector in gripper's local frame (tool Z-axis?)
GRIPPER_FORWARD_DIRECTION = np.array([0.0, 0.0, 1.0])

# distance (mm) from cube center to gripper contact point
OFFSET_DIST_HOLD_CUBE = 15    # distance when holding the cube (grasps 2 rows of cube)  
OFFSET_DIST_SPIN_CUBE = 25    # distance when spinning the cube (grasps 1 row of cube)  
OFFSET_DIST_PRE_TARGET = 50   # distance when approaching the cube (pre-grasp position)


@dataclass
class CubePose:
    """Represents a cube pose with position and a face-normal orientation vector.
    The orientation_vector is a 3D vector (can be used to build quaternions for spinning).
    """
    position: np.ndarray  # [x, y, z]
    orientation_vector: np.ndarray  # face normal [x, y, z]

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
        # Position: [x, y, z], Orientation: quaternion [x, y, z, w]
        self.cube_poses = self._init_cube_poses()
        
        # Handover pose (neutral position between robots)
        # orientation_vector defines the approach axis for the handover pose.
        self.handover_pose = CubePose(
            position=np.array(HAND_OVER_POSE_DEF["position"]),
            orientation_vector=np.array(HAND_OVER_POSE_DEF["orientation_vector"])  # default approach axis (adjust in real tests)
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

        # interner Client: RotateFace kann HandOver aufrufen
        self.hand_over_client = ActionClient(self, HandOver, 'hand_over')

        self.get_logger().info("Cube Motion Server gestartet.")

    def _init_cube_poses(self):
        """Initialize cube poses for each accessible face.
        Uses the compact CUBE_POSE_DEFS above (position, face-normal vector).
        """
        poses = {}
        for face, (pos, normal) in CUBE_POSE_DEFS.items():
            poses[face] = CubePose(
                position=np.array(pos),
                orientation_vector=np.array(normal)
            )
        return poses

    def get_gripper_pose(self, cube_pose, approach_from_front = False, offset_dist = OFFSET_DIST_HOLD_CUBE, twist_angle = 0.0):
        """
        Convert cube pose to gripper pose.

        Params:
        cube_pose: CubePose object with position and face-normal orientation vector.
        approach_from_front: bool, whether the robot should approach the cube face from the front or back.
            True: robot will grasp the cube face from the front
            False: robot will grasp the cube face from the back (opposite cube face)
        offset_dist: distance from cube center to gripper contact point (in mm).
        twist_angle: additional twist angle (radians) around the approach axis (face-normal).

        Returns:
        GripperPose object with position and quaternion orientation.
        """
        approach = cube_pose.orientation_vector / np.linalg.norm(cube_pose.orientation_vector)

        # Distance from cube center to gripper contact
        offset_dist = offset_dist / 1000.0  # convert mm to meters

        pos = None
        if approach_from_front:
            # Gripper 1: approach from +approach direction
            pos = cube_pose.position + approach * offset_dist
        else:
            # Gripper 2: approach from -approach direction (opposite side)
            pos = cube_pose.position - approach * offset_dist

        # Compute quaternion aligning gripper forward -> approach direction
        q_align = None
        if approach_from_front:
            q_align = quaternion_from_two_vectors(GRIPPER_FORWARD_DIRECTION, approach)
        else:
            q_align = quaternion_from_two_vectors(GRIPPER_FORWARD_DIRECTION, -approach)

        # Apply a twist around the approach axis
        q_twist = quaternion_from_axis_angle(approach, twist_angle)

        # Final orientations: align then twist
        q = quaternion_multiply(q_align, q_twist)

        gripper_pose = GripperPose(position=pos, quaternion=quaternion_normalize(q))

        return gripper_pose

    def get_cube_pose_for_face(self, face):
        """Get the cube pose when accessing a specific face."""
        if face not in self.cube_poses:
            raise ValueError(f"Invalid face: {face}")
        return self.cube_poses[face]

    # -------------------------------
    # HAND OVER ACTION
    # -------------------------------
    async def execute_hand_over(self, goal_handle):
        targeted_cube_spinning_robot = goal_handle.request.cube_spinning_robot

        # Wenn kein Roboter angegeben wurde, zum anderen wechseln
        if (targeted_cube_spinning_robot == 0):
            targeted_cube_spinning_robot = (self.cube_spinning_robot) % 2 +1  # Wechsel zum anderen Roboter

        if (targeted_cube_spinning_robot == self.cube_spinning_robot):
            self.get_logger().warn(">>> [HandOver] Kein Griffwechsel nötig.")
            result = HandOver.Result()
            result.success = True
            result.cube_spinning_robot = self.cube_spinning_robot
            goal_handle.abort()
            return result

        self.get_logger().info(">>> [HandOver] Ausführen des Griffwechsels")

        # Define the handover sequence
        old_spinning_robot_approach_from_front = (self.cube_spinning_robot == 1)
        new_spinning_robot_approach_from_front = (self.cube_spinning_robot == 2)

        old_spinning_robot_target = self.get_gripper_pose(self.handover_pose, approach_from_front=old_spinning_robot_approach_from_front, offset_dist=OFFSET_DIST_HOLD_CUBE)
        # TODO: MoveJ: old spinning robot, old_spinning_robot_target
        new_spinning_robot_pre_target = self.get_gripper_pose(self.handover_pose, approach_from_front=new_spinning_robot_approach_from_front, offset_dist=OFFSET_DIST_PRE_TARGET, twist_angle=np.pi/2)
        # TODO: MoveJ: new spinning robot, new_spinning_robot_pre_target
        new_spinning_robot_target = self.get_gripper_pose(self.handover_pose, approach_from_front=new_spinning_robot_approach_from_front, offset_dist=OFFSET_DIST_HOLD_CUBE, twist_angle=np.pi/2)
        # TODO: MoveL: new spinning robot, new_spinning_robot_target
        # TODO: Gripper close: new spinning robot
        # TODO: Gripper open: old spinning robot
        old_spinning_robot_post_target = self.get_gripper_pose(self.handover_pose, approach_from_front=old_spinning_robot_approach_from_front, offset_dist=OFFSET_DIST_PRE_TARGET)
        # TODO: MoveL: old spinning robot, old_spinning_robot_post_target
        
        self.cube_spinning_robot = targeted_cube_spinning_robot

        result = HandOver.Result()
        result.success = True
        result.cube_spinning_robot = self.cube_spinning_robot

        self.get_logger().info(">>> [HandOver] abgeschlossen.")
        if (result.success):
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result


    # -------------------------------
    # ROTATE FACE ACTION
    # -------------------------------
    async def execute_rotate_face(self, goal_handle):

        move = goal_handle.request.move
        self.get_logger().info(f">>> [RotateFace] Receive move {move}")

        face, angle = decode_kociemba_move(move)

        self.get_logger().info(f">>> [RotateFace] Rotating face {face} by {angle}°")

        # Beispiel: Vor dem Drehen muss evtl. umgegriffen werden
        if face not in ROBOT_FACES[self.cube_spinning_robot-1]:
            self.get_logger().info(">>> HandOver erforderlich")
            await self.call_hand_over()

        # Get cube pose for this face
        cube_pose = self.get_cube_pose_for_face(face)
        
        self.get_logger().info(f">>> [RotateFace] Cube pose for face {face}: {cube_pose.position}")
        self.get_logger().info(f">>> [RotateFace] Face normal for face {face}: {cube_pose.orientation_vector}")

        # Define gripper poses for the rotation sequence
        cube_holding_target = self.get_gripper_pose(cube_pose, approach_from_front=False, offset_dist=OFFSET_DIST_HOLD_CUBE)
        # TODO: MoveJ: cube holding robot, cube_holding_target
        cube_spinning_pre_target = self.get_gripper_pose(cube_pose, approach_from_front=True, offset_dist=OFFSET_DIST_PRE_TARGET)
        # TODO: MoveJ: cube spinning robot, cube_spinning_pre_target
        cube_spinning_start_target = self.get_gripper_pose(cube_pose, approach_from_front=True, offset_dist=OFFSET_DIST_SPIN_CUBE)
        # TODO: MoveL: cube spinning robot, cube_spinning_start_target
        # TODO: Gripper close
        cube_spinning_end_target = self.get_gripper_pose(cube_pose, approach_from_front=True, offset_dist=OFFSET_DIST_SPIN_CUBE, twist_angle=np.radians(angle))
        # TODO: Reorient: cube spinning robot, cube_spinning_end_target
        # TODO: Gripper open
        cube_spinning_post_target = self.get_gripper_pose(cube_pose, approach_from_front=True, offset_dist=OFFSET_DIST_PRE_TARGET, twist_angle=np.radians(angle))
        # TODO: MoveL: cube spinning robot, cube_spinning_post_target

        result = RotateFace.Result()
        result.success = True
        self.get_logger().info(">>> [RotateFace] abgeschlossen.")
        if (result.success):
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # -------------------------------
    # interne Hilfsfunktionen
    # -------------------------------
    async def call_hand_over(self):
        """Interner Aufruf des HandOver Actionsservers."""

        server_ready = self.hand_over_client.wait_for_server(timeout_sec=5.0)
        if not server_ready:
            self.get_logger().error("[HandOver] HandOver action server not available")
            return

        goal = HandOver.Goal()

        goal_future = self.hand_over_client.send_goal_async(goal)
        goal_handle = await goal_future

        result_future = goal_handle.get_result_async()
        result = await result_future

        if not result.result.success:
            self.get_logger().error("[HandOver] Fehler beim Umgreifen!")


def main(args=None):
    rclpy.init(args=args)
    node = CubeMotionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()