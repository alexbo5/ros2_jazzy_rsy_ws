#!/usr/bin/env python3
"""Debug script to compare joint limits from different sources."""

import subprocess
import re
import xml.etree.ElementTree as ET

def run_cmd(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return None

def get_urdf_limits():
    """Extract joint limits from URDF."""
    urdf = run_cmd("ros2 param get /robot_state_publisher robot_description")
    if not urdf:
        print("Could not get robot_description")
        return {}

    # Clean up the output (remove "String value is:" prefix if present)
    if "String value is:" in urdf:
        urdf = urdf.split("String value is:")[1].strip()

    limits = {}
    try:
        root = ET.fromstring(urdf)
        for joint in root.findall('.//joint'):
            name = joint.get('name')
            limit = joint.find('limit')
            if limit is not None and name:
                limits[name] = {
                    'lower': float(limit.get('lower', 0)),
                    'upper': float(limit.get('upper', 0)),
                }
    except ET.ParseError as e:
        print(f"XML parse error: {e}")

    return limits

def get_moveit_limits(joint_name):
    """Get MoveIt planning limits for a joint."""
    min_pos = run_cmd(f"ros2 param get /move_group robot_description_planning.joint_limits.{joint_name}.min_position 2>/dev/null")
    max_pos = run_cmd(f"ros2 param get /move_group robot_description_planning.joint_limits.{joint_name}.max_position 2>/dev/null")

    def parse_value(s):
        if s and "Double value is:" in s:
            return float(s.split(":")[-1].strip())
        return None

    return parse_value(min_pos), parse_value(max_pos)

def get_current_positions():
    """Get current joint positions."""
    output = run_cmd("ros2 topic echo /joint_states --once 2>/dev/null")
    if not output:
        return {}

    positions = {}
    lines = output.split('\n')
    names = []
    pos_values = []

    in_names = False
    in_positions = False

    for line in lines:
        if 'name:' in line:
            in_names = True
            in_positions = False
            continue
        if 'position:' in line:
            in_names = False
            in_positions = True
            continue
        if 'velocity:' in line:
            in_positions = False
            continue

        if in_names and line.strip().startswith('-'):
            names.append(line.strip().lstrip('- '))
        if in_positions and line.strip().startswith('-'):
            try:
                pos_values.append(float(line.strip().lstrip('- ')))
            except:
                pass

    for name, pos in zip(names, pos_values):
        positions[name] = pos

    return positions

def main():
    print("=" * 80)
    print("JOINT LIMITS DEBUG")
    print("=" * 80)

    # Get URDF limits
    print("\nFetching URDF limits...")
    urdf_limits = get_urdf_limits()

    # Get current positions
    print("Fetching current joint positions...")
    current_positions = get_current_positions()

    # Joints to check
    joints = [
        'robot1_shoulder_pan_joint',
        'robot1_shoulder_lift_joint',
        'robot1_elbow_joint',
        'robot1_wrist_1_joint',
        'robot1_wrist_2_joint',
        'robot1_wrist_3_joint',
        'robot2_shoulder_pan_joint',
        'robot2_shoulder_lift_joint',
        'robot2_elbow_joint',
        'robot2_wrist_1_joint',
        'robot2_wrist_2_joint',
        'robot2_wrist_3_joint',
    ]

    print("\n" + "=" * 80)
    print(f"{'Joint':<35} {'URDF Limits':<25} {'MoveIt Limits':<25} {'Current':<12} {'Status'}")
    print("=" * 80)

    for joint in joints:
        # URDF limits
        urdf = urdf_limits.get(joint, {})
        urdf_lower = urdf.get('lower', 'N/A')
        urdf_upper = urdf.get('upper', 'N/A')
        urdf_str = f"[{urdf_lower:.3f}, {urdf_upper:.3f}]" if isinstance(urdf_lower, float) else "N/A"

        # MoveIt limits
        moveit_min, moveit_max = get_moveit_limits(joint)
        if moveit_min is not None and moveit_max is not None:
            moveit_str = f"[{moveit_min:.3f}, {moveit_max:.3f}]"
        else:
            moveit_str = "N/A"

        # Current position
        current = current_positions.get(joint, None)
        current_str = f"{current:.3f}" if current is not None else "N/A"

        # Check if out of bounds
        status = "OK"
        if isinstance(urdf_lower, float) and current is not None:
            if current < urdf_lower or current > urdf_upper:
                status = "OUT OF URDF BOUNDS!"

        print(f"{joint:<35} {urdf_str:<25} {moveit_str:<25} {current_str:<12} {status}")

    print("=" * 80)
    print("\nNote: CheckStartStateBounds uses URDF limits, not MoveIt limits!")
    print("If URDF limits don't match MoveIt limits, that's the problem.")

if __name__ == "__main__":
    main()
