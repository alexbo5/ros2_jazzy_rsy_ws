# Motion step types
uint8 MOVE_J=0        # Point-to-point (PTP) motion
uint8 MOVE_L=1        # Linear (LIN) motion
uint8 GRIPPER_OPEN=2  # Open gripper
uint8 GRIPPER_CLOSE=3 # Close gripper

# Motion type (one of the constants above)
uint8 motion_type

# Robot to move ("robot1" or "robot2")
string robot_name

# Target pose for MOVE_J/MOVE_L (ignored for gripper commands)
geometry_msgs/PoseStamped target_pose
