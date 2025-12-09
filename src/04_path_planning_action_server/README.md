# Path Planning Action Server for Dual UR Robots

This package provides MoveL (linear/Cartesian) and MoveJ (joint space) path planning actions for two UR robots using MoveIt2. All target poses are specified relative to the global `world` frame.

## Package Structure

```
04_path_planning_action_server/
├── path_planning_interfaces/     # Action message definitions
│   └── action/
│       ├── MoveL.action         # Linear motion action
│       └── MoveJ.action         # Joint space motion action
└── path_planning_server/         # Action server implementation
    ├── config/
    │   ├── path_planning_params.yaml
    │   └── moveit/              # MoveIt2 configuration
    ├── launch/
    │   ├── path_planning_server.launch.py
    │   ├── moveit.launch.py
    │   └── dual_robot_planning.launch.py
    └── path_planning_server/
        ├── path_planning_action_server.py
        └── example_client.py
```

## Prerequisites

1. **Rebuild the Docker container** (to include MoveIt2 dependencies):
   ```bash
   # In the .devcontainer directory
   docker-compose build
   ```

2. **Build the workspace**:
   ```bash
   cd /root/ros2_ws
   colcon build --symlink-install
   source install/setup.bash
   ```

## Usage

### Step 1: Start the Robot Hardware/Simulation

First, start the dual robot system using the multi-robot bringup:

```bash
# For simulation (mock hardware)
ros2 launch 01_rsy_bringup multi_robot.launch.py use_mock_hardware:=true

# For real robots
ros2 launch 01_rsy_bringup multi_robot.launch.py \
    robot1_ip:=192.168.1.101 \
    robot2_ip:=192.168.1.102
```

### Step 2: Start the Path Planning System

Launch the complete path planning system (MoveIt2 + Action Server):

```bash
ros2 launch path_planning_server dual_robot_planning.launch.py
```

Or launch components separately:

```bash
# Terminal 1: Start MoveIt2 move_group nodes
ros2 launch path_planning_server moveit.launch.py

# Terminal 2: Start the action server
ros2 launch path_planning_server path_planning_server.launch.py
```

### Step 3: Send Motion Commands

#### Using the Example Client

```bash
ros2 run path_planning_server example_client.py
```

#### Using Command Line (ros2 action)

**MoveJ (Joint Space Motion)**:
```bash
ros2 action send_goal /move_j path_planning_interfaces/action/MoveJ "{
  robot_name: 'robot1',
  target_pose: {
    header: {frame_id: 'world'},
    pose: {
      position: {x: 0.3, y: 0.2, z: 0.3},
      orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}
    }
  },
  velocity_scaling: 0.3,
  acceleration_scaling: 0.3,
  execute: true
}"
```

**MoveL (Linear/Cartesian Motion)**:
```bash
ros2 action send_goal /move_l path_planning_interfaces/action/MoveL "{
  robot_name: 'robot2',
  target_pose: {
    header: {frame_id: 'world'},
    pose: {
      position: {x: -0.3, y: 0.1, z: 0.25},
      orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}
    }
  },
  velocity_scaling: 0.2,
  acceleration_scaling: 0.2,
  execute: true
}"
```

#### Using Python Code

```python
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from path_planning_interfaces.action import MoveL, MoveJ

# Initialize ROS2
rclpy.init()
node = rclpy.create_node('my_client')

# Create action client
movel_client = ActionClient(node, MoveL, 'move_l')
movel_client.wait_for_server()

# Create goal
goal = MoveL.Goal()
goal.robot_name = 'robot1'
goal.target_pose = PoseStamped()
goal.target_pose.header.frame_id = 'world'
goal.target_pose.pose.position.x = 0.3
goal.target_pose.pose.position.y = 0.2
goal.target_pose.pose.position.z = 0.3
goal.target_pose.pose.orientation.w = 1.0
goal.velocity_scaling = 0.3
goal.acceleration_scaling = 0.3
goal.execute = True  # False for plan-only

# Send goal
future = movel_client.send_goal_async(goal)
rclpy.spin_until_future_complete(node, future)
```

## Action Interfaces

### MoveL.action (Linear/Cartesian Motion)

**Goal**:
- `geometry_msgs/PoseStamped target_pose` - Target pose in world frame
- `string robot_name` - Robot identifier ("robot1" or "robot2")
- `float64 velocity_scaling` - Speed (0.0-1.0)
- `float64 acceleration_scaling` - Acceleration (0.0-1.0)
- `bool execute` - True: plan and execute, False: plan only

**Result**:
- `bool success` - Planning/execution success
- `string message` - Status/error message
- `trajectory_msgs/JointTrajectory planned_trajectory` - Planned trajectory
- `float64 planning_time` - Time to plan (seconds)
- `float64 path_length` - Cartesian path length (meters)

**Feedback**:
- `float64 progress_percentage` - Progress (0-100%)
- `string status` - Current status
- `geometry_msgs/Pose current_pose` - Current end-effector pose

### MoveJ.action (Joint Space Motion)

**Goal**: Same as MoveL

**Result**:
- Same as MoveL, except:
- `float64 joint_distance` - Total joint distance (radians)

**Feedback**:
- Same as MoveL, except:
- `float64[] current_joint_positions` - Current joint positions

## Available Topics/Services/Actions

After launching, the following are available:

**Actions**:
- `/move_l` - MoveL action (global)
- `/move_j` - MoveJ action (global)
- `/robot1/move_l`, `/robot1/move_j` - Robot 1 specific
- `/robot2/move_l`, `/robot2/move_j` - Robot 2 specific

**MoveIt2 Services** (per robot namespace):
- `/<namespace>/compute_ik` - Inverse kinematics
- `/<namespace>/plan_kinematic_path` - Motion planning
- `/<namespace>/compute_cartesian_path` - Cartesian path planning

## Configuration

Edit `config/path_planning_params.yaml` to change:

```yaml
path_planning_action_server:
  ros__parameters:
    world_frame: "world"
    ground_height: 0.0
    ground_safety_margin: 0.02

    robot1:
      name: "robot1"
      namespace: "robot1"
      ur_type: "ur3e"

    robot2:
      name: "robot2"
      namespace: "robot2"
      ur_type: "ur3e"

    planning_time: 5.0
    num_planning_attempts: 10
```

## Launch Parameters

```bash
ros2 launch path_planning_server dual_robot_planning.launch.py \
    world_frame:=world \
    robot1_namespace:=robot1 \
    robot2_namespace:=robot2 \
    robot1_ur_type:=ur3e \
    robot2_ur_type:=ur3e
```

## Troubleshooting

1. **"Service not available"**: Ensure MoveIt2 is running before the action server
2. **"IK failed"**: Target pose may be unreachable or in collision
3. **"Cartesian path only X% achievable"**: Path has collision or singularity
4. **"Below ground level"**: Increase target Z position above safety margin
