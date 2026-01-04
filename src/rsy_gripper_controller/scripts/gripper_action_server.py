#!/usr/bin/env python3
"""
ROS2 Jazzy Robotiq Gripper Action Server

This node provides a service interface to control Robotiq grippers via TCP/IP.
Supports multiple robots through prefix-based naming (e.g., robot1_gripper_service).
Supports mock hardware mode for testing without physical grippers.
"""

import rclpy
from rclpy.node import Node
from rsy_gripper_controller.srv import RobotiqGripper
import socket
import time
import re


class GripperActionServer(Node):
    """Service server node for controlling a Robotiq gripper."""

    def __init__(self, node_name: str = 'gripper_action_server'):
        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter('ip_address', '')
        self.declare_parameter('port', 63352)
        self.declare_parameter('timeout', 3.0)
        self.declare_parameter('robot_prefix', '')
        self.declare_parameter('use_mock_hardware', False)

        # Get parameters
        self.ip_address = self.get_parameter('ip_address').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value
        robot_prefix = self.get_parameter('robot_prefix').get_parameter_value().string_value
        self.use_mock_hardware = self.get_parameter('use_mock_hardware').get_parameter_value().bool_value

        # Track mock gripper state
        self.mock_position = 0  # 0 = open, 255 = closed

        # Validate IP address (only required if not in mock mode)
        if not self.use_mock_hardware and not self.ip_address:
            self.get_logger().error('ip_address parameter is required when not using mock hardware!')
            raise ValueError('ip_address parameter must be set')

        # Build service name with prefix
        if robot_prefix:
            service_name = f'{robot_prefix}_robotiq_gripper'
        else:
            service_name = 'robotiq_gripper'

        if self.use_mock_hardware:
            self.get_logger().info('Running in MOCK HARDWARE mode - no real gripper connection')
        else:
            self.get_logger().info(f'Gripper IP: {self.ip_address}:{self.port}')

        self.get_logger().info(f'Service name: {service_name}')

        # Create service
        self.service = self.create_service(
            RobotiqGripper,
            service_name,
            self.execute_service_callback
        )

        self.get_logger().info('Gripper action server is ready')

    def execute_service_callback(self, request, response):
        """Handle incoming service requests."""
        response.success = False
        response.value = -1
        response.average = -1.0

        action = request.action.upper()

        # Validate action
        if action not in ['OPEN', 'CLOSE']:
            response.message = f'ERROR: Invalid action "{request.action}". Valid commands: OPEN, CLOSE'
            self.get_logger().warn(response.message)
            return response

        # Use mock hardware if enabled
        if self.use_mock_hardware:
            return self._execute_mock_command(action, response)

        # Real hardware: Create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect((self.ip_address, self.port))
        except socket.timeout:
            response.message = f'ERROR: TCP-IP socket connection timed out to {self.ip_address}:{self.port}'
            self.get_logger().error(response.message)
            return response
        except ConnectionRefusedError:
            response.message = f'ERROR: TCP-IP socket connection refused by {self.ip_address}:{self.port}'
            self.get_logger().error(response.message)
            return response
        except Exception as e:
            response.message = f'ERROR: Failed to connect: {str(e)}'
            self.get_logger().error(response.message)
            return response

        try:
            if action == 'CLOSE':
                response = self._execute_gripper_command(sock, 255, 'CLOSE', response)
            elif action == 'OPEN':
                response = self._execute_gripper_command(sock, 0, 'OPEN', response)
        finally:
            sock.close()

        return response

    def _execute_mock_command(self, action, response):
        """Execute a mock gripper command (no real hardware)."""
        # Simulate gripper movement delay
        time.sleep(0.5)

        if action == 'CLOSE':
            self.mock_position = 255
        else:  # OPEN
            self.mock_position = 0

        percentage = round((float(self.mock_position) / 255.0) * 100.0, 2)

        response.success = True
        response.value = self.mock_position
        response.average = percentage
        response.message = (
            f'[MOCK] {action} command executed successfully. '
            f'Gripper is {percentage}% closed (raw: {self.mock_position}/255)'
        )
        self.get_logger().info(response.message)

        return response

    def _execute_gripper_command(self, sock, position, action_name, response):
        """Execute a gripper position command."""
        try:
            # Send position command
            command = f'SET POS {position}\n'.encode()
            sock.sendall(command)
            sock.recv(2**10)  # Acknowledge

            # Wait for gripper to move
            time.sleep(1.0)

            # Query current position
            sock.sendall(b'GET POS\n')
            data = sock.recv(2**10)

            # Parse position from response
            match = re.search(r'\d+', data.decode())
            if match:
                gripper_pos = int(match.group())
                percentage = round((float(gripper_pos) / 255.0) * 100.0, 2)

                response.success = True
                response.value = gripper_pos
                response.average = percentage
                response.message = (
                    f'{action_name} command executed successfully. '
                    f'Gripper is {percentage}% closed (raw: {gripper_pos}/255)'
                )
                self.get_logger().info(response.message)
            else:
                response.message = f'ERROR: Could not parse gripper position from response: {data}'
                self.get_logger().error(response.message)

        except socket.timeout:
            response.message = 'ERROR: Socket timeout during command execution'
            self.get_logger().error(response.message)
        except Exception as e:
            response.message = f'ERROR: Command execution failed: {str(e)}'
            self.get_logger().error(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = GripperActionServer()
        rclpy.spin(node)
    except ValueError as e:
        print(f'Failed to start gripper action server: {e}')
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == '__main__':
    exit(main())
