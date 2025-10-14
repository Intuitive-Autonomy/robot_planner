#!/usr/bin/env python3
"""
Robot Controller
- Subscribes to /target_eef_poses (PoseArray) for target positions
- Applies velocity and angular velocity constraints
- Publishes /target_poses (PoseArray) for next step robot control
- Publishes /current_poses (PoseArray) for robot current state

Robot state definition:
- current_poses: center point minus 20cm in opposite direction, projected to XY plane
- Each timestep: current_poses = previous target_poses

Constraints:
- Max linear velocity: 0.3 m/s
- Max angular velocity: 10 deg/s
- Control rate: 10 Hz
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Quaternion
import numpy as np
import math


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Control parameters
        self.max_linear_vel = 0.3  # m/s
        self.max_angular_vel = math.radians(10)  # 10 deg/s to rad/s
        self.control_rate = 10.0  # Hz
        self.dt = 1.0 / self.control_rate

        # EMA smoothing parameters
        self.ema_alpha = 0.5  # Smoothing factor (0 = no smoothing, 1 = no filtering)
        self.ema_poses = None  # Smoothed poses [2] - EMA filtered poses

        # State variables
        self.current_poses = None  # [2] - current robot Pose objects (left, right)
        self.target_eef_poses = None  # Target from /target_eef_poses

        # Publishers
        self.target_poses_pub = self.create_publisher(PoseArray, '/target_poses', 10)
        self.current_poses_pub = self.create_publisher(PoseArray, '/current_poses', 10)

        # Subscribers
        self.target_eef_sub = self.create_subscription(
            PoseArray, '/target_eef_poses', self.target_eef_callback, 10)

        # Control timer
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('Robot Controller initialized')
        self.get_logger().info(f'Control rate: {self.control_rate} Hz')
        self.get_logger().info(f'Max linear vel: {self.max_linear_vel} m/s')
        self.get_logger().info(f'Max angular vel: {math.degrees(self.max_angular_vel)} deg/s')
        self.get_logger().info(f'EMA alpha: {self.ema_alpha}')

    def apply_ema_smoothing(self, new_poses):
        """
        Apply Exponential Moving Average smoothing to poses

        Args:
            new_poses: list of 2 Pose objects

        Returns:
            smoothed_poses: list of 2 Pose objects with EMA applied
        """
        if self.ema_poses is None:
            # Initialize EMA with first poses
            self.ema_poses = new_poses
            return new_poses

        smoothed_poses = []
        for i in range(2):
            # EMA for position
            new_pos = np.array([new_poses[i].position.x, new_poses[i].position.y, new_poses[i].position.z])
            ema_pos = np.array([self.ema_poses[i].position.x, self.ema_poses[i].position.y, self.ema_poses[i].position.z])

            smoothed_pos = self.ema_alpha * new_pos + (1 - self.ema_alpha) * ema_pos

            # SLERP for orientation (quaternion smoothing)
            new_q = new_poses[i].orientation
            ema_q = self.ema_poses[i].orientation

            # Calculate dot product
            dot = new_q.x * ema_q.x + new_q.y * ema_q.y + new_q.z * ema_q.z + new_q.w * ema_q.w

            # If dot < 0, negate to take shorter path
            if dot < 0:
                ema_q.x = -ema_q.x
                ema_q.y = -ema_q.y
                ema_q.z = -ema_q.z
                ema_q.w = -ema_q.w
                dot = -dot

            # Clamp and calculate angle
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)

            # SLERP with ema_alpha as interpolation parameter
            if theta < 1e-6:
                # Linear interpolation for very close quaternions
                smoothed_q = Quaternion(
                    x=float(self.ema_alpha * new_q.x + (1 - self.ema_alpha) * ema_q.x),
                    y=float(self.ema_alpha * new_q.y + (1 - self.ema_alpha) * ema_q.y),
                    z=float(self.ema_alpha * new_q.z + (1 - self.ema_alpha) * ema_q.z),
                    w=float(self.ema_alpha * new_q.w + (1 - self.ema_alpha) * ema_q.w)
                )
            else:
                sin_theta = np.sin(theta)
                a = np.sin((1 - self.ema_alpha) * theta) / sin_theta
                b = np.sin(self.ema_alpha * theta) / sin_theta

                smoothed_q = Quaternion(
                    x=float(a * ema_q.x + b * new_q.x),
                    y=float(a * ema_q.y + b * new_q.y),
                    z=float(a * ema_q.z + b * new_q.z),
                    w=float(a * ema_q.w + b * new_q.w)
                )

            # Create smoothed pose
            smoothed_pose = Pose()
            smoothed_pose.position.x = float(smoothed_pos[0])
            smoothed_pose.position.y = float(smoothed_pos[1])
            smoothed_pose.position.z = float(smoothed_pos[2])
            smoothed_pose.orientation = smoothed_q

            smoothed_poses.append(smoothed_pose)

        # Update EMA state
        self.ema_poses = smoothed_poses

        return smoothed_poses

    def target_eef_callback(self, msg):
        """Receive target EEF poses"""
        if len(msg.poses) >= 2:
            self.target_eef_poses = msg
        else:
            self.get_logger().warning(f'Received target_eef_poses with {len(msg.poses)} poses, expected 2')

    def pose_to_position_orientation(self, pose):
        """
        Extract position and orientation from Pose

        Args:
            pose: Pose object

        Returns:
            position: np.array([x, y, z])
            orientation: float, angle around Z-axis (radians)
        """
        position = np.array([pose.position.x, pose.position.y, pose.position.z])

        # Extract yaw angle from quaternion
        q = pose.orientation
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y**2 + q.z**2))

        return position, yaw

    def create_pose(self, position, orientation):
        """
        Create Pose from position and orientation

        Args:
            position: np.array([x, y, z])
            orientation: float, angle around Z-axis (radians)

        Returns:
            Pose object
        """
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])

        # Convert yaw to quaternion
        half_angle = orientation / 2.0
        pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=float(np.sin(half_angle)),
            w=float(np.cos(half_angle))
        )

        return pose

    def move_towards_target(self, current_pose, target_pose):
        """
        Calculate next pose with velocity constraints

        Args:
            current_pose: Pose object
            target_pose: Pose object

        Returns:
            next_pose: Pose object
        """
        # Extract positions
        current_pos = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
        target_pos = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z])

        # Linear movement
        displacement = target_pos - current_pos
        distance = np.linalg.norm(displacement)
        max_step = self.max_linear_vel * self.dt

        if distance > max_step:
            next_pos = current_pos + (displacement / distance) * max_step
        else:
            next_pos = target_pos

        # Angular movement - use quaternion SLERP for smooth interpolation
        current_q = current_pose.orientation
        target_q = target_pose.orientation

        # Calculate angle between quaternions
        dot = current_q.x * target_q.x + current_q.y * target_q.y + current_q.z * target_q.z + current_q.w * target_q.w

        # If dot < 0, negate one quaternion to take shorter path
        if dot < 0:
            target_q.x = -target_q.x
            target_q.y = -target_q.y
            target_q.z = -target_q.z
            target_q.w = -target_q.w
            dot = -dot

        # Clamp dot to avoid numerical issues
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)

        # Calculate interpolation parameter t based on max angular velocity
        max_angle_step = self.max_angular_vel * self.dt

        if theta > max_angle_step:
            t = max_angle_step / theta
        else:
            t = 1.0

        # SLERP interpolation
        if theta < 1e-6:
            # Quaternions are very close, use linear interpolation
            next_q = Quaternion(
                x=float((1-t) * current_q.x + t * target_q.x),
                y=float((1-t) * current_q.y + t * target_q.y),
                z=float((1-t) * current_q.z + t * target_q.z),
                w=float((1-t) * current_q.w + t * target_q.w)
            )
        else:
            # Use SLERP
            sin_theta = np.sin(theta)
            a = np.sin((1-t) * theta) / sin_theta
            b = np.sin(t * theta) / sin_theta

            next_q = Quaternion(
                x=float(a * current_q.x + b * target_q.x),
                y=float(a * current_q.y + b * target_q.y),
                z=float(a * current_q.z + b * target_q.z),
                w=float(a * current_q.w + b * target_q.w)
            )

        # Create next pose
        next_pose = Pose()
        next_pose.position.x = float(next_pos[0])
        next_pose.position.y = float(next_pos[1])
        next_pose.position.z = float(next_pos[2])
        next_pose.orientation = next_q

        return next_pose

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def calculate_current_pose(self, center_pose):
        """
        Calculate current pose: center minus 20cm in opposite direction, XY projection

        Args:
            center_pose: Pose object (center position)

        Returns:
            current_pose: Pose object (projected to XY plane)
        """
        position, orientation = self.pose_to_position_orientation(center_pose)

        # 20cm back in opposite direction
        back_offset = -0.2 * np.array([np.cos(orientation), np.sin(orientation), 0.0])
        current_pos = position + back_offset

        # Project to XY plane (Z=0)
        current_pos[2] = 0.0

        return self.create_pose(current_pos, orientation)

    def control_loop(self):
        """Main control loop"""
        try:
            # Wait for target
            if self.target_eef_poses is None:
                return

            # Initialize current poses if not set
            if self.current_poses is None:
                # Initialize at target positions
                self.current_poses = [
                    self.target_eef_poses.poses[0],
                    self.target_eef_poses.poses[1]
                ]
                self.get_logger().info('Initialized robot poses from target_eef_poses')
                return

            # Move towards target with velocity constraints
            next_poses = []
            for i in range(2):
                next_pose = self.move_towards_target(
                    self.current_poses[i],
                    self.target_eef_poses.poses[i]
                )
                next_poses.append(next_pose)

            # Apply EMA smoothing
            smoothed_poses = self.apply_ema_smoothing(next_poses)

            # Publish target poses (next step)
            self.publish_target_poses(smoothed_poses)

            # Update current poses: current = previous target
            self.current_poses = next_poses

            # Publish current poses (20cm back, XY projection)
            self.publish_current_poses()

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')

    def publish_target_poses(self, poses):
        """
        Publish target poses with left/right swapped and orientation reversed

        Args:
            poses: list of 2 Pose objects [left, right]
        """
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "world"

        # Keep original order (no swap), and reverse orientation (180 degrees around Z)
        reversed_poses = []
        for pose in [poses[0], poses[1]]:  # Original order
            # Original quaternion
            q = pose.orientation
            x1, y1, z1, w1 = q.x, q.y, q.z, q.w

            # 180 degree rotation around Z: [0, 0, 1, 0]
            x2, y2, z2, w2 = 0.0, 0.0, 1.0, 0.0

            # Quaternion multiplication: q_original * q_180z
            reversed_pose = Pose()
            reversed_pose.position = pose.position
            reversed_pose.orientation = Quaternion(
                x=float(w1*x2 + x1*w2 + y1*z2 - z1*y2),
                y=float(w1*y2 - x1*z2 + y1*w2 + z1*x2),
                z=float(w1*z2 + x1*y2 - y1*x2 + z1*w2),
                w=float(w1*w2 - x1*x2 - y1*y2 - z1*z2)
            )
            reversed_poses.append(reversed_pose)

        pose_array.poses = reversed_poses

        self.target_poses_pub.publish(pose_array)

    def publish_current_poses(self):
        """
        Publish current poses (20cm back from center, XY projection)
        """
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "world"

        for center_pose in self.current_poses:
            current_pose = self.calculate_current_pose(center_pose)
            pose_array.poses.append(current_pose)

        self.current_poses_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)

    try:
        controller = RobotController()

        print("\n" + "="*60)
        print("Robot Controller Started")
        print("="*60)
        print("Topics:")
        print("  - Subscribing to: /target_eef_poses (PoseArray)")
        print("  - Publishing to: /target_poses (PoseArray)")
        print("  - Publishing to: /current_poses (PoseArray)")
        print("")
        print("Robot State Definition:")
        print("  - current_poses: center - 20cm in opposite direction")
        print("  - Projected to XY plane (Z=0)")
        print("  - Each timestep: current = previous target")
        print("")
        print("Constraints:")
        print("  - Max linear velocity: 0.3 m/s")
        print("  - Max angular velocity: 10 deg/s")
        print("  - Control rate: 10 Hz")
        print("  - Step size: 0.03m, 3 degrees per timestep (max)")
        print("="*60)
        print("Press Ctrl+C to stop...\n")

        rclpy.spin(controller)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
