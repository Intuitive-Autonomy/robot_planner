#!/usr/bin/env python3
"""
Robot Planner with State Machine
- Manages robot behavior through 4 modes: rest, approach, assist, retreat
- Publishes /mode (String) and /target_eef_poses (PoseArray)
- State transitions based on human height and distance

Modes:
- rest: Robot at 2m in front, initialization
- approach: Human height decreases, move to 0.4m front, height 1m
- assist: Distance <0.5m, follow shoulder targets
- retreat: Human height increases, return to 2.1m front
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray, Pose, Quaternion
from std_msgs.msg import String
import numpy as np
from collections import deque

class RobotPlannerNew(Node):
    def __init__(self):
        super().__init__('robot_planner_new')

        # State machine
        self.mode = 'rest'  # rest, approach, assist, retreat
        self.last_assist_position = None  # Human position at last assist frame
        self.last_assist_orientation = None  # Human orientation at last assist frame

        # Rest mode fixed target
        self.rest_target_position = None  # Fixed target position in rest mode
        self.rest_target_orientation = None  # Fixed target orientation in rest mode

        # Height tracking for state transitions
        self.height_history = deque(maxlen=30)  # 3 seconds at 10Hz
        self.initial_height = None
        self.standing_height = None  # Maximum standing height (baseline)
        self.height_threshold_down = 0.15  # 15cm decrease to trigger approach
        self.height_threshold_up = 0.10  # 10cm increase to trigger retreat
        self.stable_frames = 20  # Frames to confirm stable height

        # Distance tracking
        self.robot_distance = None  # Current distance to human

        # Publishers
        self.target_eef_publisher = self.create_publisher(
            PoseArray, '/target_eef_poses', 10)
        self.mode_publisher = self.create_publisher(
            String, '/mode', 10)

        # Subscriber
        self.pose_subscriber = self.create_subscription(
            MarkerArray, '/pose_detection', self.pose_callback, 10)

        self.get_logger().info('Robot Planner with State Machine initialized')
        self.get_logger().info(f'Initial mode: {self.mode}')

    def pose_callback(self, msg):
        """Process incoming pose and update state machine"""
        try:
            joints = self.extract_joint_positions(msg)
            if joints is None:
                return

            # Get human state
            torso_height = joints[8, 2]  # Torso Z coordinate
            human_pos_xy = joints[8, :2]  # Torso XY

            # Calculate human orientation
            r_shoulder, l_shoulder = joints[2], joints[3]
            human_ori = self.calculate_human_orientation(r_shoulder, l_shoulder)

            # Initialize and update height baseline
            if self.initial_height is None:
                self.initial_height = torso_height
                self.standing_height = torso_height
                self.get_logger().info(f'Initial human height: {self.initial_height:.3f}m')

            # Update standing height (track maximum height in rest mode)
            if self.mode == 'rest' and torso_height > self.standing_height:
                self.standing_height = torso_height
                self.get_logger().info(f'Updated standing height: {self.standing_height:.3f}m')

            self.height_history.append(torso_height)

            # Update state machine
            self.update_state_machine(torso_height, human_pos_xy, joints)

            # Generate and publish target based on current mode
            target_poses = self.generate_target_poses(human_pos_xy, human_ori, joints)
            self.publish_target_eef_poses(target_poses)

            # Publish current mode
            self.publish_mode()

        except Exception as e:
            self.get_logger().error(f'Error in pose callback: {e}')

    def update_state_machine(self, current_height, human_pos_xy, joints):
        """Update state based on human height and distance"""

        # Calculate distance (for assist/retreat transitions)
        if self.robot_distance is None:
            # Initialize robot at 2m
            self.robot_distance = 2.0

        # Calculate height change from standing baseline
        height_change = current_height - self.standing_height

        old_mode = self.mode

        # State transitions
        if self.mode == 'rest':
            # Initialize rest target position if not set
            if self.rest_target_position is None:
                human_ori = self.calculate_human_orientation(joints[2], joints[3])
                self.rest_target_position = human_pos_xy.copy()
                self.rest_target_orientation = human_ori
                self.get_logger().info(f'Rest target position fixed at: {self.rest_target_position}')

            # rest -> approach: significant height decrease
            if height_change < -self.height_threshold_down:
                self.mode = 'approach'
                self.rest_target_position = None  # Clear rest target
                self.rest_target_orientation = None
                self.get_logger().info(f'Height decreased by {-height_change:.3f}m from standing height, entering approach mode')
            else:
                # Debug: print height change periodically
                if len(self.height_history) % 10 == 0:
                    self.get_logger().info(f'[rest] Height change: {height_change:.3f}m (threshold: -{self.height_threshold_down:.3f}m, standing: {self.standing_height:.3f}m)')

        elif self.mode == 'approach':
            # approach -> assist: robot close to human (<0.5m)
            # Estimate robot distance (will be updated by controller feedback)
            human_ori = self.calculate_human_orientation(joints[2], joints[3])
            target_pos = human_pos_xy + 0.4 * np.array([np.cos(human_ori), np.sin(human_ori)])

            # Simplified: assume robot moves towards target
            self.robot_distance = max(0.0, self.robot_distance - 0.03)  # Conservative estimate

            if self.robot_distance < 0.5:
                self.mode = 'assist'
                self.get_logger().info('Robot within 0.5m, entering assist mode')

        elif self.mode == 'assist':
            # Save current position for retreat
            self.last_assist_position = human_pos_xy.copy()
            self.last_assist_orientation = self.calculate_human_orientation(joints[2], joints[3])

            # assist -> retreat: significant height increase and stable
            if len(self.height_history) >= self.stable_frames:
                recent_heights = list(self.height_history)[-self.stable_frames:]
                avg_recent_height = np.mean(recent_heights)
                height_variance = np.var(recent_heights)

                if (avg_recent_height - self.standing_height > -self.height_threshold_up and
                    height_variance < 0.01):  # Stable (low variance), close to standing height
                    self.mode = 'retreat'
                    self.get_logger().info(f'Height increased and stable (avg: {avg_recent_height:.3f}m, standing: {self.standing_height:.3f}m), entering retreat mode')

        elif self.mode == 'retreat':
            # Estimate distance increasing
            self.robot_distance = min(3.0, self.robot_distance + 0.03)  # Conservative estimate

            # retreat -> rest: distance >= 2m
            if self.robot_distance >= 2.0:
                self.mode = 'rest'
                self.initial_height = current_height  # Reset baseline
                self.rest_target_position = None  # Will be set on next iteration
                self.rest_target_orientation = None
                self.get_logger().info('Robot at 2m, entering rest mode')

        if old_mode != self.mode:
            self.get_logger().info(f'Mode transition: {old_mode} -> {self.mode}')

    def generate_target_poses(self, human_pos_xy, human_ori, joints):
        """
        Generate target EEF poses based on current mode

        Returns:
            [left_pose, right_pose]: list of 2 Pose objects
        """
        if self.mode == 'rest':
            # Use fixed rest target position
            if self.rest_target_position is not None and self.rest_target_orientation is not None:
                return self.create_front_position_poses(
                    self.rest_target_position, self.rest_target_orientation,
                    distance=1.5, height=1.0)
            else:
                # Fallback (shouldn't happen)
                return self.create_front_position_poses(human_pos_xy, human_ori, distance=1.5, height=1.0)

        elif self.mode == 'approach':
            # 0.4m in front, height 1m
            return self.create_front_position_poses(human_pos_xy, human_ori, distance=0.4, height=1.0)

        elif self.mode == 'assist':
            # Shoulder-based targets
            return self.create_shoulder_based_poses(joints)

        elif self.mode == 'retreat':
            # Return to last assist position + 2.1m
            if self.last_assist_position is not None:
                return self.create_front_position_poses(
                    self.last_assist_position, self.last_assist_orientation,
                    distance=2.1, height=1.0)
            else:
                # Fallback
                return self.create_front_position_poses(human_pos_xy, human_ori, distance=2.0, height=1.0)

    def create_front_position_poses(self, human_pos_xy, human_ori, distance, height):
        """Create poses at fixed distance in front of human"""
        # Target position
        target_xy = human_pos_xy + distance * np.array([np.cos(human_ori), np.sin(human_ori)])

        # Arm spacing: 45cm for rest/approach/retreat modes
        arm_spacing = 0.45
        lateral_vec = np.array([-np.sin(human_ori), np.cos(human_ori)])

        left_pos = np.array([
            target_xy[0] - lateral_vec[0] * arm_spacing/2,
            target_xy[1] - lateral_vec[1] * arm_spacing/2,
            height
        ])
        right_pos = np.array([
            target_xy[0] + lateral_vec[0] * arm_spacing/2,
            target_xy[1] + lateral_vec[1] * arm_spacing/2,
            height
        ])

        # Orientation: facing human (opposite direction)
        target_ori = human_ori + np.pi

        # No roll rotation
        left_pose = self.create_pose(left_pos, target_ori)
        right_pose = self.create_pose(right_pos, target_ori)

        return [left_pose, right_pose]

    def create_shoulder_based_poses(self, joints):
        """Create poses based on shoulder positions (10cm below)"""
        r_shoulder, l_shoulder = joints[2], joints[3]

        # Calculate direction
        shoulder_vec = l_shoulder - r_shoulder
        forward_dir = np.cross(shoulder_vec, np.array([0.0, 0.0, 1.0]))
        forward_norm = np.linalg.norm(forward_dir[:2])

        if forward_norm > 1e-6:
            direction = forward_dir / np.linalg.norm(forward_dir)
        else:
            direction = np.array([1.0, 0.0, 0.0])

        # Arm parameters
        arm_length = 0.4
        offset_down = 0.1

        # Centers: 10cm below shoulders
        left_center = r_shoulder.copy()  # Robot left = human right
        left_center[2] -= offset_down

        right_center = l_shoulder.copy()  # Robot right = human left
        right_center[2] -= offset_down

        # Endpoints
        left_front = left_center - (arm_length / 2.0) * direction
        left_back = left_center + (arm_length / 2.0) * direction

        right_front = right_center - (arm_length / 2.0) * direction
        right_back = right_center + (arm_length / 2.0) * direction

        # Center positions
        left_pos = (left_front + left_back) / 2.0
        right_pos = (right_front + right_back) / 2.0

        # Orientation: towards human (opposite of direction)
        target_ori = np.arctan2(-direction[1], -direction[0])

        # No roll rotation
        left_pose = self.create_pose(left_pos, target_ori)
        right_pose = self.create_pose(right_pos, target_ori)

        return [left_pose, right_pose]

    def calculate_human_orientation(self, r_shoulder, l_shoulder):
        """Calculate human facing direction"""
        shoulder_vec = l_shoulder - r_shoulder
        forward_dir = np.cross(shoulder_vec, np.array([0.0, 0.0, 1.0]))
        forward_dir_xy = forward_dir[:2]
        forward_norm = np.linalg.norm(forward_dir_xy)

        if forward_norm > 1e-6:
            forward_dir_xy = forward_dir_xy / forward_norm
            return np.arctan2(forward_dir_xy[1], forward_dir_xy[0])
        return 0.0

    def create_pose(self, position, orientation, local_roll=0.0):
        """
        Create Pose from position, orientation (yaw), and local roll
        local_roll: rotation around the local forward axis (x-axis in end-effector frame)
        """
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])

        # First create quaternion for yaw rotation (around global Z)
        half_yaw = orientation * 0.5
        q_yaw = np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)])

        # Then create quaternion for local roll (around local X, which is the forward direction)
        half_roll = local_roll * 0.5
        q_roll = np.array([np.sin(half_roll), 0.0, 0.0, np.cos(half_roll)])

        # Combine: first apply yaw, then apply roll in the rotated frame
        # Quaternion multiplication: q_yaw * q_roll
        x1, y1, z1, w1 = q_yaw
        x2, y2, z2, w2 = q_roll

        pose.orientation = Quaternion(
            x=float(w1*x2 + x1*w2 + y1*z2 - z1*y2),
            y=float(w1*y2 - x1*z2 + y1*w2 + z1*x2),
            z=float(w1*z2 + x1*y2 - y1*x2 + z1*w2),
            w=float(w1*w2 - x1*x2 - y1*y2 - z1*z2)
        )
        return pose

    def extract_joint_positions(self, marker_array):
        """Extract joint positions from MarkerArray"""
        joint_positions = np.zeros((9, 3))
        found_joints = 0

        for marker in marker_array.markers:
            if marker.ns == 'skeleton_joints' and marker.type == 2:
                joint_idx = marker.id
                if 0 <= joint_idx < 9:
                    joint_positions[joint_idx] = [
                        marker.pose.position.x,
                        marker.pose.position.y,
                        marker.pose.position.z
                    ]
                    found_joints += 1

        if found_joints < 9:
            return None
        return joint_positions

    def publish_target_eef_poses(self, poses):
        """Publish target EEF poses"""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "world"
        pose_array.poses = poses

        self.target_eef_publisher.publish(pose_array)

    def publish_mode(self):
        """Publish current mode"""
        mode_msg = String()
        mode_msg.data = self.mode
        self.mode_publisher.publish(mode_msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        planner = RobotPlannerNew()

        print("\n" + "="*60)
        print("Robot Planner with State Machine Started")
        print("="*60)
        print("Topics:")
        print("  - Subscribing to: /pose_detection (MarkerArray)")
        print("  - Publishing to: /target_eef_poses (PoseArray)")
        print("  - Publishing to: /mode (String)")
        print("")
        print("State Machine:")
        print("  - rest: 2m in front (initialization)")
        print("  - approach: Height↓ >15cm -> move to 0.4m")
        print("  - assist: Distance <0.5m -> follow shoulders")
        print("  - retreat: Height↑ >10cm stable -> return to 2.1m")
        print("  - back to rest: Distance >=2m")
        print("="*60)
        print("Press Ctrl+C to stop...\n")

        rclpy.spin(planner)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'planner' in locals():
            planner.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
