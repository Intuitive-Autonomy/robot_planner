#!/usr/bin/env python3
"""
Real-time Robot Planner
- Subscribes to /pose_detection topic for human pose data
- Applies coordinate transformation (rotate -90 degrees around X-axis, convert m to mm)
- Uses trained model for real-time inference
- Publishes predicted arm positions as MarkerArray to /target_poses
- Transforms back to ROS coordinate system and publishes in world frame
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Quaternion, PoseArray, Pose
from std_msgs.msg import Int32
import numpy as np
import torch
from collections import deque
import json
import os

# Import model and utils
from net import get_model
from utils import normalize_to_torso


class RobotPlanner(Node):
    def __init__(self, model_path, config_path):
        super().__init__('robot_planner')

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Model parameters
        self.sequence_length = self.config['sequence_length']  # 10 frames
        self.prediction_length = self.config['prediction_length']  # 5 frames
        self.skeleton_dim = 39  # 9 upper body joints * 3 + 4 robot endpoints * 3
        self.output_dim = 12   # 4 arm endpoints * 3 coordinates

        # Initialize robot position buffer (stores current robot arm positions)
        self.current_robot_pos = np.zeros((4, 3))  # [4 endpoints, 3 coords]

        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(
            skeleton_dim=self.skeleton_dim,
            output_dim=self.output_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout']
        )

        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info(f'Loaded model from {model_path}')
        self.get_logger().info(f'Using device: {self.device}')

        # Input buffer for sequence
        self.pose_buffer = deque(maxlen=self.sequence_length)

        # Publishers
        self.target_poses_publisher = self.create_publisher(PoseArray, '/target_poses', 10)
        self.phase_publisher = self.create_publisher(Int32, '/predicted_phase', 10)

        # Subscribers
        self.pose_subscriber = self.create_subscription(
            MarkerArray,  # Assuming pose_detection publishes MarkerArray
            '/pose_detection',
            self.pose_callback,
            10
        )

        # Joint names for upper body (9 joints)
        self.joint_names = [
            'Head', 'Neck', 'R_Shoulder', 'L_Shoulder',
            'R_Elbow', 'L_Elbow', 'R_Hand', 'L_Hand', 'Torso'
        ]

        self.get_logger().info('Robot Planner initialized')
        self.get_logger().info('Waiting for pose data on /pose_detection...')

    def pose_callback(self, msg):
        """Process incoming pose data"""
        try:
            # Extract joint positions from MarkerArray
            joint_positions = self.extract_joint_positions(msg)

            if joint_positions is None:
                return

            # Apply coordinate transformation: rotate -90 degrees around X-axis and convert m to mm
            transformed_joints = self.transform_ros_to_model(joint_positions)

            # Add to buffer
            self.pose_buffer.append(transformed_joints)

            # Perform inference when we have enough frames
            if len(self.pose_buffer) == self.sequence_length:
                predicted_arms, predicted_phase = self.predict_arm_positions()

                if predicted_arms is not None:
                    # Update current robot position for next prediction (use first predicted frame)
                    self.current_robot_pos = predicted_arms[0].copy()  # [4, 3]

                    # Transform back to ROS coordinate system
                    ros_arm_positions = self.transform_model_to_ros(predicted_arms)

                    # Publish predictions
                    self.publish_target_poses(ros_arm_positions)
                    self.publish_phase(predicted_phase)

        except Exception as e:
            self.get_logger().error(f'Error in pose callback: {e}')

    def extract_joint_positions(self, marker_array):
        """Extract joint positions from MarkerArray message"""
        joint_positions = np.zeros((9, 3))  # 9 joints * 3 coordinates
        found_joints = 0

        for marker in marker_array.markers:
            # Only process skeleton joint markers (not bone lines)
            if marker.ns == 'skeleton_joints' and marker.type == 2:  # SPHERE type
                joint_idx = marker.id
                if 0 <= joint_idx < 9:
                    joint_positions[joint_idx] = [
                        marker.pose.position.x,
                        marker.pose.position.y,
                        marker.pose.position.z
                    ]
                    found_joints += 1

        if found_joints < 9:
            self.get_logger().warning(f'Only found {found_joints}/9 joints in pose data')
            return None

        return joint_positions

    def transform_ros_to_model(self, joint_positions):
        """
        Transform from ROS coordinate system to model coordinate system
        - Rotate 90 degrees around X-axis: (x, y, z) -> (x, -z, y)
        - Convert from meters to millimeters
        """
        # Convert m to mm
        joint_positions_mm = joint_positions * 1000.0

        # Apply rotation around X-axis by 90 degrees
        transformed = np.zeros_like(joint_positions_mm)
        transformed[:, 0] = joint_positions_mm[:, 0]   # x unchanged
        transformed[:, 1] = joint_positions_mm[:, 2]  # y = z
        transformed[:, 2] = -joint_positions_mm[:, 1]   # z = -y

        return transformed

    def transform_model_to_ros(self, arm_positions):
        """
        Transform from model coordinate system back to ROS coordinate system
        - Rotate -90 degrees around X-axis: (x, y, z) -> (x, z, -y)
        - Convert from millimeters to meters
        """
        # Apply rotation around X-axis by -90 degrees (inverse of 90)
        transformed = np.zeros_like(arm_positions)
        transformed[:, :, 0] = arm_positions[:, :, 0]   # x unchanged
        transformed[:, :, 1] = -arm_positions[:, :, 2]   # y = -z
        transformed[:, :, 2] = arm_positions[:, :, 1]  # z = y

        # Convert mm to m
        transformed = transformed / 1000.0

        return transformed

    def predict_arm_positions(self):
        """Use model to predict arm positions"""
        try:
            # Convert buffer to numpy array
            sequence = np.array(list(self.pose_buffer))  # [seq_len, 9, 3]

            # Normalize to torso coordinate system (like in training)
            # Create dummy arms for normalization (will be ignored)
            dummy_arms = np.zeros((self.sequence_length, 4, 3))
            normalized_joints, _ = normalize_to_torso(sequence, dummy_arms)

            # Flatten joints: [seq_len, 27]
            joints_flat = normalized_joints.reshape(self.sequence_length, -1)

            # Normalize current robot position relative to torso (use last frame's torso)
            torso_pos = sequence[-1, 8, :]  # Last frame's torso position [3]
            robot_pos_normalized = self.current_robot_pos - torso_pos[None, :]  # [4, 3]
            robot_pos_flat = robot_pos_normalized.reshape(-1)  # [12]

            # Concatenate: [seq_len, 27 + 12] -> [seq_len, 39]
            robot_pos_tiled = np.tile(robot_pos_flat, (self.sequence_length, 1))  # [seq_len, 12]
            model_input = np.concatenate([joints_flat, robot_pos_tiled], axis=1)  # [seq_len, 39]

            # Convert to tensor
            input_tensor = torch.FloatTensor(model_input).unsqueeze(0).to(self.device)  # [1, seq_len, 39]

            # Model inference
            with torch.no_grad():
                predicted_robot, predicted_phase_logits = self.model(
                    input_tensor,
                    pred_len=self.prediction_length,
                    autoregressive=False,
                    return_phase=True
                )  # [1, pred_len, 12], [1, pred_len, 3]

            # Convert back to numpy: [pred_len, 12] -> [pred_len, 4, 3]
            predicted_arms = predicted_robot.cpu().numpy().squeeze(0).reshape(self.prediction_length, 4, 3)

            # Get phase prediction (use first prediction step)
            phase_logits = predicted_phase_logits.cpu().numpy().squeeze(0)  # [pred_len, 3]
            predicted_phase = int(np.argmax(phase_logits[0]))  # Use first prediction

            # Transform back to world coordinates by adding torso position
            current_torso = sequence[-1, 8]  # Latest torso position [3]
            for t in range(self.prediction_length):
                predicted_arms[t] += current_torso[None, :]  # Add torso to all 4 endpoints

            return predicted_arms, predicted_phase

        except Exception as e:
            self.get_logger().error(f'Error in prediction: {e}')
            return None, 0

    def vector_to_quaternion(self, direction_vector):
        """Convert a direction vector to a quaternion representing rotation from +X axis"""
        # Normalize the direction vector
        direction = direction_vector / (np.linalg.norm(direction_vector) + 1e-8)

        # Default forward direction is +X axis
        forward = np.array([1.0, 0.0, 0.0])

        # Calculate rotation axis (cross product)
        axis = np.cross(forward, direction)
        axis_length = np.linalg.norm(axis)

        if axis_length < 1e-6:  # Vectors are parallel
            if np.dot(forward, direction) > 0:
                # Same direction
                return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            else:
                # Opposite direction, rotate 180 degrees around Y axis
                return Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)

        # Normalize rotation axis
        axis = axis / axis_length

        # Calculate rotation angle
        angle = np.arccos(np.clip(np.dot(forward, direction), -1.0, 1.0))

        # Convert to quaternion
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        return Quaternion(
            x=float(axis[0] * sin_half),
            y=float(axis[1] * sin_half),
            z=float(axis[2] * sin_half),
            w=float(cos_half)
        )

    def publish_target_poses(self, arm_positions):
        """Publish predicted arm positions as PoseArray with 2 poses at arm centers"""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "world"

        # Create poses for the 2 arm centers (only current prediction t=0)
        current_prediction = arm_positions[0]  # [4, 3]

        # Extract positions
        left_l1 = current_prediction[0]   # [3]
        left_l2 = current_prediction[1]   # [3]
        right_r1 = current_prediction[2]  # [3]
        right_r2 = current_prediction[3]  # [3]

        # Calculate arm center positions (midpoint between endpoints)
        left_center = (left_l1 + left_l2) / 2.0   # Center of left arm
        right_center = (right_r1 + right_r2) / 2.0  # Center of right arm

        # Calculate arm directions (from L2→L1, R2→R1)
        left_direction = left_l1 - left_l2   # L2 -> L1
        right_direction = right_r1 - right_r2  # R2 -> R1

        # Create poses for arm centers
        poses_info = [
            (left_center, left_direction, "Left_Arm_Center"),
            (right_center, right_direction, "Right_Arm_Center")
        ]

        for pos, direction, name in poses_info:
            pose = Pose()
            # Position at arm center
            pose.position.x = float(pos[0])
            pose.position.y = float(pos[1])
            pose.position.z = float(pos[2])

            # Orientation pointing along arm direction
            pose.orientation = self.vector_to_quaternion(direction)

            pose_array.poses.append(pose)

        # Publish the pose array
        self.target_poses_publisher.publish(pose_array)

        # Calculate arm lengths for logging
        left_length = np.linalg.norm(left_direction)
        right_length = np.linalg.norm(right_direction)

        # Log prediction info
        self.get_logger().info(
            f'Published PoseArray (L2→L1, R2→R1): '
            f'Left_Center=({left_center[0]:.3f}, {left_center[1]:.3f}, {left_center[2]:.3f}), '
            f'Right_Center=({right_center[0]:.3f}, {right_center[1]:.3f}, {right_center[2]:.3f}) | '
            f'Lengths: L={left_length*100:.1f}cm, R={right_length*100:.1f}cm'
        )

    def publish_phase(self, phase):
        """Publish predicted phase"""
        phase_msg = Int32()
        phase_msg.data = phase
        self.phase_publisher.publish(phase_msg)

        phase_names = ['Approaching', 'Assisting', 'Leaving']
        self.get_logger().info(f'Published Phase: {phase} ({phase_names[phase]})')


def main(args=None):
    rclpy.init(args=args)

    # Model and config paths - use latest stage model
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_30_10')
    config_path = os.path.join(os.path.dirname(__file__), 'configs', '30_10.json')

    # Try to find the latest stage model (4 > 3 > 2 > 1)
    model_path = None
    for stage in [4, 3, 2, 1]:
        stage_ckpt = os.path.join(save_dir, f'best_model_stage{stage}.pth')
        if os.path.exists(stage_ckpt):
            model_path = stage_ckpt
            break

    if model_path is None:
        # Fallback to old naming
        model_path = os.path.join(save_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model has been trained and saved.")
        return

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    try:
        planner = RobotPlanner(model_path, config_path)

        print(f"Robot Planner started")
        print(f"Model: {model_path}")
        print(f"Config: {config_path}")
        print("Topics:")
        print("  - Subscribing to: /pose_detection (MarkerArray)")
        print("  - Publishing to: /target_poses (PoseArray - 2 arm centers)")
        print("  - Publishing to: /predicted_phase (Int32 - 0:Approaching, 1:Assisting, 2:Leaving)")
        print("Pose order: [Left_Arm_Center, Right_Arm_Center]")
        print("Position: Midpoint between L1-L2 and R1-R2")
        print("Orientation: Left arm L2→L1, Right arm R2→R1")
        print("Press Ctrl+C to stop...")

        rclpy.spin(planner)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()