#!/usr/bin/env python3
"""
Debug Visualizer for Real-time Human Pose
- Subscribes to /pose_detection topic (MarkerArray)
- Visualizes human skeleton in 3D with fixed origin and equal aspect ratio
- Real-time animation with smooth updates
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseArray
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import threading


class DebugVisualizer(Node):
    def __init__(self):
        super().__init__('debug_visualizer')

        # Joint names for upper body (9 joints)
        self.joint_names = [
            'Head', 'Neck', 'R_Shoulder', 'L_Shoulder',
            'R_Elbow', 'L_Elbow', 'R_Hand', 'L_Hand', 'Torso'
        ]

        # Skeleton connections (bone lines)
        self.skeleton_connections = [
            (0, 1),  # Head - Neck
            (1, 8),  # Neck - Torso
            (1, 2),  # Neck - R_Shoulder
            (1, 3),  # Neck - L_Shoulder
            (2, 4),  # R_Shoulder - R_Elbow
            (3, 5),  # L_Shoulder - L_Elbow
            (4, 6),  # R_Elbow - R_Hand
            (5, 7),  # L_Elbow - L_Hand
        ]

        # Data storage
        self.joint_positions = None  # [9, 3] in meters (ROS coordinates)
        self.target_poses = None  # PoseArray from /target_poses
        self.data_lock = threading.Lock()

        # Subscribe to pose detection
        self.pose_subscriber = self.create_subscription(
            MarkerArray,
            '/pose_detection',
            self.pose_callback,
            10
        )

        # Subscribe to target poses
        self.target_poses_subscriber = self.create_subscription(
            PoseArray,
            '/target_poses',
            self.target_poses_callback,
            10
        )

        self.get_logger().info('Debug Visualizer initialized')
        self.get_logger().info('Waiting for pose data on /pose_detection...')

        # Setup matplotlib figure
        self.setup_plot()

    def setup_plot(self):
        """Setup 3D plot with fixed axes and equal aspect ratio"""
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Fixed coordinate range (meters)
        self.ax.set_xlim(-1.0, 1.0)  # X: -1 to 1m
        self.ax.set_ylim(-1.0, 1.0)  # Y: -1 to 1m
        self.ax.set_zlim(0.0, 2.0)   # Z: 0 to 2m

        # Equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])

        # Labels
        self.ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        self.ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
        self.ax.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
        self.ax.set_title('Real-time Human Pose Visualization', fontsize=14, fontweight='bold')

        # Grid
        self.ax.grid(True, alpha=0.3)

        # Initial view angle
        self.ax.view_init(elev=20, azim=45)

        # Artists storage for efficient updates
        self.artists = {'lines': [], 'scatters': [], 'texts': []}

        self.get_logger().info('3D plot setup complete')

    def pose_callback(self, msg):
        """Process incoming pose data from MarkerArray"""
        try:
            # Extract joint positions from MarkerArray
            joint_positions_ros = self.extract_joint_positions(msg)

            if joint_positions_ros is None:
                return

            # Store with thread lock (no transformation, keep ROS coordinates in meters)
            with self.data_lock:
                self.joint_positions = joint_positions_ros

        except Exception as e:
            self.get_logger().error(f'Error in pose callback: {e}')

    def target_poses_callback(self, msg):
        """Process incoming target poses from /target_poses"""
        try:
            with self.data_lock:
                self.target_poses = msg
        except Exception as e:
            self.get_logger().error(f'Error in target_poses callback: {e}')

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

    def calculate_robot_arm_targets(self, joints):
        """
        Calculate robot arm target positions

        Args:
            joints: [9, 3] joint positions in meters

        Returns:
            left_arm: [2, 3] left arm endpoints (back, front)
            right_arm: [2, 3] right arm endpoints (back, front)
        """
        # Joint indices
        r_shoulder_idx = 2  # R_Shoulder
        l_shoulder_idx = 3  # L_Shoulder

        r_shoulder = joints[r_shoulder_idx]  # [3]
        l_shoulder = joints[l_shoulder_idx]  # [3]

        # Calculate human facing direction (perpendicular to shoulder line)
        # Shoulder vector (right to left)
        shoulder_vec = l_shoulder - r_shoulder

        # Direction perpendicular to shoulders (human facing direction)
        # Cross product with Z-axis (up) gives forward direction
        forward_dir = np.cross(shoulder_vec, np.array([0.0, 0.0, 1.0]))
        forward_norm = np.linalg.norm(forward_dir)

        if forward_norm < 1e-6:
            # Fallback if shoulders are aligned vertically
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = forward_dir / forward_norm

        # Arm parameters
        arm_length = 0.4  # 40cm
        offset_down = 0.1  # 10cm down

        # Left arm center: right shoulder - 10cm down (robot left = human's right)
        left_center = r_shoulder.copy()
        left_center[2] -= offset_down  # Z down

        # Right arm center: left shoulder - 10cm down (robot right = human's left)
        right_center = l_shoulder.copy()
        right_center[2] -= offset_down  # Z down

        # Left arm endpoints (back and front along direction)
        # Front is towards human (negative direction), back is away from human
        left_front = left_center - (arm_length / 2.0) * direction
        left_back = left_center + (arm_length / 2.0) * direction

        # Right arm endpoints (back and front along direction)
        # Front is towards human (negative direction), back is away from human
        right_front = right_center - (arm_length / 2.0) * direction
        right_back = right_center + (arm_length / 2.0) * direction

        # Return as [back, front] so arrow points from back to front (towards human)
        left_arm = np.array([left_back, left_front])    # [2, 3]
        right_arm = np.array([right_back, right_front])  # [2, 3]

        return left_arm, right_arm

    def quaternion_to_euler(self, q):
        """Convert quaternion to roll, pitch, yaw (in radians)"""
        # q = [x, y, z, w]
        x, y, z, w = q.x, q.y, q.z, q.w

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def update_plot(self, frame):
        """Update the 3D plot with current pose data"""
        # Get current joint positions and target poses with thread lock
        with self.data_lock:
            joints = self.joint_positions
            target_poses = self.target_poses

        # Remove previous artists
        for artist_list in self.artists.values():
            for artist in artist_list:
                artist.remove()
        self.artists['lines'].clear()
        self.artists['scatters'].clear()
        self.artists['texts'].clear()

        if joints is not None:
            # Draw skeleton bones (connections)
            for conn in self.skeleton_connections:
                p1, p2 = joints[conn[0]], joints[conn[1]]
                line, = self.ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    'b-', linewidth=3, alpha=0.7
                )
                self.artists['lines'].append(line)

            # Draw joint points
            scatter = self.ax.scatter(
                joints[:, 0], joints[:, 1], joints[:, 2],
                c='red', s=100, alpha=0.8, marker='o',
                edgecolors='darkred', linewidths=2
            )
            self.artists['scatters'].append(scatter)

            # Calculate and draw robot arm targets
            try:
                left_arm, right_arm = self.calculate_robot_arm_targets(joints)

                # Draw left arm (red)
                line_left, = self.ax.plot(
                    [left_arm[0, 0], left_arm[1, 0]],
                    [left_arm[0, 1], left_arm[1, 1]],
                    [left_arm[0, 2], left_arm[1, 2]],
                    'r-', linewidth=5, alpha=0.9, label='Left Arm Target'
                )
                self.artists['lines'].append(line_left)

                # Draw left arm endpoints
                scatter_left = self.ax.scatter(
                    left_arm[:, 0], left_arm[:, 1], left_arm[:, 2],
                    c='red', s=150, alpha=0.9, marker='o',
                    edgecolors='darkred', linewidths=2
                )
                self.artists['scatters'].append(scatter_left)

                # Draw left arm direction arrow (from back to front)
                left_dir = left_arm[1] - left_arm[0]  # Direction vector
                arrow_left = self.ax.quiver(
                    left_arm[0, 0], left_arm[0, 1], left_arm[0, 2],
                    left_dir[0], left_dir[1], left_dir[2],
                    color='red', arrow_length_ratio=0.2, linewidth=3, alpha=0.9
                )
                self.artists['scatters'].append(arrow_left)

                # Draw right arm (green)
                line_right, = self.ax.plot(
                    [right_arm[0, 0], right_arm[1, 0]],
                    [right_arm[0, 1], right_arm[1, 1]],
                    [right_arm[0, 2], right_arm[1, 2]],
                    'g-', linewidth=5, alpha=0.9, label='Right Arm Target'
                )
                self.artists['lines'].append(line_right)

                # Draw right arm endpoints
                scatter_right = self.ax.scatter(
                    right_arm[:, 0], right_arm[:, 1], right_arm[:, 2],
                    c='green', s=150, alpha=0.9, marker='o',
                    edgecolors='darkgreen', linewidths=2
                )
                self.artists['scatters'].append(scatter_right)

                # Draw right arm direction arrow (from back to front)
                right_dir = right_arm[1] - right_arm[0]  # Direction vector
                arrow_right = self.ax.quiver(
                    right_arm[0, 0], right_arm[0, 1], right_arm[0, 2],
                    right_dir[0], right_dir[1], right_dir[2],
                    color='green', arrow_length_ratio=0.2, linewidth=3, alpha=0.9
                )
                self.artists['scatters'].append(arrow_right)

                # Add legend
                self.ax.legend(loc='upper right', fontsize=10)

            except Exception as e:
                self.get_logger().error(f'Error calculating robot arms: {e}')

            # Add joint labels
            for i, name in enumerate(self.joint_names):
                text = self.ax.text(
                    joints[i, 0], joints[i, 1], joints[i, 2],
                    name, fontsize=8, color='black'
                )
                self.artists['texts'].append(text)

        # Visualize target poses
        if target_poses is not None and len(target_poses.poses) >= 2:
            try:
                for i, pose in enumerate(target_poses.poses[:2]):
                    pos = pose.position
                    roll, pitch, yaw = self.quaternion_to_euler(pose.orientation)

                    # Draw pose position as a sphere
                    color = 'cyan' if i == 0 else 'magenta'
                    label = f'Target {i} (L)' if i == 0 else f'Target {i} (R)'
                    scatter_target = self.ax.scatter(
                        [pos.x], [pos.y], [pos.z],
                        c=color, s=200, alpha=0.8, marker='s',
                        edgecolors='black', linewidths=2, label=label
                    )
                    self.artists['scatters'].append(scatter_target)

                    # Draw orientation axes (RPY)
                    # X-axis (roll) in red, Y-axis (pitch) in green, Z-axis (yaw) in blue
                    axis_length = 0.15

                    # Rotation matrix from RPY
                    cr, sr = np.cos(roll), np.sin(roll)
                    cp, sp = np.cos(pitch), np.sin(pitch)
                    cy, sy = np.cos(yaw), np.sin(yaw)

                    # ZYX rotation matrix
                    R = np.array([
                        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                        [-sp, cp*sr, cp*cr]
                    ])

                    # Draw axes
                    x_axis = R @ np.array([axis_length, 0, 0])
                    y_axis = R @ np.array([0, axis_length, 0])
                    z_axis = R @ np.array([0, 0, axis_length])

                    # X-axis (roll) - red
                    arrow_x = self.ax.quiver(
                        pos.x, pos.y, pos.z,
                        x_axis[0], x_axis[1], x_axis[2],
                        color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.8
                    )
                    self.artists['scatters'].append(arrow_x)

                    # Y-axis (pitch) - green
                    arrow_y = self.ax.quiver(
                        pos.x, pos.y, pos.z,
                        y_axis[0], y_axis[1], y_axis[2],
                        color='green', arrow_length_ratio=0.3, linewidth=2, alpha=0.8
                    )
                    self.artists['scatters'].append(arrow_y)

                    # Z-axis (yaw) - blue
                    arrow_z = self.ax.quiver(
                        pos.x, pos.y, pos.z,
                        z_axis[0], z_axis[1], z_axis[2],
                        color='blue', arrow_length_ratio=0.3, linewidth=2, alpha=0.8
                    )
                    self.artists['scatters'].append(arrow_z)

                    # Add text showing RPY values
                    rpy_text = f'R:{np.degrees(roll):.1f}° P:{np.degrees(pitch):.1f}° Y:{np.degrees(yaw):.1f}°'
                    text = self.ax.text(
                        pos.x, pos.y, pos.z + 0.1,
                        rpy_text, fontsize=7, color=color, weight='bold'
                    )
                    self.artists['texts'].append(text)

            except Exception as e:
                self.get_logger().error(f'Error visualizing target poses: {e}')

        else:
            # No data yet - show waiting message
            text = self.ax.text2D(
                0.5, 0.5, 'Waiting for pose data...',
                transform=self.ax.transAxes,
                fontsize=14, ha='center', color='red'
            )
            self.artists['texts'].append(text)

        return self.artists['lines'] + self.artists['scatters'] + self.artists['texts']


def main(args=None):
    rclpy.init(args=args)

    visualizer = DebugVisualizer()

    # Run ROS2 spin in background thread
    def spin():
        rclpy.spin(visualizer)

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()

    print("\n" + "="*60)
    print("Debug Visualizer Running")
    print("="*60)
    print("3D Visualization: Human pose skeleton")
    print("Coordinate system: ROS coordinates (meters), fixed origin, equal aspect ratio")
    print("Range: X[-1,1m], Y[-1,1m], Z[0,2m]")
    print("Close the plot window to exit")
    print("="*60 + "\n")

    # Create animation
    anim = FuncAnimation(
        visualizer.fig,
        visualizer.update_plot,
        interval=33,  # ~30 FPS
        repeat=True,
        blit=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
