#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import PointCloud2, PointField
import pandas as pd
import sys
import numpy as np
import os
import struct


class MocapUnifiedPublisher(Node):
    def __init__(self, csv_file, traj_idx="0", frame_rate=120.0, pointcloud_dir=None, max_points=5000):
        super().__init__('mocap_unified_publisher')

        # Point cloud downsampling parameter
        self.max_points = max_points

        # Create publishers
        self.arm_publisher = self.create_publisher(PoseArray, '/demo_target_poses', 10)
        self.arm_marker_publisher = self.create_publisher(MarkerArray, '/demo_mocap_markers', 10)
        self.skeleton_marker_publisher = self.create_publisher(MarkerArray, '/skeleton_markers', 10)
        self.frame_info_publisher = self.create_publisher(DiagnosticArray, '/mocap_frame_info', 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, '/human_pointcloud', 10)

        # 用于保证同一帧内所有消息的时间戳一致
        self.last_stamp = None

        # Load data
        self.df = pd.read_csv(csv_file)

        # Handle both numeric and string traj_id
        # If traj_idx is numeric string like "0", try to find by index in unique traj_ids
        all_traj_ids = self.df['traj_id'].unique()
        if traj_idx.isdigit():
            traj_idx_num = int(traj_idx)
            if traj_idx_num < len(all_traj_ids):
                actual_traj_id = all_traj_ids[traj_idx_num]
                self.get_logger().info(f'Using trajectory index {traj_idx_num}: {actual_traj_id}')
                self.traj_data = self.df[self.df['traj_id'] == actual_traj_id].copy()
            else:
                self.get_logger().error(f'Trajectory index {traj_idx_num} out of range (max: {len(all_traj_ids)-1})')
                self.traj_data = pd.DataFrame()
        else:
            # traj_idx is a string, match directly
            self.traj_data = self.df[self.df['traj_id'] == traj_idx].copy()

        if self.traj_data.empty:
            self.get_logger().error(f'No data found for trajectory {traj_idx}')
            return

        # Store pointcloud directory
        self.pointcloud_dir = pointcloud_dir
        if self.pointcloud_dir and os.path.exists(self.pointcloud_dir):
            self.get_logger().info(f'Pointcloud directory: {self.pointcloud_dir}')
        elif self.pointcloud_dir:
            self.get_logger().warning(f'Pointcloud directory not found: {self.pointcloud_dir}')
            self.pointcloud_dir = None

        self.get_logger().info(f'Loaded {len(self.traj_data)} frames for trajectory {traj_idx}')
        
        # Calculate publish interval based on frame rate
        self.publish_interval = 1.0 / frame_rate
        
        # Reset frame index for easy iteration
        self.traj_data.reset_index(drop=True, inplace=True)
        self.current_frame = 0
        
        # Define skeleton joint names (full body for inference_trajectory_gt.csv)
        self.joint_names = [
            'Head', 'Neck', 'R_Shoulder', 'L_Shoulder',
            'R_Elbow', 'L_Elbow', 'R_Hand', 'L_Hand', 'Torso',
            'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee', 'R_Foot', 'L_Foot'
        ]

        # Define skeleton connections (full body)
        self.connections = [
            # Upper body
            ('Torso', 'Neck'),
            ('Neck', 'Head'),
            ('Neck', 'R_Shoulder'),
            ('Neck', 'L_Shoulder'),
            ('R_Shoulder', 'R_Elbow'),
            ('R_Elbow', 'R_Hand'),
            ('L_Shoulder', 'L_Elbow'),
            ('L_Elbow', 'L_Hand'),
            # Lower body
            ('Torso', 'R_Hip'),
            ('Torso', 'L_Hip'),
            ('R_Hip', 'R_Knee'),
            ('R_Knee', 'R_Foot'),
            ('L_Hip', 'L_Knee'),
            ('L_Knee', 'L_Foot'),
            # Pelvis connection
            ('R_Hip', 'L_Hip')
        ]
        
        # Create timer for publishing
        self.timer = self.create_timer(self.publish_interval, self.publish_all_data)
        
    def publish_all_data(self):
        if self.current_frame >= len(self.traj_data):
            # Loop back to beginning
            self.current_frame = 0
            self.get_logger().info('Looping trajectory data')
        
        row = self.traj_data.iloc[self.current_frame]

        # 本帧统一时间戳（供所有消息共用）
        self.last_stamp = self.get_clock().now().to_msg()
        
        # Publish arm poses and markers
        self.publish_arm_data(row)
        
        # Publish skeleton markers
        self.publish_skeleton_data(row)

        # Publish pointcloud if available
        if self.pointcloud_dir:
            self.publish_pointcloud_data(row)

        # 发布同步信息（带 header.stamp）
        self.publish_frame_info(row)

        self.get_logger().info(
            f'Published unified data for frame {self.current_frame} '
            f'(traj {row["traj_id"]}, frame {int(row["frame_id"])})'
        )
        
        self.current_frame += 1
    
    def publish_arm_data(self, row):
        # Get raw positions from CSV trajectory format (convert mm to m)
        left1_pos = np.array([row['Left_L1_x']/1000.0, row['Left_L1_y']/1000.0, row['Left_L1_z']/1000.0])
        left2_pos = np.array([row['Left_L2_x']/1000.0, row['Left_L2_y']/1000.0, row['Left_L2_z']/1000.0])
        right1_pos = np.array([row['Right_R1_x']/1000.0, row['Right_R1_y']/1000.0, row['Right_R1_z']/1000.0])
        right2_pos = np.array([row['Right_R2_x']/1000.0, row['Right_R2_y']/1000.0, row['Right_R2_z']/1000.0])

        # Apply rotation around X-axis by -90 degrees: (x, y, z) -> (x, -z, y)
        left1_pos = self.rotate_x_minus_90(left1_pos)
        left2_pos = self.rotate_x_minus_90(left2_pos)
        right1_pos = self.rotate_x_minus_90(right1_pos)
        right2_pos = self.rotate_x_minus_90(right2_pos)
        
        # Calculate arm centers
        left_center = (left1_pos + left2_pos) / 2.0
        right_center = (right1_pos + right2_pos) / 2.0
        
        # Calculate orientations (direction from point 2 to point 1)
        left_direction = left2_pos - left1_pos
        right_direction = right2_pos - right1_pos
        
        # Create poses with calculated orientations
        left_arm_pose = Pose()
        left_arm_pose.position = Point(x=float(left_center[0]), 
                                      y=float(left_center[1]), 
                                      z=float(left_center[2]))
        left_arm_pose.orientation = self.vector_to_quaternion(left_direction)
        
        right_arm_pose = Pose()
        right_arm_pose.position = Point(x=float(right_center[0]), 
                                       y=float(right_center[1]), 
                                       z=float(right_center[2]))
        right_arm_pose.orientation = self.vector_to_quaternion(right_direction)
        
        # Create PoseArray message
        pose_array = PoseArray()
        pose_array.header.stamp = self.last_stamp
        pose_array.header.frame_id = "world"
        pose_array.poses = [left_arm_pose, right_arm_pose]
        
        # Publish arm poses
        self.arm_publisher.publish(pose_array)
        
        # Create individual point poses for visualization
        left1_pose = Pose()
        left1_pose.position = Point(x=float(left1_pos[0]), y=float(left1_pos[1]), z=float(left1_pos[2]))
        left2_pose = Pose()  
        left2_pose.position = Point(x=float(left2_pos[0]), y=float(left2_pos[1]), z=float(left2_pos[2]))
        right1_pose = Pose()
        right1_pose.position = Point(x=float(right1_pos[0]), y=float(right1_pos[1]), z=float(right1_pos[2]))
        right2_pose = Pose()
        right2_pose.position = Point(x=float(right2_pos[0]), y=float(right2_pos[1]), z=float(right2_pos[2]))
        
        # Publish arm visualization markers
        self.publish_arm_markers(left1_pose, left2_pose, right1_pose, right2_pose, left_arm_pose, right_arm_pose)
    
    def publish_skeleton_data(self, row):
        # Extract joint positions from CSV trajectory format
        joint_poses = {}
        for joint_name in self.joint_names:
            # Get position from CSV (convert mm to m)
            pos = np.array([row[f'{joint_name}_gt_x'] / 1000.0,
                           row[f'{joint_name}_gt_y'] / 1000.0,
                           row[f'{joint_name}_gt_z'] / 1000.0])

            # Apply rotation around X-axis by -90 degrees
            pos = self.rotate_x_minus_90(pos)

            pose = Pose()
            pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            joint_poses[joint_name] = pose

        # Publish skeleton visualization markers
        self.publish_skeleton_markers(joint_poses)

    def rotate_x_minus_90(self, pos):
        """
        Rotate point around X-axis by -90 degrees
        Rotation matrix for -90 degrees around X-axis:
        [1,  0,  0]
        [0,  0,  1]
        [0, -1,  0]
        This transforms: (x, y, z) -> (x, -z, y)
        """
        x, y, z = pos
        return np.array([x, -z, y])

    def vector_to_quaternion(self, direction_vector):
        """Convert a direction vector to a quaternion representing rotation from +X axis"""
        # Normalize the direction vector
        direction = direction_vector / np.linalg.norm(direction_vector)
        
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
    
    def publish_arm_markers(self, left1_pose, left2_pose, right1_pose, right2_pose, left_arm_pose, right_arm_pose):
        marker_array = MarkerArray()
        
        # Create sphere markers for each point
        points = [
            (left1_pose.position, [1.0, 0.0, 0.0]),   # Red
            (left2_pose.position, [1.0, 0.5, 0.0]),   # Orange
            (right1_pose.position, [0.0, 0.0, 1.0]),  # Blue
            (right2_pose.position, [0.0, 0.5, 1.0])   # Light blue
        ]
        
        for i, (pos, color) in enumerate(points):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.last_stamp
            marker.ns = "mocap_points"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = pos
            marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        
        # Create line markers for connections
        # Left arm connection (left1 -> left2)
        left_line = Marker()
        left_line.header.frame_id = "world"
        left_line.header.stamp = self.last_stamp
        left_line.ns = "mocap_lines"
        left_line.id = 0
        left_line.type = Marker.LINE_STRIP
        left_line.action = Marker.ADD
        left_line.points = [left1_pose.position, left2_pose.position]
        left_line.scale.x = 0.02
        left_line.color.r = 1.0
        left_line.color.g = 0.0
        left_line.color.b = 0.0
        left_line.color.a = 0.8
        marker_array.markers.append(left_line)
        
        # Right arm connection (right1 -> right2)
        right_line = Marker()
        right_line.header.frame_id = "world"
        right_line.header.stamp = self.last_stamp
        right_line.ns = "mocap_lines"
        right_line.id = 1
        right_line.type = Marker.LINE_STRIP
        right_line.action = Marker.ADD
        right_line.points = [right1_pose.position, right2_pose.position]
        right_line.scale.x = 0.02
        right_line.color.r = 0.0
        right_line.color.g = 0.0
        right_line.color.b = 1.0
        right_line.color.a = 0.8
        marker_array.markers.append(right_line)
        
        # Add center point markers with orientation arrows
        # Left arm center
        left_center_marker = Marker()
        left_center_marker.header.frame_id = "world"
        left_center_marker.header.stamp = self.last_stamp
        left_center_marker.ns = "arm_centers"
        left_center_marker.id = 0
        left_center_marker.type = Marker.ARROW
        left_center_marker.action = Marker.ADD
        left_center_marker.pose = left_arm_pose
        left_center_marker.scale.x = 0.1
        left_center_marker.scale.y = 0.02
        left_center_marker.scale.z = 0.02
        left_center_marker.color.r = 0.0
        left_center_marker.color.g = 1.0
        left_center_marker.color.b = 0.0
        left_center_marker.color.a = 1.0
        marker_array.markers.append(left_center_marker)
        
        # Right arm center
        right_center_marker = Marker()
        right_center_marker.header.frame_id = "world"
        right_center_marker.header.stamp = self.last_stamp
        right_center_marker.ns = "arm_centers"
        right_center_marker.id = 1
        right_center_marker.type = Marker.ARROW
        right_center_marker.action = Marker.ADD
        right_center_marker.pose = right_arm_pose
        right_center_marker.scale.x = 0.1
        right_center_marker.scale.y = 0.02
        right_center_marker.scale.z = 0.02
        right_center_marker.color.r = 1.0
        right_center_marker.color.g = 1.0
        right_center_marker.color.b = 0.0
        right_center_marker.color.a = 1.0
        marker_array.markers.append(right_center_marker)
        
        # Publish arm markers
        self.arm_marker_publisher.publish(marker_array)
    
    def publish_skeleton_markers(self, joint_poses):
        marker_array = MarkerArray()
        
        # Create sphere markers for each joint
        for i, (joint_name, pose) in enumerate(joint_poses.items()):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.last_stamp
            marker.ns = "skeleton_joints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = pose.position
            marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            # Different colors for different body parts
            if 'Head' in joint_name or 'Neck' in joint_name:
                color = [1.0, 1.0, 0.0]  # Yellow for head/neck
            elif 'L' in joint_name:
                color = [1.0, 0.0, 0.0]  # Red for all left side joints
            elif 'R' in joint_name:
                color = [0.0, 0.0, 1.0]  # Blue for all right side joints
            else:
                color = [0.0, 1.0, 0.0]  # Green for torso
            
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        
        # Create line markers for skeleton connections
        for i, (parent, child) in enumerate(self.connections):
            if parent in joint_poses and child in joint_poses:
                line_marker = Marker()
                line_marker.header.frame_id = "world"
                line_marker.header.stamp = self.last_stamp
                line_marker.ns = "skeleton_bones"
                line_marker.id = i
                line_marker.type = Marker.LINE_STRIP
                line_marker.action = Marker.ADD
                line_marker.points = [joint_poses[parent].position, joint_poses[child].position]
                line_marker.scale.x = 0.02
                
                # Color bones based on body part
                if 'L' in parent or 'L' in child:
                    line_marker.color.r = 1.0
                    line_marker.color.g = 0.0
                    line_marker.color.b = 0.0
                elif 'R' in parent or 'R' in child:
                    line_marker.color.r = 0.0
                    line_marker.color.g = 0.0
                    line_marker.color.b = 1.0
                else:
                    # Central connections (torso, neck, hip connection)
                    line_marker.color.r = 0.0
                    line_marker.color.g = 1.0
                    line_marker.color.b = 0.0
                    
                line_marker.color.a = 0.8
                marker_array.markers.append(line_marker)
        
        # Publish skeleton markers
        self.skeleton_marker_publisher.publish(marker_array)

    def publish_frame_info(self, row):
        """
        发布一个带 header.stamp 的同步消息（/mocap_frame_info），
        包含 traj_idx、csv中的 frame_idx、循环计数 current_frame。
        """
        arr = DiagnosticArray()
        # 统一时间戳（与本帧 markers 一致）
        arr.header.stamp = self.last_stamp

        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = "mocap_frame_info"
        status.message = "current traj/frame"
        status.hardware_id = "mocap_unified_publisher"

        status.values = [
            KeyValue(key="traj_id", value=str(row["traj_id"])),
            KeyValue(key="frame_id_csv", value=str(int(row["frame_id"]))),
            KeyValue(key="frame_idx_loop", value=str(self.current_frame)),
            KeyValue(key="fps", value=str(1.0 / self.publish_interval))
        ]

        arr.status.append(status)
        self.frame_info_publisher.publish(arr)

    def publish_pointcloud_data(self, row):
        """Load and publish pointcloud data synchronized with frame_id"""
        # Parse frame_id: format is "XX_YYYYY", extract YYYYY as the pointcloud file number
        frame_id_str = str(row['frame_id'])
        if '_' in frame_id_str:
            # Format: "03_00267" -> extract 267
            frame_number = int(frame_id_str.split('_')[1])
        else:
            # Fallback: try to convert directly to int
            frame_number = int(frame_id_str)

        # Construct pointcloud file path based on frame number
        pointcloud_file = os.path.join(self.pointcloud_dir, f'{frame_number}.npz')

        if not os.path.exists(pointcloud_file):
            self.get_logger().debug(f'Pointcloud file not found: {pointcloud_file}')
            return

        try:
            # Load pointcloud data
            data = np.load(pointcloud_file)
            points = data['arr_0'].astype(np.float32)  # Convert from float16 to float32

            # Downsample if too many points (for performance at high frame rates)
            original_count = len(points)
            if len(points) > self.max_points:
                indices = np.random.choice(len(points), self.max_points, replace=False)
                points = points[indices]
                if self.current_frame == 0:  # Only log once at start
                    self.get_logger().info(f'Downsampling pointcloud: {original_count} -> {self.max_points} points')

            # Convert from millimeters to meters
            points = points / 1000.0

            # Apply coordinate transformation (rotate -90 degrees around X-axis)
            # Transform each point: (x, y, z) -> (x, -z, y)
            # Use advanced indexing for faster transformation
            transformed_points = np.column_stack([
                points[:, 0],      # x unchanged
                -points[:, 2],     # y = -z
                points[:, 1]       # z = y
            ])

            # Create PointCloud2 message
            pointcloud_msg = PointCloud2()
            pointcloud_msg.header.stamp = self.last_stamp
            pointcloud_msg.header.frame_id = "world"
            pointcloud_msg.height = 1
            pointcloud_msg.width = len(transformed_points)
            pointcloud_msg.is_dense = True
            pointcloud_msg.is_bigendian = False

            # Define fields (XYZ)
            pointcloud_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            pointcloud_msg.point_step = 12  # 4 bytes * 3 fields
            pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width

            # Pack point data using tobytes() - MUCH faster than struct.pack
            # Ensure C-contiguous array for efficient conversion
            pointcloud_msg.data = np.ascontiguousarray(transformed_points, dtype=np.float32).tobytes()

            # Publish pointcloud
            self.pointcloud_publisher.publish(pointcloud_msg)

        except Exception as e:
            self.get_logger().error(f'Error loading pointcloud {pointcloud_file}: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 mocap_unified_publisher.py <csv_file> [traj_id] [frame_rate] [pointcloud_dir] [max_points]")
        print("Example: python3 mocap_unified_publisher.py inference_trajectory_gt_test.csv 3 10.0 /home/oliver/Documents/data/Mocap_dataset/test")
        print("         python3 mocap_unified_publisher.py inference_trajectory_gt_test.csv 3 30.0 /home/oliver/Documents/data/Mocap_dataset/test 30000")
        print("\nParameters:")
        print("  max_points: Maximum points in pointcloud (default: 5000). Lower = faster but less detail")
        return

    csv_file = sys.argv[1]
    traj_idx = sys.argv[2] if len(sys.argv) > 2 else "0"  # Keep as string to support both numeric and string traj_ids
    frame_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0  # Default to 10fps for CSV trajectory data
    pointcloud_dir = sys.argv[4] if len(sys.argv) > 4 else None
    max_points = int(sys.argv[5]) if len(sys.argv) > 5 else 5000  # Default downsample to 50k points

    try:
        publisher = MocapUnifiedPublisher(csv_file, traj_idx, frame_rate, pointcloud_dir, max_points)
        
        if publisher.traj_data.empty:
            return
            
        print(f"Publishing unified mocap data for trajectory {traj_idx} at {frame_rate} Hz")
        print("Topics:")
        print("  - /demo_target_poses: Ground truth robot arm poses")
        print("  - /demo_mocap_markers: Robot arm visualization")
        print("  - /skeleton_markers: Full body human skeleton visualization")
        print("  - /mocap_frame_info: Current traj id and frame id")
        if pointcloud_dir:
            print(f"  - /human_pointcloud: Synchronized pointcloud data (max {max_points} points)")
        print("Press Ctrl+C to stop...")
        
        rclpy.spin(publisher)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()