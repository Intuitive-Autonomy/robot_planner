#!/usr/bin/env python3
"""
Test script for 4-stage trained model with phase prediction
- Loads best model from checkpoint
- Randomly selects a trajectory from test CSV
- Performs inference with sliding window
- Evaluates robot endpoint accuracy and phase prediction
- Visualizes prediction vs ground truth in real-time 3D animation
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from net import get_model
from utils import normalize_to_torso, plane_normal, rot_from_two_vectors
import pandas as pd
from scipy.spatial.transform import Rotation as R


def load_csv_trajectories(csv_file, use_gt=True):
    """Load all trajectories from CSV file

    Args:
        csv_file: Path to CSV file
        use_gt: If True, load GT data (_gt suffix), else load Pred data (_pred suffix)
    """
    df = pd.read_csv(csv_file)
    traj_ids = df['traj_id'].unique()

    trajectories = []
    joint_names = ['Head', 'Neck', 'R_Shoulder', 'L_Shoulder',
                   'R_Elbow', 'L_Elbow', 'R_Hand', 'L_Hand', 'Torso']

    suffix = '_gt' if use_gt else '_pred'

    for traj_id in traj_ids:
        traj_df = df[df['traj_id'] == traj_id].sort_values('frame_id').reset_index(drop=True)

        # Extract joints [T, 27]
        joint_cols = []
        for jname in joint_names:
            joint_cols.extend([f'{jname}{suffix}_x', f'{jname}{suffix}_y', f'{jname}{suffix}_z'])
        joints = traj_df[joint_cols].values.astype(np.float32)

        # Extract arms [T, 12]
        arm_cols = ['Left_L1_x', 'Left_L1_y', 'Left_L1_z',
                   'Left_L2_x', 'Left_L2_y', 'Left_L2_z',
                   'Right_R1_x', 'Right_R1_y', 'Right_R1_z',
                   'Right_R2_x', 'Right_R2_y', 'Right_R2_z']
        arms = traj_df[arm_cols].values.astype(np.float32)

        # Extract phases [T]
        phase_col = f'phase{suffix}' if f'phase{suffix}' in traj_df.columns else 'phase_gt'
        phases = traj_df[phase_col].values.astype(np.int64)

        trajectories.append({
            'traj_id': traj_id,
            'joints': joints,
            'arms': arms,
            'phases': phases
        })

    return trajectories


def load_model(checkpoint_path, cfg, device):
    """Load trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model with same architecture
    model = get_model(
        skeleton_dim=39,
        output_dim=12,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        dropout=cfg['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    stage = checkpoint.get('stage', 'unknown')
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from {checkpoint_path} (Stage {stage}, Epoch {epoch})")

    return model


def apply_data_augmentation(joints, arms, augment_robot_init=True):
    """
    Apply data augmentation: random rotation around Y-axis

    Args:
        joints: [T, 27] joint positions
        arms: [T, 12] arm positions
        augment_robot_init: If True, initialize robot at hardcoded position (2m in front, 1m height)

    Returns:
        aug_joints: [T, 27] augmented joints
        aug_arms: [T, 12] augmented arms
    """
    # Random rotation angle around Y-axis (vertical)
    angle_deg = np.random.uniform(-180, 180)
    rotation = R.from_euler('y', angle_deg, degrees=True)
    rot_matrix = rotation.as_matrix()

    # Reshape to 3D points
    joints_3d = joints.reshape(-1, 9, 3)  # [T, 9, 3]
    arms_3d = arms.reshape(-1, 4, 3)      # [T, 4, 3]

    # Calculate centroid (use torso position of first frame)
    torso_init = joints_3d[0, 8, :]  # [3]

    # Apply rotation around torso
    aug_joints_3d = np.zeros_like(joints_3d)
    for t in range(len(joints_3d)):
        aug_joints_3d[t] = (rot_matrix @ (joints_3d[t] - torso_init).T).T + torso_init

    # Hardcode robot initial position if requested
    if augment_robot_init:
        aug_arms_3d = np.zeros_like(arms_3d)

        # For all frames, set robot position relative to each frame's head
        # This ensures proper orientation towards the human
        arm_length = 400  # mm
        arm_spacing = 200  # mm from center
        y_fixed = 1000  # Fixed Y height (1m)
        distance = 2000  # 2m away from head

        for t in range(len(aug_joints_3d)):
            head_pos = aug_joints_3d[t, 0, :]      # [3] head position at frame t
            r_shoulder = aug_joints_3d[t, 2, :]    # [3] right shoulder
            l_shoulder = aug_joints_3d[t, 3, :]    # [3] left shoulder

            # Calculate human facing direction (perpendicular to shoulder line)
            # Shoulder vector (right to left)
            shoulder_vec = l_shoulder - r_shoulder

            # Direction perpendicular to shoulders in XZ plane
            # Cross product with Y-axis gives forward direction
            forward_dir = np.cross(shoulder_vec, np.array([0.0, 1.0, 0.0]))
            forward_dir_xz = forward_dir.copy()
            forward_dir_xz[1] = 0  # Project to XZ plane

            # Normalize to get direction
            forward_norm = np.linalg.norm(forward_dir_xz)
            if forward_norm < 1e-6:
                # Fallback if shoulders are aligned vertically (shouldn't happen)
                dir_vec = np.array([0.0, 0.0, 1.0])
            else:
                dir_vec = forward_dir_xz / forward_norm

            # Create initial arm configuration (canonical form)
            # Arms arranged in XZ plane, centered at origin
            original_arms = np.array([
                [-arm_spacing, 0, 0],              # Left back (L1)
                [-arm_spacing, 0, arm_length],     # Left front (L2)
                [arm_spacing, 0, 0],               # Right back (R1)
                [arm_spacing, 0, arm_length],      # Right front (R2)
            ], dtype=np.float32)

            # Compute plane normal of original arms
            n_orig = plane_normal(original_arms)

            # Target normal: -Y direction (arms parallel to XZ plane)
            e_y = np.array([0.0, 1.0, 0.0])
            R_align = rot_from_two_vectors(n_orig, -e_y)

            # Compute robot midpoint position
            # Origin: head position
            # Direction: sampled within cone around human facing direction
            # Distance: 2m along this direction
            mid_pos = head_pos + distance * dir_vec
            mid_pos[1] = y_fixed  # Fix Y height at 1m

            # Apply rotation and translation
            C_orig = original_arms.mean(axis=0)
            aug_arms_3d[t] = (R_align @ (original_arms - C_orig).T).T + mid_pos
    else:
        # Apply same rotation to arms
        aug_arms_3d = np.zeros_like(arms_3d)
        for t in range(len(arms_3d)):
            aug_arms_3d[t] = (rot_matrix @ (arms_3d[t] - torso_init).T).T + torso_init

    # Flatten back
    aug_joints = aug_joints_3d.reshape(-1, 27)
    aug_arms = aug_arms_3d.reshape(-1, 12)

    return aug_joints, aug_arms


@torch.no_grad()
def predict_trajectory(model, joints, arms, sequence_length, device):
    """
    Predict full trajectory using sliding window with step=1

    Args:
        model: Trained model
        joints: [T, 27] joint positions in world coordinates
        arms: [T, 12] arm positions in world coordinates
        sequence_length: Length of input window
        device: Device to run on

    Returns:
        pred_arms: [T-seq_len+1, 12] predicted robot endpoints in world coordinates
        pred_phases: [T-seq_len+1] predicted phase labels (argmax)
        phase_probs: [T-seq_len+1, 3] predicted phase probabilities
        torso_positions: [T-seq_len+1, 3] torso positions for visualization
    """
    model.to(device)
    model.eval()

    T = joints.shape[0]
    pred_arms_list = []
    pred_phases_list = []
    phase_probs_list = []
    torso_positions_list = []

    # Reshape joints and arms for normalize_to_torso
    joints_3d = joints.reshape(T, 9, 3)  # [T, 9, 3]
    arms_3d = arms.reshape(T, 4, 3)      # [T, 4, 3]

    # Sliding window prediction with step=1
    for start_idx in range(0, T - sequence_length + 1):
        # Get input window
        window_joints = joints_3d[start_idx:start_idx + sequence_length]  # [seq_len, 9, 3]
        window_arms = arms_3d[start_idx:start_idx + sequence_length]      # [seq_len, 4, 3]

        # Normalize to torso coordinate system (like in training)
        normalized_joints, normalized_arms = normalize_to_torso(window_joints, window_arms)

        # Flatten and concatenate: [seq_len, 27+12=39]
        normalized_joints_flat = normalized_joints.reshape(sequence_length, -1)  # [seq_len, 27]
        normalized_arms_flat = normalized_arms.reshape(sequence_length, -1)      # [seq_len, 12]
        input_seq = np.concatenate([normalized_joints_flat, normalized_arms_flat], axis=1)  # [seq_len, 39]

        # Convert to tensor
        input_tensor = torch.from_numpy(input_seq).unsqueeze(0).float().to(device)

        # Predict next frame (pred_len=1 for single-step prediction)
        pred_robot, pred_phase_logits = model(input_tensor, pred_len=1,
                                              autoregressive=False,
                                              return_phase=True)

        # Convert to numpy
        pred_robot_np = pred_robot.squeeze(0).cpu().numpy()  # [1, 12]
        pred_phase_logits_np = pred_phase_logits.squeeze(0).cpu().numpy()  # [1, 3]

        # Get phase prediction
        phase_probs = torch.softmax(pred_phase_logits, dim=-1).squeeze(0).cpu().numpy()  # [1, 3]
        phase_label = np.argmax(pred_phase_logits_np, axis=-1)  # [1]

        # Transform prediction back to world coordinates
        # Get torso position at the end of the window (last frame)
        torso_pos = window_joints[-1, 8, :]  # [3] - Torso is joint index 8
        pred_robot_world = pred_robot_np.reshape(1, 4, 3) + torso_pos[None, None, :]  # Add torso offset
        pred_robot_world_flat = pred_robot_world.reshape(1, 12)  # [1, 12]

        pred_arms_list.append(pred_robot_world_flat)
        pred_phases_list.append(phase_label)
        phase_probs_list.append(phase_probs)
        torso_positions_list.append(torso_pos)

    # Concatenate predictions
    pred_arms = np.concatenate(pred_arms_list, axis=0)  # [T-seq_len+1, 12]
    pred_phases = np.concatenate(pred_phases_list, axis=0)  # [T-seq_len+1]
    phase_probs = np.concatenate(phase_probs_list, axis=0)  # [T-seq_len+1, 3]
    torso_positions = np.array(torso_positions_list)  # [T-seq_len+1, 3]

    return pred_arms, pred_phases, phase_probs, torso_positions


def evaluate_metrics(pred_arms, pred_phases, gt_arms, gt_phases):
    """
    Evaluate prediction metrics

    Returns:
        metrics: dict with MSE, phase accuracy, illegal transition ratio
    """
    # Robot endpoint MSE
    mse = np.mean((pred_arms - gt_arms) ** 2)

    # Phase accuracy
    phase_acc = np.mean(pred_phases == gt_phases)

    # Illegal transitions
    illegal_transitions = {(0, 2), (1, 0), (2, 0), (2, 1)}
    n_transitions = len(pred_phases) - 1
    n_illegal = 0

    for t in range(n_transitions):
        if (int(pred_phases[t]), int(pred_phases[t+1])) in illegal_transitions:
            n_illegal += 1

    illegal_ratio = n_illegal / max(1, n_transitions)

    return {
        'mse': mse,
        'phase_acc': phase_acc,
        'illegal_ratio': illegal_ratio,
        'n_illegal': n_illegal,
        'n_transitions': n_transitions
    }


def visualize_prediction_only(pred_arms, joints, pred_phases):
    """
    Real-time 3D animation visualization of prediction only (no GT)

    Args:
        pred_arms: [T, 12] predicted robot endpoints
        joints: [T, 27] human joints for context
        pred_phases: [T] predicted phase labels
    """
    # Reshape data
    pred_pts = pred_arms.reshape(-1, 4, 3)  # [T, 4, 3]
    joints_3d = joints.reshape(-1, 9, 3)    # [T, 9, 3]

    T = pred_pts.shape[0]

    # Phase names
    phase_names = ['Approaching', 'Assisting', 'Leaving']
    phase_colors = ['green', 'blue', 'orange']

    # Joint connections for visualization
    joint_connections = [
        (0, 1),  # Head - Neck
        (1, 8),  # Neck - Torso
        (1, 2),  # Neck - R_Shoulder
        (1, 3),  # Neck - L_Shoulder
        (2, 4),  # R_Shoulder - R_Elbow
        (3, 5),  # L_Shoulder - L_Elbow
        (4, 6),  # R_Elbow - R_Hand
        (5, 7),  # L_Elbow - L_Hand
    ]

    # Robot arm connections
    arm_connections = [(0, 1), (2, 3)]  # [lback-lfront, rback-rfront]

    # Setup figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate plot limits
    all_points = np.concatenate([pred_pts.reshape(-1, 3), joints_3d.reshape(-1, 3)], axis=0)
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    # Store artists for efficient updates
    artists = {'lines': [], 'scatters': [], 'texts': []}

    # Set initial view
    ax.view_init(elev=20, azim=45)

    # Set labels and limits (only once)
    ax.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (Up, mm)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=10, fontweight='bold')
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.grid(True, alpha=0.3)

    def update(frame):
        # Remove previous artists
        for artist_list in artists.values():
            for artist in artist_list:
                artist.remove()
        artists['lines'].clear()
        artists['scatters'].clear()
        artists['texts'].clear()

        # Draw human skeleton (gray)
        for conn in joint_connections:
            p1, p2 = joints_3d[frame, conn[0]], joints_3d[frame, conn[1]]
            line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                           'gray', linewidth=2, alpha=0.5)
            artists['lines'].append(line)

        scatter = ax.scatter(joints_3d[frame, :, 0], joints_3d[frame, :, 1], joints_3d[frame, :, 2],
                            c='gray', s=30, alpha=0.5, label='Human Skeleton')
        artists['scatters'].append(scatter)

        # Draw predicted robot arms with different colors (left=red, right=green)
        arm_colors = ['red', 'green']
        arm_labels = ['Left Arm', 'Right Arm']
        for i, conn in enumerate(arm_connections):
            p1, p2 = pred_pts[frame, conn[0]], pred_pts[frame, conn[1]]
            line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                           color=arm_colors[i], linewidth=4, alpha=0.8, label=arm_labels[i])
            artists['lines'].append(line)

        # Left arm endpoints (red)
        scatter_left = ax.scatter(pred_pts[frame, :2, 0], pred_pts[frame, :2, 1], pred_pts[frame, :2, 2],
                                 c='red', s=100, marker='^', edgecolors='darkred', linewidths=2, alpha=0.8)
        artists['scatters'].append(scatter_left)

        # Right arm endpoints (green)
        scatter_right = ax.scatter(pred_pts[frame, 2:, 0], pred_pts[frame, 2:, 1], pred_pts[frame, 2:, 2],
                                   c='green', s=100, marker='^', edgecolors='darkgreen', linewidths=2, alpha=0.8)
        artists['scatters'].append(scatter_right)

        # Get phase info
        pred_phase = int(pred_phases[frame])
        phase_text = f'Predicted Phase: {phase_names[pred_phase]}'
        phase_color = phase_colors[pred_phase]

        # Update title
        ax.set_title(f'Frame {frame+1}/{T} - Augmented Trajectory\n'
                    f'Red: Prediction Only (No GT)',
                    fontsize=12, fontweight='bold')

        # Add phase info text box at top
        text = ax.text2D(0.5, 0.95, phase_text, transform=ax.transAxes,
                        fontsize=14, fontweight='bold', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=phase_color, alpha=0.3, edgecolor='black', linewidth=2))
        artists['texts'].append(text)

        # Update legend only if needed
        if frame == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

        return artists['lines'] + artists['scatters'] + artists['texts']

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=33, repeat=True)

    plt.tight_layout()
    print(f"\nShowing real-time 3D animation ({T} frames)...")
    print("Close the window to exit.")
    plt.show()


def visualize_prediction_3d(gt_arms, pred_arms, joints, gt_phases, pred_phases):
    """
    Real-time 3D animation visualization of prediction vs ground truth

    Args:
        gt_arms: [T, 12] ground truth robot endpoints
        pred_arms: [T, 12] predicted robot endpoints
        joints: [T, 27] human joints for context
        gt_phases: [T] ground truth phase labels
        pred_phases: [T] predicted phase labels
    """
    # Reshape data
    gt_pts = gt_arms.reshape(-1, 4, 3)      # [T, 4, 3]
    pred_pts = pred_arms.reshape(-1, 4, 3)  # [T, 4, 3]
    joints_3d = joints.reshape(-1, 9, 3)    # [T, 9, 3]

    T = gt_pts.shape[0]

    # Phase names
    phase_names = ['Approaching', 'Assisting', 'Leaving']
    phase_colors = ['green', 'blue', 'orange']

    # Joint connections for visualization
    joint_connections = [
        (0, 1),  # Head - Neck
        (1, 8),  # Neck - Torso
        (1, 2),  # Neck - R_Shoulder
        (1, 3),  # Neck - L_Shoulder
        (2, 4),  # R_Shoulder - R_Elbow
        (3, 5),  # L_Shoulder - L_Elbow
        (4, 6),  # R_Elbow - R_Hand
        (5, 7),  # L_Elbow - L_Hand
    ]

    # Robot arm connections
    arm_connections = [(0, 1), (2, 3)]  # [lback-lfront, rback-rfront]

    # Setup figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate plot limits
    all_points = np.concatenate([gt_pts.reshape(-1, 3), joints_3d.reshape(-1, 3)], axis=0)
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    # Store artists for efficient updates
    artists = {'lines': [], 'scatters': [], 'texts': []}

    # Set initial view
    ax.view_init(elev=20, azim=45)

    # Set labels and limits (only once)
    ax.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (Up, mm)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (mm)', fontsize=10, fontweight='bold')
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.grid(True, alpha=0.3)

    def update(frame):
        # Remove previous artists
        for artist_list in artists.values():
            for artist in artist_list:
                artist.remove()
        artists['lines'].clear()
        artists['scatters'].clear()
        artists['texts'].clear()

        # Draw human skeleton (gray)
        for conn in joint_connections:
            p1, p2 = joints_3d[frame, conn[0]], joints_3d[frame, conn[1]]
            line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                           'gray', linewidth=2, alpha=0.5)
            artists['lines'].append(line)

        scatter = ax.scatter(joints_3d[frame, :, 0], joints_3d[frame, :, 1], joints_3d[frame, :, 2],
                            c='gray', s=30, alpha=0.5, label='Human Skeleton')
        artists['scatters'].append(scatter)

        # Draw ground truth robot arms (blue)
        for i, conn in enumerate(arm_connections):
            p1, p2 = gt_pts[frame, conn[0]], gt_pts[frame, conn[1]]
            line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                           'b-', linewidth=4, alpha=0.8, label='GT' if i == 0 else '')
            artists['lines'].append(line)

        scatter = ax.scatter(gt_pts[frame, :, 0], gt_pts[frame, :, 1], gt_pts[frame, :, 2],
                            c='blue', s=100, marker='o', edgecolors='darkblue', linewidths=2, alpha=0.8)
        artists['scatters'].append(scatter)

        # Draw predicted robot arms with different colors (left=red, right=green)
        arm_colors = ['red', 'green']
        arm_labels = ['Pred Left', 'Pred Right']
        for i, conn in enumerate(arm_connections):
            p1, p2 = pred_pts[frame, conn[0]], pred_pts[frame, conn[1]]
            line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                           color=arm_colors[i], linewidth=4, alpha=0.8, label=arm_labels[i])
            artists['lines'].append(line)

        # Left arm endpoints (red)
        scatter_left = ax.scatter(pred_pts[frame, :2, 0], pred_pts[frame, :2, 1], pred_pts[frame, :2, 2],
                                 c='red', s=100, marker='^', edgecolors='darkred', linewidths=2, alpha=0.8)
        artists['scatters'].append(scatter_left)

        # Right arm endpoints (green)
        scatter_right = ax.scatter(pred_pts[frame, 2:, 0], pred_pts[frame, 2:, 1], pred_pts[frame, 2:, 2],
                                   c='green', s=100, marker='^', edgecolors='darkgreen', linewidths=2, alpha=0.8)
        artists['scatters'].append(scatter_right)

        # Calculate MSE for current frame
        mse = np.mean((gt_pts[frame] - pred_pts[frame])**2)

        # Get phase info
        gt_phase = int(gt_phases[frame])
        pred_phase = int(pred_phases[frame])
        phase_match = '✓' if gt_phase == pred_phase else '✗'

        # Add phase visualization text box
        phase_text = (f'GT: {phase_names[gt_phase]} | '
                     f'Pred: {phase_names[pred_phase]} {phase_match}')
        phase_color = phase_colors[pred_phase]

        # Update title
        ax.set_title(f'Frame {frame+1}/{T} - MSE: {mse:.2f} mm²\n'
                    f'Blue: Ground Truth | Red: Prediction',
                    fontsize=12, fontweight='bold')

        # Add phase info text box at top
        text = ax.text2D(0.5, 0.95, phase_text, transform=ax.transAxes,
                        fontsize=14, fontweight='bold', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=phase_color, alpha=0.3, edgecolor='black', linewidth=2))
        artists['texts'].append(text)

        # Update legend only if needed
        if frame == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

        return artists['lines'] + artists['scatters'] + artists['texts']

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=33, repeat=True)

    plt.tight_layout()
    print(f"\nShowing real-time 3D animation ({T} frames)...")
    print("Close the window to exit.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test 4-stage trained model')
    parser.add_argument('--config', type=str, default='configs/30_10.json',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: uses save_dir from config)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--traj_id', type=int, default=None,
                       help='Specific trajectory ID to visualize (default: random)')
    parser.add_argument('--use_pred', action='store_true',
                       help='Use predicted data instead of GT data for testing')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation mode (random rotation + hardcoded robot init, no GT shown)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    print(f"Loaded config from {args.config}")

    # Determine checkpoint path - use latest stage model
    if args.checkpoint is None:
        # Try to find the latest stage model (4 > 3 > 2 > 1)
        save_dir = cfg['save_dir']
        for stage in [4, 3, 2, 1]:
            stage_ckpt = os.path.join(save_dir, f'best_model_stage{stage}.pth')
            if os.path.exists(stage_ckpt):
                args.checkpoint = stage_ckpt
                break
        else:
            # Fallback to old naming
            args.checkpoint = os.path.join(save_dir, 'best_model.pth')

    # Load model
    device = torch.device(args.device)
    model = load_model(args.checkpoint, cfg, device)

    # Load test CSV (use pred or gt based on flag)
    if args.use_pred:
        test_csv = cfg['test_csv_pred']
        data_type = 'Predicted'
    else:
        test_csv = cfg['test_csv_gt']
        data_type = 'Ground Truth'

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    trajectories = load_csv_trajectories(test_csv, use_gt=not args.use_pred)
    print(f"Loaded {len(trajectories)} test trajectories from {test_csv} ({data_type} data)")

    # Select trajectory
    if args.traj_id is not None and args.traj_id < len(trajectories):
        traj = trajectories[args.traj_id]
        print(f"Selected trajectory ID {args.traj_id}")
    else:
        traj = random.choice(trajectories)
        print(f"Randomly selected trajectory")

    joints = traj['joints']  # [T, 27]
    arms = traj['arms']      # [T, 12]
    phases = traj['phases']  # [T]

    print(f"Trajectory length: {len(joints)} frames")

    # Apply data augmentation if requested
    if args.augment:
        print("Applying data augmentation (random rotation + hardcoded robot init)...")
        joints, arms = apply_data_augmentation(joints, arms, augment_robot_init=True)
        print("Augmentation applied!")

    # Predict trajectory (with torso normalization)
    seq_len = cfg['sequence_length']
    print(f"Running inference with sequence_length={seq_len}...")
    pred_arms, pred_phases, phase_probs, torso_positions = predict_trajectory(model, joints, arms, seq_len, device)

    # Align prediction and ground truth lengths
    min_len = min(len(pred_arms), len(arms) - seq_len + 1)
    pred_arms = pred_arms[:min_len]
    pred_phases = pred_phases[:min_len]
    gt_arms = arms[seq_len-1:seq_len-1+min_len]
    gt_phases = phases[seq_len-1:seq_len-1+min_len]
    joints_viz = joints[seq_len-1:seq_len-1+min_len]

    # Visualize based on mode
    if args.augment:
        # Augment mode: show prediction only (no GT)
        print("\n" + "="*60)
        print("Augmented Mode: Showing prediction only (no GT comparison)")
        print("="*60 + "\n")
        visualize_prediction_only(pred_arms, joints_viz, pred_phases)
    else:
        # Normal mode: evaluate metrics and show GT vs Pred
        metrics = evaluate_metrics(pred_arms, pred_phases, gt_arms, gt_phases)

        print(f"\n{'='*60}")
        print(f"Evaluation Metrics:")
        print(f"{'='*60}")
        print(f"Robot MSE:          {metrics['mse']:.4f} mm²")
        print(f"Phase Accuracy:     {metrics['phase_acc']*100:.2f}%")
        print(f"Illegal Transitions: {metrics['n_illegal']}/{metrics['n_transitions']} ({metrics['illegal_ratio']*100:.2f}%)")
        print(f"{'='*60}\n")

        # Visualize
        visualize_prediction_3d(gt_arms, pred_arms, joints_viz, gt_phases, pred_phases)


if __name__ == "__main__":
    main()
