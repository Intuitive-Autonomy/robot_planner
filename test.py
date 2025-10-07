#!/usr/bin/env python3
"""
Test script for augmented data model
- Loads best model from checkpoint
- Randomly selects a trajectory from test set
- Performs inference with sliding window
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
from utils import AugmentedDataProcessor


def load_model_and_config(checkpoint_path):
    """Load trained model and configuration"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['config']

    # Create model
    model = get_model(
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        skeleton_dim=39,
        output_dim=12,
        num_heads=cfg["num_heads"]
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    return model, cfg


@torch.no_grad()
def predict_trajectory(model, inputs, sequence_length, prediction_length, device):
    """
    Predict full trajectory using sliding window with step=1
    Takes only the first predicted frame from each window for smooth output

    Args:
        model: Trained model
        inputs: [T, 39] input sequence
        sequence_length: Length of input window
        prediction_length: Length of prediction window (but only first frame is used)
        device: Device to run on

    Returns:
        predictions: [T, 12] predicted robot endpoints
    """
    model.to(device)
    model.eval()

    T = inputs.shape[0]
    predictions = []

    # Sliding window prediction with step=1 (only take first predicted frame)
    for start_idx in range(0, T - sequence_length + 1):
        # Get input window
        input_window = inputs[start_idx:start_idx + sequence_length]
        input_tensor = torch.from_numpy(input_window).unsqueeze(0).float().to(device)

        # Predict next frame (pred_len=1 for single-step prediction)
        pred = model(input_tensor, pred_len=1, teacher_forcing_ratio=0.0)
        pred_np = pred.squeeze(0).cpu().numpy()  # [1, 12]

        predictions.append(pred_np)

    # Concatenate predictions
    predictions = np.concatenate(predictions, axis=0)  # [T-sequence_length+1, 12]

    return predictions


def visualize_prediction_3d(gt_traj, pred_traj, joints_traj, save_path=None):
    """
    Real-time 3D animation visualization of prediction vs ground truth

    Args:
        gt_traj: [T, 12] ground truth robot endpoints
        pred_traj: [T, 12] predicted robot endpoints
        joints_traj: [T, 27] human joints for context
        save_path: Optional path to save animation (not used, for interface compatibility)
    """
    # Reshape data
    gt_pts = gt_traj.reshape(-1, 4, 3)      # [T, 4, 3]
    pred_pts = pred_traj.reshape(-1, 4, 3)  # [T, 4, 3]
    joints = joints_traj.reshape(-1, 9, 3)  # [T, 9, 3]

    T = gt_pts.shape[0]

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
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate plot limits
    all_points = np.concatenate([gt_pts.reshape(-1, 3), joints.reshape(-1, 3)], axis=0)
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    # Store view angles to preserve user interaction
    view_state = {'elev': 20, 'azim': 45}

    def update(frame):
        # Save current view before clearing
        view_state['elev'] = ax.elev
        view_state['azim'] = ax.azim

        ax.clear()

        # Draw human skeleton (gray)
        for conn in joint_connections:
            p1, p2 = joints[frame, conn[0]], joints[frame, conn[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                   'gray', linewidth=2, alpha=0.5)

        ax.scatter(joints[frame, :, 0], joints[frame, :, 1], joints[frame, :, 2],
                  c='gray', s=30, alpha=0.5, label='Human Skeleton')

        # Draw ground truth robot arms (blue)
        for conn in arm_connections:
            p1, p2 = gt_pts[frame, conn[0]], gt_pts[frame, conn[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                   'b-', linewidth=4, alpha=0.8, label='GT' if conn == arm_connections[0] else '')

        ax.scatter(gt_pts[frame, :, 0], gt_pts[frame, :, 1], gt_pts[frame, :, 2],
                  c='blue', s=100, marker='o', edgecolors='darkblue', linewidths=2, alpha=0.8)

        # Draw predicted robot arms (red)
        for conn in arm_connections:
            p1, p2 = pred_pts[frame, conn[0]], pred_pts[frame, conn[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                   'r-', linewidth=4, alpha=0.8, label='Pred' if conn == arm_connections[0] else '')

        ax.scatter(pred_pts[frame, :, 0], pred_pts[frame, :, 1], pred_pts[frame, :, 2],
                  c='red', s=100, marker='^', edgecolors='darkred', linewidths=2, alpha=0.8)

        # Calculate MSE for current frame
        mse = np.mean((gt_pts[frame] - pred_pts[frame])**2)

        # Set labels and title
        ax.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (Up, mm)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (mm)', fontsize=10, fontweight='bold')
        ax.set_title(f'Frame {frame+1}/{T} - MSE: {mse:.2f} mm²\n'
                    f'Blue: Ground Truth | Red: Prediction',
                    fontsize=12, fontweight='bold')

        # Set consistent axis limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

        ax.grid(True, alpha=0.3)

        # Restore view angles to preserve user interaction
        ax.view_init(elev=view_state['elev'], azim=view_state['azim'])

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=33, repeat=True)

    plt.tight_layout()
    print(f"\nShowing real-time 3D animation ({T} frames)...")
    print("Close the window to exit.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test model on augmented data')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_traj_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--traj_id', type=int, default=None,
                       help='Specific trajectory ID to visualize (default: random)')

    args = parser.parse_args()

    # Load model and config
    model, cfg = load_model_and_config(args.checkpoint)
    device = torch.device(args.device)

    # Load test info
    test_info_path = os.path.join(os.path.dirname(args.checkpoint), "test_info.json")
    if os.path.exists(test_info_path):
        with open(test_info_path, 'r') as f:
            test_info = json.load(f)
        print(f"Loaded test info from {test_info_path}")
    else:
        print("Warning: test_info.json not found, using default test split")
        # Fallback to creating test split
        processor = AugmentedDataProcessor(cfg["data_root"])
        base_names = processor.get_base_names()
        np.random.seed(42)
        shuffled_bases = np.random.permutation(base_names).tolist()
        n_total = len(shuffled_bases)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        test_bases = shuffled_bases[n_train+n_val:]
        test_info = {
            "test_bases": test_bases,
            "data_root": cfg["data_root"],
            "sequence_length": cfg["sequence_length"],
            "prediction_length": cfg["prediction_length"]
        }

    # Get test file paths
    processor = AugmentedDataProcessor(test_info["data_root"])
    file_groups = processor.file_groups
    test_files = [f for base in test_info["test_bases"] for f in file_groups[base]]

    print(f"\nTest set: {len(test_info['test_bases'])} base recordings ({len(test_files)} files)")

    # Randomly select a trajectory
    if args.traj_id is not None and args.traj_id < len(test_files):
        selected_file = test_files[args.traj_id]
        print(f"Selected trajectory ID {args.traj_id}: {os.path.basename(selected_file)}")
    else:
        selected_file = random.choice(test_files)
        print(f"Randomly selected: {os.path.basename(selected_file)}")

    # Load trajectory data (normalized to torso)
    inputs, targets = processor.load_trajectory_file(selected_file)
    print(f"Trajectory length: {len(inputs)} frames (at 30fps)")

    # Extract joints for visualization (first 27 dims of inputs)
    joints_normalized = inputs[:, :27]  # [T, 27]

    # Extract torso position to restore world coordinates
    # Torso is at index 8 in the 9 joints, which is dims 24-27 in the 27D vector
    joints_3d = joints_normalized.reshape(-1, 9, 3)  # [T, 9, 3]
    torso_pos = joints_3d[:, 8, :]  # [T, 3] - torso position (currently at origin)

    # Load original data to get actual torso positions in world coordinates
    inputs_world, targets_world = processor.load_trajectory_file(selected_file, normalize_to_torso_flag=False)
    joints_world = inputs_world[:, :27].reshape(-1, 9, 3)
    torso_world = joints_world[:, 8, :]  # [T, 3] - actual torso positions

    print(f"Data coordinate system: normalized to torso (will restore to world coordinates for visualization)")

    # Predict trajectory
    print(f"Running inference with sequence_length={test_info['sequence_length']}, "
          f"prediction_length={test_info['prediction_length']}...")

    predictions = predict_trajectory(
        model, inputs,
        test_info["sequence_length"],
        test_info["prediction_length"],
        device
    )

    # Align prediction and ground truth lengths
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    targets = targets[:min_len]
    joints_normalized = joints_normalized[:min_len]
    torso_world = torso_world[:min_len]

    # Restore to world coordinates by adding torso positions
    # predictions and targets are in torso-relative coordinates, need to add torso_world
    predictions_world = predictions.reshape(-1, 4, 3) + torso_world[:, None, :]  # [T, 4, 3]
    targets_world = targets.reshape(-1, 4, 3) + torso_world[:, None, :]  # [T, 4, 3]
    joints_world_viz = joints_normalized.reshape(-1, 9, 3) + torso_world[:, None, :]  # [T, 9, 3]

    # Reshape back
    predictions_world = predictions_world.reshape(-1, 12)
    targets_world = targets_world.reshape(-1, 12)
    joints_world_viz = joints_world_viz.reshape(-1, 27)

    # Calculate overall MSE (in normalized coordinates for consistency)
    mse = np.mean((predictions - targets)**2)
    print(f"Overall MSE: {mse:.4f} mm²")
    print(f"Visualizing {min_len} frames in world coordinates...")

    # Visualize in world coordinates
    visualize_prediction_3d(targets_world, predictions_world, joints_world_viz)


if __name__ == "__main__":
    main()
