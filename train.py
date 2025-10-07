#!/usr/bin/env python3
"""
Augmented Data Training:
- Input: [seq_len, 39] = 9 joints (27D) + 4 robot endpoints (12D)
- Output: [pred_len, 12] = 4 robot arm endpoints
- Data: augmented mocap data from global_csv (120fps -> 30fps)
- Split: 8:1:1 train/val/test (no leakage across augmented versions)
- Training: Two-phase (frame-level + trajectory-level)
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from net import get_model
from utils import create_augmented_data_loaders, AugmentedDataset, AugmentedDataProcessor


# ---------- Loss Functions ----------
class EndpointLoss(nn.Module):
    """
    Frame-level loss for robot arm endpoints
    - MSE on endpoint positions
    - First frame weighting
    - Smoothness regularization (velocity/acceleration)
    """
    def __init__(self,
                 first_w: float = 3.0,
                 smooth_vel_w: float = 0.05,
                 smooth_acc_w: float = 0.01):
        super().__init__()
        self.first_w = first_w
        self.sv = smooth_vel_w
        self.sa = smooth_acc_w
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: [B, T, 12] - 4 robot endpoints × 3 coords
        """
        B, T, D = pred.shape
        device = pred.device

        # 1) Endpoint position MSE
        pos = self.mse(pred, gt)

        # 2) First frame extra weight
        first = self.mse(pred[:, :1], gt[:, :1]) if T > 0 else torch.tensor(0.0, device=device)

        # 3) Smoothness regularization
        vel = ((pred[:, 1:] - pred[:, :-1])**2).mean() if T > 1 else torch.tensor(0.0, device=device)
        acc = ((pred[:, 2:] - 2*pred[:, 1:-1] + pred[:, :-2])**2).mean() if T > 2 else torch.tensor(0.0, device=device)

        total = pos + self.first_w * first + self.sv * vel + self.sa * acc
        return total


class TrajectoryJitterLoss(nn.Module):
    """
    Trajectory-level loss with jitter and speed penalties
    Penalizes jerky motion and excessive speed
    """
    def __init__(self,
                 velocity_w: float = 1.0,
                 acceleration_w: float = 2.0,
                 jerk_w: float = 3.0,
                 max_speed_mm_per_s: float = 300.0,
                 data_processor = None):
        super().__init__()
        self.velocity_w = velocity_w
        self.acceleration_w = acceleration_w
        self.jerk_w = jerk_w
        self.max_speed_mm_s = max_speed_mm_per_s
        self.base_fps = 30.0  # Fixed 30fps for augmented data
        self.mse = nn.MSELoss()

    def forward(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """
        pred_traj, gt_traj: [B, T, 12] - full trajectory prediction
        """
        B, T, _ = pred_traj.shape

        # Base MSE loss
        base_loss = self.mse(pred_traj, gt_traj)

        if T < 2:
            return base_loss

        max_speed_mm_per_frame = self.max_speed_mm_s / self.base_fps

        # Calculate velocity (1st order diff)
        pred_vel = pred_traj[:, 1:] - pred_traj[:, :-1]  # [B, T-1, 12]
        gt_vel = gt_traj[:, 1:] - gt_traj[:, :-1]

        # Velocity loss
        velocity_loss = self.mse(pred_vel, gt_vel)

        # Speed penalty
        pred_vel_points = pred_vel.view(B, T-1, 4, 3)  # [B, T-1, 4, 3]
        pred_speeds = torch.norm(pred_vel_points, dim=-1)  # [B, T-1, 4] - mm/frame
        speed_penalty = torch.relu(pred_speeds - max_speed_mm_per_frame).pow(2).mean()

        total_loss = base_loss + self.velocity_w * velocity_loss + speed_penalty

        if T < 3:
            return total_loss

        # Calculate acceleration (2nd order diff)
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]  # [B, T-2, 12]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]

        # Acceleration loss
        acceleration_loss = self.mse(pred_acc, gt_acc)

        total_loss += self.acceleration_w * acceleration_loss

        if T < 4:
            return total_loss

        # Calculate jerk (3rd order diff)
        pred_jerk = pred_acc[:, 1:] - pred_acc[:, :-1]  # [B, T-3, 12]
        gt_jerk = gt_acc[:, 1:] - gt_acc[:, :-1]

        # Jerk loss
        jerk_loss = self.mse(pred_jerk, gt_jerk)

        total_loss += self.jerk_w * jerk_loss

        return total_loss


class AugmentedFullTrajectoryDataset:
    """Full trajectory dataset for Phase 2 training with augmented data"""
    def __init__(self, file_paths, data_root, sequence_length=30, prediction_length=10):
        self.file_paths = file_paths
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.trajectories = []

        processor = AugmentedDataProcessor(data_root)

        # Load all trajectories and create segments
        for file_path in file_paths:
            try:
                inputs, targets = processor.load_trajectory_file(file_path)
                min_length = sequence_length + prediction_length
                if len(inputs) >= min_length:
                    # Create multiple overlapping training segments
                    for start_idx in range(0, len(inputs) - min_length + 1, prediction_length // 2):
                        input_seq = inputs[start_idx:start_idx + sequence_length]
                        target_seq = targets[start_idx + sequence_length:start_idx + sequence_length + prediction_length]
                        self.trajectories.append((input_seq, target_seq))
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue

        print(f"Augmented Full Trajectory Dataset: {len(self.trajectories)} segments (fixed length: {prediction_length})")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        input_seq, target_seq = self.trajectories[idx]
        return torch.from_numpy(input_seq).float(), torch.from_numpy(target_seq).float()


# ---------- Noise Scheduler ----------
def get_progressive_noise_params(epoch, total_epochs):
    """
    Calculate progressive noise parameters based on current epoch

    Stages ratio: 1:1:2 (e.g., 40 epochs = 10|10|20)
    Stage 1: skeleton_std 0→10mm, robot_std 0→10mm
    Stage 2: skeleton_std 10→30mm, robot_std 10→20mm
    Stage 3: skeleton_std 30→50mm, robot_std 20→30mm + spike noise

    Returns:
        noise_std_skeleton_mm, noise_std_robot_mm, noise_spike_prob, noise_spike_magnitude_mm
    """
    # Stage ratio: 1:1:2
    stage1_epochs = total_epochs // 4  # 1/4 of total
    stage2_epochs = total_epochs // 4  # 1/4 of total
    # stage3_epochs = total_epochs // 2  # 2/4 of total (remaining)

    stage1_end = stage1_epochs
    stage2_end = stage1_epochs + stage2_epochs

    if epoch <= stage1_end:
        # Stage 1: 0 to 10mm for both
        progress = epoch / max(1, stage1_end)
        noise_std_skeleton_mm = 0.0 + progress * 10.0
        noise_std_robot_mm = 0.0 + progress * 10.0
        noise_spike_prob = 0.0
        noise_spike_magnitude_mm = 0.0
    elif epoch <= stage2_end:
        # Stage 2: skeleton 10→30mm, robot 10→20mm
        progress = (epoch - stage1_end) / max(1, stage2_end - stage1_end)
        noise_std_skeleton_mm = 10.0 + progress * 20.0
        noise_std_robot_mm = 10.0 + progress * 10.0
        noise_spike_prob = 0.0
        noise_spike_magnitude_mm = 0.0
    else:
        # Stage 3: skeleton 30→50mm, robot 20→30mm + spike noise
        progress = (epoch - stage2_end) / max(1, total_epochs - stage2_end)
        noise_std_skeleton_mm = 30.0 + progress * 20.0
        noise_std_robot_mm = 20.0 + progress * 10.0
        noise_spike_prob = 0.3  # 30% chance to add spike noise
        noise_spike_magnitude_mm = 200.0

    return noise_std_skeleton_mm, noise_std_robot_mm, noise_spike_prob, noise_spike_magnitude_mm


# ---------- Training Functions ----------
def train_one_epoch(model, loader, opt, crit, device, epoch, num_epochs, base_tf=0.6, end_tf=0.0):
    """Train one epoch with teacher forcing decay"""
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{num_epochs}", ncols=120)
    for x, y in pbar:
        x = x.to(device)  # [B, seq_len, 39]
        y = y.to(device)  # [B, pred_len, 12]

        # Teacher forcing decay (linear from base_tf -> end_tf)
        alpha = min(1.0, epoch / max(1, num_epochs//2))
        tf_ratio = base_tf * (1.0 - alpha) + end_tf * alpha

        pred = model(x, pred_len=y.size(1), teacher_forcing_ratio=tf_ratio, y_gt=y)
        loss = crit(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss)
        n += 1
        pbar.set_postfix(loss=float(loss), tf=tf_ratio)
    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, crit, device, epoch, num_epochs):
    """Validation without teacher forcing"""
    model.eval()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Val   {epoch}/{num_epochs}", ncols=120)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        pred = model(x, pred_len=y.size(1), teacher_forcing_ratio=0.0)
        loss = crit(pred, y)
        total += float(loss)
        n += 1
        pbar.set_postfix(loss=float(loss))
    return total / max(1, n)


def train_full_trajectory_epoch(model, loader, opt, crit, device, epoch, num_epochs):
    """Full trajectory training for Phase 2"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Traj Train {epoch}/{num_epochs}", ncols=120)

    for x, y in pbar:
        x = x.to(device)  # [B, seq_len, 39]
        y = y.to(device)  # [B, pred_len, 12]

        opt.zero_grad()

        # Autoregressive prediction with reduced teacher forcing
        pred = model(x, pred_len=y.size(1), teacher_forcing_ratio=0.2)

        # Calculate trajectory-level loss
        loss = crit(pred, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += float(loss)
        n_batches += 1

        pbar.set_postfix(loss=float(loss))

    return total_loss / max(1, n_batches)


def load_config(config_path):
    """Load training configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loaded config from {config_path}")
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train motion prediction model on augmented data')

    parser.add_argument('--config', type=str, default='configs/30_10.json',
                       help='Path to configuration JSON file')

    return parser.parse_args()


# ---------- Main Training Loop ----------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    with open(os.path.join(cfg["save_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create initial augmented data loaders (noise will be updated per epoch)
    print(f"\nLoading augmented data from: {cfg['data_root']}")
    # Note: train_loader will be recreated each epoch with progressive noise
    _, val_loader, test_loader = create_augmented_data_loaders(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        sequence_length=cfg["sequence_length"],
        prediction_length=cfg["prediction_length"],
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        num_workers=cfg["num_workers"],
        noise_std_skeleton_mm=0.0,
        noise_std_robot_mm=0.0,
        noise_spike_prob=0.0,
        noise_spike_magnitude_mm=0.0
    )

    # Model with 39D input (9 joints + 4 robot endpoints)
    skeleton_dim = 39
    model = get_model(
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        skeleton_dim=skeleton_dim,
        output_dim=12,
        num_heads=cfg["num_heads"]
    ).to(device)

    print(f"\nModel created: input_dim={skeleton_dim}, output_dim=12")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== Phase 1: Frame-level training ==========
    print("\n" + "="*60)
    print("Phase 1: Frame-level training with teacher forcing")
    print("="*60)

    crit = EndpointLoss(cfg["first_w"], cfg["smooth_vel_w"], cfg["smooth_acc_w"])
    opt = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8)

    best = float("inf")
    best_path = os.path.join(cfg["save_dir"], "best_model.pth")

    t0 = time.time()
    for epoch in range(1, cfg["num_epochs"] + 1):
        # Get progressive noise parameters for this epoch
        noise_std_skel, noise_std_robot, spike_prob, spike_mag = get_progressive_noise_params(epoch, cfg["num_epochs"])

        # Recreate train_loader with updated noise parameters
        train_loader, _, _ = create_augmented_data_loaders(
            data_root=cfg["data_root"],
            batch_size=cfg["batch_size"],
            sequence_length=cfg["sequence_length"],
            prediction_length=cfg["prediction_length"],
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            num_workers=cfg["num_workers"],
            noise_std_skeleton_mm=noise_std_skel,
            noise_std_robot_mm=noise_std_robot,
            noise_spike_prob=spike_prob,
            noise_spike_magnitude_mm=spike_mag
        )

        # Display current noise parameters (1:1:2 ratio)
        stage1_end = cfg["num_epochs"] // 4
        stage2_end = cfg["num_epochs"] // 2
        stage = 1 if epoch <= stage1_end else (2 if epoch <= stage2_end else 3)
        print(f"[Epoch {epoch:03d}] Stage {stage}: skel_std={noise_std_skel:.1f}mm, robot_std={noise_std_robot:.1f}mm, spike_prob={spike_prob:.2f}")

        tr = train_one_epoch(model, train_loader, opt, crit, device, epoch, cfg["num_epochs"])
        va = validate(model, val_loader, crit, device, epoch, cfg["num_epochs"])
        sched.step(va)
        print(f"[Phase1][Epoch {epoch:03d}] train={tr:.6f}  val={va:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

        if va < best:
            best = va
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, best_path)
            print(f"[Phase1] Saved best to {best_path} (val={best:.6f})")

    print(f"Phase 1 done. Best val={best:.6f}. Elapsed {time.time()-t0:.1f}s.")

    # ========== Phase 2: Trajectory-level training ==========
    print("\n" + "="*60)
    print("Phase 2: Full trajectory training with jitter/speed penalties")
    print("="*60)

    # Get file lists for trajectory training
    processor = AugmentedDataProcessor(cfg["data_root"])
    base_names = processor.get_base_names()
    file_groups = processor.file_groups

    # Use train split for trajectory training
    np.random.seed(42)
    shuffled_bases = np.random.permutation(base_names).tolist()
    n_total = len(shuffled_bases)
    n_train = int(n_total * cfg["train_ratio"])
    train_bases = shuffled_bases[:n_train]
    train_files = [f for base in train_bases for f in file_groups[base]]

    # Create trajectory dataset
    full_traj_dataset = AugmentedFullTrajectoryDataset(
        train_files, cfg["data_root"],
        cfg["sequence_length"], cfg["prediction_length"]
    )

    traj_loader = DataLoader(
        full_traj_dataset,
        batch_size=cfg.get("traj_batch_size", 16),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True
    )

    # Trajectory-level loss
    traj_crit = TrajectoryJitterLoss(
        velocity_w=cfg.get("traj_velocity_w", 1.0),
        acceleration_w=cfg.get("traj_acceleration_w", 2.0),
        jerk_w=cfg.get("traj_jerk_w", 3.0),
        max_speed_mm_per_s=cfg.get("traj_max_speed_mm_s", 300.0),
        data_processor=None
    )

    # Use smaller learning rate
    traj_lr = cfg.get("traj_learning_rate", cfg["learning_rate"] * 0.1)
    traj_opt = optim.Adam(model.parameters(), lr=traj_lr, weight_decay=1e-5)

    traj_epochs = cfg.get("traj_epochs", max(5, cfg["num_epochs"] // 10))
    traj_best = float("inf")
    traj_best_path = os.path.join(cfg["save_dir"], "best_traj_model.pth")

    print(f"Trajectory training: {traj_epochs} epochs, lr={traj_lr:.2e}")
    print(f"Traj loader: {len(full_traj_dataset)} segments, batch={cfg.get('traj_batch_size', 16)}")

    for epoch in range(1, traj_epochs + 1):
        traj_loss = train_full_trajectory_epoch(
            model, traj_loader, traj_opt, traj_crit, device,
            epoch, traj_epochs
        )

        # Validate on original validation set
        val_loss = validate(model, val_loader, crit, device, epoch, traj_epochs)

        print(f"[Phase2][Epoch {epoch:03d}] traj_loss={traj_loss:.6f}  val={val_loss:.6f}  lr={traj_opt.param_groups[0]['lr']:.2e}")

        if val_loss < traj_best:
            traj_best = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, traj_best_path)
            print(f"[Phase2] Saved best trajectory model to {traj_best_path} (val={traj_best:.6f})")

    print(f"\nPhase 2 done. Best traj val={traj_best:.6f}. Total elapsed {time.time()-t0:.1f}s.")
    print("\nTraining completed! Models saved:")
    print(f"  - Frame-level best: {best_path}")
    print(f"  - Trajectory-level best: {traj_best_path}")

    # Save test loader info for test.py
    test_info = {
        "test_bases": shuffled_bases[n_train+int(n_total*cfg["val_ratio"]):],
        "data_root": cfg["data_root"],
        "sequence_length": cfg["sequence_length"],
        "prediction_length": cfg["prediction_length"]
    }
    test_info_path = os.path.join(cfg["save_dir"], "test_info.json")
    with open(test_info_path, "w") as f:
        json.dump(test_info, f, indent=2)
    print(f"  - Test info: {test_info_path}")


if __name__ == "__main__":
    main()
