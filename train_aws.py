#!/usr/bin/env python3
"""
AWS-optimized training script using preprocessed NPZ data (train/val/test splits)
Much faster data loading compared to reading 2000+ CSV files
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

from net import get_model
from utils import PreprocessedAugmentedDataset, AugmentedDataProcessor

# Import loss functions and training utilities from original train.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    EndpointLoss,
    TrajectoryJitterLoss,
    train_one_epoch,
    validate,
    train_full_trajectory_epoch,
    get_progressive_noise_params,
    AugmentedFullTrajectoryDataset
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train on AWS using preprocessed NPZ data')

    parser.add_argument('--config', type=str, default='configs/30_10.json',
                       help='Path to configuration JSON file')
    parser.add_argument('--npz_dir', type=str, default='/mnt/fsx/fsx/global_csv/preprocessed',
                       help='Directory containing preprocessed NPZ files (train/val/test)')

    return parser.parse_args()


def create_loaders_from_npz(npz_dir, cfg, data_root,
                            noise_std_skeleton_mm=0.0,
                            noise_std_robot_mm=0.0,
                            noise_spike_prob=0.0,
                            noise_spike_magnitude_mm=0.0):
    """Create data loaders from preprocessed NPZ files"""

    # Get train/val/test file lists (for creating the mapping)
    processor = AugmentedDataProcessor(data_root)
    base_names = processor.get_base_names()
    file_groups = processor.file_groups

    np.random.seed(42)
    shuffled_bases = np.random.permutation(base_names).tolist()

    n_total = len(shuffled_bases)
    n_train = int(n_total * cfg["train_ratio"])
    n_val = int(n_total * cfg["val_ratio"])

    train_bases = shuffled_bases[:n_train]
    val_bases = shuffled_bases[n_train:n_train+n_val]
    test_bases = shuffled_bases[n_train+n_val:]

    train_files = [f for base in train_bases for f in file_groups[base]]
    val_files = [f for base in val_bases for f in file_groups[base]]
    test_files = [f for base in test_bases for f in file_groups[base]]

    # Create datasets from NPZ files
    train_npz = os.path.join(npz_dir, "preprocessed_train.npz")
    val_npz = os.path.join(npz_dir, "preprocessed_val.npz")
    test_npz = os.path.join(npz_dir, "preprocessed_test.npz")

    train_ds = PreprocessedAugmentedDataset(
        train_npz, train_files, data_root,
        cfg["sequence_length"], cfg["prediction_length"],
        augment=True,
        noise_std_skeleton_mm=noise_std_skeleton_mm,
        noise_std_robot_mm=noise_std_robot_mm,
        noise_spike_prob=noise_spike_prob,
        noise_spike_magnitude_mm=noise_spike_magnitude_mm
    )

    val_ds = PreprocessedAugmentedDataset(
        val_npz, val_files, data_root,
        cfg["sequence_length"], cfg["prediction_length"],
        augment=False,
        noise_std_skeleton_mm=0.0,
        noise_std_robot_mm=0.0,
        noise_spike_prob=0.0,
        noise_spike_magnitude_mm=0.0
    )

    test_ds = PreprocessedAugmentedDataset(
        test_npz, test_files, data_root,
        cfg["sequence_length"], cfg["prediction_length"],
        augment=False,
        noise_std_skeleton_mm=0.0,
        noise_std_robot_mm=0.0,
        noise_spike_prob=0.0,
        noise_spike_magnitude_mm=0.0
    )

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)

    return train_loader, val_loader, test_loader


def main():
    args = parse_args()

    # Check if NPZ files exist
    train_npz = os.path.join(args.npz_dir, "preprocessed_train.npz")
    val_npz = os.path.join(args.npz_dir, "preprocessed_val.npz")
    test_npz = os.path.join(args.npz_dir, "preprocessed_test.npz")

    missing_files = []
    for npz_file in [train_npz, val_npz, test_npz]:
        if not os.path.exists(npz_file):
            missing_files.append(npz_file)

    if missing_files:
        print(f"ERROR: Preprocessed NPZ files not found:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease run preprocessing first:")
        print(f"  python preprocess_data.py --data_root /mnt/fsx/fsx/global_csv --output_dir {args.npz_dir}")
        return

    # Load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    print(f"Loaded config from {args.config}")

    os.makedirs(cfg["save_dir"], exist_ok=True)
    with open(os.path.join(cfg["save_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create data loaders from preprocessed NPZ (FAST!)
    print(f"\nLoading preprocessed data from: {args.npz_dir}")
    _, val_loader, test_loader = create_loaders_from_npz(
        args.npz_dir, cfg, cfg["data_root"],
        noise_std_skeleton_mm=0.0,
        noise_std_robot_mm=0.0,
        noise_spike_prob=0.0,
        noise_spike_magnitude_mm=0.0
    )

    # Model
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
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, verbose=True)

    best = float("inf")
    best_path = os.path.join(cfg["save_dir"], "best_model.pth")

    t0 = time.time()
    for epoch in range(1, cfg["num_epochs"] + 1):
        # Get progressive noise parameters
        noise_std_skel, noise_std_robot, spike_prob, spike_mag = get_progressive_noise_params(
            epoch, cfg["num_epochs"]
        )

        # Recreate train_loader with updated noise (FAST with NPZ!)
        train_loader, _, _ = create_loaders_from_npz(
            args.npz_dir, cfg, cfg["data_root"],
            noise_std_skeleton_mm=noise_std_skel,
            noise_std_robot_mm=noise_std_robot,
            noise_spike_prob=spike_prob,
            noise_spike_magnitude_mm=spike_mag
        )

        # Display current noise parameters
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

    # Save test loader info
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
