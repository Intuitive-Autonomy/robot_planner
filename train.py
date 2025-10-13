#!/usr/bin/env python3
"""4-stage training with progressive augmentation and phase prediction"""

import os
import json
import time
import argparse
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from net import get_model
from utils import CombinedLoss


class TrainingLogger:
    """CSV logger for training metrics"""
    def __init__(self, log_dir, config_name):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.csv')

        self.fieldnames = ['timestamp', 'stage', 'epoch', 'phase', 'train_loss', 'val_loss',
                          'robot_loss', 'phase_loss', 'lr', 'augmentation']

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

        print(f"Logging to: {self.log_file}")

    def log(self, stage, epoch, phase, train_loss, val_loss, robot_loss, phase_loss, lr, aug_info=''):
        """Log training metrics"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stage': stage,
                'epoch': epoch,
                'phase': phase,
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'robot_loss': f'{robot_loss:.6f}',
                'phase_loss': f'{phase_loss:.6f}',
                'lr': f'{lr:.6f}',
                'augmentation': aug_info
            })


def get_augmentation_config(stage, epoch, total_stage_epochs, cfg):
    """Get augmentation config for current stage and epoch"""
    progress = epoch / max(1, total_stage_epochs)

    if stage == 1:
        # Noise progression: 0→25mm→50mm in 1:1:2 ratio
        third = total_stage_epochs // 3
        if epoch < third:
            noise_std = cfg['noise_start'] + (cfg['noise_mid'] - cfg['noise_start']) * (epoch / third)
        elif epoch < 2 * third:
            noise_std = cfg['noise_mid']
        else:
            noise_std = cfg['noise_mid'] + (cfg['noise_end'] - cfg['noise_mid']) * ((epoch - 2*third) / third)

        return {'rotate_prob': cfg['rotate_prob'], 'time_scale_prob': 0.0,
                'initial_pos_prob': 0.0, 'noise_std': noise_std}

    elif stage == 2:
        # Progressive: time_scale (0→50%), initial_pos (0→30%)
        return {'rotate_prob': cfg['rotate_prob'],
                'time_scale_prob': cfg['time_scale_prob_stage2_max'] * progress,
                'initial_pos_prob': cfg['initial_pos_prob_stage2_max'] * progress,
                'noise_std': cfg['noise_end']}

    elif stage in [3, 4]:
        # Full augmentation
        return {'rotate_prob': cfg['rotate_prob'],
                'time_scale_prob': cfg['time_scale_prob_max'],
                'initial_pos_prob': cfg['initial_pos_prob_max'],
                'noise_std': cfg['noise_stage3']}

    return {}


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, stage, autoregressive=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_details = {}

    pbar = tqdm(loader, desc=f'S{stage}E{epoch:02d} Train', leave=False)
    for batch_idx, (inputs, targets_robot, targets_phase) in enumerate(pbar):
        inputs = inputs.to(device)
        targets_robot = targets_robot.to(device)
        targets_phase = targets_phase.to(device)

        optimizer.zero_grad()

        # Forward: Stage 1-3 parallel, Stage 4 autoregressive
        pred_robot, pred_phase = model(inputs, pred_len=targets_robot.shape[1],
                                       autoregressive=autoregressive,
                                       y_gt=targets_robot if autoregressive else None,
                                       return_phase=True)

        # Compute loss
        loss, details = criterion(pred_robot, pred_phase, targets_robot, targets_phase)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in details.items():
            loss_details[k] = loss_details.get(k, 0.0) + v

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.2f}'})

    avg_loss = total_loss / len(loader)
    for k in loss_details:
        loss_details[k] /= len(loader)

    return avg_loss, loss_details


@torch.no_grad()
def evaluate(model, loader, criterion, device, autoregressive=False):
    """
    Evaluate on validation set
    Note: Loss is computed in torso-normalized coordinate space,
    which is consistent with training.
    """
    model.eval()
    total_loss = 0.0
    robot_loss = 0.0
    phase_loss = 0.0

    pbar = tqdm(loader, desc='Validation', leave=False)
    for inputs, targets_robot, targets_phase in pbar:
        inputs = inputs.to(device)
        targets_robot = targets_robot.to(device)
        targets_phase = targets_phase.to(device)

        pred_robot, pred_phase = model(inputs, pred_len=targets_robot.shape[1],
                                       autoregressive=autoregressive,
                                       y_gt=None, return_phase=True)

        loss, details = criterion(pred_robot, pred_phase, targets_robot, targets_phase)
        total_loss += loss.item()
        robot_loss += details.get('robot', 0.0)
        phase_loss += details.get('phase', 0.0)

        pbar.set_postfix({'loss': f'{loss.item():.2f}'})

    return total_loss / len(loader), robot_loss / len(loader), phase_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description='4-stage training with progressive augmentation')
    parser.add_argument('--config', type=str, default='configs/30_10.json',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (e.g., best_model_stage3.pth)')
    parser.add_argument('--start_stage', type=int, default=1,
                       help='Stage to start from (1-4, default: 1)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    print(f"Loaded config from {args.config}")
    print(f"Save dir: {cfg['save_dir']}")

    device = torch.device(args.device)
    os.makedirs(cfg['save_dir'], exist_ok=True)

    # Initialize logger
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    logger = TrainingLogger(cfg['save_dir'], config_name)

    # Create model
    model = get_model(
        skeleton_dim=39,
        output_dim=12,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        dropout=cfg['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = os.path.join(cfg['save_dir'], args.resume)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Resumed from {args.resume} (Stage {checkpoint.get('stage', '?')}, Epoch {checkpoint.get('epoch', '?')})")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found, starting from scratch")

    # Create loss
    criterion = CombinedLoss(
        robot_weight=cfg['robot_loss_w'],
        phase_weight=cfg['phase_loss_w'],
        coherence_weight=cfg['coherence_loss_w'],
        speed_weight=cfg['speed_penalty_w'],
        max_speed_mm_per_frame=cfg['max_speed_mm_per_frame'],
        smooth_vel_weight=cfg['smooth_vel_w'],
        smooth_acc_weight=cfg['smooth_acc_w']
    )

    # Track best losses for each stage
    best_val_losses = {1: float('inf'), 2: float('inf'), 3: float('inf'), 4: float('inf')}
    t0 = time.time()

    # Load datasets once (use test as validation)
    from utils import CSVTrajectoryDataset
    print("\nLoading GT training data...")
    train_ds_gt = CSVTrajectoryDataset([cfg['train_csv_gt']], cfg['sequence_length'],
                                       cfg['prediction_length'], use_gt=True, augmentation_config=None)
    print("Loading Pred training data...")
    train_ds_pred = CSVTrajectoryDataset([cfg['train_csv_pred']], cfg['sequence_length'],
                                         cfg['prediction_length'], use_gt=False, augmentation_config=None)

    print("Loading GT validation data...")
    val_ds_gt = CSVTrajectoryDataset([cfg['test_csv_gt']], cfg['sequence_length'],
                                     cfg['prediction_length'], use_gt=True, augmentation_config=None)
    print("Loading Pred validation data...")
    val_ds_pred = CSVTrajectoryDataset([cfg['test_csv_pred']], cfg['sequence_length'],
                                       cfg['prediction_length'], use_gt=False, augmentation_config=None)

    # Create validation loaders (we'll use both GT and Pred for comprehensive evaluation)
    val_loader_gt = torch.utils.data.DataLoader(val_ds_gt, batch_size=cfg['batch_size'],
                                                 shuffle=False, num_workers=cfg['num_workers'])
    val_loader_pred = torch.utils.data.DataLoader(val_ds_pred, batch_size=cfg['batch_size'],
                                                   shuffle=False, num_workers=cfg['num_workers'])

    # ===== Stage 1 =====
    if args.start_stage <= 1:
        print("\n" + "="*60)
        print("Stage 1: GT only + rotation + noise (0→50mm)")
        print("="*60)

        optimizer = optim.Adam(model.parameters(), lr=cfg['stage1_lr'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

        for epoch in range(cfg['stage1_epochs']):
            aug_cfg = get_augmentation_config(1, epoch, cfg['stage1_epochs'], cfg)

            # Stage 1: Use GT data only
            train_ds_gt.set_augmentation_config(aug_cfg)
            train_loader = torch.utils.data.DataLoader(train_ds_gt, batch_size=cfg['batch_size'],
                                                       shuffle=True, num_workers=cfg['num_workers'])

            # Stage 1: parallel prediction (autoregressive=False)
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch+1, 1, autoregressive=False)

            # Evaluate on both GT and Pred data
            val_loss_gt, val_robot_gt, val_phase_gt = evaluate(model, val_loader_gt, criterion, device, autoregressive=False)
            val_loss_pred, val_robot_pred, val_phase_pred = evaluate(model, val_loader_pred, criterion, device, autoregressive=False)
            val_loss = (val_loss_gt + val_loss_pred) / 2  # Average validation loss

            scheduler.step(val_loss)

            aug_info = f"noise={aug_cfg['noise_std']:.1f}mm"
            print(f"[S1E{epoch+1:02d}] train={train_loss:.4f} val_gt={val_loss_gt:.4f} val_pred={val_loss_pred:.4f} "
                  f"rob_gt={val_robot_gt:.4f} rob_pred={val_robot_pred:.4f} {aug_info}")

            # Log metrics
            logger.log(1, epoch+1, 'train', train_loss, val_loss, val_robot_gt, val_phase_gt,
                      optimizer.param_groups[0]['lr'], aug_info)

            # Save stage-specific best
            if val_loss < best_val_losses[1]:
                best_val_losses[1] = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'stage': 1, 'epoch': epoch+1,
                           'val_loss': val_loss, 'config': cfg},
                          os.path.join(cfg['save_dir'], 'best_model_stage1.pth'))
                print(f"  → Saved Stage 1 best (val_avg={val_loss:.4f})")

    # ===== Stage 2 =====
    if args.start_stage <= 2:
        print("\n" + "="*60)
        print("Stage 2: GT/Pred mix (80%→20%) + augmentation progression")
        print("="*60)

        optimizer = optim.Adam(model.parameters(), lr=cfg['stage2_lr'], weight_decay=1e-5)

        for epoch in range(cfg['stage2_epochs']):
            progress = epoch / cfg['stage2_epochs']
            gt_ratio = cfg['stage2_gt_ratio_start'] + (cfg['stage2_gt_ratio_end'] - cfg['stage2_gt_ratio_start']) * progress

            aug_cfg = get_augmentation_config(2, epoch, cfg['stage2_epochs'], cfg)

            # Stage 2: Mix GT and Pred data based on gt_ratio
            train_ds_gt.set_augmentation_config(aug_cfg)
            train_ds_pred.set_augmentation_config(aug_cfg)

            # Create mixed dataset by concatenating subsets
            from torch.utils.data import ConcatDataset, Subset
            n_gt_samples = int(len(train_ds_gt) * gt_ratio)
            n_pred_samples = len(train_ds_gt) - n_gt_samples

            gt_subset = Subset(train_ds_gt, range(n_gt_samples))
            pred_subset = Subset(train_ds_pred, range(n_pred_samples))
            mixed_ds = ConcatDataset([gt_subset, pred_subset])

            train_loader = torch.utils.data.DataLoader(mixed_ds, batch_size=cfg['batch_size'],
                                                       shuffle=True, num_workers=cfg['num_workers'])

            train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch+1, 2, autoregressive=False)

            # Evaluate on both GT and Pred data
            val_loss_gt, val_robot_gt, val_phase_gt = evaluate(model, val_loader_gt, criterion, device, autoregressive=False)
            val_loss_pred, val_robot_pred, val_phase_pred = evaluate(model, val_loader_pred, criterion, device, autoregressive=False)
            val_loss = (val_loss_gt + val_loss_pred) / 2

            aug_info = f"gt={gt_ratio:.2f} time_scale={aug_cfg['time_scale_prob']:.2f} init_pos={aug_cfg['initial_pos_prob']:.2f}"
            print(f"[S2E{epoch+1:02d}] train={train_loss:.4f} val_gt={val_loss_gt:.4f} val_pred={val_loss_pred:.4f} {aug_info}")

            # Log metrics
            logger.log(2, epoch+1, 'train', train_loss, val_loss, val_robot_gt, val_phase_gt,
                      optimizer.param_groups[0]['lr'], aug_info)

            # Save stage-specific best
            if val_loss < best_val_losses[2]:
                best_val_losses[2] = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'stage': 2, 'epoch': epoch+1,
                           'val_loss': val_loss, 'config': cfg},
                          os.path.join(cfg['save_dir'], 'best_model_stage2.pth'))
                print(f"  → Saved Stage 2 best (val_avg={val_loss:.4f})")

    # ===== Stage 3 =====
    if args.start_stage <= 3:
        print("\n" + "="*60)
        print("Stage 3: Pred only + full augmentation")
        print("="*60)

        optimizer = optim.Adam(model.parameters(), lr=cfg['stage3_lr'], weight_decay=1e-5)

        # Stage 3: Progressive phase/coherence weight increase (0.5→1.0, 0.3→1.0)
        phase_weight_start = cfg['phase_loss_w']         # 0.5
        phase_weight_end = 1.0
        coherence_weight_start = cfg['coherence_loss_w'] # 0.3
        coherence_weight_end = 1.0

        for epoch in range(cfg['stage3_epochs']):
            aug_cfg = get_augmentation_config(3, epoch, cfg['stage3_epochs'], cfg)

            # Progressive weight increase
            progress = epoch / max(1, cfg['stage3_epochs'] - 1)
            phase_weight = phase_weight_start + (phase_weight_end - phase_weight_start) * progress
            coherence_weight = coherence_weight_start + (coherence_weight_end - coherence_weight_start) * progress
            criterion.set_phase_weights(phase_weight, coherence_weight)

            # Stage 3: Use Pred data only
            train_ds_pred.set_augmentation_config(aug_cfg)
            train_loader = torch.utils.data.DataLoader(train_ds_pred, batch_size=cfg['batch_size'],
                                                       shuffle=True, num_workers=cfg['num_workers'])

            train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch+1, 3, autoregressive=False)

            # Evaluate on both GT and Pred data
            val_loss_gt, val_robot_gt, val_phase_gt = evaluate(model, val_loader_gt, criterion, device, autoregressive=False)
            val_loss_pred, val_robot_pred, val_phase_pred = evaluate(model, val_loader_pred, criterion, device, autoregressive=False)
            val_loss = (val_loss_gt + val_loss_pred) / 2

            aug_info = f"pred_only ph_w={phase_weight:.2f} coh_w={coherence_weight:.2f}"
            print(f"[S3E{epoch+1:02d}] train={train_loss:.4f} val_gt={val_loss_gt:.4f} val_pred={val_loss_pred:.4f} {aug_info}")

            # Log metrics
            logger.log(3, epoch+1, 'train', train_loss, val_loss, val_robot_gt, val_phase_gt,
                      optimizer.param_groups[0]['lr'], aug_info)

            # Save stage-specific best
            if val_loss < best_val_losses[3]:
                best_val_losses[3] = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'stage': 3, 'epoch': epoch+1,
                           'val_loss': val_loss, 'config': cfg},
                          os.path.join(cfg['save_dir'], 'best_model_stage3.pth'))
                print(f"  → Saved Stage 3 best (val_avg={val_loss:.4f})")

    # ===== Stage 4 =====
    if args.start_stage <= 4:
        print("\n" + "="*60)
        print("Stage 4: Trajectory-level autoregressive + phase coherence (Pred data)")
        print("="*60)

        # Stage 4: Keep phase/coherence weights at 1.0
        criterion.set_phase_weights(1.0, 1.0)

        optimizer = optim.Adam(model.parameters(), lr=cfg['stage4_lr'], weight_decay=1e-5)

        for epoch in range(cfg['stage4_epochs']):
            aug_cfg = get_augmentation_config(4, epoch, cfg['stage4_epochs'], cfg)

            # Stage 4: Use Pred data with smaller batch for trajectory-level autoregressive
            train_ds_pred.set_augmentation_config(aug_cfg)
            train_loader = torch.utils.data.DataLoader(train_ds_pred, batch_size=32,
                                                       shuffle=True, num_workers=cfg['num_workers'])

            # Stage 4: Use autoregressive mode
            train_loss, train_details = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                         epoch+1, 4, autoregressive=True)

            # Evaluate on both GT and Pred data with autoregressive mode
            val_loss_gt, val_robot_gt, val_phase_gt = evaluate(model, val_loader_gt, criterion, device, autoregressive=True)
            val_loss_pred, val_robot_pred, val_phase_pred = evaluate(model, val_loader_pred, criterion, device, autoregressive=True)
            val_loss = (val_loss_gt + val_loss_pred) / 2

            aug_info = f"coh={train_details.get('coherence',0):.3f} spd={train_details.get('speed',0):.3f}"
            print(f"[S4E{epoch+1:02d}] train={train_loss:.4f} val_gt={val_loss_gt:.4f} val_pred={val_loss_pred:.4f} {aug_info}")

            # Log metrics
            logger.log(4, epoch+1, 'train', train_loss, val_loss, val_robot_gt, val_phase_gt,
                      optimizer.param_groups[0]['lr'], aug_info)

            # Save stage-specific best
            if val_loss < best_val_losses[4]:
                best_val_losses[4] = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'stage': 4, 'epoch': epoch+1,
                           'val_loss': val_loss, 'config': cfg},
                          os.path.join(cfg['save_dir'], 'best_model_stage4.pth'))
                print(f"  → Saved Stage 4 best (val_avg={val_loss:.4f})")

    print(f"\n{'='*60}")
    print(f"Done! Time: {time.time()-t0:.1f}s")
    print(f"Stage best losses: S1={best_val_losses[1]:.4f} S2={best_val_losses[2]:.4f} "
          f"S3={best_val_losses[3]:.4f} S4={best_val_losses[4]:.4f}")
    print(f"Saved to: {cfg['save_dir']}/best_model_stage[1-4].pth")
    print("="*60)


if __name__ == "__main__":
    main()
