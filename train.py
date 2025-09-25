#!/usr/bin/env python3
"""
CSV Trajectory Training:
- Input: [seq_len, 27] upper body joints (9 joints * 3 coords)
- Output: [pred_len, 12] arm endpoints (4 endpoints * 3 coords)
- Data: inference_trajectory.csv with trajectory-level training
- Augmentation: 100% rotation, scaling, time scaling, light noise
- Training: Autoregressive + Teacher Forcing (decays with epochs)
- Phase 1: Frame-level training
- Phase 2: Trajectory-level training with jitter penalties
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from net import get_model
from utils import create_csv_data_loaders, CSVTrajectoryProcessor, TrajectoryTrainer
from torch.utils.data import DataLoader

# ---------- loss ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

class EndpointLoss(nn.Module):
    """
    总损失 = 端点位置 MSE
           + 首帧权重项
           + 平滑正则（速度/加速度）
           + 段长一致性（L2-L1、R2-R1 与 GT 的长度 MSE）* length_w
           + 左右对应端点间距带约束（L1↔R1、L2↔R2 落在 [400mm, 500mm] 之外的铰链惩罚）* width_w

    端点顺序: [L1, L2, R1, R2] * 3
    """
    def __init__(self,
                 first_w: float = 3.0,
                 smooth_vel_w: float = 0.05,
                 smooth_acc_w: float = 0.01,
                 length_w: float = 5.0,   # 你要求的：比端点 MSE 大 5 倍
                 width_w: float = 3.0,    # 新增：左右对应端点间距带约束的权重（可调）
                 width_low_mm: float = 400.0,   # 40 cm
                 width_high_mm: float = 500.0   # 50 cm
                 ):
        super().__init__()
        self.first_w = first_w
        self.sv = smooth_vel_w
        self.sa = smooth_acc_w
        self.length_w = length_w
        self.width_w = width_w
        self.width_low = width_low_mm
        self.width_high = width_high_mm
        self.mse = nn.MSELoss()

    @staticmethod
    def _band_hinge_sq(d: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """
        区间 [lo, hi] 的铰链平方惩罚：
          loss = (ReLU(lo - d))^2 + (ReLU(d - hi))^2
        d: [B, T] 或任意形状
        """
        lower = F.relu(lo - d)
        upper = F.relu(d - hi)
        return lower.square() + upper.square()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: [B, T, 12]，按 [L1, L2, R1, R2] * 3
        """
        B, T, D = pred.shape
        device = pred.device

        # 1) 端点位置 MSE（主项）
        pos = self.mse(pred, gt)

        # 2) 首帧额外权重（稳定起步）
        first = self.mse(pred[:, :1], gt[:, :1]) if T > 0 else torch.tensor(0.0, device=device)

        # 3) 平滑正则
        vel = ((pred[:, 1:] - pred[:, :-1])**2).mean() if T > 1 else torch.tensor(0.0, device=device)
        acc = ((pred[:, 2:] - 2*pred[:, 1:-1] + pred[:, :-2])**2).mean() if T > 2 else torch.tensor(0.0, device=device)

        # reshape 成 [B, T, 4, 3]
        pred_pts = pred.view(B, T, 4, 3)
        gt_pts   = gt.view(B, T, 4, 3)

        # 4) 段长一致性：预测与 GT 的左右臂长度（L:1-0，R:3-2）一致
        pred_len_left  = torch.norm(pred_pts[:, :, 1] - pred_pts[:, :, 0], dim=-1)  # [B,T]
        pred_len_right = torch.norm(pred_pts[:, :, 3] - pred_pts[:, :, 2], dim=-1)  # [B,T]
        gt_len_left    = torch.norm(gt_pts[:, :, 1] - gt_pts[:, :, 0], dim=-1)      # [B,T]
        gt_len_right   = torch.norm(gt_pts[:, :, 3] - gt_pts[:, :, 2], dim=-1)      # [B,T]
        len_left_mse   = self.mse(pred_len_left,  gt_len_left)
        len_right_mse  = self.mse(pred_len_right, gt_len_right)
        len_mse = 0.5 * (len_left_mse + len_right_mse)

        # 5) 左右对应端点间距“带约束”：L1↔R1、L2↔R2 均应落在 [400mm, 500mm]
        pred_lr1 = torch.norm(pred_pts[:, :, 0] - pred_pts[:, :, 2], dim=-1)  # L1 <-> R1
        pred_lr2 = torch.norm(pred_pts[:, :, 1] - pred_pts[:, :, 3], dim=-1)  # L2 <-> R2
        band1 = self._band_hinge_sq(pred_lr1, self.width_low, self.width_high).mean()
        band2 = self._band_hinge_sq(pred_lr2, self.width_low, self.width_high).mean()
        band_penalty = 0.5 * (band1 + band2)

        # 总损失
        total = (
            pos
            + self.length_w * len_mse
            + self.width_w  * band_penalty
            + self.first_w  * first
            + self.sv * vel
            + self.sa * acc
        )
        return total


class TrajectoryJitterLoss(nn.Module):
    """
    全轨迹训练的损失函数，惩罚抖动和过快运动
    支持动态计算基于数据增强的有效帧率
    """
    def __init__(self, 
                 velocity_w: float = 1.0,     # 速度惩罚权重
                 acceleration_w: float = 2.0,  # 加速度惩罚权重
                 jerk_w: float = 3.0,         # 抖动惩罚权重
                 max_speed_mm_per_s: float = 300.0,  # 最大速度 mm/s
                 data_processor: CSVTrajectoryProcessor = None):  # 数据处理器，用于获取base_fps
        super().__init__()
        self.velocity_w = velocity_w
        self.acceleration_w = acceleration_w  
        self.jerk_w = jerk_w
        self.max_speed_mm_s = max_speed_mm_per_s
        self.base_fps = 10.0  # Fixed 10fps for CSV trajectory data
        self.mse = nn.MSELoss()
        print(f"TrajectoryJitterLoss: using base_fps={self.base_fps:.1f} for trajectory data")
        
    def estimate_effective_fps(self, gt_traj: torch.Tensor) -> float:
        """
        基于真实轨迹估算有效帧率
        通过分析运动速度模式来推断时间增强的影响
        """
        if gt_traj.size(1) < 2:
            return self.base_fps
            
        # 计算真实轨迹的速度统计
        gt_vel_points = (gt_traj[:, 1:] - gt_traj[:, :-1]).view(-1, 4, 3)  # [B*(T-1), 4, 3]
        gt_speeds = torch.norm(gt_vel_points, dim=-1)  # [B*(T-1), 4]
        
        # 使用速度统计估算有效帧率
        # 如果平均速度比基础帧率下的预期速度高，说明被加速了
        # 如果平均速度比基础帧率下的预期速度低，说明被减速了
        mean_speed = gt_speeds.mean().item()
        
        # 假设在基础帧率下的典型速度是 max_speed/5 (经验值)
        expected_speed_per_frame = self.max_speed_mm_s / self.base_fps / 5
        
        # 估算有效帧率倍数
        if mean_speed > 0 and expected_speed_per_frame > 0:
            speed_ratio = mean_speed / expected_speed_per_frame
            # 限制在合理范围内 [0.5, 2.0]
            speed_ratio = max(0.5, min(2.0, speed_ratio))
            effective_fps = self.base_fps * speed_ratio
        else:
            effective_fps = self.base_fps
            
        return effective_fps
        
    def forward(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """
        pred_traj, gt_traj: [B, T, 12] - 完整轨迹预测
        """
        B, T, _ = pred_traj.shape
        
        # 基础MSE损失
        base_loss = self.mse(pred_traj, gt_traj)
        
        if T < 2:
            return base_loss
        
        # 动态估算有效帧率
        effective_fps = self.estimate_effective_fps(gt_traj)
        max_speed_mm_per_frame = self.max_speed_mm_s / effective_fps
            
        # 计算速度 (一阶差分)
        pred_vel = pred_traj[:, 1:] - pred_traj[:, :-1]  # [B, T-1, 12]
        gt_vel = gt_traj[:, 1:] - gt_traj[:, :-1]
        
        # 速度损失
        velocity_loss = self.mse(pred_vel, gt_vel)
        
        # 速度过快惩罚 - 使用动态计算的帧率
        pred_vel_points = pred_vel.view(B, T-1, 4, 3)  # [B, T-1, 4, 3]
        pred_speeds = torch.norm(pred_vel_points, dim=-1)  # [B, T-1, 4] - mm/frame
        speed_penalty = torch.relu(pred_speeds - max_speed_mm_per_frame).pow(2).mean()
        
        total_loss = base_loss + self.velocity_w * velocity_loss + speed_penalty
        
        if T < 3:
            return total_loss
            
        # 计算加速度 (二阶差分)
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]  # [B, T-2, 12]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        
        # 加速度损失
        acceleration_loss = self.mse(pred_acc, gt_acc)
        
        total_loss += self.acceleration_w * acceleration_loss
        
        if T < 4:
            return total_loss
            
        # 计算抖动 (三阶差分)
        pred_jerk = pred_acc[:, 1:] - pred_acc[:, :-1]  # [B, T-3, 12]
        gt_jerk = gt_acc[:, 1:] - gt_acc[:, :-1]
        
        # 抖动损失
        jerk_loss = self.mse(pred_jerk, gt_jerk)
        
        total_loss += self.jerk_w * jerk_loss
        
        return total_loss


class CSVFullTrajectoryDataset:
    """CSV trajectory dataset for full trajectory training"""
    def __init__(self, csv_path: str, traj_ids: list, sequence_length: int = 30, prediction_length: int = 10):
        self.processor = CSVTrajectoryProcessor(csv_path)
        self.traj_ids = traj_ids
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.trajectories = []

        # Collect all possible training segments
        for traj_id in traj_ids:
            try:
                joints, arms = self.processor.get_trajectory_data(traj_id)
                min_length = sequence_length + prediction_length
                if len(joints) >= min_length:
                    # Create multiple overlapping training segments for each trajectory
                    for start_idx in range(0, len(joints) - min_length + 1, prediction_length // 2):  # 50% overlap
                        input_seq = joints[start_idx:start_idx + sequence_length]      # [seq_len, 27]
                        target_seq = arms[start_idx + sequence_length:start_idx + sequence_length + prediction_length]  # [pred_len, 12]
                        self.trajectories.append((input_seq, target_seq))
            except ValueError as e:
                print(f"Warning: Could not load trajectory {traj_id}: {e}")
                continue

        print(f"CSV Full trajectory dataset: {len(self.trajectories)} trajectory segments (fixed length: {prediction_length})")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        input_seq, target_seq = self.trajectories[idx]
        return torch.from_numpy(input_seq).float(), torch.from_numpy(target_seq).float()


# ---------- train/val ----------
def train_one_epoch(model, loader, opt, crit, device, epoch, num_epochs, base_tf=0.6, end_tf=0.0):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{num_epochs}", ncols=120)
    for x, y in pbar:
        x = x.to(device)     # [B, Tin, 63]
        y = y.to(device)     # [B, Tout, 12]
        # teacher forcing 衰减（线性由 base_tf -> end_tf）
        alpha = min(1.0, epoch / max(1, num_epochs//2))
        tf_ratio = base_tf * (1.0 - alpha) + end_tf * alpha

        pred = model(x, pred_len=y.size(1), teacher_forcing_ratio=tf_ratio, y_gt=y)
        loss = crit(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss); n += 1
        pbar.set_postfix(loss=float(loss), tf=tf_ratio)
    return total / max(1, n)

@torch.no_grad()
def validate(model, loader, crit, device, epoch, num_epochs):
    model.eval()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Val   {epoch}/{num_epochs}", ncols=120)
    for x, y in pbar:
        x = x.to(device); y = y.to(device)
        pred = model(x, pred_len=y.size(1), teacher_forcing_ratio=0.0)
        loss = crit(pred, y)
        total += float(loss); n += 1
        pbar.set_postfix(loss=float(loss))
    return total / max(1, n)


def train_full_trajectory_epoch(model, loader, opt, crit, device, epoch, num_epochs):
    """全轨迹训练，使用标准DataLoader处理固定长度序列"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"Traj Train {epoch}/{num_epochs}", ncols=120)
    
    for x, y in pbar:
        x = x.to(device)  # [B, seq_len, 63]
        y = y.to(device)  # [B, pred_len, 12]
        
        opt.zero_grad()
        
        # 使用自回归方式预测，减少teacher forcing比例
        pred = model(x, pred_len=y.size(1), teacher_forcing_ratio=0.2)  # [B, pred_len, 12]
        
        # 计算轨迹级损失
        loss = crit(pred, y)
        loss.backward()
        
        # 梯度裁剪
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
    parser = argparse.ArgumentParser(description='Train motion capture prediction model')
    
    parser.add_argument('--config', type=str, default='configs/csv_trajectory.json',
                       help='Path to configuration JSON file (default: configs/csv_trajectory.json)')
    
    return parser.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    with open(os.path.join(cfg["save_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use CSV trajectory dataset
    print(f"Using CSV trajectory dataset: {cfg['csv_path']}")
    train_loader, val_loader = create_csv_data_loaders(
        cfg["csv_path"],
        batch_size=cfg["batch_size"],
        sequence_length=cfg["sequence_length"],
        prediction_length=cfg["prediction_length"],
        val_ratio=cfg["val_ratio"],
        num_workers=cfg["num_workers"]
    )
    skeleton_dim = 27  # 9 upper body joints * 3 coordinates

    model = get_model(hidden_dim=cfg["hidden_dim"],
                      num_layers=cfg["num_layers"],
                      dropout=cfg["dropout"],
                      skeleton_dim=skeleton_dim,
                      output_dim=12,
                      num_heads=cfg["num_heads"]).to(device)

    crit = EndpointLoss(cfg["first_w"], cfg["smooth_vel_w"], cfg["smooth_acc_w"])
    opt = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, verbose=True)

    best = float("inf")
    best_path = os.path.join(cfg["save_dir"], "best_model.pth")

    t0 = time.time()
    for epoch in range(1, cfg["num_epochs"] + 1):
        tr = train_one_epoch(model, train_loader, opt, crit, device, epoch, cfg["num_epochs"])
        va = validate(model, val_loader, crit, device, epoch, cfg["num_epochs"])
        sched.step(va)
        print(f"[Human][Epoch {epoch:03d}] train={tr:.6f}  val={va:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

        if va < best:
            best = va
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, best_path)
            print(f"[Human] Saved best to {best_path} (val={best:.6f})")

    print(f"Phase 1 (Frame-level) done. Best val={best:.6f}. Elapsed {time.time()-t0:.1f}s.")
    
    # Phase 2: 轨迹级训练，惩罚抖动和过快运动
    print("\n" + "="*60)
    print("Phase 2: Full trajectory training with jitter/speed penalties")
    print("="*60)

    # Create trajectory-level dataset for CSV data - using all trajectories
    df = pd.read_csv(cfg["csv_path"])
    traj_ids = sorted(df['traj_id'].unique().tolist())
    all_ids = traj_ids  # Use all trajectories

    full_traj_dataset = CSVFullTrajectoryDataset(
        cfg["csv_path"], all_ids,
        cfg["sequence_length"], cfg["prediction_length"]
    )

    # Create data processor for trajectory training
    data_processor = CSVTrajectoryProcessor(cfg["csv_path"])

    # 创建轨迹级数据加载器
    traj_loader = DataLoader(
        full_traj_dataset,
        batch_size=cfg.get("traj_batch_size", 16),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True
    )
    
    # Create trajectory-level loss function
    traj_crit = TrajectoryJitterLoss(
        velocity_w=cfg.get("traj_velocity_w", 1.0),
        acceleration_w=cfg.get("traj_acceleration_w", 2.0),
        jerk_w=cfg.get("traj_jerk_w", 3.0),
        max_speed_mm_per_s=cfg.get("traj_max_speed_mm_s", 100.0),  # Reduced for 10fps data
        data_processor=data_processor
    )
    
    # Use smaller learning rate for trajectory-level training
    traj_lr = cfg.get("traj_learning_rate", cfg["learning_rate"] * 0.1)
    traj_opt = optim.Adam(model.parameters(), lr=traj_lr, weight_decay=1e-5)

    # Trajectory-level training epochs
    traj_epochs = cfg.get("traj_epochs", max(5, cfg["num_epochs"] // 10))
    
    traj_best = float("inf")
    traj_best_path = os.path.join(cfg["save_dir"], "best_traj_model.pth")
    
    print(f"Trajectory training: {traj_epochs} epochs, lr={traj_lr:.2e}")
    print(f"Traj loader: {len(full_traj_dataset)} segments, batch={cfg.get('traj_batch_size', 16)}")
    
    for epoch in range(1, traj_epochs + 1):
        # 轨迹级训练
        traj_loss = train_full_trajectory_epoch(
            model, traj_loader, traj_opt, traj_crit, device, 
            epoch, traj_epochs
        )
        
        # Use original validation set for validation
        val_loss = validate(model, val_loader, crit, device, epoch, traj_epochs)
        
        print(f"[Traj][Epoch {epoch:03d}] traj_loss={traj_loss:.6f}  val={val_loss:.6f}  lr={traj_opt.param_groups[0]['lr']:.2e}")
        
        # Save better trajectory-level model
        if val_loss < traj_best:
            traj_best = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, traj_best_path)
            print(f"[Traj] Saved best trajectory model to {traj_best_path} (val={traj_best:.6f})")
    
    print(f"Phase 2 (Trajectory-level) done. Best traj val={traj_best:.6f}. Total elapsed {time.time()-t0:.1f}s.")
    print("\nTraining completed! Models saved:")
    print(f"  - Frame-level best: {best_path}")
    print(f"  - Trajectory-level best: {traj_best_path}")
    print(f"\nTrajectory IDs used for training: {all_ids[:10]}..." + (f" (total: {len(all_ids)})" if len(all_ids) > 10 else ""))

if __name__ == "__main__":
    main()
