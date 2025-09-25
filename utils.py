# utils.py (CSV trajectory dataset training)
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---------- geometry ----------
def normalize_to_torso(joints: np.ndarray, arms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize joints and arms relative to torso position
    joints: [T, 9, 3] or [9, 3]
    arms: [T, 4, 3] or [4, 3]
    """
    if joints.ndim == 2:  # Single frame
        torso_pos = joints[8]  # Index 8 = Torso
        joints_norm = joints - torso_pos[None, :]
        arms_norm = arms - torso_pos[None, :]
    else:  # Multiple frames
        torso_pos = joints[:, 8]  # [T, 3]
        joints_norm = joints - torso_pos[:, None, :]
        arms_norm = arms - torso_pos[:, None, :]
    return joints_norm, arms_norm

def random_rotation_quat() -> np.ndarray:
    axis = np.random.normal(size=3).astype(np.float32)
    axis /= (np.linalg.norm(axis) + 1e-8)
    angle = np.random.uniform(-np.pi, np.pi)
    s = np.sin(angle / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2.0)], dtype=np.float32)

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = np.linalg.norm(q) + 1e-8
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float32)

def clip_arms_to_40cm(arms: np.ndarray) -> np.ndarray:
    """
    Clip arm lengths to 40cm (400mm)
    arms: [T, 4, 3] or [4, 3] - [L1, L2, R1, R2]
    """
    target_length = 400.0  # 40cm in mm

    if arms.ndim == 2:  # Single frame
        arms = arms[None, :]  # Add time dimension
        squeeze = True
    else:
        squeeze = False

    arms_clipped = arms.copy()

    for t in range(arms.shape[0]):
        # Left arm: L1 -> L2
        l1, l2 = arms[t, 0], arms[t, 1]
        left_vec = l1 - l2
        left_length = np.linalg.norm(left_vec)
        if left_length > target_length:
            left_dir = left_vec / left_length
            arms_clipped[t, 0] = l2 + left_dir * target_length

        # Right arm: R1 -> R2
        r1, r2 = arms[t, 2], arms[t, 3]
        right_vec = r1 - r2
        right_length = np.linalg.norm(right_vec)
        if right_length > target_length:
            right_dir = right_vec / right_length
            arms_clipped[t, 2] = r2 + right_dir * target_length

    if squeeze:
        arms_clipped = arms_clipped[0]

    return arms_clipped

def resample_time(arr: np.ndarray, new_len: int) -> np.ndarray:
    T, D = arr.shape
    if new_len == T:
        return arr.copy()
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, new_len)
    out = np.zeros((new_len, D), dtype=arr.dtype)
    for d in range(D):
        out[:, d] = np.interp(t_new, t_old, arr[:, d])
    return out

# ---------- CSV trajectory processor ----------
class CSVTrajectoryProcessor:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.traj_ids = sorted(self.df['traj_id'].unique().tolist())

        # Define joint and arm columns
        self.joint_names = ['Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow', 'R_Hand', 'L_Hand', 'Torso']
        self.joint_cols = []
        for joint in self.joint_names:
            self.joint_cols.extend([f"{joint}_pred_x", f"{joint}_pred_y", f"{joint}_pred_z"])

        self.arm_cols = [
            'Left_L1_x', 'Left_L1_y', 'Left_L1_z',
            'Left_L2_x', 'Left_L2_y', 'Left_L2_z',
            'Right_R1_x', 'Right_R1_y', 'Right_R1_z',
            'Right_R2_x', 'Right_R2_y', 'Right_R2_z'
        ]

        # Verify columns exist
        missing_joint_cols = [c for c in self.joint_cols if c not in self.df.columns]
        missing_arm_cols = [c for c in self.arm_cols if c not in self.df.columns]

        if missing_joint_cols:
            raise ValueError(f"Missing joint columns: {missing_joint_cols}")
        if missing_arm_cols:
            raise ValueError(f"Missing arm columns: {missing_arm_cols}")

        print(f"CSV Data: trajectories={len(self.traj_ids)}, frames={len(self.df)}, joints=27D, arms=12D")
    
    def get_trajectory_length(self, traj_id: int) -> int:
        """Get the number of frames in a trajectory"""
        traj_data = self.df[self.df['traj_id'] == traj_id]
        return len(traj_data)

    def get_trajectory_data(self, traj_id: int, normalize_to_torso_flag: bool = True, clip_arms_flag: bool = True):
        """
        Returns trajectory data for training
        Returns:
          joints [T,27]  : upper body joints (9 joints * 3 coords)
          arms [T,12]    : arm endpoints (4 endpoints * 3 coords)
        """
        # Get trajectory data sorted by frame_id
        traj_data = self.df[self.df['traj_id'] == traj_id].sort_values('frame_id')

        if len(traj_data) == 0:
            raise ValueError(f"No data found for trajectory {traj_id}")

        # Extract joint data [T, 27]
        joint_data = traj_data[self.joint_cols].values.astype(np.float32)  # [T, 27]
        joints = joint_data.reshape(-1, 9, 3)  # [T, 9, 3]

        # Extract arm data [T, 12]
        arm_data = traj_data[self.arm_cols].values.astype(np.float32)  # [T, 12]
        arms = arm_data.reshape(-1, 4, 3)  # [T, 4, 3]

        # Apply arm clipping
        if clip_arms_flag:
            arms = clip_arms_to_40cm(arms)

        # Apply torso normalization
        if normalize_to_torso_flag:
            joints, arms = normalize_to_torso(joints, arms)

        # Reshape back to 2D
        joints_2d = joints.reshape(-1, 27)  # [T, 27]
        arms_2d = arms.reshape(-1, 12)      # [T, 12]

        return joints_2d, arms_2d

# ---------- CSV trajectory dataset ----------
class CSVTrajectoryDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 traj_ids: List[int],
                 sequence_length: int = 30,
                 prediction_length: int = 10,
                 augment: bool = True,
                 rot_prob: float = 1.0,
                 scale_prob: float = 0.5,
                 scale_range: Tuple[float,float] = (0.9, 1.1),
                 time_scale_prob: float = 0.4,
                 time_scale_range: Tuple[float,float] = (0.8, 1.25),
                 noise_std_pos: float = 0.005):
        self.processor = CSVTrajectoryProcessor(csv_path)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.augment = augment
        self.rot_prob = rot_prob
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.time_scale_prob = time_scale_prob
        self.time_scale_range = time_scale_range
        self.noise_std_pos = noise_std_pos

        # Load trajectory data
        self.inputs, self.targets = [], []
        for traj_id in traj_ids:
            try:
                joints, arms = self.processor.get_trajectory_data(traj_id)
                self.inputs.append(joints)   # [T, 27]
                self.targets.append(arms)    # [T, 12]
            except ValueError as e:
                print(f"Warning: Could not load trajectory {traj_id}: {e}")
                continue

        # Create sequence windows
        self.sequences = []
        min_length = sequence_length + prediction_length
        for traj_idx, joints in enumerate(self.inputs):
            traj_length = joints.shape[0]
            if traj_length >= min_length:
                for start_idx in range(0, traj_length - min_length + 1):
                    self.sequences.append((traj_idx, start_idx))

        print(f"CSV Dataset: trajectories={len(traj_ids)}, loaded={len(self.inputs)}, sequences={len(self.sequences)}, "
              f"seq_len={sequence_length}, pred_len={prediction_length}, aug={'ON' if augment else 'OFF'}")

    def __len__(self):
        return len(self.sequences)

    # ---- augment ops ----
    def _apply_random_rotation(self, joints_seq, arms_seq):
        rq = random_rotation_quat()
        R = quat_to_rotmat(rq)

        joints_rot = joints_seq.copy()
        arms_rot = arms_seq.copy()

        # Rotate joints [T, 27] -> [T, 9, 3] -> rotate -> [T, 27]
        joints_3d = joints_seq.reshape(-1, 9, 3)
        joints_3d_rot = (joints_3d @ R.T)
        joints_rot = joints_3d_rot.reshape(-1, 27)

        # Rotate arms [T, 12] -> [T, 4, 3] -> rotate -> [T, 12]
        arms_3d = arms_seq.reshape(-1, 4, 3)
        arms_3d_rot = (arms_3d @ R.T)
        arms_rot = arms_3d_rot.reshape(-1, 12)

        return joints_rot, arms_rot

    def _apply_scale(self, joints_seq, arms_seq, scale_range):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        joints_scaled = joints_seq.copy() * scale
        arms_scaled = arms_seq.copy() * scale
        return joints_scaled, arms_scaled

    def _apply_time_scale(self, joints_full, arms_full, seq_len, pred_len, scale_range):
        total_len = seq_len + pred_len
        scale = np.random.uniform(scale_range[0], scale_range[1])
        new_len = max(2, int(round(total_len * scale)))

        joints_resampled = resample_time(joints_full, new_len)
        arms_resampled = resample_time(arms_full, new_len)

        start = np.random.randint(0, max(1, new_len - total_len + 1))
        joints_seq = joints_resampled[start:start+seq_len]
        arms_seq = arms_resampled[start+seq_len:start+total_len]

        # Ensure correct lengths
        if joints_seq.shape[0] < seq_len:
            pad = np.repeat(joints_seq[-1:, :], seq_len - joints_seq.shape[0], axis=0)
            joints_seq = np.concatenate([joints_seq, pad], axis=0)
        joints_seq = joints_seq[:seq_len]

        if arms_seq.shape[0] < pred_len:
            pad = np.repeat(arms_seq[-1:, :], pred_len - arms_seq.shape[0], axis=0)
            arms_seq = np.concatenate([arms_seq, pad], axis=0)
        arms_seq = arms_seq[:pred_len]

        return joints_seq, arms_seq

    def __getitem__(self, idx: int):
        traj_idx, start_idx = self.sequences[idx]
        joints_full = self.inputs[traj_idx]  # [T, 27]
        arms_full = self.targets[traj_idx]   # [T, 12]

        end_input = start_idx + self.sequence_length
        end_all = end_input + self.prediction_length

        joints_seq = joints_full[start_idx:end_input].copy()    # [seq_len, 27]
        arms_seq = arms_full[end_input:end_all].copy()          # [pred_len, 12]

        if self.augment:
            # Time scaling augmentation
            if np.random.rand() < self.time_scale_prob:
                window_joints = joints_full[start_idx:end_all].copy()
                window_arms = arms_full[start_idx:end_all].copy()
                joints_seq, arms_seq = self._apply_time_scale(
                    window_joints, window_arms,
                    self.sequence_length, self.prediction_length,
                    self.time_scale_range
                )

            # Rotation augmentation
            if np.random.rand() < self.rot_prob:
                joints_seq, arms_seq = self._apply_random_rotation(joints_seq, arms_seq)

            # Scale augmentation
            if np.random.rand() < self.scale_prob:
                joints_seq, arms_seq = self._apply_scale(joints_seq, arms_seq, self.scale_range)

            # Add noise
            if self.noise_std_pos > 0:
                noise = np.random.normal(0.0, self.noise_std_pos, joints_seq.shape).astype(np.float32)
                joints_seq += noise

        return torch.from_numpy(joints_seq).float(), torch.from_numpy(arms_seq).float()

# ---------- CSV trajectory loaders ----------
def create_csv_data_loaders(csv_path: str,
                            batch_size: int = 32,
                            sequence_length: int = 30,
                            prediction_length: int = 10,
                            val_ratio: float = 0.2,
                            num_workers: int = 4):
    """Create data loaders for CSV trajectory dataset - using same data for train and val"""
    df = pd.read_csv(csv_path)
    traj_ids = sorted(df['traj_id'].unique().tolist())

    # Use ALL trajectories for both training and validation
    all_ids = traj_ids

    train_ds = CSVTrajectoryDataset(csv_path, all_ids, sequence_length, prediction_length, augment=True)
    val_ds = CSVTrajectoryDataset(csv_path, all_ids, sequence_length, prediction_length, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"CSV Loaders: train={len(train_ds)}, val={len(val_ds)}, batch={batch_size}, seq_len={sequence_length}, pred_len={prediction_length}")
    print(f"Using same data for train and validation: {len(all_ids)} trajectories (traj_ids: {all_ids[:5]}...{all_ids[-3:]})")
    return train_loader, val_loader


# ---------- Trajectory-level training utils ----------
class TrajectoryTrainer:
    """Handles trajectory-level training with reduced teacher forcing"""
    def __init__(self, processor: CSVTrajectoryProcessor, traj_ids: List[int]):
        self.processor = processor
        self.traj_ids = traj_ids
        print(f"Trajectory Trainer: {len(traj_ids)} trajectories for trajectory-level training")

    def get_trajectory_batch(self, batch_traj_ids: List[int]):
        """Get full trajectories for trajectory-level training"""
        trajectories = []
        for traj_id in batch_traj_ids:
            try:
                joints, arms = self.processor.get_trajectory_data(traj_id)
                trajectories.append({
                    'traj_id': traj_id,
                    'joints': torch.from_numpy(joints).float(),  # [T, 27]
                    'arms': torch.from_numpy(arms).float()       # [T, 12]
                })
            except ValueError:
                continue
        return trajectories
