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


# ========== Augmented Data Processing ==========
import os
import glob
from collections import defaultdict
from typing import Dict

def clip_robot_arms_to_40cm(arms: np.ndarray) -> np.ndarray:
    """
    Clip robot arm lengths to 40cm (400mm)
    arms: [T, 4, 3] or [4, 3] - [lback, lfront, rback, rfront]
    Rule: Keep back position fixed, clip front position to 400mm away from back
    """
    target_length = 400.0  # 40cm in mm

    if arms.ndim == 2:  # Single frame
        arms = arms[None, :]
        squeeze = True
    else:
        squeeze = False

    arms_clipped = arms.copy()

    for t in range(arms.shape[0]):
        # Left arm: lback -> lfront (indices 0 -> 1)
        back, front = arms[t, 0], arms[t, 1]
        vec = front - back
        length = np.linalg.norm(vec)
        if length > target_length:
            direction = vec / length
            arms_clipped[t, 1] = back + direction * target_length

        # Right arm: rback -> rfront (indices 2 -> 3)
        back, front = arms[t, 2], arms[t, 3]
        vec = front - back
        length = np.linalg.norm(vec)
        if length > target_length:
            direction = vec / length
            arms_clipped[t, 3] = back + direction * target_length

    if squeeze:
        arms_clipped = arms_clipped[0]

    return arms_clipped


class AugmentedDataProcessor:
    """
    Process augmented mocap data from global_csv directory
    - Handles 120fps -> 30fps downsampling
    - Extracts 9 upper body joints + 4 robot endpoints
    - Normalizes to torso and clips arms to 40cm
    """
    def __init__(self, data_root: str):
        """
        Args:
            data_root: Path to global_csv directory containing csv_*_augmented_processed folders
        """
        self.data_root = data_root
        self.file_groups = self._scan_augmented_files()
        print(f"Augmented Data: {len(self.file_groups)} base recordings, "
              f"{sum(len(files) for files in self.file_groups.values())} total files")

    def _scan_augmented_files(self) -> Dict[str, List[str]]:
        """
        Scan all augmented_processed directories and group files by base name
        Returns:
            {base_name: [aug_00.csv, aug_01.csv, ..., aug_09.csv]}
        """
        file_groups = defaultdict(list)

        # Scan all augmented_processed directories
        pattern = os.path.join(self.data_root, "csv_*_augmented_processed")
        aug_dirs = glob.glob(pattern)

        for aug_dir in aug_dirs:
            person_name = os.path.basename(aug_dir).replace("_augmented_processed", "")
            csv_files = glob.glob(os.path.join(aug_dir, "*.csv"))

            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                # Extract base name (remove _aug_XX.csv suffix)
                if "_aug_" in filename:
                    base_name = filename.rsplit("_aug_", 1)[0]
                    full_key = f"{person_name}/{base_name}"
                    file_groups[full_key].append(csv_file)

        # Sort files within each group to ensure consistent ordering
        for key in file_groups:
            file_groups[key] = sorted(file_groups[key])

        return dict(file_groups)

    def get_all_file_paths(self) -> List[str]:
        """Get all augmented CSV file paths"""
        all_files = []
        for files in self.file_groups.values():
            all_files.extend(files)
        return sorted(all_files)

    def get_base_names(self) -> List[str]:
        """Get all unique base recording names"""
        return sorted(list(self.file_groups.keys()))

    def load_trajectory_file(self, file_path: str, fps_target: int = 30,
                             normalize_to_torso_flag: bool = True,
                             clip_arms_flag: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process a single trajectory CSV file

        Args:
            file_path: Path to the CSV file
            fps_target: Target frame rate (default 30fps, original is 120fps)
            normalize_to_torso_flag: Whether to normalize to torso position
            clip_arms_flag: Whether to clip arms to 40cm

        Returns:
            inputs: [T, 39] - 9 joints (27D) + 4 robot endpoints (12D)
            targets: [T, 12] - 4 robot endpoints (clipped)
        """
        df = pd.read_csv(file_path)

        # Downsample from 120fps to target fps
        fps_original = 120
        downsample_factor = fps_original // fps_target
        df = df[::downsample_factor].reset_index(drop=True)

        # Extract 9 upper body joints in the order matching original data
        # Original order: [Head, Neck, R_Shoulder, L_Shoulder, R_Elbow, L_Elbow, R_Hand, L_Hand, Torso]
        # New data order: [torso, neck, head, l_shoulder, l_elbow, l_hand, r_shoulder, r_elbow, r_hand]
        joint_names_new = ['head', 'neck', 'r_shoulder', 'l_shoulder',
                           'r_elbow', 'l_elbow', 'r_hand', 'l_hand', 'torso']

        joints_list = []
        for joint_name in joint_names_new:
            x = df[f'{joint_name}:X'].values
            y = df[f'{joint_name}:Y'].values
            z = df[f'{joint_name}:Z'].values
            joints_list.append(np.stack([x, y, z], axis=1))

        joints = np.stack(joints_list, axis=1).astype(np.float32)  # [T, 9, 3]

        # Extract 4 robot endpoints: [lback, lfront, rback, rfront]
        robot_names = ['lback', 'lfront', 'rback', 'rfront']
        robot_list = []
        for robot_name in robot_names:
            x = df[f'{robot_name}:X'].values
            y = df[f'{robot_name}:Y'].values
            z = df[f'{robot_name}:Z'].values
            robot_list.append(np.stack([x, y, z], axis=1))

        robot_endpoints = np.stack(robot_list, axis=1).astype(np.float32)  # [T, 4, 3]

        # Clip robot arms to 40cm
        if clip_arms_flag:
            robot_endpoints_clipped = clip_robot_arms_to_40cm(robot_endpoints)
        else:
            robot_endpoints_clipped = robot_endpoints.copy()

        # Normalize to torso if requested
        if normalize_to_torso_flag:
            joints_norm, robot_norm = normalize_to_torso(joints, robot_endpoints_clipped)
        else:
            joints_norm = joints
            robot_norm = robot_endpoints_clipped

        # Prepare inputs: concatenate joints and robot endpoints
        # inputs: [T, 39] = joints [T, 27] + robot [T, 12]
        joints_2d = joints_norm.reshape(-1, 27)  # [T, 27]
        robot_2d = robot_norm.reshape(-1, 12)    # [T, 12]
        inputs = np.concatenate([joints_2d, robot_2d], axis=1)  # [T, 39]

        # targets: [T, 12] = robot endpoints (clipped)
        targets = robot_norm.reshape(-1, 12)  # [T, 12]

        return inputs, targets


class AugmentedDataset(Dataset):
    """
    Dataset for augmented mocap data with 8:1:1 train/val/test split
    Supports data augmentation (rotation, scaling, time scaling, progressive noise)

    Progressive Noise Strategy:
    - Skeleton (9 joints, 27D): Stage 1 (0→10mm), Stage 2 (10→30mm), Stage 3 (30→50mm)
    - Robot (4 points, 12D): Stage 1 (0→10mm), Stage 2 (10→20mm), Stage 3 (20→30mm)
    - Stage 3: + random spike noise (~200mm on 1-3 frames)
    """
    def __init__(self,
                 file_paths: List[str],
                 sequence_length: int = 30,
                 prediction_length: int = 10,
                 augment: bool = True,
                 rot_prob: float = 1.0,
                 scale_prob: float = 0.5,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 time_scale_prob: float = 0.4,
                 time_scale_range: Tuple[float, float] = (0.8, 1.25),
                 noise_std_pos: float = 0.005,
                 noise_std_skeleton_mm: float = 0.0,
                 noise_std_robot_mm: float = 0.0,
                 noise_spike_prob: float = 0.0,
                 noise_spike_magnitude_mm: float = 200.0):
        """
        Args:
            file_paths: List of CSV file paths for this split
            sequence_length: Number of input frames
            prediction_length: Number of output frames
            augment: Whether to apply data augmentation
            noise_std_skeleton_mm: Std dev of skeleton noise (9 joints, first 27D)
            noise_std_robot_mm: Std dev of robot noise (4 points, last 12D)
        """
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.augment = augment
        self.rot_prob = rot_prob
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.time_scale_prob = time_scale_prob
        self.time_scale_range = time_scale_range
        self.noise_std_pos = noise_std_pos  # Legacy, kept for compatibility

        # Progressive noise parameters (separate for skeleton and robot)
        self.noise_std_skeleton_mm = noise_std_skeleton_mm
        self.noise_std_robot_mm = noise_std_robot_mm
        self.noise_spike_prob = noise_spike_prob
        self.noise_spike_magnitude_mm = noise_spike_magnitude_mm

        # Load all trajectories
        self.inputs_list = []   # List of [T, 39] arrays
        self.targets_list = []  # List of [T, 12] arrays

        processor = AugmentedDataProcessor(os.path.dirname(os.path.dirname(file_paths[0])))

        for file_path in file_paths:
            try:
                inputs, targets = processor.load_trajectory_file(file_path)
                self.inputs_list.append(inputs)
                self.targets_list.append(targets)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

        # Create sliding windows
        self.sequences = []
        min_length = sequence_length + prediction_length

        for traj_idx, inputs in enumerate(self.inputs_list):
            traj_length = inputs.shape[0]
            if traj_length >= min_length:
                for start_idx in range(0, traj_length - min_length + 1):
                    self.sequences.append((traj_idx, start_idx))

        print(f"Augmented Dataset: files={len(file_paths)}, loaded={len(self.inputs_list)}, "
              f"sequences={len(self.sequences)}, seq_len={sequence_length}, "
              f"pred_len={prediction_length}, aug={'ON' if augment else 'OFF'}")

    def __len__(self):
        return len(self.sequences)

    def _apply_random_rotation(self, inputs_seq, targets_seq):
        """Apply random 3D rotation to both inputs and targets"""
        rq = random_rotation_quat()
        R = quat_to_rotmat(rq)

        # Rotate inputs [T, 39] -> [T, 13, 3]
        inputs_3d = inputs_seq.reshape(-1, 13, 3)
        inputs_rot = (inputs_3d @ R.T).reshape(-1, 39)

        # Rotate targets [T, 12] -> [T, 4, 3]
        targets_3d = targets_seq.reshape(-1, 4, 3)
        targets_rot = (targets_3d @ R.T).reshape(-1, 12)

        return inputs_rot, targets_rot

    def _apply_scale(self, inputs_seq, targets_seq, scale_range):
        """Apply random uniform scaling"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return inputs_seq * scale, targets_seq * scale

    def _apply_time_scale(self, inputs_full, targets_full, seq_len, pred_len, scale_range):
        """Apply random time scaling (speed up/slow down)"""
        total_len = seq_len + pred_len
        scale = np.random.uniform(scale_range[0], scale_range[1])
        new_len = max(2, int(round(total_len * scale)))

        inputs_resampled = resample_time(inputs_full, new_len)
        targets_resampled = resample_time(targets_full, new_len)

        start = np.random.randint(0, max(1, new_len - total_len + 1))
        inputs_seq = inputs_resampled[start:start+seq_len]
        targets_seq = targets_resampled[start+seq_len:start+total_len]

        # Ensure correct lengths with padding if needed
        if inputs_seq.shape[0] < seq_len:
            pad = np.repeat(inputs_seq[-1:, :], seq_len - inputs_seq.shape[0], axis=0)
            inputs_seq = np.concatenate([inputs_seq, pad], axis=0)
        inputs_seq = inputs_seq[:seq_len]

        if targets_seq.shape[0] < pred_len:
            pad = np.repeat(targets_seq[-1:, :], pred_len - targets_seq.shape[0], axis=0)
            targets_seq = np.concatenate([targets_seq, pad], axis=0)
        targets_seq = targets_seq[:pred_len]

        return inputs_seq, targets_seq

    def _apply_progressive_noise(self, inputs_seq):
        """
        Apply progressive trajectory-level noise with separate parameters for skeleton and robot
        - Trajectory-level: same noise offset for entire sequence
        - Skeleton (first 27D): uses noise_std_skeleton_mm
        - Robot (last 12D): uses noise_std_robot_mm
        - Spike noise: random large noise on 1-3 frames in stage 3

        Args:
            inputs_seq: [T, 39] input sequence (27D skeleton + 12D robot)

        Returns:
            noisy_inputs: [T, 39] with noise applied
        """
        if self.noise_std_skeleton_mm == 0 and self.noise_std_robot_mm == 0:
            return inputs_seq

        T = inputs_seq.shape[0]
        noisy_inputs = inputs_seq.copy()

        # Split into skeleton (27D) and robot (12D)
        skeleton = noisy_inputs[:, :27]  # [T, 27]
        robot = noisy_inputs[:, 27:]     # [T, 12]

        # Apply trajectory-level noise to skeleton
        if self.noise_std_skeleton_mm > 0:
            # Trajectory-level offset (shared across all frames)
            traj_offset_skel = np.random.normal(0.0, self.noise_std_skeleton_mm, size=(1, 27)).astype(np.float32)
            # Per-frame independent noise
            per_frame_skel = np.random.normal(0.0, self.noise_std_skeleton_mm, size=(T, 27)).astype(np.float32)
            skeleton += traj_offset_skel + per_frame_skel

        # Apply trajectory-level noise to robot
        if self.noise_std_robot_mm > 0:
            # Trajectory-level offset (shared across all frames)
            traj_offset_robot = np.random.normal(0.0, self.noise_std_robot_mm, size=(1, 12)).astype(np.float32)
            # Per-frame independent noise
            per_frame_robot = np.random.normal(0.0, self.noise_std_robot_mm, size=(T, 12)).astype(np.float32)
            robot += traj_offset_robot + per_frame_robot

        # Recombine
        noisy_inputs = np.concatenate([skeleton, robot], axis=1)

        # Stage 3: Add spike noise to random frames (applies to all 39D)
        if self.noise_spike_prob > 0 and np.random.rand() < self.noise_spike_prob:
            # Randomly select 1-3 frames
            num_spike_frames = np.random.randint(1, 4)
            spike_frame_indices = np.random.choice(T, size=num_spike_frames, replace=False)

            # Add large spike noise to selected frames
            for frame_idx in spike_frame_indices:
                spike_noise = np.random.normal(0.0, self.noise_spike_magnitude_mm, size=(39,)).astype(np.float32)
                noisy_inputs[frame_idx] += spike_noise

        return noisy_inputs

    def __getitem__(self, idx: int):
        traj_idx, start_idx = self.sequences[idx]
        inputs_full = self.inputs_list[traj_idx]    # [T, 39]
        targets_full = self.targets_list[traj_idx]  # [T, 12]

        end_input = start_idx + self.sequence_length
        end_all = end_input + self.prediction_length

        inputs_seq = inputs_full[start_idx:end_input].copy()   # [seq_len, 39]
        targets_seq = targets_full[end_input:end_all].copy()   # [pred_len, 12]

        if self.augment:
            # Time scaling augmentation
            if np.random.rand() < self.time_scale_prob:
                window_inputs = inputs_full[start_idx:end_all].copy()
                window_targets = targets_full[start_idx:end_all].copy()
                inputs_seq, targets_seq = self._apply_time_scale(
                    window_inputs, window_targets,
                    self.sequence_length, self.prediction_length,
                    self.time_scale_range
                )

            # Rotation augmentation
            if np.random.rand() < self.rot_prob:
                inputs_seq, targets_seq = self._apply_random_rotation(inputs_seq, targets_seq)

            # Scale augmentation
            if np.random.rand() < self.scale_prob:
                inputs_seq, targets_seq = self._apply_scale(inputs_seq, targets_seq, self.scale_range)

            # Add progressive trajectory-level noise
            inputs_seq = self._apply_progressive_noise(inputs_seq)

        return torch.from_numpy(inputs_seq).float(), torch.from_numpy(targets_seq).float()


def create_augmented_data_loaders(data_root: str,
                                   batch_size: int = 32,
                                   sequence_length: int = 30,
                                   prediction_length: int = 10,
                                   train_ratio: float = 0.8,
                                   val_ratio: float = 0.1,
                                   test_ratio: float = 0.1,
                                   num_workers: int = 4,
                                   random_seed: int = 42,
                                   noise_std_skeleton_mm: float = 0.0,
                                   noise_std_robot_mm: float = 0.0,
                                   noise_spike_prob: float = 0.0,
                                   noise_spike_magnitude_mm: float = 200.0):
    """
    Create train/val/test data loaders for augmented data with 8:1:1 split
    Ensures no data leakage: all 10 augmented versions of the same recording
    are only in one split (train/val/test)

    Args:
        data_root: Path to global_csv directory
        batch_size: Batch size for training
        sequence_length: Number of input frames
        prediction_length: Number of prediction frames
        train_ratio: Ratio of base recordings for training (default 0.8)
        val_ratio: Ratio for validation (default 0.1)
        test_ratio: Ratio for testing (default 0.1)
        num_workers: Number of data loading workers
        random_seed: Random seed for reproducible splits
        noise_std_skeleton_mm: Std dev of skeleton noise in mm (default 0.0)
        noise_std_robot_mm: Std dev of robot noise in mm (default 0.0)
        noise_spike_prob: Probability of adding spike noise (default 0.0)
        noise_spike_magnitude_mm: Magnitude of spike noise in mm (default 200.0)

    Returns:
        train_loader, val_loader, test_loader
    """
    processor = AugmentedDataProcessor(data_root)
    base_names = processor.get_base_names()
    file_groups = processor.file_groups

    # Shuffle base names with fixed seed for reproducibility
    np.random.seed(random_seed)
    shuffled_bases = np.random.permutation(base_names).tolist()

    # Calculate split indices
    n_total = len(shuffled_bases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split base names
    train_bases = shuffled_bases[:n_train]
    val_bases = shuffled_bases[n_train:n_train+n_val]
    test_bases = shuffled_bases[n_train+n_val:]

    # Expand to file paths (each base has 10 augmented files)
    train_files = [f for base in train_bases for f in file_groups[base]]
    val_files = [f for base in val_bases for f in file_groups[base]]
    test_files = [f for base in test_bases for f in file_groups[base]]

    print(f"\nAugmented Data Split:")
    print(f"  Total base recordings: {n_total}")
    print(f"  Train: {len(train_bases)} bases ({len(train_files)} files)")
    print(f"  Val:   {len(val_bases)} bases ({len(val_files)} files)")
    print(f"  Test:  {len(test_bases)} bases ({len(test_files)} files)")

    # Create datasets (only apply progressive noise to training set)
    train_ds = AugmentedDataset(train_files, sequence_length, prediction_length, augment=True,
                                noise_std_skeleton_mm=noise_std_skeleton_mm,
                                noise_std_robot_mm=noise_std_robot_mm,
                                noise_spike_prob=noise_spike_prob,
                                noise_spike_magnitude_mm=noise_spike_magnitude_mm)
    val_ds = AugmentedDataset(val_files, sequence_length, prediction_length, augment=False,
                              noise_std_skeleton_mm=0.0, noise_std_robot_mm=0.0,
                              noise_spike_prob=0.0, noise_spike_magnitude_mm=0.0)
    test_ds = AugmentedDataset(test_files, sequence_length, prediction_length, augment=False,
                               noise_std_skeleton_mm=0.0, noise_std_robot_mm=0.0,
                               noise_spike_prob=0.0, noise_spike_magnitude_mm=0.0)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"\nData Loaders Created:")
    print(f"  Train: {len(train_ds)} sequences, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} sequences, {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds)} sequences, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
