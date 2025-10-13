# utils.py (CSV trajectory dataset training)
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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




# ========== Augmented Data Processing ==========
import os
import glob
from collections import defaultdict
from typing import Dict

def plane_normal(points):
    """Compute the normal vector of a plane from a set of points"""
    C = points.mean(axis=0)
    X = points - C
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1, :]
    n /= np.linalg.norm(n) + 1e-12
    return n


def rot_from_two_vectors(a, b):
    """Compute rotation matrix from vector a to vector b"""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180-degree rotation
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - axis.dot(a) * a
        axis /= np.linalg.norm(axis) + 1e-12
        # Rodrigues formula for 180-degree rotation
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))


def kabsch_rigid_transform(A, B):
    """Compute rigid body transform (rotation + translation) from A to B"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    assert A.shape == B.shape and A.shape[1] == 3

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def mat_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation"""
    tr = np.trace(R)
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    axis /= np.linalg.norm(axis) + 1e-12
    return axis, theta


def rodrigues(axis, angle):
    """Rodrigues formula: convert axis-angle to rotation matrix"""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def slerp_R(R, s):
    """Spherical linear interpolation for rotation matrix"""
    axis, ang = mat_to_axis_angle(R)
    return rodrigues(axis, s * ang)


def sample_arm_start_pose(original_arms, head_pos, original_distance, y_fixed=1000.0, half_angle_deg=15.0):
    """
    Sample initial arm pose (position + orientation)

    Args:
        original_arms: [4, 3] Original arm endpoints [L1, L2, R1, R2]
        head_pos: [3] Head position
        original_distance: float Original head-to-arm-midpoint distance
        y_fixed: float Fixed Y height
        half_angle_deg: float Cone half-angle (degrees)

    Returns:
        sampled_arms: [4, 3] Sampled arm endpoints
        sampled_mid: [3] Sampled midpoint position
        sampled_distance: float Sampled distance
    """
    # Original midpoint and direction
    mid_orig = original_arms.mean(axis=0)
    ref_dir = mid_orig - head_pos
    ref_norm = np.linalg.norm(ref_dir)
    if ref_norm < 1e-9:
        ref_dir = np.array([0, 0, 1.0])
        ref_norm = 1.0
    ref_dir_unit = ref_dir / ref_norm

    # Sample distance: original_distance ~ 2000mm
    sampled_distance = np.random.uniform(original_distance, 2000.0)

    # Sample direction within cone (spherical coordinates)
    half_angle_rad = np.deg2rad(half_angle_deg)
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    u = np.random.uniform(np.cos(half_angle_rad), 1.0)
    phi = np.arccos(u)

    # Construct local coordinate system
    z = ref_dir_unit
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)

    # Sample direction
    dir_vec = (np.sin(phi) * np.cos(theta)) * x + \
              (np.sin(phi) * np.sin(theta)) * y + \
              (np.cos(phi)) * z

    # Sample new midpoint position
    mid_new = head_pos + sampled_distance * dir_vec
    mid_new[1] = y_fixed  # Fix Y height

    # Compute original plane normal
    n_orig = plane_normal(original_arms)

    # Target normal: parallel to XZ plane, choose -Y direction (camera below)
    e_y = np.array([0.0, 1.0, 0.0])
    R_align = rot_from_two_vectors(n_orig, -e_y)

    # Apply rotation and translation to new midpoint
    C_orig = original_arms.mean(axis=0)
    sampled_arms = (R_align @ (original_arms - C_orig).T).T + mid_new

    return sampled_arms, mid_new, sampled_distance


def interpolate_robot_rigid_trajectory(start_arms, target_arms, max_speed=10.0):
    """
    Interpolate robot trajectory using rigid-body transform

    Args:
        start_arms: [4, 3] Start arm pose [L1, L2, R1, R2]
        target_arms: [4, 3] Target arm pose
        max_speed: float Max speed mm/frame

    Returns:
        interp_arms: [n_frames, 12] Interpolated arm trajectory
        n_frames: int Number of frames
    """
    # Compute rigid transform: start -> target
    R, t = kabsch_rigid_transform(start_arms, target_arms)

    # Compute max displacement (to determine frame count)
    disp = np.linalg.norm(target_arms - start_arms, axis=1)
    max_disp = float(np.max(disp)) if disp.size else 0.0

    # Compute number of frames
    n_frames = max(1, int(max_disp / max_speed) + 1)

    # Rigid interpolation
    interp_arms = []
    for i in range(n_frames):
        s = (i + 1) / n_frames
        Ri = slerp_R(R, s)
        ti = s * t
        Pi = (Ri @ start_arms.T).T + ti  # [4, 3]
        interp_arms.append(Pi.reshape(-1))  # [12]

    return np.array(interp_arms), n_frames


def generate_flipped_human_trajectory(joints, phases, n_frames, noise_std=2.0):
    """
    Generate corresponding human trajectory (reverse first 2/3 of approaching + noise)

    Args:
        joints: [T, 27] Original joint data
        phases: [T] Phase labels
        n_frames: int Number of frames to generate
        noise_std: float Noise std (mm)

    Returns:
        human_traj: [n_frames, 27] Human trajectory sequence
    """
    # Extract approaching phase
    approach_mask = (phases == 0)
    approach_joints = joints[approach_mask]

    if len(approach_joints) == 0:
        # If no approaching phase, use first 30 frames
        approach_joints = joints[:30]

    # Take first 2/3
    n_use = max(1, int(len(approach_joints) * 2/3))
    selected_joints = approach_joints[:n_use]

    # Time reverse
    flipped_joints = selected_joints[::-1]

    # Resample to n_frames
    if len(flipped_joints) >= n_frames:
        human_traj = flipped_joints[:n_frames]
    else:
        # Linear interpolation
        t_old = np.linspace(0, 1, len(flipped_joints))
        t_new = np.linspace(0, 1, n_frames)
        human_traj = np.zeros((n_frames, 27))
        for d in range(27):
            human_traj[:, d] = np.interp(t_new, t_old, flipped_joints[:, d])

    # Add noise
    noise = np.random.normal(0, noise_std, human_traj.shape)
    human_traj = human_traj + noise

    return human_traj


def augment_initial_position(joints, arms, phases, head_pos, y_fixed=1000.0, half_angle_deg=15.0, max_speed=10.0, noise_std=2.0):
    """
    Full initial-position data augmentation (verified version)

    Args:
        joints: [T, 27] Human joint data
        arms: [T, 12] Arm data
        phases: [T] Phase labels
        head_pos: [3] Head position
        y_fixed: float Fixed Y height
        half_angle_deg: float Cone half-angle
        max_speed: float Max speed mm/frame
        noise_std: float Human noise std (mm)

    Returns:
        aug_joints: [T_aug, 27] Augmented human joints
        aug_arms: [T_aug, 12] Augmented arms
        aug_phases: [T_aug] Augmented phase labels
        n_aug_frames: int Number of augmented frames
    """
    # 1. Extract original arm pose
    original_arms_3d = arms[0].reshape(4, 3)  # [L1, L2, R1, R2]
    original_mid = original_arms_3d.mean(axis=0)
    original_distance = np.linalg.norm(original_mid - head_pos)

    # 2. Sample new start pose
    start_arms, sampled_mid, sampled_distance = sample_arm_start_pose(
        original_arms_3d, head_pos, original_distance,
        y_fixed=y_fixed, half_angle_deg=half_angle_deg
    )

    # 3. Interpolate trajectory using rigid transform
    interp_arms, n_frames = interpolate_robot_rigid_trajectory(
        start_arms, original_arms_3d, max_speed=max_speed
    )

    # 4. Generate human trajectory
    human_traj = generate_flipped_human_trajectory(joints, phases, n_frames, noise_std=noise_std)

    # 5. Generate phase labels (all approaching)
    phase_prefix = np.zeros(n_frames, dtype=int)

    # 6. Concatenate
    aug_joints = np.vstack([human_traj, joints])
    aug_arms = np.vstack([interp_arms, arms])
    aug_phases = np.concatenate([phase_prefix, phases])

    return aug_joints, aug_arms, aug_phases, n_frames


def rotate_y_axis(joints, arms, angle=None):
    """
    Y-axis rotation augmentation

    Args:
        joints: [T, 27] or [T, 9, 3]
        arms: [T, 12] or [T, 4, 3]
        angle: float (radians), None means random

    Returns:
        joints_rot, arms_rot
    """
    if angle is None:
        angle = np.random.uniform(0, 2*np.pi)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, 0, sin_a],
                  [0, 1, 0],
                  [-sin_a, 0, cos_a]], dtype=np.float32)

    # Reshape if needed
    joints_3d = joints.reshape(-1, 9, 3) if joints.shape[-1] != 3 else joints
    arms_3d = arms.reshape(-1, 4, 3) if arms.shape[-1] != 3 else arms

    # Rotate
    joints_rot = (R @ joints_3d.reshape(-1, 3).T).T.reshape(joints_3d.shape)
    arms_rot = (R @ arms_3d.reshape(-1, 3).T).T.reshape(arms_3d.shape)

    # Reshape back
    if joints.shape[-1] != 3:
        joints_rot = joints_rot.reshape(-1, 27)
    if arms.shape[-1] != 3:
        arms_rot = arms_rot.reshape(-1, 12)

    return joints_rot, arms_rot


def time_scale_augmentation(joints, arms, phases, scale_range=(0.9, 1.1)):
    """
    Time-scale data augmentation

    Args:
        joints: [T, 27]
        arms: [T, 12]
        phases: [T]
        scale_range: (min_scale, max_scale)

    Returns:
        joints_scaled, arms_scaled, phases_scaled
    """
    T = len(joints)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    T_new = int(T * scale)
    T_new = max(10, T_new)  # At least 10 frames

    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, T_new)

    # Interpolate joints
    joints_scaled = np.zeros((T_new, 27))
    for d in range(27):
        joints_scaled[:, d] = np.interp(t_new, t_old, joints[:, d])

    # Interpolate arms
    arms_scaled = np.zeros((T_new, 12))
    for d in range(12):
        arms_scaled[:, d] = np.interp(t_new, t_old, arms[:, d])

    # Interpolate phases (nearest neighbor)
    phase_indices = np.interp(t_new, t_old, np.arange(T)).astype(int)
    phase_indices = np.clip(phase_indices, 0, T-1)
    phases_scaled = phases[phase_indices]

    return joints_scaled, arms_scaled, phases_scaled


# ==================== CSV Trajectory Dataset with Phase ====================

class CSVTrajectoryDataset(Dataset):
    """
    Dataset for CSV trajectory files with phase labels
    Supports: GT/Pred data, all augmentations, phase labels
    """
    def __init__(self, csv_files, sequence_length=30, prediction_length=10,
                 use_gt=True, augmentation_config=None):
        """
        Args:
            csv_files: List of CSV file paths
            sequence_length: Input sequence length
            prediction_length: Prediction length
            use_gt: True=use GT poses, False=use Pred poses
            augmentation_config: Dict with keys:
                - rotate_prob: float
                - time_scale_prob: float
                - initial_pos_prob: float
                - noise_std: float
        """
        self.csv_files = csv_files
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.use_gt = use_gt
        self.aug_cfg = augmentation_config or {}

        # Joint names (9 joints used for network input)
        self.joint_names = ['Head', 'Neck', 'R_Shoulder', 'L_Shoulder',
                           'R_Elbow', 'L_Elbow', 'R_Hand', 'L_Hand', 'Torso']

        # Load all trajectories
        self.trajectories = []
        self._load_all_trajectories()

    def set_augmentation_config(self, augmentation_config):
        """Update augmentation config without reloading data"""
        self.aug_cfg = augmentation_config or {}

    def _load_all_trajectories(self):
        """Load all CSV files and create overlapping trajectory segments"""
        suffix = '_gt' if self.use_gt else '_pred'
        min_length = self.sequence_length + self.prediction_length
        stride = 1  # Use every frame as starting point for maximum data utilization

        for csv_file in self.csv_files:
            try:
                df = pd.read_csv(csv_file)
                traj_ids = df['traj_id'].unique()

                for traj_id in traj_ids:
                    traj_df = df[df['traj_id'] == traj_id].sort_values('frame_id').reset_index(drop=True)

                    # Extract joints [T, 27]
                    joint_cols = []
                    for jname in self.joint_names:
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

                    # Extract head position for augmentation
                    head_pos = np.array([
                        traj_df[f'Head{suffix}_x'].values[0],
                        traj_df[f'Head{suffix}_y'].values[0],
                        traj_df[f'Head{suffix}_z'].values[0]
                    ], dtype=np.float32)

                    # Create overlapping segments with stride
                    T = len(joints)
                    if T < min_length:
                        # If trajectory too short, pad and create one segment
                        pad_len = min_length - T
                        joints_padded = np.vstack([joints, np.tile(joints[-1:], (pad_len, 1))])
                        arms_padded = np.vstack([arms, np.tile(arms[-1:], (pad_len, 1))])
                        phases_padded = np.concatenate([phases, np.tile(phases[-1:], pad_len)])

                        self.trajectories.append({
                            'joints': joints_padded,
                            'arms': arms_padded,
                            'phases': phases_padded,
                            'head_pos': head_pos,
                            'traj_id': traj_id,
                            'file': csv_file,
                            'start_idx': 0
                        })
                    else:
                        # Create multiple overlapping segments
                        for start_idx in range(0, T - min_length + 1, stride):
                            self.trajectories.append({
                                'joints': joints,
                                'arms': arms,
                                'phases': phases,
                                'head_pos': head_pos,
                                'traj_id': traj_id,
                                'file': csv_file,
                                'start_idx': start_idx
                            })

            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                continue

        print(f"Loaded {len(self.trajectories)} segments from {len(self.csv_files)} files")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        joints = traj['joints'].copy()  # [T, 27]
        arms = traj['arms'].copy()      # [T, 12]
        phases = traj['phases'].copy()  # [T]
        head_pos = traj['head_pos'].copy()
        start_idx = traj['start_idx']  # Pre-determined start index

        # Apply augmentations
        length_changed = False

        if self.aug_cfg.get('rotate_prob', 0) > 0 and np.random.rand() < self.aug_cfg['rotate_prob']:
            joints, arms = rotate_y_axis(joints, arms)

        if self.aug_cfg.get('time_scale_prob', 0) > 0 and np.random.rand() < self.aug_cfg['time_scale_prob']:
            joints, arms, phases = time_scale_augmentation(joints, arms, phases)
            length_changed = True

        if self.aug_cfg.get('initial_pos_prob', 0) > 0 and np.random.rand() < self.aug_cfg['initial_pos_prob']:
            joints, arms, phases, _ = augment_initial_position(joints, arms, phases, head_pos)
            length_changed = True

        # Add noise
        if self.aug_cfg.get('noise_std', 0) > 0:
            noise = np.random.normal(0, self.aug_cfg['noise_std'], joints.shape).astype(np.float32)
            joints = joints + noise

        # Extract segment (40 consecutive frames total)
        min_length = self.sequence_length + self.prediction_length

        # If augmentation changed length, randomly select new start_idx
        if length_changed:
            T = len(joints)
            if T < min_length:
                # Pad if too short
                pad_len = min_length - T
                joints = np.vstack([joints, np.tile(joints[-1:], (pad_len, 1))])
                arms = np.vstack([arms, np.tile(arms[-1:], (pad_len, 1))])
                phases = np.concatenate([phases, np.tile(phases[-1:], pad_len)])
                start_idx = 0
            else:
                # Randomly select start index
                max_start = T - min_length
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        # Extract input and target segments
        input_joints = joints[start_idx:start_idx + self.sequence_length]  # [30, 27]
        input_arms = arms[start_idx:start_idx + self.sequence_length]      # [30, 12]
        target_arms = arms[start_idx + self.sequence_length:start_idx + min_length]  # [10, 12]
        target_phases = phases[start_idx + self.sequence_length:start_idx + min_length]  # [10]

        # Reshape for normalization
        input_joints_3d = input_joints.reshape(self.sequence_length, 9, 3)  # [30, 9, 3]
        input_arms_3d = input_arms.reshape(self.sequence_length, 4, 3)      # [30, 4, 3]
        target_arms_3d = target_arms.reshape(self.prediction_length, 4, 3)  # [10, 4, 3]

        # Normalize to torso coordinate system (like in robot_planner)
        normalized_input_joints, normalized_input_arms = normalize_to_torso(input_joints_3d, input_arms_3d)

        # For target arms, use the last frame's torso position from input window
        torso_positions = input_joints_3d[:, 8, :]  # [30, 3]
        last_torso = torso_positions[-1:, :]  # [1, 3] - last frame's torso
        # Normalize target arms relative to last input torso
        normalized_target_arms = target_arms_3d - last_torso[None, :, :]  # [10, 4, 3]

        # Flatten back
        normalized_input_joints_flat = normalized_input_joints.reshape(self.sequence_length, -1)  # [30, 27]
        normalized_input_arms_flat = normalized_input_arms.reshape(self.sequence_length, -1)      # [30, 12]
        normalized_target_arms_flat = normalized_target_arms.reshape(self.prediction_length, -1)  # [10, 12]

        # Concatenate input: [30, 27+12=39]
        input_seq = np.concatenate([normalized_input_joints_flat, normalized_input_arms_flat], axis=1)

        return (torch.from_numpy(input_seq).float(),
                torch.from_numpy(normalized_target_arms_flat).float(),
                torch.from_numpy(target_phases).long())


# ==================== Loss Functions ====================

class PhaseCoherenceLoss(nn.Module):
    """Phase coherence loss: penalize illegal phase transitions"""
    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight
        self.illegal_transitions = {(0, 2), (1, 0), (2, 0), (2, 1)}

    def forward(self, phase_logits):
        B, T, C = phase_logits.shape
        if T < 2:
            return torch.tensor(0.0, device=phase_logits.device)

        phase_pred = torch.argmax(phase_logits, dim=-1)
        loss = 0.0
        count = 0

        for t in range(T - 1):
            phase_t = phase_pred[:, t]
            phase_t1 = phase_pred[:, t+1]

            for b in range(B):
                transition = (int(phase_t[b]), int(phase_t1[b]))
                if transition in self.illegal_transitions:
                    illegal_phase = phase_t1[b]
                    loss += phase_logits[b, t+1, illegal_phase] ** 2
                    count += 1

        return self.weight * loss / count if count > 0 else torch.tensor(0.0, device=phase_logits.device)


class SpeedPenaltyLoss(nn.Module):
    """Speed penalty loss: penalize speed exceeding max_speed_mm_per_frame"""
    def __init__(self, max_speed_mm_per_frame, weight=5.0):
        super().__init__()
        self.max_speed = max_speed_mm_per_frame
        self.weight = weight

    def forward(self, pred_arms):
        B, T, D = pred_arms.shape
        if T < 2:
            return torch.tensor(0.0, device=pred_arms.device)

        velocity = pred_arms[:, 1:] - pred_arms[:, :-1]
        velocity_points = velocity.view(B, T-1, 4, 3)
        speeds = torch.norm(velocity_points, dim=-1)
        penalty = torch.relu(speeds - self.max_speed)

        return self.weight * penalty.pow(2).mean()


class CombinedLoss(nn.Module):
    """Combined loss for CSV trajectory training with phase prediction"""
    def __init__(self, robot_weight=1.0, phase_weight=1.0, coherence_weight=10.0,
                 speed_weight=5.0, max_speed_mm_per_frame=10.0, smooth_vel_weight=0.05, smooth_acc_weight=0.01):
        super().__init__()
        self.robot_weight = robot_weight
        self.phase_weight = phase_weight
        self.coherence_weight = coherence_weight
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.phase_coherence_loss_fn = PhaseCoherenceLoss(weight=1.0)  # Base weight=1.0, will multiply by coherence_weight
        self.speed_loss = SpeedPenaltyLoss(max_speed_mm_per_frame=max_speed_mm_per_frame, weight=speed_weight)
        self.smooth_vel_w = smooth_vel_weight
        self.smooth_acc_w = smooth_acc_weight

    def set_phase_weights(self, phase_weight, coherence_weight):
        """Dynamically update phase and coherence weights"""
        self.phase_weight = phase_weight
        self.coherence_weight = coherence_weight

    def forward(self, pred_robot, pred_phase, gt_robot, gt_phase):
        B, T, _ = pred_robot.shape
        device = pred_robot.device

        robot_loss = self.mse(pred_robot, gt_robot)
        pred_phase_flat = pred_phase.reshape(-1, 3)
        gt_phase_flat = gt_phase.reshape(-1)
        phase_loss = self.ce(pred_phase_flat, gt_phase_flat)
        coherence_loss_raw = self.phase_coherence_loss_fn(pred_phase)
        speed_loss = self.speed_loss(pred_robot)
        vel_loss = ((pred_robot[:, 1:] - pred_robot[:, :-1])**2).mean() if T > 1 else torch.tensor(0.0, device=device)
        acc_loss = ((pred_robot[:, 2:] - 2*pred_robot[:, 1:-1] + pred_robot[:, :-2])**2).mean() if T > 2 else torch.tensor(0.0, device=device)

        # Apply dynamic weights to phase and coherence losses
        total_loss = (self.robot_weight * robot_loss +
                     self.phase_weight * phase_loss +
                     self.coherence_weight * coherence_loss_raw +
                     speed_loss +
                     self.smooth_vel_w * vel_loss +
                     self.smooth_acc_w * acc_loss)

        loss_dict = {
            'robot': robot_loss.item(),
            'phase': phase_loss.item(),
            'coherence': coherence_loss_raw.item() if isinstance(coherence_loss_raw, torch.Tensor) else coherence_loss_raw,
            'speed': speed_loss.item() if isinstance(speed_loss, torch.Tensor) else speed_loss,
            'smooth_vel': vel_loss.item() if isinstance(vel_loss, torch.Tensor) else vel_loss,
            'smooth_acc': acc_loss.item() if isinstance(acc_loss, torch.Tensor) else acc_loss,
            'total': total_loss.item()
        }

        return total_loss, loss_dict
