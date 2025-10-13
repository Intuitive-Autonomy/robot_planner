#!/usr/bin/env python3
"""
Add phase labels to inference trajectory CSV files based on robot-to-human distance
Phases: approaching (0), assisting (1), leaving (2)
"""

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm


def segment_trajectory(distances, window_size=5):
    """
    将轨迹切分为三个阶段：approaching, assisting, leaving
    - approaching: 机器人靠近（距离减小）
    - assisting: 机器人协助（距离保持低值）
    - leaving: 机器人离开（距离增加）

    Returns phase indices for each frame
    """
    # 平滑距离曲线
    smoothed = np.convolve(distances, np.ones(window_size)/window_size, mode='valid')

    # 计算距离变化率
    velocity = np.diff(smoothed)

    # 找到最小距离点
    min_idx = np.argmin(smoothed)

    # 在最小距离点之前找到approaching结束点（速度从负转正或接近0）
    approach_end = min_idx
    for i in range(min_idx, 0, -1):
        if i < len(velocity) and velocity[i] < -10:  # 仍在接近
            approach_end = i + 1
            break

    # 在最小距离点之后找到lifting结束点（速度从接近0转正）
    lifting_end = min_idx
    for i in range(min_idx, len(velocity)):
        if velocity[i] > 10:  # 开始远离
            lifting_end = i + 1
            break

    # 补偿平滑和diff造成的索引偏移
    offset = window_size // 2
    approach_end = min(approach_end + offset, len(distances) - 1)
    lifting_end = min(lifting_end + offset, len(distances) - 1)

    return {
        'approaching': (0, approach_end),
        'assisting': (approach_end, lifting_end),
        'leaving': (lifting_end, len(distances) - 1)
    }


def calculate_robot_to_human_distance(df, row_idx, use_gt=True):
    """计算机器人到人的距离（XZ平面投影）"""
    suffix = 'gt' if use_gt else 'pred'

    # 机器人位置：L1和R1的中点
    left_l1_x = df.loc[row_idx, 'Left_L1_x']
    left_l1_z = df.loc[row_idx, 'Left_L1_z']
    right_r1_x = df.loc[row_idx, 'Right_R1_x']
    right_r1_z = df.loc[row_idx, 'Right_R1_z']
    robot_x = (left_l1_x + right_r1_x) / 2
    robot_z = (left_l1_z + right_r1_z) / 2

    # 人的位置：Torso在XZ平面投影
    human_x = df.loc[row_idx, f'Torso_{suffix}_x']
    human_z = df.loc[row_idx, f'Torso_{suffix}_z']

    # 计算XZ平面距离
    distance = np.sqrt((robot_x - human_x)**2 + (robot_z - human_z)**2)

    return distance


def add_phase_labels_to_csv(csv_file, use_gt=True):
    """
    为CSV文件添加阶段标签列

    Args:
        csv_file: CSV文件路径
        use_gt: True使用GT数据计算距离，False使用预测数据
    """
    print(f"\nProcessing: {csv_file}")

    # 读取CSV
    df = pd.read_csv(csv_file)

    # 检查是否已有phase列
    phase_col = 'phase_gt' if use_gt else 'phase_pred'
    if phase_col in df.columns:
        print(f"  Warning: {phase_col} column already exists, will overwrite")
        df = df.drop(columns=[phase_col])

    # 获取所有unique trajectory IDs
    traj_ids = df['traj_id'].unique()
    print(f"  Found {len(traj_ids)} trajectories")

    # 为每条轨迹计算阶段标签
    phase_labels = []

    for traj_id in tqdm(traj_ids, desc="  Processing trajectories"):
        # 筛选当前轨迹
        traj_mask = df['traj_id'] == traj_id
        traj_df = df[traj_mask].reset_index(drop=True)

        # 计算距离
        distances = []
        for i in range(len(traj_df)):
            dist = calculate_robot_to_human_distance(traj_df, i, use_gt=use_gt)
            distances.append(dist)

        # 划分阶段
        phases = segment_trajectory(distances)

        # 生成标签：0=approaching, 1=assisting, 2=leaving
        traj_labels = np.zeros(len(traj_df), dtype=int)
        traj_labels[phases['approaching'][0]:phases['approaching'][1]+1] = 0
        traj_labels[phases['assisting'][0]:phases['assisting'][1]+1] = 1
        traj_labels[phases['leaving'][0]:phases['leaving'][1]+1] = 2

        phase_labels.extend(traj_labels.tolist())

    # 添加phase列
    df[phase_col] = phase_labels

    # 保存回原文件
    df.to_csv(csv_file, index=False)
    print(f"  Saved with {phase_col} column")
    print(f"  Phase distribution: 0={sum(df[phase_col]==0)}, 1={sum(df[phase_col]==1)}, 2={sum(df[phase_col]==2)}")


def main():
    # 查找所有inference_trajectory文件
    csv_files = glob.glob('inference_trajectory*.csv')

    if not csv_files:
        print("No inference_trajectory*.csv files found in current directory")
        return

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")

    # 为每个文件添加阶段标签
    for csv_file in csv_files:
        # 判断是GT文件还是预测文件
        is_gt = 'gt' in csv_file.lower()

        if is_gt:
            # GT文件：使用GT数据计算
            add_phase_labels_to_csv(csv_file, use_gt=True)
        else:
            # 预测文件：使用预测数据计算
            add_phase_labels_to_csv(csv_file, use_gt=False)

    print("\nAll files processed!")
    print("\nPhase labels:")
    print("  0 = Approaching (robot moving closer)")
    print("  1 = Assisting (robot at close distance)")
    print("  2 = Leaving (robot moving away)")


if __name__ == "__main__":
    main()
