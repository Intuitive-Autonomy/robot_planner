#!/usr/bin/env python3
"""
测试初始位置数据增强
展示增强后的人体和机械臂轨迹
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def calculate_distance_xz(pos1, pos2):
    """计算XZ平面距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2)


def plane_normal(points):
    """计算点集的平面法向量"""
    C = points.mean(axis=0)
    X = points - C
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1, :]
    n /= np.linalg.norm(n) + 1e-12
    return n


def rot_from_two_vectors(a, b):
    """计算从向量a旋转到向量b的旋转矩阵"""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180度旋转
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - axis.dot(a) * a
        axis /= np.linalg.norm(axis) + 1e-12
        # Rodrigues公式：180度旋转
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))


def sample_arm_start_pose(original_arms, head_pos, original_distance, y_fixed=1000.0, half_angle_deg=15.0):
    """
    采样机械臂起始姿态（位置+朝向）

    Args:
        original_arms: [4, 3] 原始机械臂端点 [L1, L2, R1, R2]
        head_pos: [3] 头部位置
        original_distance: float 原始head到臂中点距离
        y_fixed: float 固定的Y高度
        half_angle_deg: float 锥形半角（度）

    Returns:
        sampled_arms: [4, 3] 采样后的机械臂端点
        sampled_mid: [3] 采样的中点位置
        sampled_distance: float 采样距离
    """
    # 原始中点和方向
    mid_orig = original_arms.mean(axis=0)
    ref_dir = mid_orig - head_pos
    ref_norm = np.linalg.norm(ref_dir)
    if ref_norm < 1e-9:
        ref_dir = np.array([0, 0, 1.0])
        ref_norm = 1.0
    ref_dir_unit = ref_dir / ref_norm

    # 采样距离：原距离 ~ 2000mm
    sampled_distance = np.random.uniform(original_distance, 2000.0)

    # 在锥形内采样方向（球坐标）
    half_angle_rad = np.deg2rad(half_angle_deg)
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    u = np.random.uniform(np.cos(half_angle_rad), 1.0)
    phi = np.arccos(u)

    # 构造局部坐标系
    z = ref_dir_unit
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)

    # 采样方向
    dir_vec = (np.sin(phi) * np.cos(theta)) * x + \
              (np.sin(phi) * np.sin(theta)) * y + \
              (np.cos(phi)) * z

    # 采样中点位置
    mid_new = head_pos + sampled_distance * dir_vec
    mid_new[1] = y_fixed  # 固定Y高度

    # 计算原始平面法向量
    n_orig = plane_normal(original_arms)

    # 目标法向量：平行于XZ平面，选择-Y方向（相机在下方）
    e_y = np.array([0.0, 1.0, 0.0])
    R_align = rot_from_two_vectors(n_orig, -e_y)

    # 应用旋转并平移到新中点
    C_orig = original_arms.mean(axis=0)
    sampled_arms = (R_align @ (original_arms - C_orig).T).T + mid_new

    return sampled_arms, mid_new, sampled_distance


def kabsch_rigid_transform(A, B):
    """计算从A到B的刚体变换（旋转+平移）"""
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
    """旋转矩阵转轴角"""
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
    """Rodrigues公式：轴角到旋转矩阵"""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def slerp_R(R, s):
    """球面线性插值旋转矩阵"""
    axis, ang = mat_to_axis_angle(R)
    return rodrigues(axis, s * ang)


def interpolate_robot_trajectory(start_arms, target_arms, max_speed=10.0):
    """
    使用刚体变换插值机器人轨迹

    Args:
        start_arms: [4, 3] 起始机械臂姿态 [L1, L2, R1, R2]
        target_arms: [4, 3] 目标机械臂姿态
        max_speed: float 最大速度 mm/frame

    Returns:
        interp_arms: [n_frames, 12] 插值后的机械臂轨迹
        n_frames: int 帧数
    """
    # 计算刚体变换：start -> target
    R, t = kabsch_rigid_transform(start_arms, target_arms)

    # 计算最大位移（用于确定帧数）
    disp = np.linalg.norm(target_arms - start_arms, axis=1)
    max_disp = float(np.max(disp)) if disp.size else 0.0

    # 计算帧数
    n_frames = max(1, int(max_disp / max_speed) + 1)

    # 刚体插值
    interp_arms = []
    for i in range(n_frames):
        s = (i + 1) / n_frames
        Ri = slerp_R(R, s)
        ti = s * t
        Pi = (Ri @ start_arms.T).T + ti  # [4, 3]
        interp_arms.append(Pi.reshape(-1))  # [12]

    return np.array(interp_arms), n_frames


def generate_human_trajectory(joints, phases, n_frames):
    """
    生成对应的人体姿态序列（翻转approaching前2/3 + 噪声）

    Args:
        joints: [T, 27] 原始关节数据
        phases: [T] 阶段标签
        n_frames: int 需要生成的帧数

    Returns:
        human_traj: [n_frames, 27] 人体姿态序列
    """
    # 提取approaching阶段
    approach_mask = (phases == 0)
    approach_joints = joints[approach_mask]

    if len(approach_joints) == 0:
        print("Warning: No approaching phase found, using first frames")
        approach_joints = joints[:30]

    # 取前2/3部分
    n_use = max(1, int(len(approach_joints) * 2/3))
    selected_joints = approach_joints[:n_use]

    # 时间翻转
    flipped_joints = selected_joints[::-1]

    # 重采样到n_frames
    if len(flipped_joints) >= n_frames:
        human_traj = flipped_joints[:n_frames]
    else:
        # 线性插值
        t_old = np.linspace(0, 1, len(flipped_joints))
        t_new = np.linspace(0, 1, n_frames)
        human_traj = np.zeros((n_frames, 27))
        for d in range(27):
            human_traj[:, d] = np.interp(t_new, t_old, flipped_joints[:, d])

    # 添加噪声
    noise = np.random.normal(0, 2, human_traj.shape)
    human_traj = human_traj + noise

    return human_traj


def augment_initial_position(joints, arms, phases, head_pos, y_fixed=1000.0):
    """
    完整的初始位置数据增强

    Returns:
        aug_joints, aug_arms, aug_phases, metadata
    """
    # 1. 提取原始机械臂姿态
    original_arms_3d = arms[0].reshape(4, 3)  # [L1, L2, R1, R2]
    original_mid = original_arms_3d.mean(axis=0)
    original_distance = np.linalg.norm(original_mid - head_pos)

    # 2. 采样新起始姿态（位置+朝向，平行于XZ平面，高度=y_fixed）
    start_arms, sampled_mid, sampled_distance = sample_arm_start_pose(
        original_arms_3d, head_pos, original_distance,
        y_fixed=y_fixed, half_angle_deg=15.0
    )

    # 3. 使用刚体变换插值轨迹
    interp_arms, n_frames = interpolate_robot_trajectory(
        start_arms, original_arms_3d, max_speed=10.0
    )

    # 4. 生成人体姿态
    human_traj = generate_human_trajectory(joints, phases, n_frames)

    # 5. 生成phase标签
    phase_prefix = np.zeros(n_frames, dtype=int)

    # 6. 拼接
    aug_joints = np.vstack([human_traj, joints])
    aug_arms = np.vstack([interp_arms, arms])
    aug_phases = np.concatenate([phase_prefix, phases])

    # 元数据
    metadata = {
        'n_augmented_frames': n_frames,
        'original_distance': original_distance,
        'sampled_distance': sampled_distance,
        'travel_distance': np.linalg.norm(sampled_mid - original_mid),
        'avg_speed': np.linalg.norm(sampled_mid - original_mid) / n_frames if n_frames > 0 else 0,
        'original_start': original_mid,
        'new_start': sampled_mid
    }

    return aug_joints, aug_arms, aug_phases, metadata


def visualize_augmentation_animated(original_joints, original_arms, original_phases,
                                   aug_joints, aug_arms, aug_phases, metadata, head_pos, torso_pos):
    """3D动画可视化增强效果"""

    print("\n" + "="*60)
    print("数据增强信息")
    print("="*60)
    print(f"原始轨迹长度: {len(original_joints)} 帧")
    print(f"增强轨迹长度: {len(aug_joints)} 帧")
    print(f"新增帧数: {metadata['n_augmented_frames']} 帧")
    print(f"\n原始起始距离: {metadata['original_distance']:.1f} mm")
    print(f"采样起始距离: {metadata['sampled_distance']:.1f} mm")
    print(f"移动距离: {metadata['travel_distance']:.1f} mm")
    print(f"平均速度: {metadata['avg_speed']:.2f} mm/frame ({metadata['avg_speed']*30:.1f} mm/s)")
    print(f"\n原始起始位置: ({metadata['original_start'][0]:.1f}, {metadata['original_start'][1]:.1f}, {metadata['original_start'][2]:.1f})")
    print(f"新起始位置: ({metadata['new_start'][0]:.1f}, {metadata['new_start'][1]:.1f}, {metadata['new_start'][2]:.1f})")
    print("="*60 + "\n")

    # 预计算所有数据
    aug_centers = (aug_arms[:, 0:3] + aug_arms[:, 6:9]) / 2
    aug_L1 = aug_arms[:, 0:3]
    aug_L2 = aug_arms[:, 3:6]
    aug_R1 = aug_arms[:, 6:9]
    aug_R2 = aug_arms[:, 9:12]

    aug_dists = np.array([calculate_distance_xz(c, torso_pos) for c in aug_centers])
    aug_speeds = np.linalg.norm(np.diff(aug_centers, axis=0), axis=1)
    aug_speeds = np.concatenate([[0], aug_speeds])  # 第一帧速度为0

    # 人体关节连接定义
    skeleton_connections = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (1, 8)
    ]

    # 创建图形
    fig = plt.figure(figsize=(20, 10))

    # 1. 3D主视图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_title('Augmented Trajectory (3D View)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_zlabel('Y (mm)')

    # 设置坐标轴范围
    all_points = np.vstack([aug_centers, aug_joints.reshape(-1, 9, 3).reshape(-1, 3)])
    x_range = [all_points[:, 0].min() - 200, all_points[:, 0].max() + 200]
    y_range = [all_points[:, 2].min() - 200, all_points[:, 2].max() + 200]
    z_range = [all_points[:, 1].min() - 200, all_points[:, 1].max() + 200]
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim(z_range)

    # 绘制完整轨迹路径
    ax1.plot(aug_centers[:, 0], aug_centers[:, 2], aug_centers[:, 1],
             'gray', linewidth=1, alpha=0.3, label='Robot Path')

    # 初始化动态元素
    robot_traj_line, = ax1.plot([], [], [], 'r-', linewidth=2, alpha=0.7, label='Current Robot')
    human_skeleton_lines = [ax1.plot([], [], [], 'b-', linewidth=2)[0] for _ in skeleton_connections]
    human_joints_scatter = ax1.scatter([], [], [], c='blue', s=50, alpha=0.8)

    left_arm_line, = ax1.plot([], [], [], 'g-', linewidth=3, marker='o', markersize=8, label='Left Arm')
    right_arm_line, = ax1.plot([], [], [], 'm-', linewidth=3, marker='o', markersize=8, label='Right Arm')

    ax1.legend(loc='upper left')

    # 2. 距离曲线
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title('Robot-to-Human Distance', fontsize=14, fontweight='bold')
    ax2.plot(range(len(aug_dists)), aug_dists, 'gray', linewidth=1, alpha=0.3)
    ax2.axvline(x=metadata['n_augmented_frames'], color='orange', linestyle='--',
                linewidth=2, label=f'Aug End (frame {metadata["n_augmented_frames"]})')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Distance (mm)')
    ax2.set_ylim([aug_dists.min() - 100, aug_dists.max() + 100])
    ax2.grid(True, alpha=0.3)

    dist_line, = ax2.plot([], [], 'r-', linewidth=2)
    dist_marker = ax2.scatter([], [], c='red', s=100, zorder=5)
    dist_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.legend()

    # 3. 速度曲线
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title('Robot Speed', fontsize=14, fontweight='bold')
    ax3.plot(range(len(aug_speeds)), aug_speeds, 'gray', linewidth=1, alpha=0.3)
    ax3.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='Max Speed (10 mm/frame)')
    ax3.axvline(x=metadata['n_augmented_frames'], color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Speed (mm/frame)')
    ax3.set_ylim([0, max(15, aug_speeds.max() + 2)])
    ax3.grid(True, alpha=0.3)

    speed_line, = ax3.plot([], [], 'r-', linewidth=2)
    speed_marker = ax3.scatter([], [], c='red', s=100, zorder=5)
    speed_text = ax3.text(0.02, 0.98, '', transform=ax3.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.legend()

    # 4. 俯视图 + 采样区域（锥形，以head为中心）
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title('Top View (XZ) + Sampling Cone (±15°)', fontsize=14, fontweight='bold')
    ax4.plot(aug_centers[:, 0], aug_centers[:, 2], 'gray', linewidth=1, alpha=0.3)

    # 标记人的位置
    ax4.scatter(head_pos[0], head_pos[2], c='purple', s=200, marker='*',
                label='Head (cone center)', zorder=5, edgecolors='black', linewidths=1)
    ax4.scatter(torso_pos[0], torso_pos[2], c='green', s=150, marker='^',
                label='Torso', zorder=5)

    # 绘制采样锥形区域（以head为中心）
    original_robot_center = metadata['original_start']
    original_distance = metadata['original_distance']

    # 计算原始方向（head -> 机械臂中点）
    direction_vec = original_robot_center - head_pos
    original_angle = np.arctan2(direction_vec[2], direction_vec[0])

    # 锥形半角15度
    half_angle = 15 * np.pi / 180
    angles = np.linspace(original_angle - half_angle, original_angle + half_angle, 50)

    # 内圆弧（原始距离）
    inner_x = head_pos[0] + original_distance * np.cos(angles)
    inner_z = head_pos[2] + original_distance * np.sin(angles)
    ax4.plot(inner_x, inner_z, 'orange', linewidth=2, linestyle='--', alpha=0.6,
             label='Cone (±15°)')

    # 外圆弧（2000mm）
    outer_x = head_pos[0] + 2000 * np.cos(angles)
    outer_z = head_pos[2] + 2000 * np.sin(angles)
    ax4.plot(outer_x, outer_z, 'orange', linewidth=2, linestyle='--', alpha=0.6)

    # 锥形边界线
    left_angle = original_angle - half_angle
    right_angle = original_angle + half_angle
    ax4.plot([head_pos[0], head_pos[0] + 2000*np.cos(left_angle)],
             [head_pos[2], head_pos[2] + 2000*np.sin(left_angle)],
             'orange', linewidth=2, linestyle='--', alpha=0.6)
    ax4.plot([head_pos[0], head_pos[0] + 2000*np.cos(right_angle)],
             [head_pos[2], head_pos[2] + 2000*np.sin(right_angle)],
             'orange', linewidth=2, linestyle='--', alpha=0.6)

    # 填充锥形区域
    for r in np.linspace(original_distance, 2000, 20):
        x = head_pos[0] + r * np.cos(angles)
        z = head_pos[2] + r * np.sin(angles)
        ax4.fill(x, z, color='yellow', alpha=0.03)

    # 标记原始起始点
    ax4.scatter(original_robot_center[0], original_robot_center[2],
                c='blue', s=150, marker='o', label='Original Start', zorder=5)

    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    topview_traj, = ax4.plot([], [], 'r-', linewidth=2, label='Aug Trajectory')
    topview_robot = ax4.scatter([], [], c='red', s=150, marker='s',
                               label='Aug Start', zorder=5)
    ax4.legend(loc='upper right', fontsize=7)

    # 5. Phase显示
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title('Phase Timeline', fontsize=14, fontweight='bold')

    phase_colors = {0: 'yellow', 1: 'green', 2: 'orange'}
    phase_names = {0: 'Approaching', 1: 'Assisting', 2: 'Leaving'}
    for i, phase in enumerate(aug_phases):
        ax5.axvspan(i, i+1, alpha=0.3, color=phase_colors[phase])
    ax5.axvline(x=metadata['n_augmented_frames'], color='orange', linestyle='--',
                linewidth=2, alpha=0.7)
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Phase')
    ax5.set_ylim([-0.5, 2.5])
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(['Approaching', 'Assisting', 'Leaving'])
    ax5.grid(True, alpha=0.3, axis='x')

    phase_marker = ax5.axvline(x=0, color='red', linewidth=3, alpha=0.8)
    phase_text = ax5.text(0.02, 0.98, '', transform=ax5.transAxes,
                         verticalalignment='top', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 6. 信息面板
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    info_text = ax6.text(0.1, 0.9, '', transform=ax6.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    def init():
        robot_traj_line.set_data([], [])
        robot_traj_line.set_3d_properties([])
        for line in human_skeleton_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        human_joints_scatter._offsets3d = ([], [], [])
        left_arm_line.set_data([], [])
        left_arm_line.set_3d_properties([])
        right_arm_line.set_data([], [])
        right_arm_line.set_3d_properties([])

        dist_line.set_data([], [])
        dist_marker.set_offsets(np.empty((0, 2)))
        speed_line.set_data([], [])
        speed_marker.set_offsets(np.empty((0, 2)))
        topview_traj.set_data([], [])
        topview_robot.set_offsets(np.empty((0, 2)))

        return (robot_traj_line, left_arm_line, right_arm_line,
                dist_line, dist_marker, speed_line, speed_marker,
                topview_traj, topview_robot, phase_marker, info_text)

    def update(frame):
        # 3D视图更新
        robot_traj_line.set_data(aug_centers[:frame+1, 0], aug_centers[:frame+1, 2])
        robot_traj_line.set_3d_properties(aug_centers[:frame+1, 1])

        # 人体骨架
        joints_3d = aug_joints[frame].reshape(9, 3)
        for i, (j1, j2) in enumerate(skeleton_connections):
            xs = [joints_3d[j1, 0], joints_3d[j2, 0]]
            ys = [joints_3d[j1, 2], joints_3d[j2, 2]]
            zs = [joints_3d[j1, 1], joints_3d[j2, 1]]
            human_skeleton_lines[i].set_data(xs, ys)
            human_skeleton_lines[i].set_3d_properties(zs)

        human_joints_scatter._offsets3d = (joints_3d[:, 0], joints_3d[:, 2], joints_3d[:, 1])

        # 机械臂
        left_arm_line.set_data([aug_L1[frame, 0], aug_L2[frame, 0]],
                               [aug_L1[frame, 2], aug_L2[frame, 2]])
        left_arm_line.set_3d_properties([aug_L1[frame, 1], aug_L2[frame, 1]])

        right_arm_line.set_data([aug_R1[frame, 0], aug_R2[frame, 0]],
                                [aug_R1[frame, 2], aug_R2[frame, 2]])
        right_arm_line.set_3d_properties([aug_R1[frame, 1], aug_R2[frame, 1]])

        # 距离曲线
        dist_line.set_data(range(frame+1), aug_dists[:frame+1])
        dist_marker.set_offsets([[frame, aug_dists[frame]]])
        dist_text.set_text(f'Frame: {frame}\nDist: {aug_dists[frame]:.1f} mm')

        # 速度曲线
        speed_line.set_data(range(frame+1), aug_speeds[:frame+1])
        speed_marker.set_offsets([[frame, aug_speeds[frame]]])
        speed_text.set_text(f'Frame: {frame}\nSpeed: {aug_speeds[frame]:.2f} mm/frame\n({aug_speeds[frame]*30:.1f} mm/s)')

        # 俯视图
        topview_traj.set_data(aug_centers[:frame+1, 0], aug_centers[:frame+1, 2])
        topview_robot.set_offsets([[aug_centers[frame, 0], aug_centers[frame, 2]]])

        # Phase标记
        phase_marker.set_xdata([frame, frame])
        current_phase = aug_phases[frame]
        phase_text.set_text(f'Phase: {phase_names[current_phase]}')

        # 信息面板
        is_augmented = frame < metadata['n_augmented_frames']
        info_str = f"""Frame: {frame} / {len(aug_joints)-1}
Status: {'AUGMENTED' if is_augmented else 'ORIGINAL'}

Current State:
  Distance: {aug_dists[frame]:.1f} mm
  Speed: {aug_speeds[frame]:.2f} mm/frame
  Phase: {phase_names[current_phase]}

Augmentation Info:
  Total Frames: {len(aug_joints)}
  Aug Frames: {metadata['n_augmented_frames']}
  Orig Frames: {len(original_joints)}

  Travel Dist: {metadata['travel_distance']:.1f} mm
  Avg Speed: {metadata['avg_speed']:.2f} mm/frame
"""
        info_text.set_text(info_str)

        return (robot_traj_line, left_arm_line, right_arm_line,
                dist_line, dist_marker, speed_line, speed_marker,
                topview_traj, topview_robot, phase_marker, info_text)

    plt.tight_layout()

    # 创建动画
    n_frames = len(aug_joints)
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=50, blit=False, repeat=True)

    print("显示动画中... 关闭窗口以退出")
    plt.show()


def main():
    # 读取一条真实轨迹
    print("加载测试数据...")
    df = pd.read_csv('inference_trajectory_gt_train.csv')

    # 选择一条轨迹
    traj_ids = df['traj_id'].unique()
    test_traj_id = np.random.choice(traj_ids)
    print(f"选择轨迹: {test_traj_id}")

    traj_df = df[df['traj_id'] == test_traj_id].sort_values('frame_id').reset_index(drop=True)

    # 提取关节数据（9个关节 × 3坐标）
    joint_names = ['Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
                   'R_Hand', 'L_Hand', 'Torso']
    joint_cols = []
    for jname in joint_names:
        joint_cols.extend([f'{jname}_gt_x', f'{jname}_gt_y', f'{jname}_gt_z'])

    joints = traj_df[joint_cols].values.astype(np.float32)  # [T, 27]

    # 提取机械臂数据
    arm_cols = ['Left_L1_x', 'Left_L1_y', 'Left_L1_z',
                'Left_L2_x', 'Left_L2_y', 'Left_L2_z',
                'Right_R1_x', 'Right_R1_y', 'Right_R1_z',
                'Right_R2_x', 'Right_R2_y', 'Right_R2_z']
    arms = traj_df[arm_cols].values.astype(np.float32)  # [T, 12]

    # 提取phase标签
    phases = traj_df['phase_gt'].values.astype(int)  # [T]

    # 计算人的位置（Head用于采样）
    head_x = traj_df['Head_gt_x'].values[0]
    head_y = traj_df['Head_gt_y'].values[0]
    head_z = traj_df['Head_gt_z'].values[0]
    head_pos = np.array([head_x, head_y, head_z])

    torso_x = traj_df['Torso_gt_x'].values[0]
    torso_y = traj_df['Torso_gt_y'].values[0]
    torso_z = traj_df['Torso_gt_z'].values[0]
    torso_pos = np.array([torso_x, torso_y, torso_z])

    print(f"原始轨迹长度: {len(joints)} 帧")
    print(f"Phase分布: Approaching={np.sum(phases==0)}, Assisting={np.sum(phases==1)}, Leaving={np.sum(phases==2)}")

    # 执行数据增强
    print("\n执行初始位置数据增强...")
    aug_joints, aug_arms, aug_phases, metadata = augment_initial_position(
        joints, arms, phases, head_pos, y_fixed=1000.0
    )

    # 可视化
    visualize_augmentation_animated(joints, arms, phases,
                                   aug_joints, aug_arms, aug_phases,
                                   metadata, head_pos, torso_pos)


if __name__ == "__main__":
    # 使用随机种子以查看不同的增强结果
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    print(f"随机种子: {seed}")
    main()
