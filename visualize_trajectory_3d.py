#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# 读取CSV文件
gt_df = pd.read_csv('inference_trajectory_gt_test.csv')
pred_df = pd.read_csv('inference_trajectory_test.csv')

# 随机选择一条轨迹
unique_traj_ids = gt_df['traj_id'].unique()
random_traj_id = np.random.choice(unique_traj_ids)
print(f"随机选择的轨迹ID: {random_traj_id}")

# 筛选该条轨迹
gt_traj = gt_df[gt_df['traj_id'] == random_traj_id].reset_index(drop=True)
pred_traj = pred_df[pred_df['traj_id'] == random_traj_id].reset_index(drop=True)

# 定义关键点列表（只包含预测文件中有的上半身关键点）
keypoints_pred = ['Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
                  'R_Hand', 'L_Hand', 'Torso']

# GT包含所有关键点
keypoints_gt = ['Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow',
                'R_Hand', 'L_Hand', 'Torso', 'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee',
                'R_Foot', 'L_Foot']

# 机械臂关键点（GT和Pred都有）
robot_keypoints = ['Left_L1', 'Left_L2', 'Right_R1', 'Right_R2']

# 定义骨架连接（用于绘制骨骼）- 预测数据只有上半身
skeleton_connections_pred = [
    ('Head', 'Neck'),
    ('Neck', 'R_Shoulder'), ('Neck', 'L_Shoulder'),
    ('R_Shoulder', 'R_Elbow'), ('L_Shoulder', 'L_Elbow'),
    ('R_Elbow', 'R_Hand'), ('L_Elbow', 'L_Hand'),
    ('Neck', 'Torso'),
]

# GT骨架连接包含下半身
skeleton_connections_gt = [
    ('Head', 'Neck'),
    ('Neck', 'R_Shoulder'), ('Neck', 'L_Shoulder'),
    ('R_Shoulder', 'R_Elbow'), ('L_Shoulder', 'L_Elbow'),
    ('R_Elbow', 'R_Hand'), ('L_Elbow', 'L_Hand'),
    ('Neck', 'Torso'),
    ('Torso', 'R_Hip'), ('Torso', 'L_Hip'),
    ('R_Hip', 'R_Knee'), ('L_Hip', 'L_Knee'),
    ('R_Knee', 'R_Foot'), ('L_Knee', 'L_Foot'),
]

# 机械臂连接
robot_connections = [
    ('Left_L1', 'Left_L2'),
    ('Right_R1', 'Right_R2'),
]

# 预计算所有帧的数据用于曲线绘制
num_frames = len(gt_traj)
gt_robot_to_human_dists = []
pred_robot_to_human_dists = []
gt_human_heights = []
pred_human_heights = []
gt_arm_heights = []
pred_arm_heights = []

def get_keypoint_data(df, frame_idx, keypoint, suffix):
    """获取特定关键点的坐标"""
    x = df.loc[frame_idx, f'{keypoint}_{suffix}_x']
    y = df.loc[frame_idx, f'{keypoint}_{suffix}_y']
    z = df.loc[frame_idx, f'{keypoint}_{suffix}_z']
    return x, y, z

def get_robot_data(df, frame_idx, keypoint):
    """获取机械臂关键点的坐标（无suffix）"""
    x = df.loc[frame_idx, f'{keypoint}_x']
    y = df.loc[frame_idx, f'{keypoint}_y']
    z = df.loc[frame_idx, f'{keypoint}_z']
    return x, y, z

print("预计算所有帧的统计数据...")
for i in range(num_frames):
    # GT数据
    l1_gt = np.array(get_robot_data(gt_traj, i, 'Left_L1'))
    r1_gt = np.array(get_robot_data(gt_traj, i, 'Right_R1'))
    robot_pos_gt = (l1_gt + r1_gt) / 2  # L1和R1的中点
    robot_xz_gt = np.array([robot_pos_gt[0], robot_pos_gt[2]])  # XZ平面投影

    torso_gt = np.array(get_keypoint_data(gt_traj, i, 'Torso', 'gt'))
    human_xz_gt = np.array([torso_gt[0], torso_gt[2]])  # XZ平面投影
    human_height_gt = torso_gt[1]  # Y坐标

    # 手臂高度：L/R Hand 和 L/R Elbow 的平均Y高度
    r_hand_gt = np.array(get_keypoint_data(gt_traj, i, 'R_Hand', 'gt'))
    l_hand_gt = np.array(get_keypoint_data(gt_traj, i, 'L_Hand', 'gt'))
    r_elbow_gt = np.array(get_keypoint_data(gt_traj, i, 'R_Elbow', 'gt'))
    l_elbow_gt = np.array(get_keypoint_data(gt_traj, i, 'L_Elbow', 'gt'))
    arm_height_gt = np.mean([r_hand_gt[1], l_hand_gt[1], r_elbow_gt[1], l_elbow_gt[1]])

    dist_gt = np.linalg.norm(robot_xz_gt - human_xz_gt)
    gt_robot_to_human_dists.append(dist_gt)
    gt_human_heights.append(human_height_gt)
    gt_arm_heights.append(arm_height_gt)

    # Pred数据
    l1_pred = np.array(get_robot_data(pred_traj, i, 'Left_L1'))
    r1_pred = np.array(get_robot_data(pred_traj, i, 'Right_R1'))
    robot_pos_pred = (l1_pred + r1_pred) / 2
    robot_xz_pred = np.array([robot_pos_pred[0], robot_pos_pred[2]])

    torso_pred = np.array(get_keypoint_data(pred_traj, i, 'Torso', 'pred'))
    human_xz_pred = np.array([torso_pred[0], torso_pred[2]])
    human_height_pred = torso_pred[1]

    r_hand_pred = np.array(get_keypoint_data(pred_traj, i, 'R_Hand', 'pred'))
    l_hand_pred = np.array(get_keypoint_data(pred_traj, i, 'L_Hand', 'pred'))
    r_elbow_pred = np.array(get_keypoint_data(pred_traj, i, 'R_Elbow', 'pred'))
    l_elbow_pred = np.array(get_keypoint_data(pred_traj, i, 'L_Elbow', 'pred'))
    arm_height_pred = np.mean([r_hand_pred[1], l_hand_pred[1], r_elbow_pred[1], l_elbow_pred[1]])

    dist_pred = np.linalg.norm(robot_xz_pred - human_xz_pred)
    pred_robot_to_human_dists.append(dist_pred)
    pred_human_heights.append(human_height_pred)
    pred_arm_heights.append(arm_height_pred)

# 切分轨迹阶段（基于GT距离）
def segment_trajectory(distances, window_size=5):
    """
    将轨迹切分为三个阶段：approaching, assisting, leaving
    - approaching: 机器人靠近（距离减小）
    - assisting: 机器人协助（距离保持低值）
    - leaving: 机器人离开（距离增加）
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

phases_gt = segment_trajectory(gt_robot_to_human_dists)
phases_pred = segment_trajectory(pred_robot_to_human_dists)

print(f"\nGT Phase Segmentation:")
print(f"  Approaching: frames {phases_gt['approaching'][0]}-{phases_gt['approaching'][1]}")
print(f"  Assisting:   frames {phases_gt['assisting'][0]}-{phases_gt['assisting'][1]}")
print(f"  Leaving:     frames {phases_gt['leaving'][0]}-{phases_gt['leaving'][1]}")

print(f"\nPred Phase Segmentation:")
print(f"  Approaching: frames {phases_pred['approaching'][0]}-{phases_pred['approaching'][1]}")
print(f"  Assisting:   frames {phases_pred['assisting'][0]}-{phases_pred['assisting'][1]}")
print(f"  Leaving:     frames {phases_pred['leaving'][0]}-{phases_pred['leaving'][1]}")

# 创建图形布局：左边3D图，右边两个曲线图
fig = plt.figure(figsize=(18, 8))
gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], hspace=0.3, wspace=0.3)

ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_dist = fig.add_subplot(gs[0, 1])
ax_height = fig.add_subplot(gs[1, 1])

# 初始化线条和点
all_lines = []
all_points = []
dist_line_gt = None
dist_line_pred = None
dist_marker_gt = None
dist_marker_pred = None
height_line_gt = None
height_line_pred = None
arm_height_line_gt = None
arm_height_line_pred = None
height_marker_gt = None
height_marker_pred = None
arm_marker_gt = None
arm_marker_pred = None

def init():
    """初始化动画"""
    # 3D图设置 - Y轴朝上
    ax_3d.set_xlabel('X (mm)')
    ax_3d.set_ylabel('Y (mm) - Height')
    ax_3d.set_zlabel('Z (mm)')
    ax_3d.view_init(elev=20, azim=45)

    # 距离曲线图
    ax_dist.set_xlabel('Frame')
    ax_dist.set_ylabel('Distance (mm)')
    ax_dist.set_title('Robot-to-Human Distance (XZ plane)')
    ax_dist.grid(True, alpha=0.3)

    # 高度曲线图
    ax_height.set_xlabel('Frame')
    ax_height.set_ylabel('Height (mm)')
    ax_height.set_title('Human Height & Arm Height')
    ax_height.grid(True, alpha=0.3)

    return []

def update(frame):
    """更新动画帧"""
    global all_lines, all_points
    global dist_line_gt, dist_line_pred, dist_marker_gt, dist_marker_pred
    global height_line_gt, height_line_pred, arm_height_line_gt, arm_height_line_pred
    global height_marker_gt, height_marker_pred, arm_marker_gt, arm_marker_pred

    # 清空3D图的线条和点
    for line in all_lines:
        line.remove()
    for point in all_points:
        point.remove()
    all_lines = []
    all_points = []

    # Ground Truth 数据
    gt_xs, gt_ys, gt_zs = [], [], []
    for kp in keypoints_gt:
        x, y, z = get_keypoint_data(gt_traj, frame, kp, 'gt')
        gt_xs.append(x)
        gt_ys.append(y)
        gt_zs.append(z)

    # 绘制GT关键点
    points = ax_3d.scatter(gt_xs, gt_ys, gt_zs, c='blue', marker='o', s=50, alpha=0.8, label='GT')
    all_points.append(points)

    # 绘制GT骨架
    for conn in skeleton_connections_gt:
        idx1 = keypoints_gt.index(conn[0])
        idx2 = keypoints_gt.index(conn[1])
        line = ax_3d.plot([gt_xs[idx1], gt_xs[idx2]],
                          [gt_ys[idx1], gt_ys[idx2]],
                          [gt_zs[idx1], gt_zs[idx2]],
                          'b-', linewidth=2, alpha=0.6)[0]
        all_lines.append(line)

    # Predicted 数据
    pred_xs, pred_ys, pred_zs = [], [], []
    for kp in keypoints_pred:
        x, y, z = get_keypoint_data(pred_traj, frame, kp, 'pred')
        pred_xs.append(x)
        pred_ys.append(y)
        pred_zs.append(z)

    # 绘制Pred关键点
    points = ax_3d.scatter(pred_xs, pred_ys, pred_zs, c='red', marker='s', s=50, alpha=0.8, label='Pred')
    all_points.append(points)

    # 绘制Pred骨架
    for conn in skeleton_connections_pred:
        idx1 = keypoints_pred.index(conn[0])
        idx2 = keypoints_pred.index(conn[1])
        line = ax_3d.plot([pred_xs[idx1], pred_xs[idx2]],
                          [pred_ys[idx1], pred_ys[idx2]],
                          [pred_zs[idx1], pred_zs[idx2]],
                          'r--', linewidth=2, alpha=0.6)[0]
        all_lines.append(line)

    # 绘制机械臂（GT）
    robot_gt_xs, robot_gt_ys, robot_gt_zs = [], [], []
    for kp in robot_keypoints:
        x, y, z = get_robot_data(gt_traj, frame, kp)
        robot_gt_xs.append(x)
        robot_gt_ys.append(y)
        robot_gt_zs.append(z)

    # 绘制机械臂关键点
    points = ax_3d.scatter(robot_gt_xs, robot_gt_ys, robot_gt_zs, c='green', marker='^', s=100, alpha=0.9, label='Robot')
    all_points.append(points)

    for conn in robot_connections:
        idx1 = robot_keypoints.index(conn[0])
        idx2 = robot_keypoints.index(conn[1])
        line = ax_3d.plot([robot_gt_xs[idx1], robot_gt_xs[idx2]],
                          [robot_gt_ys[idx1], robot_gt_ys[idx2]],
                          [robot_gt_zs[idx1], robot_gt_zs[idx2]],
                          'g-', linewidth=4, alpha=0.9)[0]
        all_lines.append(line)

    # 设置3D图坐标轴范围
    all_xs = gt_xs + pred_xs + robot_gt_xs
    all_ys = gt_ys + pred_ys + robot_gt_ys
    all_zs = gt_zs + pred_zs + robot_gt_zs

    ax_3d.set_xlim([min(all_xs) - 200, max(all_xs) + 200])
    ax_3d.set_ylim([min(all_ys) - 200, max(all_ys) + 200])
    ax_3d.set_zlim([min(all_zs) - 200, max(all_zs) + 200])

    # 绘制距离曲线（整条轨迹）
    if frame == 0:  # 只在第一帧绘制完整曲线
        frames = list(range(num_frames))

        # 绘制阶段背景色（基于GT阶段）
        ax_dist.axvspan(phases_gt['approaching'][0], phases_gt['approaching'][1],
                        alpha=0.2, color='yellow', label='Approaching')
        ax_dist.axvspan(phases_gt['assisting'][0], phases_gt['assisting'][1],
                        alpha=0.2, color='green', label='Assisting')
        ax_dist.axvspan(phases_gt['leaving'][0], phases_gt['leaving'][1],
                        alpha=0.2, color='orange', label='Leaving')

        ax_dist.plot(frames, gt_robot_to_human_dists, 'b-', linewidth=2, label='GT', alpha=0.7)
        ax_dist.plot(frames, pred_robot_to_human_dists, 'r--', linewidth=2, label='Pred', alpha=0.7)
        ax_dist.set_xlabel('Frame')
        ax_dist.set_ylabel('Distance (mm)')
        ax_dist.set_title('Robot-to-Human Distance (XZ plane)')
        ax_dist.grid(True, alpha=0.3)
        ax_dist.legend(loc='upper right', fontsize=8)
        ax_dist.set_xlim(0, num_frames)

        # 绘制高度曲线（整条轨迹）
        # 绘制阶段背景色
        ax_height.axvspan(phases_gt['approaching'][0], phases_gt['approaching'][1],
                          alpha=0.2, color='yellow')
        ax_height.axvspan(phases_gt['assisting'][0], phases_gt['assisting'][1],
                          alpha=0.2, color='green')
        ax_height.axvspan(phases_gt['leaving'][0], phases_gt['leaving'][1],
                          alpha=0.2, color='orange')

        ax_height.plot(frames, gt_human_heights, 'b-', linewidth=2, label='GT Torso', alpha=0.7)
        ax_height.plot(frames, pred_human_heights, 'r--', linewidth=2, label='Pred Torso', alpha=0.7)
        ax_height.plot(frames, gt_arm_heights, 'b:', linewidth=2, label='GT Arm', alpha=0.7)
        ax_height.plot(frames, pred_arm_heights, 'r-.', linewidth=2, label='Pred Arm', alpha=0.7)
        ax_height.set_xlabel('Frame')
        ax_height.set_ylabel('Height (mm)')
        ax_height.set_title('Human Height & Arm Height')
        ax_height.grid(True, alpha=0.3)
        ax_height.legend(loc='upper right', fontsize=8)
        ax_height.set_xlim(0, num_frames)

    # 清除距离图的所有动态元素（垂直线、散点、文本）
    for artist in list(ax_dist.get_children()):
        if isinstance(artist, plt.Line2D) and artist.get_linestyle() == ':':
            artist.remove()
        elif hasattr(artist, 'get_offsets'):  # PathCollection (scatter)
            artist.remove()
        elif isinstance(artist, plt.Text) and artist.get_bbox_patch() is not None:
            artist.remove()

    # 清除高度图的所有动态元素
    for artist in list(ax_height.get_children()):
        if isinstance(artist, plt.Line2D) and artist.get_linestyle() == ':':
            artist.remove()
        elif hasattr(artist, 'get_offsets'):  # PathCollection (scatter)
            artist.remove()
        elif isinstance(artist, plt.Text) and artist.get_bbox_patch() is not None:
            artist.remove()

    # 确定当前阶段
    current_phase = 'Unknown'
    if phases_gt['approaching'][0] <= frame <= phases_gt['approaching'][1]:
        current_phase = 'Approaching'
    elif phases_gt['assisting'][0] <= frame <= phases_gt['assisting'][1]:
        current_phase = 'Assisting'
    elif phases_gt['leaving'][0] <= frame <= phases_gt['leaving'][1]:
        current_phase = 'Leaving'

    # 添加当前帧的垂直线和标记
    ax_dist.axvline(x=frame, color='gray', linestyle=':', alpha=0.5)
    ax_dist.scatter([frame], [gt_robot_to_human_dists[frame]], c='blue', s=150, zorder=5, edgecolors='black', linewidth=2)
    ax_dist.scatter([frame], [pred_robot_to_human_dists[frame]], c='red', s=150, zorder=5, edgecolors='black', linewidth=2)
    ax_dist.text(0.02, 0.98, f'Frame {frame} | Phase: {current_phase}\nGT: {gt_robot_to_human_dists[frame]:.1f} mm\nPred: {pred_robot_to_human_dists[frame]:.1f} mm',
                 transform=ax_dist.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax_height.axvline(x=frame, color='gray', linestyle=':', alpha=0.5)
    ax_height.scatter([frame], [gt_human_heights[frame]], c='blue', s=150, zorder=5, edgecolors='black', linewidth=2)
    ax_height.scatter([frame], [pred_human_heights[frame]], c='red', s=150, zorder=5, edgecolors='black', linewidth=2)
    ax_height.scatter([frame], [gt_arm_heights[frame]], c='blue', marker='x', s=150, zorder=5, linewidths=3)
    ax_height.scatter([frame], [pred_arm_heights[frame]], c='red', marker='x', s=150, zorder=5, linewidths=3)
    ax_height.text(0.02, 0.98, f'Frame {frame}:\nGT Torso: {gt_human_heights[frame]:.1f} mm\nPred Torso: {pred_human_heights[frame]:.1f} mm\n'
                               f'GT Arm: {gt_arm_heights[frame]:.1f} mm\nPred Arm: {pred_arm_heights[frame]:.1f} mm',
                   transform=ax_height.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 更新帧信息
    fig.suptitle(f'Frame: {frame}/{num_frames-1} | Trajectory: {random_traj_id}', fontsize=14, y=0.98)

    # 只在第一帧添加图例
    if frame == 0:
        ax_3d.legend(loc='upper left')

    return all_lines + all_points

# 创建动画
print(f"创建动画中，共 {num_frames} 帧...")
anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                    interval=100, blit=False, repeat=True)

# 直接显示动画
plt.tight_layout()
plt.show()
