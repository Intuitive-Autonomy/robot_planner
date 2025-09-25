#!/usr/bin/env python3
"""
Simplified motion prediction validation visualization for the new endpoint(12D) model.
- Input: skeleton only (63D), uses past 120 frames as context
- Output: EEF endpoints (12D = 4 points * 3D)
- Visualizes GT vs Pred endpoints and lines (L1-L2, R1-R2), plus skeleton

This version:
- Cache stores only RAW model predictions (no speed cap applied)
- Per-timestep speed cap is applied AT RUNTIME (based on --max-speed and --fps)
- Runtime-capped sequences are memoized in-process to avoid recomputation on the same settings
"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

from net import get_model
from utils import CSVTrajectoryProcessor  # returns skeleton 27D input, endpoints 12D target

# Skeleton connections for visualization (indices follow JOINT_NAMES order below)
SKELETON_BONES = [
    ('Ab', 'Chest'), ('Chest', 'Neck'), ('Neck', 'Head'),
    ('Chest', 'LShoulder'), ('LShoulder', 'LUArm'), ('LUArm', 'LFArm'), ('LFArm', 'LHand'),
    ('Chest', 'RShoulder'), ('RShoulder', 'RUArm'), ('RUArm', 'RFArm'), ('RFArm', 'RHand'),
    ('Ab', 'LThigh'), ('LThigh', 'LShin'), ('LShin', 'LFoot'), ('LFoot', 'LToe'),
    ('Ab', 'RThigh'), ('RThigh', 'RShin'), ('RShin', 'RFoot'), ('RFoot', 'RToe')
]

JOINT_NAMES = ['Ab', 'Chest', 'Head', 'LFArm', 'LFoot', 'LHand', 'LShin', 'LShoulder',
               'LThigh', 'LToe', 'LUArm', 'Neck', 'RFArm', 'RFoot', 'RHand', 'RShin',
               'RShoulder', 'RThigh', 'RToe', 'RUArm', 'Skeleton']


def load_config(config_path):
    """Load training configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded config from {config_path}")
    return config


class MotionPredictor:
    """Lightweight checkpoint loader for inference with the new endpoint model."""
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        self.device = device

        # load checkpoint (saved by train.py)
        ckpt = torch.load(model_path, map_location=device)

        # read config from specified config file
        cfg = load_config(config_path)

        self.seq_len  = cfg.get("sequence_length", 120)
        self.pred_len = cfg.get("prediction_length", 60)

        # Use CSV trajectory dataset (27D input)
        in_dim = 27  # CSV trajectory dataset: 9 upper body joints * 3 coordinates
        self.is_csv_dataset = True
        self.csv_path = cfg.get("csv_path", "inference_trajectory.csv")

        out_dim = cfg.get("output_dim", 12) if "output_dim" in cfg else 12

        # build model consistent with training config
        self.model = get_model(
            hidden_dim=cfg.get("hidden_dim", 256),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            skeleton_dim=in_dim,
            output_dim=out_dim,
            num_heads=cfg.get("num_heads", 2)
        ).to(device)

        # load weights
        state_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
        self.model.load_state_dict(ckpt[state_key], strict=True)
        self.model.eval()

        self.in_dim = in_dim
        print(f"Loaded model from {model_path} | in_dim={self.in_dim}, out_dim={out_dim}, "
              f"seq_len={self.seq_len}, pred_len={self.pred_len}")

    @torch.no_grad()
    def predict_one(self, past_seq: np.ndarray) -> np.ndarray:
        """
        past_seq: [Tin, in_dim] (Tin>=self.seq_len, in_dim=63)
        returns: [12]
        """
        x = torch.from_numpy(past_seq[-self.seq_len:]).float().unsqueeze(0).to(self.device)  # [1,120,63]
        y = self.model(x, pred_len=1, teacher_forcing_ratio=0.0)  # [1,1,12]
        return y[0, 0].cpu().numpy()


class SimpleVisualizer:
    """Real-time validation visualizer for 12D endpoints."""
    def __init__(self, model_path: str, config_path: str, fps: int = 120, max_speed_cm: float = 30.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fps = fps
        self.max_speed_cm = max_speed_cm
        # step limit in mm: cm/s → mm/s, then per-frame
        self.step_cap_mm = (self.max_speed_cm * 10.0) / max(1, self.fps)

        self.predictor = MotionPredictor(model_path, config_path, self.device)
        
        # Read CSV trajectory path from config
        cfg = load_config(config_path)
        csv_path = cfg.get("csv_path", "inference_trajectory.csv")
        print(f"Using CSV trajectory dataset: {csv_path}")
        self.trajectories = self._load_csv_trajectories(csv_path)

        self.current_sample = 0
        self.current_frame = 0
        self.is_animating = False
        self._last_prediction = None  # For speed capping

        print(f"Loaded {len(self.trajectories)} trajectories for real-time visualization")

    def _load_csv_trajectories(self, csv_path):
        """Build (input_traj, target_traj, status, orig_idx) list from CSV trajectory data."""
        proc = CSVTrajectoryProcessor(csv_path)
        traj_ids = proc.traj_ids

        # Pick first few trajectories for visualization
        picked = traj_ids[:5] if len(traj_ids) >= 5 else traj_ids
        print(f"📊 Loading CSV trajectories for validation: {picked}")

        out = []
        for tid in picked:
            try:
                joints, arms = proc.get_trajectory_data(tid)  # joints:[T,27], arms:[T,12]
                # Limit to reasonable length for visualization
                max_frames = min(200, len(joints))
                x = joints[:max_frames]  # [T, 27] - upper body joints
                y = arms[:max_frames]    # [T, 12] - arm endpoints
                status = "csv_trajectory"
                out.append((x, y, status, tid))
                print(f"  Loaded trajectory {tid}: {len(x)} frames")
            except ValueError as e:
                print(f"  Warning: Could not load trajectory {tid}: {e}")
                continue
        return out

    # Removed H5 loading code - only supporting CSV trajectories now

    def _extract_skeleton_frame(self, input_data, frame_idx):
        """Extract skeleton joints from 27D input data (9 upper body joints)"""
        joints = {}

        # CSV trajectory dataset - 27D upper body joints
        # Joint order: [Head, Neck, R_Shoulder, L_Shoulder, R_Elbow, L_Elbow, R_Hand, L_Hand, Torso]
        upper_body_names = ['Head', 'Neck', 'RShoulder', 'LShoulder', 'RUArm', 'LUArm', 'RHand', 'LHand', 'Chest']

        for i, name in enumerate(upper_body_names):
            if i < 9:  # Only use first 9 joints
                j = i * 3
                if j + 2 < input_data.shape[1]:
                    # Get coordinates and apply transformation if needed
                    x, y, z = input_data[frame_idx, j:j+3]
                    joints[name] = np.array([x, y, z])  # Keep original coordinate system

        return joints

    @staticmethod
    def _cap_step(prev12: np.ndarray, curr12: np.ndarray, step_cap_mm: float) -> np.ndarray:
        """
        Speed limit: per endpoint (4 points), cap Euclidean displacement to step_cap_mm (in mm).
        prev12, curr12: shape [12], ordered as [L1, L2, R1, R2] * 3
        """
        prev = prev12.reshape(4, 3)
        curr = curr12.reshape(4, 3)
        delta = curr - prev
        norms = np.linalg.norm(delta, axis=1, keepdims=True)  # [4,1]
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.minimum(1.0, step_cap_mm / np.maximum(norms, 1e-9))
        delta_capped = delta * scale
        capped = prev + delta_capped
        return capped.reshape(12)

    def _predict_current_frame(self, input_traj: np.ndarray, target_traj: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Real-time prediction for current frame.
        Returns prediction [12] or ground truth for warm-up frames.
        """
        need = self.predictor.seq_len  # 120

        if frame_idx >= need - 1:
            # Use past 120 frames for prediction
            past = input_traj[frame_idx - (need - 1): frame_idx + 1]  # [120,63]
            prediction = self.predictor.predict_one(past)  # [12]

            # Apply speed cap if needed
            if frame_idx > 0:
                prev = target_traj[frame_idx - 1] if self._last_prediction is None else self._last_prediction
                prediction = self._cap_step(prev, prediction, self.step_cap_mm)

            self._last_prediction = prediction
            return prediction
        else:
            # Warm-up: return ground truth
            return target_traj[frame_idx]

    def visualize(self, meters=False):
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.2)

        unit_scale = 0.001 if meters else 1.0
        unit_label = "m" if meters else "mm"

        # skeleton artists
        skeleton_pts = ax.scatter([], [], [], c='gray', s=30, alpha=0.6)
        skeleton_lines = [ax.plot([], [], [], 'gray', alpha=0.4)[0] for _ in SKELETON_BONES]

        # GT endpoints (4 points) & lines
        gt_p = [ax.scatter([], [], [], c='lightgreen', s=150, alpha=0.7, marker=m, edgecolors='green', linewidth=2)
                for m in ['o', 's', 'o', 's']]
        gt_L = ax.plot([], [], [], color='lightgreen', linewidth=4, alpha=0.6)[0]
        gt_R = ax.plot([], [], [], color='lightblue',  linewidth=4, alpha=0.6)[0]

        # Pred endpoints & lines (runtime-capped)
        pr_p = [ax.scatter([], [], [], c='darkgreen', s=80, alpha=0.9, marker=m)
                for m in ['o', 's', 'o', 's']]
        pr_L = ax.plot([], [], [], color='darkgreen', linewidth=2, alpha=0.9)[0]
        pr_R = ax.plot([], [], [], color='darkblue',  linewidth=2, alpha=0.9)[0]

        def update_visualization():
            x, y, status, orig_idx = self.trajectories[self.current_sample]
            t = self.current_frame

            # Real-time prediction for current frame
            prediction = self._predict_current_frame(x, y, t)
            gt = y  # Ground truth targets

            # No coordinate transformation needed for CSV trajectory data

            # skeleton
            sk = self._extract_skeleton_frame(x, t)
            if sk:
                pos = np.array([sk[n]*unit_scale for n in JOINT_NAMES if n in sk])
                if len(pos) > 0:
                    skeleton_pts._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
                for (child, parent), line in zip(SKELETON_BONES, skeleton_lines):
                    if child in sk and parent in sk:
                        p1, p2 = sk[parent]*unit_scale, sk[child]*unit_scale
                        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                        line.set_3d_properties([p1[2], p2[2]])

            # endpoints: reshape [12] -> [4,3] in order [L1, L2, R1, R2]
            # Keep original coordinate system for CSV trajectory data
            gt_pts = gt[t].reshape(4, 3)
            pr_pts = prediction.reshape(4, 3)

            gt_pts *= unit_scale
            pr_pts *= unit_scale

            # scatters
            for k in range(4):
                gt_p[k]._offsets3d = ([gt_pts[k, 0]], [gt_pts[k, 1]], [gt_pts[k, 2]])
                pr_p[k]._offsets3d = ([pr_pts[k, 0]], [pr_pts[k, 1]], [pr_pts[k, 2]])

            # lines (L1-L2, R1-R2)
            gt_L.set_data([gt_pts[0, 0], gt_pts[1, 0]], [gt_pts[0, 1], gt_pts[1, 1]])
            gt_L.set_3d_properties([gt_pts[0, 2], gt_pts[1, 2]])
            gt_R.set_data([gt_pts[2, 0], gt_pts[3, 0]], [gt_pts[2, 1], gt_pts[3, 1]])
            gt_R.set_3d_properties([gt_pts[2, 2], gt_pts[3, 2]])

            pr_L.set_data([pr_pts[0, 0], pr_pts[1, 0]], [pr_pts[0, 1], pr_pts[1, 1]])
            pr_L.set_3d_properties([pr_pts[0, 2], pr_pts[1, 2]])
            pr_R.set_data([pr_pts[2, 0], pr_pts[3, 0]], [pr_pts[2, 1], pr_pts[3, 1]])
            pr_R.set_3d_properties([pr_pts[2, 2], pr_pts[3, 2]])

            # axes range auto-center with fixed span
            allp = np.vstack([gt_pts, pr_pts])
            if sk:
                allp = np.vstack([allp, pos])
            c = allp.mean(axis=0)
            span = 1000.0  # +/-1m if unit=mm
            ax.set_xlim(c[0]-span, c[0]+span)
            ax.set_ylim(c[1]-span, c[1]+span)
            ax.set_zlim(c[2]-span, c[2]+span)

            # calculate arm lengths (L1-L2, R1-R2) for both GT and Predictions
            gt_left_length = np.linalg.norm(gt_pts[1] - gt_pts[0])   # L2 - L1
            gt_right_length = np.linalg.norm(gt_pts[3] - gt_pts[2])  # R2 - R1
            pr_left_length = np.linalg.norm(pr_pts[1] - pr_pts[0])   # L2 - L1
            pr_right_length = np.linalg.norm(pr_pts[3] - pr_pts[2])  # R2 - R1

            # convert arm lengths to centimeters for display
            if meters:
                gt_left_cm = gt_left_length * 100.0
                gt_right_cm = gt_right_length * 100.0
                pr_left_cm = pr_left_length * 100.0
                pr_right_cm = pr_right_length * 100.0
            else:
                gt_left_cm = gt_left_length / 10.0   # mm to cm
                gt_right_cm = gt_right_length / 10.0
                pr_left_cm = pr_left_length / 10.0
                pr_right_cm = pr_right_length / 10.0

            # error metrics (vs GT) using capped preds
            dist = np.linalg.norm(gt_pts - pr_pts, axis=-1).mean()  # mean over 4 points
            mse  = ((gt_pts - pr_pts)**2).sum(axis=-1).mean()       # mean pointwise squared error

            # friendly display
            if meters:
                avg_pos_error_mm = dist * 1000.0
                dist_display = f"{dist:.3f} {unit_label}"
            else:
                avg_pos_error_mm = dist
                dist_display = f"{dist:.2f} {unit_label}"

            mode = "PREDICTION" if t >= self.predictor.seq_len - 1 else "WARMUP (GT)"
            ax.set_title(
                f"Trajectory {orig_idx} ({status.upper()}) - Frame {t}/{len(x)-1} - {mode}\n"
                f"Endpoint Mean Dist: {dist_display} | Avg Pos Error: {avg_pos_error_mm:.2f}mm | MSE: {mse:.2f} {unit_label}²\n"
                f"GT Arm Lengths: L={gt_left_cm:.1f}cm, R={gt_right_cm:.1f}cm | PRED: L={pr_left_cm:.1f}cm, R={pr_right_cm:.1f}cm\n"
                f"Real-time Inference | FPS={self.fps} | Max speed={self.max_speed_cm} cm/s",
                fontsize=11
            )
            ax.set_xlabel(f"X ({unit_label})")
            ax.set_ylabel(f"Y ({unit_label})")
            ax.set_zlabel(f"Z ({unit_label})")

            # update dynamic arm length display
            left_error = abs(pr_left_cm - gt_left_cm)
            right_error = abs(pr_right_cm - gt_right_cm)

            # color coding based on accuracy (green if close to target 40cm, red if far)
            left_color = "green" if abs(pr_left_cm - 40.0) < 5.0 else "orange" if abs(pr_left_cm - 40.0) < 10.0 else "red"
            right_color = "green" if abs(pr_right_cm - 40.0) < 5.0 else "orange" if abs(pr_right_cm - 40.0) < 10.0 else "red"

            arm_length_text.set_text(
                f"ARM LENGTHS (Real-time)\n"
                f"GT Left: {gt_left_cm:5.1f} cm | GT Right: {gt_right_cm:5.1f} cm\n"
                f"PRED Left: {pr_left_cm:5.1f} cm | PRED Right: {pr_right_cm:5.1f} cm\n"
                f"Length Error: L={left_error:4.1f} cm, R={right_error:4.1f} cm\n"
                f"Target: 40.0 cm per arm\n"
                f"Status: L={left_color.upper()}, R={right_color.upper()}"
            )

            fig.canvas.draw_idle()

        def animate(_):
            if self.is_animating:
                x, _, _, _ = self.trajectories[self.current_sample]
                # step 1 frame
                self.current_frame = (self.current_frame + 1) % len(x)
                frame_slider.eventson = False
                frame_slider.set_val(self.current_frame)
                frame_slider.eventson = True
                update_visualization()
            return []

        # controls
        ax_frame = plt.axes([0.2, 0.1, 0.5, 0.03])
        x0, _, _, _ = self.trajectories[0]
        frame_slider = Slider(ax_frame, 'Frame', 0, len(x0)-1, valinit=0, valfmt='%d')

        def _on_slider(val):
            if not self.is_animating:
                self.current_frame = int(val)
                update_visualization()
        frame_slider.on_changed(_on_slider)

        ax_prev = plt.axes([0.05, 0.05, 0.08, 0.04])
        ax_next = plt.axes([0.15, 0.05, 0.08, 0.04])
        ax_anim = plt.axes([0.25, 0.05, 0.08, 0.04])
        btn_prev, btn_next, btn_anim = Button(ax_prev, 'Prev'), Button(ax_next, 'Next'), Button(ax_anim, 'Animate')

        def on_prev(_):
            self.current_sample = max(0, self.current_sample - 1)
            self.current_frame = 0
            self._last_prediction = None  # reset
            x, _, _, _ = self.trajectories[self.current_sample]
            frame_slider.valmax = len(x) - 1
            frame_slider.ax.set_xlim(0, frame_slider.valmax)
            frame_slider.set_val(0)
            update_visualization()

        def on_next(_):
            self.current_sample = min(len(self.trajectories)-1, self.current_sample + 1)
            self.current_frame = 0
            self._last_prediction = None  # reset
            x, _, _, _ = self.trajectories[self.current_sample]
            frame_slider.valmax = len(x) - 1
            frame_slider.ax.set_xlim(0, frame_slider.valmax)
            frame_slider.set_val(0)
            update_visualization()

        def on_animate(_):
            self.is_animating = not self.is_animating
            btn_anim.label.set_text('Stop' if self.is_animating else 'Animate')
            if self.is_animating:
                self.animation = FuncAnimation(fig, animate, interval=1000//self.fps,
                                               repeat=True, cache_frame_data=False)
            else:
                if hasattr(self, 'animation'):
                    self.animation.event_source.stop()

        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        btn_anim.on_clicked(on_animate)

        # static info card
        orig_indices = [t[3] for t in self.trajectories]
        traj_list = ', '.join(map(str, orig_indices))
        plt.figtext(0.75, 0.25,
                    f"CSV TRAJECTORY VALIDATION\nFPS: {self.fps}\nTrajectory: {traj_list}\n"
                    f"Data: 27D→12D (9 joints→4 endpoints)\n"
                    f"Light: Ground Truth\nDark: Model Predictions\n"
                    f"Lines: L1-L2 and R1-R2\nMode: Live prediction per frame",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        # dynamic arm length display card
        arm_length_text = plt.figtext(0.75, 0.05,
                    "ARM LENGTHS (Real-time)\n"
                    "GT Left: -- cm | GT Right: -- cm\n"
                    "PRED Left: -- cm | PRED Right: -- cm\n"
                    "Length Error: L=-- cm, R=-- cm\n"
                    "Target: 40.0 cm per arm",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
                    fontsize=10, fontfamily='monospace')

        update_visualization()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time validation viz for endpoint(12D) model"
    )
    parser.add_argument("--model", default="./checkpoints_csv_trajectory/best_model.pth", help="Model path")
    parser.add_argument("--config", default="configs/csv_trajectory.json", help="Config file path")
    parser.add_argument("--fps", type=int, default=10,
                        help="Animation & inference FPS (per-step cap = max_speed*10/fps mm)")
    parser.add_argument("--max-speed", type=float, default=40.0,
                        help="Maximum endpoint speed in cm/s (default=40). Only affects runtime capping.")
    parser.add_argument("--meters", action="store_true", help="Display in meters (scale 0.001)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}"); return
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}"); return

    vis = SimpleVisualizer(args.model, args.config, args.fps, args.max_speed)
    vis.visualize(args.meters)


if __name__ == "__main__":
    main()
