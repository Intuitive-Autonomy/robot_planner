# Robot Motion Planning with Human Motion Capture

This repository contains a deep learning-based robot motion planning system that predicts robot end-effector trajectories from human motion capture data.

## Overview

The system uses a neural network to predict 2 robot end-effector trajectories (4 endpoints total) from 9 upper body joint positions captured via motion capture. It supports both offline training and real-time ROS2-based inference.

## Core Components

### Training Pipeline

#### `train.py`
Main training script with two-phase training:
- **Phase 1**: Frame-level training with teacher forcing
- **Phase 2**: Full trajectory training with jitter and speed penalties

```bash
python train.py --config configs/csv_trajectory.json
```

Key features:
- Autoregressive prediction with teacher forcing decay
- Custom loss functions for endpoint accuracy and smoothness
- Support for data augmentation (rotation, scaling, time scaling)

#### `net.py`
Neural network architecture definitions:
- Transformer-based encoder-decoder model
- Support for variable sequence lengths
- Configurable hidden dimensions, layers, and attention heads

#### `utils.py`
Data processing utilities:
- `CSVTrajectoryProcessor`: Handles CSV trajectory data loading
- `create_csv_data_loaders`: Creates training/validation data loaders
- `TrajectoryTrainer`: Training utilities and data augmentation

### Validation and Analysis

#### `val.py`
Model validation and evaluation:
- Load trained models and evaluate on test data
- Generate prediction visualizations
- Calculate performance metrics

```bash
python val.py --config configs/csv_trajectory.json --model checkpoints_csv_trajectory/best_model.pth
```

### Motion Capture Data Publishing

#### `mocap_unified_publisher.py`
ROS2 node for publishing motion capture data:
- Publishes human body point clouds
- Publishes motion capture poses
- Supports real-time streaming of mocap data

```bash
python mocap_unified_publisher.py <csv_file> <start_frame> <duration> <data_path>
```

Example:
```bash
python mocap_unified_publisher.py inference_trajectory_gt.csv 0 10.0 /path/to/mocap/data
```

### Real-time Robot Planning

#### `robot_planner_new.py`
State machine-based robot planner with four operational modes:
- **rest**: Robot maintains 1.5m distance from human, allows manual positioning
- **approach**: Triggered by 15cm height decrease, moves to 0.4m at 1m height
- **assist**: Activates when robot is <0.5m from human, follows shoulder-based targets
- **retreat**: Triggered by 10cm stable height increase, returns to 2.1m distance

Key features:
- Dynamic standing height tracking for robust state transitions
- Publishes `/target_eef_poses` (PoseArray) for target end-effector positions
- Publishes `/mode` (String) for current operational state
- Arm spacing: 45cm for rest/approach/retreat, shoulder-based for assist

```bash
python robot_planner_new.py
```

#### `robot_controller.py`
Velocity-constrained robot controller with SLERP-based orientation interpolation:
- Subscribes to `/target_eef_poses` and applies motion constraints
- Maximum linear velocity: 0.3 m/s (0.03m per step at 10Hz)
- Maximum angular velocity: 10 deg/s (3° per step at 10Hz)
- Uses quaternion SLERP for smooth orientation changes
- Publishes `/target_poses` with 180° yaw rotation applied
- Publishes `/current_poses` for robot state feedback

```bash
python robot_controller.py
```

#### `debug_visualizer.py`
Real-time 3D visualization tool:
- Subscribes to `/pose_detection` for human skeleton
- Subscribes to `/target_poses` for robot targets
- Visualizes human joints, skeleton connections, and joint labels
- Displays robot arm targets with directional arrows
- Shows target poses with RPY orientation axes (R=red, P=green, Y=blue)
- Fixed coordinate ranges: X/Y ∈ [-1, 1]m, Z ∈ [0, 2]m
- Equal aspect ratio for accurate spatial representation

```bash
python debug_visualizer.py
```

#### Legacy Planners

**`robot_planner.py`** - Original neural network-based planner:
- Uses trained model for real-time inference
- Supports configurable prediction horizons

**`robot_planner_smoothed.py`** - Enhanced version with trajectory smoothing:
- Includes temporal smoothing filters
- Reduces jitter in predicted trajectories

## Configuration

Training and inference parameters are configured via JSON files in the `configs/` directory:

- `csv_trajectory.json`: Main configuration for CSV-based training
- Includes model architecture, training hyperparameters, and data paths

## Data Format

### Input Data
- **Motion Capture**: 9 upper body joints × 3 coordinates = 27 dimensions
- **Trajectory Format**: CSV files with `traj_id`, joint positions, and timestamps

### Output Data
- **Robot End-effectors**: 4 endpoints (2 end-effectors × 2 points each) × 3 coordinates = 12 dimensions
- **Trajectory Prediction**: Variable length sequences (typically 5-10 frames)

## Installation

### Dependencies
```bash
pip install torch numpy pandas tqdm
pip install ros2 # For ROS2 components
```

### ROS2 Setup
Ensure ROS2 is properly installed and sourced:
```bash
source /opt/ros/humble/setup.bash  # or your ROS2 distribution
```

## Usage Workflow

### Training Pipeline
1. **Data Preparation**: Prepare motion capture data in CSV format
2. **Training**: Use `train.py` to train the neural network model
3. **Validation**: Evaluate model performance with `val.py`

### Real-time Deployment Pipeline

1. **Start Motion Capture Publisher**
   ```bash
   python mocap_unified_publisher.py <csv_file> <start_frame> <duration> <data_path>
   ```
   This publishes point cloud data to ROS2 topics.

2. **Run Pose Detector** (from spike repository)
   ```bash
   cd spike
   python pose_detector.py
   ```
   This subscribes to point cloud data and publishes detected human poses to `/pose_detection`.

3. **Start Robot Planner**
   ```bash
   python robot_planner_new.py
   ```
   This node:
   - Subscribes to `/pose_detection` for human pose
   - Implements state machine (rest, approach, assist, retreat modes)
   - Publishes target end-effector poses to `/target_eef_poses`
   - Publishes current mode to `/mode`

4. **Start Robot Controller**
   ```bash
   python robot_controller.py
   ```
   This node:
   - Subscribes to `/target_eef_poses`
   - Applies velocity constraints (0.3 m/s linear, 10 deg/s angular)
   - Publishes smoothed target poses to `/target_poses` (with 180° yaw rotation)
   - Publishes current robot poses to `/current_poses`

5. **(Optional) Start Debug Visualizer**
   ```bash
   python debug_visualizer.py
   ```
   This provides real-time 3D visualization of:
   - Human skeleton
   - Robot arm targets
   - Target poses with RPY axes

## Model Architecture

The system uses a Transformer-based encoder-decoder architecture:
- **Encoder**: Processes input motion capture sequences
- **Decoder**: Generates robot endpoint trajectories
- **Loss Functions**: Custom losses for position accuracy, smoothness, and physical constraints

## Performance Features

- **Real-time Processing**: Optimized for low-latency inference
- **Trajectory Smoothing**: Reduces noise and jitter in predictions
- **Physical Constraints**: Enforces realistic arm lengths and motion limits
- **Augmentation**: Robust training through data augmentation techniques

## File Structure

```
├── train.py                    # Main training script
├── net.py                      # Neural network models
├── utils.py                    # Data processing utilities
├── val.py                      # Model validation
├── mocap_unified_publisher.py  # Motion capture data publisher
├── robot_planner.py           # Real-time robot planner
├── robot_planner_smoothed.py  # Smoothed robot planner
├── configs/                   # Configuration files
│   └── csv_trajectory.json    # Training configuration
└── checkpoints_csv_trajectory/ # Saved models
```

## License

This project is part of a robot motion planning research system.