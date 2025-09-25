# Robot Motion Planning with Human Motion Capture

This repository contains a deep learning-based robot motion planning system that predicts robot end-effector trajectories from human motion capture data.

## Overview

The system uses a neural network to predict 4 robot arm endpoints (L1, L2, R1, R2) from 9 upper body joint positions captured via motion capture. It supports both offline training and real-time ROS2-based inference.

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

#### `robot_planner.py`
ROS2-based real-time robot trajectory planning:
- Subscribes to motion capture data
- Uses trained model for real-time inference
- Publishes robot end-effector trajectories
- Supports configurable prediction horizons

```bash
ros2 run robot_planner robot_planner
```

#### `robot_planner_smoothed.py`
Enhanced version with trajectory smoothing:
- Includes temporal smoothing filters
- Reduces jitter in predicted trajectories
- Better suited for real robot deployment

```bash
ros2 run robot_planner robot_planner_smoothed
```

## Configuration

Training and inference parameters are configured via JSON files in the `configs/` directory:

- `csv_trajectory.json`: Main configuration for CSV-based training
- Includes model architecture, training hyperparameters, and data paths

## Data Format

### Input Data
- **Motion Capture**: 9 upper body joints × 3 coordinates = 27 dimensions
- **Trajectory Format**: CSV files with `traj_id`, joint positions, and timestamps

### Output Data
- **Robot Endpoints**: 4 arm endpoints × 3 coordinates = 12 dimensions
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

1. **Data Preparation**: Prepare motion capture data in CSV format
2. **Training**: Use `train.py` to train the neural network model
3. **Validation**: Evaluate model performance with `val.py`
4. **Motion Capture Publishing**: Use `mocap_unified_publisher.py` for data streaming
5. **Real-time Inference**: Deploy with `robot_planner.py` or `robot_planner_smoothed.py`

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