# Skill-based Teleoperation
This repository contains a clean codebase for controlling bimanual UR5e robots for dex manipulation tasks. It supports:
- **Quest Teleoperation** - Using Meta Quest 2 controllers for real-time control
- **BC Policy Deployment** - Running trained behavior cloning policies

![Demo](./images/teleop.gif)

## Installation
Clone this repository, then clone the required sub-repositories into it.
```bash
git clone https://github.com/x-robotics-lab/skill-teleop.git
cd skill-teleop
git clone https://github.com/x-robotics-lab/minbc.git
```

Create conda environment and install requirements.
```bash
conda create -n screw_driver python=3.10
conda activate screw_driver
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

For teleoperation, install [oculus_reader](https://github.com/rail-berkeley/oculus_reader):
```bash
pip install oculus_reader
```

## Configuration

All hardware-specific settings are centralized in `config.py`. This includes:

- **Network Configuration** - IP addresses and ports
- **Robot Configuration** - Joint limits, IPs, control parameters
- **Camera Configuration** - Serial numbers and image settings
- **Reset Positions** - Default joint positions for different tasks
- **XHand Configuration** - Gripper settings and policy paths
- **Data Collection** - Save directories and formats

### Modifying Configuration

Edit `config.py` to change hardware settings:

```python
# Example: Change robot IP addresses
ROBOT_CONFIG.left_robot_ip = "192.168.1.10"
ROBOT_CONFIG.right_robot_ip = "192.168.2.10"

# Example: Change camera
CAMERA_CONFIG.wrist_camera_id = "your_camera_serial_number"

# Example: Change data directory
DATA_CONFIG.default_data_dir = "/your/custom/path"
```

Edit `config.py` to change checkpoints:
```python
# Example: Change RL policy from dexscrew.
self.screw_task_policy_path: str = ("/your/rl/policy/path")

# Example: Change BC checkpoint from minbc.
default_checkpoint_path: str = ("/your/bc/policy/path")
```

## Usage

The system uses a client-server architecture via ZMQ for communication between hardware components.

### 1. Launch Hardware Nodes (Terminal 1)

Start the camera and robot servers:

```bash
python launch_nodes.py --robot_type bimanual_ur --use_faster_camera
```

**Arguments:**
- `--robot_type`: Robot type (default: `bimanual_ur`)
- `--use_faster_camera`: Use buffered camera mode for better performance (default: `True`)
- `--hostname`: Server hostname (default: from `config.py`)
- `--image_size`: Image size tuple (default: from `config.py`)

**Configuration:** Robot IPs are set in `config.py`:
- Left robot: `ROBOT_CONFIG.left_robot_ip` (default: `192.168.1.3`)
- Right robot: `ROBOT_CONFIG.right_robot_ip` (default: `192.168.2.3`)

### 2. Run Control (Terminal 2)

#### Quest Teleoperation with Data Collection
```bash
python run_env.py --agent quest --hz 20 --save_data
```

#### BC Policy Training and Deployment
**For BC training, please refer to [minbc](https://github.com/x-robotics-lab/minbc). To convert your collected data into the BC dataset format, use the script `./scripts/convert_to_bc_dataset.py`. Make sure to update the `SRC_DIR` and `DST_DIR` variables inside the script to match your data locations.**

```bash
python run_env.py --agent bc --hz 20 --bc_checkpoint_path /path/to/model.ckpt
```

**Common Arguments:**
- `--agent`: Agent type (`quest` or `bc`)
- `--hz`: Control frequency in Hz (default: 20)
- `--robot_port`: Robot ZMQ port (default: from `config.py`)
- `--base_camera_port`: Camera ZMQ port (default: from `config.py`)
- `--hostname`: Server hostname (default: from `config.py`)

**Data Collection Arguments:**
- `--save_data`: Enable data saving
- `--save_depth`: Save depth images (default: `True`)
- `--save_png`: Save RGB images as PNG (default: `False`)
- `--data_dir`: Data save directory (default: from `config.py`)

**BC Agent Arguments:**
- `--bc_checkpoint_path`: Path to BC model checkpoint (default: from `config.py`)
- `--bc_use_async`: Use asynchronous policy execution (default: `False`)
- `--bc_num_diffusion_iters`: Number of diffusion iterations (default: 5)

### 3. Keyboard Controls

During execution, the following keyboard controls are available:

| Key | Action |
|-----|--------|
| `L` | Stop execution and exit |
| `R` | Start/trigger data saving (when `--save_data` is enabled) |
| `C` | Mark switch event in saved data |

**Quest Controller Buttons:**
- **Left Trigger**: Activate/deactivate arm control
- **Left Joystick**: Control XHand gripper (BC agent)
- **Left Grip**: Control gripper (Quest agent)
- **Right Joystick**: Fine-tune vertical position
- **Button A/B**: Move up/down
- **Button X**: Start XHand policy (Quest agent)
- **Button Y**: Stop XHand policy (Quest agent)
- **LJ Button**: Grasp (Quest agent)
- **LG Button**: Release (Quest agent)

### 4. Cleanup

To stop all hardware nodes:
```bash
pkill -9 -f launch_nodes.py
```

## Data Format

When `--save_data` is enabled, each frame is saved as a pickle file containing:

```python
{
    'base_rgb': np.ndarray,           # RGB image(s) from camera(s)
    'base_depth': np.ndarray,         # Depth image(s) (if enabled)
    'joint_positions': np.ndarray,    # Joint angles for both arms (12D)
    'joint_velocities': np.ndarray,   # Joint velocities
    'eef_speed': np.ndarray,          # End-effector speeds
    'ee_pos_quat': np.ndarray,        # End-effector poses
    'tcp_force': np.ndarray,          # TCP force/torque readings
    'xhand_pos': np.ndarray,          # XHand joint positions (12D)
    'xhand_act': np.ndarray,          # XHand joint commands (12D)
    'xhand_tactile': np.ndarray,      # XHand tactile sensor data
    'control': np.ndarray,            # Action taken at this step
    'activated': bool,                # Whether control was active
    'xhand_rl_flag': bool,           # Whether XHand policy was running
    'switch': bool,                   # Switch event flag
}
```

A `freq.txt` file is also saved with timing statistics:
```
Average FPS: 19.85
Max FPS: 20.12
Min FPS: 19.23
Std FPS: 0.15
```

## Customization Guide

### Adding a New Robot

1. Create a new robot class in `robots/`:
```python
from robots.robot import Robot

class MyRobot(Robot):
    def num_dofs(self) -> int:
        return 6
    
    # Implement other required methods...
```

2. Add configuration in `config.py`:
```python
@dataclass
class MyRobotConfig:
    robot_ip: str = "192.168.1.100"
```

3. Update `launch_nodes.py` to support new robot type.

### Adding a New Agent

1. Create a new agent in `agents/`:
```python
from agents.agent import Agent

class MyAgent(Agent):
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        # Your control logic here
        return action
```

2. Update `run_env.py` to add agent creation function.

### Modifying Reset Positions

Edit `config.py`:
```python
@dataclass
class ResetJointPositions:
    left_arm_deg: np.ndarray = np.array([...])  # Your positions
    right_arm_deg: np.ndarray = np.array([...])
```

## Performance Tips

1. **Use Faster Camera Mode**: `--use_faster_camera` buffers frames for consistent rate
2. **Adjust Control Frequency**: Lower `--hz` for more reliable execution
3. **Disable Camera View**: Run without `--show_camera_view` for better performance
4. **Async BC Agent**: Use `--bc_use_async` for lower latency with BC policies

## Troubleshooting

### Camera Not Found
- Check camera connection: `rs-enumerate-devices`
- Verify serial number in `config.py`: `CAMERA_CONFIG.wrist_camera_id`
- Ensure RealSense SDK is installed

### Robot Connection Failed
- Verify robot IPs in `config.py`
- Check network connectivity: `ping 192.168.1.3`
- Ensure robots are powered on and in remote control mode
- Check firewall settings

### ZMQ Connection Timeout
- Ensure `launch_nodes.py` is running before `run_env.py`
- Check that ports in `config.py` are not in use: `lsof -i :5000`
- Verify hostname settings

### XHand Initialization Failed
- Check EtherCAT connection
- Verify policy path in `config.py`: `XHAND_CONFIG.screw_task_policy_path`
- Ensure `xhand_controller` package is installed

### Low Control Frequency
- Reduce `--hz` value
- Enable `--use_faster_camera`
- Disable `--show_camera_view`
- Check system load: `htop`


## Acknowledgements

This codebase is based on [HATO: Learning Visuotactile Skills with Two Multifingered Hands](https://toruowo.github.io/hato/).

Key dependencies:
- [oculus_reader](https://github.com/rail-berkeley/oculus_reader) - Quest controller interface
- [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/) - UR robot control
- [pyrealsense2](https://github.com/IntelRealSense/librealsense) - RealSense camera
- [xhand_controller](https://ai.feishu.cn/drive/folder/WGyhflqb1lRu9ddtc0scjDhwngg) - Robot Era's Xhand document


## Citations
# MinBC - Minimal Behavior Cloning

A simple and efficient implementation for robot behavior cloning with support for both Vanilla BC and Diffusion Policy.

## Features

- 🚀 **Two Policy Types**: Vanilla BC and Diffusion Policy
- 🖼️ **Flexible Vision Encoders**: Support for DINOv3, DINO, CLIP, or train from scratch
- 🎯 **Multi-Modal Input**: RGB images, joint positions, velocities, tactile sensors, etc.
- ⚡ **Multi-GPU Training**: Efficient distributed training support
- 📊 **TensorBoard Logging**: Real-time training monitoring

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- diffusers >= 0.21.0
- tyro >= 0.5.0
- tensorboard >= 2.13.0

## Quick Start

### Option 1: Training Without Images (Fastest)

If your task doesn't require vision, use only proprioceptive data:

```bash
# Single GPU
python train.py train \
  --gpu 0 \
  --data.data-key joint_positions joint_velocities eef_speed xhand_pos \
  --optim.batch-size 128 \
  --optim.num-epoch 300
```

**Benefits**: No need to configure vision encoders, faster training, lower GPU memory.

### Option 2: Training With Images
Please refer to [dinov3](https://github.com/facebookresearch/dinov3) for the model and available checkpoints.
```bash
# Single GPU with DINOv3
python train.py train \
  --gpu 0 \
  --data.im-encoder DINOv3 \
  --data.dinov3-model-dir /path/to/dinov3 \
  --data.dinov3-weights-path /path/to/dinov3/dinov3.ckpt \
  --optim.batch-size 64 \
  --optim.num-epoch 300
```

```bash
# Or use DINO (auto-downloads from PyTorch Hub)
python train.py train \
  --gpu 0 \
  --data.im-encoder DINO \
  --optim.batch-size 64
```

### Option 3: Multi-GPU Training

```bash
# Edit train.sh to set your GPU IDs
vim train.sh

# Run multi-GPU training
bash train.sh
```

Or directly:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train.py train \
  --gpu 0,1 \
  --multi-gpu \
  --optim.batch-size 256 \
  --optim.num-epoch 300
```

## Configuration

### View All Available Parameters

```bash
python train.py train --help
```

### Configuration Method

MinBC uses **command-line arguments** to configure training. There is only one configuration file: `configs/base.py`, which defines default values.

**Priority**: Command-line arguments > Default values in `configs/base.py`

### Common Configuration Options

#### Basic Training Settings
```bash
--gpu STR                    # GPU IDs (e.g., "0" or "0,1,2,3")
--multi-gpu                  # Enable multi-GPU training
--seed INT                   # Random seed (default: 0)
--optim.batch-size INT       # Batch size (default: 128)
--optim.num-epoch INT        # Number of epochs (default: 30)
--optim.learning-rate FLOAT  # Learning rate (default: 0.0002)
--output_name STR            # Experiment name
```

#### Data Configuration
```bash
--data.data-key [KEYS...]    # Data modalities to use
                             # Options: img, joint_positions, joint_velocities,
                             #          eef_speed, ee_pos_quat, xhand_pos, xhand_tactile

--data.im-encoder STR        # Vision encoder (only if using 'img')
                             # Options: DINOv3, DINO, CLIP, scratch

--data.dinov3-model-dir STR       # DINOv3 model directory (if using DINOv3)
--data.dinov3-weights-path STR    # DINOv3 weights path (if using DINOv3)
```

#### Policy Type
```bash
--policy-type STR            # Policy type: "bc" (Vanilla BC) or "dp" (Diffusion Policy)
```

#### Diffusion Policy Settings (if using policy-type=dp)
```bash
--dp.diffusion-iters INT     # Number of diffusion iterations (default: 100)
--dp.obs-horizon INT         # Observation horizon (default: 1)
--dp.act-horizon INT         # Action horizon (default: 8)
--dp.pre-horizon INT         # Prediction horizon (default: 16)
```

### How to Modify Configuration

#### Method 1: Command-Line Arguments (Recommended)

Override any parameter directly in the command:

```bash
python train.py train \
  --gpu 2 \
  --optim.batch-size 64 \
  --optim.learning-rate 0.0005 \
  --data.dinov3-model-dir /your/custom/path
```

#### Method 2: Edit Default Values

Modify `configs/base.py` to change default values:

```python
# configs/base.py

@dataclass(frozen=True)
class MinBCConfig:
    seed: int = 0
    gpu: str = '0'              # Change default GPU
    data_dir: str = 'data/'     # Change default data path
    ...

@dataclass(frozen=True)
class DataConfig:
    dinov3_model_dir: str = '/your/path/to/dinov3'  # Change default DINOv3 path
    ...
```

#### Method 3: Use Training Scripts

Create or modify training scripts like `train.sh`:

```bash
#!/bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)

python train.py train \
  --gpu 0 \
  --optim.batch-size 128 \
  --optim.num-epoch 300 \
  --data.dinov3-model-dir /your/path \
  --output_name "exp-${timestamp}"
```

## Data Format

### Directory Structure

```
data/
└── your_dataset/
    ├── train/
    │   ├── episode_000/
    │   │   ├── step_000.pkl
    │   │   ├── step_001.pkl
    │   │   └── ...
    │   ├── episode_001/
    │   └── ...
    └── test/
        ├── episode_000/
        └── ...
```

### Data Requirements

Each `.pkl` file should contain a dictionary with the following keys:

#### Required Keys
- `action`: numpy array of shape `(action_dim,)` - Robot action at this timestep

#### Optional Keys (depending on `data.data-key` configuration)

**Proprioceptive Data:**
- `joint_positions`: numpy array of shape `(12,)` - Joint positions
- `joint_velocities`: numpy array of shape `(12,)` - Joint velocities
- `eef_speed`: numpy array of shape `(12,)` - End-effector speed
- `ee_pos_quat`: numpy array of shape `(12,)` - End-effector pose (position + quaternion)
- `xhand_pos`: numpy array of shape `(12,)` - Hand position
- `xhand_tactile`: numpy array of shape `(1800,)` - Tactile sensor data

**Visual Data (if using images):**
- `base_rgb`: numpy array of shape `(H, W, 3)` - RGB image (default: 240x320x3)
  - Values should be in range [0, 255], dtype: uint8 or uint16

### Data Format Example

```python
# Example pickle file content
import pickle
import numpy as np

data = {
    'action': np.array([...]),           # Shape: (24,)
    'joint_positions': np.array([...]),  # Shape: (12,)
    'joint_velocities': np.array([...]), # Shape: (12,)
    'base_rgb': np.array([...]),         # Shape: (240, 320, 3), uint8
}

with open('step_000.pkl', 'wb') as f:
    pickle.dump(data, f)
```

### Data Configuration

Specify which data modalities to use:

```bash
# With images
python train.py train \
  --data.data-key img joint_positions xhand_pos

# Without images (only proprioceptive)
python train.py train \
  --data.data-key joint_positions joint_velocities eef_speed
```

### Data Paths

Set data paths in command line:

```bash
python train.py train \
  --data-dir /path/to/your/data \
  --train-data your_dataset/train \
  --test-data your_dataset/test
```

Or modify defaults in `configs/base.py`:

```python
@dataclass(frozen=True)
class MinBCConfig:
    data_dir: str = '/path/to/your/data'
    train_data: str = 'your_dataset/train/'
    test_data: str = 'your_dataset/test/'
```

## Training Examples

### Example 1: Minimal Setup (No Images)

```bash
python train.py train \
  --gpu 0 \
  --data.data-key joint_positions \
  --optim.batch-size 128 \
  --optim.num-epoch 100
```

### Example 2: Multi-Modal (No Images)

```bash
python train.py train \
  --gpu 0 \
  --data.data-key joint_positions joint_velocities eef_speed xhand_pos \
  --optim.batch-size 128 \
  --optim.num-epoch 300
```

### Example 3: With Vision (DINOv3)

```bash
python train.py train \
  --gpu 0 \
  --data.data-key img joint_positions xhand_pos \
  --data.im-encoder DINOv3 \
  --data.dinov3-model-dir /path/to/dinov3 \
  --data.dinov3-weights-path /path/to/dinov3/dinov3.ckpt \
  --optim.batch-size 64 \
  --optim.num-epoch 300
```

### Example 4: Diffusion Policy

```bash
python train.py train \
  --gpu 0 \
  --policy-type dp \
  --data.data-key joint_positions joint_velocities \
  --dp.diffusion-iters 100 \
  --optim.batch-size 64 \
  --optim.num-epoch 300
```

### Example 5: Multi-GPU Training

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  train.py train \
  --gpu 0,1,2,3 \
  --multi-gpu \
  --data.data-key img joint_positions xhand_pos \
  --data.im-encoder DINO \
  --optim.batch-size 256 \
  --optim.num-epoch 300
```

## Training Output

Training results are saved to `outputs/<output_name>/`:

```
outputs/bc-20251125_143022/
├── config.json              # Training configuration
├── model_last.ckpt          # Latest model checkpoint
├── model_best.ckpt          # Best model (lowest test loss)
├── stats.pkl                # Data statistics for normalization
├── norm.pkl                 # Normalization parameters
├── diff_*.patch             # Git diff at training time
└── events.out.tfevents.*    # TensorBoard logs
```

### Monitor Training

```bash
tensorboard --logdir outputs/
# Open browser to http://localhost:6006
```

## Troubleshooting

### Issue: DINOv3 Not Found

**Solution**: Either set the correct path or use a different encoder:

```bash
# Set correct path
python train.py train --data.dinov3-model-dir /correct/path

# Or use DINO (auto-downloads)
python train.py train --data.im-encoder DINO

# Or train without images
python train.py train --data.data-key joint_positions joint_velocities
```

### Issue: Out of GPU Memory

**Solutions**:
1. Reduce batch size: `--optim.batch-size 32`
2. Reduce prediction horizon: `--dp.pre-horizon 8`
3. Use fewer workers (modify `num_workers` in `dp/agent.py`)
4. Train without images if not needed

### Issue: Multi-GPU Training Hangs

**Solutions**:
1. Set `OMP_NUM_THREADS=1` before torchrun
2. Use `torchrun` instead of direct python execution
3. Check NCCL configuration

## Tips and Best Practices

1. **Start Simple**: Try training without images first to validate your pipeline
2. **Data Modalities**: Only include necessary data modalities for faster training
3. **Batch Size**: Adjust based on your GPU memory (64-128 for single GPU, 128-256 for multi-GPU)
4. **Vision Encoder**: Use DINO for ease (auto-downloads), DINOv3 for best performance (requires setup)
5. **Policy Type**: Use Vanilla BC for faster training, Diffusion Policy for better performance
6. **Monitoring**: Always check TensorBoard logs to ensure training is progressing

## Acknowledgements

MinBC is modified from [HATO](https://github.com/toruowo/hato) DP part, which is a simplification of the original Diffusion Policy.

## Citations
```
@article{hsieh2025learning,
  title={Learning Dexterous Manipulation Skills from Imperfect Simulations},
  author={Hsieh, Elvis and Hsieh, Wen-Han and Wang, Yen-Jen and Lin, Toru and Malik, Jitendra and Sreenath, Koushil and Qi, Haozhi},
  journal={arXiv:2512.02011},
  year={2025}
}
```

## Questions?
If you have any questions, please feel free to contact [Yen-Jen Wang](https://wangyenjen.github.io/) and [Haozhi Qi](https://haozhi.io/).
