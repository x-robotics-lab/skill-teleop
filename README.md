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

## Questions?
If you have any questions, please feel free to contact [Yen-Jen Wang](https://wangyenjen.github.io/) and [Haozhi Qi](https://haozhi.io/).
