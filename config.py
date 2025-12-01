"""Configuration file for screw driver hardware control system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


# =============================================================================
# Network Configuration
# =============================================================================

@dataclass
class NetworkConfig:
    """Network settings for ZMQ communication."""
    hostname: str = "127.0.0.1"
    robot_port: int = 6000
    camera_port: int = 5000


# =============================================================================
# Robot Configuration
# =============================================================================

@dataclass
class RobotConfig:
    """Robot hardware configuration."""
    # Robot IPs
    left_robot_ip: str = "192.168.1.3"
    right_robot_ip: str = "192.168.2.3"
    
    # Control parameters
    num_ur_joints: int = 6
    num_xhand_joints: int = 12
    max_joint_delta_per_step: float = 0.01
    max_interpolation_steps: int = 20


# =============================================================================
# Camera Configuration
# =============================================================================

@dataclass
class CameraConfig:
    """Camera hardware configuration."""
    # Enable/disable camera
    use_camera: bool = False
    
    # Camera serial numbers
    third_view_camera_id: str = "215122257029"  # Camera 455
    wrist_camera_id: str = "123622270261"       # Camera 405
    
    # Image settings
    default_image_size: Tuple[int, int] = (320, 240)
    
    def get_active_camera_ids(self) -> Dict[str, str]:
        """Get dictionary of active camera IDs."""
        return {
            # "third_view": self.third_view_camera_id,
            # "wrist": self.wrist_camera_id,
            # Add more cameras if needed
        }


# =============================================================================
# Reset Joint Positions (in degrees)
# =============================================================================

class ResetJointPositions:
    """Default reset positions for robot arms."""
    
    def __init__(self):
        """Initialize reset positions."""
        # Screw driver task positions (in degrees)
        self.left_arm_deg = np.array([-70.92, -103.75, -99.72, -92.57, 35.31, 32.23])
        self.right_arm_deg = np.array([-270, -30, 70, -85, 10, 0])
    
    def get_left_arm_rad(self) -> np.ndarray:
        """Get left arm reset position in radians."""
        return np.deg2rad(self.left_arm_deg)
    
    def get_right_arm_rad(self) -> np.ndarray:
        """Get right arm reset position in radians."""
        return np.deg2rad(self.right_arm_deg)
    
    def get_bimanual_reset_rad(self) -> np.ndarray:
        """Get combined bimanual reset position in radians."""
        return np.concatenate([self.get_left_arm_rad(), self.get_right_arm_rad()])


# =============================================================================
# Control Configuration
# =============================================================================

@dataclass
class ControlConfig:
    """Control loop configuration."""
    default_control_hz: int = 20
    default_show_camera: bool = False
    
    # Action smoothing for BC agent
    action_queue_size: int = 20
    action_smoothing_weight: float = 0.7


# =============================================================================
# Data Collection Configuration
# =============================================================================

@dataclass
class DataCollectionConfig:
    """Data collection settings."""
    default_data_dir: str = "/shared/data/screw_driver"
    save_depth_by_default: bool = True
    save_png_by_default: bool = False
    timestamp_format: str = "%m%d_%H%M%S"


# =============================================================================
# BC Agent Configuration
# =============================================================================

@dataclass
class BCAgentConfig:
    """Behavior cloning agent configuration."""
    default_checkpoint_path: str = (
        "/home/wangyenjen/repo/skill-teleop/minbc/outputs/"
        "bc-screw_driver_1104-hist-tactile-20251105_183647/model_last.ckpt"
    )
    num_diffusion_iterations: int = 5
    use_async_by_default: bool = False


# =============================================================================
# XHand Configuration
# =============================================================================

class XHandConfig:
    """XHand dexterous gripper configuration."""
    
    def __init__(self):
        """Initialize XHand configuration."""
        # Hardware settings
        self.hand_id: int = 0
        self.default_position: float = 0.1
        self.control_mode: int = 3
        
        # Control gains
        self.kp: int = 100
        self.ki: int = 0
        self.kd: int = 1
        self.torque_max: int = 100
        
        # Policy settings
        self.screw_task_policy_path: str = (
            "/home/wangyenjen/repo/dexscrew/xhand-deploy/checkpoints/screwdriver.pt"
        )
        self.buffer_init_path: str = "agents/screw_buffer_init.npy"
        
        # Policy dimensions
        self.observation_dim: int = 96
        self.proprioception_history_length: int = 30
        self.proprioception_dim: int = 24
        self.action_scale: float = 0.04167
        
        # Joint mapping (XHand to policy order)
        self.xhand_to_policy_indices = [3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2]
        
        # Joint limits
        self.dof_lower_limits = np.array([
            -0.1750, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, -1.0500, -0.1700
        ], dtype=np.double)
        
        self.dof_upper_limits = np.array([
            0.1750, 1.9200, 1.9200, 1.9200, 1.9200, 1.9200,
            1.9200, 1.9200, 1.9200, 1.8300, 1.5700, 1.8300
        ], dtype=np.double)
        
        # Grasp position in XHand hardware order for screw task
        # (will be converted to policy order in grasp() method)
        self.grasp_position = np.array([
            1.500, 0.961, 0.145, 0.042, 0.906, 0.174,
            0.506, 0.483, 1.214, 0.340, 0.061, -0.051
        ], dtype=np.double)
        
        # Release position in policy order
        # (directly used by _move_to_position)
        self.release_position = np.array([
            -0.17, 0.7, 0.44, 0.7, 0.44, 0.0,
            0.0, 0.0, 0.0, 1.3, -0.2, 1.5
        ], dtype=np.double)
        
        self.initial_target_position = np.array([
            -0.17, 1.2, 0.4, 1.2, 0.4, 0, 
            0, 0, 0, 1.15, 0.6, 0.1
        ], dtype=np.double)


# =============================================================================
# Keyboard Control Mapping
# =============================================================================

class KeyboardKeys:
    """Keyboard key mappings for control."""
    STOP_EXECUTION = 'l'
    START_SAVE_DATA = 'r'
    MARK_SWITCH_EVENT = 'c'
    # Quest controller virtual key
    UNUSED_KEY = 'v'


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Create global config instances
NETWORK_CONFIG = NetworkConfig()
ROBOT_CONFIG = RobotConfig()
CAMERA_CONFIG = CameraConfig()
RESET_POSITIONS = ResetJointPositions()
CONTROL_CONFIG = ControlConfig()
DATA_CONFIG = DataCollectionConfig()
BC_AGENT_CONFIG = BCAgentConfig()
XHAND_CONFIG = XHandConfig()

