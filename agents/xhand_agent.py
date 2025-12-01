"""XHand dexterous gripper control with learned policy."""

import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from xhand_controller import xhand_control

from config import XHAND_CONFIG


class XHandControl:
    """Low-level control interface for XHand hardware."""
    
    NUM_FINGERS = 12
    
    def __init__(
        self,
        hand_id: int = XHAND_CONFIG.hand_id,
        initial_position: float = XHAND_CONFIG.default_position,
        control_mode: int = XHAND_CONFIG.control_mode
    ):
        """Initialize XHand control interface.
        
        Args:
            hand_id: Hardware ID of the hand
            initial_position: Initial position for all joints
            control_mode: Control mode (3 = position control)
        """
        self._hand_id = hand_id
        self._device = xhand_control.XHandControl()
        self._hand_command = xhand_control.HandCommand_t()
        
        # Initialize all finger commands
        for finger_idx in range(self.NUM_FINGERS):
            cmd = self._hand_command.finger_command[finger_idx]
            cmd.id = finger_idx
            cmd.kp = XHAND_CONFIG.kp
            cmd.ki = XHAND_CONFIG.ki
            cmd.kd = XHAND_CONFIG.kd
            cmd.position = initial_position
            cmd.tor_max = XHAND_CONFIG.torque_max
            cmd.mode = control_mode

    def enumerate_devices(self, protocol: str) -> Optional[list]:
        """Enumerate available XHand devices.
        
        Args:
            protocol: Communication protocol ('RS485' or 'EtherCAT')
            
        Returns:
            List of available device ports
        """
        print("=" * 50)
        print(f"Enumerating {protocol} devices...")
        print("=" * 50)
        ports = self._device.enumerate_devices(protocol)
        print(f"Found devices: {ports}\n")
        return ports

    def open_device(self, device_config: dict) -> bool:
        """Open connection to XHand device.
        
        Args:
            device_config: Dictionary with 'protocol' and connection parameters
            
        Returns:
            True if successful, False otherwise
        """
        print("=" * 50)
        print("Opening XHand device...")
        print("=" * 50)
        
        protocol = device_config["protocol"]
        
        if protocol == "RS485":
            device_config["baud_rate"] = int(device_config["baud_rate"])
            response = self._device.open_serial(
                device_config["serial_port"],
                device_config["baud_rate"],
            )
            print(f"RS485 connection: {'SUCCESS' if response.error_code == 0 else 'FAILED'}\n")
            
        elif protocol == "EtherCAT":
            ethercat_devices = self.enumerate_devices("EtherCAT")
            if not ethercat_devices:
                print("ERROR: No EtherCAT devices found\n")
                return False
            response = self._device.open_ethercat(ethercat_devices[0])
            print(f"EtherCAT connection: {'SUCCESS' if response.error_code == 0 else 'FAILED'}\n")
        else:
            print(f"ERROR: Unknown protocol '{protocol}'\n")
            return False
        
        if response.error_code != 0:
            print(f"ERROR: {response.error_message}")
            print("Please check connection and try again.\n")
            return False
        
        return True
            
    def send_command(self) -> None:
        """Send current command to XHand hardware."""
        self._device.send_command(self._hand_id, self._hand_command)
        
    def read_joint_position(self, finger_id: int, force_update: bool = True) -> Optional[float]:
        """Read position of a single joint.
        
        Args:
            finger_id: Index of finger joint (0-11)
            force_update: Force hardware state update before reading
            
        Returns:
            Joint position in radians, or None if error
        """
        error_struct, state = self._device.read_state(self._hand_id, force_update)
        if error_struct.error_code != 0:
            print(f"ERROR reading joint {finger_id}: {error_struct.error_message}")
            return None
        
        return state.finger_state[finger_id].position

    def read_all_joint_positions(self) -> np.ndarray:
        """Read positions of all joints.
        
        Returns:
            Array of 12 joint positions
        """
        positions = []
        for finger_id in range(self.NUM_FINGERS):
            pos = self.read_joint_position(finger_id, force_update=(finger_id == 0))
            if pos is None:
                print(f"WARNING: Failed to read joint {finger_id}, using 0.0")
                pos = 0.0
            positions.append(pos)
        return np.array(positions, dtype=np.float64)

    def read_state(self, force_update: bool = True) -> Tuple:
        """Read full hardware state.
        
        Args:
            force_update: Force hardware state update
            
        Returns:
            Tuple of (error_struct, state)
        """
        return self._device.read_state(self._hand_id, force_update)

    def get_tactile_data(self) -> np.ndarray:
        """Read tactile sensor data from all fingers.
        
        Returns:
            Array of shape (5, 120, 3) containing force data (fx, fy, fz)
        """
        tactile_data = np.zeros((5, 120, 3), dtype=np.float64)
        error, state = self.read_state(force_update=True)
        
        if error.error_code == 0 and state is not None:
            sensor_data_list = getattr(state, "sensor_data", None)
            if sensor_data_list:
                for finger_idx in range(min(5, len(sensor_data_list))):
                    sensor_data = sensor_data_list[finger_idx]
                    forces = [[f.fx, f.fy, f.fz] for f in sensor_data.raw_force]
                    np.copyto(tactile_data[finger_idx], np.asarray(forces, dtype=np.float64))
        
        return tactile_data


class XHandPolicyController:
    """High-level controller for XHand using learned policy."""
    
    def __init__(self, use_screw_task: bool = True):
        """Initialize XHand policy controller.
        
        Args:
            use_screw_task: Whether to use screw-specific configuration
        """
        self.is_policy_running = False
        self.xhand_control = XHandControl()
        self.device = torch.device("cpu")
        self.use_screw_task = use_screw_task
        self.step_count = 0
        
        # Load policy model
        self._load_policy_model()
        
        # Initialize hardware connection
        self._initialize_hardware()
        
        # Initialize policy state
        self._initialize_policy_state()
        
        print("=" * 70)
        print("XHand Policy Controller initialized successfully")
        print("=" * 70)

    def _load_policy_model(self) -> None:
        """Load JIT-compiled policy model."""
        policy_path = XHAND_CONFIG.screw_task_policy_path
        self.policy_model = torch.jit.load(policy_path, map_location=self.device)
        self.policy_model.eval()
        print(f"[Policy] Loaded model from: {policy_path}")

    def _initialize_hardware(self) -> None:
        """Initialize hardware connection."""
        device_config = {'protocol': 'EtherCAT'}
        if not self.xhand_control.open_device(device_config):
            print("FATAL: Failed to initialize XHand hardware")
            sys.exit(1)
        print("[Hardware] XHand control initialized")

    def _initialize_policy_state(self) -> None:
        """Initialize policy state buffers and target positions."""
        # Create joint index mappings
        self.xhand_to_policy_idx = XHAND_CONFIG.xhand_to_policy_indices
        self.policy_to_xhand_idx = [0] * len(self.xhand_to_policy_idx)
        for xhand_idx, policy_idx in enumerate(self.xhand_to_policy_idx):
            self.policy_to_xhand_idx[policy_idx] = xhand_idx
        
        # Load initial history buffer for screw task
        if self.use_screw_task:
            buffer_init = np.load(XHAND_CONFIG.buffer_init_path)
            self.proprioception_history = torch.tensor(
                buffer_init, device=self.device, dtype=torch.float32
            )
            self.previous_target = buffer_init[0, -1, :12]
        else:
            # Initialize with default values
            base_proprio = torch.zeros(
                XHAND_CONFIG.proprioception_dim,
                device=self.device,
                dtype=torch.float32
            )
            self.proprioception_history = base_proprio.view(1, 1, -1).repeat(
                1, XHAND_CONFIG.proprioception_history_length, 1
            )
            self.previous_target = XHAND_CONFIG.initial_target_position.copy()
        
        self.observation_buffer = torch.zeros(
            1, XHAND_CONFIG.observation_dim,
            device=self.device,
            dtype=torch.float32
        )
        
        # Move to initial position
        self._move_to_position(self.previous_target)
        time.sleep(1)  # Allow time to reach position
        
        print("[Policy] State initialized")

    def _move_to_position(self, target_position_policy_order: np.ndarray) -> None:
        """Move hand to specified position.
        
        Args:
            target_position_policy_order: Target joint positions in policy order
        """
        target_xhand_order = self._convert_policy_to_xhand_order(target_position_policy_order)
        for finger_idx in range(XHandControl.NUM_FINGERS):
            self.xhand_control._hand_command.finger_command[finger_idx].position = \
                target_xhand_order[finger_idx]
        self.xhand_control.send_command()

    def _convert_xhand_to_policy_order(self, positions: np.ndarray) -> np.ndarray:
        """Convert joint positions from XHand order to policy order."""
        return np.asarray(positions)[self.xhand_to_policy_idx]

    def _convert_policy_to_xhand_order(self, positions: np.ndarray) -> np.ndarray:
        """Convert joint positions from policy order to XHand order."""
        return np.asarray(positions)[self.policy_to_xhand_idx]

    def start_policy(self) -> bool:
        """Start policy execution.
        
        Returns:
            True if started successfully
        """
        if self.is_policy_running:
            print("[Policy] Already running")
            return True
        
        self.is_policy_running = True
        print("[Policy] Started")
        return True
    
    def start(self) -> bool:
        """Alias for start_policy() for compatibility with Quest agent."""
        return self.start_policy()

    def stop_policy(self) -> None:
        """Stop policy execution."""
        if self.is_policy_running:
            self.is_policy_running = False
            print("[Policy] Stopped")
    
    def stop(self) -> None:
        """Alias for stop_policy() for compatibility with Quest agent."""
        self.stop_policy()

    def grasp(self) -> None:
        """Move to grasp position."""
        if self.use_screw_task:
            # XHAND_CONFIG.grasp_position is in XHand hardware order
            # Convert to policy order for _move_to_position
            grasp_pos = self._convert_xhand_to_policy_order(XHAND_CONFIG.grasp_position)
        else:
            # Default grasp position is already in policy order
            grasp_pos = np.array([
                -0.17, 1.3, 0.44, 1.3, 0.44, 0.0,
                0.0, 0.0, 0.0, 1.3, 0.2, 1.0
            ], dtype=np.double)
        
        self._move_to_position(grasp_pos)
        print("[Gripper] Moved to grasp position")

    def release(self) -> None:
        """Move to release position."""
        # XHAND_CONFIG.release_position is already in policy order
        # Directly pass to _move_to_position
        self._move_to_position(XHAND_CONFIG.release_position)
        print("[Gripper] Moved to release position")

    def step(self) -> None:
        """Execute one policy step.
        
        This should be called in a control loop when policy is running.
        """
        if not self.is_policy_running:
            return
        
        # Read current joint positions
        current_positions_xhand = self.xhand_control.read_all_joint_positions()
        current_positions_policy = self._convert_xhand_to_policy_order(current_positions_xhand)
        
        # Run policy inference
        with torch.no_grad():
            # Prepare observation
            obs = torch.clamp(self.observation_buffer, -5.0, 5.0)
            obs = F.pad(obs, (0, XHAND_CONFIG.observation_dim - obs.shape[1]))
            
            # Normalize observation
            norm_obs = (obs - self.policy_model.running_mean.unsqueeze(0)) / \
                      torch.sqrt(self.policy_model.running_var.unsqueeze(0) + 1e-5)
            norm_obs = torch.clamp(norm_obs, -5.0, 5.0)
            
            # Normalize proprioception history
            norm_proprio = (self.proprioception_history - self.policy_model.sa_mean.unsqueeze(0)) / \
                          torch.sqrt(self.policy_model.sa_var.unsqueeze(0) + 1e-5)
            norm_proprio = torch.clamp(norm_proprio, -5.0, 5.0)
            
            # Dummy point cloud info (not used)
            point_cloud_info = torch.zeros((1, 100, 3))
            
            # Policy inference
            policy_output, _, _ = self.policy_model({
                'obs': norm_obs,
                'proprio_hist': norm_proprio,
                'point_cloud_info': point_cloud_info,
            })
            
            policy_action = torch.clamp(policy_output, -1.0, 1.0).cpu().numpy().flatten()
            
            # Zero out specific joints for screw task
            if self.use_screw_task:
                policy_action[5:7] = 0
            else:
                policy_action[5:9] = 0
        
        # Compute target position
        target_position = policy_action * XHAND_CONFIG.action_scale + self.previous_target
        target_position = np.clip(
            target_position,
            XHAND_CONFIG.dof_lower_limits,
            XHAND_CONFIG.dof_upper_limits
        )
        
        # Update state
        self.previous_target = target_position.copy()
        self.observation_buffer = self.proprioception_history[:, -3:, :].reshape(1, -1).clone()
        
        # Update proprioception history
        current_dof = torch.from_numpy(current_positions_policy).float().to(self.device)
        target_dof = torch.from_numpy(target_position).float().to(self.device)
        current_proprio = torch.cat([current_dof, target_dof], dim=0).unsqueeze(0).unsqueeze(0)
        self.proprioception_history = torch.cat(
            [self.proprioception_history[:, 1:], current_proprio], 
            dim=1
        )
        
        # Send command to hardware
        self._move_to_position(target_position)
        
        self.step_count += 1
