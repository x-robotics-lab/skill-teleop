"""Main control script for robot teleoperation and policy deployment.

This script supports two modes:
1. Quest teleoperation with data collection
2. BC (Behavior Cloning) policy deployment

Usage:
    # Quest teleoperation with data collection
    python run_env.py --agent quest --hz 20 --save_data
    
    # BC policy deployment
    python run_env.py --agent bc --hz 20
"""

import datetime
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import termcolor
import tyro
from pynput import keyboard

from agents.agent import Agent
from nodes.camera_node import ZMQClientCamera
from core.env import RobotEnv
from nodes.robot_node import ZMQClientRobot
from config import (
    CAMERA_CONFIG,
    NETWORK_CONFIG,
    RESET_POSITIONS,
    CONTROL_CONFIG,
    DATA_CONFIG,
    BC_AGENT_CONFIG,
    ROBOT_CONFIG,
    KeyboardKeys,
)


# =============================================================================
# Keyboard Controller
# =============================================================================

class KeyboardController:
    """Manages keyboard input for robot control."""
    
    def __init__(self):
        """Initialize keyboard controller with listeners."""
        self.keys_pressed = {
            KeyboardKeys.STOP_EXECUTION: False,
            KeyboardKeys.START_SAVE_DATA: False,
            KeyboardKeys.MARK_SWITCH_EVENT: False,
            KeyboardKeys.UNUSED_KEY: False,
        }
        
        # Start keyboard listeners
        self._press_listener = keyboard.Listener(on_press=self._on_key_press)
        self._release_listener = keyboard.Listener(on_release=self._on_key_release)
        self._press_listener.start()
        self._release_listener.start()
    
    def _on_key_press(self, key) -> None:
        """Handle key press event."""
        try:
            if key.char in self.keys_pressed:
                self.keys_pressed[key.char] = True
        except (AttributeError, KeyError):
            pass
    
    def _on_key_release(self, key) -> None:
        """Handle key release event."""
        try:
            if key.char in self.keys_pressed:
                self.keys_pressed[key.char] = False
        except (AttributeError, KeyError):
            pass
    
    def is_pressed(self, key: str) -> bool:
        """Check if a key is currently pressed."""
        return self.keys_pressed.get(key, False)
    
    def stop_requested(self) -> bool:
        """Check if stop key was pressed."""
        return self.is_pressed(KeyboardKeys.STOP_EXECUTION)
    
    def save_data_requested(self) -> bool:
        """Check if save data key was pressed."""
        return self.is_pressed(KeyboardKeys.START_SAVE_DATA)
    
    def switch_event_requested(self) -> bool:
        """Check if switch event key was pressed."""
        return self.is_pressed(KeyboardKeys.MARK_SWITCH_EVENT)


# =============================================================================
# Data Collector
# =============================================================================

class DataCollector:
    """Handles data collection and saving."""
    
    def __init__(self, save_dir: Path, save_png: bool = False):
        """Initialize data collector.
        
        Args:
            save_dir: Directory to save data
            save_png: Whether to save RGB images as PNG files
        """
        self.save_dir = save_dir
        self.save_png = save_png
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frame_frequencies = []
        
        print(f"[Data] Saving to: {self.save_dir}")
    
    def save_frame(
        self,
        timestamp: datetime.datetime,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        is_control_active: bool,
        xhand_position: Optional[np.ndarray] = None,
        xhand_command: Optional[np.ndarray] = None,
        xhand_tactile: Optional[np.ndarray] = None,
        xhand_policy_running: bool = False,
        is_switch_event: bool = False,
    ) -> None:
        """Save a single frame of data.
        
        Args:
            timestamp: Timestamp of the frame
            observation: Robot observations
            action: Action taken
            is_control_active: Whether control was active
            xhand_position: XHand joint positions
            xhand_command: XHand joint commands
            xhand_tactile: XHand tactile data
            xhand_policy_running: Whether XHand policy was running
            is_switch_event: Whether this is a marked switch event
        """
        # Prepare data dictionary
        data = observation.copy()
        data["activated"] = is_control_active
        data["control"] = action
        data["xhand_pos"] = xhand_position if xhand_position is not None else np.zeros(12)
        data["xhand_act"] = xhand_command if xhand_command is not None else np.zeros(12)
        data["xhand_tactile"] = xhand_tactile
        data["xhand_rl_flag"] = xhand_policy_running
        data["switch"] = is_switch_event
        
        if is_switch_event:
            print("[Data] Switch event marked")
        
        # Generate filename
        timestamp_str = timestamp.isoformat().replace(":", "-").replace(".", "-")
        filepath = self.save_dir / f"{timestamp_str}.pkl"
        
        # Save as pickle
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        # Optionally save RGB images
        if self.save_png and "base_rgb" in data:
            rgb_images = data["base_rgb"]
            for cam_idx in range(rgb_images.shape[0]):
                rgb_bgr = cv2.cvtColor(rgb_images[cam_idx], cv2.COLOR_RGB2BGR)
                png_filepath = str(filepath)[:-4] + f"-cam{cam_idx}.png"
                cv2.imwrite(png_filepath, rgb_bgr)
    
    def record_frame_time(self, frame_time: float) -> None:
        """Record frame processing time.
        
        Args:
            frame_time: Time taken to process frame (seconds)
        """
        if frame_time > 0:
            self.frame_frequencies.append(1.0 / frame_time)
    
    def save_statistics(self) -> None:
        """Save timing statistics to file."""
        if not self.frame_frequencies:
            return
        
        freq_array = np.array(self.frame_frequencies[1:])  # Skip first frame
        stats_file = self.save_dir / "freq.txt"
        
        with open(stats_file, "w") as f:
            f.write(f"Average FPS: {np.mean(freq_array):.2f}\n")
            f.write(f"Max FPS: {np.max(freq_array):.2f}\n")
            f.write(f"Min FPS: {np.min(freq_array):.2f}\n")
            f.write(f"Std FPS: {np.std(freq_array):.2f}\n\n")
            
            for step, freq in enumerate(self.frame_frequencies):
                f.write(f"{step}: {freq:.2f}\n")
        
        print(f"[Data] Statistics saved to: {stats_file}")


# =============================================================================
# Helper Functions
# =============================================================================

def print_colored(
    *args,
    color: Optional[str] = None,
    attrs: tuple = (),
    **kwargs
) -> None:
    """Print colored text to console.
    
    Args:
        *args: Arguments to print
        color: Text color
        attrs: Text attributes (e.g., 'bold')
        **kwargs: Additional print arguments
    """
    if args:
        args = tuple(termcolor.colored(str(arg), color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def count_existing_trajectories(data_dir: Path) -> int:
    """Count number of existing trajectory folders.
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Number of subdirectories
    """
    if not data_dir.exists():
        return 0
    
    return sum(1 for item in data_dir.iterdir() if item.is_dir())


def interpolate_to_position(
    env: RobotEnv,
    current_joints: np.ndarray,
    target_joints: np.ndarray,
) -> None:
    """Smoothly interpolate robot from current to target position.
    
    Args:
        env: Robot environment
        current_joints: Current joint positions
        target_joints: Target joint positions
    """
    max_delta = np.abs(current_joints - target_joints).max()
    num_steps = min(
        int(max_delta / ROBOT_CONFIG.max_joint_delta_per_step),
        ROBOT_CONFIG.max_interpolation_steps
    )
    num_steps = max(num_steps, 1)  # At least 1 step
    
    for joint_positions in np.linspace(current_joints, target_joints, num_steps):
        env.step(joint_positions)


def create_bc_observation(
    robot_obs: Dict[str, np.ndarray],
    xhand_position: np.ndarray,
    xhand_command: np.ndarray,
    xhand_tactile: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Create observation dictionary for BC agent.
    
    Args:
        robot_obs: Robot observations
        xhand_position: XHand joint positions
        xhand_command: XHand joint commands
        xhand_tactile: XHand tactile data
        
    Returns:
        Formatted observation dictionary for BC agent
    """
    return {
        'base_rgb': robot_obs['base_rgb'],
        'base_depth': robot_obs['base_depth'],
        'joint_positions': robot_obs['joint_positions'],
        'joint_velocities': robot_obs['joint_velocities'],
        'eef_speed': robot_obs['eef_speed'],
        'ee_pos_quat': robot_obs['ee_pos_quat'],
        'xhand_pos': xhand_position,
        'xhand_act': xhand_command,
        'xhand_tactile': xhand_tactile.reshape(-1),  # Flatten tactile data
        'action': robot_obs['joint_positions'],  # Use current position as action
    }


def extract_arm_actions(full_action: np.ndarray, agent_type: str) -> np.ndarray:
    """Extract arm joint actions from full action vector.
    
    Different agents output actions in different formats. This function
    extracts just the arm joint commands.
    
    Args:
        full_action: Full action vector from agent
        agent_type: Type of agent ('quest' or 'bc')
        
    Returns:
        Arm joint commands (12 DOF for bimanual)
    """
    if agent_type == "bc":
        # BC: First 12 dimensions are arm joints
        return full_action[:12]
    else:  # quest
        # Quest: Remove gripper dimensions (skip indices 6 and 13)
        return np.concatenate([full_action[0:6], full_action[7:13]])


# =============================================================================
# Agent Initialization
# =============================================================================

def create_quest_agent(robot_type: str) -> Agent:
    """Create Quest teleoperation agent.
    
    Args:
        robot_type: Type of robot ('ur5')
        
    Returns:
        Initialized Quest agent
    """
    from agents.quest_agent import SingleArmQuestAgent, DualArmQuestAgent
    
    left_agent = SingleArmQuestAgent(robot_type=robot_type, which_hand="l")
    right_agent = SingleArmQuestAgent(robot_type=robot_type, which_hand="r")
    agent = DualArmQuestAgent(left_agent, right_agent)
    
    print("[Agent] Quest teleoperation agent created")
    return agent


def create_bc_agent(checkpoint_path: str, use_async: bool) -> Agent:
    """Create BC (Behavior Cloning) agent.
    
    Args:
        checkpoint_path: Path to model checkpoint
        use_async: Whether to use asynchronous execution
        
    Returns:
        Initialized BC agent
    """
    from minbc.agents.diffusion.diffusion_agent_sync import DiffusionAgent
    from minbc.agents.diffusion.diffusion_agent_client import DiffusionAgentClient
    
    if use_async:
        agent = DiffusionAgentClient(
            ckpt_path=checkpoint_path,
            temporal_ensemble_mode="avg"
        )
    else:
        agent = DiffusionAgent(ckpt_path=checkpoint_path)
    
    print(f"[Agent] BC agent created from: {checkpoint_path}")
    return agent


# =============================================================================
# Main Control Loop
# =============================================================================

@dataclass
class ControlArgs:
    """Command-line arguments for robot control."""
    
    # Network settings
    robot_port: int = NETWORK_CONFIG.robot_port
    base_camera_port: int = NETWORK_CONFIG.camera_port
    hostname: str = NETWORK_CONFIG.hostname
    
    # Control settings
    hz: int = CONTROL_CONFIG.default_control_hz
    use_camera: bool = CAMERA_CONFIG.use_camera
    show_camera_view: bool = CONTROL_CONFIG.default_show_camera
    
    # Agent selection
    agent: str = "quest"  # "quest" or "bc"
    robot_type: str = "ur5"
    
    # Data collection
    save_data: bool = False
    save_depth: bool = DATA_CONFIG.save_depth_by_default
    save_png: bool = DATA_CONFIG.save_png_by_default
    data_dir: str = DATA_CONFIG.default_data_dir
    
    # BC agent settings
    bc_checkpoint_path: str = BC_AGENT_CONFIG.default_checkpoint_path
    bc_use_async: bool = BC_AGENT_CONFIG.use_async_by_default
    bc_num_diffusion_iters: int = BC_AGENT_CONFIG.num_diffusion_iterations


def main(args: ControlArgs) -> None:
    """Main control loop.
    
    Args:
        args: Control arguments from command line
    """
    print("=" * 70)
    print("Screw Driver Robot Control")
    print("=" * 70)
    print(f"Agent: {args.agent}")
    print(f"Control frequency: {args.hz} Hz")
    print(f"Camera: {'ENABLED' if args.use_camera else 'DISABLED'}")
    print(f"Data collection: {'ENABLED' if args.save_data else 'DISABLED'}")
    print("=" * 70)
    
    # Initialize keyboard controller
    keyboard_ctrl = KeyboardController()
    
    # Connect to hardware
    # Only connect to camera if enabled
    if args.use_camera:
        camera_clients = {
            "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        print("[Setup] Camera client connected")
    else:
        camera_clients = {}
        print("[Setup] Camera disabled - using dummy observations")
    
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
        show_camera_view=args.show_camera_view and args.use_camera,
        save_depth=args.save_depth,
    )
    
    # Create agent
    if args.agent == "quest":
        agent = create_quest_agent(args.robot_type)
    elif args.agent == "bc":
        agent = create_bc_agent(args.bc_checkpoint_path, args.bc_use_async)
    else:
        raise ValueError(
            f"Invalid agent type: {args.agent}. Must be 'quest' or 'bc'"
        )
    
    # Move to reset position
    print("\n[Setup] Moving to reset position...")
    current_joints = env.get_obs()["joint_positions"]
    target_joints = RESET_POSITIONS.get_bimanual_reset_rad()
    
    print(f"[Setup] Current joints: {current_joints}")
    print(f"[Setup] Target joints: {target_joints}")
    
    interpolate_to_position(env, current_joints, target_joints)
    
    observation = env.get_obs()
    print(f"[Setup] TCP force: {observation['tcp_force']}")
    
    # Initialize XHand for BC agent
    xhand_controller = None
    if args.agent == "bc":
        print("\n[Setup] Initializing XHand controller...")
        from agents.xhand_agent import XHandPolicyController
        xhand_controller = XHandPolicyController(use_screw_task=True)
    
    # Initialize observation for agent
    if args.agent == "bc":
        xhand_pos = xhand_controller.xhand_control.read_all_joint_positions()
        xhand_cmd = np.array([
            xhand_controller.xhand_control._hand_command.finger_command[i].position
            for i in range(12)
        ])
        xhand_tactile = xhand_controller.xhand_control.get_tactile_data()
        observation = create_bc_observation(observation, xhand_pos, xhand_cmd, xhand_tactile)
    
    # Get initial position from agent
    print("\n[Setup] Getting initial position from agent...")
    initial_action = agent.act(observation)
    initial_joints = extract_arm_actions(initial_action, args.agent)
    
    current_joints = env.get_obs()["joint_positions"]
    assert len(initial_joints) == len(current_joints), \
        f"Action dimension mismatch: agent={len(initial_joints)}, env={len(current_joints)}"
    
    # Setup data collection
    data_collector = None
    if args.save_data:
        timestamp_str = datetime.datetime.now().strftime(DATA_CONFIG.timestamp_format)
        save_path = Path(args.data_dir).expanduser() / timestamp_str
        data_collector = DataCollector(save_path, args.save_png)
        
        traj_number = count_existing_trajectories(Path(args.data_dir).expanduser()) + 1
        print(f"\n[Data] Trajectory number: {traj_number}")
    
    # Initialize action smoothing for BC agent
    action_smoother = None
    if args.agent == "bc":
        from minbc_screw_drvier.utils.utils import MovingAverageQueue
        action_smoother = MovingAverageQueue(
            CONTROL_CONFIG.action_queue_size,
            ROBOT_CONFIG.num_xhand_joints,
            CONTROL_CONFIG.action_smoothing_weight
        )
    
    # Ready to start
    print_colored("\n🚀 Ready to go! 🚀", color="green", attrs=("bold",))
    print("Press 'L' to stop")
    print("Press 'R' to start saving data (if --save_data enabled)")
    print("Press 'C' to mark switch event")
    print("-" * 70)
    
    # Main control loop
    start_time = time.time()
    is_first_frame = True
    is_saving_data = False
    frame_count = 0
    
    try:
        while True:
            frame_start_time = time.time()
            
            # Check for data saving trigger
            if keyboard_ctrl.save_data_requested() and not is_saving_data:
                is_saving_data = True
                print_colored("\n[Data] Data saving ACTIVATED", color="yellow", attrs=("bold",))
            
            # Print elapsed time
            elapsed_time = frame_start_time - start_time
            print_colored(
                f"\rTime: {elapsed_time:.1f}s  Frame: {frame_count}  ",
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )
            
            # Get observation and action
            if args.agent == "bc":
                xhand_pos = xhand_controller.xhand_control.read_all_joint_positions()
                xhand_cmd = np.array([
                    xhand_controller.xhand_control._hand_command.finger_command[i].position
                    for i in range(12)
                ])
                xhand_tactile = xhand_controller.xhand_control.get_tactile_data()
                
                bc_obs = create_bc_observation(observation, xhand_pos, xhand_cmd, xhand_tactile)
                full_action = agent.act(bc_obs)
                
                # Extract arm and xhand actions (consistent with initialization)
                xhand_action = full_action[12:]
                arm_action = extract_arm_actions(full_action, args.agent)
                
                # Apply action smoothing
                if is_first_frame:
                    for _ in range(12):
                        action_smoother.add(arm_action)
                
                arm_action = action_smoother.add(arm_action)
                
            else:  # Quest
                full_action = agent.act(observation)
                arm_action = extract_arm_actions(full_action, args.agent)
                xhand_pos = xhand_cmd = xhand_tactile = None
                
                # Get XHand data from Quest agent if available
                if hasattr(agent, 'agent_left') and hasattr(agent.agent_left, 'xhand_controller'):
                    xhand_ctrl = agent.agent_left.xhand_controller.xhand_control
                    xhand_pos = xhand_ctrl.read_all_joint_positions()
                    xhand_cmd = np.array([
                        xhand_ctrl._hand_command.finger_command[i].position
                        for i in range(12)
                    ])
                    xhand_tactile = xhand_ctrl.get_tactile_data()
                    # Fixed: use correct attribute name (changed from 'running' to 'is_policy_running')
                    xhand_policy_running = agent.agent_left.xhand_controller.is_policy_running
                else:
                    xhand_policy_running = False
            
            # Save data if enabled
            if args.save_data and is_saving_data and not is_first_frame:
                data_collector.save_frame(
                    timestamp=datetime.datetime.now(),
                    observation=observation,
                    action=arm_action.tolist() if isinstance(arm_action, np.ndarray) else arm_action,
                    is_control_active=agent.trigger_state if hasattr(agent, 'trigger_state') else True,
                    xhand_position=xhand_pos,
                    xhand_command=xhand_cmd,
                    xhand_tactile=xhand_tactile,
                    xhand_policy_running=xhand_policy_running if args.agent == "quest" else False,
                    is_switch_event=keyboard_ctrl.switch_event_requested(),
                )
            
            # Execute action
            observation = env.step(arm_action)
            
            # Send XHand command for BC agent
            # BC agent output is in XHand hardware order, send directly without conversion
            if args.agent == "bc":
                for finger_idx in range(12):
                    xhand_controller.xhand_control._hand_command.finger_command[finger_idx].position = \
                        xhand_action[finger_idx]
                xhand_controller.xhand_control.send_command()
            
            # Record timing
            frame_time = time.time() - frame_start_time
            if data_collector:
                data_collector.record_frame_time(frame_time)
            
            is_first_frame = False
            frame_count += 1
            
            # Check for stop command
            if keyboard_ctrl.stop_requested():
                print_colored("\n\n[Control] Stop requested", color="red", attrs=("bold",))
                break
    
    except KeyboardInterrupt:
        print_colored("\n\n[Control] Keyboard interrupt", color="red", attrs=("bold",))
    
    finally:
        print("\n[Control] Shutting down...")
        
        if data_collector:
            data_collector.save_statistics()
        
        print("[Control] Done")
        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(ControlArgs))
