"""Launch hardware nodes for camera and robot servers.

This script starts ZMQ servers for camera and robot hardware communication.
The camera server runs in a separate process, while the robot server runs
in the main process (blocking).
"""

from dataclasses import dataclass
from multiprocessing import Process
from typing import List, Optional, Tuple

import tyro

from nodes.camera_node import ZMQServerCamera, ZMQServerCameraFaster
from nodes.robot_node import ZMQServerRobot
from robots.robot import BimanualRobot
from config import NETWORK_CONFIG, ROBOT_CONFIG, CAMERA_CONFIG


@dataclass
class LaunchArgs:
    """Arguments for launching hardware nodes."""
    robot_type: str = "bimanual_ur"
    hostname: str = NETWORK_CONFIG.hostname
    use_camera: bool = CAMERA_CONFIG.use_camera
    use_faster_camera: bool = True
    image_size: Optional[Tuple[int, int]] = CAMERA_CONFIG.default_image_size


def launch_camera_server(port: int, camera_ids: List[str], args: LaunchArgs) -> None:
    """Launch camera server with specified port and camera IDs.
    
    Args:
        port: ZMQ port number for camera server
        camera_ids: List of RealSense camera serial numbers
        args: Launch arguments containing configuration
    """
    from cameras.realsense_camera import RealSenseCamera

    camera = RealSenseCamera(camera_ids, img_size=args.image_size)

    if args.use_faster_camera:
        server = ZMQServerCameraFaster(camera, port=port, host=args.hostname)
    else:
        server = ZMQServerCamera(camera, port=port, host=args.hostname)
    
    print(f"[Camera Server] Starting on port {port} with cameras {camera_ids}")
    server.serve()


def launch_robot_server(port: int, args: LaunchArgs) -> None:
    """Launch robot server with specified port.
    
    Args:
        port: ZMQ port number for robot server
        args: Launch arguments containing configuration
        
    Raises:
        NotImplementedError: If robot type is not supported
    """
    if args.robot_type == "bimanual_ur":
        from robots.ur import URRobot

        print(f"[Robot Server] Connecting to left robot at {ROBOT_CONFIG.left_robot_ip}")
        left_robot = URRobot(robot_ip=ROBOT_CONFIG.left_robot_ip, no_gripper=True)
        
        print(f"[Robot Server] Connecting to right robot at {ROBOT_CONFIG.right_robot_ip}")
        right_robot = URRobot(robot_ip=ROBOT_CONFIG.right_robot_ip, no_gripper=True)
        
        robot = BimanualRobot(left_robot, right_robot)
    else:
        raise NotImplementedError(
            f"Robot type '{args.robot_type}' not implemented. "
            f"Supported types: ['bimanual_ur']"
        )
    
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"[Robot Server] Starting on port {port}")
    server.serve()


def create_camera_server_process(args: LaunchArgs) -> Process:
    """Create camera server process.
    
    Args:
        args: Launch arguments containing configuration
        
    Returns:
        Process object for camera server
    """
    # Use wrist-mounted camera
    active_cameras = [CAMERA_CONFIG.wrist_camera_id]
    
    print(f"[Main] Creating camera server process for cameras: {active_cameras}")
    server_process = Process(
        target=launch_camera_server,
        args=(NETWORK_CONFIG.camera_port, active_cameras, args)
    )
    return server_process


def main(args: LaunchArgs) -> None:
    """Main function to launch camera and robot servers.
    
    The camera server runs in a separate process, while the robot server
    runs in the main process (blocking).
    
    Args:
        args: Launch arguments from command line
    """
    print("=" * 70)
    print("Screw Driver Hardware Node Launcher")
    print("=" * 70)
    print(f"Camera: {'ENABLED' if args.use_camera else 'DISABLED'}")
    print("=" * 70)
    
    camera_server = None
    
    # Start camera server process (if enabled)
    if args.use_camera:
        camera_server = create_camera_server_process(args)
        print("[Main] Starting camera server process...")
        camera_server.start()
        print("[Main] Camera server process started")
        print("-" * 70)
    else:
        print("[Main] Camera disabled - skipping camera server")
        print("-" * 70)
    
    # Launch robot server (blocking call)
    print("[Main] Launching robot server (this will block)...")
    try:
        launch_robot_server(NETWORK_CONFIG.robot_port, args)
    except KeyboardInterrupt:
        print("\n[Main] Received keyboard interrupt, shutting down...")
        if camera_server:
            camera_server.terminate()
            camera_server.join()
        print("[Main] Cleanup complete")


if __name__ == "__main__":
    main(tyro.cli(LaunchArgs))
