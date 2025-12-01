#!/bin/bash
# Script to start hardware nodes

echo "=================================="
echo "Starting Hardware Nodes"
echo "=================================="
echo ""
echo "This will start:"
echo "- Camera server (port 5000)"
echo "- Robot server (port 6000)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start launch nodes
python launch_nodes.py \
    --robot_type bimanual_ur \
    --use_faster_camera

echo ""
echo "Hardware nodes stopped."

