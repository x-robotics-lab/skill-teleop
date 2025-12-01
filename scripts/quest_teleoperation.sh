#!/bin/bash
# Example script for Quest teleoperation with data collection

# This script demonstrates how to run Quest teleoperation with data collection
# Make sure to start launch_nodes.py in another terminal first!

echo "=================================="
echo "Quest Teleoperation Example"
echo "=================================="
echo ""
echo "Prerequisites:"
echo "1. Launch nodes running (python launch_nodes.py)"
echo "2. Quest controller connected"
echo "3. Robots powered on"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Run Quest teleoperation with data collection
python run_env.py \
    --agent quest \
    --hz 20 \
    --save_data \
    --data_dir /shared/data/screw_driver

echo ""
echo "Teleoperation session ended."

