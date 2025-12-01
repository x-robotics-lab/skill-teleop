#!/bin/bash
# Example script for BC policy deployment

# This script demonstrates how to run a trained BC policy
# Make sure to start launch_nodes.py in another terminal first!

echo "=================================="
echo "BC Policy Deployment Example"
echo "=================================="
echo ""
echo "Prerequisites:"
echo "1. Launch nodes running (python launch_nodes.py)"
echo "2. BC model checkpoint available"
echo "3. Robots powered on"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Run BC policy
python run_env.py \
    --agent bc \
    --hz 20

echo ""
echo "BC policy deployment ended."

