#!/bin/bash

# Function to clean up tensorboard processes on exit
cleanup() {
    echo -e "\nStopping TensorBoard instances..."
    # Kill the tensorboard processes
    [[ -n $PID1 ]] && kill $PID1 2>/dev/null
    [[ -n $PID2 ]] && kill $PID2 2>/dev/null
    
    echo "TensorBoard instances stopped."
    exit 0
}

# Trap Ctrl+C (SIGINT) and call cleanup function
trap cleanup SIGINT SIGTERM

echo "Starting TensorBoard instances..."
echo "------------------------------------"
echo "Main TensorBoard: http://localhost:6006"
echo "SSL Models TensorBoard: http://localhost:6007"
echo "------------------------------------"
echo "Press Ctrl+C to stop all TensorBoard instances"

# Start the first TensorBoard instance in the background
tensorboard --logdir=./saved_models/ --port=6006 &
PID1=$!
echo "Main TensorBoard started with PID: $PID1"

# Start the second TensorBoard instance in the background
tensorboard --logdir=./ssl_models/saved_models/ --port=6007 &
PID2=$!
echo "SSL Models TensorBoard started with PID: $PID2"

# Wait for either process to finish
# This is needed to keep the script running
wait $PID1 $PID2

# If we get here, one of the processes exited naturally
# Clean up the other one
cleanup