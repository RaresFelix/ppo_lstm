#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -s SWEEP_ID -g GPU_IDS [-n NUM_PER_GPU] [-p PROJECT_PATH]"
    echo "Example: $0 -s g1e10tqh -g '0,1' -n 2"
    echo "Options:"
    echo "  -s SWEEP_ID       Weights & Biases sweep ID"
    echo "  -g GPU_IDS        Comma-separated list of GPU IDs (e.g., '0,1,2')"
    echo "  -n NUM_PER_GPU    Number of agents to run per GPU (default: 1)"
    echo "  -p PROJECT_PATH   W&B project path (default: raresfelix/ppo_lstm)"
    exit 1
}

# Function to kill all background processes when the script is terminated
cleanup() {
    echo "Cleaning up..."
    pkill -P $$
    exit
}

# Default values
NUM_PER_GPU=1
PROJECT_PATH="raresfelix/ppo_lstm"

# Parse command line arguments
while getopts "s:g:n:p:h" opt; do
    case $opt in
        s) SWEEP_ID="$OPTARG";;
        g) GPU_IDS="$OPTARG";;
        n) NUM_PER_GPU="$OPTARG";;
        p) PROJECT_PATH="$OPTARG";;
        h) usage;;
        ?) usage;;
    esac
done

# Check required parameters
if [ -z "$SWEEP_ID" ] || [ -z "$GPU_IDS" ]; then
    echo "Error: Missing required parameters"
    usage
fi

# Trap SIGINT (Ctrl+C) and SIGTERM signals
trap cleanup SIGINT SIGTERM

# Convert comma-separated GPU IDs to array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"

# Launch agents for each GPU
for gpu in "${GPU_ARRAY[@]}"; do
    for ((i=1; i<=NUM_PER_GPU; i++)); do
        echo "Starting agent $i on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu wandb agent "$PROJECT_PATH/$SWEEP_ID" &
    done
done

# Wait for all processes to complete
wait