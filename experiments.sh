# Store the base command
CMD="python -m src.ppo_lstm.main --one-hot --use-wandb"

# Array to store PIDs
pids=()

# Trap Ctrl+C (SIGINT) and Ctrl+Z (SIGTSTP)
trap 'echo "Caught signal, killing all processes..."; for pid in "${pids[@]}"; do kill -9 $pid 2>/dev/null; done; exit 1' SIGINT SIGTSTP

# Start processes and store PIDs
for gpu_id in 0 1 2 3; do
    # First experiment on this GPU
    $CMD --gpu-id $gpu_id &
    pids+=($!)
    sleep 2  # Wait 2 seconds
    
    # Second experiment on this GPU
    $CMD --gpu-id $gpu_id &
    pids+=($!)
    sleep 2  # Wait 2 seconds

    # Third experiment on this GPU
    $CMD --gpu-id $gpu_id &
    pids+=($!)
    sleep 2  # Wait 2 seconds
done

# Wait for all processes
wait