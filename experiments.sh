# Store the base command
CMD="python -m src.ppo_lstm.main --view-size 5 --env-id MiniGrid-MemoryS9-v0 --wandb-group ppo_lstm_S9_5x5_onehot --one-hot --use-wandb --total-steps 5000000"

# Array to store PIDs
pids=()

# Trap Ctrl+C (SIGINT) and Ctrl+Z (SIGTSTP)
trap 'echo "Caught signal, killing all processes..."; for pid in "${pids[@]}"; do kill -9 $pid 2>/dev/null; done; exit 1' SIGINT SIGTSTP

# Start processes and store PIDs
for gpu_id in 2 3; do
    # First experiment on this GPU
    $CMD --gpu-id $gpu_id &
    pids+=($!)
    
    # Second experiment on this GPU
    $CMD --gpu-id $gpu_id &
    pids+=($!)
done

# Wait for all processes
wait