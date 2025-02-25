CMD="python -m src.ppo_lstm.main"

# Array to store PIDs
pids=()

# Trap Ctrl+C (SIGINT) and Ctrl+Z (SIGTSTP)
trap 'echo "Caught signal, killing all processes..."; for pid in "${pids[@]}"; do kill -9 $pid 2>/dev/null; done; exit 1' SIGINT SIGTSTP

# Start processes and store PIDs
for gpu_id in 0; do
    $CMD --gpu-id $gpu_id &
    pids+=($!)
    sleep 2
    
    $CMD --gpu-id $gpu_id &
    pids+=($!)
    sleep 2 

    $CMD --gpu-id $gpu_id &
    pids+=($!)
    sleep 2 

    $CMD --gpu-id $gpu_id &
    pids+=($!)
    sleep 2 
done

wait