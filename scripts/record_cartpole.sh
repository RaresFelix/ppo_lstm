#!/bin/bash
# Script to record CartPole agent at different training stages

# This is the best performing run from February 2nd, 2025, at 11:56:46 PM
CHECKPOINT_DIR="/home/raresfelix/remote-server/workspace/2501/ppo_lstm/checkpoints/ppo_lstm_CartPoleNoVel-v0_3x3_1738533407"
OUTPUT_DIR="/home/raresfelix/remote-server/workspace/2501/ppo_lstm/outputs/videos/cartpole_progression"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the recording script
python /home/raresfelix/remote-server/workspace/2501/ppo_lstm/scripts/record_cartpole.py \
  --checkpoint-dir $CHECKPOINT_DIR \
  --output-dir $OUTPUT_DIR \
  --num-episodes 3 \
  --max-steps 500

echo "Recording complete. Videos saved to $OUTPUT_DIR"