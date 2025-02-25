import os
import sys
import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ppo_lstm.agent import Agent
from src.ppo_lstm.config import Args
from src.environments import cartpole_no_vel  # ensure CartpoleNoVel is registered

def main():
    parser = argparse.ArgumentParser(description="Record CartPole agent at different training stages")
    parser.add_argument("--checkpoint-dir", type=str, required=True, 
                        help="Directory containing checkpoint files")
    parser.add_argument("--output-dir", type=str, default="outputs/videos/cartpole_progression",
                        help="Directory to save videos")
    parser.add_argument("--num-episodes", type=int, default=3,
                        help="Number of episodes to record for each checkpoint")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum number of steps per episode")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get checkpoint files sorted by training steps
    checkpoint_files = sorted(
        [f for f in os.listdir(args.checkpoint_dir) if f.startswith("checkpoint_")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    
    # Select checkpoints to record (first, mid-points, and final)
    if len(checkpoint_files) > 5:
        indices = [0, len(checkpoint_files)//4, len(checkpoint_files)//2, 
                   3*len(checkpoint_files)//4, len(checkpoint_files)-1]
        selected_checkpoints = [checkpoint_files[i] for i in indices]
    else:
        selected_checkpoints = checkpoint_files
    
    print(f"Selected checkpoints: {selected_checkpoints}")
    
    # Setup environment
    env_config = Args(
        env_id="CartPoleNoVel-v0",
        num_envs=1,
        hidden_size=8,
        hidden_layer_size=16,
        seq_len=8
    )
    
    for checkpoint_file in selected_checkpoints:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        checkpoint_steps = int(checkpoint_file.split("_")[1].split(".")[0])
        
        # Create a video directory for this checkpoint
        video_dir = os.path.join(args.output_dir, f"steps_{checkpoint_steps}")
        os.makedirs(video_dir, exist_ok=True)
        
        # Create environment with video recording
        env = gym.make("CartPoleNoVel-v0", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(
            env, 
            video_dir,
            episode_trigger=lambda x: True  # Record all episodes
        )
        env = gym.vector.SyncVectorEnv([lambda: env])
        
        # Initialize agent and load checkpoint
        agent = Agent(env_config, env, f"cartpole_{checkpoint_steps}", None)
        agent.load(checkpoint_path)
        
        print(f"Recording checkpoint {checkpoint_file} ({checkpoint_steps} steps)")
        
        # Record episodes
        for episode in range(args.num_episodes):
            obs, _ = env.reset()
            
            # Reset LSTM state
            hidden = torch.zeros(2, 1, agent.args.hidden_size, device=agent.device)
            
            done = False
            step = 0
            
            while not done and step < args.max_steps:
                # Get action using current LSTM state
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device)
                
                with torch.no_grad():
                    features, next_hidden = agent.feature_extractor.get_features(obs_tensor, hidden)
                    
                    # Get deterministic action (argmax of policy distribution)
                    logits = agent.actor_head(features)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.argmax(probs, dim=-1)
                    
                    value = agent.critic_head(features)
                
                # Update hidden state
                hidden = next_hidden
                obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                done = terminated[0] or truncated[0]
                step += 1
            
            print(f"  Episode {episode+1} completed: {step} steps")
        
        env.close()
        
    print(f"All videos saved to {args.output_dir}")

if __name__ == "__main__":
    main()