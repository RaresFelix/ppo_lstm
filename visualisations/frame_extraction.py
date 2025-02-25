import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
# Add parent directory to system path to access src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import minigrid
from PIL import Image
import minigrid.wrappers
from src.ppo_lstm import Agent, Args
from src.ppo_lstm.main import make_env
import torch
from tqdm import tqdm
import plotly.graph_objects as go
import os
from datetime import datetime

path = '/workspace/2501/ppo_lstm/important_checkpoints/ppo_lstm_MiniGrid-MemoryS13-v0_3x3_1737817367/checkpoint_4980736.pt'
env_id = 'MiniGrid-MemoryS17Random-v0'

def single_run(run_number, timestamp):
    num_envs = 1
    args = Args(
        env_id=env_id,
        seq_len=8,
        num_envs=num_envs,
        hidden_layer_size=16,
        hidden_size=8,
        view_size=3,
        deployment=True,
    )
    run_name = f'{timestamp}_{run_number}'
    
    # Create subdirectories for this specific run
    base_output_dir = os.path.join('visualisations', 'images', run_name)
    img_output_dir = os.path.join(base_output_dir, 'images')
    data_output_dir = os.path.join(base_output_dir, 'data')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    env = gym.vector.SyncVectorEnv([make_env(args, seed, run_name) for seed in range(1)])
    obs0, _ = env.reset()
    agent = Agent(args, envs=env, run_name=run_name)
    agent.load(path)
    agent.actor_head.rand_move_eps = .1

    for step in tqdm(range(256), desc=f"Generating frames for run {run_number}"):
        (obs_tensor, action, log_prob, value, next_observation, reward, 
                next_done, next_hidden, infos) = agent.step_env()
        
        # Save environment image
        env_img = Image.fromarray(agent.envs.render()[0])
        env_img.save(os.path.join(img_output_dir, f'env_{step:04d}.png'))
        
        hidden_data = next_hidden.flatten().detach().cpu().numpy()
        
        # Calculate dimensions for a roughly square shape
        total_units = hidden_data.shape[0]
        width = int(np.sqrt(total_units))
        height = (total_units + width - 1) // width
        
        reshaped_data = np.pad(hidden_data.flatten(), 
                              (0, width * height - total_units),
                              mode='constant',
                              constant_values=np.nan)
        reshaped_data = reshaped_data.reshape(height, width)
        
        # Save raw data as JSON
        data_dict = {
            'step': step,
            'width': width,
            'height': height,
            'values': reshaped_data.tolist()
        }
        with open(os.path.join(data_output_dir, f'memory_{step:04d}.json'), 'w') as f:
            json.dump(data_dict, f)
        
        # Create and save memory heatmap
        fig = go.Figure(data=go.Heatmap(
            z=reshaped_data,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(
                thickness=30,  # Increased thickness
                outlinewidth=0,
            )
        ))
        
        fig.update_layout(
            width=400,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False
            )
        )
        
        fig.write_image(os.path.join(img_output_dir, f'memory_{step:04d}.png'))
        
        agent.observation = next_observation
        agent.hidden = next_hidden
        agent.done = next_done

        if next_done.any():
            break

    print(f"Run {run_number} completed. Frames and data saved to {base_output_dir}")
    return base_output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_runs', type=int, help='Number of parallel runs to execute')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run multiple instances in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(single_run, run_number, timestamp) 
            for run_number in range(args.num_runs)
        ]
        
        # Wait for all runs to complete
        output_dirs = [future.result() for future in futures]
    
    print("\nAll runs completed!")
    for dir in output_dirs:
        print(f"Output saved in: {dir}")

if __name__ == "__main__":
    main()