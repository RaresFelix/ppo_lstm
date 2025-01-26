import os
import sys
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

path = '/workspace/2501/ppo_lstm/important_checkpoints/CustomMazeRandomS15_1737796730/checkpoint_31457280.pt'
env_id = 'MiniGrid-CustomMazeS13-v0'

num_envs = 1
args = Args(
    env_id=env_id,
    seq_len=64,
    num_envs = num_envs,
    hidden_layer_size=256,
    hidden_size=32,
    view_size=3,
    deployment=True,
)
run_name = 'test'
env = gym.vector.SyncVectorEnv([make_env(args, seed, run_name) for seed in range(1)])
obs0, _ = env.reset()
agent = Agent(args, envs = env, run_name = run_name)
agent.load(path)
img = Image.fromarray(agent.envs.render()[0])
agent.actor_head.rand_move_eps = .1

# Create output directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join('visualisations', 'images', f'run_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

for step in tqdm(range(40), desc="Generating frames"):
    (obs_tensor, action, log_prob, value, next_observation, reward, 
            next_done, next_hidden, infos) = agent.step_env()
    if next_done.any():
        break
    
    # Save environment image
    env_img = Image.fromarray(agent.envs.render()[0])
    env_img.save(os.path.join(output_dir, f'env_{step:04d}.png'))
    
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
    
    fig.write_image(os.path.join(output_dir, f'memory_{step:04d}.png'))
    
    agent.observation = next_observation
    agent.hidden = next_hidden
    agent.done = next_done

print(f"Frames saved to {output_dir}")