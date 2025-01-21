import sys
import random
import time
from typing import Callable
import wandb  # Add this import
import tyro

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Int
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent
from .config import Args

def create_minigrid_env(args: Args, idx: int, run_name: str, record_video: bool) -> gym.Env:
    if args.use_pixels:
        from ..enviroments.minigrid_memory import MinigridMemoryEnv
        env = MinigridMemoryEnv(
            args.env_id,
            render_mode='rgb_array',
            agent_view_size=5,
            record_video=record_video and idx == 0,
            run_name=run_name
        )
    else:
        from ..enviroments.minigrid_memory import MiniGridMemoryBasicEnv
        env = MiniGridMemoryBasicEnv(
            args.env_id,
            args.one_hot,
            render_mode='rgb_array',
            agent_view_size=5,
            record_video=record_video and idx == 0,
            run_name=run_name
        )
    return gym.wrappers.RecordEpisodeStatistics(env)

def create_standard_env(args: Args, idx: int, run_name: str, record_video: bool) -> gym.Env:
    env = gym.make(args.env_id, render_mode='rgb_array')
    if args.use_pixels:
        env = gym.wrappers.AddRenderObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if record_video and idx == 0:
        env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
    return env

def make_env(args: Args, idx: Int, run_name: str, record_video: bool = False) -> Callable:
    def thunk():
        if 'MiniGrid' in args.env_id:
            return create_minigrid_env(args, idx, run_name, record_video)
        return create_standard_env(args, idx, run_name, record_video)
    return thunk

def main() -> None:
    torch.set_float32_matmul_precision('high')
    
    # Check if running as wandb sweep
    if any(arg.startswith('--sweep') for arg in sys.argv[1:]):
        # We're in a sweep, use wandb.config
        sys.argv.remove('--sweep')
        args0 = tyro.cli(Args)
        # Initialize wandb first
        wandb.init(
            project=args0.wandb_project,
            group=args0.wandb_group,
        )
        # Now we can safely access wandb.config
        args = Args.from_wandb_config(wandb.config)
        wandb.config.update(vars(args))
    else:
        # Normal CLI usage with tyro
        args = tyro.cli(Args)
        if args.use_wandb:
            if args.wandb_group:
                wandb.init(
                    project=args.wandb_project,
                    group=args.wandb_group,
                    config=vars(args),
                )
            else:
                wandb.init(
                    project=args.wandb_project,
                    config=vars(args),
                )
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    
    run_name = f'{args.project_name}_{args.env_id}_{args.view_size}x{args.view_size}_{int(time.time())}'
    
    writer = SummaryWriter('runs/' + run_name)
    writer.add_text(
        'Hyperparameters',
        '| Key | Value |\n---|---|\n' + '\n'.join([f'| {k} | {v} |' for k, v in vars(args).items()]),
        0
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, run_name, args.record_video) for i in range(args.num_envs)]
    )
    
    agent = Agent(args, envs, run_name, writer)
    
    try:
        agent.train()
    finally:
        if args.use_wandb:
            wandb.finish()
        writer.close()

if __name__ == '__main__':
    main()
