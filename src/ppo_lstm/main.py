import random
import time
from typing import Callable

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
    args = Args(record_video=True)
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
    agent.train()

if __name__ == '__main__':
    main()
