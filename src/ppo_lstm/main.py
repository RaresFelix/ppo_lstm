import torch
import random
import minigrid
import time
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from jaxtyping import Int
from typing import Callable
from .config import Args
from .agent import Agent

#for now, assume discrete action space

def make_env(args: Args, idx: Int, run_name: str, record_video: bool = False) -> Callable:
    def thunk():
        if 'MiniGrid' in args.env_id:
            if 'video' in args.env_id:
                from ..enviroments.minigrid_memory import MinigridMemoryEnv
                env = MinigridMemoryEnv(
                    args.env_id,
                    render_mode='rgb_array', 
                    agent_view_size=5,
                    record_video=record_video and idx == 0,
                    run_name=run_name
                )
                env = gym.wrappers.RecordEpisodeStatistics(env)
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
                env = gym.wrappers.RecordEpisodeStatistics(env)
        elif args.env_id == 'CartPole-v1-CNN':
            env = gym.make('CartPole-v1', render_mode='rgb_array')
            env = gym.wrappers.AddRenderObservation(env)
            if record_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
            env = gym.wrappers.RecordEpisodeStatistics(env)
        else:
            env = gym.make(args.env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if record_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
        return env
    return thunk

def main() -> None:
    torch.set_float32_matmul_precision('high')
    args = Args(record_video=True)
    run_name = f'{args.project_name}_{args.env_id}_{args.view_size}x{args.view_size}_{int(time.time())}'
    writer = SummaryWriter('runs/' + run_name)

    #Let's log hyperparameters as text table in writer
    writer.add_text('Hyperparameters',
                    '| Key | Value |\n---|---|\n' + '\n'.join([f'| {k} | {v} |' for k, v in vars(args).items()]), 0)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, run_name, args.record_video) for i in range(args.num_envs)]
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    agent = Agent(args, envs, run_name, writer)
    agent.train()

if __name__ == '__main__':
    main()
