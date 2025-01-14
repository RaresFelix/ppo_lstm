import torch
import random
import time
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from jaxtyping import Int
from typing import Callable
from .config import Args
from .agent import Agent


#for now, assume discrete action space

def make_env(env_id: str, idx: Int, record_video: bool = False) -> Callable:
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if record_video and idx == 0:
            env = gym.wrappers.Monitor(env, f'videos/{env_id}')
        return env
    return thunk

def main() -> None:
    torch.set_float32_matmul_precision('high')
    args = Args()
    run_name = f'{args.project_name}_{args.env_id}_{int(time.time())}'
    writer = SummaryWriter('runs/' + run_name)
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i) for i in range(args.num_envs)]
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    agent = Agent(args, envs, writer)
    agent.train()

if __name__ == '__main__':
    main()
