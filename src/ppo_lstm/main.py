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



def make_env(env_id: str, idx: Int, run_name: str, record_video: bool = False) -> Callable:
    def thunk():
        if 'MiniGrid' in env_id:
            env = gym.make("MiniGrid-MemoryS7-v0", agent_view_size=7, render_mode='rgb_array')
            #env = minigrid.wrappers.DictObservationSpaceWrapper(env)
            env = minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=28)
            env = minigrid.wrappers.ImgObsWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if record_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
        elif env_id == 'CartPole-v1-CNN':
            env = gym.make('CartPole-v1', render_mode='rgb_array')
            env = gym.wrappers.AddRenderObservation(env)
            if record_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
            env = gym.wrappers.RecordEpisodeStatistics(env)
        else:
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if record_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
        return env
    return thunk

def main() -> None:
    torch.set_float32_matmul_precision('high')
    args = Args(record_video=True)
    run_name = f'{args.project_name}_{args.env_id}_{int(time.time())}'
    writer = SummaryWriter('runs/' + run_name)

    #Let's log hyperparameters as text table in writer
    writer.add_text('Hyperparameters',
                    '| Key | Value |\n---|---|\n' + '\n'.join([f'| {k} | {v} |' for k, v in vars(args).items()]), 0)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, run_name, args.record_video) for i in range(args.num_envs)]
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    agent = Agent(args, envs, writer)
    agent.train()

if __name__ == '__main__':
    main()
