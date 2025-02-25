import sys
import random
import time
from typing import Callable
import wandb  
import tyro
from dataclasses import asdict
import signal

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Int
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent
from .config import Args
from ..environments.minigrid_custom_maze import MiniGridCustomMazeEnv
from ..environments import cartpole_no_vel  # ensure CartpoleNoVel is registered

def create_minigrid_env(args: Args, idx: int, run_name: str, record_video: bool) -> gym.Env:
    if args.use_pixels:
        from ..environments.minigrid_memory import MinigridMemoryEnv
        env = MinigridMemoryEnv(
            args.env_id,
            render_mode='rgb_array',
            agent_view_size=args.view_size,
            record_video=record_video and idx == 0,
            run_name=run_name
        )
    elif 'ObstructedMaze' in args.env_id:
        from ..environments.minigrid_obstructed import MiniGridObstructedEnv
        env = MiniGridObstructedEnv(
            args.env_id,
            args.one_hot,
            render_mode='rgb_array',
            agent_view_size=args.view_size,
            record_video=record_video and idx == 0,
            run_name=run_name
        )
    elif 'CustomMaze' in args.env_id:
        env = MiniGridCustomMazeEnv(
            args.env_id,  
            args.one_hot,
            render_mode='rgb_array',
            agent_view_size=args.view_size,
            record_video=record_video and idx == 0,
            run_name=run_name
        )
    else:
        from ..environments.minigrid_memory import MiniGridMemoryBasicEnv
        env = MiniGridMemoryBasicEnv(
            args.env_id,
            args.one_hot,
            render_mode='rgb_array',
            agent_view_size=args.view_size,
            record_video=record_video and idx == 0,
            run_name=run_name
        )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    #NORMALIZE REWARD
    env = gym.wrappers.NormalizeReward(env, .99)
    return env

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

def cleanup_handler(signum, frame):
    print("\nCleaning up resources...")
    if 'agent' in globals():
        agent.cleanup()
    if 'writer' in globals():
        writer.close()
    if 'wandb' in sys.modules and wandb.run is not None:
        wandb.finish()
    sys.exit(0)

def main() -> None:
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    torch.set_float32_matmul_precision('high')
    
    if any(arg.startswith('--sweep') for arg in sys.argv[1:]):
        sys.argv.remove('--sweep')
        args0 = tyro.cli(Args)
        wandb.init(
            project=args0.wandb_project,
            group=args0.wandb_group,
            monitor_gym=True,
        )
        args = Args.from_wandb_config(wandb.config)
    else:
        args = tyro.cli(Args)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    
    run_name = f'{args.project_name}_{args.env_id}_{args.view_size}x{args.view_size}_{int(time.time())}'
    
    if any(arg.startswith('--sweep') for arg in sys.argv[1:]):
        config_dict = vars(args)
        config_dict['run_name'] = run_name
        wandb.config.update(config_dict)
    elif args.use_wandb:
        config_dict = vars(args)
        config_dict['run_name'] = run_name
        if args.wandb_group:
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=run_name,
                config=config_dict,
                monitor_gym=True,
            )
        else:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=config_dict,
                monitor_gym=True,
            )
    
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
    
    # Create evaluation environments if eval_env_id is specified
    eval_envs = None
    if args.num_eval_envs > 0:
        eval_env_id = args.eval_env_id if args.eval_env_id else args.env_id
        temp_args = Args(**{**asdict(args), 'env_id': eval_env_id})
        eval_envs = gym.vector.AsyncVectorEnv(
            [make_env(temp_args, i, f"{run_name}_eval", False) for i in range(args.num_eval_envs)]
        )
    
    agent = Agent(args, envs, run_name, writer, eval_envs=eval_envs)
    
    try:
        agent.train()
    except Exception as e:
        print(f"Error during training: {e}")
        cleanup_handler(None, None)
    finally:
        cleanup_handler(None, None)

if __name__ == '__main__':
    main()
