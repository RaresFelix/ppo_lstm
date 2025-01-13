import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from jaxtyping import Int, Float, Array
from torch import Tensor
from typing import Tuple, Optional
from dataclasses import dataclass
import gymnasium as gym

@dataclass
class Args:
    project_name: str = 'ppo_lstm'
    env_id: str = 'CartPole-v1'
    torch_deterministic: bool = True
    total_timesteps: int = int(1e6) 
    seed: int = 0
    num_steps: int = 2048
    num_nenvs: int = 2
    minibatch_size: int = 64
    buffer_size: int = int(1e5) 

    n_epochs: int = 0 # will be calculated
    batch_size: int = 0

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    eps_max: float = 1e-3
    eps_min: float = 1e-5

    hidden_size: int = 64

    def __post_init__(self):
        self.n_epochs = self.total_timesteps // (self.num_steps * self.num_nenvs)
        self.batch_size = self.num_steps * self.num_nenvs

#for now, assume discrete action space

class RolloutBuffer():
    def __init__(self, obs_dim: Int, args: Args, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.args = args

        self.observations = torch.zeros((args.buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((args.buffer_size,), dtype=torch.float32, device=device)
        self.actions = torch.zeros((args.buffer_size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((args.buffer_size,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((args.buffer_size,), dtype=torch.float32, device=device)

        self.next_obs = torch.zeros((args.buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.next_dones = torch.zeros((args.buffer_size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((args.buffer_size,), dtype=torch.float32, device=device)

        # one for actor, one for critic
        self.hidden_states = torch.zeros((args.buffer_size, 2, args.hidden_size), dtype=torch.float32, device=device)
        self.cell_states = torch.zeros((args.buffer_size, 2, args.hidden_size), dtype=torch.float32, device=device)

        self.idx = 0
        self.full = False
    
    def add(self, obs, done, act, val, next_obs, next_done, rew, hidden, cell):
        self.observations[self.idx] = obs
        self.dones[self.idx] = done
        self.actions[self.idx] = act
        self.values[self.idx] = val

        self.next_obs[self.idx] = next_obs
        self.next_dones[self.idx] = next_done
        self.rewards[self.idx] = rew
        self.hidden_states[self.idx] = hidden
        self.cell_states[self.idx] = cell

        self.idx += 1
        if self.idx == self.args.buffer_size:
            self.idx = 0
            self.full = True

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def compute_gae(args, rewards: Float[Tensor, 'num_envs num_steps'], values, dones, next_dones, next_values: Float[Tensor, 'num_envs']):
    gamma = args.gamma
    gae_lambda = args.gae_lambda

    all_values = torch.cat([values, next_values.unsqueeze(1)], dim=1)
    all_dones = torch.cat([dones, next_dones.unsqueeze(1)], dim=1)
    # s0, a0, r0, ....
    all_values *= 1 - all_dones # Enforce V(s) = 0 if s is terminal
    deltas = rewards + gamma * all_values[:, 1:] * (1 - all_dones[:, :-1]) - all_values[:, :-1]
    advantages = deltas.clone()
    
    factors = gamma * gae_lambda * (1 - all_dones[:, :-1])
    for t in range(deltas.size(1) - 2, -1, -1):
        advantages[:, t] += factors[:, t] * advantages[:, t + 1]

    return advantages

class Actor(nn.Module):
    def __init__(self, obs_dim: Int, act_dim: Int, args: Args):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(obs_dim, args.hidden_size)
        self.fc = nn.Linear(args.hidden_size, act_dim)
        self.args = args
    
    def foward(self, x: Float[Tensor, 'batch obs_dim'], hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

class Critic(nn.Module):
    def __init__(self, obs_dim: Int, args: Args):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(obs_dim, args.hidden_size)
        self.fc = nn.Linear(args.hidden_size, 1)
    
    def forward(self, x: Float[Tensor, 'batch obs_dim'], hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x).squeeze(-1)
        return x, hidden

class Agent():
    def __init__(self, args: Args, envs, writer: Optional[SummaryWriter] = None, device = None):
        self.args = args
        self.writer = writer
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device

        self.obs_dim = np.prod(envs.single_observation_space.shape)
        self.act_dim = envs.single_action_space.n
        
        self.actor = Actor(self.obs_dim, self.act_dim, args).to(device)
        self.critic = Critic(self.obs_dim, args).to(device)

        self.envs = envs

        self.buffer = RolloutBuffer(self.obs_dim, args)

    def rollout(self):


        pass

    def train(self):
        pass

def make_env(env_id: str, idx: int, record_video: bool = False):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if record_video and idx == 0:
            env = gym.wrappers.Monitor(env, f'videos/{env_id}')
        return env
    return thunk

def main():
    args = Args()
    run_name = f'{args.project_name}_{args.env_id}_{int(time.time())}'
    writer = SummaryWriter('runs/' + run_name)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i) for i in range(args.num_nenvs)]
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    agent = Agent(args, envs, writer)
    agent.train()

if __name__ == 'main':
    main()
