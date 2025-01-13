import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from jaxtyping import Int, Float, Array
from torch import Tensor
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Args:
    project_name: str = 'ppo_lstm'
    total_timesteps: int = int(1e6) 
    n_steps: int = 2048
    n_envs: int = 2
    minibatch_size: int = 64

    n_epochs: int = 0 # will be calculated
    buffer_size: int = int(1e5) 

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
        self.n_epochs = self.total_timesteps // (self.n_steps * self.n_envs)

#for now, assume discrete action space

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

        self.next_obs = torch.zeros((args.buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.next_dones = torch.zeros((args.buffer_size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((args.buffer_size,), dtype=torch.float32, device=device)
        self.hidden_states = torch.zeros((args.buffer_size, args.hidden_size), dtype=torch.float32, device=device)
        self.cell_states = torch.zeros((args.buffer_size, args.hidden_size), dtype=torch.float32, device=device)

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

class Agent():
    def __init__(self, args: Args, envs):
        self.args = args
        self.actor = Actor(envs.single_observation_space.shape[0], envs.single_action_space.n, args)
        self.critic = Critic(envs.single_observation_space.shape[0], args)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=args.lr)
        self.envs = envs