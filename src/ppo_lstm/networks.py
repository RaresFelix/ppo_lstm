import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from jaxtyping import Int, Float
from torch import Tensor
from .config import Args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim: Int, act_dim: Int, args: Args):
        super(Actor, self).__init__()
        self.lstm = nn.LSTMCell(obs_dim, args.hidden_size)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
        #self.fc = nn.Linear(args.hidden_size, act_dim)
        self.fc = nn.Sequential(
            layer_init(nn.Linear(args.hidden_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=.01)
        )
        self.args = args
    
    def forward(self,
                x: Float[Tensor, "batch obs_dim"],
                hidden: Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]
                ) -> Tuple[Float[Tensor, "batch act_dim"], 
                          Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]]:
        #lst expects (batch, seq, features) but here seq = 1
        hidden = self.lstm(x, (hidden[0], hidden[1]))
        hidden = torch.stack(hidden)
        y = self.fc(hidden[0])
        return y, hidden
    
    def get_action(self,
                   x: Float[Tensor, "batch obs_dim"],
                   hidden: Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]
                   ) -> Tuple[Float[Tensor, "batch"],
                            Float[Tensor, "batch"],
                            Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]]:
        y, hidden = self(x, hidden)
        dist = torch.distributions.Categorical(logits=y)
        act = dist.sample()
        log_prob = dist.log_prob(act)
        return act, log_prob, hidden
    
    def get_action_logprob_and_entropy(self, 
                           obs: Float[Tensor, "batch obs_dim"],
                           act: Float[Tensor, "batch act_dim"],
                           hidden: Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]
                           ) -> Tuple[Float[Tensor, "batch"],
                                    Float[Tensor, "batch"],
                                    Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]]:
        y, hidden = self(obs, hidden)
        dist = torch.distributions.Categorical(logits=y)
        log_prob = dist.log_prob(act)
        entropy = dist.entropy()
        return log_prob, entropy, hidden


class Critic(nn.Module):
    def __init__(self, obs_dim: Int, args: Args):
        super(Critic, self).__init__()
        self.lstm = nn.LSTMCell(obs_dim, args.hidden_size)
        #self.fc = nn.Linear(args.hidden_size, 1)
        self.fc = nn.Sequential(
            layer_init(nn.Linear(args.hidden_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.)
        )
    
    def forward(self,
                x: Float[Tensor, "batch obs_dim"],
                hidden: Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]
                ) -> Tuple[Float[Tensor, "batch"],
                          Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]]:
        hidden = self.lstm(x, (hidden[0], hidden[1]))
        hidden = torch.stack(hidden)
        y = self.fc(hidden[0]).squeeze(-1)                                                                                                                                                                                                                      
        return y, hidden
