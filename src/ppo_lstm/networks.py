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

class BaseEncoder(nn.Module):
    def __init__(self, obs_dim: Tuple[int, ...]):
        super().__init__()
        print('The obs_dim is: ', obs_dim)
        self.is_cnn = len(obs_dim) > 1
        
        if self.is_cnn:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(obs_dim[0], 32, 8, 4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, 2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, 1)),
                nn.ReLU(),
                nn.Flatten()
            )
            dummy_input = torch.zeros((1, *obs_dim))
            dummy_output = self.encoder(dummy_input)
            self.encoder_output_size = dummy_output.shape[1]
        else:
            self.encoder = nn.Identity()
            self.encoder_output_size = obs_dim[0]
    
    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

class Actor(BaseEncoder):
    def __init__(self, obs_dim: Tuple[int, ...], act_dim: Int, args: Args):
        super().__init__(obs_dim)
        
        self.lstm = nn.LSTMCell(self.encoder_output_size, args.hidden_size)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)

        self.fc = nn.Sequential(
            layer_init(nn.Linear(args.hidden_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=.01)
        )
        self.args = args
    
    def forward(self,
                x: Float[Tensor, "batch *obs_dim"],
                hidden: Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]
                ) -> Tuple[Float[Tensor, "batch act_dim"], 
                          Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]]:
        x = self.encode(x)
        hidden = self.lstm(x, (hidden[0], hidden[1]))
        hidden = torch.stack(hidden)
        y = self.fc(hidden[0])
        return y, hidden
    
    def get_action(self,
                   x: Float[Tensor, "batch *obs_dim"],
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
                           obs: Float[Tensor, "batch *obs_dim"],
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


class Critic(BaseEncoder):
    def __init__(self, obs_dim: Tuple[int, ...], args: Args):
        super().__init__(obs_dim)
        
        self.lstm = nn.LSTMCell(self.encoder_output_size, args.hidden_size)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
                
        self.fc = nn.Sequential(
            layer_init(nn.Linear(args.hidden_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.)
        )
    
    def forward(self,
                x: Float[Tensor, "batch *obs_dim"],  # Changed to handle variable obs dimensions
                hidden: Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]] # or single tensor of shape (2 batch hidden_size)
                ) -> Tuple[Float[Tensor, "batch"],
                          Float[Tensor, "2 batch hidden_size"]]:
        x = self.encode(x)
        hidden = self.lstm(x, (hidden[0], hidden[1]))
        hidden = torch.stack(hidden)
        y = self.fc(hidden[0]).squeeze(-1)                                                                                                                                                                                                                      
        return y, hidden
