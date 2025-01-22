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

class LSTMFeatureExtractor(BaseEncoder):
    def __init__(self, obs_dim: Tuple[int, ...], args: Args):
        super().__init__(obs_dim)
        
        self.lstm = nn.LSTMCell(self.encoder_output_size, args.hidden_size)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
        
        self.args = args
    
    def get_features(self, 
                    x: Float[Tensor, "batch *obs_dim"],
                    hidden: Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]
                    ) -> Tuple[Float[Tensor, "batch hidden_size"],
                              Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]]:
        x = self.encode(x)
        hidden = self.lstm(x, (hidden[0], hidden[1]))
        return hidden[0], torch.stack(hidden)  # features, hidden
    
    def get_sequential_features(self,
                              obs: Float[Tensor, "batch seq_len *obs_dim"],
                              dones: Float[Tensor, "batch seq_len"],
                              hidden: Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]
                              ) -> Tuple[Float[Tensor, "batch seq_len hidden_size"],
                                       Tuple[Float[Tensor, "batch hidden_size"], Float[Tensor, "batch hidden_size"]]]:
        batch_size, seq_len, _ = obs.shape
        obs = self.encode(obs)
        
        features = torch.zeros((batch_size, seq_len, self.args.hidden_size), device=obs.device)
        for i in range(seq_len):
            hidden = self.lstm(obs[:, i], (hidden[0], hidden[1]))
            hidden = torch.stack(hidden)
            hidden *= (1 - dones[:, i].unsqueeze(0).unsqueeze(-1))
            features[:, i] = hidden[0]
        
        return features, hidden

class ActorHead(nn.Module):
    def __init__(self, feature_dim: int, act_dim: int, args: Args):
        super().__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(feature_dim, args.hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_layer_size, args.hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_layer_size, act_dim), std=.01)
        )
    
    def forward(self, features: Float[Tensor, "*batch hidden_size"]) -> Float[Tensor, "*batch act_dim"]:
        return self.fc(features)
    
    def get_action_dist(self, features: Float[Tensor, "*batch hidden_size"]):
        logits = self.forward(features)
        return torch.distributions.Categorical(logits=logits)
    
    def get_action(self, features: Float[Tensor, "batch hidden_size"]):
        dist = self.get_action_dist(features)
        act = dist.sample()
        log_prob = dist.log_prob(act)
        return act, log_prob
    
    def get_log_prob(self, features: Float[Tensor, "batch hidden_size"], actions: Float[Tensor, "batch"]) -> Float[Tensor, "batch"]:
        dist = self.get_action_dist(features)
        return dist.log_prob(actions)

    def evaluate_actions(self, features: Float[Tensor, "*batch hidden_size"], actions: Float[Tensor, "*batch"]):
        dist = self.get_action_dist(features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_seq_action_logprob_and_entropy(self,
                              features: Float[Tensor, "batch seq_len hidden_size"],
                              actions: Float[Tensor, "batch seq_len"]
                              ) -> Tuple[Float[Tensor, "batch seq_len"],
                                       Float[Tensor, "batch seq_len"]]:
        """Process sequence of features to evaluate actions"""
        dist = self.get_action_dist(features)  # works on any batch dimensions
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy

class CriticHead(nn.Module):
    def __init__(self, feature_dim: int, args: Args):
        super().__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(feature_dim, args.hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_layer_size, args.hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_layer_size, 1), std=1.)
        )
    
    def forward(self, features: Float[Tensor, "*batch hidden_size"]) -> Float[Tensor, "*batch"]:
        return self.fc(features).squeeze(-1)

    def get_seq_value(self,
                     features: Float[Tensor, "batch seq_len hidden_size"]
                     ) -> Float[Tensor, "batch seq_len"]:
        """Process sequence of features to get values"""
        return self.forward(features)  # forward already handles any batch dimensions
