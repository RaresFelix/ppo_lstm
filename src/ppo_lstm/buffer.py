import torch
from jaxtyping import Int, Float
import einops
from torch import Tensor
from typing import List, Tuple
from .config import Args

def compute_gae(args: Args,
                rewards: Float[Tensor, "num_steps num_envs"],
                values: Float[Tensor, "num_steps num_envs"],
                dones: Float[Tensor, "num_steps num_envs"],
                last_dones: Float[Tensor, "num_envs"],
                last_values: Float[Tensor, "num_envs"]) -> Float[Tensor, "num_steps num_envs"]:
    gamma = args.gamma
    gae_lambda = args.gae_lambda

    all_values = torch.cat([values, last_values.unsqueeze(0)], dim=0)
    all_dones = torch.cat([dones, last_dones.unsqueeze(0)], dim=0)
    # s0, a0, r0, ....
    all_values *= 1 - all_dones # Enforce V(s) = 0 if s is terminal
    deltas = rewards + gamma * all_values[1:] * (1 - all_dones[:-1]) - all_values[:-1]
    advantages = deltas.clone()
    
    factors = gamma * gae_lambda * (1 - all_dones[:-1])
    for t in range(deltas.size(0) - 2, -1, -1):
        advantages[t] += factors[t] * advantages[t + 1]

    return advantages


def reshape_sequence_data(tensor: Tensor, sequence_length: int, is_lstm_hidden: bool = False) -> Tensor:
    """
    Reshapes a tensor from (timesteps, environments, ...) to (batch, sequence_length, ...).
    Handles both regular tensors and LSTM hidden states differently.
    
    Args:
        tensor: Input tensor of shape:
            - Regular: (timesteps, environments, ...) or (timesteps, environments)
            - LSTM hidden: (timesteps, 2, environments, hidden_size)
        sequence_length: Length of each sequence in the output
        is_lstm_hidden: Whether the tensor is an LSTM hidden state
        
    Returns:
        Reshaped tensor of shape:
            - Regular: (batch, sequence_length, ...) where batch = num_environments * num_sequences
            - LSTM hidden: (batch, sequence_length, 2, hidden_size)
    """
    # Calculate number of complete sequences
    num_timesteps = tensor.shape[0]
    num_sequences_per_env = num_timesteps // sequence_length
    num_valid_timesteps = num_sequences_per_env * sequence_length
    
    # Truncate to complete sequences
    tensor = tensor[:num_valid_timesteps]
    
    if is_lstm_hidden:
        # LSTM hidden states have shape (timesteps, 2, environments, hidden_size)
        pattern = '(sequences sequence_length) lstm_states environments hidden_size -> (environments sequences) sequence_length lstm_states hidden_size'
        return einops.rearrange(
            tensor,
            pattern,
            sequence_length=sequence_length,
            sequences=num_sequences_per_env,
            lstm_states=2
        )
    else:
        # Regular tensors have shape (timesteps, environments, ...)
        if len(tensor.shape) > 2:  # Has additional dimensions
            pattern = '(sequences sequence_length) environments ... -> (environments sequences) sequence_length ...'
        else:  # Just (timesteps, environments)
            pattern = '(sequences sequence_length) environments -> (environments sequences) sequence_length'
            
        return einops.rearrange(
            tensor,
            pattern,
            sequence_length=sequence_length,
            sequences=num_sequences_per_env
        )

class RolloutBuffer():
    def __init__(self, obs_dim: Tuple[int, ...], args: Args, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.args = args

        self.observations = torch.zeros((args.buffer_size, args.num_envs, *obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.actions = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.values = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.log_prob = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)

        self.next_obs = torch.zeros((args.buffer_size, args.num_envs, *obs_dim), dtype=torch.float32, device=device)
        self.next_dones = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)

        self.hidden = torch.zeros((args.buffer_size, 2, args.num_envs, args.hidden_size), dtype=torch.float32, device=device)

        self.idx = 0
        self.full = False
    
    def add(self, 
            obs: Float[Tensor, "num_envs *obs_dim"],
            done: Float[Tensor, "num_envs"],
            act: Float[Tensor, "num_envs"],
            log_prob: Float[Tensor, "num_envs"],
            val: Float[Tensor, "num_envs"],
            next_obs: Float[Tensor, "num_envs *obs_dim"],
            next_done: Float[Tensor, "num_envs"],
            rew: Float[Tensor, "num_envs"],
            hidden: Float[Tensor, "2 num_envs hidden_size"]) -> None:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        log_prob = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        val = torch.as_tensor(val, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        next_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        hidden = torch.as_tensor(hidden, dtype=torch.float32, device=self.device)
        
        self.observations[self.idx] = obs
        self.dones[self.idx] = done
        self.actions[self.idx] = act
        self.values[self.idx] = val
        self.log_prob[self.idx] = log_prob

        self.next_obs[self.idx] = next_obs
        self.next_dones[self.idx] = next_done
        self.rewards[self.idx] = rew
        self.hidden[self.idx] = hidden

        self.idx += 1
        if self.idx == self.args.buffer_size:
            self.idx = 0
            self.full = True

    def add_last(self, 
                next_done: Float[Tensor, "num_envs"],
                next_value: Float[Tensor, "num_envs"],
                last_hidden: Float[Tensor, "2 num_envs hidden_size"]) -> None:
        self.last_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device) 
        self.last_value = torch.as_tensor(next_value, dtype=torch.float32, device=self.device)
        self.last_hidden = torch.as_tensor(last_hidden, dtype=torch.float32, device=self.device)
    
    def reset(self):
        self.idx = 0
        self.full = False

    def get_batch(self) -> List[Float[Tensor, "..."]]:
        with torch.no_grad():
            if self.full:
                self.idx = len(self.observations)
            if self.idx % self.args.seq_len != 0:
                self.idx = self.idx // self.args.seq_len * self.args.seq_len  # cut off incomplete sequence
                
            self.advantages = compute_gae(
                self.args,
                self.rewards[:self.idx],
                self.values[:self.idx],
                self.dones[:self.idx],
                self.last_done,
                self.last_value
            )
            
            # Regular tensors
            regular_tensors = [
                self.observations[:self.idx],
                self.dones[:self.idx],
                self.values[:self.idx],
                self.actions[:self.idx],
                self.log_prob[:self.idx],
                self.advantages[:self.idx],
                self.rewards[:self.idx],
                self.next_obs[:self.idx],
                self.next_dones[:self.idx]
            ]
            
            # LSTM hidden states
            lstm_tensors = [
                self.hidden[:self.idx],  # Single hidden state repeated twice to maintain compatibility
            ]
            
            # Process regular tensors and LSTM hidden states separately
            result = [
                reshape_sequence_data(tensor, self.args.seq_len)
                for tensor in regular_tensors
            ] + [
                reshape_sequence_data(tensor, self.args.seq_len, is_lstm_hidden=True)
                for tensor in lstm_tensors
            ]

            self.reset()
                
            return result