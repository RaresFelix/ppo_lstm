import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
import einops
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from jaxtyping import Int, Float, Array
from torch import Tensor
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import gymnasium as gym

@dataclass
class Args:
    project_name: str = 'ppo_lstm'
    env_id: str = 'CartPole-v1'
    torch_deterministic: bool = True
    total_steps: int = int(1e6) 
    seed: int = 0
    num_steps: int = 1024
    num_envs: int = 1
    minibatch_size: int = 256
    buffer_size: int = int(1e5) 
    debug_probes: bool = False

    n_epochs: int = 0 # will be calculated
    batch_size: int = 0
    num_iterations: int = 0
    num_minibatches: int = 0

    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 2.5e-4
    eps_max: float = 1e-3
    eps_min: float = 1e-5

    hidden_size: int = 64

    def __post_init__(self):
        self.n_epochs = self.total_steps // (self.num_steps * self.num_envs)
        self.batch_size = self.num_steps * self.num_envs
        self.num_iterations = self.total_steps // self.batch_size
        self.num_minibatches = self.batch_size // (self.minibatch_size * self.num_envs)

#for now, assume discrete action space

class RolloutBuffer():
    def __init__(self, obs_dim: Int, args: Args, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.args = args

        self.observations = torch.zeros((args.buffer_size, args.num_envs, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.actions = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.values = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.log_prob = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)

        self.next_obs = torch.zeros((args.buffer_size, args.num_envs, obs_dim), dtype=torch.float32, device=device)
        self.next_dones = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((args.buffer_size, args.num_envs,), dtype=torch.float32, device=device)

        self.actor_hidden = torch.zeros((args.buffer_size, 2, args.num_envs, args.hidden_size), dtype=torch.float32, device=device)
        self.critic_hidden = torch.zeros((args.buffer_size, 2, args.num_envs, args.hidden_size), dtype=torch.float32, device=device)

        self.idx = 0
        self.full = False
    
    def add(self, 
            obs: Float[Tensor, "num_envs obs_dim"],
            done: Float[Tensor, "num_envs"],
            act: Float[Tensor, "num_envs"],
            log_prob: Float[Tensor, "num_envs"],
            val: Float[Tensor, "num_envs"],
            next_obs: Float[Tensor, "num_envs obs_dim"],
            next_done: Float[Tensor, "num_envs"],
            rew: Float[Tensor, "num_envs"],
            actor_hidden: Float[Tensor, "2 num_envs hidden_size"],
            critic_hidden: Float[Tensor, "2 num_envs hidden_size"]) -> None:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        log_prob = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        val = torch.as_tensor(val, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        next_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        actor_hidden = torch.as_tensor(actor_hidden, dtype=torch.float32, device=self.device)
        critic_hidden = torch.as_tensor(critic_hidden, dtype=torch.float32, device=self.device)
        
        self.observations[self.idx] = obs
        self.dones[self.idx] = done
        self.actions[self.idx] = act
        self.values[self.idx] = val
        self.log_prob[self.idx] = log_prob

        self.next_obs[self.idx] = next_obs
        self.next_dones[self.idx] = next_done
        self.rewards[self.idx] = rew
        self.actor_hidden[self.idx] = actor_hidden
        self.critic_hidden[self.idx] = critic_hidden

        self.idx += 1
        if self.idx == self.args.buffer_size:
            self.idx = 0
            self.full = True

    def add_last(self, 
                next_done: Float[Tensor, "num_envs"],
                next_value: Float[Tensor, "num_envs"],
                last_critic_hidden: Float[Tensor, "2 num_envs hidden_size"]) -> None:
        #this is for gae calculation
        self.last_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device) 
        self.last_value = torch.as_tensor(next_value, dtype=torch.float32, device=self.device)
        self.last_critic_hidden = torch.as_tensor(last_critic_hidden, dtype=torch.float32, device=self.device)
    
    def reset(self):
        self.idx = 0
        self.full = False
    
    def get_batch(self) -> list[Float[Tensor, "..."]]:
        with torch.no_grad():
            if self.full:
                self.advantages = compute_gae(self.args, self.rewards, self.values, self.dones, self.last_done, self.last_value)
                result = [self.observations, self.dones, self.values, self.actions, self.log_prob, self.advantages, self.rewards, self.next_obs, self.next_dones, self.actor_hidden, self.critic_hidden]
                return result
            else:
                self.advantages = compute_gae(self.args, self.rewards[:self.idx], self.values[:self.idx], self.dones[:self.idx], self.last_done, self.last_value)
                result = [self.observations[:self.idx], self.dones[:self.idx], self.values[:self.idx], self.actions[:self.idx], self.log_prob[:self.idx], self.advantages[:self.idx], self.rewards[:self.idx], self.next_obs[:self.idx], self.next_dones[:self.idx], self.actor_hidden[:self.idx], self.critic_hidden[:self.idx]]
                return result
                

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

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
    

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        self.envs = envs

        self.buffer = RolloutBuffer(self.obs_dim, args)

    @torch.inference_mode()
    def rollout(self) -> None:
        self.critic.eval()
        self.actor.eval()

        if self.step == 0:
            observation = self.envs.reset()[0]
            done = np.zeros(self.args.num_envs)
            actor_hidden = torch.zeros(2, self.args.num_envs, self.args.hidden_size, device=self.device)
            critic_hidden = torch.zeros(2, self.args.num_envs, self.args.hidden_size, device=self.device)
        else:
            observation = self.last_observation
            done = self.last_done
            actor_hidden = self.last_actor_hidden
            critic_hidden = self.last_critic_hidden

        self.buffer.reset()
        freq_terminal_log = 2
        cnt_terminal_log = 0

        for step in range(self.args.num_steps):
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            self.step += self.args.num_envs

            with torch.no_grad():
                action, log_prob, next_actor_hidden = self.actor.get_action(obs_tensor, actor_hidden)

            next_observation, reward, termination, truncation, infos = self.envs.step(action.cpu().numpy())

            next_done = np.logical_or(termination, truncation)

            if 'episode' in infos and self.writer and infos['_episode'].sum():
                cnt_terminal_log += 1
                if cnt_terminal_log % freq_terminal_log == 0:
                    avg_returns = infos['episode']['r'].sum() / infos['_episode'].sum()
                    avg_lengths = infos['episode']['l'].sum() / infos['_episode'].sum()
                    number_terminals = infos['_episode'].sum()

                    self.writer.add_scalar('episode/return', avg_returns, self.step)
                    self.writer.add_scalar('episode/length', avg_lengths, self.step)
                    self.writer.add_scalar('episode/terminals', number_terminals, self.step)

            value, next_critic_hidden = self.critic(obs_tensor, critic_hidden)
            
            self.buffer.add(obs_tensor, done, action, log_prob, value, next_observation, next_done, reward, actor_hidden, critic_hidden)

            actor_hidden = next_actor_hidden
            critic_hidden = next_critic_hidden

            for i in range(self.args.num_envs):
                #if done[i] or next_done[i]:
                actor_hidden[0, i].zero_()
                actor_hidden[1, i].zero_()
                critic_hidden[0, i].zero_()
                critic_hidden[1, i].zero_()

            observation = next_observation
            done = next_done

        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        next_value, last_critic_hidden = self.critic(obs_tensor, critic_hidden)
        
        self.buffer.add_last(next_done, next_value, last_critic_hidden)
        self.actor.train()
        self.critic.train()

        self.last_observation = observation
        self.last_done = done
        self.last_actor_hidden = actor_hidden
        self.last_critic_hidden = critic_hidden
            

    def train(self) -> None:
        progress_bar = tqdm(range(self.args.total_steps))
        self.step = 0

        print('num_mini_batches', self.args.num_minibatches)
        for step in range(self.args.num_iterations):
            progress_bar.update(self.args.batch_size)
            progress_bar.set_description(f'Step {step} / {self.args.num_iterations}')

            time0 = time.time()
            self.rollout()
            rollout_time = time.time() - time0
            time0 = time.time()

            rollout_batch = self.buffer.get_batch()

            if self.args.debug_probes:
                with torch.no_grad():
                    for obs_v in np.linspace(-1., 1., 5):
                        obs_v_tens = torch.tensor(obs_v, dtype=torch.float32, device=self.device).unsqueeze(0)
                        v_hidden = torch.zeros((2, self.args.hidden_size,), device=self.device)
                        val = self.critic(obs_v_tens, v_hidden)[0].item()

                        self.writer.add_scalar(f'probes/critic_value_{obs_v}', val, self.step)

            log_on_this_step = (step % 1 == 0)

            for epoch in range(self.args.update_epochs):
                length = rollout_batch[0].size(0)
                idx = torch.randperm(length, device = self.device)
                for i in range(self.args.num_minibatches):
                    mini_idx = idx[i * self.args.minibatch_size: (i + 1) * self.args.minibatch_size]
                    minibatch = [x[mini_idx] for x in rollout_batch]

                    with torch.no_grad():
                        observations, dones, values, actions, log_probs, advantages, rewards, next_obs, next_dones, actor_hidden, critic_hidden = minibatch

                    # Critic loss
                    
                    # critic accpts only one batch size, so we flatten
                    obs_view = observations.view(-1, observations.shape[-1])
                    #critic_hidden_view = critic_hidden.view(2, -1, critic_hidden.shape[-1])
                    critic_hidden_view = einops.rearrange(critic_hidden, 'batch n2 num_envs hidden_size -> n2 (batch num_envs) hidden_size')
                    values_pred_view, _ = self.critic(obs_view, critic_hidden_view)

                    values_pred = values_pred_view.view(-1, self.args.num_envs) # back to (batch_size, num_envs)

                    with torch.no_grad():
                        returns = values + advantages
                    VF_deltas = (values_pred - returns) * (1 - dones)
                    VF_loss = VF_deltas.pow(2).sum() / ((1 - dones).sum() + 1e-5)

                    # Actor loss
                    actor_hidden_view = einops.rearrange(actor_hidden, 'batch n2 num_envs hidden_size -> n2 (batch num_envs) hidden_size')
                    action_logits_view, _ = self.actor(obs_view, actor_hidden_view)
                    action_logits = einops.rearrange(action_logits_view, '(batch num_envs) act_dim -> batch num_envs act_dim', batch=self.args.minibatch_size)
                    dist = torch.distributions.Categorical(logits=action_logits)
                    entropy = dist.entropy().mean()
                    new_log_probs = dist.log_prob(actions)

                    log_ratio = new_log_probs - log_probs
                    ratio = torch.exp(log_ratio)

                    approx_kl = ((ratio - 1) - log_ratio).mean()

                    clipped_ratio = ratio.clamp(1 - self.args.clip_range, 1 + self.args.clip_range)
                    with torch.no_grad():
                        is_clipped = (ratio - 1).abs() > self.args.clip_range
                        fraction_clipped = is_clipped.float().mean()
                    
                    surr1 = advantages * ratio
                    surr2 = advantages * clipped_ratio
                    actor_loss = -torch.min(surr1, surr2).mean()

                    if self.writer and i == 0 and epoch == 0 and log_on_this_step:
                        self.writer.add_scalar('training/approx_kl', approx_kl, self.step)
                        self.writer.add_scalar('training/clip_fraction', fraction_clipped, self.step)
                        self.writer.add_scalar('training/entropy', entropy, self.step)
                        self.writer.add_scalar('training/actor_loss', actor_loss, self.step)
                        self.writer.add_scalar('training/critic_loss', VF_loss, self.step)
                        self.writer.add_scalar('training/total_loss', actor_loss + VF_loss, self.step)
                        self.writer.add_scalar('training/advantages', advantages.mean(), self.step)
                        self.writer.add_scalar('training/values', values.mean(), self.step)
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    VF_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                    self.critic_optimizer.step()
            train_time = time.time() - time0
            print(f'Rollout time: {rollout_time:.2f}s, Train time: {train_time:.2f}s')

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
