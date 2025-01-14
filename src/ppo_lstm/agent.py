import torch
import random
import numpy as np
import time
import einops
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from jaxtyping import Float
from torch import Tensor
from typing import Optional
from .config import Args
from .networks import Actor, Critic
from .buffer import RolloutBuffer

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
    
    def _log_episode_stats(self, infos):
        """Log episode statistics to tensorboard.
        Args:
            infos: Info dict from environment containing episode stats
        """
        if not (self.writer and infos['_episode'].sum()):
            return
            
        avg_returns = infos['episode']['r'].sum() / infos['_episode'].sum()
        avg_lengths = infos['episode']['l'].sum() / infos['_episode'].sum()
        number_terminals = infos['_episode'].sum()

        self.writer.add_scalar('episode/return', avg_returns, self.step)
        self.writer.add_scalar('episode/length', avg_lengths, self.step)
        self.writer.add_scalar('episode/terminals', number_terminals, self.step)
    
    def _run_debug_probes(self):
        """Run debug value probes for critic network visualization"""
        with torch.no_grad():
            for obs_v in np.linspace(-1., 1., 5):
                obs_v_tens = torch.tensor(obs_v, dtype=torch.float32, device=self.device).unsqueeze(0)
                v_hidden = torch.zeros((2, self.args.hidden_size,), device=self.device)
                val = self.critic(obs_v_tens, v_hidden)[0].item()
                self.writer.add_scalar(f'probes/critic_value_{obs_v}', val, self.step)

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
                    self._log_episode_stats(infos)

            value, next_critic_hidden = self.critic(obs_tensor, critic_hidden)
            
            self.buffer.add(obs_tensor, done, action, log_prob, value, next_observation, next_done, reward, actor_hidden, critic_hidden)

            actor_hidden = next_actor_hidden
            critic_hidden = next_critic_hidden

            for i in range(self.args.num_envs):
                if done[i] or next_done[i]:
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
            
    def _log_training_stats(self, approx_kl, fraction_clipped, entropy, actor_loss, VF_loss, advantages, values):
        """Log training statistics to tensorboard.
        Args:
            approx_kl: Approximate KL divergence
            fraction_clipped: Fraction of clipped objectives
            entropy: Policy entropy
            actor_loss: Actor loss value
            VF_loss: Value function loss
            advantages: Mean advantages
            values: Mean values
        """
        if not self.writer:
            return
            
        self.writer.add_scalar('training/approx_kl', approx_kl, self.step)
        self.writer.add_scalar('training/clip_fraction', fraction_clipped, self.step)
        self.writer.add_scalar('training/entropy', entropy, self.step)
        self.writer.add_scalar('training/actor_loss', actor_loss, self.step)
        self.writer.add_scalar('training/critic_loss', VF_loss, self.step)
        self.writer.add_scalar('training/total_loss', actor_loss + VF_loss, self.step)
        self.writer.add_scalar('training/advantages', advantages.mean(), self.step)
        self.writer.add_scalar('training/values', values.mean(), self.step)

    def _get_critic_loss(self, observations, dones, values, advantages, critic_hidden):
        """Simulate LSTM and get BTT loss for critic."""
        returns = values + advantages
        pred_values = torch.zeros_like(returns)

        for step in range(len(observations)):
            obs_tensor = observations[step]
            current_value, critic_hidden = self.critic(obs_tensor, critic_hidden)
            pred_values[step] = current_value
        
        VF_deltas = (pred_values - returns) * (1 - dones)
        VF_loss = VF_deltas.pow(2).sum() / ((1 - dones).sum() + 1e-5)

        return VF_loss
    
    def _get_actor_loss(self, observations, actions, prev_logprob, dones, advantages, actor_hidden):
        """Simulate LSTM and get BTT loss for actor."""

        log_probs = torch.zeros_like(advantages)
        entropy = torch.zeros_like(advantages)
        for step in range(len(observations)):
            obs_tensor = observations[step]
            act_tensor = actions[step]
            log_probs[step], entropy[step], actor_hidden = self.actor.get_action_logprob_and_entropy(obs_tensor, act_tensor, actor_hidden)
        
        log_ratio = log_probs - prev_logprob
        ratio = torch.exp(log_ratio)

        approx_kl = ((ratio - 1) - log_ratio).mean()
        
        with torch.no_grad():
            is_clipped = (ratio - 1).abs() > self.args.clip_range
            fraction_clipped = is_clipped.float().mean()
        
        clipped_ratio = ratio.clamp(1 - self.args.clip_range, 1 + self.args.clip_range)

        surr1 = advantages * ratio
        surr2 = advantages * clipped_ratio
        actor_loss = -torch.min(surr1, surr2).mean()

        return actor_loss, entropy.mean(), approx_kl, fraction_clipped
        

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
                self._run_debug_probes()

            log_on_this_step = (step % 1 == 0)

            early_stop = False
            freq_train_log = self.args.num_minibatches * self.args.update_epochs // 4
            for epoch in range(self.args.update_epochs):
                if early_stop:
                    break

                length = rollout_batch[0].size(0)
                with torch.no_grad():
                    observations, dones, values, actions, log_probs, advantages, rewards, next_obs, next_dones, actor_hidden, critic_hidden = rollout_batch

                # Critic loss
                VF_loss = self._get_critic_loss(observations, dones, values, advantages, critic_hidden[0])

                # Actor loss

                actor_loss, entropy, approx_kl, fraction_clipped = self._get_actor_loss(observations, actions, log_probs, dones, advantages, actor_hidden[0])
                if self.writer and (random.randint(0, freq_train_log) == 0) and log_on_this_step:
                    self._log_training_stats(approx_kl, fraction_clipped, entropy, actor_loss, VF_loss, advantages, values)

                total_actor_loss = actor_loss + self.args.entropy_coef * entropy 
                
                if approx_kl > self.args.max_kl:
                    early_stop = True
                    break
                
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                VF_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()
            train_time = time.time() - time0
            print(f'Rollout time: {rollout_time:.2f}s, Train time: {train_time:.2f}s')