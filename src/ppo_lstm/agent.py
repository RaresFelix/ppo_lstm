import torch
import gymnasium as gym
import random
import numpy as np
import time
import einops
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from jaxtyping import Float
from torch import Tensor
from typing import Optional
from .config import Args
from .networks import LSTMFeatureExtractor, ActorHead, CriticHead 
from .buffer import RolloutBuffer
import os
from pathlib import Path
import wandb  

class Agent():
    def __init__(self, args: Args, envs, run_name: str, writer: Optional[SummaryWriter] = None, device = None, eval_envs: Optional[gym.vector.VectorEnv] = None):
        self.args = args
        self.writer = writer
        self.run_name = run_name
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device

        self.obs_dim = envs.single_observation_space.shape
        self.act_dim = envs.single_action_space.n
        
        self.feature_extractor = LSTMFeatureExtractor(self.obs_dim, args).to(device)
        self.actor_head = ActorHead(args.hidden_size, self.act_dim, args).to(device)
        self.critic_head = CriticHead(args.hidden_size, args).to(device)

#        self.feature_extractor = torch.compile(self.feature_extractor)
#        self.actor_head = torch.compile(self.actor_head)
#        self.critic_head = torch.compile(self.critic_head)

        self.feature_optimizer = optim.AdamW(self.feature_extractor.parameters(), lr=args.learning_rate, betas=args.betas)
        self.actor_optimizer = optim.AdamW(self.actor_head.parameters(), lr=args.learning_rate, betas=args.betas)
        self.critic_optimizer = optim.AdamW(self.critic_head.parameters(), lr=args.learning_rate, betas=args.betas)

        self.envs = envs
        self.eval_envs = eval_envs

        self.buffer = RolloutBuffer(self.obs_dim, args)
        self.checkpoint_dir = Path(args.save_dir) / self.run_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.ema_return = None
        self.observation = None
        self.done = None
        self.hidden = None
    
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

        # Update EMA return
        if self.ema_return is None:
            self.ema_return = avg_returns
        else:
            self.ema_return = self.args.ema_decay * self.ema_return + (1 - self.args.ema_decay) * avg_returns

        stats = {
            'episode/return': avg_returns,
            'episode/return_ema': self.ema_return,
            'episode/length': avg_lengths,
            'episode/terminals': number_terminals
        }

        for key, value in stats.items():
            self.writer.add_scalar(key, value, self.step)

        if self.args.use_wandb:
            wandb.log(stats, step=self.step)
    
    def _run_debug_probes(self):
        """Run debug value probes for critic network visualization"""
        if not self.writer:
            return
        with torch.no_grad():
            for obs_v in np.linspace(-1., 1., 5):
                obs_v_tens = torch.tensor(obs_v, dtype=torch.float32, device=self.device).unsqueeze(0)
                v_hidden = torch.zeros((2, self.args.hidden_size,), device=self.device)
                features, _ = self.feature_extractor.get_features(obs_v_tens, v_hidden)
                val = self.critic_head(features).item()
                self.writer.add_scalar(f'probes/critic_value_{obs_v}', val, self.step)

    def step_env(self):
        """
        Take one step in the environment using the current policy.
        Returns:
            Tuple containing (observation, action, log_prob, value, next_observation, 
                            reward, next_done, next_hidden, info)
        """
        if self.observation is None:
            self.observation = self.envs.reset()[0]
            self.done = np.zeros(self.args.num_envs)
            self.hidden = torch.zeros(2, self.args.num_envs, self.args.hidden_size, device=self.device)

        obs_tensor = torch.tensor(self.observation, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            features, next_hidden = self.feature_extractor.get_features(obs_tensor, self.hidden)
            action, log_prob = self.actor_head.get_action(features)
            value = self.critic_head(features)

        next_observation, reward, termination, truncation, infos = self.envs.step(action.cpu().numpy())
        next_done = np.logical_or(termination, truncation)

        # Reset hidden state for done environments
        for i in range(self.args.num_envs):
            if self.done[i] or next_done[i]:
                next_hidden[0, i].zero_()
                next_hidden[1, i].zero_()

        return (obs_tensor, action, log_prob, value, next_observation, reward, 
                next_done, next_hidden, infos)

    @torch.inference_mode()
    def rollout(self) -> None:
        self.feature_extractor.eval()
        self.actor_head.eval()
        self.critic_head.eval()

        self.buffer.reset()
        freq_terminal_log = 1
        cnt_terminal_log = 0
        
        for step in range(self.args.num_steps):
            self.step += self.args.num_envs
            
            (obs_tensor, action, log_prob, value, next_observation, reward, 
             next_done, next_hidden, infos) = self.step_env()

            if 'episode' in infos and self.writer and infos['_episode'].sum():
                cnt_terminal_log += 1
                if cnt_terminal_log % freq_terminal_log == 0:
                    self._log_episode_stats(infos)
            
            self.buffer.add(obs_tensor, self.done, action, log_prob, value, 
                          next_observation, next_done, reward, self.hidden)

            self.observation = next_observation
            self.done = next_done
            self.hidden = next_hidden

        # Get final value estimate
        obs_tensor = torch.tensor(self.observation, dtype=torch.float32, device=self.device)
        features, _ = self.feature_extractor.get_features(obs_tensor, self.hidden)
        next_value = self.critic_head(features)
        
        self.buffer.add_last(self.done, next_value, self.hidden)
        
        self.feature_extractor.train()
        self.actor_head.train()
        self.critic_head.train()

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
            
        stats = {
            'training/approx_kl': approx_kl,
            'training/clip_fraction': fraction_clipped,
            'training/entropy': entropy,
            'training/actor_loss': actor_loss,
            'training/critic_loss': VF_loss,
            'training/total_loss': actor_loss + VF_loss,
            'training/advantages': advantages.mean(),
            'training/values': values.mean()
        }

        for key, value in stats.items():
            self.writer.add_scalar(key, value, self.step)

        if self.args.use_wandb:
            wandb.log(stats, step=self.step)

    def _get_critic_loss(self, features: Float[Tensor, "batch seq_len hidden_size"], 
                        dones: Float[Tensor, "batch seq_len"], 
                        values: Float[Tensor, "batch seq_len"], 
                        advantages: Float[Tensor, "batch seq_len"]) -> Float[Tensor, ""]:
        """Get critic loss from pre-calculated features."""
        returns = values + advantages
        pred_values = self.critic_head.get_seq_value(features)
     
        VF_deltas = (pred_values - returns) * (1 - dones)
        VF_loss = VF_deltas.pow(2).sum() / ((1 - dones).sum() + 1e-5)

        return VF_loss
        
    def _get_actor_loss(self, features: Float[Tensor, "batch seq_len hidden_size"],
                       actions: Float[Tensor, "batch seq_len"],
                       prev_logprob: Float[Tensor, "batch seq_len"],
                       dones: Float[Tensor, "batch seq_len"],
                       advantages: Float[Tensor, "batch seq_len"]) -> Tuple[Float[Tensor, ""], ...]:
        """Get actor loss from pre-calculated features."""
        log_probs, entropy = self.actor_head.get_seq_action_logprob_and_entropy(features, actions)
        
        log_ratio = log_probs - prev_logprob
        ratio = torch.exp(log_ratio)
        
        approx_kl = ((ratio - 1) - log_ratio).mean()
        
        with torch.no_grad():
            is_clipped = (ratio - 1).abs() > self.args.clip_range
            fraction_clipped = is_clipped.float().mean()
        
        clipped_ratio = ratio.clamp(1 - self.args.clip_range, 1 + self.args.clip_range)
        surr1 = advantages * ratio
        surr2 = advantages * clipped_ratio
        
        valid_steps = (1 - dones)
        actor_loss = -(torch.min(surr1, surr2) * valid_steps).sum() / (valid_steps.sum() + 1e-5)
        entropy_mean = (entropy * valid_steps).sum() / (valid_steps.sum() + 1e-5)

        return actor_loss, entropy_mean, approx_kl, fraction_clipped

    def save(self, step: int) -> None:
        """Save model checkpoint"""
        save_path = self.checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save({
            'step': step,
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'actor_head_state_dict': self.actor_head.state_dict(),
            'critic_head_state_dict': self.critic_head.state_dict(),
            'feature_optimizer_state_dict': self.feature_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, save_path)
    
    def load(self, path: str) -> int:
        """Load model checkpoint and return the step number"""
        checkpoint = torch.load(path)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.actor_head.load_state_dict(checkpoint['actor_head_state_dict'])
        self.critic_head.load_state_dict(checkpoint['critic_head_state_dict'])
        self.feature_optimizer.load_state_dict(checkpoint['feature_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        return checkpoint['step']

    @torch.inference_mode()
    def evaluate(self, global_step: int):
        if not self.eval_envs:
            return
            
        self.feature_extractor.eval()
        self.actor_head.eval()
        self.critic_head.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        obs = self.eval_envs.reset()[0]
        hidden = torch.zeros(2, self.args.num_eval_envs, self.args.hidden_size, device=self.device)
        
        while len(episode_rewards) < self.args.num_eval_episodes:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            features, next_hidden = self.feature_extractor.get_features(obs_tensor, hidden)
            action, _ = self.actor_head.get_action(features)
            
            next_obs, reward, terminated, truncated, info = self.eval_envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            if "final_info" in info:
                for item in info["final_info"]:
                    if item is not None:
                        episode_rewards.append(item["episode"]["r"])
                        episode_lengths.append(item["episode"]["l"])
            
            # Reset hidden state for done environments
            for i in range(self.args.num_eval_envs):
                if done[i]:
                    next_hidden[:, i].zero_()
            
            obs = next_obs
            hidden = next_hidden
            
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        self.writer.add_scalar("eval/mean_reward", mean_reward, global_step)
        self.writer.add_scalar("eval/mean_episode_length", mean_length, global_step)
        
        if self.args.use_wandb:
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/mean_episode_length": mean_length
            }, step=global_step)
            
        self.feature_extractor.train()
        self.actor_head.train()
        self.critic_head.train()
        
    def train(self) -> None:
        progress_bar = tqdm(range(self.args.total_steps))
        self.step = 0

        wandb.watch(self.feature_extractor, log = 'all')
        wandb.watch(self.actor_head, log = 'all') 
        wandb.watch(self.critic_head, log = 'all')

        print('num_mini_batches', self.args.num_minibatches)
        for step in range(self.args.num_iterations):
            progress_bar.update(self.args.batch_size)
            progress_bar.set_description(f'Step {step} / {self.args.num_iterations}')

            time0 = time.time()
            self.rollout()
            rollout_time = time.time() - time0
            time0 = time.time()

            rollout_batch = self.buffer.get_batch() # each tensor here has shape (batch_size, seq_len, ...)

            if self.args.debug_probes:
                self._run_debug_probes()

            log_on_this_step = (step % 1 == 0)

            early_stop = False
            freq_train_log = self.args.num_minibatches * self.args.update_epochs // 4
            for epoch in range(self.args.update_epochs):
                if early_stop:
                    break

                for i in range(self.args.num_minibatches):
                    minibatch = [x[i*self.args.minibatch_size:(i+1)*self.args.minibatch_size] for x in rollout_batch]
                    with torch.no_grad():
                        observations, dones, values, actions, log_probs, advantages, rewards, next_obs, next_dones, hidden = minibatch

                    # Calculate features once for both actor and critic
                    # hidden : batch seq_len c2 hidden
                    hidden = hidden[:, 0] # take the first hidden state
                    hidden = einops.rearrange(hidden, 'batch c2 hidden -> c2 batch hidden', c2=2)
                    features, _ = self.feature_extractor.get_sequential_features(observations, dones, hidden)

                    # Critic loss
                    VF_loss = self._get_critic_loss(features, dones, values, advantages)

                    # Actor loss
                    actor_loss, entropy, approx_kl, fraction_clipped = self._get_actor_loss(
                        features, actions, log_probs, dones, advantages)
                    
                    if self.writer and (random.randint(0, freq_train_log) == 0) and log_on_this_step:
                        self._log_training_stats(approx_kl, fraction_clipped, entropy, actor_loss, VF_loss, advantages, values)

                    total_actor_loss = actor_loss - self.args.entropy_coef * entropy 
                    total_loss = VF_loss + total_actor_loss

                    self.feature_optimizer.zero_grad()
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    
                    total_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.actor_head.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic_head.parameters(), self.args.max_grad_norm)
                    
                    self.feature_optimizer.step()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

            train_time = time.time() - time0
            print(f'Rollout time: {rollout_time:.2f}s, Train time: {train_time:.2f}s')
            if self.step % self.args.save_freq == 0:
                self.save(self.step)
            if self.args.debug_probes:
                self._run_debug_probes()
            
            # Add evaluation using step directly
            if self.args.eval_freq > 0 and step % self.args.eval_freq == 0:
                self.evaluate(self.step)