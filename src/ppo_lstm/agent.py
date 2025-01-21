import torch
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
from .networks import LSTMFeatureExtractor, ActorHead, CriticHead  # Update imports
from .buffer import RolloutBuffer
import os
from pathlib import Path
import wandb  # Add this import

class Agent():
    def __init__(self, args: Args, envs, run_name: str, writer: Optional[SummaryWriter] = None, device = None):
        self.args = args
        self.writer = writer
        self.run_name = run_name
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device

        self.obs_dim = envs.single_observation_space.shape
        self.act_dim = envs.single_action_space.n
        
        # Replace actor/critic with components
        self.feature_extractor = LSTMFeatureExtractor(self.obs_dim, args).to(device)
        self.actor_head = ActorHead(args.hidden_size, self.act_dim, args).to(device)
        self.critic_head = CriticHead(args.hidden_size, args).to(device)

        self.feature_extractor = torch.compile(self.feature_extractor)
        self.actor_head = torch.compile(self.actor_head)
        self.critic_head = torch.compile(self.critic_head)

        # Update optimizers
        self.feature_optimizer = optim.AdamW(self.feature_extractor.parameters(), lr=args.learning_rate, betas=args.betas)
        self.actor_optimizer = optim.AdamW(self.actor_head.parameters(), lr=args.learning_rate, betas=args.betas)
        self.critic_optimizer = optim.AdamW(self.critic_head.parameters(), lr=args.learning_rate, betas=args.betas)

        self.envs = envs

        self.buffer = RolloutBuffer(self.obs_dim, args)
        self.checkpoint_dir = Path(args.save_dir) / self.run_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
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

        if self.args.use_wandb:
            wandb.log({
                'episode/return': avg_returns,
                'episode/length': avg_lengths,
                'episode/terminals': number_terminals
            }, step=self.step)
    
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

    @torch.inference_mode()
    def rollout(self) -> None:
        self.feature_extractor.eval()
        self.actor_head.eval()
        self.critic_head.eval()

        if self.step == 0:
            observation = self.envs.reset()[0]
            done = np.zeros(self.args.num_envs)
            hidden = torch.zeros(2, self.args.num_envs, self.args.hidden_size, device=self.device)
        else:
            observation = self.last_observation
            done = self.last_done
            hidden = self.last_hidden

        self.buffer.reset()
        freq_terminal_log = 1
        cnt_terminal_log = 0
        for step in range(self.args.num_steps):
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            self.step += self.args.num_envs

            with torch.no_grad():
                features, next_hidden = self.feature_extractor.get_features(obs_tensor, hidden)
                action, log_prob = self.actor_head.get_action(features)
                value = self.critic_head(features)

            next_observation, reward, termination, truncation, infos = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(termination, truncation)

            if 'episode' in infos and self.writer and infos['_episode'].sum():
                cnt_terminal_log += 1
                if cnt_terminal_log % freq_terminal_log == 0:
                    self._log_episode_stats(infos)
            
            self.buffer.add(obs_tensor, done, action, log_prob, value, next_observation, next_done, reward, hidden)

            hidden = next_hidden

            for i in range(self.args.num_envs):
                if done[i] or next_done[i]:
                    hidden[0, i].zero_()
                    hidden[1, i].zero_()

            observation = next_observation
            done = next_done

        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        features, _ = self.feature_extractor.get_features(obs_tensor, hidden)
        next_value = self.critic_head(features)
        
        self.buffer.add_last(next_done, next_value, hidden)
        self.feature_extractor.train()
        self.actor_head.train()
        self.critic_head.train()

        self.last_observation = observation
        self.last_done = done
        self.last_hidden = hidden
            
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

        # Log to tensorboard
        for key, value in stats.items():
            self.writer.add_scalar(key, value, self.step)

        # Log to wandb
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
                    
                    # Apply gradients to all components
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