import pytest
import torch
import time
import numpy as np
import gymnasium as gym
import torch.nn.functional as F
from probes import Probe1, Probe2, Probe3, Probe4, Probe5
from ppo import Args, Agent, make_env
from torch.utils.tensorboard import SummaryWriter

def train_agent(env_id, total_timesteps, seed = 1, gamma = .2):
    """Helper function to train agent"""
    run_name = f'{env_id}__{int(time.time())}'
    writer = SummaryWriter('runs/debug/' + run_name, max_queue=100)
    args = Args(
        env_id=env_id,
        total_steps=total_timesteps,
        seed=seed,
        num_envs=1,
        gamma=gamma,
        gae_lambda=.95,
        clip_range=0.2,
        vf_coef=0.5,
        learning_rate=5e-3,
        ent_coef=0.01,
        max_grad_norm=0.5,
        num_steps=4,  
        minibatch_size=4,
        n_epochs=8,
        hidden_size=4,
        debug_probes=True,
    )
    envs = gym.vector.SyncVectorEnv([make_env(env_id, idx) for idx in range(args.num_envs)])
    agent = Agent(args, envs, writer)
    agent.train()
    return agent

def get_value(agent, obs):
    """Helper function to get value"""
    with torch.no_grad():
        obs = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        value, hidden = agent.critic(obs, torch.zeros((2, agent.args.hidden_size,), device=agent.device))

    return value

def test_probe1_learning(capsys):
    """Tests if ppo can learn probe1"""
    gym.envs.registration.register(id='Probe1-v0', entry_point=Probe1)

    with capsys.disabled():
        agent = train_agent('Probe1-v0', 200)
    
    obs = np.array([0.], dtype=np.float32)
    value = get_value(agent, obs)

    expected_value = 1.0
    tolerance = .02

    assert abs(value.item() - expected_value) < tolerance, \
        f'Q-value: {value.item()} is not within {tolerance * 100}% of {expected_value}'

def test_probe2_learning(capsys):
    """Tests if ppo can learn probe2"""
    gym.envs.registration.register(id='Probe2-v0', entry_point=Probe2)

    with capsys.disabled():
        agent = train_agent('Probe2-v0', 500)
    
    # Test both possible observations
    pos_obs = np.array([1.], dtype=np.float32)
    neg_obs = np.array([-1.], dtype=np.float32)
    
    pos_value = get_value(agent, pos_obs)
    neg_value = get_value(agent, neg_obs)

    tolerance = .02
    
    # Positive observation should predict reward of 1
    assert abs(pos_value.item() - 1.0) < tolerance, \
        f'Value for pos obs: {pos_value.item()} is not within {tolerance * 100}% of 1.0'
    
    # Negative observation should predict reward of -1
    assert abs(neg_value.item() - (-1.0)) < tolerance, \
        f'Value for neg obs: {neg_value.item()} is not within {tolerance * 100}% of -1.0'

def test_probe3_learning(capsys):
    """Tests if ppo can learn probe3 with correct discounted values"""
    gym.envs.registration.register(id='Probe3-v0', entry_point=Probe3)

    gamma = .2

    with capsys.disabled():
        agent = train_agent('Probe3-v0', 400, gamma=gamma)
    
    # Test both observations
    first_obs = np.array([0.], dtype=np.float32)
    second_obs = np.array([1.], dtype=np.float32)
    
    first_value = get_value(agent, first_obs)
    second_value = get_value(agent, second_obs)

    expected_first_value = gamma * 1.0  # Discounted future reward
    expected_second_value = 1.0  # Immediate reward
    tolerance = .02
    
    assert abs(first_value.item() - expected_first_value) < tolerance, \
        f'Value for first obs: {first_value.item()} is not within {tolerance * 100}% of {expected_first_value}'
    
    assert abs(second_value.item() - expected_second_value) < tolerance, \
        f'Value for second obs: {second_value.item()} is not within {tolerance * 100}% of {expected_second_value}'

def test_probe4_learning(capsys):
    """Tests if ppo can learn to select the better action in probe4"""
    gym.envs.registration.register(id='Probe4-v0', entry_point=Probe4)

    with capsys.disabled():
        agent = train_agent('Probe4-v0', 400)  # Needs more steps to learn policy
    
    # Get logits for the zero observation
    obs = np.array([0.], dtype=np.float32)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    
    with torch.no_grad():
        hidden = torch.zeros((2, agent.args.hidden_size,), device=agent.device)
        logits = agent.actor(obs_tensor, hidden)[0]
        probs = F.softmax(logits, dim=-1)

        values = agent.critic(obs_tensor, hidden)[0].squeeze().item()
    
    tolerance = .05
    expected_value = 1.

    assert abs(values - expected_value) < tolerance, \
        f'Value for zero obs: {values} is not within {tolerance * 100}% of {expected_value}'
    
    # Check if probability of taking action 0 (reward +1) is significantly higher
    assert probs[0] > 0.8, f'Agent failed to learn optimal policy. Probs: {probs}'

def test_probe5_learning(capsys):
    """Tests if ppo can learn to select correct action based on observation in probe5"""
    gym.envs.registration.register(id='Probe5-v0', entry_point=Probe5)

    with capsys.disabled():
        agent = train_agent('Probe5-v0', 400)  # Needs more steps to learn observation-dependent policy
    
    # Test both possible observations
    pos_obs = np.array([1.], dtype=np.float32)
    neg_obs = np.array([-1.], dtype=np.float32)
    
    hidden = torch.zeros((2, agent.args.hidden_size,), device=agent.device)
    obs_tensor_pos = torch.tensor(pos_obs, dtype=torch.float32, device=agent.device)
    obs_tensor_neg = torch.tensor(neg_obs, dtype=torch.float32, device=agent.device)
    
    with torch.no_grad():
        logits_pos = agent.actor(obs_tensor_pos, hidden)[0]
        logits_neg = agent.actor(obs_tensor_neg, hidden)[0]
        
        probs_pos = F.softmax(logits_pos, dim=-1)
        probs_neg = F.softmax(logits_neg, dim=-1)
        
        values_pos = agent.critic(obs_tensor_pos, hidden)[0].squeeze()
        values_neg = agent.critic(obs_tensor_neg, hidden)[0].squeeze()
    
    value_tolerance = 0.1
    prob_tolerance = 0.8  # Minimum probability for correct action
    expected_value = 1.0

    # Check values are close to expected reward of 1
    assert abs(values_pos - expected_value) < value_tolerance, \
        f'Value for positive obs: {values_pos} not within {value_tolerance} of {expected_value}'
    assert abs(values_neg - expected_value) < value_tolerance, \
        f'Value for negative obs: {values_neg} not within {value_tolerance} of {expected_value}'
    
    # For positive observation, action 1 should have high probability
    assert probs_pos[1] > prob_tolerance, \
        f'Agent failed to learn optimal policy for positive obs. Probs: {probs_pos}'
    
    # For negative observation, action 0 should have high probability
    assert probs_neg[0] > prob_tolerance, \
        f'Agent failed to learn optimal policy for negative obs. Probs: {probs_neg}'

