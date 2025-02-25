import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
import random

class Probe1(gym.Env):
    """
    Constant observation. +1 reward. One timestep long
    Tests if agent can learn constant value function
    """

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0], dtype=np.float32),
                                      np.array([1e-9], dtype=np.float32)) # not 0 just to supress warning
        self.action_space = Discrete(1)
    
    def step(self, action: int):
        return (np.array([0.], dtype=np.float32), 1, True, False, {})

    def reset(self, seed: int | None = None, options = None) -> tuple[np.array, int]:
        super().reset(seed=seed)
        return np.array([0.], dtype=np.float32), {}
    
    def seed(self, seed: int | None = None) -> list[int]:
        super().reset(seed=seed)
        return [seed] if seed is not None else []

class Probe2(gym.Env):
    """
    +-1 Observation. Reward corresponding to observation. One timestep long
    Tests if agent can learn simple value function
    """

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([-1], dtype=np.float32),
                                      np.array([1], dtype=np.float32))
        self.action_space = Discrete(1)
    
    def step(self, action: int):
        return (np.array([0.], dtype=np.float32), self.reward, True, False, {}) # Go to 0, get reward
    
    def reset(self, seed: int | None = None, options = None) -> tuple[np.array, int]:
        super().reset(seed=seed)
        np.random.seed(seed)
        self.reward = np.random.choice([-1, 1])
        return np.array([self.reward], dtype=np.float32), {}
    
    def seed(self, seed: int | None = None) -> list[int]:
        super().reset(seed=seed)
        return [seed] if seed is not None else []

class Probe3(gym.Env):
    """
    Two timesteps long, different observations(0, then 1). Reward on second observations
    Checks time discounting
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = gym.spaces.Discrete(1)
        self.current_step = 0
        
    def reset(self, seed: int | None = None, options = None) -> tuple[np.array, int]:
        super().reset(seed=seed)
        self.current_step = 0
        return np.array([0.], dtype=np.float32), {}
        
    def step(self, action):
        reward = 0
        if self.current_step == 0:
            self.current_step += 1
            return np.array([1.], dtype=np.float32), 0, False, False, {}
        else:
            return np.array([1.], dtype=np.float32), 1, True, False, {}
    
    def seed(self, seed: int | None = None) -> list[int]:
        super().reset(seed=seed)
        return [seed] if seed is not None else []

class Probe4(gym.Env):
    """
    Zero observation. Two actions with +1/-1 rewards. One timestep long.
    Tests if agent can learn to select the better action.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0], dtype=np.float32),
                                   np.array([1e-9], dtype=np.float32))  # not 0 just to suppress warning
        self.action_space = Discrete(2)
    
    def step(self, action: int):
        reward = 1 if action == 0 else -1
        return np.array([0.], dtype=np.float32), reward, True, False, {}
    
    def reset(self, seed: int | None = None, options = None) -> tuple[np.array, int]:
        super().reset(seed=seed)
        return np.array([0.], dtype=np.float32), {}
    
    def seed(self, seed: int | None = None) -> list[int]:
        super().reset(seed=seed)
        return [seed] if seed is not None else []

class Probe5(gym.Env):
    """
    +-1 observation. Reward if action == observation. One timestep long.
    Tests if agent can learns policy
    """

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([-1], dtype=np.float32),
                                      np.array([1], dtype=np.float32))
        self.action_space = Discrete(2)
    
    def step(self, action: int):
        reward = 0.
        if action == 1 and self.observation == 1:
            reward = 1.
        elif action == 0 and self.observation == -1:
            reward = 1.
        return np.array([0.], dtype=np.float32), reward, True, False, {}
    
    def reset(self, seed: int | None = None, options = None) -> tuple[np.array, int]:
        super().reset(seed=seed)
        self.observation = np.random.choice([-1, 1])
        return np.array([self.observation], dtype=np.float32), {}
    
    def seed(self, seed: int | None = None) -> list[int]:
        super().reset(seed=seed)
        return [seed] if seed is not None else []

