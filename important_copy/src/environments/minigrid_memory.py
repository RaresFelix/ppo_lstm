import gymnasium as gym
import minigrid
import numpy as np
from gymnasium import spaces

class MinigridMemoryEnv(gym.Env):
    def __init__(self, env_id: str, render_mode='rgb_array', agent_view_size=3, record_video=False, run_name=None):
        super().__init__()
        
        # Create and wrap the environment
        self._env = gym.make(env_id, agent_view_size=agent_view_size, render_mode=render_mode)
        self._env = minigrid.wrappers.RGBImgPartialObsWrapper(self._env, tile_size=28)
        self._env = minigrid.wrappers.ImgObsWrapper(self._env)
        
        if record_video and run_name is not None:
            self._env = gym.wrappers.RecordVideo(self._env, f'videos/{run_name}')
        
        # Transform the observation space to match the transformed observations
        obs_shape = self._env.observation_space.shape  # Expected (H, W, C)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),  # Transform to (C, H, W)
            dtype=np.uint8
        )
        
        # Other spaces remain the same
        self.action_space = spaces.Discrete(3)
        
    def _transform_observation(self, obs):
        return np.transpose(obs, (2, 0, 1))
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        transformed_obs = self._transform_observation(obs)
        return transformed_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        transformed_obs = self._transform_observation(obs)
        return transformed_obs, info
    
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()

class MiniGridMemoryBasicEnv(gym.Env):
    def __init__(self, env_id: str, use_one_hot: bool, render_mode='rgb_array', agent_view_size=3, record_video=False, run_name=None):
        super().__init__()
        
        full_obs = False
        if agent_view_size == 0:
            full_obs = True
            agent_view_size = 7

        self._env = gym.make(env_id, agent_view_size=agent_view_size, render_mode=render_mode)
        if full_obs:
            self._env = minigrid.wrappers.FullyObsWrapper(self._env)
        if use_one_hot:
            self._env = minigrid.wrappers.OneHotPartialObsWrapper(self._env)
        self._env = gym.wrappers.FilterObservation(self._env, ['image', 'direction'])
        self._env = gym.wrappers.FlattenObservation(self._env)
        
        if record_video and run_name is not None:
            self._env = gym.wrappers.RecordVideo(self._env, f'videos/{run_name}')
        
        # Limit action space to first 3 actions
        self.action_space = spaces.Discrete(3)
        self.observation_space = self._env.observation_space
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)
    
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()