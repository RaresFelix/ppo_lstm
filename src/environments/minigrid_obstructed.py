import gymnasium as gym
import minigrid
from gymnasium import spaces

class MiniGridObstructedEnv(gym.Env):
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
        
        # Use full action space for ObstructedMaze as it requires more actions
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        
    def step(self, action):
        return self._env.step(action)
    
    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)
    
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()
