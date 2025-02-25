import gymnasium as gym
import numpy as np

class CartpoleNoVelWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        orig_space = env.observation_space  # expected shape (4,)
        # Create new observation space with only cart position (index 0) and pole angle (index 2)
        self.observation_space = gym.spaces.Box(
            low=np.array([orig_space.low[0], orig_space.low[2]]),
            high=np.array([orig_space.high[0], orig_space.high[2]]),
            dtype=orig_space.dtype
        )
        self.action_space = env.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        transformed_obs = self._transform_obs(obs)
        return transformed_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        transformed_obs = self._transform_obs(obs)
        return transformed_obs, info

    def _transform_obs(self, obs):
        return np.array([obs[0], obs[2]])

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def make_cartpole_no_vel(*args, **kwargs):
    env = gym.make('CartPole-v1', *args, **kwargs)
    return CartpoleNoVelWrapper(env)

# Register CartpoleNoVel environment
gym.envs.registration.register(
    id='CartPoleNoVel-v0',
    entry_point='src.environments.cartpole_no_vel:make_cartpole_no_vel'
)
