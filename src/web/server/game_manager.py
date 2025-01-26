from src.environments.minigrid_custom_maze import MiniGridCustomMazeEnv, MiniGridCustomMaze
import minigrid
import numpy as np

class GameManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, session_id, agent_view=False):
        if agent_view:
            # Use simple CustomMaze with partial observation for agent view
            env = MiniGridCustomMaze(size=13, agent_view_size=3)
            env = minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=32)
        else:
            # Use CustomMazeEnv for full view
            env = MiniGridCustomMazeEnv(
                env_id='CustomMazeS13',
                use_one_hot=False,
                render_mode='rgb_array'
            )
        
        obs, _ = env.reset()
        render = obs['image'] if agent_view else env.render()
        
        self.sessions[session_id] = {
            'env': env,
            'current_obs': obs,
            'render': render,
            'agent_view': agent_view
        }
        return render.tolist()

    def get_render(self, session_id):
        if session_id in self.sessions:
            return self.sessions[session_id]['render'].tolist()
        return None

    def perform_action(self, session_id, action):
        if session_id not in self.sessions:
            return None
            
        session_data = self.sessions[session_id]
        obs, reward, terminated, truncated, _ = session_data['env'].step(action)
        
        if terminated or truncated:
            obs, _ = session_data['env'].reset()
            
        render = obs['image'] if session_data['agent_view'] else session_data['env'].render()
        session_data['current_obs'] = obs
        session_data['render'] = render
        
        return render.tolist()

    def toggle_view(self, session_id, agent_view):
        """Completely regenerate the environment with the new view mode"""
        if session_id in self.sessions:
            # Clean up old environment if needed
            if 'env' in self.sessions[session_id]:
                self.sessions[session_id]['env'].close()
                
            # Delete the old session and create a new one
            del self.sessions[session_id]
            return self.create_session(session_id, agent_view)
        return None
