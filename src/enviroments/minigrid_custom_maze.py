import gymnasium as gym
import minigrid
import numpy as np
from gymnasium import spaces
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Key, Wall, Goal
from abc import abstractmethod
import re
from minigrid.core.constants import (
    COLOR_NAMES, 
    DIR_TO_VEC,
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    STATE_TO_IDX
)

class DisjointSetUnion:
    def __init__(self, n):
        self.e = [-1] * n
    
    def parent(self, u):
        while self.e[u] >= 0:
            u = self.e[u]
        return u
    
    def join(self, u, v):
        u, v = self.parent(u), self.parent(v)
        if u == v:
            return
        if self.e[u] > self.e[v]:
            u, v = v, u
        self.e[u] += self.e[v]
        self.e[v] = u
    
    def same(self, u, v):
        return self.parent(u) == self.parent(v)


class MiniGridCustomMaze(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if size > 13:
            print("WARNING: max_steps(128) is not enough for large grid size, consider increasing max_steps")
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            see_through_walls=True,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return ''
    
    @abstractmethod
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Generate a tree of free places with kruskal
        def _gen_maze_kruskal(w, h):
            is_free = np.zeros((w, h), dtype=bool)
            for x in range(0, w, 2):
                for y in range(0, h, 2):
                    is_free[x, y] = True
            walls = [(x, y) for x in range(w) for y in range(h) if (x % 2) + (y % 2) == 1]
            DSU = DisjointSetUnion(w * h)
            while walls:
                x, y = walls.pop(self._rand_int(0, len(walls))) 
                if is_free[x, y]:
                    continue
                #check if all free neighbors are in different sets
                neighbors = [(x + dx, y + dy) for dx, dy in DIR_TO_VEC if 0 <= x + dx < w and 0 <= y + dy < h]
                neighbors = [(nx, ny) for nx, ny in neighbors if is_free[nx, ny]]
                neighbors_sets = set(DSU.parent(nx + ny * w) for nx, ny in neighbors)
                ok = len(neighbors_sets) == len(neighbors)

                if ok:
                    is_free[x, y] = True
                    for nx, ny in neighbors:
                        DSU.join(x + y * w, nx + ny * w)
            return is_free
        
        inside_maze_free = _gen_maze_kruskal(width - 2, height - 2)
       # UF_free = DisjointSetUnion(width * height)
       # for x in range(1, width - 1):
       #     for y in range(1, height - 1):
       #         if inside_maze_free[x - 1, y - 1]:
       #             for dx, dy in DIR_TO_VEC:
       #                 nx, ny = x + dx, y + dy
       #                 if nx < 1 or nx >= width - 1 or ny < 1 or ny >= height - 1:
       #                     continue
       #                 if inside_maze_free[nx - 1, ny - 1]:
       #                     UF_free.join(x + y * width, nx + ny * width)
        free_cells = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1) if inside_maze_free[x - 1, y - 1]]

        # Place the agent and goal
        self.agent_pos = free_cells.pop(self._rand_int(0, len(free_cells)))
        self.agent_dir = 0
        self.goal_pos = free_cells.pop(self._rand_int(0, len(free_cells)))
        self.grid.set(*self.goal_pos, Goal())

     #   free_parents_set = set(UF_free.parent(x + y * width) for x, y in free_cells)
     #   if len(free_parents_set) > 1:
     #       print(f"free_parents_set: {len(free_parents_set)}")
        #assert UF_free.same(self.agent_pos[0] + self.agent_pos[1] * width, self.goal_pos[0] + self.goal_pos[1] * width)

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if not inside_maze_free[x - 1, y - 1]:
                    self.grid.set(x, y, Wall())
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info 
    def render(self):
        return super().render()

class MiniGridCustomMazeRandom(MiniGridCustomMaze):
    def __init__(
        self,
        max_size=8,
        min_size=6,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.max_size = max_size
        self.min_size = min_size
        size = self._rand_int(min_size, max_size + 1)
        self.width = size
        self.height = size
        super().__init__(
            size=size,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            max_steps=max_steps,
            **kwargs,
        )
    
    def reset(self, seed=None, options=None):
        # Choose new random size before reset
        size = self._rand_int(self.min_size, self.max_size + 1)
        self.width = size
        self.height = size
        self.grid_size = size
        self.grid = None
        return super().reset(seed=seed, options=options)

class MiniGridCustomMazeEnv(gym.Env):
    def __init__(self, env_id: str, use_one_hot: bool, render_mode='rgb_array', agent_view_size=3, record_video=False, run_name=None):
        super().__init__()
        
        # Parse size from env_id using regex, supporting both fixed and random sizes
        # e.g., 'CustomMazeS8', 'CustomMazeRandomS8'
        size_match = re.search(r'CustomMaze(?:Random)?S(\d+)', env_id)
        size = int(size_match.group(1)) if size_match else 8
        
        is_random = 'Random' in env_id
        
        full_obs = False
        if agent_view_size == 0:
            full_obs = True
            agent_view_size = 7

        # Use gym.make() instead of direct instantiation
        self._env = gym.make(
            f'MiniGrid-CustomMaze{"Random" if is_random else ""}-S{size}-v0',
            render_mode=render_mode,
            agent_view_size=agent_view_size
        )

        if full_obs:
            self._env = minigrid.wrappers.FullyObsWrapper(self._env)
        self._env = gym.wrappers.FilterObservation(self._env, ['image', 'direction'])
        if use_one_hot:
            self._env = minigrid.wrappers.OneHotPartialObsWrapper(self._env)
        self._env = gym.wrappers.FlattenObservation(self._env)
        
        if record_video and run_name is not None:
            self._env = gym.wrappers.RecordVideo(self._env, f'videos/{run_name}')
        
        self.action_space = spaces.Discrete(3)  # Only use first 3 actions like other envs
        self.observation_space = self._env.observation_space
        
    def step(self, action):
        return self._env.step(action)
    
    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)
    
    def render(self):
        return self._env.render()

from gymnasium.envs.registration import register

for i in range(4, 25):
    register(
        id=f'MiniGrid-CustomMaze-S{i}-v0',
        entry_point='src.enviroments.minigrid_custom_maze:MiniGridCustomMaze',
        kwargs={
            'size': i,
        }
    )

for i in range(4, 25):
    register(
        id=f'MiniGrid-CustomMazeRandom-S{i}-v0',
        entry_point='src.enviroments.minigrid_custom_maze:MiniGridCustomMazeRandom',
        kwargs={
            'max_size': i,
            'min_size': 6,
        }
    )