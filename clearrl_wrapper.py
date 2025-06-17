import numpy as np
import gymnasium as gym
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils.wrappers import BaseWrapper

from env import TerraformingMarsEnv

class ClearRLWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = aec_to_parallel(env)
        
        self.agents = self.env.possible_agents
        #self.action_space = self.env.action_space(self.agents[0])
        #self.observation_space = self.env.observation_space(self.agents[0])

    @property
    def num_agents(self):
        return len(self.env.possible_agents)
    
    @num_agents.setter
    def num_agents(self, value):
        self.env=TerraformingMarsEnv([str(i) for i in range(value)])

    def observation_space(self,agent):
        return self.env.observation_spaces[agent]
    
    def action_space(self,agent):
            return self.env.action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        dones = {agent: terminations[agent] or truncations[agent] for agent in self.agents}
        return obs, rewards, dones, infos

    def render(self, render_mode="human"):
        self.env.render(render_mode=render_mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    def get_spaces(self):
        return self.observation_space, self.action_space


