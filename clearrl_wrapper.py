import numpy as np
import gymnasium as gym
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils.wrappers import BaseWrapper

class ClearRLWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = aec_to_parallel(env)
        self.num_agents = len(self.env.possible_agents)
        self.agents = self.env.possible_agents
        self.action_space = self.env.action_space(self.agents[0])
        self.observation_space = self.env.observation_space(self.agents[0])

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        dones = {agent: terminations[agent] or truncations[agent] for agent in self.agents}
        return obs, rewards, dones, infos

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    def get_spaces(self):
        return self.observation_space, self.action_space
