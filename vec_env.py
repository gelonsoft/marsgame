import numpy as np

class VectorizedTMEnv:
    def __init__(self, make_env_fn, num_envs):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.agents = self.envs[0].possible_agents

    def reset(self):
        obses = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            obses.append(obs)
            infos.append(info)
        return obses, infos

    def step(self, actions_batch):
        next_obses, rewards, terms, truncs, infos = [], [], [], [], []

        for env, actions in zip(self.envs, actions_batch):
            o, r, t, tr, i = env.step(actions)
            next_obses.append(o)
            rewards.append(r)
            terms.append(t)
            truncs.append(tr)
            infos.append(i)

        return next_obses, rewards, terms, truncs, infos

    def get_action_masks(self):
        return [env.get_action_mask() for env in self.envs]
