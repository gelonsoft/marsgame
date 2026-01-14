import torch
import time
from env import parallel_env
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Evaluator:
    def __init__(self, model, manager, games=10):
        self.model = model
        self.manager = manager
        self.games = games

    def run(self):
        while True:
            wins = 0

            for _ in range(self.games):
                env = parallel_env()
                obs, _ = env.reset()
                done = False

                while not done:
                    action_map={}
                    for agent in ["1"]:
                        
                        obs_vec = torch.tensor(obs[agent], dtype=torch.float32)
                        policy, value, h = self.model.initial_inference(obs_vec.unsqueeze(0).to(DEVICE))
                        acts=list(env.action_lookup[agent].keys())
                        if len(acts)==0:
                            action_map[agent]=0
                        else:
                            policy=policy.detach().cpu()
                            mask = env.get_action_mask(agent)
                            masked_policy = policy * torch.tensor(mask, device=policy.device)
                            masked_policy_sum=masked_policy.sum()
                            if (masked_policy_sum.abs()>1e-18).any():
                                masked_policy = masked_policy / masked_policy.sum()
                            masked_policy2=torch.clamp(masked_policy,1e-18,1e6)
                            action = int(torch.multinomial(masked_policy2, 1).detach().cpu())
                            if action not in acts:
                                action_map[agent]=random.choice(acts)
                            else:
                                action_map[agent]=action
                    for agent in ["2"]:
                        mask=env.get_action_mask(agent)
                        acts=list(env.action_lookup[agent].keys())
                        if len(acts)==0:
                            acts=[0]
                        action_map[agent]=random.choice(acts)
                    obs, rewards, terms, truncs, infos = env.step(action_map)
                    done = any(terms.values())

                if rewards["1"] > rewards["2"]:
                    wins += 1

            winrate = wins / self.games
            self.manager.log_metric("eval/winrate", winrate)

            time.sleep(300)  # evaluate every 5 minutes
