import torch
import time
from env_all_actions import parallel_env
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
            all_rewards={1:-1,2:-1}
            for _ in range(self.games):
                env = parallel_env()
                obs, infos, action_count, action_list = env.reset()
                done = False

                while not done:
                    action=0
                    for agent in ["1"]:
                        obs_vec = torch.tensor(obs, dtype=torch.float32)
                        policy, value, h = self.model.initial_inference(obs_vec.unsqueeze(0).to(DEVICE))
                        if action_count>0:
                            policy=policy.detach().cpu()
                            mask = env.get_action_mask(action_list)
                            masked_policy = policy * torch.tensor(mask, device=policy.device)
                            masked_policy_sum=masked_policy.sum()
                            if (masked_policy_sum.abs()>1e-18).any():
                                masked_policy = masked_policy / masked_policy.sum()
                            masked_policy2=torch.clamp(masked_policy,1e-18,1e6)
                            action = int(torch.multinomial(masked_policy2, 1).detach().cpu())
                            if action not in action_list:
                                action=random.choice(action_list)
                        obs, rewards, terms, action_count,action_list,all_rewards,curr_eid = env.step(action)
                        done = terms

                if all_rewards[1] > all_rewards[2]:
                    wins += 1

            winrate = wins / self.games
            self.manager.log_metric("eval/winrate", winrate)

            time.sleep(120)  # evaluate every 5 minutes
