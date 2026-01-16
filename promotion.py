import random
import torch
from env_all_actions import parallel_env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def play_match(agent_a, agent_b, device="cpu"):
    env = parallel_env()
    obs, _, action_count, action_list = env.reset()
    all_rewards={1:-1,2:-1}
    done = False
    while not done:
        action_map={}
        with torch.no_grad():
            for agent in ["1","2"]:
                action=0
                obs_vec = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
                if agent=="1":
                    policy, value, h = agent_a.initial_inference(obs_vec.unsqueeze(0).to(DEVICE))
                else:
                    policy, value, h = agent_b.initial_inference(obs_vec.unsqueeze(0).to(DEVICE))
                if len(action_list)>0:
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
                    # obs,rewards,terms,action_count,action_list,all_rewards,curr_eid
                    obs, rewards, terms, action_count,action_list,all_rewards,curr_eid = env.step(action)
                    done = terms

    if all_rewards[0] >= all_rewards[1]:
        return 1   # agent_a wins
    else:
        return 0
