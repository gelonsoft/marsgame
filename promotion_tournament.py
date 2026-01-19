import torch
import random
import numpy as np
from env_all_actions import parallel_env



def play_five_player_game(models,replay, device="cpu"):
    env = parallel_env(num_players=5,num_human_players=5)
    obs,infos,action_count,action_list,current_env_id = env.reset()

    done = False
    p_rewards = {str(i): 0.0 for i in range(5)}

    while not done:
        actions = {}
        model_id=current_env_id
        model=models[model_id]
        
        agent_id = str(model_id)


        with torch.no_grad():
            obs_vec = torch.tensor(obs, dtype=torch.float32)
            policy, value, h = model.initial_inference(obs_vec.unsqueeze(0).to(device))
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
            obs, rewards, terms, action_count,action_list,all_rewards,current_env_id = env.step(action)
            next_vec = torch.tensor(obs, dtype=torch.float32)
            if replay:
                replay.add(obs_vec.cpu(), action, rewards, next_vec, done)

        for k in all_rewards:
            p_rewards[str(k)] += 0.0+rewards

        done = terms

    # Victory points ranking
    vp = {}
    for i in range(5):
        agent_id = str(i)
        vp[agent_id] = all_rewards[int(agent_id)]

    return all_rewards,p_rewards, vp
