import numpy as np
from env_all_actions import AllActionsEnv
import json 
import random
from decode_observation import decode_observation

env=AllActionsEnv(num_players=4,num_human_players=3,safe_mode=False)
obs,infos,action_count,action_list,current_env_id=env.reset()
MAX_ROWS=10000

#with open("")
rand=random.Random()
i=0
curr_eid=-1
rewards, terms,all_rewards=(None,None,None)
while True:
    is_no_actions=True
    if action_count>0:
        action=rand.choice(action_list)
        for eid in range(env.num_players):
            print(f"EID={eid}({curr_eid}) act_cnt={env.envs[eid].actions_count} actions={env.envs[eid].actions_list}")
        obs,rewards,terms,action_count,action_list,all_rewards,curr_eid=env.step(action)
        #env.current_obs,-1,True,0,[],{i:-1 for i in range(self.num_players)},self.last_env
        for e in env.envs:
            print()
        #obs_new=decode_observation(obs)
    elif terms:
        terms=False
        print(f"Done game rewards={all_rewards}")
        obs,infos,action_count,action_list,current_env=env.reset()
    else:
        print("Strange")
        raise Exception(f"Strange {action_count} {terms}")
        
    i+=1
    if i>=MAX_ROWS:
        print("Done")
        break

