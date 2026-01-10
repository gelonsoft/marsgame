import numpy as np
from env import SERVER_BASE_URL, TerraformingMarsEnv
import json 
import random
import pyarrow as pa

#player_state=""""""
#waiting_for={}
#player_id="pbc07ea787b11"
#player_state=json.loads(player_state)
#waiting_for=None
#NON_STOP=True
#env=TerraformingMarsEnv(["1","2"],init_from_player_state=True,player_state=player_state if not player_id else None,player_id=player_id,waiting_for=waiting_for)
#print(env.player_states['1'].get('waitingFor'))
#i=0
#for id,action in env.action_lookup['1'].items():
#    if i>30:
#        break
#    i=i+1
#    print(f"Action id: {id}, action: {json.dumps(action)}")
#    
#if NON_STOP:



env=TerraformingMarsEnv(["1","2"])
MAX_ROWS=10000
num_actions=env.observation_space(env.possible_agents[0]).shape[0]

#with open("")
rand=random.Random()
i=0
next_obs, rewards, terms, truncs, infos=(None,None,None,None,None)
while True:
    actions={}
    is_no_actions=True
    for agent in env.action_lookup:
        max_actions=len(env.action_lookup[agent].keys())+1
        
        if max_actions>1:
            actions[agent]=rand.randint(0,max_actions-1)
            is_no_actions=False
        elif max_actions==1:
            actions[agent]=0
        else:
            is_no_actions=False
            actions[agent]=0
        #print(f"F {agent}:{actions[agent]}/{max_actions}")
    if terms and terms.get('1',False):
        terms=None
        env=TerraformingMarsEnv(["1","2"])
    else:
        next_obs, rewards, terms, truncs, infos=env.step(actions)
        #result[i]=next_obs['1']
        i+=1
        if i>=MAX_ROWS:
            print("Done")
            break
        #result[i]=next_obs['2']
        i+=1
        if i>=MAX_ROWS:
            print("Done")
            break   
        print(f"Encoder data step done {i}")

for agent in env.agents:
    print(f"player_link={SERVER_BASE_URL}/player?id={env.agent_id_to_player_id[agent]}")
    
#np.save("test.npy",result)