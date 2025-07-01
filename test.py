import numpy as np
from env import TerraformingMarsEnv
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
MAX_ROWS=10240
result=np.zeros((MAX_ROWS,env.action_space(env.possible_agents[0]).shape[0]))
rand=random.Random()
i=0
next_obs, rewards, terms, truncs, infos=(None,None,None,None,None)
while True:
    actions={}
    is_no_actions=True
    for agent in env.action_lookup:
        max_actions=len(env.action_lookup[agent].keys())
        if max_actions>1:
            actions[agent]=int(rand.randrange(0,max_actions-1))
            is_no_actions=False
        elif max_actions==0:
            actions[agent]=0
        else:
            is_no_actions=False
            actions[agent]=0
    if terms and terms.get('1',False):
        terms=None
        env=TerraformingMarsEnv(["1","2"])
    else:
        next_obs, rewards, terms, truncs, infos=env.step(actions)
        result[i]=next_obs['1']
        i+=1
        if i>=MAX_ROWS:
            print("Done")
            break
        result[i]=next_obs['2']
        i+=1
        if i>=MAX_ROWS:
            print("Done")
            break   

pa_table = pa.table({"data": result})
pa.parquet.write_table(pa_table, "test.parquet")