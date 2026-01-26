import random
import time
from typing import List
from env import TerraformingMarsEnv, start_new_game
from pettingzoo import AECEnv, ParallelEnv

from myconfig import MAX_ACTIONS


class AllActionsEnv(ParallelEnv):
    def __init__(self,num_players=4,num_human_players=3,safe_mode=True):
        self.num_players=num_players
        self.num_human_players=num_human_players
        self.num_random_players=num_players-num_human_players
        self.safe_mode=safe_mode
        self.new_game_response=None
        self.player_ids=[]
        self.envs:List[TerraformingMarsEnv]=[]
        self.agents:List[str]=[]
        self.player_types:List[str]=[]
        self.last_env:int=0
        self.current_obs=None
        self.infos=None
        self.rewards={}
        self.terminations={}
        self.game_id=""
        self.stale_states=[False for p in range(num_players)]
        self.no_steps_count=0
        
    def reset(self,seed=None,options=None):
        self.no_steps_count=0
        self.new_game_response=start_new_game(num_players=self.num_players)
        self.game_id=self.new_game_response['id']
        self.envs=[]
        self.player_ids=[]
        self.rewards={}
        self.terminations={}
        self.stale_states=[False for p in range(self.num_players)]
        self.infos=None
        self.agents:List[str]=[]
        self.current_obs=None
        i=0
        for p in self.new_game_response['players']:
            agent_id=str(i+1)
            self.agents.append(agent_id)
            player_id=p['id']
            self.player_ids.append(player_id)
            env=TerraformingMarsEnv(init_from_player_state=True,player_id=player_id,safe_mode=self.safe_mode)
            zz=0
            while env.actions_count==1 and zz<10:
                zz=zz+1
                env.step(env.actions_list[0])
            if env.actions_count>0:
                self.last_env=i          
            self.player_types.append("h" if i>=self.num_human_players else "r")
            self.envs.append(env)
            i+=1
        if not self.last_env:
            raise Exception(f"Not found active env {self.envs}")
        env=self.envs[self.last_env]
        if env.current_obs is None:
            raise Exception("Bad obs")
        return env.current_obs,env.infos,env.actions_count,env.actions_list,self.last_env

    def stale_state_except(self,except_env_num:int):
        self.stale_states=[p!=except_env_num for p in range(self.num_players)]

    def get_env(self,env_id) -> TerraformingMarsEnv:
        env=self.envs[env_id]
        if self.stale_states[env_id]:
            env.update_state()
        return env

    def step(self,action: int):
        rewards:float=0.0
        env=self.envs[self.last_env]
        no_steps=0
        if self.no_steps_count>3:
            print(f"Bad no steps2 game={self.game_id}")
            print(f"Reward={-1.0}")
            return env.current_obs,-1.0,True,0,[],{i:self.envs[i].player_states.get('thisPlayer',{}).get('victoryPointsBreakdown',{}).get('total',0) for i in range(self.num_players)},self.last_env
        if env.actions_count>0:
            env.save_metrics_for_reward_calc()
            actions_count,action_list,obs,terminations,failed=env.step(action)
            no_steps=0 if not failed else no_steps+1
            while env.actions_count==1:
                actions_count,action_list,obs,terminations,failed=env.step(env.actions_list[0])
                no_steps=0 if not failed else no_steps+1
            self.stale_state_except(self.last_env)
            rewards=env.get_reward_by_saved_metrics()
            self.rewards[self.last_env]=rewards
            if actions_count>0 and not terminations:
                self.no_steps_count=0 if not failed else self.no_steps_count+1
                #print(f"Changed no_steps_count #1 to {self.no_steps_count}")
                print(f"Reward={rewards}")
                return obs,rewards,terminations,actions_count,action_list,{i:self.envs[i].player_states.get('thisPlayer',{}).get('victoryPointsBreakdown',{}).get('total',0) for i in range(self.num_players)},self.last_env
        z=0
        eid=self.last_env
        
        while z<100 and no_steps<=self.num_players*2:
            z=z+1
            eid=eid+1
            if eid==self.num_players:
                eid=0
            env=self.get_env(eid)
            while env.actions_count==1 and not failed:
                _,_,_,_,failed=env.step(env.actions_list[0])
                self.stale_state_except(eid)
                no_steps=0 if not failed else no_steps+1
            if env.actions_count>0:
                if self.player_types[eid]=="r":
                    env.save_metrics_for_reward_calc()
                    zz=0
                    while env.actions_count>0 and zz<100:
                        zz+=1
                        rac,ral,robs,rterm,failed=env.step(random.choice(env.actions_list))
                        self.stale_state_except(eid)
                        no_steps=0 if not failed else no_steps+1
                    self.rewards[eid]=env.get_reward_by_saved_metrics()
                    no_steps+=1
                else: #human
                    self.last_env=eid
                    while env.actions_count==1:
                        _,_,_,_,failed=env.step(env.actions_list[0])
                        no_steps=0 if not failed else no_steps+1
                    self.no_steps_count=0 if not failed else self.no_steps_count+1
                    print(f"Reward={rewards}")
                    return env.current_obs,rewards,env.terminations,env.actions_count,env.actions_list,self.rewards,self.last_env
            else:
                no_steps+=1
            if no_steps>=self.num_players and not (all([env.terminations for env in self.envs])):
                print(f"No steps for all, wait no_steps={no_steps} self.no_steps_count={self.no_steps_count}")
                self.no_steps_count+=1
                #print(f"Changed no_steps_count #3 to {self.no_steps_count}")
                self.stale_state_except(-1)
                time.sleep(0.3)
                continue

                
        if all([env.terminations for env in self.envs]):
            print(f"Game {self.game_id} completed rewards={self.rewards}")
            self.no_steps_count=0 if not failed else self.no_steps_count+1
            #print(f"Changed no_steps_count #4 to {self.no_steps_count}")
            print(f"Reward={rewards}")
            return env.current_obs,rewards,env.terminations,env.actions_count,env.actions_list,self.rewards,self.last_env
        else:
            print(f"Bad no steps game={self.game_id}")
            self.no_steps_count+=1
            #print(f"Changed no_steps_count #5 to {self.no_steps_count}")
            print(f"Reward={-1}")
            return env.current_obs,-1,True,0,[],{i:self.envs[i].player_states.get('thisPlayer',{}).get('victoryPointsBreakdown',{}).get('total',0) for i in range(self.num_players)},self.last_env

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def get_action_mask(self,action_list):
        mask = [0] * MAX_ACTIONS

        for slot in action_list:
            mask[slot] = 1

        return mask

    def render(self,render_mode):
        pass

    def close(self):
        pass 


def parallel_env(num_players=4,num_human_players=2):
    return AllActionsEnv(num_players=num_players, num_human_players=num_human_players, safe_mode=True)