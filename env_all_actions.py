"""
AllActionsEnv - Multi-player Parallel Environment for Terraforming Mars

This environment manages multiple TerraformingMarsEnv instances (one per player)
and coordinates turn-taking between human-controlled and random AI players.

FACTORED ACTION INTEGRATION:
----------------------------
The step() method has been updated to return factored action metadata from
the active player's environment so that MCTS can use it for the next action
selection. Returns extended 12-tuple instead of legacy 7-tuple.
"""

import random
import time
from typing import List
from env import TerraformingMarsEnv, start_new_game
from pettingzoo import AECEnv, ParallelEnv

from myconfig import MAX_ACTIONS


class AllActionsEnv(ParallelEnv):
    def __init__(self,num_players=4,num_human_players=3,safe_mode=True):
        self.server_id=random.choice([0,1,2,3])
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
        self.new_game_response=start_new_game(num_players=self.num_players,server_id=self.server_id)
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
            env=TerraformingMarsEnv(init_from_player_state=True,player_id=player_id,safe_mode=self.safe_mode,server_id=self.server_id)
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
        """
        Take a step in the environment.
        
        Returns
        -------
        12-tuple: (obs, rewards, terminations, actions_count, action_list, 
                   all_rewards, curr_env_id, head_masks, cond_obj, cond_pay, 
                   cond_extra, factored_legal)
        
        The last 5 elements are factored action metadata from the active
        player's TerraformingMarsEnv, needed for MCTS action selection.
        """
        rewards:float=0.0
        env=self.envs[self.last_env]
        no_steps=0
        
        # Early exit: too many failed steps
        if self.no_steps_count>3:
            # Return with None factored data (terminal state)
            return (env.current_obs, -1.0, True, 0, [],
                    {i:self.envs[i].player_states.get('thisPlayer',{}).get('victoryPointsBreakdown',{}).get('total',0) for i in range(self.num_players)},
                    self.last_env,
                    None, None, None, None, None)  # factored data = None
        
        # Take step in active env
        if env.actions_count>0:
            env.save_metrics_for_reward_calc()
            actions_count,action_list,obs,terminations,failed,hm,co,cp,ce,fl=env.step(action)
            no_steps=0 if not failed else no_steps+1
            
            # Auto-step through single-action states
            while env.actions_count==1:
                actions_count,action_list,obs,terminations,failed,hm,co,cp,ce,fl=env.step(env.actions_list[0])
                no_steps=0 if not failed else no_steps+1
            
            self.stale_state_except(self.last_env)
            rewards=env.get_reward_by_saved_metrics()
            self.rewards[self.last_env]=rewards
            
            if actions_count>0 and not terminations:
                self.no_steps_count=0 if not failed else self.no_steps_count+1
                # Return with factored data from active env
                return (obs, rewards, terminations, actions_count, action_list,
                        {i:self.envs[i].player_states.get('thisPlayer',{}).get('victoryPointsBreakdown',{}).get('total',0) for i in range(self.num_players)},
                        self.last_env,
                        hm, co, cp, ce, fl)  # factored action metadata
        
        # Search for next active player
        z=0
        eid=self.last_env
        
        while z<100 and no_steps<=self.num_players*2:
            z=z+1
            eid=eid+1
            if eid==self.num_players:
                eid=0
            env=self.get_env(eid)
            
            # Auto-step through single-action states
            while env.actions_count==1 and not failed:
                _,_,_,_,failed,_hm,_co,_cp,_ce,_fl=env.step(env.actions_list[0])
                self.stale_state_except(eid)
                no_steps=0 if not failed else no_steps+1
            
            if env.actions_count>0:
                if self.player_types[eid]=="r":
                    # Random player: take steps until done
                    env.save_metrics_for_reward_calc()
                    zz=0
                    while env.actions_count>0 and zz<100:
                        zz+=1
                        rac,ral,robs,rterm,failed,_hm,_co,_cp,_ce,_fl=env.step(random.choice(env.actions_list))
                        self.stale_state_except(eid)
                        no_steps=0 if not failed else no_steps+1
                    self.rewards[eid]=env.get_reward_by_saved_metrics()
                    no_steps+=1
                else: # human player found
                    self.last_env=eid
                    
                    # Auto-step through single-action states
                    while env.actions_count==1:
                        _,_,_,_,failed,hm,co,cp,ce,fl=env.step(env.actions_list[0])
                        no_steps=0 if not failed else no_steps+1
                    
                    self.no_steps_count=0 if not failed else self.no_steps_count+1
                    
                    # Return with factored data from new active env
                    return (env.current_obs, rewards, env.terminations, 
                            env.actions_count, env.actions_list, self.rewards, 
                            self.last_env,
                            env.head_masks, env.cond_masks_obj_by_type,
                            env.cond_masks_pay_by_obj, env.cond_masks_extra_by_pay,
                            env.factored_legal)
            else:
                no_steps+=1
            
            # Stall detection
            if no_steps>=self.num_players and not (all([env.terminations for env in self.envs])):
                print(f"No steps for all, wait no_steps={no_steps} self.no_steps_count={self.no_steps_count}")
                self.no_steps_count+=1
                self.stale_state_except(-1)
                time.sleep(0.3)
                continue

        # Game completed        
        if all([env.terminations for env in self.envs]):
            print(f"Game {self.game_id} completed rewards={self.rewards}")
            self.no_steps_count=0 if not failed else self.no_steps_count+1
            print(f"Reward={rewards}")
            # Return with None factored data (game ended)
            return (env.current_obs, rewards, env.terminations, 
                    env.actions_count, env.actions_list, self.rewards, 
                    self.last_env,
                    None, None, None, None, None)
        else:
            # Bad state: no valid moves found
            self.no_steps_count+=1
            return (env.current_obs, -1, True, 0, [],
                    {i:self.envs[i].player_states.get('thisPlayer',{}).get('victoryPointsBreakdown',{}).get('total',0) for i in range(self.num_players)},
                    self.last_env,
                    None, None, None, None, None)

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