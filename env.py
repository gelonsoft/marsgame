import string
from time import sleep
import time
import traceback
import numpy as np
import json
from gymnasium import spaces
from myutils import find_first_with_nested_attr, get_stat
from typing import Any, List, Dict
from myconfig import GAME_START_JSON, MAX_ACTIONS, SERVER_BASE_URL, PLAYER_COLORS
from observe_gamestate import get_actions_shape, observe  # assuming observe() is defined in another module
import requests
import random
from decision_mapper import TerraformingMarsDecisionMapper
import logging
import os
from decode_observation import decode_observation
from reward import reward_function

logging.basicConfig(level=logging.INFO)  # Set the logging level to DEBUG



global mmax_actions
mmax_actions=5

request_number=0
request_responses={}
USE_MOCK_SERVER=False
if USE_MOCK_SERVER:
    with open("response.json","rb") as f:
        request_responses=json.loads(f.read())
        
LOG_REQUESTS=False


def get_player_state(player_id):
    url = f"{SERVER_BASE_URL}/api/player"
    logging.debug(f"get_player_state {player_id}")
    global request_responses,request_number
    if USE_MOCK_SERVER:
        response=request_responses[str(request_number)]["response"]
        logging.debug(f"Request url={url} {player_id} request_mock={request_responses[str(request_number)]['request']}")
        request_number+=1
        return response
    response = requests.get(url,params={"id":player_id},headers={"content-type":"application/json"})
    response.raise_for_status()
    try:
        response_json=response.json()
        if LOG_REQUESTS:
            request_responses[request_number]={"request":response.request.url,"method":"get","response":response_json}
        request_number+=1
        return response_json
    except Exception as e:
        print(f"Bad get_player_state response:\n{response.text}")
        raise e

def post_player_input(run_id,player_id, player_input_model):
    url = f"{SERVER_BASE_URL}/player/input"
    player_input_model['runId']=run_id
    global request_responses,request_number
    if USE_MOCK_SERVER:
        response=request_responses[str(request_number)]["response"]
        logging.debug(f"Request url={url} {player_id} request_mock={request_responses[str(request_number)]['request']}")
        request_number+=1
        return response
    response = requests.post(url, params={"id":player_id}, json=player_input_model)
    try:
        response.raise_for_status()
        resp=response.json()
        if LOG_REQUESTS:
            request_responses[request_number]={"request":response.request.url,"method":"get","response":resp}
        request_number+=1
        logging.debug(f"Response: post_player_input\n---\n{resp}\n---")
        return resp
    except Exception as e:
        print(f"Bad post_player_input response:\n{response.text}\npayload:\n{json.dumps(player_input_model)}\n")
        return response.text

def start_new_game(num_players):

    GAME_START_JSON['players']=[{
            "name": str(i),
            "color": PLAYER_COLORS[i],
            "beginner": False,
            "handicap": 0,
            "first": False
        } for i in range(num_players)]
    url = f"{SERVER_BASE_URL}/api/creategame"
    global request_responses,request_number
    if USE_MOCK_SERVER:
        response=request_responses[str(request_number)]["response"]
        logging.debug(f"Request url={url} {num_players} request_mock={request_responses[str(request_number)]['request']}")
        request_number+=1
        return response
    response = requests.put(url, json=GAME_START_JSON)
    response.raise_for_status()
    response_json=response.json()
    if LOG_REQUESTS:
        request_responses[str(request_number)]={"request":response.request.url,"method":"get","response":response_json}
    request_number+=1
    print(f"Started new game {SERVER_BASE_URL}/spectator?id={response_json['spectatorId']}")
    return response_json


class TerraformingMarsEnv():
    metadata = {"render_modes": ["human"], "name": "terraforming_mars_aec_v0","is_parallelizable":True}

    def __init__(self, init_from_player_state=False, player_id=None,player_state=None,waiting_for=None,safe_mode=True):
        self.render_mode='human'
        self.action_slot_map = {}
        self.raise_exceptions=not safe_mode
        self.decision_mapper=TerraformingMarsDecisionMapper(None)
        self.game_id=None
        self.init_from_player_state=init_from_player_state
        self.start_player_state=None
        if self.init_from_player_state:
            if player_id:
                self.start_player_state=get_player_state(player_id)
            else:
                self.start_player_state=player_state
        self.observations_made=0
        self.waiting_for=waiting_for
        self.player_id=""
        self.rewards = 0.0
        self.infos = {}
        self.saved_player_state = {}
        self.action_lookup = {}
        self.actions_list=[]
        self.actions_count=0
        self.terminations= False
        self.truncations = False
        self.deffered_actions=None
        self.skip_full_observation=False
        self.reset()
        if self.current_obs is None:
            raise Exception("Bad obs state")
        self.observation_shape = self.current_obs.shape
        self.action_shape=get_actions_shape()
        self.observation_spaces = spaces.Box(low=0.0, high=1.0, shape=self.observation_shape, dtype=np.float32)
        self.action_spaces = spaces.Discrete(MAX_ACTIONS)


    def flatten_actions(self, action):
        flat = []

        if action["type"] == "or":
            for sub in action["options"]:
                flat.extend(self.flatten_actions(sub))

        elif action["type"] == "and":
            combo = []
            for sub in action["actions"]:
                combo.extend(self.flatten_actions(sub))
            flat.append({"type": "and", "actions": combo})

        else:
            flat.append(action)

        return flat

    def _generate_agent_action_map(self):
        action_map={}
        if self.deffered_actions is None:
            action_map = self.decision_mapper.generate_action_space(self.player_states.get("waitingFor"),self.player_states,True,None)  # Initialize the action map
        else:
            action_map = self.decision_mapper.generate_action_space(self.player_states.get("waitingFor"),self.player_states,True,self.deffered_actions)  # Initialize the action map

        self.legal_actions = action_map 

        slots = list(range(MAX_ACTIONS))
        random.shuffle(slots)

        legal = self.legal_actions
        slot_map = {}
        reverse_map = {}

        for i, action in enumerate(legal.values()):
            if i>=MAX_ACTIONS:
                continue
            slot = slots[i]
            slot_map[slot] = action
            reverse_map[slot] = i

        self.action_slot_map = slot_map
        self.reverse_action_slot_map = reverse_map

        self.action_lookup = slot_map
        self.actions_list=list(slot_map.keys())
        self.actions_count=len(self.actions_list)
        

    def _update_agent_state(self,force_update=False):
        if self.start_player_state and self.observations_made==0:
                self.player_states = self.start_player_state
        else:
            if force_update or (self.deffered_actions is None and not self.skip_full_observation):
                self.player_states = get_player_state(self.player_id)
            if self.waiting_for and self.observations_made==0:
                self.player_states["waitingFor"] =self.waiting_for

    def _update_agent_observation(self,force_update=False):
        self._update_agent_state(force_update)
        self._generate_agent_action_map()
        self.current_obs = observe(self.player_states,self.action_slot_map)
        
    def update_state(self):
        self._update_all_observations(force=True)

    def _update_all_observations(self,force=False):
        self.current_obs = None
        self.legal_actions = {}
        self.action_spaces = {}
        self._update_agent_observation(force)
        self.observations_made+=1

    def observe(self):
        return self.current_obs

    def _extract_player_metrics(self):
        p = self.player_states["thisPlayer"]

        tr = p["terraformRating"]

        vp=int(self.player_states['thisPlayer']['victoryPointsBreakdown']['total'])

        production = (
            p["megaCreditProduction"]
            + p["steelProduction"]
            + p["titaniumProduction"]
            + p["plantProduction"]
            + p["energyProduction"]
            + p["heatProduction"]
        )

        return {
            "tr": tr,
            "production": production,
            "vp": vp
        }
    def reset(self, seed=None, options=None):
        self.skip_full_observation=False
        self.saved_player_state = {}
        self.terminations=False
        self.truncations=False
        new_game_response=self.start_player_state or {}
        if self.init_from_player_state:
            pass
        else:
            new_game_response=start_new_game(2)
            self.game_id=new_game_response['id']
        self.player_id=None
        self.deffered_actions=None

        if self.init_from_player_state:
            self.player_id=new_game_response['id']
        else:
            player=new_game_response['players'][0]
            self.player_id=player['id']
            print(f"Player links for new game {SERVER_BASE_URL}/player?id={player['id']}")

        self.rewards = 0.0
        self.infos = {}
            
        self._update_all_observations()
        return self.current_obs,self.infos


    def get_action_mask(self):
        mask = [0] * MAX_ACTIONS

        for slot in self.action_slot_map.keys():
            mask[slot] = 1

        return mask


    def post_player_input(self,player_input):
        return post_player_input(self.player_states['runId'],self.player_id,player_input)

    def _calc_reward_by_metrics(self,prev)->float:
        delta_tr:float=0.0
        delta_prod:float=0.0
        delta_vp:float=0.0
        reward:float=0.0
        if prev is not None:
            try:
                curr = self._extract_player_metrics()
                delta_tr = curr["tr"] - prev["tr"]
                delta_prod = curr["production"] - prev["production"]
                delta_vp = curr["vp"] - prev["vp"]
            except Exception:
                pass
            reward += 0.01                     # valid move bonus
            reward += 0.1 * delta_tr           # TR shaping
            reward += 0.2 * delta_vp           # TR shaping
            reward += 0.02 * delta_prod        # production shaping
                
        return reward


    def _step(self,action): 
        #old_player_input=None
        failed=False
        need_sleep=False
        action_lookup=self.action_lookup
        acts=len(action_lookup.keys())
        if acts==0:
            return True
        self.skip_full_observation=True
        #player_input = action_lookup.get(action)
        player_input = self.action_slot_map.get(action)
        if player_input is not None:
            if self.deffered_actions is None:
                first_deffered_action=find_first_with_nested_attr(player_input,"__deferred_action")
                if first_deffered_action is not None:
                    self.deffered_actions=player_input
                    player_input=None
                else:
                    self.deffered_actions=None
            else:
                first_deffered_action=find_first_with_nested_attr(self.deffered_actions,"__deferred_action")
                if first_deffered_action is None:
                    raise Exception("first_deffered_action should not be None")
                if player_input['type']!="deffered":
                    raise Exception("player_input should be deffered")
                parent,deffered=first_deffered_action
                parent=parent or {}
                if player_input["xtype"]=="xcard_choose":
                    selected=deffered.get('selected',[])
                    selected.append(player_input['xoption']['name'])
                    if len(selected)>=deffered['xmax']:
                        parent['cards']=selected
                        need_sleep=True
                    else:
                        deffered['selected']=selected
                elif player_input["xtype"]=="xpayment":
                    parent['payment']=player_input['xoption']
                    need_sleep=True
                elif player_input["xtype"]=="xconfirm_card_choose":
                    selected=deffered.get('selected',[])
                    if len(selected)<deffered['xmin']:
                        raise Exception("len(selected)<deffered['xmin']")
                    parent['cards']=selected
                    need_sleep=True
                elif player_input["xtype"]=="xremove_first_card_choose":
                    selected=deffered.get('selected',[])
                    if len(selected)<=0:
                        raise Exception("len(selected)<0")
                    selected.pop(0)
                    deffered['selected']=selected
                first_deffered_action=find_first_with_nested_attr(self.deffered_actions,"__deferred_action")
                if first_deffered_action is not None:
                    player_input=None
                else:
                    player_input=self.deffered_actions
                    self.deffered_actions=None

            if player_input is not None:
                res=self.post_player_input(player_input)
                if need_sleep:
                    #sleep(0.3)
                    pass
                if isinstance(res,str):
                    reward = -0.05
                    print(f"Failed to post player input with input player_link={SERVER_BASE_URL}/player?id={self.player_id}: \n{json.dumps(player_input, indent=2)}\n and waiting steps \n{json.dumps(self.player_states.get('waitingSteps',{}), indent=2)}\n")
                    failed=True
                    with open(os.path.join("data","failed_actions",''.join(random.choices(string.ascii_uppercase + string.digits, k=12))+".json"),'w',encoding='utf-8') as f:
                        f.write(json.dumps({
                            "player_link": f"{SERVER_BASE_URL}/player?id={self.player_id}",
                            "player_id": self.player_id,
                            "player_input":player_input,
                            "error":res,
                            "player_state.waitingFor":self.player_states
                        }))
                    if self.raise_exceptions:
                        raise Exception("Bad player actions")
                    player_input=None
                    self._update_agent_observation(True)
                else:
                    self.skip_full_observation=True
                    self.player_states=res
        else:
            raise Exception(f"Bad action id action={action} of {self.action_lookup}")
        
        self._update_agent_state()
        self._generate_agent_action_map()
        acts=len(self.action_lookup.keys())
        if acts==1:
            self._step(list(self.action_lookup.keys())[0])

        return failed


    def save_metrics_for_reward_calc(self):
        self.saved_player_state=self.player_states

    def get_reward_by_saved_metrics(self):
        # if self.terminations:
        #     players=self.player_states.get('players',[])
        #     if len(players)>1:
        #         for p in players:
        #             if p.get('id','')==self.player_id:
        #                 self.rewards=p.get('victoryPointsBreakdown',{}).get('total',0)/100.0
        #                 return self.rewards
        return reward_function(self.saved_player_state,self.player_states)


    def step(self, action):
        failed=self._step(action)
        self._update_agent_observation()
        if self.player_states.get('game',{}).get('phase',"")=="end":
            self.terminations=True
        else:
            self.terminations=False
        if self.terminations:
            players=self.player_states.get('players',[])
            if len(players)>1:
                for p in players:
                    if p.get('id','')==self.player_id:
                        self.rewards=p.get('victoryPointsBreakdown',{}).get('total',0)/100.0
        
        print(f"Step {SERVER_BASE_URL}/game?id={self.game_id} ... actions: {action} of {list(self.action_lookup.keys())} isTerraformed={self.player_states.get('game',{}).get('isTerraformed',False)} self.terminations={self.terminations} phases={self.player_states.get('game',{}).get('phase',False)}")

        return self.actions_count,self.actions_list,self.current_obs,self.terminations,failed


    

def parallel_env(opponent=None):
    env = TerraformingMarsEnv()
    return env

