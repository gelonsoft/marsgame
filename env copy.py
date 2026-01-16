import string
from time import sleep
import time
import traceback
import numpy as np
import json
from gymnasium import spaces
from pettingzoo import AECEnv, ParallelEnv
from myutils import find_first_with_nested_attr, get_stat
try:
    from pettingzoo.utils import agent_selector
    x=agent_selector(['1','2'])
except:
    from pettingzoo.utils import AgentSelector
    x=AgentSelector(['1','2'])
from typing import List, Dict
from myconfig import GAME_START_JSON, MAX_ACTIONS
from observe_gamestate import get_actions_shape, observe  # assuming observe() is defined in another module
import requests
import random
from decision_mapper import TerraformingMarsDecisionMapper
import logging
import os
from decode_observation import decode_observation

logging.basicConfig(level=logging.INFO)  # Set the logging level to DEBUG

SERVER_BASE_URL=os.environ.get('SERVER_BASE_URL','http://localhost:9976') #,"http://lev-rworker-3:9976")

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
        #with open(os.path.join("debug",f"{request_number}.json"),"w") as f:
        #    f.write(json.dumps(response_json,indent=2))
        #logging.debug(f"Request url={url} {player_id} response:\n{json.dumps(response_json,indent=2)}")
        #print(f"state cards in hand: {[c['name'] for c in response_json['cardsInHand']]}")
        return response_json
    except Exception as e:
        print(f"Bad get_player_state response:\n{response.text}")
        raise e

def post_player_input(run_id,player_id, player_input_model):
    url = f"{SERVER_BASE_URL}/player/input"
    #logging.debug(f"post_player_input Request:---\n{json.dumps(player_input_model)}\n---\n")
    #print(f"post_player_input Request:---\n{json.dumps(player_input_model)}\n---\n")
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


class TerraformingMarsEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "terraforming_mars_aec_v0","is_parallelizable":True}

    def __init__(self, agent_ids: List[str], init_from_player_state=False, player_id=None,player_state=None,waiting_for=None,safe_mode=True):
        super().__init__()
        self.controlled_agents = ["1"]
        self.random_agents = [a for a in agent_ids if a not in self.controlled_agents]
        self.action_slot_map = {}
        self.reverse_action_slot_map = {}
        self.raise_exceptions=not safe_mode
        self.decision_mapper=TerraformingMarsDecisionMapper(None)
        self.opponent_policy=None
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
        #self.run_id
        #self.run_id=None
        self.spectator_id=None
        self.player_name_to_id={}
        self.agent_id_to_player_id={}
        self.possible_agents = agent_ids
        self.agents = self.possible_agents[:]
        try:
            self._agent_selector = agent_selector(self.agents)
        except:
            self._agent_selector = AgentSelector(self.agents)
        self.action_spaces = {}
        self.agent_selection = self._agent_selector.reset()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.prev_player_metrics = {agent: {} for agent in self.agents}
        self.player_states={}
        self.action_lookup = {}
        self.reverse_action_lookup = {}  # For debugging
        self.observation_shape=None
        self.render_mode='human'
        self.terminations= {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.reset()
        self.deffered_actions={agent: None for agent in self.agents}
        self.skip_full_observation={agent: False for agent in self.agents}
        #self.player_states = {agent: get_player_state(self.agent_id_to_player_id[agent]) for agent in self.agents}
        sample_obs = self.current_obs[self.agents[0]] #observe(self.player_states[self.agents[0]],self.decision_mapper.generate_action_space(self.player_states[agent].get("waitingFor")))
        self.observation_shape = sample_obs.shape
        self.action_shape=get_actions_shape()

        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=self.observation_shape, dtype=np.float32)
            for agent in self.controlled_agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=self.action_shape, dtype=np.float32)
            for agent in self.controlled_agents
        }


        #self._update_all_observations()

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

    def _generate_agent_action_map(self,agent):
        action_map={}
        if self.deffered_actions[agent] is None:
            action_map = self.decision_mapper.generate_action_space(self.player_states[agent].get("waitingFor"),self.player_states[agent],True,None)  # Initialize the action map
        else:
            action_map = self.decision_mapper.generate_action_space(self.player_states[agent].get("waitingFor"),self.player_states[agent],True,self.deffered_actions[agent])  # Initialize the action map

        server_actions= action_map #list(action_map.keys())[:MAX_ACTIONS]
        self.legal_actions[agent] = server_actions 
        #self.action_lookup[agent] = {i: action_map[i] for i in self.legal_actions[agent]}
        #self.reverse_action_lookup[agent] = action_map
        #self.action_spaces[agent] = spaces.Discrete(len(self.action_lookup[agent]) or 1)
        slots = list(range(MAX_ACTIONS))
        random.shuffle(slots)

        legal = self.legal_actions[agent]
        slot_map = {}
        reverse_map = {}

        for i, action in enumerate(legal.values()):
            if i>=MAX_ACTIONS:
                continue
            slot = slots[i]
            slot_map[slot] = action
            reverse_map[slot] = i

        self.action_slot_map[agent] = slot_map
        self.reverse_action_slot_map[agent] = reverse_map

        self.action_lookup[agent] = slot_map
        self.action_spaces[agent] = spaces.Discrete(MAX_ACTIONS)

    def _update_agent_state(self,agent,force_update=False):
        
        if self.start_player_state and self.observations_made==0:
                self.player_states[agent] = self.start_player_state
        else:
            if force_update or (self.deffered_actions[agent] is None and not self.skip_full_observation[agent]):
                #print(f"Update agent state for agent={agent}")
                self.player_states[agent] = get_player_state(self.agent_id_to_player_id[agent])
            if self.waiting_for and self.observations_made==0:
                self.player_states[agent]["waitingFor"] =self.waiting_for

    def _update_agent_observation(self,agent,force_update=False):
        self._update_agent_state(agent,force_update)
        self._generate_agent_action_map(agent)
        if agent in self.random_agents:
            return
        self.current_obs[agent] = observe(self.player_states[agent],self.action_slot_map[agent])
        

    def _update_all_observations(self,force=False):
        self.current_obs = {}
        self.legal_actions = {}
        self.action_spaces = {}
        for agent in self.agents:
            self._update_agent_observation(agent,force)
        self.observations_made+=1

    def observe(self, agent):
        return self.current_obs[agent]

    def _extract_player_metrics(self, agent):
        p = self.player_states[agent]["thisPlayer"]

        tr = p["terraformRating"]

        vp=int(self.player_states[agent]['thisPlayer']['victoryPointsBreakdown']['total'])

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
    def reset_all(self, seed=None, options=None):
        self.skip_full_observation={agent: False for agent in self.agents}
        self.agents = self.possible_agents[:]
        try:
            self._agent_selector = agent_selector(self.agents)
        except:
            self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.prev_player_metrics = {agent: {} for agent in self.agents}
        self.terminations={agent: False for agent in self.agents}
        self.truncations={agent: False for agent in self.agents}
        new_game_response=self.start_player_state
        if self.init_from_player_state:
            self.spectator_id=self.start_player_state.get('game',{}).get('spectatorId','')
        else:
            new_game_response=start_new_game(len(self.agents))
            self.game_id=new_game_response['id']
            #self.run_id=new_game_response['runId']
            self.spectator_id=new_game_response['spectatorId']
        self.player_name_to_id={}
        self.agent_id_to_player_id={}
        self.deffered_actions={agent: None for agent in self.agents}

        for i in range(len(self.agents)):
            agent=self.agents[i]
            if self.init_from_player_state:
                self.agent_id_to_player_id[agent]=new_game_response['id']
                self.player_name_to_id[new_game_response['id']]=new_game_response['thisPlayer']['color']
            else:
                player=new_game_response['players'][i]
                self.agent_id_to_player_id[agent]=player['id']
                self.player_name_to_id[player['id']]=player['color']
                print(f"Player links for new game {SERVER_BASE_URL}/player?id={player['id']}")
            self.dones[agent] = False
            self.rewards[agent] = 0.0
            self.infos[agent] = {}
            
        self._update_all_observations()
        return self.current_obs,self.infos
    
    # Single-agent reset helper (agent "1" only)
    def reset(self, seed=None, options=None):
        main_agent="1"
        obs, infos = self.reset_all(seed=seed, options=options)
        return {main_agent:obs[main_agent]},{main_agent:infos[main_agent]}
    
    #def get_action_mask(self):
        #return [len(self.legal_actions[agent]) or 1 for agent in self.agents]
    def get_action_mask(self, agent):
        mask = [0] * MAX_ACTIONS

        for slot in self.action_slot_map[agent].keys():
            mask[slot] = 1

        return mask


    def post_player_input(self,agent_id,player_input):
        return post_player_input(self.player_states[agent_id]['runId'],self.agent_id_to_player_id[agent_id],player_input)

    def _calc_reward_by_metrics(self,agent,prev):
        delta_tr=0
        delta_prod=0
        delta_vp=0
        reward=0
        if prev is not None:
            try:
                curr = self._extract_player_metrics(agent)
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


    def _step(self,agent,action): 
        #old_player_input=None
        need_sleep=False
        prev_player_metrics=self._extract_player_metrics(agent)
        action_lookup=self.action_lookup[agent]
        acts=len(action_lookup.keys())
        if acts==0:
            return 0
        self.skip_full_observation[agent]=True
        #player_input = action_lookup.get(action)
        player_input = self.action_slot_map[agent].get(action)
        reward=0
        if player_input is not None:
            if self.deffered_actions[agent] is None:
                first_deffered_action=find_first_with_nested_attr(player_input,"__deferred_action")
                if first_deffered_action is not None:
                    self.deffered_actions[agent]=player_input
                    player_input=None
                else:
                    self.deffered_actions[agent]=None
            else:
                first_deffered_action=find_first_with_nested_attr(self.deffered_actions[agent],"__deferred_action")
                if first_deffered_action is None:
                    raise Exception("first_deffered_action should not be None")
                if player_input['type']!="deffered":
                    raise Exception("player_input should be deffered")
                parent,deffered=first_deffered_action
                if player_input["xtype"]=="xcard_choose":
                    selected=deffered.get('selected',[])
                    selected.append(player_input['xoption']['name'])
                    #print(f"a={agent} xcard_choose={player_input['xoption']['name']} selected_after={selected} deffered={deffered}" )
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
                first_deffered_action=find_first_with_nested_attr(self.deffered_actions[agent],"__deferred_action")
                if first_deffered_action is not None:
                    player_input=None
                else:
                    player_input=self.deffered_actions[agent]
                    self.deffered_actions[agent]=None

            if player_input is not None:
                #print(f"a={agent} player_input={player_input}")
                #z=[c['name'] for c in self.player_states[agent]['cardsInHand']]
                #print(f"a={agent} cardsInHand={z}")
                res=self.post_player_input(agent, player_input)
                if need_sleep:
                    #sleep(0.3)
                    pass
                if isinstance(res,str):
                    reward = -0.05
                    print(f"Failed to post player input for agent {agent} with input player_link={SERVER_BASE_URL}/player?id={self.agent_id_to_player_id[agent]}: \n{json.dumps(player_input, indent=2)}\n and waiting steps \n{json.dumps(self.player_states[agent].get('waitingSteps',{}), indent=2)}\n")
                    if 'already exists' in res or 'Not waiting for anything' in res:

                        #sleep(0.1)
                        pass
                    else:
                        with open(os.path.join("data","failed_actions",''.join(random.choices(string.ascii_uppercase + string.digits, k=12))+".json"),'w',encoding='utf-8') as f:
                            f.write(json.dumps({
                                "player_link": f"{SERVER_BASE_URL}/player?id={self.agent_id_to_player_id[agent]}",
                                "player_id": self.agent_id_to_player_id[agent],
                                "player_input":player_input,
                                "error":res,
                                "player_state.waitingFor":self.player_states[agent]
                            }))
                        if self.raise_exceptions:
                            raise Exception("Bad player actions")
                        player_input=None
                    self._update_agent_observation(agent,True)
                else:
                    #print(f"{agent}=agent")
                    #print(f"action_lookup={action_lookup}")
                    #print(f"player_input={player_input}")
                    #print(f"res.waitingFor={res.get('waitingFor')}")
                    #print(f"deffered_actions={self.deffered_actions[agent]}")
                    #print(f"res={res}")
                    #if need_sleep:
                    self.skip_full_observation[agent]=True
                    self.player_states[agent]=res
        else:
            raise Exception(f"Bad action id a={agent} action={action} of {self.action_lookup[agent]}")
        
        self.dones[agent] = True
        self._update_agent_state(agent)
        self._generate_agent_action_map(agent)
        #print(f"action_lookup2={action_lookup}")
        acts=len(self.action_lookup[agent].keys())
        if acts==1:
            #print(f"Substate for agent {agent}")
            self._step(agent,list(self.action_lookup[agent].keys())[0])

        reward+=self._calc_reward_by_metrics(agent,prev_player_metrics)
        return reward



    #action= {'1': 2774, '2': 6487}
    def step_all(self, actions):
        acts=[{agent:len(self.action_lookup[agent].keys())} for agent in self.action_lookup]
        for agent in actions:
            self.rewards[agent]=self._step(agent,actions[agent])
            self._update_agent_observation(agent)

        max_actions=max([len(self.action_lookup[agent].keys()) for agent in self.agents])
        is_terminate=False
        if max_actions==0:
            for ii in range(3):
                print(f"Warning {ii}, no actions: waiting 1 second and checking again... max_actions={max_actions}, actions={[len(self.action_lookup[agent].keys()) for agent in self.agents]}")

                if all([self.player_states[agent].get('game',{}).get('phase',"")=="end" for agent in self.agents]):
                    print("End game detected. Terminating...")
                    self.terminations = {agent: True for agent in self.agents}
                    is_terminate=True
                    break
                else:
                    if ii>0:
                        time.sleep(1)
                    self._update_all_observations(True)
                    max_actions=max([len(self.action_lookup[agent].keys()) for agent in self.agents])
                    if max_actions>0:
                        break
        
        if max_actions==0 and not is_terminate:
            for agent in self.agents:
                with open(os.path.join("data","failed_actions",''.join(random.choices(string.ascii_uppercase + string.digits, k=12))+".json"),'w',encoding='utf-8') as f:
                    f.write(json.dumps({
                        "player_link": f"{SERVER_BASE_URL}/player?id={self.agent_id_to_player_id[agent]}",
                        "player_id": self.agent_id_to_player_id[agent],
                        "error": "No steps",
                        "player_state":self.player_states[agent]
                    }))
            self.terminations = {agent: True for agent in self.agents}
            print(f"No steps players={[self.agent_id_to_player_id[agent] for agent in self.agents]}")
            if self.raise_exceptions:
                raise Exception(f"No steps players={[self.agent_id_to_player_id[agent] for agent in self.agents]}")
        

        #terms=[self.player_states[agent].get('game',{}).get('isTerraformed',False) for agent in self.agents]
        #phases=[self.player_states[agent].get('game',{}).get('phase',False) for agent in self.agents]
        
        return self.current_obs,self.rewards,self.terminations,self.truncations, self.infos


    # New single-agent step: only agent "1" is controlled externally.
    # Other agents act randomly if they have legal actions.
    # This method blocks until agent "1" has at least one legal action.
    def step(self, action, return_both_reward=False):
        main_agent = "1"
        #print(f"actions_num={self.action_lookup[main_agent].keys()}")
        prev_player_metrics={}

        action_map={main_agent:action}

        for agent in self.agents:
                prev_player_metrics[agent]=self._extract_player_metrics(agent)
                if agent == main_agent:
                    continue
                legal = list(self.action_lookup.get(agent, {}).keys())
                if len(legal) > 0:
                    action_map[agent] = random.choice(legal)
                else:
                    action_map[agent]=0
        self.step_all(action_map)
        self._update_all_observations(force=True)
        zz=0
        # Loop until main agent has available actions
        while zz<30:
            zz=zz+1
            # Ensure observations and action maps are current
            main_actions = len(self.action_lookup.get(main_agent, {}).keys())

            # If main agent has actions, execute step
            if main_actions > 0:
                break

            # Otherwise let other agents play randomly (no wait-action)
            random_actions = {}
            for agent in self.agents:
                if agent == main_agent:
                    continue

                legal = list(self.action_lookup.get(agent, {}).keys())
                if len(legal) > 0:
                    random_actions[agent] = random.choice(legal)

            # If no one has actions, check termination
            if len(random_actions) == 0:
                if all(self.player_states[agent].get('game',{}).get('phase',"") == "end" for agent in self.agents):
                    self.terminations = {agent: True for agent in self.agents}
                    players=self.player_states[main_agent].get('players',[])
                    for agent in self.agents:
                        self.rewards[agent]=0.0
                    if len(players)>1:
                        for p in players:
                            for agent in self.agents:
                                if p.get('id','')==self.agent_id_to_player_id[agent]:
                                    self.rewards[agent]=p.get('victoryPointsBreakdown',{}).get('total',0)/100.0

                    print(f"Step {SERVER_BASE_URL}/game?id={self.game_id} ... rewards={self.rewards} actions: {action} of {list(self.action_lookup[main_agent].keys())} isTerraformed={self.player_states[main_agent].get('game',{}).get('isTerraformed',False)} self.terminations={self.terminations} phases={self.player_states[main_agent].get('game',{}).get('phase',False)}")
                    if return_both_reward:
                        return {main_agent:self.current_obs[main_agent]},self.rewards,{main_agent: self.terminations[main_agent]},{main_agent: self.truncations[main_agent]},{main_agent: self.infos[main_agent]}
                    else:
                        return {main_agent:self.current_obs[main_agent]},{main_agent: self.rewards[main_agent]},{main_agent: self.terminations[main_agent]},{main_agent: self.truncations[main_agent]},{main_agent: self.infos[main_agent]}

                # Otherwise refresh state and continue loop
                time.sleep(0.2)
                continue

            # Execute random opponent actions
            self.step_all(random_actions)
        if self.terminations[main_agent] or self.terminations["2"]:
            for agent in self.agents:
                self.terminations[agent]=True
            players=self.player_states[main_agent].get('players',[])
            if len(players)>1:
                for p in players:
                    for agent in self.agents:
                        if p.get('id','')==self.agent_id_to_player_id[agent]:
                            self.rewards[agent]=p.get('victoryPointsBreakdown',{}).get('total',0)/100.0
        else:
            self.rewards={agent:self._calc_reward_by_metrics(agent,prev_player_metrics[agent]) for agent in self.agents}
        print(f"Step {SERVER_BASE_URL}/game?id={self.game_id} ... rewards={self.rewards} actions: {action} of {list(self.action_lookup[main_agent].keys())} isTerraformed={self.player_states[main_agent].get('game',{}).get('isTerraformed',False)} self.terminations={self.terminations} phases={self.player_states[main_agent].get('game',{}).get('phase',False)}")
        if return_both_reward:
            return {main_agent:self.current_obs[main_agent]},self.rewards,{main_agent: self.terminations[main_agent]},{main_agent: self.truncations[main_agent]},{main_agent: self.infos[main_agent]}
        else:
            return {main_agent:self.current_obs[main_agent]},{main_agent: self.rewards[main_agent]},{main_agent: self.terminations[main_agent]},{main_agent: self.truncations[main_agent]},{main_agent: self.infos[main_agent]}

    def render(self,render_mode):
        logging.debug("Rendering not implemented")

    def close(self):
        pass 
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    

def parallel_env(opponent=None):
    env = TerraformingMarsEnv(['1','2'])
    env.opponent_policy = opponent
    return env

