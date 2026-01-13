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
    print(f"post_player_input Request:---\n{json.dumps(player_input_model)}\n---\n")
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

    def __init__(self, agent_ids: List[str], init_from_player_state=False, player_id=None,player_state=None,waiting_for=None):
        super().__init__()
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
        #self.player_states = {agent: get_player_state(self.agent_id_to_player_id[agent]) for agent in self.agents}
        sample_obs = self.current_obs[self.agents[0]] #observe(self.player_states[self.agents[0]],self.decision_mapper.generate_action_space(self.player_states[agent].get("waitingFor")))
        self.observation_shape = sample_obs.shape
        self.action_shape=get_actions_shape()

        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=self.observation_shape, dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=self.action_shape, dtype=np.float32)
            for agent in self.agents
        }


        #self._update_all_observations()

    def _update_all_observations(self):
        self.current_obs = {}
        self.legal_actions = {}
        self.action_spaces = {}
        for agent in self.agents:
            if self.start_player_state and self.observations_made==0:
                self.player_states[agent] = self.start_player_state
            else:
                if self.deffered_actions[agent] is None:
                    self.player_states[agent] = get_player_state(self.agent_id_to_player_id[agent])
                    
                    #print(f"Updated state for agent {agent}")
                    #z=[c['name'] for c in self.player_states[agent]['cardsInHand']]
                    #print(f"cards in hand: {z}")
                #print(f"self.observations_made={self.observations_made}")
                if self.waiting_for and self.observations_made==0:
                    self.player_states[agent]["waitingFor"] =self.waiting_for
            action_map={}
            if self.deffered_actions[agent] is None:
                action_map = self.decision_mapper.generate_action_space(self.player_states[agent].get("waitingFor"),self.player_states[agent],True,None)  # Initialize the action map
            else:
                action_map = self.decision_mapper.generate_action_space(self.player_states[agent].get("waitingFor"),self.player_states[agent],True,self.deffered_actions[agent])  # Initialize the action map
            
            self.current_obs[agent] = observe(self.player_states[agent],action_map)
            global mmax_actions
            if len(action_map.keys())>mmax_actions:
                with open(os.path.join("data","max_actions",f"{mmax_actions}.json"),'w',encoding='utf-8') as f:
                        f.write(json.dumps({
                            "action_map": action_map,
                            "player_state":self.player_states[agent]
                        }))
                mmax_actions=len(action_map.keys())
            #print(f"Action map: \n{json.dumps(action_map,indent=2)}")
            self.legal_actions[agent] = list(action_map.keys())[:MAX_ACTIONS]
            self.action_lookup[agent] = {i: action_map[i] for i in self.legal_actions[agent]}
            self.reverse_action_lookup[agent] = action_map

            self.action_spaces[agent] = spaces.Discrete(len(self.action_lookup[agent]) or 1)
        self.observations_made+=1

    def observe(self, agent):
        return self.current_obs[agent]

    def _extract_player_metrics(self, agent):
        p = self.player_states[agent]["thisPlayer"]

        tr = p["terraformRating"]

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
            "production": production
        }
    def reset(self, seed=None, options=None):
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
            self.spectator_id=self.start_player_state['game']['spectatorId']
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
    
    def get_action_mask(self):
        return [len(self.legal_actions[agent]) or 1 for agent in self.agents]


    def post_player_input(self,agent_id,player_input):
        return post_player_input(self.player_states[agent_id]['runId'],self.agent_id_to_player_id[agent_id],player_input)

    #action= {'1': 2774, '2': 6487}
    def step(self, actions):
        old_player_input=None
        self.prev_player_metrics = {
            agent: self._extract_player_metrics(agent)
            for agent in self.agents
        }
        
        acts=[{agent:len(self.action_lookup[agent].keys())} for agent in self.action_lookup]
        for agent in actions:
            need_sleep=False
            action=int(actions[agent])
            #if action==0:
            #    self.dones[agent] = True
            #    self.rewards[agent] = -5 if len(self.action_lookup[agent].keys())<=0 else -7
            #    continue
            
            
            #action=action-1
            
            

            player_input = self.action_lookup[agent].get(action)
            #print(f"Action: {action}/{len(self.action_lookup[agent].keys())} player_input={player_input}")
            if player_input:
                if self.deffered_actions[agent] is None:
                    first_deffered_action=find_first_with_nested_attr(player_input,"__deferred_action")
                    if first_deffered_action is not None:
                        #print(f"new deffered action: {player_input}")
                        self.deffered_actions[agent]=player_input
                        self.rewards[agent]=0
                        continue
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
                        continue
                    else:
                        old_player_input=player_input
                        player_input=self.deffered_actions[agent]
                        #print(f"Agent {agent}")
                        #print(f"parent: {parent}")
                        #print(f"deffered: {deffered}")
                        #print(f"old_player_input: {old_player_input}")
                        #print(f"player_input: {player_input}")
                        
                        self.deffered_actions[agent]=None


                #print(f"Agent {agent} selected input: {player_input}")
                res=self.post_player_input(agent, player_input)
                if need_sleep:
                    sleep(0.3)
                if isinstance(res,str) and  old_player_input is not None:
                    #print(f"res.text={res}")
                    pass
                if isinstance(res,str):
                    self.rewards[agent] = -0.05
                    print(f"Failed to post player input for agent {agent} with input player_link={SERVER_BASE_URL}/player?id={self.agent_id_to_player_id[agent]}: \n{json.dumps(player_input, indent=2)}\n and waiting steps \n{json.dumps(self.player_states[agent].get('waitingSteps',{}), indent=2)}\n")
                    if 'already exists' in res:
                        sleep(0.5)
                        continue
                    with open(os.path.join("data","failed_actions",''.join(random.choices(string.ascii_uppercase + string.digits, k=12))+".json"),'w',encoding='utf-8') as f:
                        f.write(json.dumps({
                            "player_link": f"{SERVER_BASE_URL}/player?id={self.agent_id_to_player_id[agent]}",
                            "player_id": self.agent_id_to_player_id[agent],
                            "player_input":player_input,
                            "error":res,
                            "player_state":self.player_states[agent]
                        }))
                    #raise Exception("Bad player actions")
                    player_input=None
                    self.rewards[agent]=-5
                else:
                    self.rewards[agent]=0

            self.dones[agent] = True

        self._update_all_observations()
        for agent in self.agents:
            prev = self.prev_player_metrics.get(agent)
            if prev is None:
                continue

            try:
                curr = self._extract_player_metrics(agent)

                delta_tr = curr["tr"] - prev["tr"]
                delta_prod = curr["production"] - prev["production"]

                # Shaping
                self.rewards[agent] += 0.01                     # valid move bonus
                self.rewards[agent] += 0.1 * delta_tr           # TR shaping
                self.rewards[agent] += 0.02 * delta_prod        # production shaping

            except Exception:
                pass
            if self.rewards[agent]<0:
                if self.rewards[agent]==-5:
                    self.rewards[agent]=0.0
                elif self.rewards[agent]==-7:
                    self.rewards[agent]=-1.0
                continue
            
            try:
                vp = int(self.player_states[agent]['thisPlayer']['victoryPointsBreakdown']['total'])-20
                if vp<0:
                    vp=0
                self.rewards[agent] += vp / 1000.0   # scaled terminal reward
                self.rewards[agent] += 0.01
                #self.rewards[agent]=self.rewards[agent]
                if self.rewards[agent]>1:
                    self.rewards[agent]=1.0
                if self.rewards[agent]<-1:
                    self.rewards[agent]=-1
                #print(f"VP={self.player_states[agent]['thisPlayer']['victoryPointsBreakdown']}")
            except Exception as e:
                #print(f"Exception vp= {e}")
                self.rewards[agent]=0
        
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
                    time.sleep(1)
                    self._update_all_observations()
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
                        "player_input":player_input,
                        "player_state":self.player_states[agent]
                    }))
            self.terminations = {agent: True for agent in self.agents}
            print(f"No steps players={[self.agent_id_to_player_id[agent] for agent in self.agents]}")
            #raise Exception(f"No steps players={[self.agent_id_to_player_id[agent] for agent in self.agents]}")


        # Placeholder reward logic

        #if all(self.dones.values()):
        #    self.agents = []
        #obs, rewards, terminations, truncations, infos
        terms=[self.player_states[agent].get('game',{}).get('isTerraformed',False) for agent in self.agents]
        
        phases=[self.player_states[agent].get('game',{}).get('phase',False) for agent in self.agents]
        print(f"Executing step {self.game_id}... rewards={self.rewards} actions: {actions} of {acts} isTerraformed={terms} self.terminations={self.terminations} phases={phases}")

        return self.current_obs,self.rewards,self.terminations,self.truncations, self.infos

    def render(self,render_mode):
        logging.debug("Rendering not implemented")

    def close(self):
        pass 
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    

def parallel_env():
    return TerraformingMarsEnv(['1','2'])

