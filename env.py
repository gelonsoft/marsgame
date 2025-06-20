from time import sleep
import traceback
import numpy as np
import json
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv, ParallelEnv
try:
    from pettingzoo.utils import agent_selector
    x=agent_selector(['1','2'])
except:
    from pettingzoo.utils import AgentSelector
    x=AgentSelector(['1','2'])
from typing import List, Dict
from myconfig import MAX_ACTIONS
from observe_gamestate import get_actions_shape, observe  # assuming observe() is defined in another module
import requests
import random
from decision_mapper import TerraformingMarsDecisionMapper
import logging
import os

logging.basicConfig(level=logging.INFO)  # Set the logging level to DEBUG

SERVER_BASE_URL=os.environ.get('SERVER_BASE_URL',"http://lev-rworker-3:9976")

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
        return response_json
    except Exception as e:
        print(f"Bad get_player_state response:\n{response.text}")
        raise e

def post_player_input(run_id,player_id, player_input_model):
    url = f"{SERVER_BASE_URL}/player/input"
    logging.debug(f"post_player_input Request:---\n{json.dumps(player_input_model)}\n---\n")
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
        logging.error(f"Bad post_player_input response:\n{response.text}\npayload:\n{json.dumps(player_input_model)}\n")
        return None

def start_new_game(num_players):
    
    url = f"{SERVER_BASE_URL}/api/creategame"
    global request_responses,request_number
    if USE_MOCK_SERVER:
        response=request_responses[str(request_number)]["response"]
        logging.debug(f"Request url={url} {num_players} request_mock={request_responses[str(request_number)]['request']}")
        request_number+=1
        return response
    response = requests.put(url, json={
    "players": [
        {
            "name": "Red",
            "color": "red",
            "beginner": False,
            "handicap": 0,
            "first": False
        },
        {
            "name": "Green",
            "color": "green",
            "beginner": False,
            "handicap": 0,
            "first": False
        }
    ],
    "expansions": {
        "corpera": True,
        "promo": True,
        "venus": True,
        "colonies": True,
        "prelude": True,
        "prelude2": False,
        "turmoil": True,
        "community": False,
        "ares": False,
        "moon": False,
        "pathfinders": False,
        "ceo": False,
        "starwars": False,
        "underworld": False
    },
    "draftVariant": True,
    "showOtherPlayersVP": False,
    "customCorporationsList": [],
    "customColoniesList": [],
    "customPreludes": [],
    "bannedCards": [],
    "includedCards": [],
    "board": "tharsis",
    "seed": random.random() ,
    "solarPhaseOption": True,
    "aresExtremeVariant": False,
    "politicalAgendasExtension": "Standard",
    "undoOption": False,
    "showTimers": True,
    "fastModeOption": False,
    "removeNegativeGlobalEventsOption": False,
    "includeFanMA": False,
    "modularMA": False,
    "startingCorporations": 2,
    "soloTR": False,
    "initialDraft": False,
    "preludeDraftVariant": True,
    "randomMA": "No randomization",
    "shuffleMapOption": False,
    "randomFirstPlayer": True,
    "requiresVenusTrackCompletion": False,
    "requiresMoonTrackCompletion": False,
    "moonStandardProjectVariant": False,
    "moonStandardProjectVariant1": False,
    "altVenusBoard": False,
    "escapeVelocityMode": False,
    "escapeVelocityBonusSeconds": 2,
    "twoCorpsVariant": False,
    "customCeos": [],
    "startingCeos": 3,
    "startingPreludes": 4
})
    response.raise_for_status()
    response_json=response.json()
    if LOG_REQUESTS:
        request_responses[str(request_number)]={"request":response.request.url,"method":"get","response":response_json}
    request_number+=1
    logging.info(f"Started new game {SERVER_BASE_URL}/spectator?id={response_json['spectatorId']}")
    return response_json


class TerraformingMarsEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "terraforming_mars_aec_v0","is_parallelizable":True}

    def __init__(self, agent_ids: List[str], init_from_player_state=False, player_state=None):
        super().__init__()
        self.decision_mapper=TerraformingMarsDecisionMapper(None)
        
        self.game_id=None
        self.init_from_player_state=init_from_player_state
        self.start_player_state=player_state
        self.observations_made=0
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
        self.player_states={}
        self.action_lookup = {}
        self.reverse_action_lookup = {}  # For debugging
        self.observation_shape=None
        self.render_mode='human'
        self.terminations= {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.reset()
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
            if self.init_from_player_state and self.observations_made==0:
                self.player_states[agent] = self.start_player_state
            else:
                self.player_states[agent] = get_player_state(self.agent_id_to_player_id[agent])
            action_map = self.decision_mapper.generate_action_space(self.player_states[agent].get("waitingFor"))  # Initialize the action map
            self.current_obs[agent] = observe(self.player_states[agent],action_map)
            #print(f"Action map: \n{json.dumps(action_map,indent=2)}")
            self.legal_actions[agent] = list(action_map.keys())[:MAX_ACTIONS]
            self.action_lookup[agent] = {i: action_map[i] for i in self.legal_actions[agent]}
            self.reverse_action_lookup[agent] = action_map

            self.action_spaces[agent] = spaces.Discrete(len(self.action_lookup[agent]) or 1)
        self.observations_made+=1

    def observe(self, agent):
        return self.current_obs[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        try:
            self._agent_selector = agent_selector(self.agents)
        except:
            self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        new_game_response=None
        if self.init_from_player_state:
            self.spectator_id=self.start_player_state['game']['spectatorId']
            new_game_response=self.start_player_state
        else:
            new_game_response=start_new_game(len(self.agents))
            self.game_id=new_game_response['id']
            #self.run_id=new_game_response['runId']
            self.spectator_id=new_game_response['spectatorId']
        self.player_name_to_id={}
        self.agent_id_to_player_id={}

        for i in range(len(self.agents)):
            agent=self.agents[i]
            if self.init_from_player_state:
                self.agent_id_to_player_id[agent]=new_game_response['id']
                self.player_name_to_id[new_game_response['id']]=new_game_response['thisPlayer']['color']
            else:
                player=new_game_response['players'][i]
                self.agent_id_to_player_id[agent]=player['id']
                self.player_name_to_id[player['id']]=player['color']
            self.dones[agent] = False
            self.rewards[agent] = 0.0
            self.infos[agent] = {}

        self._update_all_observations()
        return self.current_obs,self.infos
    
    def get_action_mask(self):
        return [len(self.legal_actions[agent]) for agent in self.agents]


    def post_player_input(self,agent_id,player_input):
        return post_player_input(self.player_states[agent_id]['runId'],self.agent_id_to_player_id[agent_id],player_input)

    #action= {'1': 2774, '2': 6487}
    def step(self, actions):
        for agent in actions:
            action=actions[agent]
            #print(f"Action: {action}")
            player_input = self.action_lookup[agent].get(action)
            if player_input:
                print(f"Agent {agent} selected input: {player_input.get('type')}")
                res=self.post_player_input(agent, player_input)
                if res is None:
                    print(f"Failed to post player input for agent {agent} with input player_link={SERVER_BASE_URL}/player?id={self.agent_id_to_player_id[agent]}: \n{json.dumps(player_input, indent=2)}\n and waiting steps \n{json.dumps(self.player_states[agent].get('waitingSteps',{}), indent=2)}\n")
                    raise Exception("Bad player actions")
                    player_input=None

            self.rewards[agent] = 1.0 if player_input else -1.0
            self.dones[agent] = True

        self._update_all_observations()

        # Placeholder reward logic

        #if all(self.dones.values()):
        #    self.agents = []
        #obs, rewards, terminations, truncations, infos
        print(f"Executing step {self.game_id}... rewards={self.rewards} actions: {actions} of {[{agent:len(self.action_lookup[agent].keys())} for agent in self.action_lookup]}")
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

if __name__ == '__main__':
    env=TerraformingMarsEnv(["1","2"])
    print(f"Agents and players: {env.agent_id_to_player_id}")
    print(f"Spectator id: {SERVER_BASE_URL}/spectator?id={env.spectator_id}")
    print(f"Game id: {env.game_id}")
    #print(f"Actions: {json.dumps(env.action_lookup,indent=4)}")
    print(f"Observes: {env.current_obs}")
    print(f"Observes len: {len(env.current_obs['1'])}")
    sleep(10)
    env.step(1)
    sleep(10)
    env.step(2)
    sleep(10)
    #http://localhost:9976/spectator?id=se97c54ea92af
    if not USE_MOCK_SERVER and LOG_REQUESTS:
        with open('response.json', 'w') as f:
            f.write(json.dumps(request_responses,indent=2))