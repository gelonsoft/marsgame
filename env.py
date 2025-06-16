import numpy as np
import json
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from typing import List, Dict
from observe_function import observe  # assuming observe() is defined in another module
import requests
import random

SERVER_BASE_URL="http://localhost:9976"

MAX_ACTIONS = 20  # Must match observe()

def get_player_state(player_id):
    url = f"{SERVER_BASE_URL}/api/player"
    print(f"get_player_state {player_id}")
    response = requests.get(url,params={"id":player_id},headers={"content-type":"application/json"})
    response.raise_for_status()
    try:
        return response.json()
    except Exception as e:
        print(f"Bad get_player_state response:\n{response.text}")
        raise e

def post_player_input(game_id,player_id, player_input_model):
    url = f"{SERVER_BASE_URL}/player/input"
    print(f"post_player_input Request:---\n{json.dumps(player_input_model)}\n---\n")
    player_input_model['runId']=game_id
    response = requests.post(url, params={"id":player_id}, json=player_input_model)
    response.raise_for_status()
    try:
        resp=response.json()
        print(f"Response: post_player_input\n---\n{resp}\n---")
        return resp
    except Exception as e:
        print(f"Bad post_player_input response:\n{response.text}")
        raise e

def start_new_game(num_players):
    print("Start new game")
    url = f"{SERVER_BASE_URL}/api/creategame"
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
    return response.json()


class TerraformingMarsEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "terraforming_mars_aec_v0"}

    def __init__(self, agent_ids: List[str]):
        super().__init__()
        self.game_id=None
        self.spectator_id=None
        self.player_name_to_id={}
        self.agent_id_to_player_id={}
        self.possible_agents = agent_ids
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.action_spaces = {}
        self.agent_selection = self._agent_selector.reset()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.player_states={}
        self.action_lookup = {}
        self.reverse_action_lookup = {}  # For debugging
        self.observation_shape=None
        self.reset()
        #self.player_states = {agent: get_player_state(self.agent_id_to_player_id[agent]) for agent in self.agents}
        sample_obs = observe(None, json.dumps(self.player_states[self.agents[0]]))
        self.observation_shape = sample_obs.shape

        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=self.observation_shape, dtype=np.float32)
            for agent in self.agents
        }


        #self._update_all_observations()

    def _update_all_observations(self):
        self.current_obs = {}
        self.legal_actions = {}
        self.action_spaces = {}
        for agent in self.agents:
            self.player_states[agent] = get_player_state(self.agent_id_to_player_id[agent])
            json_str = json.dumps(self.player_states[agent])
            self.current_obs[agent] = observe(None, json_str)

            waiting_input = self.player_states[agent].get("waitingFor")
            action_map = {}

            if isinstance(waiting_input, dict) and "options" in waiting_input:
                options = waiting_input["options"]
                input_type = waiting_input.get("type")

                if input_type == "initialCards":
                    from itertools import combinations
                    max_comb = min(len(options), MAX_ACTIONS)
                    combo_index = 0
                    for r in range(1, max_comb + 1):
                        for combo in combinations(options, r):
                            if combo_index < MAX_ACTIONS:
                                action_map[combo_index] = {
                                    "type": input_type,
                                    "responses": [{"type": "card", "cards": list(combo)}]
                                }
                                combo_index += 1
                            else:
                                break
                elif input_type in ["standardProject", "action", "tile", "milestone", "award"]:
                    for i, option in enumerate(options[:MAX_ACTIONS]):
                        response = {"type": input_type}
                        if isinstance(option, dict):
                            response.update(option)
                        elif isinstance(option, str):
                            response["choice"] = option
                        action_map[i] = response
                else:
                    for i, option in enumerate(options[:MAX_ACTIONS]):
                        action_map[i] = {"type": input_type, "choice": option}

            elif isinstance(waiting_input, dict):
                action_map[0] = waiting_input

            self.legal_actions[agent] = list(action_map.keys())[:MAX_ACTIONS]
            self.action_lookup[agent] = {i: action_map[i] for i in self.legal_actions[agent]}
            self.reverse_action_lookup[agent] = action_map

            self.action_spaces[agent] = spaces.Discrete(len(self.action_lookup[agent]) or 1)

    def observe(self, agent):
        return self.current_obs[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        new_game_response=start_new_game(len(self.agents))
        self.game_id=new_game_response['id']
        self.spectator_id=new_game_response['spectatorId']
        self.player_name_to_id={}
        self.agent_id_to_player_id={}

        for i in range(len(self.agents)):
            agent=self.agents[i]
            player=new_game_response['players'][i]
            self.agent_id_to_player_id[agent]=player['id']
            self.player_name_to_id[player['id']]=player['color']
            self.dones[agent] = False
            self.rewards[agent] = 0.0
            self.infos[agent] = {}

        self._update_all_observations()

    def post_player_input(self,agent_id,player_input):
        post_player_input(self.game_id,self.agent_id_to_player_id[agent_id],player_input)

    def step(self, action):
        agent = self.agent_selection

        if self.dones[agent]:
            self.agent_selection = self._agent_selector.next()
            return

        player_input = self.action_lookup[agent].get(action)
        if player_input:
            print(f"[DEBUG] Agent {agent} selected input: {json.dumps(player_input, indent=2)}")
            self.post_player_input(agent, player_input)

        self._update_all_observations()

        # Placeholder reward logic
        self.rewards[agent] = 1.0 if player_input else 0.0
        self.dones[agent] = True

        self.agent_selection = self._agent_selector.next()
        if all(self.dones.values()):
            self.agents = []

    def render(self):
        print("Rendering not implemented")

    def close(self):
        pass 


if __name__ == '__main__':
    env=TerraformingMarsEnv(["1","2"])
    print(f"Agents and players: {env.agent_id_to_player_id}")
    print(f"Spectator id: {env.spectator_id}")
    print(f"Game id: {env.game_id}")
    print(f"Actions: {json.dumps(env.action_lookup,indent=4)}")
    print(f"Observes: {env.current_obs}")
    env.step(1)