import numpy as np
import random
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import os
import json


def load_cards(filepath="terraforming_mars_cards_full.json", expansions=None):
    """
    Load card data from a JSON file and filter by expansions if provided.

    Args:
        filepath (str): Path to JSON file containing card data.
        expansions (list or None): Filter by expansions (e.g., ["Base", "Prelude"]).

    Returns:
        list of dict: Card entries
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Card file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cards = data.get("cards", [])
    if expansions:
        cards = [card for card in cards if card.get("expansion", "Base") in expansions]
    standard_projects = data.get("standard_projects", [])
    if expansions:
        standard_projects = [card for card in standard_projects if card.get("expansion", "Base") in expansions]
    return cards,standard_projects

def filter_playable_cards(cards, player_state, global_params):
    """
    Given player and global game state, filter cards that are playable.
    """
    def meets_requirements(card):
        reqs = card.get("requirements", {})
        temp = global_params.get("temperature", -30)
        oxy = global_params.get("oxygen", 0)
        venus = global_params.get("venus", 0)

        if reqs.get("temperature_min") is not None and temp < reqs["temperature_min"]:
            return False
        if reqs.get("oxygen_min") is not None and oxy < reqs["oxygen_min"]:
            return False
        if reqs.get("venus_min") is not None and venus < reqs["venus_min"]:
            return False
        if player_state.get("mc", 0) < card.get("cost", 0):
            return False
        return True

    return [card for card in cards if meets_requirements(card)]

def get_all_cards():
    return load_cards()

class TerraformingMarsEnv(AECEnv):
    def __init__(self, num_players=2, render_callback=None):
        super().__init__()
        self.num_players = num_players
        self.players = [self.create_player() for _ in range(num_players)]
        self.current_player = 0
        self.passed_players = set()
        self.generation = 1
        self.tiles = []
        self.ocean_positions = {}
        self.render_callback = render_callback
        self.deck = []
        all_cards,all_standard_projects=get_all_cards()
        self.all_cards=all_cards
        self.standard_projects=all_standard_projects
        self.draft_phase = True
        self.place_tile_phase = False
        self.place_entity_phase=False
        self.action_phase = False
        self.draft_max_available=100
        self.current_player_actions_left=2
        self.reset()

    def create_player(self):
        return {
            'mc': 40,
            'steel': 0,
            'titanium': 0,
            'plants': 0,
            'energy': 0,
            'heat': 0,
            'production': {'mc': 1, 'steel': 0, 'titanium': 0, 'plants': 0, 'energy': 0, 'heat': 0},
            'played_cards': [],
            'hand': [],
            'vp': 0,
            'terraform_rating': 20,
            'draft_hand': [],
            'available_draft_cards':1000,
            'available_free_draft_cards':0
        }

    def reset(self):
        self.generation = 1
        self.current_player = 0
        self.passed_players = set()
        self.map_width = 9
        self.map_height = 5
        self.tiles = [[{'type': 'empty', 'owner': None} for _ in range(self.map_width)] for _ in range(self.map_height)]
        self.ocean_positions = {(1, 1), (1, 7), (2, 3), (2, 5), (3, 1), (3, 7)}
        self.players = [self.create_player() for _ in range(self.num_players)]
        self.global_parameters = {'temperature': -30, 'oxygen': 0, 'oceans': 0}
        self.deck = self.all_cards.copy()
        self.draft_phase = True
        self.place_tile_phase = False
        self.place_entity_phase=False
        self.action_phase = False
        self.draft_max_available=100
        self.current_player_actions_left=2
        random.shuffle(self.deck)
        print(f"Deck {self.deck}")
        for player in self.players:
            player['draft_hand'] = [self.deck.pop() for _ in range(10)]
        self.done = False
        return self.observe()

    def observe(self):
        player = self.players[self.current_player]
        return np.array([
            self.global_parameters['temperature'],
            self.global_parameters['oxygen'],
            self.global_parameters['oceans'],
            player['terraform_rating'],
            player['mc'],
            player['heat'],
            player['plants']
        ], dtype=np.float32)

    def step(self, action):
        print(f"Step triggered action={action}")
        player = self.players[self.current_player]

        if action == "pass":
            self.player_pass()
            return self.observe(), 0, False, {}
        
        reward = self.handle_action(action)
        if reward<0:
            return self.observe(), reward, False, {}
        action_type = action.get('type')
        if action_type == "buy_card":
            return self.observe(), reward, False, {}

        if self.action_phase:
            self.current_player_actions_left-=1    
            if self.current_player_actions_left<=0:
                self.next_turn()
        else:
            self.next_turn()
        self.update_front()
        return self.observe(), reward, False, {}

    def can_play_card(self, player, card):
        reqs = card.get('requirements', {})
        temp = self.global_parameters['temperature']
        oxy = self.global_parameters['oxygen']

        if 'temperature_min' in reqs and temp < reqs['temperature_min']:
            return False
        if 'oxygen_min' in reqs and oxy < reqs['oxygen_min']:
            return False
        if player['mc'] < self.calc_card_cost(card):
            return False
        if card['effects']:
            for e in card['effects']:
                if e['type']=='global' and self.global_parameters['oceans']+e['amount']>9:
                    return False
        return True
    
    def calc_card_cost(self,card):
        return card['cost']

    def apply_card_effect(self, player, card):
        for effect in card.get("effects", []):
            etype = effect.get("type")
            if etype == "global":
                target = effect.get("target")
                amount = effect.get("amount", 0)
                if target in self.global_parameters:
                    self.global_parameters[target] += amount

            elif etype == "tr":
                player['terraform_rating'] += effect.get("amount", 0)

            elif etype == "production":
                resource = effect.get("resource")
                amount = effect.get("amount", 0)
                if resource in player['production']:
                    player['production'][resource] += amount

            elif etype == "draw" and self.deck:
                for _ in range(effect.get("amount", 1)):
                    if self.deck:
                        player['hand'].append(self.deck.pop())

            elif etype == "place_tile":
                # Mark intent — actual tile placement must be handled elsewhere in UI
                tile_type = effect.get("tile")
                print(f"[Effect] Would place tile: {tile_type} (UI required)")

            elif etype == "remove_plants":
                player['plants'] = max(0, player['plants'] - effect.get("amount", 0))

            elif etype == "discount":
                print(f"[Effect] Player receives discount on {effect.get('scope')} by {effect.get('amount')}")
                # Apply this to a session variable or future cost calculations

    def next_turn(self):
        self.current_player = (self.current_player + 1) % self.num_players
        if self.current_player==0 and self.draft_phase:
            self.action_phase=True
            self.draft_phase=False
            for p in self.players:
                p['draft_hand']=[]
        if len(self.passed_players) == self.num_players:
            self.end_generation()
        print(f"End turn. Current player: {self.current_player}")
        self.update_front()

    def player_pass(self):
        self.passed_players.add(self.current_player)
        self.next_turn()

    def end_generation(self):
        self.generation += 1
        self.passed_players = set()
        for player in self.players:
            player['heat'] += player['energy']
            player['energy'] = 0
            for res in ['mc', 'steel', 'titanium', 'plants', 'energy', 'heat']:
                player[res] += player['production'].get(res, 0)
            player['mc'] += player['terraform_rating']
            if self.deck:
                for i in range(4):
                    player['draft_hand'].append(self.deck.pop())
            player['available_draft_cards']=100
            player['available_free_draft_cards']=0
        self.action_phase=False
        self.draft_phase=True
        
        self.current_player = 0
        print(f"End generation done")

    def handle_action_play_card(self,player,action):
        name = action['card_name']
        for card in player['hand']:
            if card['name'] == name and self.can_play_card(player, card):
                if card['type']!='standard_project':
                    player['hand'].remove(card)
                    player['played_cards'].append(card)
                player['mc'] -= self.calc_card_cost(card=card)
                self.apply_card_effect(player, card)
                print(f"Played card {card}")
                return 1
        else:
            return -1

    def handle_action_place_tile(self,player,action):
        tile_type = action['tile_type']
        x, y = action['position']
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return -1

        tile = self.tiles[y][x]
        if tile['type'] != 'empty':
            return -1
        if tile_type == 'city':
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if 0 <= y+dy < self.map_height and 0 <= x+dx < self.map_width:
                        if self.tiles[y+dy][x+dx]['type'] == 'city':
                            return -1
            self.tiles[y][x] = {'type': tile_type, 'owner': self.current_player}
            player['terraform_rating'] += 1
            return 1
        if tile_type == 'ocean' and (y, x) not in self.ocean_positions:
            return -1
        self.tiles[y][x] = {'type': tile_type, 'owner': self.current_player}
        if tile_type in ['city', 'greenery', 'ocean']:
            player['terraform_rating'] += 1
        return 1
    
    def handle_action_buy_card(self, player, action):
        if not self.draft_phase:
            return -1

        if player['available_draft_cards']<1:
            return -1
        if player['available_free_draft_cards']>0:
            player['available_free_draft_cards']-=1
        else:
            if player['mc'] < 4:
                return -1
            player['mc'] -= 4
        name = action['card_name']
        for card in player['draft_hand']:
            if card['name'] == name:
                player['hand'].append(card)
                player['draft_hand'].remove(card)
                return 1
        return 1
    
    def handle_action(self, action):
        player = self.players[self.current_player]
        reward = 0
        action_type = action.get('type')
        print(f"Handle action started, type={action_type}")

        if action_type == 'play_card':
            reward=self.handle_action_play_card(player,action)
        elif action_type == 'place_tile':
            reward=self.handle_action_place_tile(player,action)
        elif action_type == 'buy_card':
            reward=self.handle_action_buy_card(player,action)
            print(f"Buy card done, reward={reward}")
        elif action_type == 'end_turn':
            pass
        else:
            reward = -1
        if reward<0:
            print(f"Invalid action: {action} by Player {self.current_player + 1}")
        print(f"Handle action done, type={action_type}, reward={reward}")
        return reward

    def update_front(self):
        if self.render_callback:
            self.render_callback()

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return np.zeros((500, 500, 3), dtype=np.uint8)

    def close(self):
        pass
