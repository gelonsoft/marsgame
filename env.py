import numpy as np
import random
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import os
import json
from gymnasium import spaces


MILESTONES = {
    "Builder": lambda p: sum(p['production'].values()) >= 6,
    "Gardener": lambda p: p['plants'] >= 6,
    "Planner": lambda p: len(p['played_cards']) >= 16
}

AWARDS = ["Landlord", "Banker", "Thermalist"] 

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
    corporations = data.get("corporations", [])
    if expansions:
        corporations = [card for card in corporations if card.get("expansion", "Base") in expansions]
    return cards,standard_projects,corporations

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
        all_cards,all_standard_projects,all_corporations=get_all_cards()
        self.all_corporations=all_corporations
        self.corporations=[]
        self.all_cards=all_cards
        self.standard_projects=all_standard_projects
        self.draft_phase = True
        self.place_tile_phase = False
        self.place_entity_phase=False
        self.action_phase = False
        self.draft_max_available=100
        self.current_player_actions_left=2
        self.claimed_milestones = []
        self.funded_awards = []
        self.max_milestones = 3
        self.max_awards = 3
        self.milestones=MILESTONES
        self.awards=AWARDS
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
            'available_free_draft_cards':0,
            'corporation':None,
            'corporation_choices':[],
            'played_active_cards_in_round':{}
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
        self.corporations=self.all_corporations.copy()
        print(f"Corporations: {self.corporations}")
        random.shuffle(self.corporations)
        for player in self.players:
            player['corporation_choices'] = [self.corporations.pop() for _ in range(2)]
            player['corporation'] = None
        self.global_parameters = {'temperature': -30, 'oxygen': 0, 'oceans': 0}
        self.deck = self.all_cards.copy()
        self.draft_phase = False
        self.place_tile_phase = False
        self.place_entity_phase=False
        self.action_phase = False
        self.choose_corporation_phase=True
        self.draft_max_available=100
        self.current_player_actions_left=2
        random.shuffle(self.deck)
        for player in self.players:
            player['draft_hand'] = [self.deck.pop() for _ in range(10)]
        self.done = False
        self.claimed_milestones = []
        self.funded_awards = []
        self.max_milestones = 3
        self.max_awards = 3
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


    def get_observation_space():
        return spaces.Dict({
            "global_parameters": spaces.Dict({
                "temperature": spaces.Box(low=-30, high=8, shape=(), dtype=np.float32),
                "oxygen": spaces.Box(low=0, high=14, shape=(), dtype=np.float32),
                "oceans": spaces.Box(low=0, high=9, shape=(), dtype=np.float32)
            }),
            "player": spaces.Dict({
                "terraform_rating": spaces.Box(low=0, high=100, shape=(), dtype=np.float32),
                "resources": spaces.Dict({
                    "mc": spaces.Box(0, 1000, shape=(), dtype=np.float32),
                    "steel": spaces.Box(0, 100, shape=(), dtype=np.float32),
                    "titanium": spaces.Box(0, 100, shape=(), dtype=np.float32),
                    "plants": spaces.Box(0, 100, shape=(), dtype=np.float32),
                    "energy": spaces.Box(0, 100, shape=(), dtype=np.float32),
                    "heat": spaces.Box(0, 200, shape=(), dtype=np.float32),
                }),
                "production": spaces.Dict({
                    "mc": spaces.Box(0, 10, shape=(), dtype=np.float32),
                    "steel": spaces.Box(0, 10, shape=(), dtype=np.float32),
                    "titanium": spaces.Box(0, 10, shape=(), dtype=np.float32),
                    "plants": spaces.Box(0, 10, shape=(), dtype=np.float32),
                    "energy": spaces.Box(0, 10, shape=(), dtype=np.float32),
                    "heat": spaces.Box(0, 10, shape=(), dtype=np.float32),
                }),
                "hand_size": spaces.Discrete(100),
                "draft_hand_size": spaces.Discrete(10),
                "played_cards": spaces.MultiBinary(200),  # assume 200 unique cards max
                "milestones": spaces.MultiBinary(len(MILESTONES)),
                "funded_awards": spaces.MultiBinary(len(AWARDS)),
                "corporation": spaces.Discrete(20)  # max 20 corporations assumed
            }),
            "tiles": spaces.Box(low=0, high=3, shape=(5, 9), dtype=np.int8),
            "claimed_milestones": spaces.MultiBinary(len(MILESTONES)),
            "funded_awards": spaces.MultiBinary(len(AWARDS)),
            "phase": spaces.Dict({
                "draft": spaces.Discrete(2),
                "action": spaces.Discrete(2),
                "place_entity": spaces.Discrete(2)
            }),
            "generation": spaces.Discrete(25),
            "current_player": spaces.Discrete(5)
        })

    def get_action_space():
        return spaces.Dict({
            "type": spaces.Discrete(6),  # 0=play_card, 1=place_tile, 2=buy_card, 3=end_turn, 4=active_card_action, 5=milestone/award
            "card_index": spaces.Discrete(100),
            "tile_type": spaces.Discrete(3),  # 0=greenery, 1=city, 2=ocean
            "tile_position": spaces.MultiDiscrete([9, 5]),
            "milestone_index": spaces.Discrete(len(MILESTONES)),
            "award_index": spaces.Discrete(len(AWARDS))
        })


    def compute_final_scores(self):
        scores = []
        for i, player in enumerate(self.players):
            tr = player['terraform_rating']
            vp_cards = sum(c.get('effects', [{}])[0].get('amount', 0) for c in player['played_cards'] if 'vp' in str(c))
            milestone_vp = len(player.get('milestones', [])) * 5
            award_vp = 0

            for award in self.funded_awards:
                # Simplified scoring — award 5 VP to most qualifying player
                if award == "Landlord":
                    tiles = sum(row.count({'type': 'city', 'owner': i}) for row in self.tiles)
                    award_vp += 5 if tiles >= 2 else 0
                elif award == "Banker":
                    if player['production'].get('mc', 0) >= 4:
                        award_vp += 5
                elif award == "Thermalist":
                    if player['heat'] >= 8:
                        award_vp += 5

            total = tr + vp_cards + milestone_vp + award_vp
            scores.append((f"Player {i+1}", total))
        return scores

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
        if player['played_active_cards_in_round'][card['name']]:
            return False
        return True
    
    def calc_card_cost(self, card):
        cost = card['cost']
        tags = set(card.get('tags', []))
        player = self.players[self.current_player]
        discounts = player.get('discounts', {})
        for tag in tags:
            if tag in discounts:
                cost -= discounts[tag]
        return max(0, cost)
    
    def apply_corporation_effect(self, player, effect):
        etype = effect.get("type")
        if etype == "production":
            res = effect["resource"]
            amt = effect["amount"]
            player['production'][res] += amt
        elif etype == "gain":
            res = effect["resource"]
            amt = effect["amount"]
            player[res] += amt
        elif etype == "discount":
            if 'discounts' not in player:
                player['discounts'] = {}
            scope = effect['scope']
            amt = effect['amount']
            player['discounts'][scope] = amt
        elif etype == "conversion":
            player['conversion'] = effect

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
        print("Next turn started")
        self.current_player = (self.current_player + 1) % self.num_players
        if self.current_player==0 and self.draft_phase:
            self.action_phase=True
            self.draft_phase=False
            for p in self.players:
                p['draft_hand']=[]
        
        if self.current_player==0 and self.choose_corporation_phase:
            self.choose_corporation_phase=False
            self.current_player_actions_left=1
            self.draft_phase=True
        else:
            self.current_player_actions_left=2
        
        
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
            player['played_active_cards_in_round']={}
        self.action_phase=False
        self.draft_phase=True
        
        
        self.current_player = 0
        print(f"End generation done")

    def handle_action_play_card(self,player,action):
        name = action['card_name']
        for card in player['hand'] + self.standard_projects:
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
    
    def handle_action_choose_corporation(self,player,action):
        corp_name = action['name']
        for corp in player['corporation_choices']:
            if corp['name'] == corp_name:
                player['corporation'] = corp['name']
                player['mc'] = corp['mc']
                for effect in corp['effect']:
                    self.apply_corporation_effect(player, effect)
                return 1
        return -1
        
    def handle_active_card_action(self, player, action):
        card_name=action['card_name']
        for card in player['played_cards']:
            if card['name'] != card_name or card.get('type') != 'active':
                continue

            for ae in card.get('active_effects', []):
                if ae['trigger'] != 'action':
                    continue

                # Respect optional per-card limits
                limit = ae.get('limit')
                res = ae['effect'].get('resource', 'science')
                if 'resources' not in card:
                    card['resources'] = {}
                if limit is not None and card['resources'].get(res, 0) >= limit:
                    return -1  # Already reached the limit

                # Apply the effect
                effect = ae['effect']
                etype = effect['type']
                amt = effect.get('amount', 1)
                reward=-1
                if etype == 'add_science':
                    card['resources'][res] = card['resources'].get(res, 0) + amt
                    reward= 1

                elif etype == 'draw':
                    for _ in range(amt):
                        if self.deck:
                            player['hand'].append(self.deck.pop())
                    reward= 1

                elif etype == 'gain':
                    resource = effect['resource']
                    player[resource] += amt
                    reward= 1
                
                if reward>0:
                    player['played_active_cards_in_round'][card['name']]=True
            return -1
        return -1

    def handle_action_claim_milestone(self, player, action):
        name=action['name']
        if name in self.claimed_milestones:
            return -1
        if len(self.claimed_milestones) >= self.max_milestones:
            return -1
        if player['mc'] < 8:
            return -1
        condition = MILESTONES.get(name)
        if not condition or not condition(player):
            return -1
        player['mc'] -= 8
        self.claimed_milestones.append(name)
        player.setdefault('milestones', []).append(name)
        return 1

    def handle_action_fund_award(self, player, action):
        name=action['name']
        if name in self.funded_awards:
            return -1
        if len(self.funded_awards) >= self.max_awards:
            return -1
        if player['mc'] < 8:
            return -1
        player['mc'] -= 8
        self.funded_awards.append(name)
        player.setdefault('funded_awards', []).append(name)
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
        elif action_type == 'choose_corporation':
            reward=self.handle_action_choose_corporation(player,action)
        elif action_type == 'active_card_action':
            return self.handle_active_card_action(player, action)
        elif action_type == 'claim_milestone':
            return self.handle_action_claim_milestone(player, action)
        elif action_type == 'fund_award':
            return self.handle_action_fund_award(player, action)
        elif action_type == 'buy_card':
            reward=self.handle_action_buy_card(player,action)
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
