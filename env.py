import numpy as np
import random
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import os
import json
from gymnasium import spaces
import enum
import uuid


DEBUG=1

MILESTONES = {
    "Builder": lambda p: sum(p['production'].values()) >= 6,
    "Gardener": lambda p: p['plants'] >= 6,
    "Planner": lambda p: len(p['played_cards']) >= 16
}

AWARDS = ["Landlord", "Banker", "Thermalist"] 

# Action type map
ACTION_TYPES = {
    0: 'play_card',
    1: 'place_tile',
    2: 'buy_card',
    3: 'end_turn',
    4: 'active_card_action',
    5: 'claim_milestone',
    6: 'fund_award'
}

@enum.unique
class GamePhase(enum.Enum):
    choose_corp=1
    prelude_draft=2
    prelude_play=3
    start_draft=4
    main_action=5
    new_round_draft=6
    deffered_actions=7
    draft_card=9
    discount_card=10
    last_actions=11

def load_cards(filepath="terraforming_mars_cards_full.json", expansions=None):
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

def load_maps(filepath="terraforming_mars_maps.json"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Card file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("maps")

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
    def __init__(self, num_players=2, render_callback=None,seed=42):
        super().__init__()
        self.all_maps=load_maps()
        self.map=[]
        self.num_players = num_players
        self.players = []
        self.current_player = 0
        self.passed_players = set()
        self.generation = 1
        self.render_callback = render_callback
        self.deck = []
        self.phase=GamePhase.choose_corp
        all_cards,all_standard_projects,all_corporations=get_all_cards()
        self.all_corporations=all_corporations
        self.corporations=[]
        self.all_cards=all_cards
        self.standard_projects=all_standard_projects
        self.draft_max_available=100
        self.current_player_actions_left=2
        self.claimed_milestones = []
        self.funded_awards = []
        self.max_milestones = 3
        self.max_awards = 3
        self.milestones=MILESTONES
        self.awards=AWARDS
        self.discard_pile = []
        self.seed=seed
        self.reset()
        
    @property
    def draft_phase(self):
        return self.phase==GamePhase.start_draft or self.phase==GamePhase.new_round_draft
    
    @property
    def choose_corporation_phase(self):
        return self.phase==GamePhase.choose_corp
    
    @property
    def action_phase(self):
        return self.phase==GamePhase.main_action
    
    @property
    def place_entity_phase(self):
        return self.phase==GamePhase.place_resource
    
    @property
    def place_tile_phase(self):
        return self.phase==GamePhase.place_tile

    @property
    def deffered_actions_phase(self):
        return self.phase==GamePhase.deffered_actions
        
    def reset(self):
        self.generation = 1
        self.first_player=int(random.Random(self.seed).random()*self.num_players)
        self.current_player = self.first_player
        self.passed_players = set()
        self.map=self.all_maps[int(random.Random(self.seed).random()*len(self.all_maps))]['tiles'].copy()
        self.players = [self.create_player() for _ in range(self.num_players)]
        self.corporations=self.all_corporations.copy()
        random.Random(self.seed).shuffle(self.corporations)
        for player in self.players:
            player['corporation_choices'] = [self.corporations.pop() for _ in range(2)]
            player['corporation'] = None
        self.global_parameters = {'temperature': -30, 'oxygen': 0, 'oceans': 0}
        self.deck = self.all_cards.copy()
        self.draft_max_available=100
        self.current_player_actions_left=2
        random.Random(self.seed).shuffle(self.deck)
        for player in self.players:
            player['draft_hand'] = [self.get_card_from_deck() for _ in range(10)]
        self.claimed_milestones = []
        self.funded_awards = []
        self.max_milestones = 3
        self.deffered_player_actions=[]
        self.max_awards = 3
        self.discard_pile=[]
        if DEBUG==1:
            player=self.players[self.current_player]
            for p in self.players:
                p['corporation']=self.corporations.pop()
            player['hand'].extend([c for c in self.all_cards if c['name']=="Lake Marineris" ])
            player['hand'].extend([c for c in self.all_cards if c['name']=="Nuclear Power" ])
            self.phase=GamePhase.main_action
            
        return self.observe()

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
            'played_active_cards_in_round':{},
            'funded_awards':[]
        }

    def get_map_tile_by_coord(self,x,y):
        for t in self.map:
            if t['x']==x and t['y']==y:
                return t
        return None
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
        
        print(f"Check len(deffered_player_actions)={len(self.deffered_player_actions)}")
        if len(self.deffered_player_actions)>0:
            self.phase=GamePhase.deffered_actions
            self.update_front()
            return self.observe(), reward, False, {}
        elif self.phase==GamePhase.deffered_actions:
            self.phase=GamePhase.main_action 
            self.current_player_actions_left-=1  
            if self.current_player_actions_left<=0:
                self.next_turn()
            
        if action_type == "buy_card":
            return self.observe(), reward, False, {}

        if action_type=='place_tile':
            return self.observe(), reward, False, {}

        if self.action_phase:
            self.current_player_actions_left-=1    
            if self.current_player_actions_left<=0:
                self.next_turn()
        else:
            self.next_turn()
        self.update_front()
        return self.observe(), reward, False, {}

    def get_observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 1 + 6 + 6 + 2 + 200 + len(MILESTONES)*2 + 1 + 45 + len(AWARDS)*2 + 3 + 2,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(low=0, high=255, shape=(3,), dtype=np.int32)

    def encode_observation(self,flat_obs):
        obs = {}
        i = 0

        # Global parameters
        obs['global_parameters'] = {
            'temperature': flat_obs[i],
            'oxygen': flat_obs[i+1],
            'oceans': flat_obs[i+2]
        }
        i += 3

        # Terraform rating
        tr = flat_obs[i]
        i += 1

        # Resources
        resources = dict(zip(['mc', 'steel', 'titanium', 'plants', 'energy', 'heat'], flat_obs[i:i+6]))
        i += 6

        # Production
        production = dict(zip(['mc', 'steel', 'titanium', 'plants', 'energy', 'heat'], flat_obs[i:i+6]))
        i += 6

        # Hand sizes
        hand_size = int(flat_obs[i])
        draft_hand_size = int(flat_obs[i+1])
        i += 2

        # Played cards
        played_cards = [i for i, v in enumerate(flat_obs[i:i+200]) if v == 1]
        i += 200

        # Milestones and Awards
        milestones = [i for i, v in enumerate(flat_obs[i:i+len(MILESTONES)]) if v == 1]
        i += len(MILESTONES)
        funded_awards = [i for i, v in enumerate(flat_obs[i:i+len(AWARDS)]) if v == 1]
        i += len(AWARDS)

        # Corporation
        corporation = int(flat_obs[i])
        i += 1

        # Tiles (flattened 5x9)
        tiles = np.array(flat_obs[i:i+45]).reshape((5, 9)).tolist()
        i += 45

        # Global milestone/award flags
        claimed_milestones = [i for i, v in enumerate(flat_obs[i:i+len(MILESTONES)]) if v == 1]
        i += len(MILESTONES)
        global_awards = [i for i, v in enumerate(flat_obs[i:i+len(AWARDS)]) if v == 1]
        i += len(AWARDS)

        # Phase flags
        phase_flags = flat_obs[i:i+3].astype(int)
        i += 3

        # Meta
        generation = int(flat_obs[i])
        current_player = int(flat_obs[i+1])

        # Build nested dict
        obs['player'] = {
            'terraform_rating': tr,
            'resources': resources,
            'production': production,
            'hand_size': hand_size,
            'draft_hand_size': draft_hand_size,
            'played_card_indices': played_cards,
            'milestones': milestones,
            'funded_awards': funded_awards,
            'corporation_index': corporation
        }
        obs['tiles'] = tiles
        obs['claimed_milestones'] = claimed_milestones
        obs['funded_awards'] = global_awards
        obs['phase'] = {
            'draft': bool(phase_flags[0]),
            'action': bool(phase_flags[1]),
            'place_entity': bool(phase_flags[2])
        }
        obs['generation'] = generation
        obs['current_player'] = current_player

        return obs

    def get_vp_from_cards(self,cards):
        return sum(c.get('effects', [{}])[0].get('amount', 0) for c in player['played_cards'] if 'vp' in str(c))

    def compute_final_scores(self):
        scores = []
        for i, player in enumerate(self.players):
            tr = player['terraform_rating']
            vp_cards = self.get_vp_from_cards(player['played_cards'])
            milestone_vp = len(player.get('milestones', [])) * 5
            award_vp = 0

            for award in self.funded_awards:
                # Simplified scoring — award 5 VP to most qualifying player
                if award == "Landlord":
                    tiles = sum([1 for tile in self.map if tile['type']=='city' and tile['owner']==i])
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

    def get_card_from_deck(self):
        if len(self.deck)>0:
            return self.deck.pop()
        else:
            if len(self.discard_pile)>0:
                self.deck=self.discard_pile.copy()
                self.discard_pile=[]
                random.Random(self.seed).shuffle(self.deck)
                return self.deck.pop()
            else:
                raise Exception("Deck is empty")

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
                if e['type']=="production" and e['amount']<0:
                    if e['resource']=='mc' and player['production'][e['resource']]+e['amount']<-5:
                        return False
                    elif e['resource']!='mc' and player['production'][e['resource']]+e['amount']<0:
                        return False
        if card['name'] in player['played_active_cards_in_round'].keys():
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

    def add_global_parameter(self,player,parameter_name):
        if parameter_name=='temperature':
            if self.global_parameters[parameter_name]+2<=8:
                player['terraform_rating']+=1
                self.global_parameters[parameter_name]+=2
        elif parameter_name=='oxygen':
            if self.global_parameters[parameter_name]+2<=14:
                player['terraform_rating']+=1
                self.global_parameters[parameter_name]+=1
        
        
    def apply_card_effect(self, player, card):
        for effect in card.get("effects", []):
            etype = effect.get("type")
            if etype == "global":
                target = effect.get("target")
                amount = effect.get("amount", 0)
                for i in range(amount):
                    self.add_global_parameter(player,target)

            elif etype == "tr":
                player['terraform_rating'] += effect.get("amount", 0)

            elif etype == "production":
                resource = effect.get("resource")
                amount = effect.get("amount", 0)
                if resource in player['production']:
                    player['production'][resource] += amount

            elif etype == "resource":
                resource = effect.get("resource")
                amount = effect.get("amount", 0)
                if resource in player:
                    player[resource] += amount
            elif etype == "draw" and self.deck:
                for _ in range(effect.get("amount", 1)):
                    if self.deck:
                        player['hand'].append(self.get_card_from_deck())

            elif etype == "place_tile":
                # Mark intent — actual tile placement must be handled elsewhere in UI
                tile_type = effect.get("tile")
                self.deffered_player_actions.append({"type":"place_tile","tile":tile_type,"id":str(uuid.uuid4())})
                print(f"[Effect] Would place tile: {tile_type} (UI required)")

            elif etype == "remove_plants":
                player['plants'] = max(0, player['plants'] - effect.get("amount", 0))

            elif etype == "discount":
                print(f"[Effect] Player receives discount on {effect.get('scope')} by {effect.get('amount')}")
                # Apply this to a session variable or future cost calculations
                

    def next_turn(self):
        print("Next turn started")
        self.current_player = (self.current_player + 1) % self.num_players
        if self.current_player==self.first_player and self.draft_phase:
            self.phase=GamePhase.main_action
            for p in self.players:
                for c in p['draft_hand']:
                    self.discard_pile.append(c)
                p['draft_hand']=[]
        
        if self.current_player==self.first_player and self.choose_corporation_phase:
            self.current_player_actions_left=1
            self.phase=GamePhase.start_draft
        else:
            self.current_player_actions_left=2
        
                           
        
        if len(self.passed_players) == self.num_players:
            self.end_generation()
        print(f"End turn. Current player: {self.current_player}")

    def player_pass(self):
        self.passed_players.add(self.current_player)
        self.next_turn()

    def end_generation(self):
        self.generation += 1
        self.passed_players = set()
        self.first_player=(self.first_player + 1) % self.num_players
        for player in self.players:
            player['heat'] += player['energy']
            player['energy'] = 0
            for res in ['mc', 'steel', 'titanium', 'plants', 'energy', 'heat']:
                player[res] += player['production'].get(res, 0)
            player['mc'] += player['terraform_rating']
            if self.deck:
                for i in range(4):
                    player['draft_hand'].append(self.get_card_from_deck())
            player['available_draft_cards']=100
            player['available_free_draft_cards']=0
            player['played_active_cards_in_round']={}
        if self.global_parameters['oceans']==9 and self.global_parameters['temperature']==8 and self.global_parameters['oxygen']==14:
            self.phase=GamePhase.last_actions
        else:
            self.phase=GamePhase.new_round_draft
        
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
        tile = self.get_map_tile_by_coord(x,y)
        if not tile:
            return -1
        if tile_type == 'city' and tile['type'] == 'empty':
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    test_tile=self.get_map_tile_by_coord(x+dx,y+dy)
                    if test_tile and test_tile['type']=='city':
                        return -1
        if tile_type == 'ocean' and tile['type']!='ocean_area':
            return -1
        tile['type']=tile_type
        tile['owner']=self.current_player
        if tile_type=='greenery':
            self.add_global_parameter(player,'oxygen')
        if tile_type=='ocean':
            player['terraform_rating'] += 1
        for r in tile.get('resources',[]):
            if r['resource'] in player:
                player[r['resource']]+=r['amount']
            elif r['resource']=='draw':
                for _ in r['amount']:
                    player['hand'].append(self.get_card_from_deck())
        if action['id']:
            action_id=action['id']
            self.deffered_player_actions=list( filter(lambda x: x['id'] != action_id, self.deffered_player_actions) )
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
                            player['hand'].append(self.get_card_from_deck())
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
        player['funded_awards'].append(name)
        return 1

    def handle_action_sell_card(self, player, action):
        card_name = action.get('card_name')
        for card in player['hand']:
            if card['name'] == card_name:
                if card in player['hand']:
                    player['hand'].remove(card)
                    player['mc']+=1
                    self.discard_pile.append(card)
                    return 1
                else:
                    return -1
        return -1
    
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
        elif action_type == 'sell_card':
            reward=self.handle_action_sell_card(player,action)
        elif action_type == 'end_turn':
            pass
        else:
            reward = -1
        if reward<0:
            print(f"Invalid action: {action} by Player {self.current_player + 1}")
        print(f"Handle action done, type={action_type}, reward={reward}")
        return reward

    def decode_action(self,flat_action):
        a = flat_action.astype(int)
        action_type = ACTION_TYPES.get(a[0])

        if action_type == 'play_card':
            return {'type': 'play_card', 'card_index': a[1]}
        elif action_type == 'place_tile':
            tile_map = {0: 'greenery', 1: 'city', 2: 'ocean'}
            tile_type = tile_map.get(a[1] % 3)
            position = (a[2] % 9, a[1] % 5)
            return {'type': 'place_tile', 'tile_type': tile_type, 'position': position}
        elif action_type == 'buy_card':
            return {'type': 'buy_card', 'card_index': a[1]}
        elif action_type == 'active_card_action':
            return {'type': 'active_card_action', 'card_name_index': a[1]}
        elif action_type == 'claim_milestone':
            milestone_list = list(MILESTONES)
            return {'type': 'claim_milestone', 'name': milestone_list[a[1] % len(milestone_list)]}
        elif action_type == 'fund_award':
            return {'type': 'fund_award', 'name': AWARDS[a[1] % len(AWARDS)]}
        elif action_type == 'end_turn':
            return {'type': 'end_turn'}

        return {'type': 'noop'}
    
    def update_front(self):
        if self.render_callback:
            self.render_callback()

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return np.zeros((500, 500, 3), dtype=np.uint8)

    def close(self):
        pass



class TerraformingMarsPettingZooWrapper(AECEnv):
    metadata = {'render_modes': ['human'], 'name': 'terraforming_mars_aec'}

    def __init__(self):
        super().__init__()
        self.env = TerraformingMarsEnv(num_players=2)
        self.num_agents = self.env.num_players
        self.agents = [f"player_{i}" for i in range(self.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        self.observation_spaces = {agent: self.env.get_observation_space() for agent in self.agents}
        self.action_spaces = {agent: self.env.get_action_space() for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.agent_selection = self.agents[self.env.current_player]
        return self.observe(self.agent_selection)

    def observe(self, agent):
        if agent not in self.agents:
            return np.zeros(self.observation_spaces[agent].shape, dtype=np.float32)
        return self.env.observe()

    def step(self, action):
        agent = self.agent_selection
        action_dict = self.env.decode_action(action)
        _, reward, done, _ = self.env.step(action_dict)
        self._cumulative_rewards[agent] += reward
        self.agent_selection = self.agents[self.env.current_player]
        self._was_done_step = done

        if done:
            self.agents = []

    def render(self):
        pass  # UI can be routed through streamlit or just print()

    def close(self):
        self.env.close()