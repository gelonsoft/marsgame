import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

class TerraformingMarsEnv(AECEnv):
    def __init__(self, num_players=2, render_callback=None):
        super().__init__()
        self.num_players = num_players
        self.players = [self.create_player() for _ in range(num_players)]
        self.current_player = 0
        self.passed_players = set()
        self.generation = 1  # ✅ Add this line
        self.tiles=[]
        self.ocean_positions={}
        self.render_callback=render_callback
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
            'vp': 0,
            'terraform_rating': 20
        }

    def reset(self):
        self.generation = 1
        self.current_player = 0
        self.passed_players = set()
        self.map_width = 9
        self.map_height = 5
        self.tiles = [
            [{'type': 'empty', 'owner': None} for _ in range(self.map_width)]
            for _ in range(self.map_height)
        ]

        # Define preset ocean zones (row, col)
        self.ocean_positions = {(1, 1), (1, 7), (2, 3), (2, 5), (3, 1), (3, 7)}

        self.players = [self.create_player() for _ in range(self.num_players)]
        self.global_parameters = {'temperature': -30, 'oxygen': 0, 'oceans': 0}
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
        player = self.players[self.current_player]

        if action == "pass":
            self.player_pass()
            return self.observe(), 0, False, {}

        # Handle normal action (e.g. play_card, standard_project...)
        reward = self.handle_action(action)

        self.next_turn()
        self.update_front()
        return self.observe(), reward, False, {}

    def next_turn(self):
        self.current_player = (self.current_player + 1) % self.num_players
        # If all players have passed → end generation
        if len(self.passed_players) == self.num_players:
            self.end_generation()
        
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            # Generate and return an RGB image representing the current game state
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            # Add your rendering logic here
            return img
    def update_front(self):
        if self.render_callback:
            self.render_callback()

        
    def player_pass(self):
        self.passed_players.add(self.current_player)
        self.next_turn()
    
    def end_generation(self):
        self.generation += 1
        self.passed_players = set()

        # Convert energy to heat, then produce resources
        for player in self.players:
            player['heat'] += player['energy']
            player['energy'] = 0

            for resource in ['mc', 'steel', 'titanium', 'plants', 'energy', 'heat']:
                prod = player['production'].get(resource, 0)
                player[resource] += prod

            player['mc'] += player['terraform_rating']  # +TR as income

        self.current_player = 0

    
    def close(self):
        pass

    def handle_action(self, action):
        player = self.players[self.current_player]
        reward = 0

        action_type = action.get('type')
        print(f"Handle action type={action_type}")
        if action_type == 'standard_project':
            name = action['name']
            if name == 'asteroid' and player['mc'] >= 14:
                player['mc'] -= 14
                self.global_parameters['temperature'] += 2
                player['terraform_rating'] += 1
                reward = 1

            elif name == 'power_plant' and player['mc'] >= 11:
                player['mc'] -= 11
                player['production']['energy'] += 1
                reward = 1

            elif name == 'aquifer' and player['mc'] >= 18:
                player['mc'] -= 18
                self.global_parameters['oceans'] += 1
                player['terraform_rating'] += 1
                reward = 1
            else:
                reward=-1
                print("Bad standard project")

        elif action_type == 'play_card':
            name = action['card_name']
            if name == 'Comet' and player['mc'] >= 21:
                player['mc'] -= 21
                self.global_parameters['temperature'] += 2
                player['terraform_rating'] += 1
                player['played_cards'].append({'name': name})
                reward = 2

            elif name == 'Lichen' and player['mc'] >= 7:
                player['mc'] -= 7
                player['production']['plants'] += 1
                player['played_cards'].append({'name': name})
                reward = 1
            else:
                print("Bad card")
                reward = -1  # not enough MC or invalid card name


        elif action_type == 'place_tile':
            tile_type = action['tile_type']
            x, y = action['position']

            # Check bounds
            if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                reward=-1  # invalid
            else:
                tile = self.tiles[y][x]

                if tile['type'] != 'empty':
                    reward= -1  # already occupied
                else:
                    if tile_type == 'city':
                        # Check no adjacent cities
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if (0 <= y+dy < self.map_height) and (0 <= x+dx < self.map_width):
                                    neighbor = self.tiles[y+dy][x+dx]
                                    if neighbor['type'] == 'city':
                                        return -1  # invalid city adjacency

                    if tile_type == 'ocean' and (y, x) not in self.ocean_positions:
                        reward=-1  # invalid ocean placement
                    else:
                        self.tiles[y][x] = {'type': tile_type, 'owner': self.current_player}


                        # TR bonuses
                        if tile_type in ['city', 'greenery', 'ocean']:
                            self.players[self.current_player]['terraform_rating'] += 1

                        reward = 1
        else:
            print(f"Unknown action type {action_type}")
            reward=-1
        print(f"Reward: {reward} player_mc={player['mc']}")

        return reward
