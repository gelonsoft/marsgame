import json
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from core.components import Deck, GridBoard
from games.terraformingmars.actions import TMAction
from games.terraformingmars.components import Award, Milestone, TMCard, TMMapTile
from games.terraformingmars.rules.effects import Bonus
from utilities import Utils
from utilities import Vector2D

class ActionType(Enum):
    PlayCard = "PlayCard"
    StandardProject = "StandardProject"
    ClaimMilestone = "ClaimMilestone"
    FundAward = "FundAward"
    ActiveAction = "ActiveAction"
    BasicResourceAction = "BasicResourceAction"
    BuyProject = "BuyProject"  # TODO ignore in GUI

class BasicResourceAction(Enum):
    HeatToTemp = "HeatToTemp"
    PlantToGreenery = "PlantToGreenery"

class StandardProject(Enum):
    SellPatents = "SellPatents"
    PowerPlant = "PowerPlant"
    Asteroid = "Asteroid"
    Aquifer = "Aquifer"
    Greenery = "Greenery"
    City = "City"
    AirScraping = "AirScraping"

class MapTileType(Enum):
    Ground = "Ground"
    Ocean = "Ocean"
    City = "City"
    Volcanic = "Volcanic"

    def __init__(self, outline):
        self.outline = outline

    def get_outline_color(self):
        return self.outline
        
class TMTypes:
    # Odd r: (odd rows offset to the right)
    neighbor_directions = [
        [Vector2D(1, 0), Vector2D(0, -1), Vector2D(-1, -1), 
        Vector2D(-1, 0), Vector2D(-1, 1), Vector2D(0, 1)],
        [Vector2D(1, 0), Vector2D(1, -1), Vector2D(0, -1), 
         Vector2D(-1, 0), Vector2D(0, 1), Vector2D(1, 1)]
    ]



MapTileType.Ground.outline = Color.lightGray
MapTileType.Ocean.outline = Color.blue
MapTileType.City.outline = Utils.string_to_color("purple")
MapTileType.Volcanic.outline = Color.red

class Tile(Enum):
    Ocean = "data/terraformingmars/images/tiles/ocean.png"
    Greenery = "data/terraformingmars/images/tiles/greenery_no_O2.png"
    City = "data/terraformingmars/images/tiles/city.png"
    CommercialBuilding = "data/terraformingmars/images/tiles/CommercialBuilding.png"
    NuclearExplosion = "data/terraformingmars/images/tiles/NuclearExplosion.png"
    IndustrialBuilding = "data/terraformingmars/images/tiles/IndustrialBuilding.png"
    Mine = "data/terraformingmars/images/tiles/mine.png"
    Moonhole = "data/terraformingmars/images/tiles/Moonhole.png"
    Nature = "data/terraformingmars/images/tiles/Nature.png"
    Park = "data/terraformingmars/images/tiles/park.png"
    Restricted = "data/terraformingmars/images/tiles/Restricted.png"
    Volcano = "data/terraformingmars/images/tiles/Volcano.png"

    def __init__(self, image_path):
        self.image_path = image_path

    def get_image_path(self):
        return self.image_path

    def get_regular_legal_tile_type(self):
        if self == Tile.Ocean:
            return TMTypes.MapTileType.Ocean
        return TMTypes.MapTileType.Ground

    def can_be_owned(self):
        return self != Tile.Ocean

    def get_global_parameter_to_increase(self):
        if self == Tile.Ocean:
            return TMTypes.GlobalParameter.OceanTiles
        if self == Tile.Greenery:
            return TMTypes.GlobalParameter.Oxygen
        return None

class Resource(Enum):
    MegaCredit = ("data/terraformingmars/images/megacredits/megacredit.png", True, False)
    Steel = ("data/terraformingmars/images/resources/steel.png", True, False)
    Titanium = ("data/terraformingmars/images/resources/titanium.png", True, False)
    Plant = ("data/terraformingmars/images/resources/plant.png", True, False)
    Energy = ("data/terraformingmars/images/resources/power.png", True, False)
    Heat = ("data/terraformingmars/images/resources/heat.png", True, False)
    Card = ("data/terraformingmars/images/resources/card.png", False, False)
    TR = ("data/terraformingmars/images/resources/TR.png", False, False)
    Microbe = ("data/terraformingmars/images/resources/microbe.png", False, True)
    Animal = ("data/terraformingmars/images/resources/animal.png", False, True)
    Science = ("data/terraformingmars/images/resources/science.png", False, True)
    Fighter = ("data/terraformingmars/images/resources/fighter.png", False, True)
    Floater = ("data/terraformingmars/images/resources/floater.png", False, True)

    _n_player_board_res = -1

    def __init__(self, image_path, player_board_res, can_go_on_card):
        self.image_path = image_path
        self.player_board_res = player_board_res
        self.can_go_on_card = can_go_on_card

    def get_image_path(self):
        return self.image_path

    def is_player_board_res(self):
        return self.player_board_res

    @classmethod
    def n_player_board_res(cls):
        if cls._n_player_board_res == -1:
            cls._n_player_board_res = sum(1 for res in cls if res.is_player_board_res())
        return cls._n_player_board_res

    def can_go_on_card(self):
        return self.can_go_on_card

    @classmethod
    def get_player_board_resources(cls):
        return [res for res in cls if res.is_player_board_res()]

class Tag(Enum):
    Plant = "data/terraformingmars/images/tags/plant.png"
    Microbe = "data/terraformingmars/images/tags/microbe.png"
    Animal = "data/terraformingmars/images/tags/animal.png"
    Science = "data/terraformingmars/images/tags/science.png"
    Earth = "data/terraformingmars/images/tags/earth.png"
    Space = "data/terraformingmars/images/tags/space.png"
    Event = "data/terraformingmars/images/tags/event.png"
    Building = "data/terraformingmars/images/tags/building.png"
    Power = "data/terraformingmars/images/tags/power.png"
    Jovian = "data/terraformingmars/images/tags/jovian.png"
    City = "data/terraformingmars/images/tags/city.png"
    Venus = "data/terraformingmars/images/tags/venus.png"
    Wild = "data/terraformingmars/images/tags/wild.png"

    def __init__(self, image_path):
        self.image_path = image_path

    def get_image_path(self):
        return self.image_path

class CardType(Enum):
    Automated = ("data/terraformingmars/images/cards/card-automated.png", True, Color.green)
    Active = ("data/terraformingmars/images/cards/card-active.png", True, Color.cyan)
    Event = ("data/terraformingmars/images/cards/card-event.png", True, Color.orange)
    Corporation = ("data/terraformingmars/images/cards/corp-card-bg.png", False, Color.gray)
    Prelude = ("data/terraformingmars/images/cards/proj-card-bg.png", False, Color.pink)
    Colony = ("data/terraformingmars/images/cards/proj-card-bg.png", False, Color.lightGray)
    GlobalEvent = ("data/terraformingmars/images/cards/proj-card-bg.png", False, Color.blue)

    def __init__(self, image_path, is_playable_standard, color):
        self.image_path = image_path
        self.is_playable_standard = is_playable_standard
        self.color = color

    def get_image_path(self):
        return self.image_path

    def is_playable_standard(self):
        return self.is_playable_standard

    def get_color(self):
        return self.color

class GlobalParameter(Enum):
    Oxygen = ("data/terraformingmars/images/global-parameters/oxygen.png", Color.lightGray, True, "O2")
    Temperature = ("data/terraformingmars/images/global-parameters/temperature.png", Color.white, True, "°C")
    OceanTiles = ("data/terraformingmars/images/tiles/ocean.png", Color.yellow, True, "Ocean")
    Venus = ("data/terraformingmars/images/global-parameters/venus.png", Color.blue, False, "Venus")

    def __init__(self, image_path, color, counts_for_end_game, short_string):
        self.image_path = image_path
        self.color = color
        self.counts_for_end_game = counts_for_end_game
        self.short_string = short_string

    def get_image_path(self):
        return self.image_path

    def get_color(self):
        return self.color

    def counts_for_end_game(self):
        return self.counts_for_end_game

    def get_short_string(self):
        return self.short_string

    @classmethod
    def get_draw_order(cls, params):
        order = []
        if params.expansions.contains(Expansion.Venus):
            order.append(cls.Venus)
        order.extend([cls.Temperature, cls.Oxygen, cls.OceanTiles])
        return order

class Expansion(Enum):
    Base = "Base"
    CorporateEra = "CorporateEra"
    Prelude = "Prelude"
    Venus = "Venus"
    Turmoil = "Turmoil"
    Colonies = "Colonies"
    Promo = "Promo"
    Hellas = "Hellas"
    Elysium = "Elysium"

    def get_board_path(self):
        return f"data/terraformingmars/boards/{self.name.lower()}.json"

    def get_corp_cards_path(self):
        return f"data/terraformingmars/corporationCards/{self.name.lower()}.json"

    def get_project_cards_path(self):
        return f"data/terraformingmars/projectCards/{self.name.lower()}.json"

    def get_other_cards_path(self):
        return f"data/terraformingmars/otherCards/{self.name.lower()}.json"

    def load_board(self, board: GridBoard, extra_tiles: Set[TMMapTile], bonuses: Set[Bonus],
                    milestones: Set[Milestone], awards: Set[Award], 
                    global_parameters: Dict['GlobalParameter', 'GlobalParameter']):
        try:
            with open(self.get_board_path(), 'r') as f:
                data = json.load(f)

                # Process main map
                if 'board' in data:
                    for y, row in enumerate(data['board']):
                        for x, tile_str in enumerate(row):
                            board.set_element(x, y, TMMapTile.parse_map_tile(tile_str, x, y))

                # Process extra tiles not on regular board
                if 'extra' in data:
                    for tile_str in data['extra']:
                        extra_tiles.add(TMMapTile.parse_map_tile(tile_str))

                # Process milestones and awards
                if 'milestones' in data:
                    for ms_str in data['milestones']:
                        parts = ms_str.split(':')
                        milestones.add(Milestone(parts[0], int(parts[2]), parts[1]))

                if 'awards' in data:
                    for award_str in data['awards']:
                        parts = award_str.split(':')
                        awards.add(Award(parts[0], parts[1]))

                # Process global parameters
                if 'globalParameters' in data:
                    for gp_data in data['globalParameters']:
                        param = TMTypes.GlobalParameter[gp_data['name']]
                        values = [int(v) for v in gp_data['range']]
                        global_parameters[param] = games.terraformingmars.components.GlobalParameter(values, param.name)

                        # Process bonuses
                        if 'bonus' in gp_data:
                            for bonus_data in gp_data['bonus']:
                                effect_str = bonus_data['effect']
                                threshold = int(bonus_data['threshold'])
                                bonuses.add(Bonus(param, threshold, 
                                                TMAction.parse_action_on_card(effect_str, None, True)))
        except Exception as e:
            print(f"Error loading board: {e}")

    def load_project_cards(self, deck: Deck[TMCard]):
        self._load_cards(deck, self.get_project_cards_path())

    def load_corp_cards(self, deck: Deck[TMCard]):
        self._load_cards(deck, self.get_corp_cards_path())

    def _load_cards(self, deck: Deck[TMCard], path: str):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for card_data in data:
                    if deck.component_name.lower() == "corporations":
                        card = TMCard.load_corporation(card_data)
                    else:
                        card = TMCard.load_card_json(card_data)
                    deck.add(card)
        except Exception as e:
            print(f"Error loading cards: {e}")