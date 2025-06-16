from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass


class TMPhase(Enum):
    CORPORATION_SELECT = "CorporationSelect"
    RESEARCH = "Research"
    ACTIONS = "Actions"
    PRODUCTION = "Production"


class TMGameState:
    def __init__(self, game_parameters, n_players: int):
        # Initialize parent class equivalent
        self.game_parameters = game_parameters
        self.n_players = n_players
        self.turn_order = self._create_turn_order(n_players)
        self.game_status = None
        self.game_phase = None
        self.player_results = [None] * n_players
        self.current_player = 0
        self.rnd = None  # Random generator equivalent
        
        # General state info
        self.generation = 0
        self.board = None  # GridBoard equivalent
        self.extra_tiles = set()  # HashSet<TMMapTile>
        self.global_parameters = {}  # HashMap<GlobalParameter, GlobalParameter>
        self.bonuses = set()  # HashSet<Bonus>
        self.project_cards = None  # Deck<TMCard>
        self.corp_cards = None     # Deck<TMCard>
        self.discard_cards = None  # Deck<TMCard>
        
        # Effects and actions played - arrays indexed by player
        self.player_extra_actions = [set() for _ in range(n_players)]
        self.player_resource_map = [set() for _ in range(n_players)]
        self.player_discount_effects = [{} for _ in range(n_players)]
        self.player_persisting_effects = [set() for _ in range(n_players)]
        
        # Player-specific counters
        self.player_resources = [{} for _ in range(n_players)]
        self.player_resource_increase_gen = [{} for _ in range(n_players)]
        self.player_production = [{} for _ in range(n_players)]
        self.player_cards_played_tags = [{} for _ in range(n_players)]
        self.player_cards_played_types = [{} for _ in range(n_players)]
        self.player_tiles_placed = [{} for _ in range(n_players)]
        self.player_card_points = [0] * n_players  # Counter equivalent
        
        # Player cards
        self.player_hands = [[] for _ in range(n_players)]  # Deck<TMCard> equivalent
        self.player_complicated_point_cards = [[] for _ in range(n_players)]
        self.played_cards = [[] for _ in range(n_players)]
        self.player_card_choice = [[] for _ in range(n_players)]
        self.player_corporations = [None] * n_players
        
        # Milestones and awards
        self.milestones = set()
        self.awards = set()
        self.n_milestones_claimed = 0  # Counter equivalent
        self.n_awards_funded = 0       # Counter equivalent

    def _create_turn_order(self, n_players: int):
        # Equivalent to TMTurnOrder
        return None  # Placeholder

    def _get_game_type(self):
        return "TerraformingMars"

    def _get_all_components(self) -> List:
        components = []
        components.append(self.board)
        components.append(self.project_cards)
        components.append(self.corp_cards)
        components.append(self.discard_cards)
        components.append(self.n_awards_funded)
        components.append(self.n_milestones_claimed)
        components.extend(self.extra_tiles)
        components.extend(self.milestones)
        components.extend(self.awards)
        components.extend(self.global_parameters.values())
        components.extend(self.player_hands)
        components.extend(self.player_card_choice)
        components.extend(self.player_complicated_point_cards)
        components.extend(self.played_cards)
        components.extend(self.player_card_points)
        
        for i in range(self.n_players):
            components.extend(self.player_resources[i].values())
            components.extend(self.player_production[i].values())
            components.extend(self.player_cards_played_tags[i].values())
            components.extend(self.player_tiles_placed[i].values())
            components.extend(self.player_cards_played_types[i].values())
            if self.player_corporations[i] is not None:
                components.append(self.player_corporations[i])
        
        return components

    def copy(self, player_id: int = -1):
        """Deep copy of the game state"""
        copy_state = TMGameState(self.game_parameters.copy(), self.n_players)
        
        # General public info
        copy_state.generation = self.generation
        copy_state.board = self.board.copy() if self.board else None
        copy_state.extra_tiles = {tile.copy() for tile in self.extra_tiles}
        copy_state.global_parameters = {k: v.copy() for k, v in self.global_parameters.items()}
        copy_state.bonuses = {bonus.copy() for bonus in self.bonuses}
        copy_state.milestones = {milestone.copy() for milestone in self.milestones}
        copy_state.awards = {award.copy() for award in self.awards}
        copy_state.n_milestones_claimed = self.n_milestones_claimed
        copy_state.n_awards_funded = self.n_awards_funded
        
        # Face-down decks
        copy_state.project_cards = self.project_cards.copy() if self.project_cards else None
        copy_state.corp_cards = self.corp_cards.copy() if self.corp_cards else None
        copy_state.discard_cards = self.discard_cards.copy() if self.discard_cards else None
        
        # Player-specific info
        for i in range(self.n_players):
            copy_state.player_extra_actions[i] = {action.copy() for action in self.player_extra_actions[i]}
            copy_state.player_resource_map[i] = {rm.copy() for rm in self.player_resource_map[i]}
            copy_state.player_persisting_effects[i] = {effect.copy() for effect in self.player_persisting_effects[i]}
            copy_state.player_discount_effects[i] = {req.copy(): val for req, val in self.player_discount_effects[i].items()}
            
            copy_state.player_resources[i] = {k: v.copy() for k, v in self.player_resources[i].items()}
            copy_state.player_resource_increase_gen[i] = self.player_resource_increase_gen[i].copy()
            copy_state.player_production[i] = {k: v.copy() for k, v in self.player_production[i].items()}
            copy_state.player_cards_played_tags[i] = {k: v.copy() for k, v in self.player_cards_played_tags[i].items()}
            copy_state.player_cards_played_types[i] = {k: v.copy() for k, v in self.player_cards_played_types[i].items()}
            copy_state.player_tiles_placed[i] = {k: v.copy() for k, v in self.player_tiles_placed[i].items()}
            
            copy_state.player_card_points[i] = self.player_card_points[i]
            copy_state.player_complicated_point_cards[i] = self.player_complicated_point_cards[i].copy()
            copy_state.played_cards[i] = self.played_cards[i].copy()
            
            if self.player_corporations[i] is not None:
                copy_state.player_corporations[i] = self.player_corporations[i].copy()
        
        # Handle hidden information based on partial observability
        if player_id != -1 and self.game_parameters.partial_observable:
            for i in range(self.n_players):
                copy_state.player_hands[i] = self.player_hands[i].copy()
                copy_state.player_card_choice[i] = self.player_card_choice[i].copy()
                
                if i != player_id:  # Shuffle opponent's cards
                    copy_state.project_cards.extend(copy_state.player_hands[i])
                    if self.game_phase != TMPhase.CORPORATION_SELECT:
                        copy_state.project_cards.extend(copy_state.player_card_choice[i])
                        copy_state.player_card_choice[i] = []
                    copy_state.player_hands[i] = []
            
            # Reshuffle and deal new cards for opponents
            if copy_state.corp_cards:
                copy_state.corp_cards.shuffle()
            if copy_state.project_cards:
                copy_state.project_cards.shuffle()
                
            for i in range(self.n_players):
                if i != player_id:
                    for _ in range(len(self.player_hands[i])):
                        copy_state.player_hands[i].append(copy_state.draw_card())
                    if self.game_phase != TMPhase.CORPORATION_SELECT:
                        for _ in range(len(self.player_card_choice[i])):
                            copy_state.player_card_choice[i].append(copy_state.draw_card())
        else:
            for i in range(self.n_players):
                copy_state.player_hands[i] = self.player_hands[i].copy()
                copy_state.player_card_choice[i] = self.player_card_choice[i].copy()
        
        return copy_state

    def draw_card(self):
        """Draw a card from project deck, reshuffling discard if needed"""
        if len(self.project_cards) == 0:
            self.project_cards.extend(self.discard_cards)
            self.discard_cards.clear()
            self.project_cards.shuffle()
        return self.project_cards.pop()

    def _get_heuristic_score(self, player_id: int) -> float:
        return self.count_points(player_id)

    def get_game_score(self, player_id: int) -> float:
        return self.player_resources[player_id].get("TR", 0)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TMGameState):
            return False
        
        return (self.generation == other.generation and
                self.board == other.board and
                self.extra_tiles == other.extra_tiles and
                self.global_parameters == other.global_parameters and
                self.bonuses == other.bonuses and
                self.project_cards == other.project_cards and
                self.corp_cards == other.corp_cards and
                self.discard_cards == other.discard_cards and
                self.player_extra_actions == other.player_extra_actions and
                self.player_resource_map == other.player_resource_map and
                self.player_discount_effects == other.player_discount_effects and
                self.player_persisting_effects == other.player_persisting_effects and
                self.player_resources == other.player_resources and
                self.player_resource_increase_gen == other.player_resource_increase_gen and
                self.player_production == other.player_production and
                self.player_cards_played_tags == other.player_cards_played_tags and
                self.player_cards_played_types == other.player_cards_played_types and
                self.player_tiles_placed == other.player_tiles_placed and
                self.player_card_points == other.player_card_points and
                self.player_hands == other.player_hands and
                self.player_complicated_point_cards == other.player_complicated_point_cards and
                self.player_card_choice == other.player_card_choice and
                self.player_corporations == other.player_corporations and
                self.milestones == other.milestones and
                self.awards == other.awards and
                self.n_milestones_claimed == other.n_milestones_claimed and
                self.n_awards_funded == other.n_awards_funded)

    def __hash__(self) -> int:
        # Simplified hash implementation
        return hash((self.generation, str(self.board), len(self.extra_tiles),
                    len(self.global_parameters), len(self.bonuses)))

    def __str__(self) -> str:
        components = [
            str(hash(self.game_parameters)),
            str(hash(self.turn_order)),
            str(hash(tuple(self._get_all_components()))),
            str(hash(self.game_status)),
            str(hash(self.game_phase)),
            str(hash(tuple(self.player_results))),
            str(hash(self.generation)),
            str(hash(self.board)),
            str(hash(tuple(self.extra_tiles))),
            str(hash(tuple(self.global_parameters.items()))),
            str(hash(tuple(self.bonuses))),
            str(hash(self.project_cards)),
            str(hash(self.corp_cards)),
            str(hash(self.discard_cards)),
            str(hash(tuple(self.milestones))),
            str(hash(tuple(self.awards))),
            str(hash((self.n_milestones_claimed, self.n_awards_funded))),
        ]
        return "|".join(components)

    # Public API methods
    def get_player_production(self):
        return self.player_production

    def get_player_resources(self):
        return self.player_resources

    def get_board(self):
        return self.board

    def get_bonuses(self):
        return self.bonuses

    def get_global_parameters(self):
        return self.global_parameters

    def get_extra_tiles(self):
        return self.extra_tiles

    def get_player_hands(self):
        return self.player_hands

    def get_player_cards_played_tags(self):
        return self.player_cards_played_tags

    def get_player_cards_played_types(self):
        return self.player_cards_played_types

    def get_player_extra_actions(self):
        return self.player_extra_actions

    def get_player_tiles_placed(self):
        return self.player_tiles_placed

    def get_milestones(self):
        return self.milestones

    def get_awards(self):
        return self.awards

    def get_player_corporations(self):
        return self.player_corporations

    def all_corp_chosen(self) -> bool:
        return all(corp is not None for corp in self.player_corporations)

    def get_player_card_choice(self):
        return self.player_card_choice

    def get_discard_cards(self):
        return self.discard_cards

    def get_corp_cards(self):
        return self.corp_cards

    def get_project_cards(self):
        return self.project_cards

    def get_player_resource_map(self):
        return self.player_resource_map

    def get_n_awards_funded(self):
        return self.n_awards_funded

    def get_n_milestones_claimed(self):
        return self.n_milestones_claimed

    def get_generation(self):
        return self.generation

    def get_player_resource_increase_gen(self):
        return self.player_resource_increase_gen

    def get_player_persisting_effects(self):
        return self.player_persisting_effects

    def get_player_discount_effects(self):
        return self.player_discount_effects

    def get_player_complicated_point_cards(self):
        return self.player_complicated_point_cards

    def get_player_card_points(self):
        return self.player_card_points

    def get_played_cards(self):
        return self.played_cards

    def discount_action_type_cost(self, action, player: int = -1) -> int:
        """Apply tag discount effects for actions"""
        discount = 0
        if player == -1:
            player = self.current_player
            
        for requirement, amount in self.player_discount_effects[player].items():
            if hasattr(requirement, '__class__') and 'ActionTypeRequirement' in str(requirement.__class__):
                if requirement.test_condition(action):
                    discount += amount
        return discount

    def discount_card_cost(self, card, player: int = -1) -> int:
        """Apply tag discount effects for cards"""
        discount = 0
        if player == -1:
            player = self.current_player
            
        for tag in card.tags:
            for requirement, amount in self.player_discount_effects[player].items():
                if hasattr(requirement, '__class__') and 'TagsPlayedRequirement' in str(requirement.__class__):
                    if tag in requirement.tags:
                        discount += amount
        return discount

    def is_card_free(self, card, amount_paid: int = 0, player: int = -1) -> bool:
        if player == -1:
            player = self.current_player
        return card.cost - self.discount_card_cost(card, player) - amount_paid <= 0

    def string_to_gp_counter(self, s: str):
        """Convert string to global parameter counter"""
        # Implementation would depend on TMTypes.GlobalParameter enum
        return self.global_parameters.get(s)

    def string_to_gp_or_player_res_counter(self, s: str, player: int = -1):
        """Convert string to global parameter or player resource counter"""
        if player == -1:
            player = self.current_player
            
        counter = self.string_to_gp_counter(s)
        if counter is None:
            # A resource or production instead
            resource = s.split("prod")[0]
            if "prod" in s:
                counter = self.player_production[player].get(resource)
            else:
                counter = self.player_resources[player].get(resource)
        return counter

    def can_player_pay(self, player: int, resource_or_card=None, from_resources=None, 
                      to_resource=None, amount: int = 0, production: bool = False) -> bool:
        """Check if player can pay for something"""
        if player == -3:  # Neutral player in solo play
            return True
            
        if production:
            counter = self.player_production[player].get(to_resource, 0)
            # Handle minimum values for production
            return counter >= amount
            
        if isinstance(resource_or_card, str):  # Simple resource payment
            return self.player_resources[player].get(resource_or_card, 0) >= amount
        else:  # Card payment with resource transformation
            total = self.player_resource_sum(player, resource_or_card, from_resources, to_resource, True)
            return (self.is_card_free(resource_or_card, total, player) if resource_or_card 
                   else total >= amount)

    def player_resource_sum(self, player: int, card, from_resources: Set, 
                           to_resource, include_itself: bool) -> int:
        """Calculate total resources available for payment"""
        if from_resources is None or len(from_resources) > 0:
            total = 0
            if include_itself or (from_resources and to_resource in from_resources):
                total = self.player_resources[player].get(to_resource, 0)
                
            # Add resources that can be transformed
            for res_map in self.player_resource_map[player]:
                if ((from_resources is None or res_map.from_resource in from_resources) and
                    res_map.to_resource == to_resource and
                    (res_map.requirement is None or res_map.requirement.test_condition(card))):
                    n = self.player_resources[player].get(res_map.from_resource, 0)
                    total += int(n * res_map.rate)
            return total
        return 0

    def can_player_transform(self, player: int, card, from_resource=None, to_resource=None) -> Set:
        """Check which resources can be transformed"""
        resources = set()
        for res_map in self.player_resource_map[player]:
            if ((from_resource is None or res_map.from_resource == from_resource) and
                res_map.to_resource == to_resource and
                (res_map.requirement is None or res_map.requirement.test_condition(card))):
                if self.player_resources[player].get(res_map.from_resource, 0) > 0:
                    resources.add(res_map.from_resource)
        return resources

    def player_pay(self, player: int, resource: str, amount: int):
        """Make player pay resources"""
        current = self.player_resources[player].get(resource, 0)
        self.player_resources[player][resource] = current - abs(amount)

    def get_resource_map_rate(self, from_resource: str, to_resource: str) -> float:
        """Get resource transformation rate"""
        rate = 1.0
        for rm in self.player_resource_map[self.current_player]:
            if rm.from_resource == from_resource and rm.to_resource == to_resource:
                rate = rm.rate
                break
        return rate

    def add_discount_effects(self, discounts: List):
        """Add discount effects to current player"""
        player = self.current_player
        for discount in discounts:
            requirement = discount.requirement
            amount = discount.amount
            if requirement in self.player_discount_effects[player]:
                self.player_discount_effects[player][requirement] += amount
            else:
                self.player_discount_effects[player][requirement] = amount

    def add_persisting_effects(self, effects: List):
        """Add persisting effects to current player"""
        player = self.current_player
        self.player_persisting_effects[player].update(effects)

    def add_resource_mappings(self, mappings: Set, add: bool = True):
        """Add or replace resource mappings"""
        player = self.current_player
        to_remove = set()
        to_add = set()
        
        for new_mapping in mappings:
            added = False
            for existing_mapping in self.player_resource_map[player]:
                if (existing_mapping.from_resource == new_mapping.from_resource and
                    existing_mapping.to_resource == new_mapping.to_resource):
                    if (new_mapping.requirement is None or 
                        new_mapping.requirement == existing_mapping.requirement):
                        if add:
                            existing_mapping.rate += new_mapping.rate
                        else:
                            to_remove.add(existing_mapping)
                            to_add.add(new_mapping)
                        added = True
            if not added:
                to_add.add(new_mapping)
                
        self.player_resource_map[player] -= to_remove
        self.player_resource_map[player] |= to_add

    def has_placed_tile(self, player: int) -> bool:
        """Check if player has placed any ownable tiles"""
        for tile_type, count in self.player_tiles_placed[player].items():
            # Assuming tile_type has can_be_owned() method
            if hasattr(tile_type, 'can_be_owned') and tile_type.can_be_owned() and count > 0:
                return True
        return False

    def any_tiles_placed(self, tile_type=None) -> bool:
        """Check if any tiles have been placed"""
        if tile_type is None:
            for i in range(self.n_players):
                for count in self.player_tiles_placed[i].values():
                    if count > 0:
                        return True
            return self.n_players == 1
        else:
            for i in range(self.n_players):
                if self.player_tiles_placed[i].get(tile_type, 0) > 0:
                    return True
            return (self.n_players == 1 and 
                   tile_type in ["City", "Greenery"])

    def count_points(self, player: int) -> int:
        """Count total points for a player"""
        points = 0
        # Add TR (Terraform Rating)
        points += self.player_resources[player].get("TR", 0)
        # Add milestone points
        points += self.count_points_milestones(player)
        # Add award points
        points += self.count_points_awards(player)
        # Add board points
        points += self.count_points_board(player)
        # Add card points
        points += self.count_points_cards(player)
        return points

    def count_points_milestones(self, player: int) -> int:
        """Count points from milestones"""
        points = 0
        milestone_points = getattr(self.game_parameters, 'n_points_milestone', 5)
        for milestone in self.milestones:
            if milestone.is_claimed() and milestone.claimed == player:
                points += milestone_points
        return points

    def count_points_awards(self, player: int) -> int:
        """Count points from awards"""
        points = 0
        first_place_points = getattr(self.game_parameters, 'n_points_award_first', 5)
        second_place_points = getattr(self.game_parameters, 'n_points_award_second', 2)
        
        for award in self.awards:
            winners = self.award_winner(award)
            if winners:
                first_place, second_place = winners
                if player in first_place:
                    points += first_place_points
                if player in second_place and len(first_place) == 1:
                    points += second_place_points
        return points

    def award_winner(self, award) -> Optional[Tuple[Set[int], Set[int]]]:
        """Determine award winners (first and second place)"""
        if not award.is_claimed():
            return None
            
        best = -1
        second_best = -1
        best_players = set()
        second_best_players = set()
        
        # Find best and second best scores
        for i in range(self.n_players):
            player_points = award.check_progress(self, i)
            if player_points > best:
                second_best_players = best_players.copy()
                second_best = best
                best_players = {i}
                best = player_points
            elif player_points == best:
                best_players.add(i)
            elif player_points > second_best:
                second_best_players = {i}
                second_best = player_points
            elif player_points == second_best:
                second_best_players.add(i)
        
        # No second place if 2 or fewer players, or if multiple got first
        if self.n_players <= 2 or len(best_players) > 1:
            second_best_players = set()
            
        return (best_players, second_best_players)

    def count_points_board(self, player: int) -> int:
        """Count points from board placement"""
        points = 0
        # Points from greeneries
        points += self.player_tiles_placed[player].get("Greenery", 0)
        
        # Points from cities (adjacent greeneries)
        if self.board:
            for i in range(self.board.height):
                for j in range(self.board.width):
                    tile = self.board.get_element(j, i)
                    if tile and tile.get_tile_placed() == "City" and tile.owner == player:
                        # Count adjacent greeneries - would need PlaceTile.nAdjacentTiles equivalent
                        adjacent_greeneries = 0  # Placeholder
                        points += adjacent_greeneries
        return points

    def count_points_cards(self, player: int) -> int:
        """Count points from cards"""
        points = 0
        
        # Normal card points
        points += self.player_card_points[player]
        
        # Complicated point cards
        for card in self.player_complicated_point_cards[player]:
            if card is None:
                continue
                
            if hasattr(card, 'points_threshold') and card.points_threshold is not None:
                if (hasattr(card, 'points_resource') and card.points_resource is not None and
                    card.n_resources_on_card >= card.points_threshold):
                    points += card.n_points
            else:
                if hasattr(card, 'points_resource') and card.points_resource is not None:
                    points += card.n_points * card.n_resources_on_card
                elif hasattr(card, 'points_tag') and card.points_tag is not None:
                    tag_count = self.player_cards_played_tags[player].get(card.points_tag, 0)
                    points += card.n_points * tag_count
                elif hasattr(card, 'points_tile') and card.points_tile is not None:
                    if (hasattr(card, 'points_tile_adjacent') and card.points_tile_adjacent and
                        hasattr(card, 'map_tile_id_tile_placed') and card.map_tile_id_tile_placed >= 0):
                        # Adjacent tile points - would need board neighbor logic
                        pass  # Placeholder for adjacent tile counting
                    else:
                        tile_count = self.player_tiles_placed[player].get(card.points_tile, 0)
                        points += card.n_points * tile_count
                elif hasattr(card, 'component_name') and card.component_name.lower() == "capital":
                    # Special case for Capital card - adjacent oceans
                    pass  # Placeholder for ocean adjacency counting
                    
        return points


@dataclass
class ResourceMapping:
    """Resource transformation mapping"""
    from_resource: str
    to_resource: str
    rate: float
    requirement: Optional[Any] = None
    
    def copy(self):
        return ResourceMapping(
            self.from_resource,
            self.to_resource, 
            self.rate,
            self.requirement.copy() if self.requirement else None
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ResourceMapping):
            return False
        return (self.from_resource == other.from_resource and
                self.to_resource == other.to_resource and
                self.requirement == other.requirement)
    
    def __hash__(self) -> int:
        return hash((self.from_resource, self.to_resource, str(self.requirement)))