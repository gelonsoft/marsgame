from typing import List
from core.CoreConstants import VisibilityMode
from games.terraformingmars.TMGameState import TMGameState, TMPhase
from games.terraformingmars.TMTypes import (
    ActionType, Resource, StandardProject, Expansion, GlobalParameter,
    Tile, MapTileType, Tag, CardType, BasicResourceAction
)
from core import AbstractGameState
from core import  GameResult
from core import StandardForwardModelWithTurnOrder
from core.components.Counter import Counter
from core.components.Deck import Deck
from core.components.GridBoard import GridBoard
from games.terraformingmars.actions import BuyCard
from games.terraformingmars.actions import ClaimAwardMilestone
from games.terraformingmars.actions import CompoundAction
from games.terraformingmars.actions import DiscardCard
from games.terraformingmars.actions import ModifyGlobalParameter
from games.terraformingmars.actions import ModifyPlayerResource
from games.terraformingmars.actions import PayForAction
from games.terraformingmars.actions import PlaceTile
from games.terraformingmars.actions import PlayCard
from games.terraformingmars.actions import SellProjects
from games.terraformingmars.actions import TMAction
from games.terraformingmars.rules.requirements import TagOnCardRequirement
from utilities import Vector2D


class TMForwardModel(StandardForwardModelWithTurnOrder):
    
    def _setup(self, first_state: AbstractGameState) -> None:
        gs = first_state
        params = first_state.get_game_parameters()
        
        # Initialize player resources and production
        gs.player_resources = [dict() for _ in range(gs.get_n_players())]
        gs.player_production = [dict() for _ in range(gs.get_n_players())]
        gs.player_resource_map = [set() for _ in range(gs.get_n_players())]
        gs.player_discount_effects = [dict() for _ in range(gs.get_n_players())]
        gs.player_resource_increase_gen = [dict() for _ in range(gs.get_n_players())]
        
        for i in range(gs.get_n_players()):
            gs.player_resources[i] = {}
            gs.player_production[i] = {}
            gs.player_resource_increase_gen[i] = {}
            
            for res in Resource:
                starting_res = params.starting_resources.get(res, 0)
                if res == Resource.TR and gs.get_n_players() == 1:
                    starting_res = params.solo_tr
                
                gs.player_resources[i][res] = Counter(
                    starting_res, 0, params.max_points, f"{res}-{i}"
                )
                
                if res in params.starting_production:
                    starting_production = params.starting_production[res]
                    if Expansion.CORPORATE_ERA in params.expansions:
                        starting_production = 0  # No production in corporate era
                    
                    gs.player_production[i][res] = Counter(
                        starting_production,
                        params.minimum_production.get(res, 0),
                        params.max_points,
                        f"{res}-prod-{i}"
                    )
                
                gs.player_resource_increase_gen[i][res] = False
            
            gs.player_resource_map[i] = set()
            # By default, players can exchange steel for X MC and titanium for X MC
            gs.player_resource_map[i].add(
                TMGameState.ResourceMapping(
                    Resource.STEEL, Resource.MEGA_CREDIT,
                    params.n_steel_mc,
                    TagOnCardRequirement([Tag.BUILDING])
                )
            )
            gs.player_resource_map[i].add(
                TMGameState.ResourceMapping(
                    Resource.TITANIUM, Resource.MEGA_CREDIT,
                    params.n_titanium_mc,
                    TagOnCardRequirement([Tag.SPACE])
                )
            )
            
            # Set up player discount maps
            gs.player_discount_effects[i] = {}
        
        # Initialize card decks
        gs.project_cards = Deck("Projects", VisibilityMode.HIDDEN_TO_ALL)
        gs.corp_cards = Deck("Corporations", VisibilityMode.HIDDEN_TO_ALL)
        gs.discard_cards = Deck("Discard", VisibilityMode.HIDDEN_TO_ALL)
        
        # Initialize board and game components
        gs.board = GridBoard(params.board_size, params.board_size)
        gs.extra_tiles = set()
        gs.bonuses = set()
        gs.milestones = set()
        gs.awards = set()
        gs.global_parameters = {}
        
        # Load base expansion
        Expansion.BASE.load_project_cards(gs.project_cards)
        Expansion.BASE.load_corp_cards(gs.corp_cards)
        Expansion.BASE.load_board(
            gs.board, gs.extra_tiles, gs.bonuses,
            gs.milestones, gs.awards, gs.global_parameters
        )
        
        # Handle Hellas/Elysium expansions
        if (Expansion.HELLAS in params.expansions or 
            Expansion.ELYSIUM in params.expansions):
            # Clear milestones and awards, they'll be replaced by these expansions
            gs.milestones.clear()
            gs.awards.clear()
        
        # Load other expansions
        for expansion in params.expansions:
            if expansion not in [Expansion.HELLAS, Expansion.ELYSIUM]:
                # Hellas and Elysium don't have project or corporation cards
                expansion.load_project_cards(gs.project_cards)
                expansion.load_corp_cards(gs.corp_cards)
            expansion.load_board(
                gs.board, gs.extra_tiles, gs.bonuses,
                gs.milestones, gs.awards, gs.global_parameters
            )
        
        # Disable milestones and awards for solo play
        if gs.get_n_players() == 1:
            gs.milestones = set()
            gs.awards = set()
        
        # Shuffle decks
        gs.project_cards.shuffle(gs.get_rnd())
        gs.corp_cards.shuffle(gs.get_rnd())
        
        # Initialize player-specific components
        gs.player_corporations = [None] * gs.get_n_players()
        gs.player_card_choice = [None] * gs.get_n_players()
        gs.player_hands = [None] * gs.get_n_players()
        gs.player_complicated_point_cards = [None] * gs.get_n_players()
        gs.played_cards = [None] * gs.get_n_players()
        gs.player_card_points = [None] * gs.get_n_players()
        
        for i in range(gs.get_n_players()):
            gs.player_hands[i] = Deck(f"Hand of p{i}", i, VisibilityMode.VISIBLE_TO_OWNER)
            gs.player_card_choice[i] = Deck(
                f"Card Choice for p{i}", i, VisibilityMode.VISIBLE_TO_OWNER
            )
            gs.player_complicated_point_cards[i] = Deck(
                f"Resource or Points Cards Played by p{i}", i, VisibilityMode.VISIBLE_TO_ALL
            )
            gs.played_cards[i] = Deck(
                f"Other Cards Played by p{i}", i, VisibilityMode.VISIBLE_TO_ALL
            )
            gs.player_card_points[i] = Counter(0, 0, params.max_points, f"Points of p{i}")
        
        # Initialize tracking dictionaries
        gs.player_tiles_placed = [dict() for _ in range(gs.get_n_players())]
        gs.player_cards_played_types = [dict() for _ in range(gs.get_n_players())]
        gs.player_cards_played_tags = [dict() for _ in range(gs.get_n_players())]
        gs.player_extra_actions = [set() for _ in range(gs.get_n_players())]
        gs.player_persisting_effects = [set() for _ in range(gs.get_n_players())]
        
        for i in range(gs.get_n_players()):
            gs.player_tiles_placed[i] = {}
            for tile in Tile:
                gs.player_tiles_placed[i][tile] = Counter(
                    0, 0, params.max_points, f"{tile.name} tiles placed player {i}"
                )
            
            gs.player_cards_played_types[i] = {}
            for card_type in CardType:
                gs.player_cards_played_types[i][card_type] = Counter(
                    0, 0, params.max_points, f"{card_type.name} cards played player {i}"
                )
            
            gs.player_cards_played_tags[i] = {}
            for tag in Tag:
                gs.player_cards_played_tags[i][tag] = Counter(
                    0, 0, params.max_points, f"{tag.name} cards played player {i}"
                )
            
            gs.player_extra_actions[i] = set()
            gs.player_persisting_effects[i] = set()
        
        gs.n_awards_funded = Counter(0, 0, len(params.n_cost_awards), "Awards funded")
        gs.n_milestones_claimed = Counter(0, 0, len(params.n_cost_milestone), "Milestones claimed")
        
        # Set initial game phase and deal corporation cards
        gs.set_game_phase(TMPhase.CORPORATION_SELECT)
        for i in range(gs.get_n_players()):
            for j in range(params.n_corp_choice_start):
                gs.player_card_choice[i].add(gs.corp_cards.pick(0))
        
        # Solo setup: place cities randomly with adjacent greeneries
        if gs.get_n_players() == 1:
            gs.get_turn_order().set_turn_owner(1)
            for i in range(params.solo_cities):
                # Place city + greenery adjacent
                pt = PlaceTile(1, Tile.CITY, MapTileType.GROUND, True)
                actions = pt._compute_available_actions(gs)
                action = actions[gs.get_rnd().randint(0, len(actions) - 1)]
                action.execute(gs)
                
                mt = gs.get_component_by_id(action.map_tile_id)
                neighbours = PlaceTile.get_neighbours(Vector2D(mt.get_x(), mt.get_y()))
                placed = False
                
                while not placed:
                    v = neighbours[gs.get_rnd().randint(0, len(neighbours) - 1)]
                    mtn = gs.board.get_element(v.get_x(), v.get_y())
                    if (mtn is not None and mtn.get_owner_id() == -1 and 
                        mtn.get_tile_type() == MapTileType.GROUND):
                        mtn.set_tile_placed(Tile.GREENERY, gs)
                        placed = True
            
            gs.get_turn_order().set_turn_owner(0)
            gs.global_parameters[GlobalParameter.OXYGEN].set_value(0)
        
        gs.generation = 1
    
    def _after_action(self, current_state: AbstractGameState, action) -> None:
        gs = current_state
        params = gs.get_game_parameters()
        
        if gs.get_game_phase() == TMPhase.CORPORATION_SELECT:
            all_chosen = all(card is not None for card in gs.get_player_corporations())
            if all_chosen:
                gs.set_game_phase(TMPhase.RESEARCH)
                gs.get_turn_order().end_round(gs)
                for i in range(gs.get_n_players()):
                    for j in range(params.n_projects_start):
                        gs.player_card_choice[i].add(gs.draw_card())
        
        elif gs.get_game_phase() == TMPhase.RESEARCH:
            # Check if finished: no more cards in card choice decks
            all_done = all(deck.get_size() == 0 for deck in gs.get_player_card_choice())
            if all_done:
                gs.set_game_phase(TMPhase.ACTIONS)
                gs.get_turn_order().end_round(gs)
        
        elif gs.get_game_phase() == TMPhase.ACTIONS:
            # Check if finished: all players passed
            if gs.get_turn_order().n_passed == gs.get_n_players():
                # Production phase
                for i in range(gs.get_n_players()):
                    # First, energy turns to heat
                    energy_val = gs.get_player_resources()[i][Resource.ENERGY].get_value()
                    gs.get_player_resources()[i][Resource.HEAT].increment(energy_val)
                    gs.get_player_resources()[i][Resource.ENERGY].set_value(0)
                    
                    # Then, all production values are added to resources
                    for res in Resource:
                        if res.is_player_board_res():
                            production_val = gs.get_player_production()[i][res].get_value()
                            gs.get_player_resources()[i][res].increment(production_val)
                    
                    # TR also adds to mega credits
                    tr_val = gs.player_resources[i][Resource.TR].get_value()
                    gs.get_player_resources()[i][Resource.MEGA_CREDIT].increment(tr_val)
                
                # Check game end before next research phase
                if self.check_game_end(gs):
                    if gs.get_n_players() == 1:
                        # Solo game end condition
                        won = GameResult.WIN_GAME
                        for param in gs.global_parameters:
                            if (param is not None and param.counts_for_end_game() and 
                                not gs.global_parameters[param].is_maximum()):
                                won = GameResult.LOSE_GAME
                        gs.set_game_status(GameResult.GAME_END)
                        gs.set_player_result(won, 0)
                    else:
                        self.end_game(gs)
                    return
                
                # Move to research phase
                gs.get_turn_order().end_round(gs)
                gs.set_game_phase(TMPhase.RESEARCH)
                
                # Deal research cards
                for j in range(params.n_projects_research):
                    for i in range(gs.get_n_players()):
                        c = gs.draw_card()
                        if c is not None:
                            gs.player_card_choice[i].add(c)
                        else:
                            break
                
                # Reset player states for new generation
                for i in range(gs.get_n_players()):
                    # Mark player actions unused
                    for c in gs.player_complicated_point_cards[i].get_components():
                        c.action_played = False
                    
                    # Reset resource increase
                    for res in Resource:
                        gs.player_resource_increase_gen[i][res] = False
                
                # Next generation
                gs.generation += 1
    
    def _compute_available_actions(self, game_state: AbstractGameState) -> List:
        actions = []
        gs = game_state
        player = gs.get_current_player()
        
        possible_actions = self.get_all_actions(gs)
        
        # Wrap actions that can actually be played and must be paid for
        for action in possible_actions:
            if action is not None and action.can_be_played(gs):
                if action.get_cost() != 0:
                    actions.append(PayForAction(player, action))
                else:
                    actions.append(action)
        
        return actions
    
    def get_all_actions(self, gs: TMGameState) -> List:
        """
        Bypass regular compute_actions function call to list all actions possible
        in the current state, some of which might not be playable at the moment.
        """
        # If there is an action in progress, delegate to that
        if gs.is_action_in_progress():
            return gs.get_actions_in_progress().peek()._compute_available_actions(gs)
        
        params = gs.get_game_parameters()
        player = gs.get_current_player()
        possible_actions = []
        
        if gs.get_game_phase() == TMPhase.CORPORATION_SELECT:
            # Decide one card at a time
            card_choice = gs.get_player_card_choice()[player]
            if card_choice.get_size() == 0:
                possible_actions.append(TMAction(player))  # Pass
            else:
                for i in range(card_choice.get_size()):
                    possible_actions.append(
                        BuyCard(player, card_choice.get(i).get_component_id(), 0)
                    )
        
        elif gs.get_game_phase() == TMPhase.RESEARCH:
            # Decide one card at a time
            card_choice = gs.get_player_card_choice()[player]
            if card_choice.get_size() == 0:
                possible_actions.append(TMAction(player))  # Pass
            else:
                card_choice.get(0).action_played = False
                buy_action = BuyCard(
                    player, card_choice.get(0).get_component_id(),
                    params.get_project_purchase_cost()
                )
                if buy_action.can_be_played(gs):
                    possible_actions.append(buy_action)
                possible_actions.append(
                    DiscardCard(player, card_choice.get(0).get_component_id(), True)
                )
        
        else:  # Actions phase
            if gs.generation == 1:
                # Check if any players have decided first action from corporations
                corp_card = gs.player_corporations[player]
                if (corp_card.first_action is not None and 
                    not corp_card.first_action_executed):
                    possible_actions.append(corp_card.first_action)
                    return possible_actions
            
            possible_actions.append(TMAction(player))  # Can always pass
            
            # Play a card actions
            for i in range(gs.player_hands[player].get_size()):
                possible_actions.append(
                    PlayCard(player, gs.player_hands[player].get(i), False)
                )
            
            # Standard projects
            # Discard cards for MC
            if gs.player_hands[player].get_size() > 0:
                possible_actions.append(SellProjects(player))
            
            # Increase energy production for 11 MC
            possible_actions.append(
                ModifyPlayerResource(
                    StandardProject.POWER_PLANT, params.get_n_cost_sp_energy(),
                    player, 1, Resource.ENERGY
                )
            )
            
            # Increase temperature for 14 MC
            possible_actions.append(
                ModifyGlobalParameter(
                    ActionType.STANDARD_PROJECT, Resource.MEGA_CREDIT,
                    params.get_n_cost_sp_temp(), GlobalParameter.TEMPERATURE, 1, False
                )
            )
            
            # Place ocean tile for 18 MC
            possible_actions.append(
                PlaceTile(
                    StandardProject.AQUIFER, params.get_n_cost_sp_ocean(),
                    player, Tile.OCEAN, MapTileType.OCEAN
                )
            )
            
            # Place greenery tile for 23 MC
            possible_actions.append(
                PlaceTile(
                    StandardProject.GREENERY, params.get_n_cost_sp_greenery(),
                    player, Tile.GREENERY, MapTileType.GROUND
                )
            )
            
            # Place city tile and increase MC prod for 25 MC
            a1 = PlaceTile(player, Tile.CITY, MapTileType.GROUND, True)
            a2 = ModifyPlayerResource(player, params.n_sp_city_mc_gain, Resource.MEGA_CREDIT, True)
            possible_actions.append(
                CompoundAction(
                    ActionType.STANDARD_PROJECT, player, [a1, a2], params.n_cost_sp_city
                )
            )
            
            # Venus air scraping if Venus expansion enabled
            if Expansion.VENUS in params.expansions:
                possible_actions.append(
                    ModifyGlobalParameter(
                        ActionType.STANDARD_PROJECT, Resource.MEGA_CREDIT,
                        params.n_cost_venus, GlobalParameter.VENUS, 1, False
                    )
                )
            
            # Claim milestones
            n_milestones_claimed = gs.get_n_milestones_claimed().get_value()
            milestone_cost = 0
            if not gs.get_n_milestones_claimed().is_maximum():
                milestone_cost = params.get_n_cost_milestone()[n_milestones_claimed]
            for milestone in gs.milestones:
                possible_actions.append(
                    ClaimAwardMilestone(player, milestone, milestone_cost)
                )
            
            # Fund awards
            n_awards_funded = gs.get_n_awards_funded().get_value()
            award_cost = 0
            if not gs.get_n_awards_funded().is_maximum():
                award_cost = params.get_n_cost_awards()[n_awards_funded]
            for award in gs.awards:
                possible_actions.append(
                    ClaimAwardMilestone(player, award, award_cost)
                )
            
            # Use active card actions
            possible_actions.extend(gs.player_extra_actions[player])
            
            # 8 plants into greenery tile
            possible_actions.append(
                PlaceTile(
                    BasicResourceAction.PLANT_TO_GREENERY,
                    params.get_n_cost_greenery_plant(), player,
                    Tile.GREENERY, MapTileType.GROUND
                )
            )
            
            # 8 heat into temperature increase
            possible_actions.append(
                ModifyGlobalParameter(
                    ActionType.BASIC_RESOURCE_ACTION, Resource.HEAT,
                    params.get_n_cost_temp_heat(), GlobalParameter.TEMPERATURE, 1, False
                )
            )
        
        return possible_actions
    
    def illegal_action_played(self, game_state: AbstractGameState, action) -> None:
        self.next(game_state, TMAction(game_state.get_current_player()))
    
    def check_game_end(self, gs: TMGameState) -> bool:
        ended = True
        if gs.get_n_players() == 1:
            # Solo game goes for 14 generations regardless of global parameters
            if gs.generation < gs.get_game_parameters().solo_max_gen:
                ended = False
        else:
            for param in gs.global_parameters:
                if (param is not None and param.counts_for_end_game() and 
                    not gs.global_parameters[param].is_maximum()):
                    ended = False
        
        return ended