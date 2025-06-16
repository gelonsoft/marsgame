from typing import Optional, List, Set
from games.terraformingmars.rules.effects.effect import Effect
from games.terraformingmars.tm_types import Tile, Resource
from games.terraformingmars.actions.pay_for_action import PayForAction
from games.terraformingmars.actions.place_tile import PlaceTile
from games.terraformingmars.actions.tm_action import TMAction
from games.terraformingmars.tm_game_state import TMGameState
from games.terraformingmars.components.tm_map_tile import TMMapTile


class PlaceTileEffect(Effect):
    def __init__(self, must_be_current_player: bool, effect_action: TMAction, 
                 on_mars: bool, tile: Optional[Tile] = None, 
                 resource_type_gained: Optional[List[Resource]] = None):
        super().__init__(must_be_current_player, effect_action)
        self.on_mars = on_mars  # tile must've been placed on mars
        self.tile = tile
        self.resource_type_gained = resource_type_gained

    def can_execute(self, game_state: TMGameState, action_taken: TMAction, player: int) -> bool:
        # Check if action is PlaceTile or PayForAction containing PlaceTile
        if not (isinstance(action_taken, PlaceTile) or 
               (isinstance(action_taken, PayForAction) and isinstance(action_taken.action, PlaceTile))) or \
           not super().can_execute(game_state, action_taken, player):
            return False

        # Extract the PlaceTile action
        if isinstance(action_taken, PayForAction):
            action = action_taken.action
        else:
            action = action_taken

        # Check Mars condition
        mars_condition = not self.on_mars or action.on_mars
        
        # Check tile condition
        tile_condition = self.tile is None or action.tile == self.tile

        # Check resource type condition
        gained: Set[Resource] = set()
        if action.map_tile_id != -1 and action.on_mars:
            mt = game_state.get_component_by_id(action.map_tile_id)
            if isinstance(mt, TMMapTile):
                gained.update(mt.get_resources())

        resource_type_condition = self.resource_type_gained is None
        if self.resource_type_gained is not None:
            for r in self.resource_type_gained:
                if r in gained:
                    resource_type_condition = True
                    break

        return mars_condition and tile_condition and resource_type_condition

    def copy(self) -> 'PlaceTileEffect':
        ef = PlaceTileEffect(
            self.must_be_current_player,
            self.effect_action.copy(),
            self.on_mars,
            self.tile,
            self.resource_type_gained
        )
        if self.resource_type_gained is not None:
            ef.resource_type_gained = self.resource_type_gained.copy()
        return ef

    def copy_serializable(self) -> 'PlaceTileEffect':
        ef = PlaceTileEffect(
            self.must_be_current_player,
            self.effect_action.copy_serializable(),
            self.on_mars,
            self.tile,
            self.resource_type_gained
        )
        if self.resource_type_gained is not None and len(self.resource_type_gained) > 0:
            ef.resource_type_gained = self.resource_type_gained.copy()
        return ef