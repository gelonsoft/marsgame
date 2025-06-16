from core.actions import AbstractAction
from core.interfaces import IExtendedSequence
from games.terraformingmars.actions.TMAction import TMAction
from games.terraformingmars.components.TMMapTile import TMMapTile
from games.terraformingmars.TMTypes import MapTileType
from typing import List, Optional
import copy

class ReserveTile(TMAction, IExtendedSequence):
    def __init__(self, player: int = -1, map_tile_id: int = -1, free: bool = False, map_tile_type: Optional[MapTileType] = None):
        super().__init__(player, free)
        self.map_tile_id = map_tile_id
        self.map_type = map_tile_type
        self.placed = False
        self.impossible = False

    def _execute(self, gs) -> bool:
        if self.map_tile_id != -1:
            mt = gs.get_component_by_id(self.map_tile_id)
            mt.set_reserved(self.player)
            return True
        gs.set_action_in_progress(self)
        return True

    def _copy(self) -> 'ReserveTile':
        copy_tile = ReserveTile(self.player, self.map_tile_id, self.free_action_point, self.map_type)
        copy_tile.impossible = self.impossible
        copy_tile.placed = self.placed
        return copy_tile

    def copy(self) -> 'ReserveTile':
        return super().copy()

    def _compute_available_actions(self, state) -> List[AbstractAction]:
        actions = []
        if self.map_tile_id == -1:
            gs = state
            for i in range(gs.get_board().get_height()):
                for j in range(gs.get_board().get_width()):
                    mt = gs.get_board().get_element(j, i)
                    if (mt is not None and 
                        mt.get_tile_placed() is None and 
                        (mt.get_tile_type() == self.map_type)):
                        actions.append(ReserveTile(self.player, mt.get_component_id(), True))
            if not actions:
                self.impossible = True
                actions.append(TMAction(self.player))
        else:
            self.impossible = True
            actions.append(TMAction(self.player))
        return actions

    def get_current_player(self, state) -> int:
        return self.player

    def _after_action(self, state, action) -> None:
        self.placed = True

    def execution_complete(self, state) -> bool:
        return self.placed or self.impossible

    def __eq__(self, other) -> bool:
        if not isinstance(other, ReserveTile):
            return False
        return (super().__eq__(other) and 
               (self.map_tile_id == other.map_tile_id) and 
               (self.placed == other.placed) and 
               (self.impossible == other.impossible) and 
               (self.map_type == other.map_type))

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.map_tile_id, self.map_type, self.placed, self.impossible))

    def get_string(self, game_state) -> str:
        return "Reserve tile"

    def __str__(self) -> str:
        return "Reserve tile"