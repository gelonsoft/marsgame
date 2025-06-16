from typing import Any, Dict, Optional, TypeVar, Generic
from enum import Enum
from games.terraformingmars import TMTypes
from games.terraformingmars import TMGameState
from games.terraformingmars.components import TMMapTile



class Group(Generic[TMGameState, TMMapTile, int]):
    def __init__(self, a: TMGameState, b: TMMapTile, c: int):
        self.a = a
        self.b = b
        self.c = c

class TMTypes:
    class Tile(Enum):
        pass  # Define your Tile enum values here

class AdjacencyRequirement:
    def __init__(self, tile_types: Optional[Dict[TMTypes.Tile, int]] = None):
        self.tile_types = tile_types if tile_types is not None else None
        self.owned = False
        self.none_adjacent = False
        self.reversed = False
    
    def test_condition(self, group: Group[TMGameState, TMMapTile, int]) -> bool:
        if self.owned:
            return self.is_adjacent_to_player_owned_tiles(group.a, group.b, group.c)
        if self.none_adjacent:
            is_adjacent = self.is_adjacent_to_any(group.a, group.b)
            return is_adjacent if self.reversed else not is_adjacent
        if self.tile_types is not None:
            for tile, count in self.tile_types.items():
                adjacent_count = self.n_adjacent_tiles(group.a, group.b, tile)
                if not self.reversed and adjacent_count < count:
                    return False
                if self.reversed and adjacent_count >= count:
                    return False
        return True
    
    def is_max(self) -> bool:
        return False
    
    def applies_when_any_player(self) -> bool:
        return False
    
    def get_display_text(self, gs: TMGameState) -> Optional[str]:
        return None
    
    def get_reason_for_failure(self, gs: TMGameState) -> Optional[str]:
        return None
    
    def get_display_images(self) -> Optional[list[Any]]:  # Adjust Image type as needed
        return None
    
    def copy(self) -> 'AdjacencyRequirement':
        copy = AdjacencyRequirement()
        if self.tile_types is not None:
            copy.tile_types = self.tile_types.copy()
        copy.none_adjacent = self.none_adjacent
        copy.owned = self.owned
        copy.reversed = self.reversed
        return copy
    
    def copy_serializable(self) -> 'AdjacencyRequirement':
        copy = AdjacencyRequirement()
        if self.tile_types is not None and len(self.tile_types) > 0:
            copy.tile_types = self.tile_types.copy()
        copy.none_adjacent = self.none_adjacent
        copy.owned = self.owned
        copy.reversed = self.reversed
        return copy
    
    def __str__(self) -> str:
        return "Adjacency"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdjacencyRequirement):
            return False
        return (self.owned == other.owned and 
                self.none_adjacent == other.none_adjacent and 
                self.reversed == other.reversed and 
                self.tile_types == other.tile_types)
    
    def __hash__(self) -> int:
        return hash((frozenset(self.tile_types.items()) if self.tile_types else None, 
                   self.owned, self.none_adjacent, self.reversed))

    # Placeholder for static methods - implement these as needed
    @staticmethod
    def is_adjacent_to_player_owned_tiles(gs: TMGameState, tile: TMMapTile, player: int) -> bool:
        raise NotImplementedError("Implement this method")
    
    @staticmethod
    def is_adjacent_to_any(gs: TMGameState, tile: TMMapTile) -> bool:
        raise NotImplementedError("Implement this method")
    
    @staticmethod
    def n_adjacent_tiles(gs: TMGameState, tile: TMMapTile, tile_type: TMTypes.Tile) -> int:
        raise NotImplementedError("Implement this method")