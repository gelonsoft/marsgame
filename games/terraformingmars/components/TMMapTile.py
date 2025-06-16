from typing import Optional, List
from enum import Enum
from games.terraformingmars import TMTypes
from games.terraformingmars import TMGameState


class TMMapTile:
    def __init__(self, x: int, y: int, component_id: int = -1):
        self.component_id = component_id
        self.component_name = "Tile"
        self.x = x
        self.y = y
        self.tile_placed: Optional[TMTypes.Tile] = None
        self.type: Optional[TMTypes.MapTileType] = None
        self.resources: List[TMTypes.Resource] = []
        self.volcanic = False
        self.reserved = -1
        self.owner_id = -1
    
    def is_reserved(self) -> bool:
        return self.reserved != -1
    
    def get_reserved(self) -> int:
        return self.reserved
    
    def set_reserved(self, reserved: int) -> None:
        self.reserved = reserved
    
    def is_volcanic(self) -> bool:
        return self.volcanic
    
    def set_volcanic(self, volcanic: bool) -> None:
        self.volcanic = volcanic
    
    def set_type(self, tile_type: TMTypes.MapTileType) -> None:
        self.type = tile_type
    
    def set_resources(self, resources: List[TMTypes.Resource]) -> None:
        self.resources = resources.copy() if resources else []
    
    def get_tile_type(self) -> Optional[TMTypes.MapTileType]:
        return self.type
    
    def get_resources(self) -> List[TMTypes.Resource]:
        return self.resources.copy()
    
    def get_tile_placed(self) -> Optional[TMTypes.Tile]:
        return self.tile_placed
    
    def get_x(self) -> int:
        return self.x
    
    def get_y(self) -> int:
        return self.y
    
    def set_tile_placed(self, tile: TMTypes.Tile, gs: TMGameState) -> None:
        self.tile_placed = tile
        player = gs.get_current_player()
        
        if tile.can_be_owned():  # Assuming Tile enum has this method
            self.owner_id = player
        
        if 0 <= player < gs.get_n_players():
            # Increment tile count for player
            gs.get_player_tiles_placed().setdefault(player, {}).setdefault(tile, 0)
            gs.get_player_tiles_placed()[player][tile] += 1
            
            # Add resources to player
            for res in self.resources:
                gs.get_player_resources().setdefault(player, {}).setdefault(res, 0)
                gs.get_player_resources()[player][res] += 1
                gs.get_player_resource_increase_gen().setdefault(player, {}).setdefault(res, True)
    
    def place_tile(self, tile: TMTypes.Tile, gs: TMGameState) -> bool:
        if self.tile_placed is None:
            gp = tile.get_global_parameter_to_increase()  # Assuming Tile enum has this method
            self.set_tile_placed(tile, gs)
            if gp is not None:
                # Increase global parameter - implement ModifyGlobalParameter in Python
                return ModifyGlobalParameter(gp, 1, True).execute(gs)
            return True
        return False
    
    def remove_tile(self) -> None:
        self.owner_id = -1
        self.tile_placed = None
    
    def copy(self) -> 'TMMapTile':
        new_tile = TMMapTile(self.x, self.y, self.component_id)
        new_tile.component_name = self.component_name
        new_tile.tile_placed = self.tile_placed
        new_tile.type = self.type
        new_tile.resources = self.resources.copy()
        new_tile.volcanic = self.volcanic
        new_tile.reserved = self.reserved
        new_tile.owner_id = self.owner_id
        return new_tile
    
    @staticmethod
    def parse_map_tile(s: str, x: int = -1, y: int = -1) -> Optional['TMMapTile']:
        if s == "0":
            return None
        
        mt = TMMapTile(x, y)
        split = s.split(":")
        
        # First element is tile type
        try:
            tile_type = TMTypes.MapTileType(split[0])
            if tile_type == TMTypes.MapTileType.Volcanic:
                tile_type = TMTypes.MapTileType.Ground
                mt.set_volcanic(True)
            mt.set_type(tile_type)
        except ValueError:
            mt.set_type(TMTypes.MapTileType.City)
            mt.component_name = split[0]  # Keep city name
        
        # The rest are resources
        resources = []
        for res_str in split[1:]:
            try:
                res = TMTypes.Resource(res_str)
                resources.append(res)
            except ValueError:
                pass  # Handle special cases like Ocean or MegaCredit/-6
        
        mt.set_resources(resources)
        return mt
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TMMapTile):
            return False
        return (self.owner_id == other.owner_id and 
                self.x == other.x and 
                self.y == other.y and 
                self.volcanic == other.volcanic and 
                self.reserved == other.reserved and 
                self.tile_placed == other.tile_placed and 
                self.type == other.type and 
                self.resources == other.resources)
    
    def __hash__(self) -> int:
        return hash((self.owner_id, self.x, self.y, self.tile_placed, self.type, 
                    tuple(self.resources), self.volcanic, self.reserved))

class ModifyGlobalParameter:
    def __init__(self, param: TMTypes.GlobalParameter, amount: int, increase: bool):
        self.param = param
        self.amount = amount
        self.increase = increase
    
    def execute(self, gs: TMGameState) -> bool:
        # Implement global parameter modification logic
        return True