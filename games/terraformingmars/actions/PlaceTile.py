from typing import List, Set, Optional, Dict
from copy import deepcopy

from core import AbstractGameState
from core.actions import AbstractAction
from core.interfaces import IExtendedSequence
from games.terraformingmars import TMAction
from games.terraformingmars import TMGameParameters
from games.terraformingmars import TMGameState
from games.terraformingmars import TMTypes, neighbor_directions
from games.terraformingmars.components import TMCard
from games.terraformingmars.components import TMMapTile
from games.terraformingmars.rules.requirements import AdjacencyRequirement
from games.terraformingmars.actions import ModifyPlayerResource
from utilities import Group
from utilities import Vector2D


class PlaceTile(TMAction, IExtendedSequence):
    def __init__(self, player: int = -1, map_tile_id: int = -1, tile: Optional[TMTypes.Tile] = None,
                 respecting_adjacency: bool = True, on_mars: bool = True, tile_name: Optional[str] = None,
                 map_type: Optional[TMTypes.MapTileType] = None, legal_positions: Optional[Set[int]] = None,
                 resources_gained_restriction: Optional[List[TMTypes.Resource]] = None,
                 volcanic_restriction: bool = False, adjacency_requirement: Optional[AdjacencyRequirement] = None,
                 free: bool = False, standard_project: Optional[TMTypes.StandardProject] = None,
                 basic_resource_action: Optional[TMTypes.BasicResourceAction] = None, cost: int = 0):
        
        if standard_project is not None:
            super().__init__(standard_project=standard_project, player=player, free=False)
            self.set_action_cost(TMTypes.Resource.MegaCredit, cost, -1)
        elif basic_resource_action is not None:
            super().__init__(basic_resource_action=basic_resource_action, player=player, free=False)
            self.set_action_cost(TMTypes.Resource.Plant, cost, -1)
        else:
            super().__init__(player=player, free=free)
        
        self.respecting_adjacency = respecting_adjacency
        self.on_mars = on_mars
        self.tile_name = tile_name
        self.map_tile_id = map_tile_id
        self.tile = tile
        self.map_type = map_type
        self.legal_positions = legal_positions
        self.resources_gained_restriction = resources_gained_restriction
        self.volcanic_restriction = volcanic_restriction
        self.adjacency_requirement = adjacency_requirement
        
        self.remove_resources_adjacent_owner = False
        self.remove_resources_amount = 0
        self.remove_resources_res = None
        self.remove_resources_prod = False
        
        self.placed = False
        self.impossible = False
        
        if tile is not None:
            self._set_tile(tile, tile_name)

    def _set_tile(self, tile: TMTypes.Tile, name: Optional[str]):
        self.tile = tile
        if tile == TMTypes.Tile.City and name is None:
            # Cities can't be adjacent to other cities, unless named
            self.adjacency_requirement = AdjacencyRequirement({TMTypes.Tile.City: 1})
            self.adjacency_requirement.reversed = True
        elif tile == TMTypes.Tile.Greenery:
            # Greeneries must be adjacent to other owned tiles
            self.adjacency_requirement = AdjacencyRequirement()
            self.adjacency_requirement.owned = True

    def _execute(self, gs: TMGameState) -> bool:
        if self.map_tile_id != -1 and self.tile is not None:
            mt = gs.get_component_by_id(self.map_tile_id)
            success = mt.place_tile(self.tile, gs)
            
            if success and self.on_mars:
                if self.get_card_id() != -1:
                    # Save location of tile placed on card
                    card = gs.get_component_by_id(self.get_card_id())
                    card.map_tile_id_tile_placed = mt.get_component_id()
                
                if self.player < 0 or self.player >= gs.get_n_players():
                    return super()._execute(gs)

                # Add money earned from adjacent oceans
                n_oceans = self.n_adjacent_tiles(gs, mt, TMTypes.Tile.Ocean)
                gs.get_player_resources()[self.player][TMTypes.Resource.MegaCredit].increment(
                    n_oceans * gs.get_game_parameters().get_n_mc_gained_ocean()
                )

                if self.resources_gained_restriction is not None:
                    # Production of each resource type gained increased by 1
                    gained = mt.get_resources()
                    types_added = set()
                    for r in gained:
                        if self._contains_resource(self.resources_gained_restriction, r) and r not in types_added:
                            gs.get_player_production()[self.player][r].increment(1)
                            types_added.add(r)

                if self.remove_resources_adjacent_owner:
                    adjacent_owners = set()
                    neighbours = self.get_neighbours(Vector2D(mt.get_x(), mt.get_y()))
                    for n in neighbours:
                        other = gs.get_board().get_element(n.get_x(), n.get_y())
                        if other is not None and other.get_tile_placed() is not None:
                            adjacent_owners.add(other.get_owner_id())
                    
                    if len(adjacent_owners) > 0:
                        mpr = ModifyPlayerResource(self.player, -self.remove_resources_amount, 
                                                 self.remove_resources_res, self.remove_resources_prod)
                        mpr.target_player = -2
                        mpr.target_player_options = adjacent_owners
                        mpr.execute(gs)

            return success and super()._execute(gs)
        
        gs.set_action_in_progress(self)
        return True

    def _copy(self):
        copy = PlaceTile(
            player=self.player,
            map_tile_id=self.map_tile_id,
            tile=self.tile,
            respecting_adjacency=self.respecting_adjacency,
            on_mars=self.on_mars,
            tile_name=self.tile_name,
            map_type=self.map_type,
            legal_positions=deepcopy(self.legal_positions) if self.legal_positions else None,
            resources_gained_restriction=self.resources_gained_restriction.copy() if self.resources_gained_restriction else None,
            volcanic_restriction=self.volcanic_restriction,
            adjacency_requirement=self.adjacency_requirement.copy() if self.adjacency_requirement else None,
            free=self.free_action_point
        )
        
        copy.impossible = self.impossible
        copy.placed = self.placed
        copy.remove_resources_adjacent_owner = self.remove_resources_adjacent_owner
        copy.remove_resources_amount = self.remove_resources_amount
        copy.remove_resources_res = self.remove_resources_res
        copy.remove_resources_prod = self.remove_resources_prod
        
        return copy

    def copy(self):
        return super().copy()

    def _compute_available_actions(self, state: AbstractGameState) -> List[AbstractAction]:
        actions = []
        if self.map_tile_id == -1:
            # Need to choose where to place it
            gs = state
            if self.legal_positions is not None:
                for pos in self.legal_positions:
                    mt = gs.get_component_by_id(pos)
                    if mt is not None and mt.get_tile_placed() is None:
                        actions.append(PlaceTile(
                            player=self.player, map_tile_id=pos, tile=self.tile,
                            respecting_adjacency=self.respecting_adjacency, on_mars=self.on_mars,
                            tile_name=self.tile_name, map_type=self.map_type,
                            legal_positions=self.legal_positions,
                            resources_gained_restriction=self.resources_gained_restriction,
                            volcanic_restriction=self.volcanic_restriction,
                            adjacency_requirement=self.adjacency_requirement, free=True
                        ))
            else:
                if self.on_mars:
                    for i in range(gs.get_board().get_height()):
                        for j in range(gs.get_board().get_width()):
                            mt = gs.get_board().get_element(j, i)

                            # Check if we can place tile here
                            if mt is None or mt.get_tile_placed() is not None:
                                continue
                            if mt.is_reserved() and mt.get_reserved() != self.player:
                                continue
                            if self.tile_name is not None and not mt.get_component_name().lower() == self.tile_name.lower():
                                continue
                            if self.map_type is not None and mt.get_tile_type() != self.map_type:
                                continue
                            if self.volcanic_restriction and not mt.is_volcanic():
                                continue
                            if (self.resources_gained_restriction is not None and 
                                not self._contains_resources(mt.get_resources(), self.resources_gained_restriction)):
                                continue

                            # Check placement rules
                            if self.respecting_adjacency and self.adjacency_requirement is not None:
                                if self.adjacency_requirement.test_condition(Group(gs, mt, self.player)):
                                    actions.append(PlaceTile(
                                        player=self.player, map_tile_id=mt.get_component_id(), tile=self.tile,
                                        respecting_adjacency=self.respecting_adjacency, on_mars=self.on_mars,
                                        tile_name=self.tile_name, map_type=self.map_type,
                                        legal_positions=self.legal_positions,
                                        resources_gained_restriction=self.resources_gained_restriction,
                                        volcanic_restriction=self.volcanic_restriction,
                                        adjacency_requirement=self.adjacency_requirement, free=True
                                    ))
                            else:
                                actions.append(PlaceTile(
                                    player=self.player, map_tile_id=mt.get_component_id(), tile=self.tile,
                                    respecting_adjacency=self.respecting_adjacency, on_mars=self.on_mars,
                                    tile_name=self.tile_name, map_type=self.map_type,
                                    legal_positions=self.legal_positions,
                                    resources_gained_restriction=self.resources_gained_restriction,
                                    volcanic_restriction=self.volcanic_restriction,
                                    adjacency_requirement=self.adjacency_requirement, free=True
                                ))
                else:
                    for mt in gs.get_extra_tiles():
                        if mt.get_component_name().lower() == self.tile_name.lower():
                            actions.append(PlaceTile(
                                player=self.player, map_tile_id=mt.get_component_id(), tile=self.tile,
                                respecting_adjacency=self.respecting_adjacency, on_mars=self.on_mars,
                                tile_name=self.tile_name, map_type=self.map_type,
                                legal_positions=self.legal_positions,
                                resources_gained_restriction=self.resources_gained_restriction,
                                volcanic_restriction=self.volcanic_restriction,
                                adjacency_requirement=self.adjacency_requirement, free=True
                            ))
                            break

            if len(actions) == 0:
                self.impossible = True
                actions.append(TMAction(self.player))
        else:
            self.impossible = True
            actions.append(TMAction(self.player))
        
        return actions

    def get_current_player(self, state: AbstractGameState) -> int:
        return self.player

    def _after_action(self, state: AbstractGameState, action: AbstractAction):
        self.placed = True

    def execution_complete(self, state: AbstractGameState) -> bool:
        return self.placed or self.impossible

    def get_string(self, game_state: AbstractGameState) -> str:
        mt = game_state.get_component_by_id(self.map_tile_id)
        if mt is not None:
            tile_desc = self.tile_name if self.tile_name is not None else f"{self.tile.name} on {mt.get_tile_type()}"
            return f"Place {tile_desc}"
        else:
            tile_desc = self.tile_name if self.tile_name is not None else self.tile.name
            return f"Place {tile_desc}"

    def __str__(self):
        return f"Place {self.tile.name}"

    def __eq__(self, other):
        if not isinstance(other, PlaceTile):
            return False
        if not super().__eq__(other):
            return False
        
        return (self.respecting_adjacency == other.respecting_adjacency and
                self.on_mars == other.on_mars and
                self.map_tile_id == other.map_tile_id and
                self.volcanic_restriction == other.volcanic_restriction and
                self.remove_resources_adjacent_owner == other.remove_resources_adjacent_owner and
                self.remove_resources_amount == other.remove_resources_amount and
                self.remove_resources_prod == other.remove_resources_prod and
                self.placed == other.placed and
                self.impossible == other.impossible and
                self.tile_name == other.tile_name and
                self.tile == other.tile and
                self.map_type == other.map_type and
                self.legal_positions == other.legal_positions and
                self.resources_gained_restriction == other.resources_gained_restriction and
                self.adjacency_requirement == other.adjacency_requirement and
                self.remove_resources_res == other.remove_resources_res)

    def __hash__(self):
        return hash((
            super().__hash__(),
            self.respecting_adjacency,
            self.on_mars,
            self.tile_name,
            self.map_tile_id,
            self.tile,
            self.map_type,
            tuple(self.legal_positions) if self.legal_positions else None,
            self.volcanic_restriction,
            self.adjacency_requirement,
            self.remove_resources_adjacent_owner,
            self.remove_resources_amount,
            self.remove_resources_res,
            self.remove_resources_prod,
            self.placed,
            self.impossible,
            tuple(self.resources_gained_restriction) if self.resources_gained_restriction else None
        ))

    @staticmethod
    def is_adjacent_to_player_owned_tiles(gs: TMGameState, mt: TMMapTile, player: int) -> bool:
        placed_any_tiles = gs.has_placed_tile(player)
        if placed_any_tiles:
            neighbours = PlaceTile.get_neighbours(Vector2D(mt.get_x(), mt.get_y()))
            for n in neighbours:
                other = gs.get_board().get_element(n.get_x(), n.get_y())
                if other is not None and other.get_owner_id() == player:
                    return True
            return False
        return True

    @staticmethod
    def is_adjacent_to_any(gs: TMGameState, mt: TMMapTile) -> bool:
        placed_any_tiles = gs.any_tiles_placed()
        if placed_any_tiles:
            neighbours = PlaceTile.get_neighbours(Vector2D(mt.get_x(), mt.get_y()))
            for n in neighbours:
                other = gs.get_board().get_element(n.get_x(), n.get_y())
                if other is not None and other.get_tile_placed() is not None:
                    return True
            return False
        return False

    @staticmethod
    def n_adjacent_tiles(gs: TMGameState, mt: TMMapTile, tile_type: Optional[TMTypes.Tile] = None) -> int:
        placed_any_tiles = gs.any_tiles_placed()
        count = 0
        if placed_any_tiles:
            neighbours = PlaceTile.get_neighbours(Vector2D(mt.get_x(), mt.get_y()))
            for n in neighbours:
                other = gs.get_board().get_element(n.get_x(), n.get_y())
                if other is not None:
                    if tile_type is None and other.get_tile_placed() is not None:
                        count += 1
                    elif other.get_tile_placed() == tile_type:
                        count += 1
        return count

    @staticmethod
    def is_adjacent_to_tile(gs: TMGameState, mt: TMMapTile, t: TMTypes.Tile) -> bool:
        placed_any_tiles = gs.any_tiles_placed()
        if placed_any_tiles:
            neighbours = PlaceTile.get_neighbours(Vector2D(mt.get_x(), mt.get_y()))
            for n in neighbours:
                other = gs.get_board().get_element(n.get_x(), n.get_y())
                if other is not None and other.get_tile_placed() == t:
                    return True
            return False
        return True

    @staticmethod
    def get_neighbours(cell: Vector2D) -> List[Vector2D]:
        neighbours = []
        parity = abs(cell.get_y() % 2)
        for v in neighbor_directions[parity]:
            neighbours.append(cell.add(v))
        return neighbours

    @staticmethod
    def _contains_resources(array: List[TMTypes.Resource], objects: List[TMTypes.Resource]) -> bool:
        for r1 in array:
            for r2 in objects:
                if r1 == r2:
                    return True
        return False

    @staticmethod
    def _contains_resource(array: List[TMTypes.Resource], r2: TMTypes.Resource) -> bool:
        return r2 in array