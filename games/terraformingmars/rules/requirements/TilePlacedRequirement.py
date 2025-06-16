from typing import List, Dict, Any
from games.terraformingmars.rules.requirements import Requirement
from games.terraformingmars import TMGameState, TMTypes
import utilities.ImageIO as ImageIO

class TilePlacedRequirement(Requirement):
    def __init__(self, tile: TMTypes.Tile, threshold: int, max: bool, any: bool):
        self.tile = tile
        self.threshold = threshold
        self.max = max
        self.any = any

    def test_condition(self, gs: TMGameState) -> bool:
        n_placed = self.n_placed(gs)
        if self.max and n_placed <= self.threshold:
            return True
        return not self.max and n_placed >= self.threshold

    def n_placed(self, gs: TMGameState) -> int:
        player = gs.getCurrentPlayer()
        n_placed = 0
        if not self.any:
            n_placed = gs.getPlayerTilesPlaced()[player].get(self.tile).getValue()
        else:
            for i in range(gs.getNPlayers()):
                n_placed = gs.getPlayerTilesPlaced()[i].get(self.tile).getValue()
            if gs.getNPlayers() == 1:
                if self.tile == TMTypes.Tile.City or self.tile == TMTypes.Tile.Greenery:
                    n_placed += (gs.getGameParameters()).getSoloCities()
        return n_placed

    def is_max(self) -> bool:
        return self.max

    def applies_when_any_player(self) -> bool:
        return self.any

    def get_display_text(self, gs: TMGameState) -> str:
        return None

    def get_reason_for_failure(self, gs: TMGameState) -> str:
        n_placed = self.n_placed(gs)
        return f"{n_placed}/{self.threshold} {self.tile} tiles placed{' by you' if not self.any else ''}"

    def get_display_images(self) -> List[Any]:
        return [ImageIO.GetInstance().getImage(self.tile.getImagePath())]

    def copy(self) -> 'TilePlacedRequirement':
        return self

    def __str__(self) -> str:
        return "Tile Placed"

    def __eq__(self, o: object) -> bool:
        if self is o:
            return True
        if not isinstance(o, TilePlacedRequirement):
            return False
        that = o
        return (self.threshold == that.threshold and 
                self.max == that.max and 
                self.any == that.any and 
                self.tile == that.tile)

    def __hash__(self) -> int:
        return hash((self.tile, self.threshold, self.max, self.any))