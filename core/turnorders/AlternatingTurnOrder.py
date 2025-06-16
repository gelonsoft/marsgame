from core.turnorders.TurnOrder import TurnOrder
from core import AbstractGameState

class AlternatingTurnOrder(TurnOrder):
    def __init__(self, nPlayers: int = None):
        super().__init__(nPlayers)
        self.direction = 1

    def _copy(self) -> 'AlternatingTurnOrder':
        to = AlternatingTurnOrder()
        to.direction = self.direction
        return to

    def _endRound(self, gameState: AbstractGameState) -> None:
        pass

    def _startRound(self, gameState: AbstractGameState) -> None:
        pass

    def _reset(self) -> None:
        self.direction = 1

    def nextPlayer(self, gameState: AbstractGameState) -> int:
        return (self.nPlayers + self.turnOwner + self.direction) % self.nPlayers

    def reverse(self) -> None:
        self.direction *= -1

    def __eq__(self, o: object) -> bool:
        if self is o:
            return True
        if not isinstance(o, AlternatingTurnOrder):
            return False
        if not super().__eq__(o):
            return False
        return self.direction == o.direction

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.direction))