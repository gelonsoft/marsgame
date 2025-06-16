from core.turnorders.TurnOrder import TurnOrder
from core import AbstractGameState

class StandardTurnOrder(TurnOrder):
    def __init__(self, nPlayers: int = None, nMaxRounds: int = None):
        super().__init__(nPlayers, nMaxRounds)

    def _reset(self) -> None:
        pass

    def _copy(self) -> 'StandardTurnOrder':
        return StandardTurnOrder()

    def _endRound(self, gameState: AbstractGameState) -> None:
        pass

    def _startRound(self, gameState: AbstractGameState) -> None:
        pass