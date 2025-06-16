from typing import Queue
from queue import Queue as SimpleQueue
from core.turnorders.TurnOrder import TurnOrder
from core import AbstractGameState
from core.CoreConstants import GameResult

class ReactiveTurnOrder(TurnOrder):
    def __init__(self, nPlayers: int = None, nMaxRounds: int = None):
        super().__init__(nPlayers, nMaxRounds)
        self.reactivePlayers: Queue[int] = SimpleQueue()

    def getCurrentPlayer(self, gameState: AbstractGameState) -> int:
        if self.reactivePlayers.qsize() > 0:
            return self.reactivePlayers.queue[0]
        return self.turnOwner

    def _copy(self) -> 'ReactiveTurnOrder':
        to = ReactiveTurnOrder()
        to.reactivePlayers = SimpleQueue()
        for player in list(self.reactivePlayers.queue):
            to.reactivePlayers.put(player)
        return to

    def _endRound(self, gameState: AbstractGameState) -> None:
        pass

    def _startRound(self, gameState: AbstractGameState) -> None:
        pass

    def _reset(self) -> None:
        self.reactivePlayers = SimpleQueue()

    def reactionsFinished(self) -> bool:
        return self.reactivePlayers.qsize() <= 0

    def getReactivePlayers(self) -> Queue[int]:
        return self.reactivePlayers

    def addReactivePlayer(self, player: int) -> None:
        self.reactivePlayers.put(player)

    def addCurrentPlayerReaction(self, gameState: AbstractGameState) -> None:
        self.reactivePlayers.put(self.getCurrentPlayer(gameState))

    def addAllReactivePlayers(self, gameState: AbstractGameState) -> None:
        for i in range(gameState.getNPlayers()):
            if gameState.getPlayerResults()[i] == GameResult.GAME_ONGOING:
                self.reactivePlayers.put(i)

    def addAllReactivePlayersButCurrent(self, gameState: AbstractGameState) -> None:
        currentPlayer = self.getCurrentPlayer(gameState)
        for i in range(gameState.getNPlayers()):
            if i != currentPlayer and gameState.getPlayerResults()[i] == GameResult.GAME_ONGOING:
                self.reactivePlayers.put(i)

    def __eq__(self, o: object) -> bool:
        if self is o:
            return True
        if not isinstance(o, ReactiveTurnOrder):
            return False
        if not super().__eq__(o):
            return False
        return list(self.reactivePlayers.queue) == list(o.reactivePlayers.queue)

    def __hash__(self) -> int:
        return hash((super().__hash__(), tuple(self.reactivePlayers.queue)))