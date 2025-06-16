from enum import Enum
from typing import List, Callable, Optional
from core import AbstractGameState, CoreConstants
from evaluation.listeners import IGameListener
from evaluation.metrics import Event
from core.actions import LogEvent

class GameResult(Enum):
    DRAW_GAME = "DRAW_GAME"
    GAME_ONGOING = "GAME_ONGOING"
    LOSE_GAME = "LOSE_GAME"
    WIN_GAME = "WIN_GAME"

class TurnOrder:
    def __init__(self, nPlayers: int = None, nMaxRounds: int = None):
        self.reset()
        if nPlayers is not None:
            self.nPlayers = nPlayers
        if nMaxRounds is not None:
            self.nMaxRounds = nMaxRounds
        self.listeners: List[IGameListener] = []

    def setStartingPlayer(self, player: int) -> None:
        self.firstPlayer = player
        self.turnOwner = player

    def getTurnOwner(self) -> int:
        return self.turnOwner

    def nPlayers(self) -> int:
        return self.nPlayers

    def getRoundCounter(self) -> int:
        return self.roundCounter

    def getTurnCounter(self) -> int:
        return self.turnCounter

    # Abstract methods
    def _reset(self) -> None:
        raise NotImplementedError

    def _copy(self) -> 'TurnOrder':
        raise NotImplementedError

    def copyTo(self, turnOrder: 'TurnOrder') -> 'TurnOrder':
        turnOrder.turnOwner = self.turnOwner
        turnOrder.turnCounter = self.turnCounter
        turnOrder.roundCounter = self.roundCounter
        turnOrder.firstPlayer = self.firstPlayer
        turnOrder.nMaxRounds = self.nMaxRounds
        turnOrder.nPlayers = self.nPlayers
        return turnOrder

    def reset(self) -> None:
        self._reset()
        self.firstPlayer = 0
        self.turnOwner = 0
        self.turnCounter = 0
        self.roundCounter = 0

    def copy(self) -> 'TurnOrder':
        to = self._copy()
        return self.copyTo(to)

    def endPlayerTurn(self, gameState: AbstractGameState) -> None:
        if gameState.getGameStatus() != GameResult.GAME_ONGOING:
            return

        gameState.getPlayerTimer()[self.getCurrentPlayer(gameState)].incrementTurn()

        for listener in self.listeners:
            listener.onEvent(Event.createEvent(Event.GameEvent.TURN_OVER, gameState, self.getCurrentPlayer(gameState)))

        self.turnCounter += 1
        if self.turnCounter >= self.nPlayers:
            self.endRound(gameState)
        else:
            self.moveToNextPlayer(gameState, self.nextPlayer(gameState))

    def logEvent(self, eventText: Callable[[], str] | str, state: AbstractGameState) -> None:
        if isinstance(eventText, Callable):
            if not self.listeners and not state.getCoreGameParameters().recordEventHistory:
                return
            text = eventText()
        else:
            text = eventText

        logAction = LogEvent(text)
        for listener in self.listeners:
            listener.onEvent(Event.createEvent(Event.GameEvent.GAME_EVENT, state, logAction))
        if state.getCoreGameParameters().recordEventHistory:
            state.recordHistory(text)

    def endRound(self, gameState: AbstractGameState) -> None:
        self._endRound(gameState)
        if gameState.getGameStatus() != GameResult.GAME_ONGOING:
            return

        gameState.getPlayerTimer()[self.getCurrentPlayer(gameState)].incrementRound()

        for listener in self.listeners:
            listener.onEvent(Event.createEvent(Event.GameEvent.ROUND_OVER, gameState, self.getCurrentPlayer(gameState)))
        if gameState.getCoreGameParameters().recordEventHistory:
            gameState.recordHistory(Event.GameEvent.ROUND_OVER.name())

        self.roundCounter += 1
        if self.nMaxRounds != -1 and self.roundCounter == self.nMaxRounds:
            self.endGame(gameState)
        else:
            self.turnCounter = 0
            self.moveToNextPlayer(gameState, self.firstPlayer)
            self._startRound(gameState)

    def _endRound(self, gameState: AbstractGameState) -> None:
        raise NotImplementedError

    def _startRound(self, gameState: AbstractGameState) -> None:
        raise NotImplementedError

    def getCurrentPlayer(self, gameState: AbstractGameState) -> int:
        if gameState.isActionInProgress():
            return gameState.currentActionInProgress().getCurrentPlayer(gameState)
        return self.turnOwner

    def setTurnOwner(self, owner: int) -> None:
        self.turnOwner = owner

    def nextPlayer(self, gameState: AbstractGameState) -> int:
        return (self.nPlayers + self.turnOwner + 1) % self.nPlayers

    def moveToNextPlayer(self, gameState: AbstractGameState, newTurnOwner: int) -> None:
        self.turnOwner = newTurnOwner
        n = 0
        while gameState.getPlayerResults()[self.turnOwner] != GameResult.GAME_ONGOING:
            self.turnOwner = self.nextPlayer(gameState)
            n += 1
            if n >= self.nPlayers:
                self.endGame(gameState)
                break

    def endGame(self, gs: AbstractGameState) -> None:
        gs.setGameStatus(CoreConstants.GameResult.GAME_END)
        drawn = sum(1 for p in range(gs.getNPlayers()) if gs.getOrdinalPosition(p) == 1) > 1
        for p in range(gs.getNPlayers()):
            o = gs.getOrdinalPosition(p)
            if o == 1 and drawn:
                gs.setPlayerResult(GameResult.DRAW_GAME, p)
            elif o == 1:
                gs.setPlayerResult(GameResult.WIN_GAME, p)
            else:
                gs.setPlayerResult(GameResult.LOSE_GAME, p)
        if gs.getCoreGameParameters().verbose:
            print(gs.getPlayerResults())

    def __eq__(self, o: object) -> bool:
        if self is o:
            return True
        if not isinstance(o, TurnOrder):
            return False
        return (self.nPlayers == o.nPlayers and
                self.turnOwner == o.turnOwner and
                self.roundCounter == o.roundCounter and
                self.firstPlayer == o.firstPlayer and
                self.turnCounter == o.turnCounter and
                self.nMaxRounds == o.nMaxRounds)

    def __hash__(self) -> int:
        return hash((self.nPlayers, self.turnOwner, self.turnCounter, 
                    self.roundCounter, self.firstPlayer, self.nMaxRounds))

    def addListener(self, listener: IGameListener) -> None:
        if listener not in self.listeners:
            self.listeners.append(listener)

    def clearListeners(self) -> None:
        self.listeners.clear()

    def getFirstPlayer(self) -> int:
        return self.firstPlayer