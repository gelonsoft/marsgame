from enum import Enum
from typing import Optional, Set
from core import AbstractGameState

class IGameEvent:
    def isOncePerGame(self) -> bool:
        pass

    def getValues(self) -> Set['IGameEvent']:
        pass

class GameEvent(Enum, IGameEvent):
    ABOUT_TO_START = (True,)
    GAME_OVER = (True,)
    ROUND_OVER = (False,)
    TURN_OVER = (False,)
    ACTION_CHOSEN = (False,)
    ACTION_TAKEN = (False,)
    GAME_EVENT = (False,)

    def __init__(self, oncePerGame: bool):
        self.oncePerGame = oncePerGame

    def isOncePerGame(self) -> bool:
        return self.oncePerGame

    def getValues(self) -> Set['IGameEvent']:
        return set(GameEvent)

class Event:
    def __init__(self):
        self.type: Optional['IGameEvent'] = None
        self.state: Optional['AbstractGameState'] = None
        self.action: Optional['AbstractAction'] = None
        self.playerID: int = -1

    @staticmethod
    def createEvent(type: IGameEvent, 
                   gameState: Optional['AbstractGameState'] = None,
                   action: Optional['AbstractAction'] = None,
                   playerID: int = -1) -> 'Event':
        e = Event()
        e.type = type
        e.state = gameState
        e.action = action
        e.playerID = playerID
        return e