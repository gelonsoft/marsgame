from core import AbstractForwardModel, AbstractGameState
from core.actions import AbstractAction
from core.interfaces import IPlayerDecorator
from evaluation.metrics import Event
from players import PlayerParameters
from typing import List, Dict, Optional, Any
import random
import abc

class AbstractPlayer(abc.ABC):
    
    def __init__(self, params: Optional[PlayerParameters], name: str) -> None:
        self.player_id: int = -1  # Will be assigned by the game
        self.name: str = name
        self.rnd: random.Random = random.Random()
        self._forward_model: Optional[AbstractForwardModel] = None
        self.parameters: PlayerParameters = params if params is not None else PlayerParameters()
        self.decorators: List[IPlayerDecorator] = []
        if self.parameters.decorator is not None:
            self.decorators.append(self.parameters.decorator)

    # Final methods

    @property
    def playerID(self) -> int:
        return self.player_id

    def setName(self, name: str) -> None:
        self.name = name

    def getAction(self, gameState: AbstractGameState, observedActions: List[AbstractAction]) -> AbstractAction:
        for decorator in self.decorators:
            observedActions = decorator.actionFilter(gameState, observedActions)
        
        if not observedActions:
            raise AssertionError(f"No actions available for player {self}")
        elif len(observedActions) == 1:
            action = observedActions[0]
        else:
            gameState.rnd = self.rnd
            action = self._getAction(gameState, observedActions)
        
        for decorator in self.decorators:
            decorator.recordDecision(gameState, action)
        
        return action

    def setForwardModel(self, model: AbstractForwardModel) -> None:
        self._forward_model = model
        for decorator in self.decorators:
            model.addPlayerDecorator(decorator)

    @property
    def forwardModel(self) -> AbstractForwardModel:
        return self._forward_model

    def addDecorator(self, decorator: IPlayerDecorator) -> None:
        self.decorators.append(decorator)

    def clearDecorators(self) -> None:
        self.decorators.clear()

    def removeDecorator(self, decorator: IPlayerDecorator) -> None:
        self.decorators.remove(decorator)

    def __str__(self) -> str:
        return self.name if self.name is not None else self.__class__.__name__

    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    def _getAction(self, gameState: AbstractGameState, possibleActions: List[AbstractAction]) -> AbstractAction:
        pass

    # Optional methods that can be overridden by subclasses

    def initializePlayer(self, gameState: AbstractGameState) -> None:
        pass

    def finalizePlayer(self, gameState: AbstractGameState) -> None:
        pass

    def registerUpdatedObservation(self, gameState: AbstractGameState) -> None:
        pass

    def onEvent(self, event: Event) -> None:
        pass

    @abc.abstractmethod
    def copy(self) -> 'AbstractPlayer':
        pass

    def getDecisionStats(self) -> Dict[AbstractAction, Dict[str, Any]]:
        return {}

    @property
    def getParameters(self) -> PlayerParameters:
        return self.parameters

    @property
    def getRnd(self) -> random.Random:
        return self.rnd