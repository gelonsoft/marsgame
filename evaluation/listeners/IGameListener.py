from abc import ABC, abstractmethod
import os
from typing import List, Set

class IGameListener(ABC):
    
    @abstractmethod
    def onEvent(self, event):
        """Manages all events."""
        pass
    
    @abstractmethod
    def report(self):
        """Called when all processing is finished."""
        pass
    
    def setOutputDirectory(self, *nestedDirectories) -> bool:
        return True
    
    @abstractmethod
    def setGame(self, game):
        pass
    
    @abstractmethod
    def getGame(self):
        pass
    
    @staticmethod
    def createListener(listenerName: str):
        if not listenerName:
            raise ValueError("A listenerName must be specified")
        
        # In Python, we'd typically use a different mechanism for dynamic class loading
        # This is a simplified version - actual implementation would depend on your project
        raise NotImplementedError("Dynamic listener creation not implemented in Python version")
    
    def reset(self):
        pass
    
    def init(self, game, nPlayersPerGame: int, playerNames: Set[str]):
        pass