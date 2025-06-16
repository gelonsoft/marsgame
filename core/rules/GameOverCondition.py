from abc import ABC, abstractmethod
from core import AbstractGameState
from core.CoreConstants import GameResult


class GameOverCondition(ABC):
    @abstractmethod
    def test(self, gs: AbstractGameState) -> GameResult:
        pass