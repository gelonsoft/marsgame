from enum import Enum
from typing import Any, Set, Optional
from core import Game
from evaluation.metrics import IDataProcessor

class ReportType(Enum):
    RawData = 1
    Summary = 2
    Plot = 3
    RawDataPerEvent = 4

class ReportDestination(Enum):
    ToFile = 1
    ToConsole = 2
    ToBoth = 3

class IDataLogger:
    def reset(self) -> None:
        pass

    def init(self, game: Game, nPlayersPerGame: int, playerNames: Set[str]) -> None:
        pass

    def addData(self, columnName: str, data: Any) -> None:
        pass

    def getDefaultProcessor(self) -> 'IDataProcessor':
        pass

    def flush(self) -> None:
        pass

    def copy(self) -> 'IDataLogger':
        pass

    def emptyCopy(self) -> 'IDataLogger':
        pass

    def create(self) -> 'IDataLogger':
        pass