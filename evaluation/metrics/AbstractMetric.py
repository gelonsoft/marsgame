from abc import ABC, abstractmethod
from typing import Dict, Optional, Set, List, Type, Any
from core import Game
from evaluation.listeners import MetricsGameListener
from evaluation.metrics import IDataLogger, IDataProcessor, Event

class AbstractMetric(ABC):
    def __init__(self, *args):
        self.dataLogger: Optional[IDataLogger] = None
        self.eventTypes: Set[Event.IGameEvent] = self.getDefaultEventTypes()
        self.args = args
        self.columnNames: Set[str] = set()
        self.gamesCompleted: int = 0

    @abstractmethod
    def _run(self, listener: 'MetricsGameListener', e: Event, records: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def getDefaultEventTypes(self) -> Set[Event.IGameEvent]:
        pass

    def reset(self) -> None:
        self.gamesCompleted = 0
        self.columnNames.clear()
        if self.dataLogger:
            self.dataLogger.reset()

    def init(self, game: 'Game', nPlayers: int, playerNames: Set[str]) -> None:
        if self.dataLogger:
            self.dataLogger.init(game, nPlayers, playerNames)

    def run(self, listener: 'MetricsGameListener', e: Event) -> None:
        records = {name: None for name in self.columnNames}
        
        if self._run(listener, e, records):
            self.addDefaultData(e)
            for key, value in records.items():
                if self.dataLogger:
                    self.dataLogger.addData(key, value)

    @abstractmethod
    def getColumns(self, nPlayersPerGame: int, playerNames: Set[str]) -> Dict[str, Type]:
        pass

    def getDefaultColumns(self) -> Dict[str, Type]:
        return {
            "GameID": str,
            "GameName": str,
            "PlayerCount": str,
            "GameSeed": str,
            "Tick": int,
            "Turn": int,
            "Round": int,
            "Event": str
        }

    def addDefaultData(self, e: Event) -> None:
        if self.dataLogger:
            self.dataLogger.addData("GameID", str(e.state.getGameID()))
            self.dataLogger.addData("GameName", e.state.getGameType().name())
            self.dataLogger.addData("PlayerCount", str(e.state.getNPlayers()))
            self.dataLogger.addData("GameSeed", str(e.state.getGameParameters().getRandomSeed()))
            self.dataLogger.addData("Tick", e.state.getGameTick())
            self.dataLogger.addData("Turn", e.state.getTurnCounter())
            self.dataLogger.addData("Round", e.state.getRoundCounter())
            self.dataLogger.addData("Event", e.type.name())

    def getEventTypes(self) -> Set[Event.IGameEvent]:
        return self.eventTypes

    def getName(self) -> str:
        return self.__class__.__name__

    def listens(self, eventType: Event.IGameEvent) -> bool:
        return eventType in self.eventTypes if self.eventTypes else True

    def filterByEventTypeWhenReporting(self) -> bool:
        return True

    def report(self, folderName: str, 
               reportTypes: List[IDataLogger.ReportType],
               reportDestinations: List[IDataLogger.ReportDestination],
               append: bool) -> None:
        
        dataProcessor = self.getDataProcessor()
        assert isinstance(dataProcessor, type(self.dataLogger.getDefaultProcessor())), \
            f"Data Processor {dataProcessor.__class__.__name__} is not compatible with Data Logger {self.dataLogger.__class__.__name__}"

        for i, reportType in enumerate(reportTypes):
            reportDestination = reportDestinations[0] if len(reportDestinations) == 1 else reportDestinations[i]
            
            to_file = reportDestination in [IDataLogger.ReportDestination.ToFile, IDataLogger.ReportDestination.ToBoth]
            to_console = reportDestination in [IDataLogger.ReportDestination.ToConsole, IDataLogger.ReportDestination.ToBoth]

            if reportType == IDataLogger.ReportType.RawData:
                if to_file:
                    dataProcessor.processRawDataToFile(self.dataLogger, folderName, append)
                if to_console:
                    dataProcessor.processRawDataToConsole(self.dataLogger)
            elif reportType == IDataLogger.ReportType.Summary:
                if to_file:
                    dataProcessor.processSummaryToFile(self.dataLogger, folderName)
                if to_console:
                    dataProcessor.processSummaryToConsole(self.dataLogger)
            elif reportType == IDataLogger.ReportType.Plot:
                if to_file:
                    dataProcessor.processPlotToFile(self.dataLogger, folderName)
                if to_console:
                    dataProcessor.processPlotToConsole(self.dataLogger)

    def getDataProcessor(self) -> IDataProcessor:
        return self.dataLogger.getDefaultProcessor()

    def addColumnName(self, name: str) -> None:
        self.columnNames.add(name)

    def getColumnNames(self) -> Set[str]:
        return self.columnNames

    def notifyGameOver(self) -> None:
        self.gamesCompleted += 1

    def getGamesCompleted(self) -> int:
        return self.gamesCompleted

    def setDataLogger(self, logger: IDataLogger) -> None:
        self.dataLogger = logger

    def getDataLogger(self) -> IDataLogger:
        return self.dataLogger

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AbstractMetric):
            return False
        return (self.eventTypes == other.eventTypes and 
                self.args == other.args and 
                self.columnNames == other.columnNames)

    def __hash__(self) -> int:
        return hash((frozenset(self.eventTypes), tuple(self.args), frozenset(self.columnNames)))