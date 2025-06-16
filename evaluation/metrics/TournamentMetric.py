from typing import Any, Set, Dict, List
from core import AbstractPlayer, Game
from evaluation.listeners import MetricsGameListener
from evaluation.metrics import AbstractMetric, Event, IDataLogger

class TournamentMetric(AbstractMetric):
    def __init__(self, metric: AbstractMetric):
        super().__init__(metric.getEventTypes())
        self.wrappedMetric = metric
        self.dataLoggers: Dict[Set[AbstractPlayer], IDataLogger] = {}
        self.firstReport = True

    def _run(self, listener: 'MetricsGameListener', e: Event, records: Dict[str, Any]) -> bool:
        return self.wrappedMetric._run(listener, e, records)

    def getDefaultEventTypes(self) -> Set[Event.IGameEvent]:
        return self.wrappedMetric.getDefaultEventTypes()

    def reset(self) -> None:
        super().reset()
        for logger in self.dataLoggers.values():
            logger.reset()

    def init(self, game: Game, nPlayers: int, playerNames: Set[str]) -> None:
        pass  # Initialized specially in tournamentInit

    def getColumns(self, nPlayersPerGame: int, playerNames: Set[str]) -> Dict[str, Any]:
        return self.wrappedMetric.getColumns(nPlayersPerGame, playerNames)

    def tournamentInit(self, game: Game, nPlayers: int, playerNames: Set[str], 
                      matchup: Set[AbstractPlayer]) -> None:
        matchup_set = set(matchup)
        logger = next((dl for key, dl in self.dataLoggers.items() 
                      if set(key) == matchup_set), None)
        
        if logger:
            self.setDataLogger(logger)
        else:
            new_logger = self.dataLogger.create()
            self.dataLoggers[matchup_set] = new_logger
            new_logger.init(game, nPlayers, playerNames)
            self.setDataLogger(new_logger)

    def report(self, folderName: str, 
              reportTypes: List[IDataLogger.ReportType],
              reportDestinations: List[IDataLogger.ReportDestination]) -> None:
        
        dataProcessor = self.getDataProcessor()
        assert isinstance(dataProcessor, type(self.dataLogger.getDefaultProcessor())), \
            "Incompatible Data Processor and Logger"

        for matchup, logger in self.dataLoggers.items():
            matchup_folder = f"{folderName}/{'_'.join(str(p) for p in matchup)}"
            import os
            os.makedirs(matchup_folder, exist_ok=True)

            for i, reportType in enumerate(reportTypes):
                destination = (reportDestinations[0] if len(reportDestinations) == 1 
                              else reportDestinations[i])
                
                to_file = destination in [IDataLogger.ReportDestination.ToFile, 
                                        IDataLogger.ReportDestination.ToBoth]
                to_console = destination in [IDataLogger.ReportDestination.ToConsole, 
                                           IDataLogger.ReportDestination.ToBoth]

                if reportType == IDataLogger.ReportType.RawData:
                    if to_file:
                        dataProcessor.processRawDataToFile(logger, matchup_folder, not self.firstReport)
                    if to_console:
                        dataProcessor.processRawDataToConsole(logger)
                elif reportType == IDataLogger.ReportType.Summary:
                    if to_file:
                        dataProcessor.processSummaryToFile(logger, matchup_folder)
                    if to_console:
                        dataProcessor.processSummaryToConsole(logger)
                elif reportType == IDataLogger.ReportType.Plot:
                    if to_file:
                        dataProcessor.processPlotToFile(logger, matchup_folder)
                    if to_console:
                        dataProcessor.processPlotToConsole(logger)

        self.firstReport = False