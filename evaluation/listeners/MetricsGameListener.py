import os
from typing import Set
from collections import defaultdict

from evaluation.listeners.IGameListener import IGameListener

class MetricsGameListener(IGameListener):
    def __init__(self, metrics=None):
        self.metrics = {}
        self.eventsOfInterest = set()
        self.game = None
        self.reportTypes = []
        self.reportDestinations = []
        self.destDir = "metrics/out/"
        self.firstReport = True
        
        if metrics is not None:
            self._initialize_with_metrics(metrics)
    
    def _initialize_with_metrics(self, metrics):
        self.reportDestinations = ["ToConsole"]
        self.reportTypes = ["Summary", "Plot"]
        self.metrics = {m.getName(): m for m in metrics}
        self.firstReport = True
        
        for m in self.metrics.values():
            #m.setDataLogger(DataTableSaw(m))  # Assuming DataTableSaw is available
            self.eventsOfInterest.update(m.getEventTypes())
        
        self.eventsOfInterest.add("GAME_OVER")
    
    def onEvent(self, event):
        if event.type not in self.eventsOfInterest:
            return
        
        for metric in self.metrics.values():
            if metric.listens(event.type):
                metric.run(self, event)
            
            if event.type == "GAME_OVER":
                metric.notifyGameOver()
    
    def setOutputDirectory(self, *nestedDirectories) -> bool:
        success = True
        
        if "ToFile" in self.reportDestinations or "ToBoth" in self.reportDestinations:
            # Create directory logic would go here
            self.destDir = os.path.join(*nestedDirectories) + os.sep
        
        return success
    
    def report(self):
        success = True
        
        if "ToFile" in self.reportDestinations or "ToBoth" in self.reportDestinations:
            if not os.path.exists(self.destDir):
                os.makedirs(self.destDir)
        
        if not (len(self.reportTypes) == 1 and "RawDataPerEvent" in self.reportTypes):
            for metric in self.metrics.values():
                metric.report(self.destDir, self.reportTypes, self.reportDestinations, not self.firstReport)
        
        if "RawDataPerEvent" in self.reportTypes:
            event_metrics = defaultdict(list)
            for event in self.eventsOfInterest:
                for metric in self.metrics.values():
                    if metric.listens(event):
                        event_metrics[event].append(metric)
            
            for event, metrics in event_metrics.items():
                if metrics:
                    pass
                    #TODO: ZAZA
                    #data_logger = DataTableSaw(metrics, event, self._eventToIndexingColumn(event))
                    #data_logger.getDefaultProcessor().processRawDataToFile(data_logger, self.destDir, not self.firstReport)
            
            for metric in self.metrics.values():
                metric.getDataLogger().flush()
            
            self.firstReport = False
    
    def _eventToIndexingColumn(self, event):
        if event in ["ABOUT_TO_START", "GAME_OVER"]:
            return "GameID"
        elif event == "ROUND_OVER":
            return "Round"
        elif event in ["TURN_OVER", "ACTION_CHOSEN", "ACTION_TAKEN", "GAME_EVENT"]:
            return "Tick"
        return None
    
    def setGame(self, game):
        self.game = game
    
    def getGame(self):
        return self.game
    
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
    
    def init(self, game, nPlayersPerGame: int, playerNames: Set[str]):
        self.game = game
        for metric in self.metrics.values():
            metric.init(game, nPlayersPerGame, playerNames)