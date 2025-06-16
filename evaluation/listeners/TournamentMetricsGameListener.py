from typing import Set

from evaluation.listeners.MetricsGameListener import MetricsGameListener
from evaluation.metrics import TournamentMetric
from players import AbstractPlayer

class TournamentMetricsGameListener(MetricsGameListener):
    def __init__(self, metrics=None, logTo="ToBoth", dataTypes=None):
        if dataTypes is None:
            dataTypes = ["RawData", "Summary", "Plot"]
        
        if metrics is not None:
            tournament_metrics = [TournamentMetric(m) for m in metrics]
            super().__init__(metrics=tournament_metrics)
            self.reportDestinations = [logTo]
            self.reportTypes = dataTypes
    
    def tournamentInit(self, game, nPlayersPerGame: int, playerNames: Set[str], matchup: Set[AbstractPlayer]):
        for metric in self.metrics.values():
            if isinstance(metric, TournamentMetric):
                metric.tournamentInit(game, nPlayersPerGame, playerNames, matchup)