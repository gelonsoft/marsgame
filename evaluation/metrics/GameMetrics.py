from typing import Dict, Set, List, Type, Any
from core import AbstractGameState
from core.interfaces import IComponentContainer
from evaluation.listeners import MetricsGameListener
from evaluation.metrics import AbstractMetric, Event, IMetricsCollection

class GameMetrics(IMetricsCollection):
    class GameScore(AbstractMetric):
        def _run(self, listener: 'MetricsGameListener', e: Event, records: Dict[str, Any]) -> bool:
            sum_scores = 0.0
            leader_id = -1
            second_id = -1
            
            for i in range(e.state.getNPlayers()):
                score = e.state.getGameScore(i)
                sum_scores += score
                records[f"Player-{i}"] = score
                records[f"PlayerName-{i}"] = str(listener.getGame().getPlayers()[i])
                
                if e.state.getOrdinalPosition(i) == 1:
                    leader_id = i
                if e.state.getNPlayers() > 1 and e.state.getOrdinalPosition(i) == 2:
                    second_id = i
            
            records["Average"] = sum_scores / e.state.getNPlayers()
            records["LeaderGap"] = (e.state.getGameScore(leader_id) - e.state.getGameScore(second_id)) if second_id != -1 else 0.0
            return True

        def getDefaultEventTypes(self) -> Set[Event.IGameEvent]:
            return {Event.GameEvent.ACTION_CHOSEN, Event.GameEvent.ROUND_OVER, Event.GameEvent.GAME_OVER}

        def getColumns(self, nPlayersPerGame: int, playerNames: Set[str]) -> Dict[str, Type]:
            columns = {f"Player-{i}": float for i in range(nPlayersPerGame)}
            columns.update({f"PlayerName-{i}": str for i in range(nPlayersPerGame)})
            columns.update({"Average": float, "LeaderGap": float})
            return columns

    # Similar implementations for other metrics (FinalScore, RoundCounter, etc.)
    # Would follow the same pattern as above
    
    @staticmethod
    def countComponents(state: AbstractGameState) -> tuple[int, List[int]]:
        hidden_by_player = [0] * state.getNPlayers()
        total = sum(1 for c in state.getAllComponents() if not isinstance(c, IComponentContainer))
        for p in range(len(hidden_by_player)):
            hidden_by_player[p] = len(state.getUnknownComponentsIds(p))
        return (total, hidden_by_player)