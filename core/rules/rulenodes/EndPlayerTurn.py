from core import AbstractGameStateWithTurnOrder
from core.CoreConstants import DefaultGamePhase
from core.rules.nodetypes import RuleNode


class EndPlayerTurn(RuleNode):
    def __init__(self):
        super().__init__()
        self.set_next_player_node()

    def __init__(self, end_player_turn: 'EndPlayerTurn'):
        super().__init__(end_player_turn)

    def run(self, gs: AbstractGameStateWithTurnOrder) -> bool:
        gs.get_turn_order().end_player_turn(gs)
        gs.set_game_phase(DefaultGamePhase.Main)
        return True

    def _copy(self) -> 'EndPlayerTurn':
        return EndPlayerTurn(self)