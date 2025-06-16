from core import AbstractGameStateWithTurnOrder

from core.CoreConstants import DefaultGamePhase
from core.rules.nodetypes import RuleNode
from core.turnorders import ReactiveTurnOrder


class ForceAllPlayerReaction(RuleNode):
    def __init__(self):
        super().__init__()

    def __init__(self, force_all_player_reaction: 'ForceAllPlayerReaction'):
        super().__init__(force_all_player_reaction)

    def run(self, gs: AbstractGameStateWithTurnOrder) -> bool:
        gs.get_turn_order().add_all_reactive_players(gs)
        gs.set_game_phase(DefaultGamePhase.PlayerReaction)
        return False

    def _copy(self) -> 'ForceAllPlayerReaction':
        return ForceAllPlayerReaction(self)