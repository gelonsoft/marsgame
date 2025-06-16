from core import AbstractGameStateWithTurnOrder
from core.rules.nodetypes import RuleNode


class PlayerAction(RuleNode):
    def __init__(self):
        super().__init__(True)

    def __init__(self, player_action: 'PlayerAction'):
        super().__init__(player_action)

    def run(self, gs: AbstractGameStateWithTurnOrder) -> bool:
        if self.action is not None:
            self.action.execute(gs)
            return True
        return False

    def _copy(self) -> 'PlayerAction':
        return PlayerAction(self)