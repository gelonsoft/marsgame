from abc import abstractmethod
from typing import List, Optional
from core import AbstractGameStateWithTurnOrder
from core.CoreConstants import GameResult
from core.rules import GameOverCondition, Node


class RuleNode(Node):
    def __init__(self):
        super().__init__()
        self.child_next: Optional[Node] = None
        self.game_over_conditions: List[GameOverCondition] = []

    def __init__(self, action_node: bool):
        super().__init__()
        self.action_node = action_node
        self.game_over_conditions = []

    def __init__(self, node: 'RuleNode'):
        super().__init__(node)
        self.game_over_conditions = node.game_over_conditions

    @abstractmethod
    def run(self, gs: AbstractGameStateWithTurnOrder) -> bool:
        pass

    def add_game_over_condition(self, condition: GameOverCondition) -> None:
        self.game_over_conditions.append(condition)

    def execute(self, gs: AbstractGameStateWithTurnOrder) -> Optional[Node]:
        if self.require_action() and self.action is None:
            return None

        interrupted = not self.run(gs)
        
        if self.game_over_conditions:
            for goc in self.game_over_conditions:
                result = goc.test(gs)
                if result != GameResult.GAME_ONGOING:
                    gs.set_game_status(result)
        
        return None if interrupted else self.child_next

    def set_next(self, child_next: Node) -> None:
        self.child_next = child_next

    def get_next(self) -> Optional[Node]:
        return self.child_next

    def get_game_over_conditions(self) -> List[GameOverCondition]:
        return self.game_over_conditions