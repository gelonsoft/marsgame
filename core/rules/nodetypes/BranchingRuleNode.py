from abc import abstractmethod
from typing import Optional, List
from core import AbstractGameStateWithTurnOrder
from core.rules import Node
from .RuleNode import RuleNode

class BranchingRuleNode(RuleNode):
    def __init__(self):
        super().__init__()
        self.children: List[Optional[Node]] = []

    def __init__(self, action_node: bool):
        super().__init__(action_node)

    def __init__(self, node: 'BranchingRuleNode'):
        super().__init__(node)

    @abstractmethod
    def run(self, gs: AbstractGameStateWithTurnOrder) -> bool:
        pass

    def set_next(self, children: List[Node]) -> None:
        self.children = children

    def get_children(self) -> List[Optional[Node]]:
        return self.children