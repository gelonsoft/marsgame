from abc import abstractmethod
from typing import Optional
from core import AbstractGameState, AbstractGameStateWithTurnOrder
from core.rules import Node


class ConditionNode(Node):
    def __init__(self):
        super().__init__()
        self.child_yes: Optional[Node] = None
        self.child_no: Optional[Node] = None
        self.passed = False

    def __init__(self, node: 'ConditionNode'):
        super().__init__(node)
        self.child_yes = node.child_yes
        self.child_no = node.child_no
        self.passed = node.passed

    @abstractmethod
    def test(self, gs: AbstractGameState) -> bool:
        pass

    def execute(self, gs: AbstractGameStateWithTurnOrder) -> Optional[Node]:
        self.passed = self.test(gs)
        return self.get_next()

    def get_next(self) -> Optional[Node]:
        return self.child_yes if self.passed else self.child_no

    def set_yes_no(self, child_yes: Node, child_no: Node) -> None:
        self.child_yes = child_yes
        self.child_no = child_no

    def get_yes_no(self) -> list[Optional[Node]]:
        return [self.child_yes, self.child_no]