from abc import ABC, abstractmethod
from typing import Optional
from core.actions import AbstractAction
from core import AbstractGameStateWithTurnOrder

class Node(ABC):
    _next_id = 0

    def __init__(self):
        self.id = Node._next_id
        Node._next_id += 1
        self.action_node = False
        self.next_player_node = False
        self.action: Optional[AbstractAction] = None
        self.parent: Optional['Node'] = None

    def __init__(self, node: 'Node'):
        self.id = node.id
        self.action_node = node.action_node
        self.next_player_node = node.next_player_node
        self.action = node.action

    def copy(self) -> 'Node':
        return self._copy()

    @abstractmethod
    def _copy(self) -> 'Node':
        pass

    @abstractmethod
    def execute(self, gs: AbstractGameStateWithTurnOrder) -> Optional['Node']:
        pass

    @abstractmethod
    def get_next(self) -> Optional['Node']:
        pass

    def set_action(self, action: AbstractAction) -> None:
        self.action = action

    def require_action(self) -> bool:
        return self.action_node

    def set_next_player_node(self) -> None:
        self.next_player_node = True

    def is_next_player_node(self) -> bool:
        return self.next_player_node

    def get_id(self) -> int:
        return self.id

    def get_parent(self) -> Optional['Node']:
        return self.parent

    def set_parent(self, parent: 'Node') -> None:
        self.parent = parent

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return self.__class__.__name__