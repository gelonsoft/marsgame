# OneShotExtendedAction.py
from core.actions import IExtendedSequence
from typing import List, Callable
from core import AbstractGameState, AbstractAction

class OneShotExtendedAction(IExtendedSequence):
    def __init__(self, name: str, player: int, compute_available_actions: Callable[[AbstractGameState], List[AbstractAction]]):
        self.action_function = compute_available_actions
        self.player = player
        self.name = name
        self.executed = False

    def _compute_available_actions(self, state: AbstractGameState) -> List[AbstractAction]:
        return self.action_function(state)

    def get_current_player(self, state: AbstractGameState) -> int:
        return self.player

    def _after_action(self, state: AbstractGameState, action: AbstractAction):
        self.executed = True

    def execution_complete(self, state: AbstractGameState) -> bool:
        return self.executed

    def copy(self):
        ret_value = OneShotExtendedAction(self.name, self.player, self.action_function)
        ret_value.executed = self.executed
        return ret_value

    def __hash__(self):
        return hash((self.player, self.executed, self.name))

    def __eq__(self, other):
        if isinstance(other, OneShotExtendedAction):
            return (self.player == other.player and 
                    self.executed == other.executed and 
                    self.name == other.name)
        return False

    def __str__(self):
        return self.name