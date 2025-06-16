from typing import Any
from games.terraformingmars.actions import TMAction
from games.terraformingmars import TMGameState
from abc import ABC, abstractmethod

class Effect(ABC):
    def __init__(self, must_be_current_player: bool, effect_action: TMAction):
        self.must_be_current_player = must_be_current_player  # if true, only applies when player is current player
        self.effect_action = effect_action
        # self.applies_to_player = False  # if true, applies to the player, otherwise can apply to any TODO
        # self.must_apply = False  # "up to X" type effects don't have to apply TODO

    def can_execute(self, game_state: TMGameState, action_taken: TMAction, player: int) -> bool:
        return not self.must_be_current_player or game_state.get_current_player() == player

    def execute(self, gs: TMGameState, action_taken: TMAction, player: int) -> None:
        if self.can_execute(gs, action_taken, player):
            self.effect_action.player = player
            self.effect_action.execute(gs)

    def __eq__(self, o: Any) -> bool:
        if self is o:
            return True
        if not isinstance(o, Effect):
            return False
        effect = o
        return (self.must_be_current_player == effect.must_be_current_player and 
                self.effect_action == effect.effect_action)

    def __hash__(self) -> int:
        return hash((self.must_be_current_player, self.effect_action))

    @abstractmethod
    def copy(self) -> 'Effect':
        pass

    @abstractmethod
    def copy_serializable(self) -> 'Effect':
        pass