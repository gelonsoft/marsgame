from typing import Any
from core.components import Counter
from games.terraformingmars.actions import TMAction
from games.terraformingmars import TMTypes
from games.terraformingmars import TMGameState

class Bonus:
    def __init__(self, p: TMTypes.GlobalParameter, threshold: int, effect: TMAction):
        self.param = p
        self.threshold = threshold
        self.effect = effect
        self.claimed = -1  # Can only execute once

    def get_effect(self) -> TMAction:
        return self.effect

    def check_bonus(self, gs: TMGameState) -> None:
        if self.claimed == -1:
            c = gs.get_global_parameters().get(self.param)
            if c.get_value_idx() >= self.threshold - 1:  # -1 because this is checked right before the increase
                self.effect.player = gs.get_current_player()
                self.claimed = self.effect.player
                self.effect.execute(gs)

    def copy(self) -> 'Bonus':
        b = Bonus(self.param, self.threshold, self.effect.copy())
        b.claimed = self.claimed
        return b

    def __eq__(self, o: Any) -> bool:
        if self is o:
            return True
        if not isinstance(o, Bonus):
            return False
        bonus = o
        return (self.threshold == bonus.threshold and 
                self.claimed == bonus.claimed and 
                self.param == bonus.param and 
                self.effect == bonus.effect)

    def __hash__(self) -> int:
        return hash((self.threshold, self.param, self.effect, self.claimed))