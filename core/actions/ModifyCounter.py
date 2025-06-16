# ModifyCounter.py
from core.actions import AbstractAction
from core.components import Counter
from core import AbstractGameState

class ModifyCounter(AbstractAction):
    def __init__(self, counter_id, change):
        self.counter_id = counter_id
        self.change = change

    def execute(self, gs):
        counter = gs.get_component_by_id(self.counter_id)
        if self.change > 0 and not counter.is_maximum():
            counter.increment(self.change)
            return True
        elif self.change < 0 and not counter.is_minimum():
            counter.increment(self.change)
            return True
        return False

    def copy(self):
        return ModifyCounter(self.counter_id, self.change)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ModifyCounter):
            return False
        return self.counter_id == other.counter_id and self.change == other.change

    def __hash__(self):
        return hash((self.counter_id, self.change))

    def get_string(self, game_state):
        return f"Modify counter {game_state.get_component_by_id(self.counter_id).get_component_name()} by {self.change}"