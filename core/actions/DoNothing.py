# DoNothing.py
from core.actions import AbstractAction
from core import AbstractGameState

class DoNothing(AbstractAction):
    def execute(self, gs):
        return True

    def copy(self):
        return DoNothing()

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, DoNothing)

    def __hash__(self):
        return 0

    def get_string(self, game_state):
        return self.__str__()

    def __str__(self):
        return "DoNothing"