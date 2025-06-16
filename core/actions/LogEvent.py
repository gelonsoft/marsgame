# LogEvent.py
from core.actions.AbstractAction import AbstractAction

class LogEvent(AbstractAction):
    def __init__(self, message):
        self.text = message

    def execute(self, gs):
        return True

    def copy(self):
        return self

    def __eq__(self, other):
        return isinstance(other, LogEvent) and other.text == self.text

    def __hash__(self):
        return hash(self.text) - 31

    def get_string(self, game_state):
        return self.text

    def __str__(self):
        return self.text