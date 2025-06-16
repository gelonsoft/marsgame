# RemoveComponentFromDeck.py
from core.actions import DrawCard
from core.components import Component, Deck
from core import AbstractGameState

class RemoveComponentFromDeck(DrawCard):
    def __init__(self, deck_from, deck_to, from_index, deck_remove_id, component_remove_idx):
        super().__init__(deck_from, deck_to, from_index)
        self.deck = deck_remove_id
        self.component_idx = component_remove_idx

    def execute(self, gs):
        gs.get_component_by_id(self.deck).remove(self.component_idx)  # card removed from the game
        return super().execute(gs)  # Discard other card from player hand

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, RemoveComponentFromDeck):
            return False
        if not super().__eq__(other):
            return False
        return self.deck == other.deck and self.component_idx == other.component_idx

    def __hash__(self):
        return hash((super().__hash__(), self.deck, self.component_idx))

    def __str__(self):
        return f"RemoveComponentFromDeck{{deck={self.deck}, component={self.component_idx}}}"

    def get_string(self, game_state):
        return (f"RemoveComponentFromDeck{{deck={self.deck}, component={self.component_idx}, "
                f"cardPlayed={self.get_card(game_state).get_component_name()}}}")

    def get_component_idx(self):
        return self.component_idx

    def get_deck(self):
        return self.deck

    def copy(self):
        return RemoveComponentFromDeck(self.deck_from, self.deck_to, self.from_index, self.deck, self.component_idx)