# ReplaceComponents.py
from core.actions import DrawComponents
from core.components import Component, Deck
from core import AbstractGameState

class ReplaceComponents(DrawComponents):
    def __init__(self, deck_from, deck_to, n_components, deck_draw):
        super().__init__(deck_from, deck_to, n_components)
        self.deck_draw = deck_draw

    def execute(self, gs):
        super().execute(gs)
        from_deck = gs.get_component_by_id(self.deck_from)
        draw_deck = gs.get_component_by_id(self.deck_draw)

        for i in range(self.n_components):
            from_deck.add(draw_deck.draw())
        return True

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ReplaceComponents):
            return False
        return self.deck_draw == other.deck_draw

    def __hash__(self):
        return hash(self.deck_draw)

    def print_to_console(self):
        print("Replace components")

    def get_string(self, game_state):
        return (f"ReplaceComponents{{deckDraw={game_state.get_component_by_id(self.deck_draw).get_component_name()}"
                f"deckTo={game_state.get_component_by_id(self.deck_to).get_component_name()}"
                f"drawFrom={game_state.get_component_by_id(self.deck_from).get_component_name()}}}")

    def __str__(self):
        return f"ReplaceComponents{{deckDraw={self.deck_draw}}}"

    def copy(self):
        return ReplaceComponents(self.deck_from, self.deck_to, self.n_components, self.deck_draw)