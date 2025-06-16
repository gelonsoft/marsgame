# MoveCardById.py
from core.actions import AbstractAction
from core.components import Card, Deck

class MoveCardById(AbstractAction):
    def __init__(self, deck_from_id, deck_to_id, card_id):
        self.deck_from_id = deck_from_id
        self.deck_to_id = deck_to_id
        self.card_id = card_id

    def execute(self, gs):
        from_deck = gs.get_component_by_id(self.deck_from_id)
        to_deck = gs.get_component_by_id(self.deck_to_id)
        card = gs.get_component_by_id(self.card_id)
        
        if from_deck is None or to_deck is None or card is None:
            return False
            
        to_deck.add(card)
        from_deck.remove(card)
        return True

    def copy(self):
        return self

    def __eq__(self, other):
        if isinstance(other, MoveCardById):
            return (other.deck_from_id == self.deck_from_id and 
                    other.deck_to_id == self.deck_to_id and 
                    other.card_id == self.card_id)
        return False

    def __hash__(self):
        return hash((self.deck_from_id, self.deck_to_id, self.card_id))

    def get_string(self, game_state):
        from_deck = game_state.get_component_by_id(self.deck_from_id)
        to_deck = game_state.get_component_by_id(self.deck_to_id)
        card = game_state.get_component_by_id(self.card_id)
        return f"Move card {card.get_component_name()} from {from_deck.get_component_name()} to {to_deck.get_component_name()}"

    def __str__(self):
        return f"Move card {self.card_id} from deck {self.deck_from_id} to deck {self.deck_to_id}"

    def get_card_id(self):
        return self.card_id

    def get_deck_to_id(self):
        return self.deck_to_id

    def get_deck_from_id(self):
        return self.deck_from_id