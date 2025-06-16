import streamlit as st
from typing import List
from core.components import Deck
from games.terraformingmars import TMGameState
from games.terraformingmars.components import TMCard

class TMDeckDisplay:
    def __init__(self, gui, gs: TMGameState, deck: Deck[TMCard], horizontal: bool):
        self.gui = gui
        self.gs = gs
        self.deck = deck
        self.horizontal = horizontal
        self.max_cards = 200
        self.card_height = 200
        self.card_width = 150  # Placeholder
        
        self.cards = [None] * self.max_cards
    
    def get_highlight_index(self) -> int:
        for i, card in enumerate(self.cards):
            if card and card.get('clicked', False):
                return i
        return -1
    
    def clear_highlights(self):
        for card in self.cards:
            if card:
                card['clicked'] = False
    
    def update(self, deck: Deck[TMCard], highlight_first: bool):
        self.deck = deck
        
        # Update card displays
        for i in range(min(len(deck.getComponents()), self.max_cards)):
            card = deck.getComponents()[i]
            if card:
                self.cards[i] = {
                    'name': card.getComponentName(),
                    'cost': card.getCost(),
                    'tags': card.getTags(),
                    'clicked': self.cards[i].get('clicked', False) if self.cards[i] else False
                }
        
        # Clear remaining slots
        for i in range(len(deck.getComponents()), self.max_cards):
            self.cards[i] = None
        
        # Highlight first card if requested
        if highlight_first and len(deck.getComponents()) > 0 and not self.cards[0]['clicked']:
            self.cards[0]['clicked'] = True
    
    def display(self):
        if not self.deck or self.deck.getSize() == 0:
            st.write("No cards in deck")
            return
            
        if self.horizontal:
            cols = st.columns(len(self.deck.getComponents()))
            for i, card in enumerate(self.cards[:len(self.deck.getComponents())]):
                with cols[i]:
                    self._display_card(card, i)
        else:
            for i, card in enumerate(self.cards[:len(self.deck.getComponents())]):
                self._display_card(card, i)
    
    def _display_card(self, card_data, index):
        if not card_data:
            return
            
        with st.container():
            st.write(f"**{card_data['name']}**")
            st.write(f"Cost: {card_data['cost']}")
            st.write("Tags: " + ", ".join(tag.name() for tag in card_data['tags']))
            
            if st.button("Select", key=f"card_{index}"):
                self.clear_highlights()
                card_data['clicked'] = True
                self.gui.update_buttons = True