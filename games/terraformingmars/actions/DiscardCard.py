
from games.terraformingmars import TMGameState
from games.terraformingmars.actions import TMAction


class DiscardCard(TMAction, IExtendedSequence):
    """Action to discard a card from hand or card choice."""
    
    def __init__(self, player: int = -1, card_id: int = -1, card_choice: bool = False):
        super().__init__(player, True)
        self.set_card_id(card_id)
        self.card_choice = card_choice
    
    def _execute(self, gs: TMGameState) -> bool:
        card = gs.get_component_by_id(self.get_card_id())
        if card:
            if card.card_type != TMTypes.CardType.CORPORATION:
                gs.get_discard_cards().add(card)
            
            if self.card_choice:
                gs.get_player_card_choice()[self.player].remove(card)
            else:
                gs.get_player_hands()[self.player].remove(card)
        else:
            gs.set_action_in_progress(self)
        
        return True
    
    def _compute_available_actions(self, state: AbstractGameState) -> List[AbstractAction]:
        """Choose which card to discard."""
        actions = []
        gs = state
        
        if self.card_choice:
            for i in range(gs.get_player_card_choice()[self.player].get_size()):
                card = gs.get_player_card_choice()[self.player].get(i)
                actions.append(DiscardCard(self.player, card.get_component_id(), self.card_choice))
        else:
            for i in range(gs.get_player_hands()[self.player].get_size()):
                card = gs.get_player_hands()[self.player].get(i)
                actions.append(DiscardCard(self.player, card.get_component_id(), self.card_choice))
        
        if not actions:
            actions.append(TMAction(self.player))
        
        return actions
    
    def get_current_player(self, state: AbstractGameState) -> int:
        return self.player
    
    def _after_action(self, state: AbstractGameState, action: AbstractAction) -> None:
        self.set_card_id(action.get_card_id())
    
    def execution_complete(self, state: AbstractGameState) -> bool:
        return self.get_card_id() != -1
    
    def _copy(self) -> 'DiscardCard':
        return DiscardCard(self.player, self.get_card_id(), self.card_choice)
    
    def copy(self) -> 'DiscardCard':
        return super().copy()
    
    def can_be_played(self, gs: TMGameState) -> bool:
        return True
    
    def get_string(self, game_state: AbstractGameState) -> str:
        component = game_state.get_component_by_id(self.get_card_id())
        if not component:
            return "Discard a card"
        return f"Discard {component.get_component_name()}"
    
    def __str__(self) -> str:
        if self.get_card_id() == -1:
            return "Discard a card"
        return f"Discard card id {self.get_card_id()}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, DiscardCard):
            return False
        return super().__eq__(other) and self.card_choice == other.card_choice
    
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.card_choice))