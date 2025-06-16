from typing import List
from core.actions import AbstractAction
from core.interfaces import IExtendedSequence
from games.terraformingmars.actions.TMAction import TMAction
from games.terraformingmars.actions.BuyCard import BuyCard
from games.terraformingmars.actions.DiscardCard import DiscardCard
from games.terraformingmars.actions.PayForAction import PayForAction
from games.terraformingmars.components.TMCard import TMCard
from games.terraformingmars.TMGameState import TMGameState

class TopCardDecision(TMAction, IExtendedSequence):
    def __init__(self, n_cards_look: int = 0, n_cards_keep: int = 0, buy: bool = False):
        super().__init__(player=-1, free=True)
        self.stage: int = 0
        self.n_cards_kept: int = 0
        self.n_cards_look: int = n_cards_look
        self.n_cards_keep: int = n_cards_keep
        self.buy: bool = buy

    def _execute(self, game_state: TMGameState) -> bool:
        for i in range(self.n_cards_look):
            card = game_state.draw_card()
            if card is not None:
                game_state.get_player_card_choice()[self.player].add(card)
            else:
                self.n_cards_look = i
                break
        
        self.stage = 0
        self.n_cards_kept = 0
        game_state.set_action_in_progress(self)
        return True

    def _compute_available_actions(self, state: TMGameState) -> List[AbstractAction]:
        actions = []
        player_choice = state.get_player_card_choice()[self.player]
        
        if player_choice.get_size() == 0:
            return actions
            
        card_id = player_choice.get(0).get_component_id()
        
        # Option to keep the card
        if self.n_cards_look == 1 or self.n_cards_kept < self.n_cards_keep:
            if self.buy:
                cost = state.get_game_parameters().get_project_purchase_cost()
                actions.append(PayForAction(self.player, BuyCard(self.player, card_id, cost)))
            else:
                actions.append(BuyCard(self.player, card_id, 0))
        
        # Option to discard the card
        if self.n_cards_look == 1 or (self.n_cards_look - self.stage) > (self.n_cards_keep - self.n_cards_kept):
            actions.append(DiscardCard(self.player, card_id, True))
        
        return actions

    def get_current_player(self, state: TMGameState) -> int:
        return self.player

    def _after_action(self, state: TMGameState, action: AbstractAction) -> None:
        self.stage += 1
        if isinstance(action, BuyCard):
            self.n_cards_kept += 1

    def execution_complete(self, state: TMGameState) -> bool:
        return state.get_player_card_choice()[self.player].get_size() == 0

    def _copy(self) -> 'TopCardDecision':
        copy = TopCardDecision(self.n_cards_look, self.n_cards_keep, self.buy)
        copy.n_cards_kept = self.n_cards_kept
        copy.stage = self.stage
        return copy

    def copy(self) -> 'TopCardDecision':
        return super().copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopCardDecision):
            return False
        return (super().__eq__(other) and
                self.stage == other.stage and
                self.n_cards_kept == other.n_cards_kept and
                self.n_cards_look == other.n_cards_look and
                self.n_cards_keep == other.n_cards_keep and
                self.buy == other.buy)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.stage, self.n_cards_kept, 
                    self.n_cards_look, self.n_cards_keep, self.buy))

    def __str__(self) -> str:
        text = f"Look at the top {self.n_cards_look} card{'s' if self.n_cards_look > 1 else ''}."
        if self.buy:
            text += f" Buy {self.n_cards_keep if self.n_cards_keep > 1 else 'it'} or discard all."
        else:
            text += f" Take {self.n_cards_keep} of them into your hand and discard the rest."
        return text

    def get_string(self, game_state: TMGameState) -> str:
        return self.__str__()