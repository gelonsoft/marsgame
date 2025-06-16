from typing import List, Set, Optional
from core.actions import AbstractAction
from core.interfaces import IExtendedSequence
from games.terraformingmars.actions.TMAction import TMAction
from games.terraformingmars.components.TMCard import TMCard
from games.terraformingmars.TMTypes import Resource, ActionType
import copy

class SellProjects(TMAction, IExtendedSequence):
    def __init__(self, player: int = -1, card_id: int = -1):
        super().__init__(player, free_action_point=(card_id != -1))
        self.action_type = ActionType.StandardProject if card_id == -1 else None
        self.card_ids_sold: Set[int] = set()
        self.complete = False
        if card_id != -1:
            self.set_card_id(card_id)

    def _execute(self, gs) -> bool:
        if self.get_card_id() != -1:
            gp = gs.get_game_parameters()
            current_mc = gs.get_player_resources()[self.player].get(Resource.MegaCredit).get_value()
            card = gs.get_component_by_id(self.get_card_id())
            if card is not None:
                gs.get_discard_cards().add(card)
                gs.get_player_hands()[self.player].remove(card)
            gs.get_player_resources()[self.player].get(Resource.MegaCredit).set_value(current_mc + 1)
            return True
        gs.set_action_in_progress(self)
        return True

    def _compute_available_actions(self, state) -> List[AbstractAction]:
        actions = []
        gs = state
        for i in range(gs.get_player_hands()[self.player].get_size()):
            card = gs.get_player_hands()[self.player].get(i)
            if card.get_component_id() not in self.card_ids_sold:
                actions.append(SellProjects(self.player, card.get_component_id()))
        if self.card_ids_sold:
            actions.append(TMAction(self.player))  # Pass to stop the action
        return actions

    def get_current_player(self, state) -> int:
        return self.player

    def _after_action(self, state, action) -> None:
        if action.pass_action:
            self.complete = True
        else:
            self.card_ids_sold.add(action.get_card_id())

    def execution_complete(self, state) -> bool:
        return self.complete

    def _copy(self) -> 'SellProjects':
        copy_obj = SellProjects(self.player, self.get_card_id())
        copy_obj.complete = self.complete
        copy_obj.card_ids_sold = set(self.card_ids_sold)
        return copy_obj

    def copy(self) -> 'SellProjects':
        return super().copy()

    def get_string(self, game_state) -> str:
        if self.get_card_id() == -1:
            return "Sell projects"
        c = game_state.get_component_by_id(self.get_card_id())
        return f"Sell {c.get_component_name()}"

    def __str__(self) -> str:
        if self.get_card_id() == -1:
            return "Sell projects"
        return f"Sell card id {self.get_card_id()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, SellProjects):
            return False
        return (super().__eq__(other) and
                self.complete == other.complete and
                self.card_ids_sold == other.card_ids_sold)

    def __hash__(self) -> int:
        return hash((super().__hash__(), frozenset(self.card_ids_sold), self.complete))