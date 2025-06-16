from typing import List, Optional, TypeVar
from core.actions import AbstractAction
from core.interfaces import IExtendedSequence
from games.terraformingmars import TMGameState, TMTypes
from games.terraformingmars.components import TMCard
from games.terraformingmars.actions import TMAction


T = TypeVar('T')

class AddResourceOnCard(TMAction, IExtendedSequence):
    def __init__(self, player: int = -1, card_id: int = -1, resource: Optional[TMTypes.Resource] = None, 
                 amount: int = 0, free: bool = False):
        super().__init__(player, free)
        self.resource = resource
        self.amount = amount
        self.set_card_id(card_id)
        self.choose_any = False
        self.tag_requirement: Optional[TMTypes.Tag] = None
        self.min_res_requirement = 0
        self.tag_top_card_draw_deck: Optional[TMTypes.Tag] = None
        self.last_top_card_draw_deck_tag: Optional[TMTypes.Tag] = None

    def _execute(self, gs: TMGameState) -> bool:
        if self.get_card_id() != -1:
            can_execute = True
            if self.tag_top_card_draw_deck is not None:
                can_execute = False
                top_card = gs.draw_card()
                if top_card is not None:
                    for t in top_card.tags:
                        self.last_top_card_draw_deck_tag = t  # todo show all?
                        if t == self.tag_top_card_draw_deck:
                            can_execute = True
                            break
                    gs.get_discard_cards().add(top_card)
            if can_execute:
                card = gs.get_component_by_id(self.get_card_id())
                if card is not None:
                    # It's null if solo game and action chosen is for a card of the neutral opponent
                    card.n_resources_on_card += self.amount
                return True
            return False
        gs.set_action_in_progress(self)
        return True

    def _compute_available_actions(self, state: AbstractGameState) -> List[AbstractAction]:
        actions = []
        gs = state
        if self.choose_any:
            for i in range(state.get_n_players()):
                self._add_deck_actions(actions, gs, i)
            if state.get_n_players() == 1:
                actions.add(AddResourceOnCard(self.player, -2, self.resource, self.amount, True))
        else:
            self._add_deck_actions(actions, gs, self.player)
        if actions.size() == 0:
            actions.add(TMAction(self.player))  # Pass, can't do any legal actions
        return actions

    def _add_deck_actions(self, actions: List[AbstractAction], gs: TMGameState, player: int) -> None:
        for card in gs.get_player_complicated_point_cards()[player].get_components():
            if card.resource_on_card is not None:
                if (self.resource is not None and card.resource_on_card == self.resource or
                    self.resource is None and card.n_resources_on_card > self.min_res_requirement):
                    if (self.amount > 0 or 
                        self.amount < 0 and card.n_resources_on_card > self.amount and card.can_resources_be_removed):
                        if (self.tag_requirement is None or 
                            self._contains(card.tags, self.tag_requirement)):
                            actions.add(AddResourceOnCard(player, card.get_component_id(), 
                                                        self.resource, self.amount, True))

    def _contains(self, array: List[TMTypes.Tag], target: TMTypes.Tag) -> bool:
        for t in array:
            if target == t:
                return True
        return False

    def get_current_player(self, state: AbstractGameState) -> int:
        return self.player

    def _after_action(self, state: AbstractGameState, action: AbstractAction) -> None:
        if isinstance(action, AddResourceOnCard):
            self.set_card_id(action.get_card_id())
        else:
            self.set_card_id(-2)

    def execution_complete(self, state: AbstractGameState) -> bool:
        complete = self.get_card_id() != -1
        self.set_card_id(-1)
        return complete

    def _copy(self) -> 'AddResourceOnCard':
        copy = AddResourceOnCard(self.player, self.get_card_id(), self.resource, 
                               self.amount, self.free_action_point)
        copy.choose_any = self.choose_any
        copy.tag_requirement = self.tag_requirement
        copy.min_res_requirement = self.min_res_requirement
        copy.tag_top_card_draw_deck = self.tag_top_card_draw_deck
        copy.last_top_card_draw_deck_tag = self.last_top_card_draw_deck_tag
        return copy

    def copy(self) -> 'AddResourceOnCard':
        return super().copy()

    def __eq__(self, o: object) -> bool:
        if self is o:
            return True
        if not isinstance(o, AddResourceOnCard):
            return False
        if not super().__eq__(o):
            return False
        that = o
        return (self.amount == that.amount and 
                self.choose_any == that.choose_any and 
                self.min_res_requirement == that.min_res_requirement and 
                self.resource == that.resource and 
                self.tag_requirement == that.tag_requirement and 
                self.tag_top_card_draw_deck == that.tag_top_card_draw_deck)

    def __hash__(self) -> int:
        return Objects.hash(super().__hash__(), self.resource, self.amount, 
                          self.choose_any, self.tag_requirement, 
                          self.min_res_requirement, self.tag_top_card_draw_deck)

    def get_string(self, game_state: AbstractGameState) -> str:
        if self.get_card_id() != -1:
            card = game_state.get_component_by_id(self.get_card_id())
            if card is None:
                a = 0  # Debug placeholder
            return f"Add {self.amount} {self.resource} on card {card.get_component_name()}"
        return f"Add {self.amount} {self.resource} on {'any ' if self.choose_any else 'another '}card"

    def __str__(self) -> str:
        return (f"Add {self.amount} {self.resource} on " 
                f"{'any ' if self.choose_any else 'another ' if self.get_card_id() == -1 else ''}card")