from typing import Set, Optional
from games.terraformingmars.rules.effects.effect import Effect
from games.terraformingmars.tm_types import Tag
from games.terraformingmars.actions.pay_for_action import PayForAction
from games.terraformingmars.actions.play_card import PlayCard
from games.terraformingmars.actions.tm_action import TMAction
from games.terraformingmars.tm_game_state import TMGameState
from games.terraformingmars.components.tm_card import TMCard


class PlayCardEffect(Effect):
    def __init__(self, must_be_current_player: bool, effect_action: TMAction, 
                 tags_on_card: Set[Tag]):
        super().__init__(must_be_current_player, effect_action)
        self.tags_on_card = tags_on_card

    def execute(self, gs: TMGameState, action_taken: TMAction, player: int) -> None:
        if self.can_execute(gs, action_taken, player):
            action = action_taken.action
            self.effect_action.player = player
            
            if self.effect_action.get_card_id() == -1:
                # Effect based on card played, e.g. add resource to that card
                self.effect_action.set_card_id(action.get_play_card_id())
            
            self.effect_action.execute(gs)  # TODO execute multiple times

    def copy(self) -> 'PlayCardEffect':
        return PlayCardEffect(
            self.must_be_current_player,
            self.effect_action.copy() if self.effect_action is not None else None,
            set(self.tags_on_card)
        )

    def copy_serializable(self) -> 'PlayCardEffect':
        tags_copy = None
        if self.tags_on_card is not None and len(self.tags_on_card) > 0:
            tags_copy = set(self.tags_on_card)
        
        return PlayCardEffect(
            self.must_be_current_player,
            self.effect_action.copy_serializable() if self.effect_action is not None else None,
            tags_copy
        )

    def can_execute(self, game_state: TMGameState, action_taken: TMAction, player: int) -> bool:
        # PlayCard is always wrapped in PayForAction
        if not isinstance(action_taken, PayForAction) or \
           not super().can_execute(game_state, action_taken, player):
            return False
        
        aa = action_taken
        if not isinstance(aa.action, PlayCard):
            return False
        
        action = aa.action
        card = game_state.get_component_by_id(action.get_play_card_id())
        
        if isinstance(card, TMCard):
            for t in card.tags:
                if t in self.tags_on_card:
                    return True
        
        return False