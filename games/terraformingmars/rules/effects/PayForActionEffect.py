from typing import Optional
from games.terraformingmars.rules.effects.effect import Effect
from games.terraformingmars.tm_types import ActionType
from games.terraformingmars.actions.pay_for_action import PayForAction
from games.terraformingmars.actions.tm_action import TMAction
from games.terraformingmars.tm_game_state import TMGameState


class PayForActionEffect(Effect):
    def __init__(self, must_be_current_player: bool, effect_action: TMAction, 
                 action_type: Optional[ActionType] = None, min_cost: Optional[int] = None):
        super().__init__(must_be_current_player, effect_action)
        self.action_type = action_type
        self.min_cost = min_cost or 0

    def can_execute(self, game_state: TMGameState, action_taken: TMAction, player: int) -> bool:
        if not isinstance(action_taken, PayForAction) or not super().can_execute(game_state, action_taken, player):
            return False
        
        action = action_taken
        if self.action_type is not None:
            return action.action_type == self.action_type
        
        return action.get_cost() >= self.min_cost

    def copy(self) -> 'PayForActionEffect':
        copy_obj = PayForActionEffect(
            self.must_be_current_player,
            self.effect_action.copy(),
            self.action_type
        )
        copy_obj.min_cost = self.min_cost
        return copy_obj

    def copy_serializable(self) -> 'PayForActionEffect':
        copy_obj = PayForActionEffect(
            self.must_be_current_player,
            self.effect_action.copy_serializable(),
            self.action_type
        )
        copy_obj.min_cost = self.min_cost
        return copy_obj