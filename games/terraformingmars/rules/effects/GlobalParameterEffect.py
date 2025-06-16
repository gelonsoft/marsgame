from games.terraformingmars.rules.effects import Effect
from games.terraformingmars import TMGameState, TMTypes
from games.terraformingmars.actions import ModifyGlobalParameter, TMAction
from typing import Optional

class GlobalParameterEffect(Effect):
    def __init__(self, must_be_current_player: bool, effect_action: TMAction, param: TMTypes.GlobalParameter):
        """
        Initialize a GlobalParameterEffect
        
        Args:
            must_be_current_player: Whether the effect only applies to current player
            effect_action: The action to execute when triggered
            param: The global parameter this effect monitors
        """
        super().__init__(must_be_current_player, effect_action)
        self.global_parameter = param

    def execute(self, gs: TMGameState, action_taken: TMAction, player: int) -> None:
        """
        Execute the effect if conditions are met
        
        Args:
            gs: The current game state
            action_taken: The action that triggered this effect
            player: The player who triggered the effect
        """
        if self.can_execute(gs, action_taken, player):
            action = action_taken  # type: ModifyGlobalParameter
            self.effect_action.player = player
            if action.param == self.global_parameter:
                self.effect_action.execute(gs)

    def copy(self) -> 'GlobalParameterEffect':
        """
        Create a copy of this effect
        
        Returns:
            A new GlobalParameterEffect instance with the same properties
        """
        return GlobalParameterEffect(
            self.must_be_current_player,
            self.effect_action.copy(),
            self.global_parameter
        )

    def copy_serializable(self) -> 'GlobalParameterEffect':
        """
        Create a serializable copy of this effect
        
        Returns:
            A new GlobalParameterEffect instance with serializable properties
        """
        return GlobalParameterEffect(
            self.must_be_current_player,
            self.effect_action.copy_serializable(),
            self.global_parameter
        )

    def can_execute(self, game_state: TMGameState, action_taken: TMAction, player: int) -> bool:
        """
        Check if this effect can execute
        
        Args:
            game_state: The current game state
            action_taken: The action that might trigger this effect
            player: The player attempting the action
            
        Returns:
            bool: True if the effect can execute, False otherwise
        """
        if not isinstance(action_taken, ModifyGlobalParameter):
            return False
        if not super().can_execute(game_state, action_taken, player):
            return False
        action = action_taken  # type: ModifyGlobalParameter
        return action.param == self.global_parameter