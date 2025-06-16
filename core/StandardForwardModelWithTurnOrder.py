from abc import abstractmethod
from typing import Optional

from core import AbstractForwardModel
from core.actions import AbstractAction
from core.interfaces import IExtendedSequence


class StandardForwardModelWithTurnOrder(AbstractForwardModel):
    """
    This is purely for old-style game implementations from before January 2023 that use the now deprecated TurnOrder.

    This has been deprecated because it all too often led to a mixture of logic and state, and ambiguity over where 
    any individual piece of game logic should be implemented.
    
    The new standard (See StandardForwardModel) is to have a clean separation of:
     - state within something that extends AbstractGameState
     - game logic within something that extends AbstractForwardModel (and this has new method hooks to help)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the deprecated StandardForwardModelWithTurnOrder."""
        super().__init__(*args, **kwargs)
        import warnings
        warnings.warn(
            "StandardForwardModelWithTurnOrder is deprecated. Use StandardForwardModel instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def _next(self, current_state, action: Optional[AbstractAction]) -> None:
        """
        Execute the action and handle extended sequence logic.
        
        Args:
            current_state: Current game state to be modified
            action: Action to execute (must not be None)
        """
        if action is not None:
            action.execute(current_state)
        else:
            raise AssertionError("No action selected by current player")
        
        # We then register the action with the top of the stack ... unless the top of the stack is this action
        # in which case go to the next action
        # We can't just register with all items in the Stack, as this may represent some complex dependency
        # For example in Dominion where one can Throne Room a Throne Room, which then Thrones a Smithy
        if len(current_state.actions_in_progress) > 0:
            top_of_stack = current_state.actions_in_progress.pop()
            if top_of_stack != action:
                top_of_stack._after_action(current_state, action)
            else:
                if len(current_state.actions_in_progress) > 0:
                    next_on_stack = current_state.actions_in_progress[-1]  # peek at top
                    next_on_stack._after_action(current_state, action)
            current_state.actions_in_progress.append(top_of_stack)  # push back
        
        self._after_action(current_state, action)

    @abstractmethod
    def _after_action(self, current_state, action_taken: AbstractAction) -> None:
        """
        Handle any logic that should occur after an action is executed.
        
        Args:
            current_state: Current game state
            action_taken: The action that was just executed
        """
        pass

    def end_player_turn(self, gs) -> None:
        """
        End the current player's turn using the turn order system.
        
        Args:
            gs: Game state (must be AbstractGameStateWithTurnOrder)
        """
        # Note: This assumes gs is an AbstractGameStateWithTurnOrder
        # In Python, we'll need to ensure this is properly typed or add a check
        gs.get_turn_order().end_player_turn(gs)