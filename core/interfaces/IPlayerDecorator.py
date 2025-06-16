
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.interfaces import AbstractGameState, AbstractAction

class IPlayerDecorator:
    """
    This is the core interface to be implemented by all player decorators.

    It takes the list of possible actions and returns a filtered list of actions. This filtered list
    is then passed to the underlying AbstractPlayer.
    """
    
    def action_filter(self, state: 'AbstractGameState', possible_actions: List['AbstractAction']) -> List['AbstractAction']:
        """
        Filters the list of possible actions.
        
        Args:
            state: The current game state.
            possible_actions: List of possible actions to filter.
            
        Returns:
            List[AbstractAction]: The filtered list of actions.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")
    
    def record_decision(self, state: 'AbstractGameState', action: 'AbstractAction') -> None:
        """
        Applies logic after the decision is made.
        Provides the actual decision selected by the underlying AbstractPlayer.
        
        Args:
            state: The current game state.
            action: The action selected by the player.
        """
        pass  # do nothing as default; override if needed
    
    def decision_player_only(self) -> bool:
        """
        Indicates if the Decorator only applies to the decision player.
        The default is to use the same decorator for all players (e.g., when constructing the MCTS Search tree).
        But we may also want to see the impact of, "We cannot use X; but we will plan for other players to do so".
        
        Returns:
            bool: True if the decorator applies only to the decision player; False otherwise.
        """
        return False