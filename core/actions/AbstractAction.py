from typing import Set, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

from core.interfaces.IPrintable import IPrintable

if TYPE_CHECKING:
    from core.actions import AbstractGameState

class AbstractAction(IPrintable, ABC):
    """
    Abstract base class for game actions.
    """
    
    @abstractmethod
    def execute(self, gs: 'AbstractGameState') -> bool:
        """
        Executes this action, applying its effect to the given game state. Can access any component IDs stored
        through the AbstractGameState.getComponentById(int id) method.

        Args:
            gs: game state which should be modified by this action.

        Returns:
            bool: True if successfully executed, False otherwise.
        """
        pass
    
    @abstractmethod
    def copy(self) -> 'AbstractAction':
        """
        Create a copy of this action, with all of its variables.
        NO REFERENCES TO COMPONENTS TO BE KEPT IN ACTIONS, PRIMITIVE TYPES ONLY.

        Returns:
            AbstractAction: new AbstractAction object with the same properties.
        """
        pass
    
    @abstractmethod
    def __eq__(self, obj: Any) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def get_string(self, game_state: 'AbstractGameState') -> str:
        pass
    
    def get_string_for_perspective(self, gs: 'AbstractGameState', perspective_player: int) -> str:
        """
        Returns the string representation of this action from the perspective of the given player.

        Args:
            gs: game state to be used to generate the string.
            perspective_player: player to whom the action should be represented.

        Returns:
            str: string representation of this action.
        """
        return self.get_string(gs)
    
    def get_string_for_perspective_set(self, gs: 'AbstractGameState', perspective_set: Set[int]) -> str:
        """
        Helper method to get string representation for a set of perspective players.

        Args:
            gs: game state to be used to generate the string.
            perspective_set: set of player IDs who are viewing the action.

        Returns:
            str: string representation of this action.
        """
        current_player = gs.get_current_player()
        perspective = current_player if current_player in perspective_set else next(iter(perspective_set), current_player)
        return self.get_string_for_perspective(gs, perspective)
    
    def get_tooltip(self, gs: 'AbstractGameState') -> str:
        """
        Returns a tooltip string for this action (empty by default).

        Args:
            gs: game state to be used to generate the tooltip.

        Returns:
            str: tooltip string.
        """
        return ""