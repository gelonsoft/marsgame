from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core import AbstractGameState

class IPrintable(ABC):
    """
    Interface for objects that can be printed to console with optional game state context.
    """
    
    def get_string(self, game_state: Optional['AbstractGameState'] = None) -> str:
        """
        Retrieves a short string for this object, optionally using game state for context.
        
        Args:
            game_state: Optional game state provided for context
            
        Returns:
            str: String representation of the object
        """
        return str(self)

    def print_to_console(self, game_state: Optional['AbstractGameState'] = None) -> None:
        """
        Prints the object's state to console, optionally using game state for context.
        
        Args:
            game_state: Optional game state provided for context
        """
        print(self.get_string(game_state))

    # Python's built-in __str__ serves the same purpose as toString() in Java
    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        
        Returns:
            str: String representation of the object
        """
        pass