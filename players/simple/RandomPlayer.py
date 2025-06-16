import random
from typing import List
from core import AbstractGameState
from core import AbstractPlayer
from core.actions import AbstractAction

class RandomPlayer(AbstractPlayer):
    def __init__(self, rnd: random.Random = None):
        """
        Initialize a random player.
        
        Args:
            rnd: Optional random number generator. If None, a new one will be created.
        """
        super().__init__(None, "RandomPlayer")
        self.rnd = rnd if rnd is not None else random.Random()

    def _get_action(self, observation: AbstractGameState, actions: List[AbstractAction]) -> AbstractAction:
        """
        Select a random action from the available actions.
        
        Args:
            observation: Current game state
            actions: List of available actions
            
        Returns:
            Randomly selected action
        """
        random_action = self.rnd.randint(0, len(actions) - 1)
        return actions[random_action]

    def copy(self) -> 'RandomPlayer':
        """
        Create a copy of this player with the same random seed.
        
        Returns:
            A new RandomPlayer instance
        """
        new_player = RandomPlayer(random.Random(self.rnd.getrandbits(32)))
        new_player.decorators = self.decorators.copy()
        new_player.set_name(str(self))
        return new_player

    def __eq__(self, obj: object) -> bool:
        """
        Check if another object is a RandomPlayer.
        
        Args:
            obj: Object to compare
            
        Returns:
            True if obj is a RandomPlayer, False otherwise
        """
        return isinstance(obj, RandomPlayer)