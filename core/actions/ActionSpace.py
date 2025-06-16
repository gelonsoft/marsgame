from enum import Enum
from typing import Any

class ActionSpace:
    """
    <p>See games.loveletter.LoveLetterForwardModel._computeAvailableActions(AbstractGameState, ActionSpace)
    for example implementations of structured action spaces.</p>
    
    <p>These are used within the forward model to compute the available actions for a given game state.
    Each call to this function can request different lists of actions for different types of action spaces.
    Ideally, the same action space should be kept throughout a game (otherwise deep action spaces might not receive the
    correct sequence of decisions required).</p>
    """
    
    Default = None  # Will be initialized after the class definition
    
    def __init__(self, structure: 'Structure' = None, flexibility: 'Flexibility' = None, context: 'Context' = None):
        """
        Constructor for ActionSpace. Can be initialized with any combination of Structure, Flexibility, and Context.
        If any parameter is None, it defaults to the corresponding Default enum value.
        """
        self.structure = structure if structure is not None else Structure.Default
        self.flexibility = flexibility if flexibility is not None else Flexibility.Default
        self.context = context if context is not None else Context.Default
    
    def is_default(self) -> bool:
        """Returns True if this ActionSpace is equal to the Default ActionSpace."""
        return self == ActionSpace.Default
    

    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ActionSpace):
            return False
        return (self.structure == other.structure and 
                self.flexibility == other.flexibility and 
                self.context == other.context)
    
    def __hash__(self) -> int:
        return hash((self.structure, self.flexibility, self.context))

class Structure(Enum):
    """
    Defines the structure of the action space. Current options supported in some games:
    - Default: whichever option the game considers by default, usually 'Flat'. Used when no other structure specified.
    - Flat: combinatorial list of full actions, considering all combinations of parameter values.
    - Deep: Actions with multiple parameters/decisions required are split into several decisions.
    """
    Default = "Default"
    Flat = "Flat"
    Deep = "Deep"

class Flexibility(Enum):
    """
    Defines the flexibility of the action space. The options are currently NOT supported in any of the games.
    - Default: whichever option the game considers by default, usually 'Rigid'. Used when no other flexibility specified.
    - Rigid: if multiple decisions need to be taken for an action, the order for these decisions is fixed.
    - Elastic: if multiple decisions need to be taken for an action, the agent has an additional choice to make, for
        the order in which the decisions will be taken. Useful for e.g. decision trees to maximise use of information gain.
    """
    Default = "Default"
    Rigid = "Rigid"
    Elastic = "Elastic"

class Context(Enum):
    """
    Defines the context of the action space. Current options supported in some games:
    - Default: whichever option the game considers by default, usually 'Independent'. Used when no other context specified.
    - Independent: the actions are self-contained and have the same effect in any game state they are applied to (e.g. play card 'King').
    - Dependent: actions require context from the game state, and effect will differ (e.g. play 3rd card in hand).
    """
    Default = "Default"
    Independent = "Independent"
    Dependent = "Dependent"
# Initialize the Default ActionSpace after the class is fully defined
ActionSpace.Default = ActionSpace()