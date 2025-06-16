from abc import ABC, abstractmethod
from typing import List,TYPE_CHECKING
if TYPE_CHECKING:
    from core.AbstractGameState import AbstractGameState
    from core.actions.AbstractAction import AbstractAction
    from core.actions.ActionSpace import ActionSpace


class IExtendedSequence(ABC):
    """
    This is a mini-ForwardModel that takes temporary control of:
          i) which player is currently making a decision (the getCurrentPlayer()),
          ii) what actions they have (computeAvailableActions()), and
          iii) what happens after an action is taken (_afterAction()).
    """
    
    @abstractmethod
    def _compute_available_actions(self, state: 'AbstractGameState') -> List['AbstractAction']:
        """
        Forward Model delegates to this from computeAvailableActions() if this Extended Sequence is currently active.
        """
        pass
    
    def _compute_available_actions_with_space(self, state: 'AbstractGameState', action_space: 'ActionSpace') -> List['AbstractAction']:
        """
        Default implementation that delegates to the basic method
        """
        return self._compute_available_actions(state)
    
    @abstractmethod
    def get_current_player(self, state: 'AbstractGameState') -> int:
        """
        TurnOrder delegates to this from getCurrentPlayer() if this Extended Sequence is currently active.
        """
        pass
    
    @abstractmethod
    def _after_action(self, state: 'AbstractGameState', action: 'AbstractAction') -> None:
        """
        Called by ForwardModel whenever an action has just been taken.
        """
        pass
    
    def after_removal_from_queue(self, state: 'AbstractGameState', completed_sequence: 'IExtendedSequence') -> None:
        """
        Called whenever the IExtendedSequence is moved to the top of the queue.
        """
        if isinstance(completed_sequence, AbstractAction):
            self._after_action(state, completed_sequence)
    
    @abstractmethod
    def execution_complete(self, state: 'AbstractGameState') -> bool:
        """
        Return true if this extended sequence has now completed.
        """
        pass
    
    @abstractmethod
    def copy(self) -> 'IExtendedSequence':
        """
        Create a deep copy of the object.
        """
        pass