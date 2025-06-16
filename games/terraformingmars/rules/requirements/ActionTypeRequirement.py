from enum import Enum
from typing import Optional, Any

from games.terraformingmars import TMTypes
from games.terraformingmars.actions import TMAction
from games.terraformingmars import TMGameState


class ActionTypeRequirement:
    def __init__(self, action_type: TMTypes.ActionType, standard_project: Optional[TMTypes.StandardProject] = None):
        self.action_type = action_type
        self.standard_project = standard_project
    
    def copy(self) -> 'ActionTypeRequirement':
        return self  # Since it's immutable, returning self is fine
    
    def test_condition(self, action: TMAction) -> bool:
        return (action.action_type == self.action_type and 
                ((self.standard_project is None and action.standard_project is None) or 
                 action.standard_project == self.standard_project))
    
    def is_max(self) -> bool:
        return False
    
    def applies_when_any_player(self) -> bool:
        return False
    
    def get_display_text(self, gs: TMGameState) -> Optional[str]:
        return None
    
    def get_reason_for_failure(self, gs: TMGameState) -> Optional[str]:
        return None
    
    def get_display_images(self) -> Optional[list[Any]]:  # Adjust Image type as needed
        return None
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ActionTypeRequirement):
            return False
        return (self.action_type == other.action_type and 
                self.standard_project == other.standard_project)
    
    def __hash__(self) -> int:
        return hash((self.action_type, self.standard_project))
    
    def __str__(self) -> str:
        return "Action Type"