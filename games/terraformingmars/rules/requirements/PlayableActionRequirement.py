from typing import Any, Optional, List
from games.terraformingmars import TMGameState
from games.terraformingmars.actions import TMAction

class PlayableActionRequirement:
    def __init__(self, action: Optional[TMAction] = None):
        self.action = action

    def test_condition(self, gs: TMGameState) -> bool:
        if self.action is None:
            return False
        return self.action.can_be_played(gs)

    def is_max(self) -> bool:
        return False

    def applies_when_any_player(self) -> bool:
        return False

    def get_display_text(self, gs: TMGameState) -> str:
        return "Playable action"

    def get_reason_for_failure(self, gs: TMGameState) -> str:
        if self.action is None or not hasattr(self.action, 'requirements'):
            return "No action or requirements defined"
            
        reasons = []
        for req in self.action.requirements:
            if req.test_condition(gs):
                reasons.append(f"OK: {req}")
            else:
                failure_reason = req.get_reason_for_failure(gs) if hasattr(req, 'get_reason_for_failure') else "No reason provided"
                reasons.append(f"FAIL: {req} \\\\ {failure_reason}")
        return "\n".join(reasons)

    def get_display_images(self) -> Optional[List[Any]]:
        return None

    def copy(self) -> 'PlayableActionRequirement':
        action_copy = self.action.copy() if self.action is not None else None
        return PlayableActionRequirement(action_copy)

    def copy_serializable(self) -> 'PlayableActionRequirement':
        action_copy = self.action.copy_serializable() if self.action is not None else None
        return PlayableActionRequirement(action_copy)

    def __str__(self) -> str:
        return "Playable Action"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlayableActionRequirement):
            return False
        return self.action == other.action

    def __hash__(self) -> int:
        return hash(self.action)