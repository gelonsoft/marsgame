from typing import Optional, List
from games.terraformingmars.components import Award, Milestone
from games.terraformingmars import TMGameState


class ClaimableAwardMilestoneRequirement:
    def __init__(self, award_milestone_id: int, player: int = -1):
        self.award_milestone_id = award_milestone_id
        self.player = player
    
    def test_condition(self, gs: TMGameState) -> bool:
        p = self.player if self.player != -1 else gs.get_current_player()
        am = gs.get_component_by_id(self.award_milestone_id)
        if isinstance(am, (Award, Milestone)):
            return am.can_claim(gs, p)
        return False
    
    def is_max(self) -> bool:
        return False
    
    def applies_when_any_player(self) -> bool:
        return False
    
    def get_display_text(self, gs: TMGameState) -> Optional[str]:
        return None
    
    def get_reason_for_failure(self, gs: TMGameState) -> str:
        am = gs.get_component_by_id(self.award_milestone_id)
        if not isinstance(am, (Award, Milestone)):
            return "Invalid award/milestone"
        
        reasons = []
        if am.is_claimed():
            reasons.append("Already claimed")
        
        if (isinstance(am, Milestone) and gs.get_n_milestones_claimed().is_maximum()) or \
           (not isinstance(am, Milestone) and gs.get_n_awards_funded().is_maximum()):
            reasons.append("Max claimed")
        
        if isinstance(am, Milestone):
            progress = am.check_progress(gs, self.player)
            reasons.append(f"Not enough: {progress} / {am.min} {am.counter_id}")
        
        return ". ".join(reasons) + "." if reasons else ""
    
    def get_display_images(self) -> Optional[List[Any]]:
        return None
    
    def copy(self) -> 'ClaimableAwardMilestoneRequirement':
        return ClaimableAwardMilestoneRequirement(self.award_milestone_id, self.player)
    
    def __str__(self) -> str:
        return "Award/Milestone"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClaimableAwardMilestoneRequirement):
            return False
        return (self.award_milestone_id == other.award_milestone_id and 
                self.player == other.player)
    
    def __hash__(self) -> int:
        return hash((self.award_milestone_id, self.player))