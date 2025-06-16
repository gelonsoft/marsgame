from abc import ABC, abstractmethod
from typing import List, Set, Optional, Any
import copy
from dataclasses import dataclass, field
from enum import Enum

# Assuming these imports are available in the project
from core import AbstractGameState
from core.actions import AbstractAction
from core.interfaces import IExtendedSequence
from games.terraforming_mars.TMGameState import TMGameState
from games.terraforming_mars import TMTypes
from games.terraforming_mars import TMAction
from games.terraforming_mars.components import TMCard
from games.terraforming_mars.components import Award
from games.terraforming_mars.components import Milestone
from games.terraforming_mars import TMModifyCounter
from games.terraforming_mars.rules.requirements import PlayableActionRequirement
from games.terraforming_mars.rules.requirements import ClaimableAwardMilestoneRequirement
from games.terraforming_mars.rules.requirements import CounterRequirement
from games.terraforming_mars.rules.requirements import ResourceRequirement
from games.terraforming_mars.rules.requirements import Requirement
from games.terraforming_mars.rules.effects import Effect
from games.terraforming_mars.rules.effects import GlobalParameterEffect

class ClaimAwardMilestone(TMAction):
    """Action to claim a milestone or fund an award."""
    
    def __init__(self, player: int = -1, to_claim: Award = None, cost: int = 0,
                 to_claim_id: int = -1, action_type: TMTypes.ActionType = None):
        if to_claim:
            at = TMTypes.ActionType.CLAIM_MILESTONE if isinstance(to_claim, Milestone) else TMTypes.ActionType.FUND_AWARD
            super().__init__(at, player, False)
            self.to_claim_id = to_claim.get_component_id()
        else:
            super().__init__(action_type, player, False)
            self.to_claim_id = to_claim_id
        
        self.set_action_cost(TMTypes.Resource.MEGA_CREDIT, cost, -1)
        self.requirements.add(ClaimableAwardMilestoneRequirement(self.to_claim_id, player))
    
    def _execute(self, gs: TMGameState) -> bool:
        player = self.player if self.player != -1 else gs.get_current_player()
        
        to_claim = gs.get_component_by_id(self.to_claim_id)
        if to_claim.claim(gs, player):
            if isinstance(to_claim, Milestone):
                gs.get_n_milestones_claimed().increment(1)
            else:
                gs.get_n_awards_funded().increment(1)
            return True
        return False
    
    def _copy(self) -> 'ClaimAwardMilestone':
        return ClaimAwardMilestone(self.player, None, self.get_cost(), 
                                 self.to_claim_id, self.action_type)
    
    def get_string(self, game_state: AbstractGameState) -> str:
        to_claim = game_state.get_component_by_id(self.to_claim_id)
        if isinstance(to_claim, Milestone):
            return f"Claim milestone {to_claim.get_component_name()}"
        else:
            return f"Fund award {to_claim.get_component_name()}"
    
    def __str__(self) -> str:
        if self.action_type == TMTypes.ActionType.CLAIM_MILESTONE:
            return f"Claim milestone {self.to_claim_id}"
        else:
            return f"Fund award {self.to_claim_id}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ClaimAwardMilestone):
            return False
        return super().__eq__(other) and self.to_claim_id == other.to_claim_id
    
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.to_claim_id))