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


class ChoiceAction(TMAction, IExtendedSequence):
    """Action that presents a choice between multiple actions."""
    
    def __init__(self, player: int = -1, actions: List[TMAction] = None, 
                 free: bool = True, requirements: Set[Requirement] = None):
        super().__init__(player, free)
        self.actions = actions or []
        self.finished = False
        
        if requirements:
            self.requirements = requirements
        else:
            for action in self.actions:
                self.requirements.add(PlayableActionRequirement(action))
    
    def _execute(self, game_state: TMGameState) -> bool:
        for action in self.actions:
            action.player = self.player
            action.set_card_id(self.get_card_id())
        game_state.set_action_in_progress(self)
        return True
    
    def can_be_played(self, gs: TMGameState) -> bool:
        """OR behavior on requirements instead of default AND."""
        played = False
        if self.get_card_id() != -1:
            card = gs.get_component_by_id(self.get_card_id())
            if card and card.action_played:
                played = True
        
        if played and self.standard_project is None and self.basic_resource_action is None:
            return False
            
        if self.requirements:
            for requirement in self.requirements:
                if requirement.test_condition(gs):
                    return True
        return False
    
    def _compute_available_actions(self, state: AbstractGameState) -> List[AbstractAction]:
        return list(self.actions)
    
    def get_current_player(self, state: AbstractGameState) -> int:
        return self.player
    
    def _after_action(self, state: AbstractGameState, action: AbstractAction) -> None:
        self.finished = True
    
    def execution_complete(self, state: AbstractGameState) -> bool:
        return self.finished
    
    def _copy(self) -> 'ChoiceAction':
        actions_copy = [action.copy() if action else None for action in self.actions]
        copy_obj = ChoiceAction(self.player, actions_copy)
        copy_obj.finished = self.finished
        return copy_obj
    
    def copy(self) -> 'ChoiceAction':
        return super().copy()
    
    def get_string(self, game_state: AbstractGameState) -> str:
        if not self.actions:
            return "Choose from: (no actions)"
        
        s = "Choose from: "
        for action in self.actions:
            s += action.get_string(game_state) + " or "
        return s[:-4]  # Remove last " or "
    
    def __str__(self) -> str:
        if not self.actions:
            return "Choose from: (no actions)"
            
        s = "Choose from: "
        for action in self.actions:
            s += str(action) + " or "
        return s[:-4]  # Remove last " or "
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ChoiceAction):
            return False
        return (super().__eq__(other) and 
                self.finished == other.finished and 
                self.actions == other.actions)
    
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.finished, tuple(self.actions)))