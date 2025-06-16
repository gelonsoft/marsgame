from typing import Dict, List, Set, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import re
from core.actions import AbstractAction
from core.components import Counter
from games.terraformingmars import TMGameState, TMTypes
from games.terraformingmars.components import TMCard
from games.terraformingmars.rules.effects import Effect
from games.terraformingmars.rules.requirements import AdjacencyRequirement, Requirement, ResourceRequirement
from utilities import Utils, Pair

class TMAction(AbstractAction):
    def __init__(self, *args):
        """
        Multiple constructors handled via *args:
        - TMAction(action_type: TMTypes.ActionType, player: int, free: bool)
        - TMAction(project: TMTypes.StandardProject, player: int, free: bool)
        - TMAction(basic_resource_action: TMTypes.BasicResourceAction, player: int, free: bool)
        - TMAction(player: int)
        - TMAction(player: int, free: bool)
        - TMAction(player: int, free: bool, requirements: Set[Requirement])
        """
        self.free_action_point: bool = False
        self.player: int = -1
        self.pass_action: bool = False
        self.cost_requirement: Optional[Requirement] = None
        self.requirements: Set[Requirement] = set()
        self.action_type: Optional[TMTypes.ActionType] = None
        self.standard_project: Optional[TMTypes.StandardProject] = None
        self.basic_resource_action: Optional[TMTypes.BasicResourceAction] = None
        self._cost: int = 0
        self._cost_resource: Optional[TMTypes.Resource] = None
        self._play_card_id: int = -1  # Card used to play this action
        self._card_id: int = -1  # Card related to the action

        if len(args) == 3:
            if isinstance(args[0], TMTypes.ActionType):
                self._init_action_type(*args)
            elif isinstance(args[0], TMTypes.StandardProject):
                self._init_standard_project(*args)
            elif isinstance(args[0], TMTypes.BasicResourceAction):
                self._init_basic_resource_action(*args)
        elif len(args) == 1:
            self._init_pass_action(args[0])
        elif len(args) == 2:
            self._init_simple_action(*args)
        elif len(args) == 3 and isinstance(args[2], set):
            self._init_with_requirements(*args)

    def _init_action_type(self, action_type: TMTypes.ActionType, player: int, free: bool):
        self.player = player
        self.free_action_point = free
        self.action_type = action_type

    def _init_standard_project(self, project: TMTypes.StandardProject, player: int, free: bool):
        self.player = player
        self.free_action_point = free
        self.action_type = TMTypes.ActionType.StandardProject
        self.standard_project = project

    def _init_basic_resource_action(self, basic_action: TMTypes.BasicResourceAction, player: int, free: bool):
        self.player = player
        self.free_action_point = free
        self.action_type = TMTypes.ActionType.BasicResourceAction
        self.basic_resource_action = basic_action

    def _init_pass_action(self, player: int):
        self.player = player
        self.pass_action = True

    def _init_simple_action(self, player: int, free: bool):
        self.player = player
        self.free_action_point = free

    def _init_with_requirements(self, player: int, free: bool, requirements: Set[Requirement]):
        self.player = player
        self.free_action_point = free
        self.requirements = set(requirements)

    def set_action_cost(self, resource: TMTypes.Resource, cost: int, card_id: int) -> None:
        self._cost_resource = resource
        self._cost = cost
        self._play_card_id = card_id
        self.cost_requirement = ResourceRequirement(resource, abs(cost), False, self.player, card_id)
        self.requirements.add(self.cost_requirement)

    def can_be_played(self, gs: TMGameState) -> bool:
        if self._card_id != -1:
            card = gs.get_component_by_id(self._card_id)
            if card and card.action_played and not (self.standard_project or self.basic_resource_action):
                return False
        
        return all(req.test_condition(gs) for req in self.requirements) if self.requirements else True

    def _execute(self, gs: TMGameState) -> bool:
        return True

    def execute(self, game_state: AbstractGameState) -> bool:
        gs = game_state  # type: TMGameState
        gs.get_all_components()  # Force recalculate components
        
        if self.player == -1:
            self.player = game_state.get_current_player()
        
        if not self.can_be_played(gs):
            raise AssertionError(f"Card cannot be played {self}")
        
        result = self._execute(gs)
        self.post_execute(gs)
        return result

    def post_execute(self, gs: TMGameState) -> None:
        player = self.player if self.player != -1 else gs.get_current_player()
        
        if not (0 <= player < gs.get_n_players()):
            return
        
        if not self.free_action_point:
            gs.get_turn_order().register_action_taken(gs, self, player)
        
        if self._card_id != -1 and not isinstance(self, (BuyCard, PlayCard, DiscardCard)):
            card = gs.get_component_by_id(self._card_id)
            if card:
                if not card.first_action_executed and card.first_action:
                    card.first_action_executed = True
                elif card.card_type in {TMTypes.CardType.Active, TMTypes.CardType.Corporation}:
                    card.action_played = True
        
        # Check persisting effects for all players
        for i in range(gs.get_n_players()):
            for effect in gs.get_player_persisting_effects()[i]:
                if effect:
                    effect.execute(gs, self, i)
        
        # Handle card resources
        card_counter = gs.get_player_resources()[player].get(TMTypes.Resource.Card)
        n_cards = card_counter.get_value()
        
        if n_cards > 0:
            for _ in range(n_cards):
                card = gs.draw_card()
                if card:
                    gs.get_player_hands()[player].add(card)
                else:
                    break
            card_counter.set_value(0)
        elif n_cards < 0:
            for _ in range(abs(n_cards)):
                DiscardCard(player, False).execute(gs)

    def _copy(self) -> 'TMAction':
        return TMAction(self.player, self.free_action_point)

    def copy(self) -> 'TMAction':
        action = self._copy()
        action.free_action_point = self.free_action_point
        action.player = self.player
        action.pass_action = self.pass_action
        
        if self.cost_requirement:
            action.cost_requirement = self.cost_requirement.copy()
        
        action.requirements = {req.copy() for req in self.requirements} if self.requirements else set()
        action.action_type = self.action_type
        action.standard_project = self.standard_project
        action.basic_resource_action = self.basic_resource_action
        action._cost = self._cost
        action._cost_resource = self._cost_resource
        action._play_card_id = self._play_card_id
        action._card_id = self._card_id
        
        return action

    def copy_serializable(self) -> 'TMAction':
        action = self._copy()
        action.free_action_point = self.free_action_point
        action.player = self.player
        action.pass_action = self.pass_action
        
        if self.cost_requirement:
            action.cost_requirement = self.cost_requirement.copy()
        
        action.requirements = {req.copy() for req in self.requirements} if self.requirements else None
        action.action_type = self.action_type
        action.standard_project = self.standard_project
        action.basic_resource_action = self.basic_resource_action
        action._cost = self._cost
        action._cost_resource = self._cost_resource
        action._play_card_id = self._play_card_id
        action._card_id = self._card_id
        
        return action

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TMAction):
            return False
        return (self.free_action_point == other.free_action_point and
                self.player == other.player and
                self.pass_action == other.pass_action and
                self._cost == other._cost and
                self._play_card_id == other._play_card_id and
                self._card_id == other._card_id and
                self.cost_requirement == other.cost_requirement and
                self.requirements == other.requirements and
                self.action_type == other.action_type and
                self.standard_project == other.standard_project and
                self.basic_resource_action == other.basic_resource_action and
                self._cost_resource == other._cost_resource)

    def __hash__(self) -> int:
        return hash((self.free_action_point, self.player, self.pass_action, self.cost_requirement,
                    frozenset(self.requirements) if self.requirements else None, self.action_type,
                    self.standard_project, self.basic_resource_action, self._cost, self._cost_resource,
                    self._play_card_id, self._card_id))

    def get_string(self, game_state: AbstractGameState) -> str:
        return self.action_type.name if self.action_type else "Pass"

    def __str__(self) -> str:
        return self.action_type.name if self.action_type else "Pass"

    @property
    def cost(self) -> int:
        return self._cost

    @cost.setter
    def cost(self, value: int) -> None:
        self._cost = value

    @property
    def cost_resource(self) -> Optional[TMTypes.Resource]:
        return self._cost_resource

    @property
    def play_card_id(self) -> int:
        return self._play_card_id

    def set_card_id(self, card_id: int) -> None:
        self._card_id = card_id

    @property
    def card_id(self) -> int:
        return self._card_id

    # Parsing methods
    @staticmethod
    def parse_action_on_card(s: str, card: Optional[TMCard], free: bool) -> Optional['TMAction']:
        or_split = s.split(" or ")
        and_split = s.split(" and ")
        card_id = card.get_component_id() if card else -1

        if len(or_split) == 1:
            if len(and_split) == 1:
                action = TMAction._parse_action(s, free, card_id).a
                if action:
                    if isinstance(action, PlaceTile):
                        action.set_card_id(card_id)
                    elif card and isinstance(action, AddResourceOnCard) and (not action.choose_any or action.card_id == -1):
                        card.resource_on_card = action.resource
                    return action
            else:
                return TMAction._process_compound_action(free, and_split, card_id)
        else:
            action_choices = []
            for s2 in or_split:
                s2 = s2.strip()
                and_split = s2.split(" and ")
                if len(and_split) == 1:
                    action = TMAction._parse_action(s2, free, card_id).a
                    if action and isinstance(action, PlaceTile):
                        action.set_card_id(card_id)
                    action_choices.append(action)
                else:
                    action_choices.append(TMAction._process_compound_action(free, and_split, card_id))
            return ChoiceAction(-1, [a for a in action_choices if a is not None])

        return None

    @staticmethod
    def _process_compound_action(free: bool, and_split: List[str], card_id: int) -> 'CompoundAction':
        actions = []
        for s3 in and_split:
            s3 = s3.strip()
            action = TMAction._parse_action(s3, free, card_id).a
            if action and isinstance(action, PlaceTile):
                action.set_card_id(card_id)
            actions.append(action)
        return CompoundAction(-1, [a for a in actions if a is not None])

    @staticmethod
    def _parse_action(encoding: str, free: bool, card_id: int) -> Pair:
        effect = None
        effect_string = ""
        player = -1

        if "inc" in encoding or "dec" in encoding:
            # Handle increase/decrease actions
            pass  # Implementation omitted for brevity
        elif "placetile" in encoding:
            # Handle place tile actions
            pass  # Implementation omitted for brevity
        elif "reserve" in encoding:
            # Handle reserve tile actions
            pass  # Implementation omitted for brevity
        elif "add" in encoding or "rem" in encoding:
            # Handle add/remove resource actions
            pass  # Implementation omitted for brevity
        elif "duplicate" in encoding:
            # Handle duplicate actions
            pass  # Implementation omitted for brevity
        elif "look" in encoding:
            # Handle look at top cards actions
            pass  # Implementation omitted for brevity

        return Pair(effect, effect_string)