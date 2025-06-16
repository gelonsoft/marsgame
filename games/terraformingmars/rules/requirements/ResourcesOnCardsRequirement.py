from typing import Optional, List, Any
from games.terraformingmars.rules.requirements.requirement import Requirement
from games.terraformingmars.tm_types import Resource
from games.terraformingmars.tm_game_state import TMGameState
from games.terraformingmars.components.tm_card import TMCard
from utilities.image_io import ImageIO


class ResourcesOnCardsRequirement(Requirement[TMGameState]):
    
    def __init__(self, resources: Optional[List[Resource]], n_min: Optional[List[int]]):
        self.resources = resources
        self.n_min = n_min
        
        self.n_resources = 0
        if resources is not None:
            for k in range(len(resources)):
                self.n_resources += n_min[k]

    def test_condition(self, gs: TMGameState) -> bool:
        for i in range(len(self.n_min)):
            res = self.resources[i]
            n_res = 0
            for c in gs.get_player_complicated_point_cards()[gs.get_current_player()].get_components():
                if isinstance(c, TMCard) and c.resource_on_card == res:
                    n_res += c.n_resources_on_card
            if n_res < self.n_min[i]:
                return False
        return True

    def is_max(self) -> bool:
        return False

    def applies_when_any_player(self) -> bool:
        return False

    def get_display_text(self, gs: TMGameState) -> Optional[str]:
        if self.n_resources > 4:
            return str(self.n_min[0])
        return None

    def get_reason_for_failure(self, gs: TMGameState) -> str:
        reasons = ""
        for i in range(len(self.n_min)):
            res = self.resources[i]
            n_res = 0
            for c in gs.get_player_complicated_point_cards()[gs.get_current_player()].get_components():
                if isinstance(c, TMCard) and c.resource_on_card == res:
                    n_res += c.n_resources_on_card
            if n_res < self.n_min[i]:
                reasons += f"Need {self.n_min[i]} {res}s on cards. "
            else:
                reasons += f"Enough {res}s on cards. "
        return reasons

    def get_display_images(self) -> Optional[List[Any]]:
        n = self.n_resources
        if n > 4:
            n = len(self.resources)
        
        imgs = [None] * n
        i = 0
        
        for k in range(len(self.resources)):
            path = self.resources[k].get_image_path()
            if n == self.n_resources:
                for j in range(self.n_min[k]):
                    imgs[i] = ImageIO.get_instance().get_image(path)
                    i += 1
            else:
                imgs[i] = ImageIO.get_instance().get_image(path)
                i += 1
        
        return imgs

    def copy(self) -> 'ResourcesOnCardsRequirement':
        resources_copy = self.resources.copy() if self.resources is not None else None
        n_min_copy = self.n_min.copy() if self.n_min is not None else None
        return ResourcesOnCardsRequirement(resources_copy, n_min_copy)

    def copy_serializable(self) -> 'Requirement[TMGameState]':
        resources_copy = None
        if self.resources is not None and len(self.resources) > 0:
            resources_copy = self.resources.copy()
        
        n_min_copy = None
        if self.n_min is not None and len(self.n_min) > 0:
            n_min_copy = self.n_min.copy()
        
        return ResourcesOnCardsRequirement(resources_copy, n_min_copy)

    def __str__(self) -> str:
        return "Tags Played"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, ResourcesOnCardsRequirement):
            return False
        return (self.n_resources == other.n_resources and 
                self.resources == other.resources and 
                self.n_min == other.n_min)

    def __hash__(self) -> int:
        resources_tuple = tuple(self.resources) if self.resources is not None else None
        n_min_tuple = tuple(self.n_min) if self.n_min is not None else None
        return hash((self.n_resources, resources_tuple, n_min_tuple))