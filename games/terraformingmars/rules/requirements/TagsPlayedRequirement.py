from typing import Optional, List, Any
from games.terraformingmars.rules.requirements import Requirement
from games.terraformingmars import Tag
from games.terraformingmars import TMGameState
from utilities.image_io import ImageIO


class TagsPlayedRequirement(Requirement[TMGameState]):
    
    def __init__(self, tags: Optional[List[Tag]], n_min: Optional[List[int]]):
        self.tags = tags
        self.n_min = n_min
        
        self.n_tags = 0
        if tags is not None and n_min is not None:
            for k in range(len(tags)):
                self.n_tags += n_min[k]

    def test_condition(self, gs: TMGameState) -> bool:
        for i in range(len(self.n_min)):
            tag = self.tags[i]
            if gs.get_player_cards_played_tags()[gs.get_current_player()].get(tag).get_value() < self.n_min[i]:
                return False
        return True

    def is_max(self) -> bool:
        return False

    def applies_when_any_player(self) -> bool:
        return False

    def get_display_text(self, gs: TMGameState) -> Optional[str]:
        if self.n_tags > 4:
            return str(self.n_min[0])
        return None

    def get_reason_for_failure(self, gs: TMGameState) -> str:
        reasons = ""
        for i in range(len(self.n_min)):
            tag = self.tags[i]
            if gs.get_player_cards_played_tags()[gs.get_current_player()].get(tag).get_value() < self.n_min[i]:
                reasons += f"Need {self.n_min[i]} {tag} tags. "
            else:
                reasons += f"Enough {tag} tags. "
        return reasons

    def get_display_images(self) -> Optional[List[Any]]:
        n = self.n_tags
        if n > 4:
            n = len(self.tags)
        
        imgs = [None] * n
        i = 0
        
        for k in range(len(self.tags)):
            path = self.tags[k].get_image_path()
            if n == self.n_tags:
                for j in range(self.n_min[k]):
                    imgs[i] = ImageIO.get_instance().get_image(path)
                    i += 1
            else:
                imgs[i] = ImageIO.get_instance().get_image(path)
                i += 1
        
        return imgs

    def copy(self) -> 'TagsPlayedRequirement':
        tags_copy = self.tags.copy() if self.tags is not None else None
        n_min_copy = self.n_min.copy() if self.n_min is not None else None
        return TagsPlayedRequirement(tags_copy, n_min_copy)

    def copy_serializable(self) -> 'Requirement[TMGameState]':
        tags_copy = None
        if self.tags is not None and len(self.tags) > 0:
            tags_copy = self.tags.copy()
        
        n_min_copy = None
        if self.n_min is not None and len(self.n_min) > 0:
            n_min_copy = self.n_min.copy()
        
        return TagsPlayedRequirement(tags_copy, n_min_copy)

    def __str__(self) -> str:
        return "Tags Played"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, TagsPlayedRequirement):
            return False
        return (self.n_tags == other.n_tags and 
                self.tags == other.tags and 
                self.n_min == other.n_min)

    def __hash__(self) -> int:
        tags_tuple = tuple(self.tags) if self.tags is not None else None
        n_min_tuple = tuple(self.n_min) if self.n_min is not None else None
        return hash((self.n_tags, tags_tuple, n_min_tuple))