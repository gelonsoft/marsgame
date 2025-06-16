from typing import Optional, List, Any
from games.terraformingmars.rules.requirements import Requirement
from games.terraformingmars import Tag
from games.terraformingmars import TMGameState
from games.terraformingmars.components import TMCard


class TagOnCardRequirement(Requirement[TMCard]):
    
    def __init__(self, tags: Optional[List[Tag]]):
        self.tags = tags  # card must contain all of these tags

    def test_condition(self, card: TMCard) -> bool:
        if card is None:
            return False
        if self.tags is None:
            return True
        
        for tag in self.tags:
            found = False
            for t in card.tags:
                if t == tag:
                    found = True
                    break
            if not found:
                return False
        
        return True

    def is_max(self) -> bool:
        return False

    def applies_when_any_player(self) -> bool:
        return False

    def get_display_text(self, gs: TMGameState) -> Optional[str]:
        return None

    def get_reason_for_failure(self, gs: TMGameState) -> Optional[str]:
        return None

    def get_display_images(self) -> Optional[List[Any]]:
        return None

    def copy(self) -> 'TagOnCardRequirement':
        tags_copy = self.tags.copy() if self.tags is not None else None
        return TagOnCardRequirement(tags_copy)

    def copy_serializable(self) -> 'Requirement[TMCard]':
        tags_copy = None
        if self.tags is not None and len(self.tags) > 0:
            tags_copy = self.tags.copy()
        return TagOnCardRequirement(tags_copy)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, TagOnCardRequirement):
            return False
        return self.tags == other.tags

    def __hash__(self) -> int:
        tags_tuple = tuple(self.tags) if self.tags is not None else None
        return hash(tags_tuple)

    def __str__(self) -> str:
        return "Tag On Card"