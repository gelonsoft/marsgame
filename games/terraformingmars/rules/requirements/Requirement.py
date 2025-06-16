from typing import Any, TypeVar, Generic, Optional, List
from abc import ABC, abstractmethod

from games.terraformingmars import TMTypes
from games.terraformingmars.rules.requirements import CounterRequirement, TagsPlayedRequirement
from games.terraformingmars import TMGameState

T = TypeVar('T')

class Requirement(Generic[T], ABC):
    """
    2 cases implemented:
        - counter: global parameter / player resource / player production min or max
        - minimum N tags on cards played by player
    """
    
    @abstractmethod
    def test_condition(self, obj: T) -> bool:
        pass
    
    @abstractmethod
    def is_max(self) -> bool:
        pass
    
    @abstractmethod
    def applies_when_any_player(self) -> bool:
        pass
    
    @abstractmethod
    def get_display_text(self, gs: 'TMGameState') -> str:
        pass
    
    @abstractmethod
    def get_reason_for_failure(self, gs: 'TMGameState') -> str:
        pass
    
    @abstractmethod
    def get_display_images(self) -> Optional[List[Any]]:
        pass
    
    @abstractmethod
    def copy(self) -> 'Requirement[T]':
        pass
    
    def copy_serializable(self) -> 'Requirement[T]':
        return self.copy()
    
    @staticmethod
    def string_to_requirement(s: str) -> 'Requirement':
        split = s.split(":")
        
        # First is counter
        if "tag" in split[0]:
            # Format: tag-tag1-tag2:min1-min2
            tag_part = split[0].replace("tag-", "")
            tag_def = tag_part.split("-")
            min_def = split[1].split("-")
            
            tags = [TMTypes.Tag(tag) for tag in tag_def]
            min_values = [int(min_val) for min_val in min_def]
            
            return TagsPlayedRequirement(tags, min_values)
        else:
            return CounterRequirement(
                split[0], 
                int(split[1]), 
                split[2].lower() == "max"
            )