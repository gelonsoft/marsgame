from typing import Optional, List
from games.terraformingmars import TMGameState, TMTypes


class ResourceIncGenRequirement:
    def __init__(self, resource: TMTypes.Resource):
        self.resource = resource

    def test_condition(self, gs: TMGameState) -> bool:
        """Check if this resource was increased for current player in this generation"""
        player_resources = gs.get_player_resource_increase_gen()
        current_player = gs.get_current_player()
        return player_resources.get(current_player, {}).get(self.resource, False)

    def is_max(self) -> bool:
        return False

    def applies_when_any_player(self) -> bool:
        return False

    def get_display_text(self, gs: TMGameState) -> str:
        return f"inc {self.resource.name} this gen"

    def get_reason_for_failure(self, gs: TMGameState) -> str:
        return f"{self.resource} not increased this generation"

    def get_display_images(self) -> Optional[List[Any]]:
        return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceIncGenRequirement):
            return False
        return self.resource == other.resource

    def __hash__(self) -> int:
        return hash(self.resource)

    def copy(self) -> 'ResourceIncGenRequirement':
        return self  # Immutable, so return self

    def __str__(self) -> str:
        return "Resource Increased This Generation"