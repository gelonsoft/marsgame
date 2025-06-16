from typing import Optional, List, Any
from games.terraformingmars.rules.requirements import Requirement
from games.terraformingmars import Resource
from games.terraformingmars import TMGameState
from games.terraformingmars.components import TMCard


class ResourceRequirement(Requirement[TMGameState]):
    
    def __init__(self, resource: Resource, amount: int, production: bool, player: int, card_id: int):
        self.resource = resource
        self.amount = amount
        self.production = production
        self.player = player
        self.card_id = card_id

    def test_condition(self, gs: TMGameState) -> bool:
        if self.amount == 0:
            return True

        card = None
        if self.card_id != -1:
            card = gs.get_component_by_id(self.card_id)
            if not isinstance(card, TMCard):
                card = None

        p = self.player
        if p == -1:
            p = gs.get_current_player()
        elif p == -2:
            # Can any player pay?
            if self.resource == Resource.Card:
                for i in range(gs.get_n_players()):
                    if gs.get_player_hands()[i].get_size() >= self.amount:
                        return True
            else:
                for i in range(gs.get_n_players()):
                    if gs.can_player_pay(i, card, None, self.resource, self.amount, self.production):
                        return True
            return gs.get_n_players() == 1  # In solo play this is always true

        if self.resource == Resource.Card:
            return gs.get_player_hands()[p].get_size() >= self.amount

        return gs.can_player_pay(p, card, None, self.resource, self.amount, self.production)

    def is_max(self) -> bool:
        return False

    def applies_when_any_player(self) -> bool:
        return False

    def get_display_text(self, gs: TMGameState) -> Optional[str]:
        return None

    def get_reason_for_failure(self, gs: TMGameState) -> str:
        production_text = " production" if self.production else ""
        return f"Need {self.amount} {self.resource}{production_text}"

    def get_display_images(self) -> Optional[List[Any]]:
        return None

    def copy(self) -> 'ResourceRequirement':
        return self

    def __str__(self) -> str:
        production_text = " production" if self.production else ""
        return f"Resource{production_text}"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, ResourceRequirement):
            return False
        return (self.amount == other.amount and 
                self.production == other.production and 
                self.card_id == other.card_id and 
                self.player == other.player and 
                self.resource == other.resource)

    def __hash__(self) -> int:
        return hash((self.resource, self.amount, self.production, self.card_id, self.player))