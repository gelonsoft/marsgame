from typing import Any, Optional, List, Dict
from games.terraformingmars import TMGameState
from core.components import Counter
from games.terraformingmars import TMTypes

class CounterRequirement:
    def __init__(self, counter_code: str, threshold: int, is_max: bool):
        self.counter_code = counter_code
        self.threshold_idx = threshold
        self.max = is_max
        self.counter_id = -1

    def test_condition(self, gs: TMGameState) -> bool:
        counter = self._get_counter(gs)
        if not counter:
            return False
            
        value = counter.get_value_idx()
        discount = self._get_discount(gs)

        if self.max:
            return (value - discount) <= self.threshold_idx
        else:
            return (value + discount) >= self.threshold_idx

    def _get_discount(self, gs: TMGameState) -> int:
        discount = 0
        player = gs.get_current_player()
        for req, amount in gs.get_player_discount_effects()[player].items():
            if isinstance(req, CounterRequirement) and req.counter_code.lower() == self.counter_code.lower():
                discount = amount
                break
        return discount

    def is_max(self) -> bool:
        return self.max

    def applies_when_any_player(self) -> bool:
        return False

    def get_display_text(self, gs: TMGameState) -> str:
        counter = self._get_counter(gs)
        if not counter:
            return ""
            
        param = TMTypes.GlobalParameter.from_string(counter.get_component_name())
        if param:
            return f"{counter.get_values()[self.threshold_idx]} {param.get_short_string()}"
        else:
            return f"{counter.get_value()} {counter.get_component_name()}"

    def get_reason_for_failure(self, gs: TMGameState) -> str:
        counter = self._get_counter(gs)
        if not counter:
            return "Counter not found"
            
        value = counter.get_value()
        discount = self._get_discount(gs)

        if self.max:
            return f"value {value - discount} when max {self.get_display_text(gs)}"
        else:
            return f"value {value + discount} when min {self.get_display_text(gs)}"

    def get_display_images(self) -> Optional[List[Any]]:
        return None  # TODO: Implement if needed

    def copy(self) -> 'CounterRequirement':
        copy = CounterRequirement(self.counter_code, self.threshold_idx, self.max)
        copy.counter_id = self.counter_id
        return copy

    def _get_counter(self, gs: TMGameState) -> Optional[Counter]:
        if self.counter_id == -1:
            counter = gs.string_to_gp_or_player_res_counter(self.counter_code, -1)
            if not counter:
                return None
                
            self.counter_id = counter.get_component_id()
            
            # Convert threshold to index for certain counters
            if counter.get_component_name().lower() in ["temperature", "venus"]:
                self.threshold_idx = counter.get_values().index(self.threshold_idx)
        
        else:
            counter = gs.get_component_by_id(self.counter_id)
            if not isinstance(counter, Counter):
                return None

        if self.max and self.threshold_idx == -1:
            self.threshold_idx = counter.get_maximum() - 1

        return counter

    def __str__(self) -> str:
        return "Counter Value"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CounterRequirement):
            return False
        return (self.counter_code.lower() == other.counter_code.lower() and 
                self.max == other.max)

    def __hash__(self) -> int:
        return hash((self.counter_code.lower(), self.max))