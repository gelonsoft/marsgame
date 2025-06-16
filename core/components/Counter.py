import json
from typing import List, Optional
from core.components import Component
from core import CoreConstants

class Counter(Component):
    def __init__(self, value_idx: int = 0, minimum: int = 0, maximum: int = None, name: str = "", values: Optional[List[int]] = None, component_id: int = None):
        """
        Initialize a Counter object.
        
        :param value_idx: Current value index of the counter
        :param minimum: Minimum value (inclusive)
        :param maximum: Maximum value (inclusive). Defaults to sys.maxsize if None.
        :param name: Name of the counter
        :param values: Optional list of specific values the counter can take
        :param component_id: Optional ID for the component
        """
        if maximum is None:
            maximum = float('inf')  # Using float('inf') as equivalent to Integer.MAX_VALUE
        
        super().__init__(CoreConstants.ComponentType.COUNTER, name, component_id)
        self.value_idx = value_idx
        self.minimum = minimum
        self.maximum = maximum
        self.values = values.copy() if values is not None else None
        
        # Special case for values array initialization
        if values is not None and maximum == float('inf'):
            self.maximum = len(values) - 1

    def copy(self) -> 'Counter':
        """
        Create a deep copy of this Counter.
        """
        copy = Counter(
            values=self.values.copy() if self.values is not None else None,
            value_idx=self.value_idx,
            minimum=self.minimum,
            maximum=self.maximum,
            name=self.component_name,
            component_id=self.component_id
        )
        self.copy_component_to(copy)
        return copy

    def increment(self, amount: int = 1) -> bool:
        """
        Increment the counter by specified amount.
        
        :param amount: Amount to increment by (default 1)
        :return: True if succeeded, False if capped at max
        """
        self.value_idx += amount
        return self._clamp()

    def decrement(self, amount: int = 1) -> bool:
        """
        Decrement the counter by specified amount.
        
        :param amount: Amount to decrement by (default 1)
        :return: True if succeeded, False if capped at min
        """
        self.value_idx -= amount
        return self._clamp()

    def _clamp(self) -> bool:
        """
        Ensure value stays within bounds.
        
        :return: True if value was within bounds, False if clamped
        """
        if self.value_idx > self.maximum:
            self.value_idx = self.maximum
            return False
        if self.value_idx < self.minimum:
            self.value_idx = self.minimum
            return False
        return True

    def is_minimum(self) -> bool:
        """
        Check if counter is at minimum value.
        """
        return self.value_idx <= self.minimum

    def is_maximum(self) -> bool:
        """
        Check if counter is at maximum value.
        """
        return self.value_idx >= self.maximum

    def get_value(self) -> int:
        """
        Get current counter value.
        """
        if self.values is not None:
            return self.values[self.value_idx]
        return self.value_idx

    def set_value(self, value: int) -> None:
        """
        Set current counter value.
        """
        self.value_idx = value

    def set_to_max(self) -> None:
        """
        Set counter to maximum value.
        """
        self.value_idx = self.maximum

    def set_to_min(self) -> None:
        """
        Set counter to minimum value.
        """
        self.value_idx = self.minimum

    @staticmethod
    def load_counters(filename: str) -> List['Counter']:
        """
        Load counters from JSON file.
        
        :param filename: Path to JSON file
        :return: List of Counter objects
        """
        counters = []
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                for item in data:
                    new_counter = Counter()
                    new_counter.load_counter(item)
                    counters.append(new_counter)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading counters: {e}")
        return counters

    def load_counter(self, counter_data: dict) -> None:
        """
        Load counter data from JSON object.
        """
        self.minimum = counter_data["min"][1]
        self.maximum = counter_data["max"][1]
        self.component_name = counter_data["id"]
        
        if "count" not in counter_data:
            self.value_idx = self.minimum
        else:
            self.value_idx = counter_data["count"][1]
        
        self.parse_component(self, counter_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Counter):
            return False
        if not super().__eq__(other):
            return False
        return (self.value_idx == other.value_idx and 
                self.minimum == other.minimum and 
                self.maximum == other.maximum and 
                self.values == other.values)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.value_idx, self.minimum, self.maximum, tuple(self.values) if self.values is not None else None))

    def __str__(self) -> str:
        return str(self.get_value())

    # Property accessors for Java-style getters
    @property
    def minimum(self) -> int:
        return self._minimum

    @minimum.setter
    def minimum(self, value: int) -> None:
        self._minimum = value

    @property
    def maximum(self) -> int:
        return self._maximum

    @maximum.setter
    def maximum(self, value: int) -> None:
        self._maximum = value

    @property
    def value_idx(self) -> int:
        return self._value_idx

    @value_idx.setter
    def value_idx(self, value: int) -> None:
        self._value_idx = value

    @property
    def values(self) -> Optional[List[int]]:
        return self._values

    @values.setter
    def values(self, value: Optional[List[int]]) -> None:
        self._values = value