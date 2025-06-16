from dataclasses import dataclass
from typing import Optional
from core.properties import Property

@dataclass
class PropertyLong(Property):
    value: int  # Using int since Python handles both int and long transparently

    def __init__(self, hash_string: str, value: int, hash_key: Optional[int] = None):
        if hash_key is not None:
            super().__init__(hash_string, hash_key)
        else:
            super().__init__(hash_string)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropertyLong):
            return False
        return self.value == other.value

    def _copy(self) -> 'PropertyLong':
        return PropertyLong(self.hash_string, self.value, self.hash_key)