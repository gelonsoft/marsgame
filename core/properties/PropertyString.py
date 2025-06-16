from core.properties.Property import Property
from typing import Any

class PropertyString(Property):
    def __init__(self, value: str):
        super().__init__("")
        self.value = value
    
    def __init_alt__(self, hash_string: str, value: str):
        super().__init__(hash_string)
        self.value = value
    
    def __init_alt2__(self, hash_string: str, hash_key: int, value: str):
        super().__init__(hash_string, hash_key)
        self.value = value
    
    def __str__(self) -> str:
        return self.value
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PropertyString):
            return self.value == other.value
        return False
    
    def _copy(self) -> 'Property':
        return PropertyString(self.hash_string, self.hash_key, self.value)