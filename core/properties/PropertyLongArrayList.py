from core.properties.Property import Property
from typing import List, Any
import json

class PropertyLongArrayList(Property):
    def __init__(self, hash_string: str, values: List[Any]):
        super().__init__(hash_string)
        self.values: List[int] = []
        for value in values:
            self.values.append(int(value))
    
    def __init_alt__(self, hash_string: str, hash_key: int, values: List[int]):
        super().__init__(hash_string, hash_key)
        self.values: List[int] = []
        self.values.extend(values)
    
    def get_values(self) -> List[int]:
        return self.values
    
    def __str__(self) -> str:
        return str(self.values)
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PropertyLongArrayList):
            psto = other
            if len(psto.values) == len(self.values):
                for i in range(len(self.values)):
                    if self.values[i] != psto.values[i]:
                        return False
            else:
                return False
        else:
            return False
        return True
    
    def _copy(self) -> 'Property':
        return PropertyLongArrayList(self.hash_string, self.hash_key, self.values)