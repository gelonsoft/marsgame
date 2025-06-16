import json
from typing import Type, TypeVar, Generic, Optional
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class Pair(Generic[T, U]):
    """Generic pair class to replace utilities.Pair"""
    a: T
    b: U

class Discount(Pair['Requirement', int]):
    """A discount rule that applies when certain requirements are met"""
    
    def __init__(self, requirement: Optional['Requirement'] = None, amount: Optional[int] = None):
        """
        Initialize a Discount
        
        Args:
            requirement: The requirement that must be met for the discount
            amount: The discount amount to apply when requirement is met
        """
        super().__init__(requirement, amount)
    
    def serialize(self) -> dict:
        """
        Serialize the discount to a dictionary
        
        Returns:
            dict: A dictionary representation of the discount
        """
        return {
            'requirement': self._serialize_requirement(self.a),
            'amount': self.b
        }
    
    @classmethod
    def deserialize(cls, data: dict) -> 'Discount':
        """
        Deserialize a discount from a dictionary
        
        Args:
            data: Dictionary containing serialized discount data
            
        Returns:
            Discount: A new Discount instance
        """
        requirement = cls._deserialize_requirement(data.get('requirement'))
        amount = data.get('amount', 0)
        return cls(requirement, amount)
    
    @staticmethod
    def _serialize_requirement(req: 'Requirement') -> dict:
        """Helper method to serialize a requirement"""
        # Implement requirement serialization logic here
        # This would depend on your Requirement class implementation
        if req is None:
            return {}
        return req.serialize()  # Assuming Requirement has serialize() method
    
    @staticmethod
    def _deserialize_requirement(data: dict) -> Optional['Requirement']:
        """Helper method to deserialize a requirement"""
        # Implement requirement deserialization logic here
        # This would depend on your Requirement class implementation
        if not data:
            return None
        return Requirement.deserialize(data)  # Assuming Requirement has deserialize() method
    
    def to_json(self) -> str:
        """Convert the discount to JSON string"""
        return json.dumps(self.serialize())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Discount':
        """Create a discount from JSON string"""
        data = json.loads(json_str)
        return cls.deserialize(data)