from core.properties.Property import Property
from utilities.Utils import Utils  # Assuming this utility exists
from typing import Tuple  # For type hints

class PropertyColor(Property):
    """
    A Property subclass representing a color value.
    """
    
    def __init__(self, hash_string: str, val_str: str):
        """
        Initializes the PropertyColor with a color string and a hash string.
        
        :param hash_string: The string representing the property name.
        :param val_str: The string representation of the color (e.g., "RED" or "#FF0000").
        """
        super().__init__(hash_string)
        self.value = Utils.string_to_color(val_str)  # Convert string to Color object
        self.value_str = val_str
    
    @classmethod
    def _from_components(cls, hash_string: str, hash_key: int, value: Tuple[int, int, int], value_str: str) -> 'PropertyColor':
        """
        Private constructor used internally for copying the property.
        
        :param hash_string: The hash string of the property.
        :param hash_key: The precomputed hash key of the property.
        :param value: The Color object.
        :param value_str: The string representation of the color.
        :return: A new PropertyColor instance.
        """
        instance = cls.__new__(cls)
        super(Property, instance).__init__(hash_string, hash_key)
        instance.value = value
        instance.value_str = value_str
        return instance
    
    def __str__(self) -> str:
        """Returns the string representation of the color value."""
        return str(self.value)
    
    def __eq__(self, other: object) -> bool:
        """
        Checks equality with another object.
        
        :param other: The object to compare with.
        :return: True if the other object is a PropertyColor with the same color value, False otherwise.
        """
        if isinstance(other, PropertyColor):
            return self.value == other.value
        return False
    
    def _copy(self) -> 'PropertyColor':
        """
        Creates a copy of this PropertyColor.
        
        :return: A new PropertyColor instance with the same hash_string, hash_key, color value, and value_str.
        """
        return PropertyColor._from_components(self.hash_string, self.hash_key, self.value, self.value_str)