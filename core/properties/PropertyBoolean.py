from core.properties.Property import Property

class PropertyBoolean(Property):
    """
    A Property subclass representing a boolean value.
    """
    
    def __init__(self, value: bool, hash_string: str = ""):
        """
        Initializes the PropertyBoolean with a boolean value and an optional hash string.
        
        :param value: The boolean value of the property.
        :param hash_string: Optional string representing the property name (defaults to "").
        """
        super().__init__(hash_string)
        self.value = value
    
    @classmethod
    def _from_hash_key(cls, hash_string: str, hash_key: int, value: bool) -> 'PropertyBoolean':
        """
        Private constructor used internally for copying the property.
        
        :param hash_string: The hash string of the property.
        :param hash_key: The precomputed hash key of the property.
        :param value: The boolean value of the property.
        :return: A new PropertyBoolean instance.
        """
        instance = cls.__new__(cls)
        super(Property, instance).__init__(hash_string, hash_key)
        instance.value = value
        return instance
    
    def __str__(self) -> str:
        """Returns the string representation of the boolean value."""
        return str(self.value)
    
    def __eq__(self, other: object) -> bool:
        """
        Checks equality with another object.
        
        :param other: The object to compare with.
        :return: True if the other object is a PropertyBoolean with the same value, False otherwise.
        """
        if isinstance(other, PropertyBoolean):
            return self.value == other.value
        return False
    
    def _copy(self) -> 'PropertyBoolean':
        """
        Creates a copy of this PropertyBoolean.
        
        :return: A new PropertyBoolean instance with the same hash_string, hash_key, and value.
        """
        return PropertyBoolean._from_hash_key(self.hash_string, self.hash_key, self.value)