from utilities.Hash import Hash  # Assuming Hash is implemented as a singleton

class Property:
    """
    Abstract base class for properties. Each property has a name (hashString) and a hash key.
    """
    
    def __init__(self, hash_string: str, hash_key: int = None):
        """
        Initializes the Property with a hash string and an optional hash key.
        If hash_key is not provided, it is computed using the Hash singleton.
        
        :param hash_string: The string representing the property name.
        :param hash_key: Optional precomputed hash key for the property.
        """
        self.hash_string = hash_string
        self.hash_key = hash_key if hash_key is not None else Hash.get_instance().hash(hash_string)
    
    @property
    def get_hash_string(self) -> str:
        """Returns the hash string of the property."""
        return self.hash_string
    
    @property
    def get_hash_key(self) -> int:
        """Returns the hash key of the property."""
        return self.hash_key
    
    def _copy(self) -> 'Property':
        """
        Abstract method to create a copy of this property.
        To be implemented by subclasses.
        
        :return: A new Property object with the same hash_string and hash_key.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def copy(self) -> 'Property':
        """
        Creates a copy of this property. This is a final method that calls _copy().
        
        :return: A new Property object with the same hash_string and hash_key.
        """
        return self._copy()
    
    def __str__(self) -> str:
        """Abstract method for string representation. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __eq__(self, other: object) -> bool:
        """Abstract method for equality comparison. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __hash__(self) -> int:
        """Returns the hash key of the property."""
        return self.hash_key