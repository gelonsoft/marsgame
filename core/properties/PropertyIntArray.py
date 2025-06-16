from core.properties.Property import Property


class PropertyIntArray(Property):
    def __init__(self, hash_string, values, hash_key=None):
        if hash_key is None:
            super().__init__(hash_string)
            # Convert JSON array/list to Python list of integers
            self.values = [int(val) for val in values]
        else:
            super().__init__(hash_string, hash_key)
            # Create a copy of the input array
            self.values = values.copy()

    def get_values(self):
        return self.values.copy()

    def __str__(self):
        return str(self.values)

    def __eq__(self, other):
        if not isinstance(other, PropertyIntArray):
            return False
        return self.values == other.values

    def _copy(self):
        return PropertyIntArray(self.hashString, self.values, self.hashKey)