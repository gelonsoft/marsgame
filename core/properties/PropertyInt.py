from core.properties.Property import Property

class PropertyInt(Property):
    def __init__(self, hash_string, value, hash_key=None):
        if hash_key is None:
            super().__init__(hash_string)
        else:
            super().__init__(hash_string, hash_key)
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, PropertyInt):
            return self.value == other.value
        return False

    def _copy(self):
        return PropertyInt(self.hashString, self.value, self.hashKey)