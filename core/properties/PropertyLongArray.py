
from core.properties.Property import Property


class PropertyLongArray(Property):
    def __init__(self, hashString, values, hashKey=None):
        if hashKey is not None:
            super().__init__(hashString, hashKey)
            self.values = values.copy()
        else:
            super().__init__(hashString)
            if isinstance(values, []):
                self.values = [int(values.get(i)) for i in range(values.size())]
            else:
                self.values = values.copy()

    def getValues(self):
        return self.values

    def __str__(self):
        return self.values.join("")

    def __eq__(self, other):
        if not isinstance(other, PropertyLongArray):
            return False
        if len(other.values) != len(self.values):
            return False
        for i in range(len(self.values)):
            if self.values[i] != other.values[i]:
                return False
        return True

    def _copy(self):
        return PropertyLongArray(self.hashString, self.hashKey, self.values)