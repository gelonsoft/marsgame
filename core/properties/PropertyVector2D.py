from core.properties.Property import Property
from utilities import Vector2D

class PropertyVector2D(Property):
    def __init__(self, hashString, values, hashKey=None):
        super().__init__(hashString, hashKey) if hashKey is not None else super().__init__(hashString)
        if isinstance(values, list):
            self.values = Vector2D(int(int(values.get(0))), int(int(values.get(1))))
        else:
            self.values = Vector2D(values.getX(), values.getY())

    def __str__(self):
        return str(self.values)

    def __eq__(self, o):
        if self is o:
            return True
        if o is None or self.__class__ != o.__class__:
            return False
        return self.values== o.values

    def _copy(self):
        return PropertyVector2D(self.hashString, self.hashKey, self.values)