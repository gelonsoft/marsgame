class Hash:
    _instance = None
    _hashmap = {}

    @classmethod
    def GetInstance(cls):
        if cls._instance is None:
            cls._instance = Hash()
        return cls._instance

    def __init__(self):
        self._hashmap = {}

    def hash(self, key):
        if key in self._hashmap:
            return self._hashmap[key]

        new_value = hash(key)  # Using Python's built-in hash function
        self._hashmap[key] = new_value
        return new_value