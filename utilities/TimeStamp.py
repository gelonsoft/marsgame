class TimeStamp:
    def __init__(self, x: int, value: float):
        self.x = x
        self.v = value

    def __str__(self) -> str:
        return f"[x: {self.x}, y: {self.v}]"