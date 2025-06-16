from typing import TypeVar, Generic, Any
from dataclasses import dataclass

T = TypeVar('T')
V = TypeVar('V')

@dataclass
class Pair(Generic[T, V]):
    a: T
    b: V

    @classmethod
    def of(cls, a: T, b: V) -> 'Pair[T, V]':
        return cls(a, b)

    def swap(self) -> None:
        self.a, self.b = self.b, self.a  # type: ignore

    def copy(self) -> 'Pair[T, V]':
        return Pair(self.a, self.b)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Pair):
            return False
        return self.a == other.a and self.b == other.b

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def __str__(self) -> str:
        return f"<{self.a};{self.b}>"