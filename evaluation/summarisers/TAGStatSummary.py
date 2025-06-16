from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, Any, List, Optional

from utilities import TimeStamp

class StatType(Enum):
    Numeric = 1
    Occurrence = 2
    Time = 3

class TAGStatSummary(ABC):
    def __init__(self, name: str = "", stat_type: StatType = None):
        self.name = name
        self.type = stat_type
        self.n = 0

    def reset(self) -> None:
        self.n = 0

    def n(self) -> int:
        return self.n

    def add(self, ss: 'TAGStatSummary') -> None:
        self.n += ss.n

    @abstractmethod
    def getElements(self) -> Any:
        pass

    @abstractmethod
    def copy(self) -> 'TAGStatSummary':
        pass

    @abstractmethod
    def getSummary(self) -> Dict[str, Any]:
        pass

    @staticmethod
    def construct(stat_type: StatType) -> Optional['TAGStatSummary']:
        if stat_type == StatType.Numeric:
            return TAGNumericStatSummary()
        elif stat_type == StatType.Occurrence:
            return TAGOccurrenceStatSummary()
        return None

    @staticmethod
    def construct_with_name(name: str, stat_type: StatType) -> Optional['TAGStatSummary']:
        if stat_type == StatType.Numeric:
            return TAGNumericStatSummary(name)
        elif stat_type == StatType.Occurrence:
            return TAGOccurrenceStatSummary(name)
        return None
    
from typing import List, Dict, Any, TYPE_CHECKING
import math
from statistics import median
from .TAGStatSummary import TAGStatSummary, StatType

class TAGNumericStatSummary(TAGStatSummary):
    def __init__(self, name: str = ""):
        super().__init__(name, StatType.Numeric)
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float('inf')
        self.max = float('-inf')
        self.mean = 0.0
        self.median = 0.0
        self.sd = 0.0
        self.last_added = 0.0
        self.valid = False
        self.elements: List[float] = []

    def reset(self) -> None:
        super().reset()
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float('inf')
        self.max = float('-inf')
        self.valid = False
        self.elements = []

    def max(self) -> float:
        if not self.valid:
            self._compute_stats()
        return self.max

    def min(self) -> float:
        if not self.valid:
            self._compute_stats()
        return self.min

    def mean(self) -> float:
        if not self.valid:
            self._compute_stats()
        return self.mean

    def median(self) -> float:
        if not self.valid:
            self._compute_stats()
        return self.median

    def kurtosis(self) -> float:
        if self.n < 4 or self.sd < 0.001:
            return 0.0
        if not self.valid:
            self._compute_stats()
        sum_quartic = sum(math.pow(d - self.mean, 4) for d in self.elements)
        return (sum_quartic / math.pow(self.sd, 4) * self.n * (self.n + 1) / 
                (self.n - 1) / (self.n - 2) / (self.n - 3))

    def skew(self) -> float:
        if self.n < 3 or self.sd < 0.001:
            return 0.0
        if not self.valid:
            self._compute_stats()
        sum_cube = sum(math.pow(d - self.mean, 3) for d in self.elements)
        return (sum_cube / math.pow(self.sd, 3) * self.n / 
                (self.n - 1) / (self.n - 2))

    def sum_square_diff(self) -> float:
        return self.sumsq - self.n * self.mean() * self.mean()

    def _compute_stats(self) -> None:
        if not self.valid and self.elements:
            self.max = max(self.elements)
            self.min = min(self.elements)
            self.mean = self.sum / self.n
            num = self.sumsq - (self.n * self.mean * self.mean)
            num = max(num, 0)  # Avoid tiny negative numbers
            self.sd = math.sqrt(num / (self.n - 1))
            sorted_elements = sorted(self.elements)
            self.median = median(sorted_elements)
            self.valid = True

    def sd(self) -> float:
        if not self.valid:
            self._compute_stats()
        return self.sd

    def std_err(self) -> float:
        return self.sd() / math.sqrt(self.n)

    def add(self, ss: 'TAGNumericStatSummary') -> None:
        super().add(ss)
        self.sum += ss.sum
        self.sumsq += ss.sumsq
        self.last_added = ss.last_added
        self.valid = False
        self.elements.extend(ss.getElements())

    def add(self, d: float) -> None:
        self.n += 1
        self.sum += d
        self.sumsq += d * d
        self.last_added = d
        self.valid = False
        self.elements.append(d)

    def sum(self) -> float:
        return self.sum

    def get_last_added(self) -> float:
        return self.last_added

    def discount_last(self, discount: float) -> None:
        self.last_added *= discount

    def __str__(self) -> str:
        s = f"{self.name}\n" if self.name else ""
        s += (f" min   = {self.min()}\n"
              f" max   = {self.max()}\n"
              f" ave   = {self.mean()}\n"
              f" sd    = {self.sd()}\n"
              f" se    = {self.std_err()}\n"
              f" sum   = {self.sum}\n"
              f" sumsq = {self.sumsq}\n"
              f" n     = {self.n}\n")
        return s

    def short_string(self, scientific_notation: bool = False) -> str:
        prefix = f"{self.name}: [" if self.name else "["
        mean_fmt = "%6.3e" if scientific_notation else "%.2f"
        sd_fmt = "%6.3e" if scientific_notation else "%.2f"
        se_fmt = "%6.3e" if scientific_notation else "%.2f"
        return (f"{prefix}{self.min()}, {self.max()}] "
                f"avg={mean_fmt % self.mean()}; "
                f"sd={sd_fmt % self.sd()}; "
                f"se={se_fmt % self.std_err()}")

    def getElements(self) -> List[float]:
        return self.elements

    def copy(self) -> 'TAGNumericStatSummary':
        ss = TAGNumericStatSummary()
        ss.name = self.name
        ss.n = self.n
        ss.type = self.type
        ss.sum = self.sum
        ss.sumsq = self.sumsq
        ss.min = self.min
        ss.max = self.max
        ss.mean = self.mean
        ss.sd = self.sd
        ss.valid = self.valid
        ss.last_added = self.last_added
        return ss

    def getSummary(self) -> Dict[str, Any]:
        return {
            "Median": self.median(),
            "Mean": self.mean(),
            "Max": self.max(),
            "Min": self.min(),
            "VarCoeff": abs(self.sd() / self.mean()) if self.mean() != 0 else 0.0,
            "Skew": self.skew(),
            "Kurtosis": self.kurtosis(),
            "Delta": self._calculate_delta()
        }

    def _calculate_delta(self) -> float:
        if len(self.elements) <= 1:
            return 0.0
        changes = sum(1 for i in range(len(self.elements)-1) 
                  if self.elements[i+1] != self.elements[i])
        return changes / (len(self.elements) - 1)
    
    
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from .TAGStatSummary import TAGStatSummary, StatType

@dataclass
class DataMeasure:
    data: str
    count: int

    def __lt__(self, other: 'DataMeasure') -> bool:
        if self.count == other.count:
            return self.data < other.data
        return self.count > other.count  # Reverse for descending count

class TAGOccurrenceStatSummary(TAGStatSummary):
    def __init__(self, name: str = ""):
        super().__init__(name, StatType.Occurrence)
        self.elements: Dict[Any, int] = {}

    def reset(self) -> None:
        super().reset()
        self.elements = {}

    def add(self, ss: 'TAGOccurrenceStatSummary') -> None:
        super().add(ss)
        for key, count in ss.elements.items():
            self.elements[key] = self.elements.get(key, 0) + count

    def add_single(self, o: Any) -> None:
        self.elements[o] = self.elements.get(o, 0) + 1
        self.n += 1

    def add(self, s: str) -> None:
        for e in s.split(","):
            if e:
                self.add_single(e.strip())

    def add(self, o: Any) -> None:
        if isinstance(o, str):
            self.add(o)
        else:
            self.add_single(o)

    def get_highest_occurrence(self) -> Optional[tuple[Any, int]]:
        if not self.elements:
            return None
        max_key = max(self.elements.keys(), key=lambda k: self.elements[k])
        return (max_key, self.elements[max_key])

    def get_lowest_occurrence(self) -> Optional[tuple[Any, int]]:
        if not self.elements:
            return None
        min_key = min(self.elements.keys(), key=lambda k: self.elements[k])
        return (min_key, self.elements[min_key])

    def __str__(self) -> str:
        s = f"{self.name}\n" if self.name else ""
        s += f" n     = {self.n}\n{self.elements}"
        return s

    def short_string(self) -> str:
        prefix = f"{self.name}: [" if self.name else "["
        return f"{prefix}{self.elements}]"

    def string_summary(self) -> str:
        sorted_measures = sorted(
            (DataMeasure(str(k), v) for k, v in self.elements.items()),
            key=lambda x: x
        )
        
        result = ["\tCount - Measure"]
        for dm in sorted_measures:
            result.append(f"\t{dm.count} - {dm.data}")
        return "\n".join(result)

    def getElements(self) -> Dict[Any, int]:
        return self.elements

    def copy(self) -> 'TAGOccurrenceStatSummary':
        ss = TAGOccurrenceStatSummary()
        ss.name = self.name
        ss.n = self.n
        ss.type = self.type
        ss.elements = self.elements.copy()
        return ss

    def getSummary(self) -> Dict[str, Any]:
        return {str(k): v for k, v in self.elements.items()}
    
class TimeStampSummary(TimeStamp):
    def __init__(self, x: int, values: List[float]):
        super().__init__(x, float('nan'))
        self.values = TAGNumericStatSummary()
        for d in values:
            self.values.add(d)

    def __str__(self) -> str:
        return f"[x: {self.x}, y: ({self.values})]"
    
    
class TAGSummariser:
    def __init__(self):
        self.supplier = lambda: TAGNumericStatSummary()
        self.accumulator = lambda ss, num: ss.add(num)
        self.combiner = lambda ss1, ss2: ss1.add(ss2) or ss1
        self.finisher = lambda ss: ss
        self.characteristics = {'IDENTITY_FINISH', 'UNORDERED'}

    def supplier(self) -> Callable[[], TAGNumericStatSummary]:
        return self.supplier

    def accumulator(self) -> Callable[[TAGNumericStatSummary, float], None]:
        return self.accumulator

    def combiner(self) -> Callable[[TAGNumericStatSummary, TAGNumericStatSummary], TAGNumericStatSummary]:
        return self.combiner

    def finisher(self) -> Callable[[TAGNumericStatSummary], TAGNumericStatSummary]:
        return self.finisher

    def characteristics(self) -> Set[str]:
        return self.characteristics