from typing import List, Dict, Any, Optional

from evaluation.summarisers import TimeStampSummary
from .TAGStatSummary import TAGStatSummary, StatType
from utilities import TimeStamp

class TAGTimeSeriesSummary(TAGStatSummary):
    def __init__(self, name: str = ""):
        super().__init__(name, StatType.Time)
        self.series: List[TimeStamp] = []

    def reset(self) -> None:
        super().reset()
        self.series = []

    def append(self, ktp: TimeStamp) -> None:
        self.series.append(ktp)
        self.n += 1

    def append(self, x: int, v: float) -> None:
        self.series.append(TimeStamp(x, v))
        self.n += 1

    def append(self, ktp: TimeStampSummary) -> None:
        self.series.append(ktp)
        self.n += 1

    def __str__(self) -> str:
        s = f"{self.name}\n" if self.name else ""
        s += f" n     = {self.n}\n"
        return s

    def getElements(self) -> List[TimeStamp]:
        return self.series

    def copy(self) -> 'TAGTimeSeriesSummary':
        ss = TAGTimeSeriesSummary(self.name)
        ss.series = self.series.copy()
        return ss

    def getSummary(self) -> Dict[str, Any]:
        return {self.name: self._process_values(self.series)}

    def _process_values(self, values: List[TimeStamp]) -> Any:
        if len(values) <= 1:
            return values

        if values[0].x != values[1].x:
            return values

        aggregated = []
        y_data = []
        curr_x = values[0].x
        
        for ts in values:
            if ts.x == curr_x:
                y_data.append(ts.v)
            else:
                aggregated.append(TimeStampSummary(curr_x, y_data))
                curr_x = ts.x
                y_data = [ts.v]
        
        if y_data:
            aggregated.append(TimeStampSummary(curr_x, y_data))
        
        return aggregated