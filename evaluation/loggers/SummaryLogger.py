from typing import Dict, Any, List, Optional
import os
from dataclasses import dataclass
import heapq

from core.interfaces import IStatisticLogger
from evaluation.summarisers import TAGNumericStatSummary, TAGOccurrenceStatSummary, TAGStatSummary, TAGTimeSeriesSummary, TimeStampSummary
from utilities import Pair, TimeStamp

class SummaryLogger(IStatisticLogger):
    def __init__(self, logFile: Optional[str] = None):
        self.logFile = logFile
        self.printToConsole = True
        self.data: Dict[str, TAGStatSummary] = {}

    def record(self, key: str, value: Any) -> None:
        summary = self.data.get(key)
        
        if isinstance(value, (int, float)):
            if key not in self.data:
                summary = TAGNumericStatSummary(key)
                self.data[key] = summary
            summary.add(value)
        elif isinstance(value, TimeStamp):
            if key not in self.data:
                summary = TAGTimeSeriesSummary(key)
                self.data[key] = summary
            summary.append(value)
        elif isinstance(value, list) and value and isinstance(value[0], TimeStamp):
            ts = value[0]
            if not isinstance(ts, TimeStampSummary):
                return
                
            if key not in self.data:
                summary = TAGTimeSeriesSummary(key)
                self.data[key] = summary
            for tst in value:
                if isinstance(tst, TimeStampSummary):
                    summary.append(tst)
        else:
            if isinstance(value, dict) and next(iter(value.keys())) is str:
                self.record(value)
            else:
                if key not in self.data:
                    summary = TAGOccurrenceStatSummary(key)
                    self.data[key] = summary
                summary.add(value)

    def record(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            self.record(key, value)

    def summary(self) -> Dict[str, TAGStatSummary]:
        return self.data

    def emptyCopy(self, id: str) -> 'SummaryLogger':
        if self.logFile is None:
            return SummaryLogger()
        return SummaryLogger(self.logFile)

    def processDataAndFinish(self) -> None:
        if self.printToConsole and self.data:
            print("\n" + str(self))

        if self.logFile and os.path.exists(self.logFile):
            try:
                data = self.getFileOutput()
                if data:
                    with open(self.logFile, "a") as writer:
                        writer.write(data.a)  # header
                        writer.write(data.b)  # body
            except IOError as e:
                print(f"Error writing to log file: {e}")

    def getFileOutput(self) -> Optional[Pair]:
        if not self.data:
            return None

        header = []
        outputData = []
        
        for key, summary in self.data.items():
            if isinstance(summary, TAGOccurrenceStatSummary):
                header.append(key)
                outputData.append(str(summary))
            elif isinstance(summary, TAGNumericStatSummary):
                if summary.n() == 1:
                    header.append(key)
                    outputData.append("%.3g" % summary.mean())
                else:
                    header.extend([
                        key, f"{key}_se", f"{key}_sd", 
                        f"{key}_median", f"{key}_min", 
                        f"{key}_max", f"{key}_skew", 
                        f"{key}_kurtosis"
                    ])
                    outputData.extend([
                        "%.3g" % summary.mean(),
                        "%.2g" % summary.stdErr(),
                        "%.3g" % summary.sd(),
                        "%.3g" % summary.median(),
                        "%.3g" % summary.min(),
                        "%.3g" % summary.max(),
                        "%.3g" % summary.skew(),
                        "%.3g" % summary.kurtosis()
                    ])
            elif isinstance(summary, TAGTimeSeriesSummary):
                header.append(f"{key}\tNot implemented\n")
        
        header_str = "\t".join(header) + "\n"
        output_str = "\t".join(outputData) + "\n"
        return Pair(header_str, output_str)

    def processDataAndNotFinish(self) -> None:
        pass

    def __str__(self) -> str:
        sb = []
        groupedData = {}
        keyMaxLength = 0

        for key, summary in self.data.items():
            split = key.split(":")
            group = split[-1] if len(split) == 3 else split[-1]
            
            if group not in groupedData:
                groupedData[group] = {}
            
            split[0] = split[0].replace(")(", " > ")
            split2 = split[0].split("(")
            metricName = split2[0]
            if len(split) > 2:
                metricName += ":" + split[1]
            
            params = split2[1].replace(")", "") if len(split2) == 2 else ""
            
            if metricName not in groupedData[group]:
                groupedData[group][metricName] = {}
            groupedData[group][metricName][params] = summary
            
            keyMaxLength = max(keyMaxLength, len(params), len(metricName))

        for event, eventData in groupedData.items():
            sb.append(f"\n{'#' * len(f'Event: {event}')}")
            sb.append(f"\nEvent: {event}\n")
            sb.append(f"{'#' * len(f'Event: {event}')}\n")
            
            printQueue = []
            for metric, d in eventData.items():
                heapq.heappush(printQueue, (len(d), metric, d))
            
            while printQueue:
                size, metric, d = heapq.heappop(printQueue)
                
                if size > 1:
                    sb.append("\n")
                sb.append(f"{metric.ljust(keyMaxLength)}")
                if size > 1:
                    sb.append("\n")
                
                for key in sorted(d.keys()):
                    summary = d[key]
                    if isinstance(summary, TAGNumericStatSummary):
                        if size > 1:
                            sb.append(f" * {key.ljust(keyMaxLength)}\t")
                        if summary.n() == 1:
                            sb.append(f"\tValue: {summary.mean():8.3g}\n")
                        else:
                            sb.append(
                                f"\tMean: {summary.mean():8.3g} +/- {summary.stdErr():6.2g}, "
                                f"\tMedian: {summary.median():8.3g}, "
                                f"\tSum: {summary.sum():8.3g}, "
                                f"\tRange: [{int(summary.min())}, {int(summary.max())}], "
                                f"\tPop sd: {summary.sd():8.3g}, "
                                f"\tSkew: {summary.skew():8.3g}, "
                                f"\tKurtosis: {summary.kurtosis():8.3g}, "
                                f"\tN: {summary.n()}\n"
                            )
                    elif isinstance(summary, TAGTimeSeriesSummary):
                        sb.append(f"{key}\n")
                        series = summary.getElements()
                        lastX = -1
                        oneSeries = []
                        
                        for i, ts in enumerate(series):
                            x = ts.x
                            if x <= lastX:
                                series_str = self._seriesToString(oneSeries, isinstance(ts, TimeStampSummary))
                                sb.append(series_str)
                                oneSeries = [ts]
                            else:
                                oneSeries.append(ts)
                            
                            if i == len(series) - 1:
                                series_str = self._seriesToString(oneSeries, isinstance(ts, TimeStampSummary))
                                sb.append(series_str)
                            
                            lastX = x
                    else:
                        sb.append(f"\n{summary.stringSummary()}")
        
        return "".join(sb)

    def _seriesToString(self, oneSeries: List[TimeStamp], isSummary: bool) -> str:
        sb = [f"[n: {len(oneSeries)}; "]
        for ts in oneSeries:
            if not isSummary:
                sb.append(f"{ts.v},")
            else:
                tss = ts
                sb.append(f"{tss.values.mean()};")
        sb.append("]\n")
        return "".join(sb).replace(",]", "]")