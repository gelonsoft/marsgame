from typing import List
import inspect

from evaluation.metrics import AbstractMetric

class IMetricsCollection:
    def getAllMetrics(self) -> List['AbstractMetric']:
        metrics = []
        for name, obj in inspect.getmembers(self):
            if inspect.isclass(obj) and issubclass(obj, AbstractMetric) and obj != AbstractMetric:
                try:
                    metric = obj()
                    metrics.append(metric)
                except Exception as e:
                    print(f"Error creating metric {name}: {e}")
        return metrics