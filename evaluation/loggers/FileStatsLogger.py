from abc import ABC, abstractmethod
import os
from typing import Dict, Any, Set, TYPE_CHECKING
from evaluation.loggers import SummaryLogger
from evaluation.summarisers import TAGOccurrenceStatSummary, TAGStatSummary

from core.interfaces import IStatisticLogger

class FileStatsLogger(IStatisticLogger):
    def __init__(self, fileName: str, delimiter: str = "\t", append: bool = True):
        self.fileName = fileName
        self.actionName = ""
        self.append = append
        self.delimiter = delimiter
        self.writer = None
        self.doubleFormat = "%.3g"
        self.intFormat = "%d"
        self.headerNeeded = True
        self.allKeys: Set[str] = set()

    def setOutPutDirectory(self, *nestedDirectories) -> None:
        if self.writer is not None:
            raise AssertionError("Cannot set output directory after initialisation")
        folder = os.path.join(*nestedDirectories)
        self.fileName = os.path.join(folder, self.fileName)

    def _initialise(self) -> None:
        try:
            if os.path.exists(self.fileName) and self.append:
                self.headerNeeded = False
            self.writer = open(self.fileName, "a" if self.append else "w")
        except Exception as e:
            raise AssertionError(f"Problem opening file {self.fileName} : {str(e)}")

    def record(self, rawData: Dict[str, Any]) -> None:
        if self.writer is None:
            self._initialise()
        
        # Preprocess data to remove nesting
        data = {}
        for key, value in rawData.items():
            if isinstance(value, dict):
                data.update(value)
            else:
                data[key] = value
        
        try:
            if not self.allKeys:
                self.allKeys = set(data.keys())
                if self.headerNeeded:
                    outputLine = self.delimiter.join(self.allKeys) + "\n"
                    outputLine = outputLine.replace(f":{self.actionName}{self.delimiter}", self.delimiter)
                    outputLine = outputLine.replace(f":{self.actionName}\n", "\n")
                    self.writer.write(outputLine)
            else:
                for key in data.keys():
                    if key not in self.allKeys:
                        pass  # Unknown key, could log if needed
            
            outputData = []
            for key in self.allKeys:
                datum = data.get(key)
                if datum is None:
                    outputData.append("NA")
                    continue
                
                if isinstance(datum, TAGOccurrenceStatSummary):
                    datum = datum.getHighestOccurrence().a
                
                if isinstance(datum, int):
                    outputData.append(self.intFormat % datum)
                elif isinstance(datum, float):
                    outputData.append(self.doubleFormat % datum)
                elif isinstance(datum, dict):
                    if len(datum) == 1:
                        outputData.append(str(next(iter(datum.values()))))
                    else:
                        outputData.append(str(datum))
                else:
                    outputData.append(str(datum))
            
            if outputData:
                outputLine = self.delimiter.join(outputData) + "\n"
                self.writer.write(outputLine)
        except IOError as e:
            raise AssertionError(f"Problem writing to file {self.writer} : {str(e)}")

    def record(self, key: str, datum: Any) -> None:
        pass  # Single record not implemented

    def processDataAndFinish(self) -> None:
        if self.writer is None:
            return
        try:
            self.writer.flush()
            self.writer.close()
        except Exception as e:
            raise AssertionError(f"Problem closing file {self.writer} : {str(e)}")

    def processDataAndNotFinish(self) -> None:
        if self.writer is None:
            return
        try:
            self.writer.flush()
        except Exception as e:
            raise AssertionError(f"Problem flushing file {self.writer} : {str(e)}")

    def summary(self) -> Dict[str, TAGStatSummary]:
        return {}

    def emptyCopy(self, id: str) -> 'FileStatsLogger':
        fileParts = self.fileName.split(".")
        if len(fileParts) != 2:
            raise AssertionError("Filename does not conform to expected <stem>.<type>")
        newFileName = f"{fileParts[0]}_{id}.{fileParts[1]}"
        retValue = FileStatsLogger(newFileName, self.delimiter, self.append)
        retValue.actionName = id
        return retValue
    
class IStatisticLogger(ABC):
    @abstractmethod
    def record(self, data: Dict[str, Any]) -> None:
        """Use to register a set of data in one go"""
        pass

    @abstractmethod
    def record(self, key: str, datum: Any) -> None:
        """Use to record a single datum"""
        pass

    @abstractmethod
    def processDataAndFinish(self) -> None:
        """Trigger any specific batch processing of data by this Logger"""
        pass

    @abstractmethod
    def processDataAndNotFinish(self) -> None:
        pass

    @abstractmethod
    def summary(self) -> Dict[str, TAGStatSummary]:
        """Return a summary of the data"""
        pass

    @abstractmethod
    def emptyCopy(self, id: str) -> 'IStatisticLogger':
        pass

    @staticmethod
    def createLogger(loggerClass: str, logFile: str) -> 'IStatisticLogger':
        if not logFile:
            raise ValueError("Must specify logFile")
        
        try:
            # In Python we'd typically use importlib for dynamic class loading
            module_name, class_name = loggerClass.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            try:
                # Try constructor with logFile parameter
                logger = cls(logFile)
            except TypeError:
                # Fall back to default constructor
                logger = cls()
                
            return logger
        except Exception as e:
            print(f"Error creating logger: {e}")
            return SummaryLogger()