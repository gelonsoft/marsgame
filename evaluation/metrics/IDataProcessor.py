from evaluation.metrics import IDataLogger

class IDataProcessor:
    def processRawDataToConsole(self, logger: IDataLogger) -> None:
        pass

    def processSummaryToConsole(self, logger: IDataLogger) -> None:
        pass

    def processPlotToConsole(self, logger: IDataLogger) -> None:
        pass

    def processRawDataToFile(self, logger: IDataLogger, folderName: str, append: bool) -> None:
        pass

    def processSummaryToFile(self, logger: IDataLogger, folderName: str) -> None:
        pass

    def processPlotToFile(self, logger: IDataLogger, folderName: str) -> None:
        pass