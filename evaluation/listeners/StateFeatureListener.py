import re
from typing import Dict, List

from evaluation.listeners.FeatureListener import FeatureListener
from evaluation.loggers import FileStatsLogger

class StateFeatureListener(FeatureListener):
    def __init__(self, phi, frequency, currentPlayerOnly: bool, fileName: str):
        super().__init__(frequency, currentPlayerOnly)
        self.phiFn = phi
        self.logger = FileStatsLogger(fileName)
    
    def names(self) -> List[str]:
        return self.phiFn.names()
    
    def extractFeatureVector(self, action, state, perspectivePlayer: int) -> List[float]:
        return self.phiFn.featureVector(state, perspectivePlayer)
    
    def injectAgentAttributes(self, raw: str) -> str:
        return raw.replace(re.escape("*PHI*"), self.phiFn.__class__.__module__ + "." + self.phiFn.__class__.__name__)