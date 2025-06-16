from typing import Any, Dict, List
import re

from evaluation.listeners.FeatureListener import FeatureListener
from evaluation.listeners.StateFeatureListener import StateFeatureListener
from evaluation.loggers import FileStatsLogger

class ActionFeatureListener(FeatureListener):
    def __init__(self, psi, phi, frequency, includeActionsNotTaken: bool, fileName: str):
        super().__init__(frequency, True)
        if psi is None:
            raise AssertionError("Action Features must be provided and cannot be null")
        self.psiFn = psi
        self.phiFn = phi
        self.includeActionsNotTaken = includeActionsNotTaken
        self.logger = FileStatsLogger(fileName)
        self.actionValues = {}
        self.cachedPhi = None
    
    def names(self) -> List[str]:
        psiNames = self.psiFn.names()
        phiNames = self.phiFn.names() if self.phiFn else []
        return phiNames + psiNames
    
    def extractFeatureVector(self, action, state, perspectivePlayer: int) -> List[float]:
        phi = self.cachedPhi if self.cachedPhi is not None else (
            self.phiFn.featureVector(state, perspectivePlayer) if self.phiFn else []
        )
        psi = self.psiFn.featureVector(action, state, perspectivePlayer)
        return phi + psi
    
    def processStateWithTargets(self, state, action, targets: Dict[str, Dict[Any, float]]):
        self.actionValues = targets
        self.processState(state, action)
    
    def processState(self, state, action):
        if action is None:
            return
        
        self.cachedPhi = None
        availableActions = self.game.getForwardModel().computeAvailableActions(state)
        
        if len(availableActions) == 1:
            return
        
        if not self.actionValues:
            av = {a: 0.0 for a in availableActions}
            av[action] = 1.0
            self.actionValues["CHOSEN"] = av
        
        p = state.getCurrentPlayer()
        phi = self.extractFeatureVector(action, state, p)
        self.currentData.append(StateFeatureListener.LocalDataWrapper(p, phi, state, self.getActionScores(action)))
        
        if self.includeActionsNotTaken:
            for alternativeAction in availableActions:
                if alternativeAction == action:
                    continue
                phi = self.extractFeatureVector(alternativeAction, state, p)
                self.currentData.append(StateFeatureListener.LocalDataWrapper(p, phi, state, self.getActionScores(alternativeAction)))
        
        self.actionValues.clear()
    
    def getActionScores(self, action) -> Dict[str, float]:
        retValue = {}
        for key, actionMap in self.actionValues.items():
            retValue[key] = actionMap.get(action)
        
        if not retValue:
            raise AssertionError(f"Action {action} not found in action values map")
        return retValue
    
    def injectAgentAttributes(self, raw: str) -> str:
        psi_class = self.psiFn.__class__.__module__ + "." + self.psiFn.__class__.__name__
        phi_class = (self.phiFn.__class__.__module__ + "." + self.phiFn.__class__.__name__ 
                    if self.phiFn else "NONE")
        return raw.replace(re.escape("*PSI*"), psi_class).replace(re.escape("*PHI*"), phi_class)