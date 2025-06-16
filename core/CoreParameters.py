from core import AbstractParameters
from core.actions import ActionSpace
from evaluation.optimisation import TunableParameters
from typing import Any, List
from enum import Enum
import copy

class CoreParameters(TunableParameters):
    def __init__(self):
        super().__init__()
        self.verbose = False
        self.recordEventHistory = True  # include in history text game events
        self.partialObservable = True
        self.competitionMode = False
        self.disqualifyPlayerOnIllegalActionPlayed = False
        self.disqualifyPlayerOnTimeout = False
        self.alwaysDisplayFullObservable = False
        self.alwaysDisplayCurrentPlayer = False
        self.frameSleepMS = 100
        
        # Action space type for this game
        self.actionSpace = ActionSpace(ActionSpace.Structure.Flat, ActionSpace.Flexibility.Default, ActionSpace.Context.Dependent)
        
        self.add_tunable_parameter("verbose", self.verbose, [False, True])
        self.add_tunable_parameter("recordEventHistory", self.recordEventHistory, [False, True])
        self.add_tunable_parameter("partial observable", self.partialObservable, [False, True])
        self.add_tunable_parameter("competition mode", self.competitionMode, [False, True])
        self.add_tunable_parameter("disqualify player on illegal action played", 
                                  self.disqualifyPlayerOnIllegalActionPlayed, [False, True])
        self.add_tunable_parameter("disqualify player on timeout", 
                                  self.disqualifyPlayerOnTimeout, [False, True])
        self.add_tunable_parameter("always display full observable", 
                                  self.alwaysDisplayFullObservable, [False, True])
        self.add_tunable_parameter("always display current player", 
                                  self.alwaysDisplayCurrentPlayer, [False, True])
        self.add_tunable_parameter("frame sleep MS", self.frameSleepMS, [0, 100, 500, 1000, 5000])
        self.add_tunable_parameter("actionSpaceStructure", ActionSpace.Structure.Default, 
                                  list(ActionSpace.Structure))
        self.add_tunable_parameter("actionSpaceFlexibility", ActionSpace.Flexibility.Default, 
                                  list(ActionSpace.Flexibility))
        self.add_tunable_parameter("actionSpaceContext", ActionSpace.Context.Default, 
                                  list(ActionSpace.Context))
    
    def _copy(self) -> 'AbstractParameters':
        return CoreParameters()
    
    def _equals(self, o: object) -> bool:
        if self is o:
            return True
        if not isinstance(o, CoreParameters):
            return False
        if not super().equals(o):
            return False
        that = o
        return (self.verbose == that.verbose and 
                self.recordEventHistory == that.recordEventHistory and 
                self.partialObservable == that.partialObservable and 
                self.competitionMode == that.competitionMode and 
                self.disqualifyPlayerOnIllegalActionPlayed == that.disqualifyPlayerOnIllegalActionPlayed and 
                self.disqualifyPlayerOnTimeout == that.disqualifyPlayerOnTimeout and 
                self.alwaysDisplayFullObservable == that.alwaysDisplayFullObservable and 
                self.alwaysDisplayCurrentPlayer == that.alwaysDisplayCurrentPlayer and 
                self.frameSleepMS == that.frameSleepMS and 
                self.actionSpace == that.actionSpace)
    
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.verbose, self.recordEventHistory, 
                     self.partialObservable, self.competitionMode, 
                     self.disqualifyPlayerOnIllegalActionPlayed, 
                     self.disqualifyPlayerOnTimeout, 
                     self.alwaysDisplayFullObservable, 
                     self.alwaysDisplayCurrentPlayer, 
                     self.frameSleepMS, 
                     self.actionSpace))
    
    def instantiate(self) -> Any:
        return None
    
    def _reset(self) -> None:
        self.verbose = self.get_parameter_value("verbose")
        self.recordEventHistory = self.get_parameter_value("recordEventHistory")
        self.partialObservable = self.get_parameter_value("partial observable")
        self.competitionMode = self.get_parameter_value("competition mode")
        self.disqualifyPlayerOnIllegalActionPlayed = self.get_parameter_value("disqualify player on illegal action played")
        self.disqualifyPlayerOnTimeout = self.get_parameter_value("disqualify player on timeout")
        self.alwaysDisplayFullObservable = self.get_parameter_value("always display full observable")
        self.alwaysDisplayCurrentPlayer = self.get_parameter_value("always display current player")
        self.frameSleepMS = int(str(self.get_parameter_value("frame sleep MS")))
        self.actionSpace = ActionSpace(
            self.get_parameter_value("actionSpaceStructure"),
            self.get_parameter_value("actionSpaceFlexibility"),
            self.get_parameter_value("actionSpaceContext")
        )