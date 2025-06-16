from abc import abstractmethod
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

from evaluation.metrics.listeners import IGameListener

@dataclass
class LocalDataWrapper:
    player: int
    gameTurn: int
    gameRound: int
    currentScore: float
    actionScores: List[float]
    actionScoreNames: List[str]
    array: List[float]

class FeatureListener(IGameListener):
    def __init__(self, frequency, currentPlayerOnly: bool):
        self.currentData = []
        self.frequency = frequency
        self.currentPlayerOnly = currentPlayerOnly
        self.logger = None
        self.game = None
    
    def setLogger(self, logger):
        self.logger = logger
    
    def onEvent(self, event):
        if event.type == self.frequency and self.frequency != "GAME_OVER":
            self.processState(event.state, event.action)
        
        if event.type == "GAME_OVER":
            self.processState(event.state, None)
            self.writeDataWithStandardHeaders(event.state)
    
    def setOutputDirectory(self, *nestedDirectories) -> bool:
        if hasattr(self.logger, 'setOutPutDirectory'):
            self.logger.setOutPutDirectory(nestedDirectories)
        return True
    
    def writeDataWithStandardHeaders(self, state):
        totP = state.getNPlayers()
        finalScores = [state.getGameScore(p) for p in range(totP)]
        winLoss = [1.0 if r == "WIN_GAME" else 0.5 if r == "DRAW_GAME" else 0.0 
                  for r in state.getPlayerResults()]
        ordinal = [state.getOrdinalPosition(p) for p in range(totP)]
        finalRound = state.getRoundCounter()
        
        for record in self.currentData:
            data = {
                "GameID": float(state.getGameID()),
                "Player": float(record.player),
                "Round": float(record.gameRound),
                "Turn": float(record.gameTurn),
                "CurrentScore": record.currentScore,
                "PlayerCount": float(len(self.game.getPlayers())),
                "TotalRounds": finalRound,
                "TotalTurns": float(state.getTurnCounter()),
                "TotalTicks": float(state.getGameTick()),
                "Win": winLoss[record.player],
                "Ordinal": ordinal[record.player],
                "FinalScore": finalScores[record.player]
            }
            
            # Add feature vector values
            for i, val in enumerate(record.array):
                data[self.names()[i]] = val
            
            # Add action scores
            for name, score in zip(record.actionScoreNames, record.actionScores):
                data[name] = score
            
            # Calculate final score advantage
            bestOtherScore = max(finalScores[p] for p in range(totP) if p != record.player)
            data["FinalScoreAdv"] = finalScores[record.player] - bestOtherScore
            
            self.logger.record(data)
        
        self.logger.processDataAndNotFinish()
        self.currentData = []
    
    def report(self):
        self.logger.processDataAndFinish()
    
    def setGame(self, game):
        self.game = game
    
    def getGame(self):
        return self.game
    
    @abstractmethod
    def names(self) -> List[str]:
        pass
    
    @abstractmethod
    def extractFeatureVector(self, action, state, perspectivePlayer: int) -> List[float]:
        pass
    
    @abstractmethod
    def injectAgentAttributes(self, raw: str) -> str:
        pass
    
    def processState(self, state, action):
        if self.currentPlayerOnly and not state.isNotTerminal():
            p = state.getCurrentPlayer()
            phi = self.extractFeatureVector(action, state, p)
            self.currentData.append(LocalDataWrapper(
                p, phi, state, {}
            ))
        else:
            for p in range(state.getNPlayers()):
                phi = self.extractFeatureVector(action, state, p)
                self.currentData.append(LocalDataWrapper(
                    p, phi, state, {}
                ))