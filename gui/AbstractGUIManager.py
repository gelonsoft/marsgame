from typing import Set
from core import AbstractGameState, AbstractPlayer, CoreConstants, Game
from core.actions import AbstractAction
from players.human import ActionController
from evaluation.listeners import IGameListener
from evaluation.metrics import Event

class AbstractGUIManager:
    def __init__(self, parent, game: Game, ac: ActionController, human: Set[int]):
        self.ac = ac
        self.max_action_space = self.getMaxActionSpace()
        self.parent = parent
        self.game = game
        self.human_player_ids = human
        
        # Initialize history tracking
        self.history_perspective = set()
        self.history = []
        self.actions_at_last_update = 0
    
    def getMaxActionSpace(self) -> int:
        raise NotImplementedError()
    
    def _update(self, player: AbstractPlayer, gameState: AbstractGameState):
        raise NotImplementedError()
    
    def updateActionButtons(self, player: AbstractPlayer, gameState: AbstractGameState):
        if (gameState.getGameStatus() == CoreConstants.GameResult.GAME_ONGOING and 
            hasattr(self, 'action_buttons') and self.action_buttons is not None):
            
            actions = player.getForwardModel().computeAvailableActions(
                gameState, 
                gameState.getCoreGameParameters().action_space
            )
            
            for i in range(min(len(actions), self.max_action_space)):
                self.action_buttons[i].setVisible(True)
                self.action_buttons[i].setButtonAction(actions[i], gameState)
                self.action_buttons[i].setBackground("white")
            
            for i in range(len(actions), len(self.action_buttons)):
                self.action_buttons[i].setVisible(False)
                self.action_buttons[i].setButtonAction(None, "")
    
    def createActionHistoryPanel(self, width: int, height: int, perspective_set: Set[int]):
        self.history_perspective = perspective_set
        if not perspective_set:
            return
            
        # Create game listener for action events
        self.game.addListener(IGameListener(
            onEvent=lambda event: self._handle_game_event(event),
            report=lambda: None,
            setGame=lambda game: None,
            getGame=lambda: None
        ))
    
    def _handle_game_event(self, event: Event):
        if event.type == Event.GameEvent.ACTION_CHOSEN:
            self.history.append(
                f"Player {event.state.getCurrentPlayer()} : " +
                event.action.getString(self.game.getGameState(), self.history_perspective)
            )
        elif event.type == Event.GameEvent.GAME_EVENT:
            self.history.append(event.action.toString())
        elif event.type == Event.GameEvent.GAME_OVER:
            for i in range(event.state.getNPlayers()):
                self.history.append(
                    f"Player {i} finishes at position {event.state.getOrdinalPosition(i)} " +
                    f"with score: {event.state.getGameScore(i)}"
                )
    
    def updateGameStateInfo(self, gameState: AbstractGameState):
        if not self.history_perspective:
            self.history = gameState.getHistoryAsText()
            
        if len(self.history) > self.actions_at_last_update:
            self.actions_at_last_update = len(self.history)
            
        # In Streamlit, we would update the displayed info here
        # For this base class, we just provide the structure
    
    def update(self, player: AbstractPlayer, gameState: AbstractGameState, showActions: bool):
        self.updateGameStateInfo(gameState)
        self._update(player, gameState)
        
        if showActions:
            self.updateActionButtons(player, gameState)
        else:
            self.resetActionButtons()
    
    def resetActionButtons(self):
        if hasattr(self, 'action_buttons') and self.action_buttons is not None:
            for button in self.action_buttons:
                button.setVisible(False)
                button.setButtonAction(None, "")

class ActionButton:
    def __init__(self, ac: ActionController, highlights=None, 
                 on_action_selected=None, on_mouse_enter=None, on_mouse_exit=None):
        self.ac = ac
        self.highlights = highlights
        self.action = None
        self.action_buttons = None
        self.on_action_selected = on_action_selected
        self.on_mouse_enter = on_mouse_enter
        self.on_mouse_exit = on_mouse_exit
    
    def setButtonAction(self, action: AbstractAction, gameState_or_text):
        self.action = action
        if isinstance(gameState_or_text, AbstractGameState):
            self.setText(action.getString(gameState_or_text))
        else:
            self.setText(str(gameState_or_text))
    
    def setText(self, text: str):
        # In Streamlit, this would update the button text
        pass
    
    def setVisible(self, visible: bool):
        # In Streamlit, this would show/hide the button
        pass
    
    def setBackground(self, color: str):
        # In Streamlit, this would set the button color
        pass
    
    def informAllActionButtons(self, action_buttons):
        self.action_buttons = action_buttons
    
    def getButtonAction(self) -> AbstractAction:
        return self.action