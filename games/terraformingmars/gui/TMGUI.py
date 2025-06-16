import streamlit as st
from typing import Any, Set
from core import AbstractGameState, AbstractPlayer, CoreConstants, Game
from games.terraformingmars import TMGameState, TMTypes
from games.terraformingmars.components import TMCard
from gui import AbstractGUIManager
from players.human import ActionController

class TMGUI(AbstractGUIManager):
    def __init__(self, parent, game: Game, ac: ActionController, human_id: Set[int]):
        super().__init__(parent, game, ac, human_id)
        if game is None:
            return
            
        self.view = TMBoardView(self, game.getGameState())
        self.player_view = TMPlayerView(game.getGameState(), 0)
        
        self.focus_player = 0
        self.current_player_idx = 0
        self.focus_current_player = False
        
        self.last_action = None
        self.turn_order = None
        
        # Initialize Streamlit components
        st.set_page_config(layout="wide")
        self.init_ui()
        
    def init_ui(self):
        st.title("Terraforming Mars")
        
        # Main game area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display board view
            if hasattr(self.view, 'draw'):
                board_img = self.view.draw()
                st.image(board_img, use_column_width=True)
            
        with col2:
            # Player info
            st.header(f"Player {self.focus_player}")
            self.display_player_info()
            
            # Hand cards
            st.subheader("Hand")
            self.display_hand_cards()
            
            # Card choices
            st.subheader("Card Choices")
            self.display_card_choices()
            
        # Action buttons at bottom
        st.header("Actions")
        self.display_action_buttons()
        
    def display_player_info(self):
        if self.game and self.game.getGameState():
            gs = self.game.getGameState()
            st.write(f"Generation: {gs.getGeneration()}")
            st.write(f"Phase: {gs.getGamePhase()}")
            
            # Display resources
            st.write("Resources:")
            resources = gs.getPlayerResources()[self.focus_player]
            for res, val in resources.items():
                st.write(f"{res}: {val}")
    
    def display_hand_cards(self):
        if self.game and self.game.getGameState():
            gs = self.game.getGameState()
            hand = gs.getPlayerHands()[self.focus_player]
            
            cols = st.columns(4)
            for i, card in enumerate(hand.getComponents()):
                with cols[i % 4]:
                    st.image(self.get_card_image(card), caption=card.getComponentName())
    
    def display_card_choices(self):
        if self.game and self.game.getGameState():
            gs = self.game.getGameState()
            choices = gs.getPlayerCardChoice()[self.focus_player]
            
            cols = st.columns(4)
            for i, card in enumerate(choices.getComponents()):
                with cols[i % 4]:
                    st.image(self.get_card_image(card), caption=card.getComponentName())
    
    def display_action_buttons(self):
        if self.game and self.game.getGameState():
            gs = self.game.getGameState()
            player = self.game.getPlayers()[gs.getCurrentPlayer()]
            
            actions = player.getForwardModel().computeAvailableActions(gs)
            
            cols = st.columns(4)
            for i, action in enumerate(actions):
                with cols[i % 4]:
                    if st.button(str(action)):
                        self.ac.addAction(action)
    
    def get_card_image(self, card: TMCard) -> Any:
        # Placeholder for card image rendering
        #img = Surface((100, 150))
        #img.fill((255, 255, 255))
        #font = pygame.font.SysFont(None, 20)
        #text = font.render(card.getComponentName(), True, (0, 0, 0))
        #img.blit(text, (10, 10))
        #return img
        return None
    
    def getMaxActionSpace(self) -> int:
        return 500
    
    def _update(self, player: AbstractPlayer, gameState: AbstractGameState):
        if gameState is None:
            return
            
        gs = gameState
        
        if gs.getGameStatus() == CoreConstants.GameResult.GAME_END:
            self.display_game_end(gs)
            
        self.current_player_idx = gs.getCurrentPlayer()
        if self.focus_current_player:
            self.focus_player = self.current_player_idx
            
        # Update all views
        if hasattr(self, 'view'):
            self.view.update(gs)
        if hasattr(self, 'player_view'):
            self.player_view.update(gs)
            
        # Rerun Streamlit to update UI
        st.experimental_rerun()
    
    def display_game_end(self, gs: TMGameState):
        st.header("Game Over")
        
        win = -1
        for i in range(gs.getNPlayers()):
            if gs.getPlayerResults()[i] == CoreConstants.GameResult.WIN_GAME:
                win = i
        
        st.write(f"Winner: Player {win}")
        
        for i in range(gs.getNPlayers()):
            tr = gs.getPlayerResources()[i].get(TMTypes.Resource.TR).getValue()
            milestones = gs.countPointsMilestones(i)
            awards = gs.countPointsAwards(i)
            board = gs.countPointsBoard(i)
            cards = gs.countPointsCards(i)
            total = tr + milestones + awards + board + cards
            
            st.write(f"Player {i}: TR={tr}, Milestones={milestones}, Awards={awards}, Board={board}, Cards={cards}, Total={total}")

# Placeholder classes for views - would need proper implementation
class TMBoardView:
    def __init__(self, gui, game_state):
        self.gui = gui
        self.game_state = game_state
        
    def update(self, game_state):
        self.game_state = game_state
        
    def draw(self) -> Any:
        #img = Surface((800, 600))
        #img.fill((0, 0, 100))  # Blue background for space
        #return img
        return None

class TMPlayerView:
    def __init__(self, game_state, player_idx):
        self.game_state = game_state
        self.player_idx = player_idx
        
    def update(self, game_state):
        self.game_state = game_state