from typing import List
from core import AbstractGameState
from core import AbstractPlayer
from core.actions import AbstractAction, DoNothing
from .ActionController import ActionController
import streamlit as st

class HumanGUIPlayer(AbstractPlayer):
    def __init__(self, ac: ActionController):
        super().__init__(None, "HumanGUIPlayer")
        self.ac = ac

    def _get_action(self, observation: AbstractGameState, actions: List[AbstractAction]) -> AbstractAction:
        try:
            # For Streamlit integration, we might want to display the available actions
            # and let the user select one through the GUI
            if not self.ac.has_action():
                # Display actions in Streamlit and wait for user selection
                action_idx = st.selectbox(
                    "Select your action:",
                    range(len(actions)),
                    format_func=lambda x: actions[x].get_string(observation)
                )
                self.ac.add_action(actions[action_idx])
            
            return self.ac.get_action()
        except Exception as e:
            # If interrupted, return DoNothing
            return DoNothing()

    def copy(self) -> 'HumanGUIPlayer':
        return self