from typing import List
from core import AbstractGameState
from core import AbstractPlayer
from core.actions import AbstractAction
from core.interfaces import IPrintable

class HumanConsolePlayer(AbstractPlayer):
    def __init__(self):
        super().__init__(None, "HumanConsolePlayer")

    def _get_action(self, observation: AbstractGameState, actions: List[AbstractAction]) -> AbstractAction:
        if isinstance(observation, IPrintable):
            observation.print_to_console()

        for i, action in enumerate(actions):
            if action is not None:
                print(f"Action {i}: {action.get_string(observation)}")
            else:
                print("Null action")

        print("Type the index of your desired action:")
        invalid = True
        player_action = 0
        while invalid:
            try:
                player_action = int(input())
                if player_action < 0 or player_action >= len(actions):
                    print(f"Chosen index {player_action} is invalid. Choose any number in the range of [0, {len(actions)-1}]:")
                else:
                    invalid = False
            except ValueError:
                print("Please enter a valid integer.")

        return actions[player_action]

    def register_updated_observation(self, observation: AbstractGameState) -> None:
        print("No actions available. End turn by pressing any key...")
        input()

    def copy(self) -> 'HumanConsolePlayer':
        return self