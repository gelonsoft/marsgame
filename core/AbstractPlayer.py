from typing import List, Dict, Optional, Any
import random
from core.actions import AbstractAction
from core.interfaces import IPlayerDecorator
from evaluation.metrics import Event
from players import PlayerParameters
from core import AbstractForwardModel
from core import AbstractGameState

class AbstractPlayer:
    def __init__(self, params: Optional[PlayerParameters], name: str):
        self.player_id: int = -1  # Will be assigned by the game
        self.name = name
        self.rnd = random.Random()
        self.forward_model: Optional[AbstractForwardModel] = None
        self.parameters = params if params is not None else PlayerParameters()
        self.decorators: List[IPlayerDecorator] = []
        
        if self.parameters.decorator is not None:
            self.decorators.append(self.parameters.decorator)

    # Final methods (should not be overridden by subclasses)

    @property
    def player_id(self) -> int:
        return self._player_id

    @player_id.setter
    def player_id(self, value: int):
        self._player_id = value

    def set_name(self, name: str) -> None:
        self.name = name

    def get_action(self, game_state: AbstractGameState, observed_actions: List[AbstractAction]) -> AbstractAction:
        # Apply decorators to filter actions
        current_actions = observed_actions.copy()
        for decorator in self.decorators:
            current_actions = decorator.action_filter(game_state, current_actions)

        if not current_actions:
            raise AssertionError(f"No actions available for player {self}")

        if len(current_actions) == 1:
            action = current_actions[0]
        else:
            game_state.rnd = self.rnd
            action = self._get_action(game_state, current_actions)

        # Apply decorators to record decision
        for decorator in self.decorators:
            decorator.record_decision(game_state, action)

        return action

    def set_forward_model(self, model: AbstractForwardModel) -> None:
        self.forward_model = model
        for decorator in self.decorators:
            model.add_player_decorator(decorator)

    @property
    def forward_model(self) -> Optional[AbstractForwardModel]:
        return self._forward_model

    @forward_model.setter
    def forward_model(self, model: AbstractForwardModel):
        self._forward_model = model

    def add_decorator(self, decorator: IPlayerDecorator) -> None:
        self.decorators.append(decorator)

    def clear_decorators(self) -> None:
        self.decorators.clear()

    def remove_decorator(self, decorator: IPlayerDecorator) -> None:
        self.decorators.remove(decorator)

    def __str__(self) -> str:
        return self.name if self.name else self.__class__.__name__

    # Abstract methods (must be implemented by subclasses)
    def _get_action(self, game_state: AbstractGameState, possible_actions: List[AbstractAction]) -> AbstractAction:
        raise NotImplementedError("Subclasses must implement this method")

    def copy(self) -> 'AbstractPlayer':
        raise NotImplementedError("Subclasses must implement this method")

    # Optional methods (can be implemented by subclasses)
    def initialize_player(self, game_state: AbstractGameState) -> None:
        pass

    def finalize_player(self, game_state: AbstractGameState) -> None:
        pass

    def register_updated_observation(self, game_state: AbstractGameState) -> None:
        pass

    def on_event(self, event: Event) -> None:
        pass

    def get_decision_stats(self) -> Dict[AbstractAction, Dict[str, Any]]:
        return {}

    @property
    def parameters(self) -> PlayerParameters:
        return self._parameters

    @parameters.setter
    def parameters(self, params: PlayerParameters):
        self._parameters = params

    @property
    def rnd(self) -> random.Random:
        return self._rnd

    @rnd.setter
    def rnd(self, value: random.Random):
        self._rnd = value