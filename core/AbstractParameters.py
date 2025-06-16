import time
from core.interfaces import IStateHeuristic, ITunableParameters
from evaluation.optimisation import TunableParameters
from games import GameType
from players.heuristics import NullHeuristic
from typing import Dict, List, Optional, Any

class AbstractParameters:
    def __init__(self):
        self.random_seed = int(time.time() * 1000)
        self.max_rounds = -1
        self.timeout_rounds = -1
        self.thinking_time_mins = 90
        self.increment_action_s = 0
        self.increment_turn_s = 0
        self.increment_round_s = 0
        self.increment_milestone_s = 0

    def _copy(self) -> 'AbstractParameters':
        raise NotImplementedError("Subclasses must implement this method")

    def _equals(self, o: object) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

    def get_random_seed(self) -> int:
        return self.random_seed

    def set_random_seed(self, random_seed: int) -> None:
        self.random_seed = random_seed

    def set_thinking_time_mins(self, thinking_time_mins: int) -> None:
        self.thinking_time_mins = thinking_time_mins

    def set_max_rounds(self, max_rounds: int) -> None:
        self.max_rounds = max_rounds

    def set_timeout_rounds(self, timeout_rounds: int) -> None:
        self.timeout_rounds = timeout_rounds

    def get_thinking_time_mins(self) -> int:
        return self.thinking_time_mins

    def get_increment_action_s(self) -> int:
        return self.increment_action_s

    def get_increment_turn_s(self) -> int:
        return self.increment_turn_s

    def get_increment_round_s(self) -> int:
        return self.increment_round_s

    def get_increment_milestone_s(self) -> int:
        return self.increment_milestone_s

    def get_max_rounds(self) -> int:
        return self.max_rounds

    def get_timeout_rounds(self) -> int:
        return self.timeout_rounds

    def copy(self) -> 'AbstractParameters':
        copy = self._copy()
        copy.random_seed = int(time.time() * 1000)
        return copy

    def randomize(self) -> None:
        if isinstance(self, ITunableParameters):
            import random
            rnd = random.Random(self.random_seed)
            for name in self.get_parameter_names():
                possible_values = self.get_possible_values(name)
                n_values = len(possible_values)
                random_choice = rnd.randint(0, n_values - 1)
                self.set_parameter_value(name, possible_values[random_choice])
            self._reset()
        else:
            print("Error: Not implementing the TunableParameters interface. Not randomizing")

    def reset(self) -> None:
        if isinstance(self, ITunableParameters):
            default_values = self.get_default_parameter_values()
            self.set_parameter_values(default_values)
        else:
            print("Error: Not implementing the TunableParameters interface. Not resetting.")

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, AbstractParameters):
            return False
        return (self.thinking_time_mins == o.thinking_time_mins and
                self.increment_action_s == o.increment_action_s and
                self.increment_turn_s == o.increment_turn_s and
                self.increment_round_s == o.increment_round_s and
                self.max_rounds == o.max_rounds and 
                self.timeout_rounds == o.timeout_rounds and
                self.increment_milestone_s == o.increment_milestone_s)

    def __hash__(self) -> int:
        return hash((self.thinking_time_mins, self.increment_action_s, self.increment_turn_s, 
                    self.increment_round_s, self.increment_milestone_s, self.max_rounds, 
                    self.timeout_rounds))

    @staticmethod
    def create_from_file(game: GameType, file_name: str) -> 'AbstractParameters':
        params = game.create_parameters(int(time.time() * 1000))
        if not file_name:
            return params
        if isinstance(params, TunableParameters):
            TunableParameters.load_from_json_file(params, file_name)
            return params
        else:
            raise AssertionError(f"JSON parameter initialisation not supported for {game}")

    def get_state_heuristic(self) -> IStateHeuristic:
        return NullHeuristic()