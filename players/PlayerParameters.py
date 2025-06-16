from core.interfaces import IStateHeuristic, IPlayerDecorator
from core.actions import ActionSpace
from evaluation.optimisation import TunableParameters
from typing import List, Optional
import time
from enum import Enum

class PlayerConstants(Enum):
    BUDGET_FM_CALLS = "BUDGET_FM_CALLS"
    # Add other constants as needed

class PlayerParameters(TunableParameters):
    def __init__(self):
        super().__init__()
        self.noise_epsilon: float = 1e-6

        # Budget settings
        self.budget_type: PlayerConstants = PlayerConstants.BUDGET_FM_CALLS
        self.budget: int = 4000
        # break_ms is the number of milliseconds prior to the end of the turn that the player will stop searching
        # this is intended mainly for competition situations, in which overrunning the time limit leads to disqualification.
        # setting break_ms to some number greater than zero then adds a safety margin
        self.break_ms: int = 0
        # reset_seed_each_game is a dangerous parameter. If true then the random seed will be reset at the start of each game.
        # otherwise the Random() object will be used from the old game, ensuring that we do not take exactly the same
        # set of actions
        self.reset_seed_each_game: bool = False

        # Heuristic
        self.game_heuristic: Optional[IStateHeuristic] = None

        # Action space type for this player
        self.action_space: ActionSpace = ActionSpace()
        self.decorator: Optional[IPlayerDecorator] = None

        self.add_tunable_parameter("budgetType", PlayerConstants.BUDGET_FM_CALLS, list(PlayerConstants))
        self.add_tunable_parameter("budget", 4000, [100, 300, 1000, 3000, 10000, 30000, 100000])
        self.add_tunable_parameter("breakMS", 0)
        self.add_tunable_parameter("actionSpaceStructure", ActionSpace.Structure.Default, list(ActionSpace.Structure))
        self.add_tunable_parameter("actionSpaceFlexibility", ActionSpace.Flexibility.Default, list(ActionSpace.Flexibility))
        self.add_tunable_parameter("actionSpaceContext", ActionSpace.Context.Default, list(ActionSpace.Context))
        self.add_tunable_parameter("randomSeed", int(time.time() * 1000))
        self.add_tunable_parameter("resetSeedEachGame", False)
        self.add_tunable_parameter("epsilon", 1e-6)
        self.add_tunable_parameter("actionRestriction", IPlayerDecorator)

    def _copy(self) -> 'PlayerParameters':
        params = PlayerParameters()
        # only need to copy fields that are not Tuned (those are done in the super class)
        params.game_heuristic = self.game_heuristic
        return params

    def _reset(self) -> None:
        self.set_random_seed(int(self.get_parameter_value("randomSeed")))
        self.budget = int(self.get_parameter_value("budget"))
        self.reset_seed_each_game = bool(self.get_parameter_value("resetSeedEachGame"))
        self.break_ms = int(self.get_parameter_value("breakMS"))
        self.noise_epsilon = float(self.get_parameter_value("epsilon"))
        self.budget_type = PlayerConstants(self.get_parameter_value("budgetType"))
        self.action_space = ActionSpace(
            ActionSpace.Structure(self.get_parameter_value("actionSpaceStructure")),
            ActionSpace.Flexibility(self.get_parameter_value("actionSpaceFlexibility")),
            ActionSpace.Context(self.get_parameter_value("actionSpaceContext"))
        )
        self.decorator = self.get_parameter_value("actionRestriction")

    def _equals(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, PlayerParameters):
            if self.game_heuristic is None and other.game_heuristic is None:
                return True
            if self.game_heuristic is None or other.game_heuristic is None:
                return False
            return self.game_heuristic.equals(other.game_heuristic)
        return False

    def instantiate(self) -> None:
        raise RuntimeError("PlayerParameters should not be instantiated directly.")