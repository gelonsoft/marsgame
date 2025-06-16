from abc import ABC, abstractmethod
from typing import List, Optional
import random

from core.CoreConstants import GameResult, DefaultGamePhase
from core.actions.AbstractAction import AbstractAction
from core.actions import ActionSpace
from core.actions import DoNothing
from core.interfaces import IPlayerDecorator
from utilities import ActionTreeNode
from utilities import ElapsedCpuChessTimer


class AbstractForwardModel(ABC):
    """
    Abstract base class for forward models in game implementations.
    Handles game state transitions, action validation, and player decorators.
    """
    
    def __init__(self, decorators: Optional[List[IPlayerDecorator]] = None, 
                 player_id: int = -1):
        """
        Initialize the forward model.
        
        Args:
            decorators: List of player decorators to modify available actions
            player_id: ID of the decision player
        """
        self.root: Optional[ActionTreeNode] = None
        self.leaves: List[ActionTreeNode] = []
        
        # Decorator modify (restrict) the actions available to the player.
        # This enables the Forward Model to be passed to the decision algorithm (e.g. MCTS), and ensure that any
        # restrictions are applied to the actions available to the player during search, and not just
        # in the main game loop.
        self.decorators = decorators.copy() if decorators else []
        self.decision_player_id = player_id

    def abstract_setup(self, first_state) -> None:
        """
        Combines both super class and sub class setup methods. Called from the game loop.
        
        Args:
            first_state: Initial game state to be set up
        """
        first_state.game_status = GameResult.GAME_ONGOING
        first_state.player_results = [GameResult.GAME_ONGOING] * first_state.get_n_players()
        first_state.game_phase = DefaultGamePhase.MAIN
        first_state.player_timer = []
        
        for i in range(first_state.get_n_players()):
            timer = ElapsedCpuChessTimer(
                first_state.game_parameters.thinking_time_mins,
                first_state.game_parameters.increment_action_s,
                first_state.game_parameters.increment_turn_s,
                first_state.game_parameters.increment_round_s,
                first_state.game_parameters.increment_milestone_s
            )
            first_state.player_timer.append(timer)

        self._setup(first_state)
        first_state.add_all_components()

    # Methods to be implemented by subclasses, unavailable to AI players

    @abstractmethod
    def _setup(self, first_state) -> None:
        """
        Performs initial game setup according to game rules:
        - sets up decks and shuffles
        - gives player cards
        - places tokens on boards
        etc.
        
        Args:
            first_state: The state to be modified to the initial game state
        """
        pass

    @abstractmethod
    def _next(self, current_state, action: AbstractAction) -> None:
        """
        Applies the given action to the game state and executes any other game rules. Steps to follow:
        - execute player action
        - execute any game rules applicable
        - check game over conditions, and if any trigger, set the game_status and player_results variables
          appropriately (and return)
        - move to the next player where applicable
        
        Args:
            current_state: Current game state, to be modified by the action
            action: Action requested to be played by a player
        """
        pass

    @abstractmethod
    def _compute_available_actions(self, game_state) -> List[AbstractAction]:
        """
        Calculates the list of currently available actions, possibly depending on the game phase.
        
        Args:
            game_state: Current game state
            
        Returns:
            List of available AbstractAction objects
        """
        pass

    def _compute_available_actions_with_space(self, game_state, action_space: ActionSpace) -> List[AbstractAction]:
        """
        Calculates available actions with a specific action space.
        Default implementation delegates to _compute_available_actions.
        
        Args:
            game_state: Current game state
            action_space: Action space to constrain actions
            
        Returns:
            List of available AbstractAction objects
        """
        return self._compute_available_actions(game_state)

    @abstractmethod
    def end_player_turn(self, state) -> None:
        """
        Handles end of player turn logic.
        
        Args:
            state: Current game state
        """
        pass

    def illegal_action_played(self, game_state, action: AbstractAction) -> None:
        """
        Current player tried to play an illegal action.
        Subclasses can override for their own behavior.
        
        Args:
            game_state: Game state in which illegal action was attempted
            action: Action that was played illegally
        """
        self.disqualify_or_random_action(
            game_state.core_game_parameters.disqualify_player_on_illegal_action_played,
            game_state
        )

    def disqualify_or_random_action(self, flag: bool, game_state) -> AbstractAction:
        """
        Either disqualify (automatic loss and no more playing), or play a random action for the player instead.
        
        Args:
            flag: Boolean to check if player should be disqualified, or random action should be played
            game_state: Current game state
            
        Returns:
            The action that was taken (DoNothing if disqualified, or random action)
        """
        if flag:
            game_state.set_player_result(GameResult.DISQUALIFY, game_state.get_current_player())
            self.end_player_turn(game_state)
            return DoNothing()
        else:
            possible_actions = self.compute_available_actions(game_state)
            rng = random.Random(game_state.get_game_parameters().get_random_seed())
            random_action_idx = rng.randint(0, len(possible_actions) - 1)
            random_action = possible_actions[random_action_idx]
            self.next(game_state, random_action)
            return random_action

    # Public API for AI players

    def setup(self, game_state) -> None:
        """
        Sets up the given game state for game start according to game rules, with a new random seed.
        
        Args:
            game_state: Game state to be modified
        """
        game_state.reset()
        self.abstract_setup(game_state)

    def next(self, current_state, action: Optional[AbstractAction]) -> None:
        """
        Applies the given action to the game state and executes any other game rules.
        
        Args:
            current_state: Current game state, to be modified by the action
            action: Action requested to be played by a player
        """
        if action is not None:
            player = current_state.get_current_player()
            current_state.record_action(action, player)
            self._next(current_state, action)
        else:
            if current_state.core_game_parameters.verbose:
                print("Invalid action.")
            self.illegal_action_played(current_state, action)
        current_state.advance_game_tick()

    def compute_available_actions(self, game_state) -> List[AbstractAction]:
        """
        Computes the available actions and updates the game state accordingly.
        
        Args:
            game_state: Game state to update with the available actions
            
        Returns:
            The list of actions available
        """
        return self.compute_available_actions_with_space(
            game_state, 
            game_state.core_game_parameters.action_space
        )

    def compute_available_actions_with_space(self, game_state, action_space: ActionSpace) -> List[AbstractAction]:
        """
        Computes available actions with a specific action space.
        
        Args:
            game_state: Game state to compute actions for
            action_space: Action space to use
            
        Returns:
            List of available actions after applying decorators
        """
        # If there is an action in progress (see IExtendedSequence), then delegate to that
        if game_state.is_action_in_progress():
            ret_value = game_state.actions_in_progress[-1]._compute_available_actions(game_state, action_space)
        elif action_space is not None and not action_space.is_default():
            ret_value = self._compute_available_actions_with_space(game_state, action_space)
        else:
            ret_value = self._compute_available_actions(game_state)

        # Then apply Decorators regardless of source of actions
        for decorator in self.decorators:
            if decorator.decision_player_only() and game_state.get_current_player() != self.decision_player_id:
                continue
            ret_value = decorator.action_filter(game_state, ret_value)
        
        return ret_value

    def end_game(self, gs) -> None:
        """
        Performs any end of game computations, as needed.
        This should not normally need to be overridden - but can be. For example if a game is purely co-operative
        or has an insta-win situation without the concept of a game score.
        The last thing to be called in the game loop, after the game is finished.
        
        Args:
            gs: Game state to finalize
        """
        gs.set_game_status(GameResult.GAME_END)
        
        # If we have more than one person in Ordinal position of 1, then this is a draw
        first_place_count = sum(1 for p in range(gs.get_n_players()) if gs.get_ordinal_position(p) == 1)
        drawn = first_place_count > 1
        
        for p in range(gs.get_n_players()):
            ordinal_pos = gs.get_ordinal_position(p)
            if ordinal_pos == 1 and drawn:
                gs.set_player_result(GameResult.DRAW_GAME, p)
            elif ordinal_pos == 1:
                gs.set_player_result(GameResult.WIN_GAME, p)
            else:
                gs.set_player_result(GameResult.LOSE_GAME, p)
        
        if gs.get_core_game_parameters().verbose:
            print(gs.get_player_results())

    def add_player_decorator(self, decorator: IPlayerDecorator) -> None:
        """
        Add a player decorator to modify available actions.
        
        Args:
            decorator: Decorator to add
        """
        self.decorators.append(decorator)

    def clear_player_decorators(self) -> None:
        """
        Clear all player decorators.
        """
        self.decorators.clear()