from typing import List, Set, Optional, Any, Callable
from abc import ABC, abstractmethod
import random
from core.actions.LogEvent import LogEvent
from core.components.Area import Area
from core.components.Component import Component,IComponentContainer, PartialObservableDeck
from core.interfaces.IExtendedSequence import IExtendedSequence
from core.interfaces.IGamePhase import IGamePhase
from evaluation.listeners.IGameListener import IGameListener
from evaluation.metrics import Event
from games import GameType
from utilities import Pair
from utilities.ElapsedCpuChessTimer import ElapsedCpuChessTimer
from core import AbstractParameters, CoreConstants, CoreParameters


class AbstractGameState(ABC):
    """
    Contains all game state information.
    This is distinct from the Game, of which it is a component. The Game also controls the players in the game,
    and this information is not present in (and must not be present in) the AbstractGameState.
    """

    def __init__(self, game_parameters: 'AbstractParameters', n_players: int):
        self.game_parameters = game_parameters
        self.game_type = self._get_game_type()
        self.all_components = Area(-1, "All Components")
        self.tick = 0
        self.round_counter = 0
        self.turn_counter = 0
        self.turn_owner = 0
        self.first_player = 0
        self.n_players = n_players
        self.n_teams = n_players  # Default to one team per player
        self.listeners: List['IGameListener'] = []
        self.player_timer: List['ElapsedCpuChessTimer'] = [None] * n_players
        self.history: List[Pair[int, Any]] = []
        self.history_text: List[str] = []
        self.game_status = CoreConstants.GameResult.GAME_ONGOING
        self.player_results = [CoreConstants.GameResult.GAME_ONGOING] * n_players
        self.game_phase: Optional['IGamePhase'] = None
        self.actions_in_progress = []
        self.core_game_parameters = CoreParameters()
        self.game_id = 0
        self.rnd = random.Random(game_parameters.random_seed)
        self.redeterminisation_rnd = random.Random()

    @abstractmethod
    def _get_game_type(self) -> GameType:
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets variables initialized for this game state."""
        if seed is not None:
            self.game_parameters.random_seed = seed
        self.all_components = Area(-1, "All Components")
        self.game_status = CoreConstants.GameResult.GAME_ONGOING
        self.player_results = [CoreConstants.GameResult.GAME_ONGOING] * self.n_players
        self.history = []
        self.history_text = []
        self.player_timer = [None] * self.n_players
        self.tick = 0
        self.turn_owner = 0
        self.turn_counter = 0
        self.round_counter = 0
        self.first_player = 0
        self.actions_in_progress.clear()
        self.rnd = random.Random(self.game_parameters.random_seed)

    # Getters
    def get_core_game_parameters(self) -> 'CoreParameters':
        return self.core_game_parameters

    def get_game_status(self) -> CoreConstants.GameResult:
        return self.game_status

    def get_game_parameters(self) -> 'AbstractParameters':
        return self.game_parameters

    def get_n_players(self) -> int:
        return self.n_players

    def get_n_teams(self) -> int:
        return self.n_teams

    def get_team(self, player: int) -> int:
        return player  # Default to one team per player

    def get_current_player(self) -> int:
        return self.actions_in_progress[-1].get_current_player(self) if self.is_action_in_progress() else self.turn_owner

    def get_player_results(self) -> List[CoreConstants.GameResult]:
        return self.player_results

    def get_winners(self) -> Set[int]:
        return {i for i, result in enumerate(self.player_results) if result == CoreConstants.GameResult.WIN_GAME}

    def get_tied(self) -> Set[int]:
        return {i for i, result in enumerate(self.player_results) if result == CoreConstants.GameResult.DRAW_GAME}

    def get_game_phase(self) -> Optional[IGamePhase]:
        return self.game_phase

    def get_player_timer(self) -> List[ElapsedCpuChessTimer]:
        return self.player_timer

    def get_game_type(self) -> GameType:
        return self.game_type

    def set_history_at(self, index: int, action: Pair[int, Any]) -> None:
        self.history[index] = action

    def get_history(self) -> List[Pair[int, Any]]:
        return list(self.history)

    def get_history_as_text(self) -> List[str]:
        return list(self.history_text)

    def get_game_id(self) -> int:
        return self.game_id

    def get_round_counter(self) -> int:
        return self.round_counter

    def get_turn_counter(self) -> int:
        return self.turn_counter

    def get_turn_owner(self) -> int:
        return self.turn_owner

    def get_first_player(self) -> int:
        return self.first_player

    # Setters
    def set_core_game_parameters(self, core_game_parameters: 'CoreParameters') -> None:
        self.core_game_parameters = core_game_parameters

    def set_game_status(self, status: CoreConstants.GameResult) -> None:
        self.game_status = status

    def set_player_result(self, result: CoreConstants.GameResult, player_idx: int) -> None:
        self.player_results[player_idx] = result

    def set_game_phase(self, game_phase: IGamePhase) -> None:
        self.game_phase = game_phase

    def set_game_id(self, id: int) -> None:
        self.game_id = id

    def advance_game_tick(self) -> None:
        self.tick += 1

    def set_turn_owner(self, new_turn_owner: int) -> None:
        self.turn_owner = new_turn_owner

    def set_first_player(self, new_first_player: int) -> None:
        self.first_player = new_first_player
        self.turn_owner = new_first_player

    def get_rnd(self) -> random.Random:
        return self.rnd

    def add_listener(self, listener: 'IGameListener') -> None:
        if listener not in self.listeners:
            self.listeners.append(listener)

    def clear_listeners(self) -> None:
        self.listeners.clear()

    # Game state checks
    def is_not_terminal(self) -> bool:
        return self.game_status == CoreConstants.GameResult.GAME_ONGOING

    def is_not_terminal_for_player(self, player: int) -> bool:
        return self.player_results[player] == CoreConstants.GameResult.GAME_ONGOING and self.game_status == CoreConstants.GameResult.GAME_ONGOING

    def get_game_tick(self) -> int:
        return self.tick

    def get_component_by_id(self, id: int) -> Optional['Component']:
        c = self.all_components.get_component(id)
        if c is None:
            try:
                self.add_all_components()
                c = self.all_components.get_component(id)
            except Exception:
                pass  # Can crash from concurrent modifications if running with GUI
        return c

    def get_all_components(self) -> Area:
        self.add_all_components()
        return self.all_components

    def get_all_top_level_components(self) -> List[Component]:
        return self._get_all_components()

    def add_all_components(self) -> None:
        self.all_components.clear()
        self.all_components.put_components(self._get_all_components())

    def copy(self, player_id: int = -1) -> 'AbstractGameState':
        """Creates a copy of the game state, optionally reduced for a specific player's perspective."""
        s = self._copy(player_id)
        s.all_components = self.all_components.empty_copy()
        s.game_status = self.game_status
        s.player_results = self.player_results.copy()
        s.game_phase = self.game_phase
        s.core_game_parameters = self.core_game_parameters
        s.tick = self.tick
        s.n_players = self.n_players
        s.round_counter = self.round_counter
        s.turn_counter = self.turn_counter
        s.turn_owner = self.turn_owner
        s.first_player = self.first_player
        s.rnd = random.Random(self.redeterminisation_rnd.getrandbits(64))

        if not self.core_game_parameters.competition_mode:
            s.history = list(self.history)
            s.history_text = list(self.history_text)

        s.actions_in_progress = [a.copy() for a in self.actions_in_progress]
        s.player_timer = [timer.copy() for timer in self.player_timer]
        s.add_all_components()
        return s

    def record_action(self, action: Any, player: int) -> None:
        """Records an action in the game history."""
        self.history.append(Pair(player, action.copy()))
        self.history_text.append(f"Player {player} : {action.get_string(self)}")

    def log_event(self, event: 'Event.IGameEvent', event_text: Optional[str] = None, text_supplier: Optional[Callable[[], str]] = None) -> None:
        """Logs an event to listeners and history if configured."""
        if not self.listeners and not self.get_core_game_parameters().record_event_history:
            return
        if text_supplier is not None:
            event_text = text_supplier()
        log_action = LogEvent(event_text or event.name())
        for listener in self.listeners:
            listener.on_event(Event.create_event(event, self, log_action))
        if self.get_core_game_parameters().record_event_history:
            self.record_history(event_text or event.name())

    def record_history(self, history_text: str) -> None:
        self.history_text.append(history_text)

    # Extended actions handling
    def current_action_in_progress(self) -> Optional[IExtendedSequence]:
        return self.actions_in_progress[-1] if self.actions_in_progress else None

    def get_queued_action(self, index: int) -> IExtendedSequence:
        if index < 0 or index >= len(self.actions_in_progress):
            raise IndexError(f"Index {index} out of bounds for action stack of size {len(self.actions_in_progress)}")
        return self.actions_in_progress[index]

    def is_action_in_progress(self) -> bool:
        self.check_actions_in_progress()
        return bool(self.actions_in_progress)

    def set_action_in_progress(self, action: IExtendedSequence) -> None:
        if action is not None:
            self.actions_in_progress.append(action)

    def check_actions_in_progress(self) -> None:
        while self.actions_in_progress:
            top_of_stack = self.actions_in_progress[-1]
            if top_of_stack.execution_complete(self):
                self.actions_in_progress.pop()
                if self.actions_in_progress:
                    self.actions_in_progress[-1].after_removal_from_queue(self, top_of_stack)
            else:
                break

    def get_actions_in_progress(self) -> List[IExtendedSequence]:
        return self.actions_in_progress

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def _get_all_components(self) -> List[Component]:
        pass

    @abstractmethod
    def _copy(self, player_id: int) -> 'AbstractGameState':
        pass

    @abstractmethod
    def _get_heuristic_score(self, player_id: int) -> float:
        pass

    @abstractmethod
    def get_game_score(self, player_id: int) -> float:
        pass

    def get_tiebreak(self, player_id: int, tier: int) -> float:
        return float('inf')

    def get_tiebreak_levels(self) -> int:
        return 5

    def get_ordinal_position(self, player_id: int, 
                            score_function: Optional[Callable[[int], float]] = None,
                            tiebreak_function: Optional[Callable[[int, int], float]] = None) -> int:
        if score_function is None:
            score_function = self.get_game_score
        if tiebreak_function is None:
            tiebreak_function = self.get_tiebreak

        ordinal = 1
        player_score = score_function(player_id)
        for i in range(self.n_players):
            other_score = score_function(i)
            if other_score > player_score:
                ordinal += 1
            elif other_score == player_score and tiebreak_function is not None and tiebreak_function(i, 1) != float('inf'):
                tier = 1
                while tier <= self.get_tiebreak_levels():
                    other_tiebreak = tiebreak_function(i, tier)
                    player_tiebreak = tiebreak_function(player_id, tier)
                    if other_tiebreak == player_tiebreak:
                        tier += 1
                    else:
                        if other_tiebreak > player_tiebreak:
                            ordinal += 1
                        break
        return ordinal

    def _get_unknown_components_ids(self, player_id: int) -> List[int]:
        return []

    def unknown_components(self, container: Component.IComponentContainer, player: int) -> List[int]:
        ret_value = []
        if isinstance(container, 'PartialObservableDeck'):
            for i in range(container.get_size()):
                if not container.get_visibility_for_player(i, player):
                    ret_value.append(container.get(i).get_component_id())
        else:
            visibility_mode = container.get_visibility_mode()
            if visibility_mode == "VISIBLE_TO_ALL":
                pass
            elif visibility_mode == "HIDDEN_TO_ALL":
                ret_value.extend([c.get_component_id() for c in container.get_components()])
            elif visibility_mode == "VISIBLE_TO_OWNER":
                if container.get_owner_id() != player:
                    ret_value.extend([c.get_component_id() for c in container.get_components()])
            elif visibility_mode == "TOP_VISIBLE_TO_ALL":
                ret_value.extend([c.get_component_id() for c in container.get_components()])
                if container.get_components():
                    ret_value.remove(container.get_components()[0].get_component_id())
            elif visibility_mode == "BOTTOM_VISIBLE_TO_ALL":
                if container.get_components():
                    ret_value.append(container.get_components()[-1].get_component_id())
            elif visibility_mode == "MIXED_VISIBILITY":
                raise AssertionError("Mixed visibility mode requires custom implementation")

        for c in container.get_components():
            if isinstance(c, IComponentContainer):
                ret_value.extend(self.unknown_components(c, player))
        return ret_value

    def get_unknown_components_ids(self, player_id: int) -> List[int]:
        everything = self.get_all_top_level_components()
        ret_value = []
        for c in everything:
            if isinstance(c, IComponentContainer):
                ret_value.extend(self.unknown_components(c, player_id))
        ret_value.extend(self._get_unknown_components_ids(player_id))
        return ret_value

    @abstractmethod
    def _equals(self, other: Any) -> bool:
        pass

    def get_heuristic_score(self, player_id: int) -> float:
        return self._get_heuristic_score(player_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AbstractGameState):
            return False
        return (self.game_parameters == other.game_parameters and
                self.game_status == other.game_status and
                self.n_players == other.n_players and
                self.round_counter == other.round_counter and
                self.turn_counter == other.turn_counter and
                self.turn_owner == other.turn_owner and
                self.first_player == other.first_player and
                self.tick == other.tick and
                self.player_results == other.player_results and
                self.game_phase == other.game_phase and
                self.actions_in_progress == other.actions_in_progress and
                self._equals(other))

    def __hash__(self) -> int:
        result = hash((self.game_parameters, self.game_status, self.game_phase, tuple(self.actions_in_progress)))
        result = 31 * result + hash((self.tick, self.n_players, self.round_counter, self.turn_counter, 
                                    self.turn_owner, self.first_player))
        result = 31 * result + hash(tuple(self.player_results))
        return result

    def is_game_over(self) -> bool:
        return self.game_status == CoreConstants.GameResult.GAME_END

    def get_winner(self) -> int:
        if self.is_game_over():
            for player_id in range(self.n_players):
                if self.player_results[player_id] == CoreConstants.GameResult.WIN_GAME:
                    return player_id
        return -1