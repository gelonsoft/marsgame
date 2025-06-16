from core import AbstractGameState, AbstractParameters
from core.turnorders import TurnOrder
from evaluation.listeners import IGameListener

DEPRECATED_WARNING = ("This class is deprecated and should not be used for new development. "
                      "Instead, use AbstractGameStateWithTurnOrderV2")

class AbstractGameStateWithTurnOrder(AbstractGameState):
    def __init__(self, game_parameters: AbstractParameters, n_players: int):
        super().__init__(game_parameters, n_players)
        self._turn_order = self._create_turn_order(n_players)
        self.reset()

    @staticmethod
    def _create_turn_order(n_players: int) -> TurnOrder:
        raise NotImplementedError

    def reset(self):
        super().reset()
        self._turn_order.reset()

    @property
    def turn_order(self) -> TurnOrder:
        return self._turn_order

    def get_round_counter(self) -> int:
        return self._turn_order.get_round_counter()

    def get_turn_counter(self) -> int:
        return self._turn_order.get_turn_counter()

    def get_first_player(self) -> int:
        return self._turn_order.get_first_player()

    def get_n_players(self) -> int:
        return self._turn_order.n_players()

    def get_current_player(self, player_id: int) -> int:
        return self._turn_order.get_current_player(player_id)

    def add_listener(self, listener: IGameListener) -> None:
        self._turn_order.add_listener(listener)

    def clear_listeners(self) -> None:
        self._turn_order.clear_listeners()

    def set_turn_owner(self, new_turn_owner: int) -> None:
        self._turn_order.set_turn_owner(new_turn_owner)

    def set_first_player(self, new_first_player: int) -> None:
        self._turn_order.set_starting_player(new_first_player)

    def set_turn_order(self, turn_order: TurnOrder) -> None:
        self._turn_order = turn_order

    def _copy(self, player_id: int) -> 'AbstractGameStateWithTurnOrder':
        copied_game_state = self.__class__.__copy__(player_id)
        copied_game_state._turn_order = self._turn_order.copy()
        return copied_game_state

    def __eq__(self, other: 'AbstractGameStateWithTurnOrder') -> bool:
        return (
            super().__eq__(other) and
            self._turn_order == other._turn_order
        )

    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self._turn_order)
