from typing import Optional
from core.components import Counter
from games.terraformingmars.actions.TMAction import TMAction
from games.terraformingmars.TMTypes import ActionType, StandardProject
from games.terraformingmars.components.GlobalParameter import GlobalParameter

class TMModifyCounter(TMAction):
    def __init__(self, 
                 counter_id: int = -1, 
                 change: float = 0.0, 
                 free: bool = False,
                 action_type: Optional[ActionType] = None,
                 standard_project: Optional[StandardProject] = None):
        
        # Initialize based on which constructor parameters are provided
        if action_type is not None:
            super().__init__(action_type=action_type, player=-1, free=free)
        elif standard_project is not None:
            super().__init__(standard_project=standard_project, player=-1, free=free)
        else:
            super().__init__(player=-1, free=free)
            
        self.counter_id = counter_id
        self.change = change

    def _execute(self, gs) -> bool:
        c = gs.get_component_by_id(self.counter_id)
        
        # Handle null case for solo play
        if gs.get_n_players() == 1 and c is None:
            return True
            
        # Special handling for GlobalParameter
        if isinstance(c, GlobalParameter):
            return c.increment(int(self.change), gs)
            
        # Normal counter increment
        return c.increment(int(self.change))

    def _copy(self) -> 'TMModifyCounter':
        return TMModifyCounter(
            counter_id=self.counter_id,
            change=self.change,
            free=self.free_action_point
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, TMModifyCounter):
            return False
        return (super().__eq__(other) and \
               (self.counter_id == other.counter_id) and \
               (self.change == other.change))

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.counter_id, self.change))

    def get_string(self, game_state) -> str:
        component = game_state.get_component_by_id(self.counter_id)
        return f"Modify counter {component.get_component_name()} by {self.change}"

    def __str__(self) -> str:
        return f"Modify counter {self.counter_id} by {self.change}"