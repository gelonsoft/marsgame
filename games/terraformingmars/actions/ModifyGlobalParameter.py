class ModifyGlobalParameter(TMModifyCounter):
    """Action to modify global parameters like temperature, oxygen, etc."""
    
    def __init__(self, param: TMTypes.GlobalParameter = None, change: float = 0, 
                 free: bool = True, action_type: TMTypes.ActionType = None,
                 cost_resource: TMTypes.Resource = None, cost: int = 0):
        super().__init__(action_type, -1, change, free)
        self.param = param
        if param:
            self.requirements.add(CounterRequirement(param.name, -1, True))
        if cost_resource and cost > 0:
            self.set_action_cost(cost_resource, cost, -1)
    
    def _execute(self, gs: TMGameState) -> bool:
        # When global parameters change, TR is increased
        counter = gs.get_global_parameters().get(self.param)
        if self.counter_id == -1:
            self.counter_id = counter.get_component_id()
        
        if ((self.change > 0 and not counter.is_maximum()) or 
            (self.change < 0 and not counter.is_minimum())):
            # Check persisting global param effects for all players
            for i in range(gs.get_n_players()):
                for effect in gs.get_player_persisting_effects()[i]:
                    if isinstance(effect, GlobalParameterEffect):
                        effect.execute(gs, self, i)
        
        return super()._execute(gs)
    
    def get_string(self, game_state: AbstractGameState) -> str:
        return f"Modify global parameter {self.param} by {self.change}"
    
    def __str__(self) -> str:
        return f"Modify global parameter {self.param} by {self.change}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ModifyGlobalParameter):
            return False
        return super().__eq__(other) and self.param == other.param
    
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.param))
    
    def _copy(self) -> 'ModifyGlobalParameter':
        copy_obj = ModifyGlobalParameter(self.param, self.change, self.free_action_point)
        copy_obj.counter_id = self.counter_id
        return copy_obj