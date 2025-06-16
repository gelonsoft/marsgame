class CompoundAction(TMAction):
    """Action that executes multiple actions in sequence."""
    
    def __init__(self, player: int = -1, actions: List[TMAction] = None,
                 action_type: TMTypes.ActionType = None, cost: int = 0):
        super().__init__(action_type if action_type else TMTypes.ActionType.COMPOUND, player, action_type is None)
        self.actions = actions or []
        
        if cost > 0:
            self.set_action_cost(TMTypes.Resource.MEGA_CREDIT, cost, -1)
        
        for action in self.actions:
            self.requirements.add(PlayableActionRequirement(action))
    
    def _execute(self, game_state: TMGameState) -> bool:
        success = True
        card = None
        if self.get_card_id() != -1:
            card = game_state.get_component_by_id(self.get_card_id())
        
        for action in self.actions:
            if card:
                # Reset flag to allow all actions to execute
                card.action_played = False
            action.player = self.player
            action.set_card_id(self.get_card_id())
            success &= action.execute(game_state)
        
        return success
    
    def _copy(self) -> 'CompoundAction':
        actions_copy = [action.copy() if action else None for action in self.actions]
        return CompoundAction(self.player, actions_copy)
    
    def get_string(self, game_state: AbstractGameState) -> str:
        if not self.actions:
            return "(no actions)"
        
        s = ""
        for action in self.actions:
            s += action.get_string(game_state) + " and "
        return s[:-5]  # Remove last " and "
    
    def __str__(self) -> str:
        if not self.actions:
            return "(no actions)"
        
        s = ""
        for action in self.actions:
            s += str(action) + " and "
        return s[:-5]  # Remove last " and "
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CompoundAction):
            return False
        return super().__eq__(other) and self.actions == other.actions
    
    def __hash__(self) -> int:
        return hash((super().__hash__(), tuple(self.actions)))
    