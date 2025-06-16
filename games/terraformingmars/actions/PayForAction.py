class PayForAction(TMAction, IExtendedSequence):
    def __init__(self, player: int = -1, action: Optional[TMAction] = None):
        super().__init__(player=player, free=True)
        self.action = action
        self.cost_paid = 0
        self.stage = 0
        self.resources_to_pay_with: Optional[List] = None
        
        if action:
            self.set_action_cost(action.get_cost_resource(), abs(action.get_cost()), 
                               action.get_play_card_id())
            self.requirements.append(PlayableActionRequirement(action))
    
    def _execute(self, state: TMGameState) -> bool:
        gp = state.get_game_parameters()
        
        card = None
        cost = self.get_cost()
        if self.get_play_card_id() > -1:
            card = state.get_component_by_id(self.get_play_card_id())
            cost -= state.discount_card_cost(card, self.player)
        
        cost -= state.discount_action_type_cost(self.action, self.player)
        self.set_cost(cost)
        
        resources = state.can_player_transform(self.player, card, None, self.get_cost_resource())
        resources.add(self.get_cost_resource())
        
        self.resources_to_pay_with = list(resources)
        self.stage = 0
        self.cost_paid = 0
        
        if self.stage == len(self.resources_to_pay_with) - 1:
            actions = self._compute_available_actions(state)
            if len(actions) == 1:
                a = actions[0]
                s1 = a.execute(state)
                self.action.player = self.player
                if hasattr(self.action, 'cost_requirement') and self.action.cost_requirement in self.action.requirements:
                    self.action.requirements.remove(self.action.cost_requirement)
                s2 = self.action.execute(state)
                self.stage = len(self.resources_to_pay_with)
                return s1 and s2
            else:
                state.set_action_in_progress(self)
                return True
        else:
            state.set_action_in_progress(self)
            return True
    
    def _compute_available_actions(self, state: AbstractGameState) -> List[AbstractAction]:
        gs = state
        actions = []
        
        if self.stage >= len(self.resources_to_pay_with):
            actions.append(self.action)
            return actions
        
        res = self.resources_to_pay_with[self.stage]
        resources_remaining = set(self.resources_to_pay_with[self.stage + 1:])
        
        card = gs.get_component_by_id(self.get_play_card_id())
        rate = gs.get_resource_map_rate(res, self.get_cost_resource())
        sum_val = gs.player_resource_sum(self.player, card, resources_remaining, 
                                        self.get_cost_resource(), False)
        remaining = self.get_cost() - self.cost_paid - sum_val
        min_val = max(0, int(remaining / rate))
        max_val = min(gs.player_resources[self.player][res].get_value(), 
                     int((self.get_cost() - self.cost_paid) / rate))
        
        for i in range(min_val, max_val + 1):
            actions.append(ModifyPlayerResource(self.player, change=-i, resource=res, 
                                              production=False))
        
        if not actions:
            actions.append(self.action)
        
        return actions
    
    def get_current_player(self, state: AbstractGameState) -> int:
        return self.player
    
    def _after_action(self, state: AbstractGameState, action: AbstractAction):
        if not isinstance(action, ModifyPlayerResource):
            self.stage = len(self.resources_to_pay_with)
            self.action.player = self.player
            if hasattr(self, 'cost_requirement'):
                self.action.requirements.remove(self.cost_requirement)
            self.action.execute(state)
            return
        
        gs = state
        res = self.resources_to_pay_with[self.stage]
        self.cost_paid += abs(action.change) * gs.get_resource_map_rate(res, self.get_cost_resource())
        self.stage += 1
        
        if self.cost_paid >= self.get_cost():
            self.action.player = self.player
            if hasattr(self, 'cost_requirement'):
                self.action.requirements.remove(self.cost_requirement)
            self.action.execute(state)
            self.stage = len(self.resources_to_pay_with)
    
    def execution_complete(self, state: AbstractGameState) -> bool:
        return (self.stage == len(self.resources_to_pay_with) or 
                self.cost_paid == self.get_cost())
    
    def __str__(self) -> str:
        return f"Pay {self.get_cost()} {self.get_cost_resource()} for {self.action}"
    
    def get_string(self, game_state: AbstractGameState) -> str:
        return f"Pay {self.get_cost()} {self.get_cost_resource()} for {self.action.get_string(game_state)}"