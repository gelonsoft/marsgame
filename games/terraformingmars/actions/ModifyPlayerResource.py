class ModifyPlayerResource(TMModifyCounter, IExtendedSequence):
    def __init__(self, player: int = -1, target_player: int = -1, change: float = 0, 
                 resource=None, production: bool = False, tag_to_count=None, 
                 tile_to_count=None, any_player: bool = False, opponents: bool = False, 
                 on_mars: bool = False, counter_resource=None, 
                 counter_resource_production: bool = False, free: bool = False):
        super().__init__(-1, change, free)
        self.resource = resource
        self.production = production
        self.target_player = target_player
        self.target_player_options: Optional[Set[int]] = None
        self.counter_resource = counter_resource
        self.counter_resource_production = counter_resource_production
        self.tag_to_count = tag_to_count
        self.tile_to_count = tile_to_count
        self.any = any_player
        self.opponents = opponents
        self.on_mars = on_mars
        self.complete = False
        self.player = player
        
        if change < 0 and production:
            self.requirements.append(ResourceRequirement(resource, abs(int(change)), 
                                                        production, target_player, -1))
    
    @classmethod
    def for_standard_project(cls, standard_project, cost: int, player: int, 
                           change: float, resource):
        instance = cls(player=player, change=change, resource=resource, production=True)
        instance.set_action_cost(TMTypes.Resource.MEGA_CREDIT, cost, -1)
        if change < 0:
            instance.requirements.append(ResourceRequirement(resource, abs(int(change)), 
                                                            True, player, -1))
        return instance
    
    def _execute(self, state: TMGameState) -> bool:
        if not self.complete and (self.target_player == -2 or self.counter_resource is not None):
            state.set_action_in_progress(self)
            return True
        else:
            if self.target_player in [-1, -2]:
                if self.player not in [-1, -2]:
                    self.target_player = self.player
                else:
                    self.target_player = state.get_current_player()
                    self.player = self.target_player
            elif self.target_player == -3:
                return super()._execute(state)
            
            if self.production:
                self.counter_id = state.player_production[self.target_player][self.resource].get_component_id()
            else:
                self.counter_id = state.player_resources[self.target_player][self.resource].get_component_id()
            
            if self.tag_to_count is not None:
                if self.any or self.opponents:
                    count = 0
                    for i in range(state.get_n_players()):
                        if self.opponents and i == self.player:
                            continue
                        count += state.player_cards_played_tags[i][self.tag_to_count].get_value()
                    self.change *= count
                else:
                    self.change *= state.player_cards_played_tags[self.player][self.tag_to_count].get_value()
            
            elif self.tile_to_count is not None:
                if self.on_mars:
                    count = 0
                    for i in range(state.board.get_height()):
                        for j in range(state.board.get_width()):
                            mt = state.board.get_element(j, i)
                            if (mt is not None and 
                                isinstance(mt, TMMapTile) and 
                                mt.get_tile_placed() == self.tile_to_count):
                                if self.any:
                                    count += 1
                                elif ((self.opponents and mt.get_owner_id() != self.player) or 
                                      (not self.opponents and mt.get_owner_id() == self.player)):
                                    count += 1
                    self.change *= count
                else:
                    if self.any or self.opponents:
                        count = 0
                        for i in range(state.get_n_players()):
                            if self.opponents and i == self.player:
                                continue
                            count += state.player_tiles_placed[i][self.tile_to_count].get_value()
                        self.change *= count
                    else:
                        self.change *= state.player_tiles_placed[self.player][self.tile_to_count].get_value()
            
            if self.counter_resource is not None:
                if self.counter_resource_production:
                    c = state.player_production[self.target_player][self.counter_resource]
                else:
                    c = state.player_resources[self.target_player][self.counter_resource]
                c.increment(int(-1 * self.change))
                if -1 * self.change > 0 and not self.counter_resource_production:
                    state.player_resource_increase_gen[self.target_player][self.counter_resource] = True
            
            if self.change > 0 and not self.production:
                state.player_resource_increase_gen[self.target_player][self.resource] = True
            
            return super()._execute(state)
    
    def _compute_available_actions(self, state: AbstractGameState) -> List[AbstractAction]:
        gs = state
        actions = []
        
        if self.target_player == -2:
            c = gs.player_resources[self.player][self.resource]
            max_val = -1 * min(abs(self.change), 
                              c.get_value() + abs(c.get_minimum()) if c.get_minimum() < 0 
                              else c.get_value())
            
            if self.target_player_options is not None:
                for i in self.target_player_options:
                    if i == -1:
                        actions.append(TMAction(player=self.player))
                        continue
                    if not self.production and self.change < 0:
                        for k in range(int(max_val), 0):
                            self._create_actions(gs, actions, i, k)
                    else:
                        self._create_actions(gs, actions, i, self.change)
            else:
                for i in range(state.get_n_players()):
                    if not self.production and self.change < 0:
                        for k in range(int(max_val), 0):
                            self._create_actions(gs, actions, i, k)
                    else:
                        self._create_actions(gs, actions, i, self.change)
                
                if state.get_n_players() == 1:
                    if not self.production and self.change < 0:
                        for k in range(int(max_val), 0):
                            self._create_actions(gs, actions, -3, k)
                    else:
                        self._create_actions(gs, actions, -3, self.change)
        
        elif self.counter_resource is not None:
            if self.production:
                c = gs.player_production[self.player][self.resource]
            else:
                c = gs.player_resources[self.player][self.resource]
            
            max_val = (c.get_value() + abs(c.get_minimum()) if c.get_minimum() < 0 
                      else c.get_value())
            
            for i in range(max_val + 1):
                a = ModifyPlayerResource(self.player, self.target_player, -i, self.resource, 
                                       self.production, self.tag_to_count, self.tile_to_count,
                                       self.any, self.opponents, self.on_mars, 
                                       self.counter_resource, self.counter_resource_production, True)
                a.complete = True
                actions.append(a)
        
        if not actions:
            actions.append(TMAction(player=self.player))
        
        return actions
    
    def _create_actions(self, state: TMGameState, actions: List[AbstractAction], 
                       target: int, change_val: float):
        a = ModifyPlayerResource(self.player, target, change_val, self.resource, 
                               self.production, self.tag_to_count, self.tile_to_count,
                               self.any, self.opponents, self.on_mars, 
                               self.counter_resource, self.counter_resource_production, True)
        a.complete = True
        if a.can_be_played(state):
            actions.append(a)
    
    def get_current_player(self, state: AbstractGameState) -> int:
        return self.player
    
    def _after_action(self, state: AbstractGameState, action: AbstractAction):
        self.complete = True
    
    def execution_complete(self, state: AbstractGameState) -> bool:
        return self.complete
    
    def __str__(self) -> str:
        s = "Modify "
        if self.target_player == -1:
            s += "your "
        elif self.target_player == -2:
            s += "any "
        else:
            s += f"p{self.target_player} "
        
        s += f"{self.resource}"
        if self.production:
            s += " production "
        else:
            s += " "
        
        if self.counter_resource is not None:
            if not self.complete:
                s += f"by -X and {self.counter_resource}"
                if self.counter_resource_production:
                    s += " production "
                else:
                    s += " "
                s += "by X"
            else:
                s += f"by {self.change} and {self.counter_resource}"
                if self.counter_resource_production:
                    s += " production "
                else:
                    s += " "
                s += f"by {-1 * self.change}"
        else:
            s += f"by {self.change}"
        
        if self.tag_to_count is not None:
            s += f" for each {self.tag_to_count}"
            if self.any:
                s += " ever played"
            elif self.opponents:
                s += " opponents played"
            else:
                s += " you played"
        
        if self.tile_to_count is not None:
            s += f" for each {self.tile_to_count}"
            if self.on_mars:
                s += " (on Mars)"
            if self.any:
                s += " ever played"
            elif self.opponents:
                s += " opponents played"
            else:
                s += " you played"
        
        return s