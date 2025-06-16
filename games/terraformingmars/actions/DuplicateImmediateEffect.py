class DuplicateImmediateEffect(TMAction, IExtendedSequence):
    """Action to duplicate immediate effects from played cards with specific tags."""
    
    def __init__(self, tag_requirement: TMTypes.Tag = None, action_class_name: str = None,
                 production: bool = False, player: int = -1, card_id: int = -1):
        super().__init__(player, True)
        self.tag_requirement = tag_requirement
        self.action_class_name = action_class_name
        self.production = production
        self.set_card_id(card_id)
    
    def _execute(self, game_state: TMGameState) -> bool:
        if self.get_card_id() == -1:
            # Find viable cards and put them in card choice deck
            found = False
            for card in game_state.get_played_cards()[self.player].get_components():
                if self.tag_requirement in card.tags:
                    for action in card.immediate_effects:
                        class_name = action.__class__.__name__
                        if (class_name.lower() == self.action_class_name.lower() and
                            (self.action_class_name.lower() != "modifyplayerresource" or 
                             action.production == self.production)):
                            game_state.get_player_card_choice()[self.player].add(card)
                            found = True
                            break
                    if found:
                        game_state.set_action_in_progress(self)
                        break
        else:
            # Execute all matching effects on the chosen card
            card = game_state.get_component_by_id(self.get_card_id())
            for action in card.immediate_effects:
                class_name = action.__class__.__name__
                if (class_name.lower() == self.action_class_name.lower() and
                    (self.action_class_name.lower() != "modifyplayerresource" or 
                     action.production == self.production)):
                    action.player = self.player
                    action.execute(game_state)
        
        return True
    
    def _compute_available_actions(self, state: AbstractGameState) -> List[AbstractAction]:
        """Choose card from card choice."""
        gs = state
        actions = []
        for card in gs.get_player_card_choice()[self.player].get_components():
            actions.append(DuplicateImmediateEffect(
                self.tag_requirement, self.action_class_name, self.production,
                self.player, card.get_component_id()))
        return actions
    
    def get_current_player(self, state: AbstractGameState) -> int:
        return self.player
    
    def _after_action(self, state: AbstractGameState, action: AbstractAction) -> None:
        self.set_card_id(action.get_card_id())
        gs = state
        gs.get_player_card_choice()[self.player].clear()
    
    def execution_complete(self, state: AbstractGameState) -> bool:
        return self.get_card_id() != -1
    
    def copy(self) -> 'DuplicateImmediateEffect':
        return super().copy()
    
    def _copy(self) -> 'DuplicateImmediateEffect':
        return DuplicateImmediateEffect(
            self.tag_requirement, self.action_class_name, self.production,
            self.player, self.get_card_id())
    
    def get_string(self, game_state: AbstractGameState) -> str:
        return "Duplicate card"
    
    def __str__(self) -> str:
        return "Duplicate card"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, DuplicateImmediateEffect):
            return False
        return (super().__eq__(other) and
                self.tag_requirement == other.tag_requirement and
                self.action_class_name == other.action_class_name and
                self.production == other.production)
    
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.tag_requirement, 
                    self.action_class_name, self.production))