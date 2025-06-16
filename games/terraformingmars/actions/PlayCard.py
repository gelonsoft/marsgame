from core import AbstractGameState
from games.terraformingmars import TMAction
from games.terraformingmars import TMGameParameters
from games.terraformingmars import TMGameState
from games.terraformingmars import TMTypes
from games.terraformingmars.components import TMCard
from games.terraformingmars.rules.requirements import PlayableActionRequirement


class PlayCard(TMAction):
    def __init__(self, player: int = -1, card: TMCard = None, free: bool = False):
        if card is not None:
            super().__init__(action_type=TMTypes.ActionType.PlayCard, player=player, free=free)
            self.set_action_cost(TMTypes.Resource.MegaCredit, card.cost, card.get_component_id())
            self.set_card_id(card.get_component_id())

            self.requirements.extend(card.requirements)
            for aa in card.immediate_effects:
                # All immediate effects must also be playable in order for this card to be playable
                self.requirements.append(PlayableActionRequirement(aa))
        else:
            super().__init__(player=player, free=free)

    def _execute(self, gs: TMGameState) -> bool:
        gp = gs.get_game_parameters()
        card = gs.get_component_by_id(self.get_play_card_id())
        self._play_card(gs, card)
        return True

    def _play_card(self, gs: TMGameState, card: TMCard):
        # Remove from hand, resolve on-play effects and add tags etc. to cards played lists
        gs.get_player_hands()[self.player].remove(card)

        # Add info to played cards stats
        if card.card_type != TMTypes.CardType.Event:  # Event tags don't count for regular tag counts
            for t in card.tags:
                gs.get_player_cards_played_tags()[self.player][t].increment(1)
        else:
            gs.get_player_cards_played_tags()[self.player][TMTypes.Tag.Event].increment(1)

        gs.get_player_cards_played_types()[self.player][card.card_type].increment(1)
        
        if card.should_save_card():
            gs.get_player_complicated_point_cards()[self.player].append(card)
            gs.get_played_cards()[self.player].append(card)
        else:
            gs.get_played_cards()[self.player].append(card)
            if card.n_points != 0:
                gs.get_player_card_points()[self.player].increment(int(card.n_points))

        # Add actions
        for a in card.actions:
            a.player = self.player
            a.set_card_id(card.get_component_id())
            gs.get_player_extra_actions()[self.player].append(a)

        # Add discount_effects to player's discounts
        gs.add_discount_effects(card.discount_effects)
        gs.add_resource_mappings(card.resource_mappings, False)
        # Add persisting effects
        gs.add_persisting_effects(card.persisting_effects)

        # Execute on-play effects
        for aa in card.immediate_effects:
            card.action_played = False  # This is set by each action, preventing the next ones, but we want all to be executed
            aa.player = self.player
            aa.execute(gs)
        card.action_played = False  # We've not executed the active card action

    def _copy(self):
        return PlayCard(player=self.player, free=self.free_action_point)

    def get_string(self, game_state: AbstractGameState) -> str:
        card = game_state.get_component_by_id(self.get_play_card_id())
        return f"Play card {card.get_component_name()}"

    def __str__(self) -> str:
        return f"Play card id {self.get_play_card_id()}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, PlayCard):
            return False
        return super().__eq__(other)