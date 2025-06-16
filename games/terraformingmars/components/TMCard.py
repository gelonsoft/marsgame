from typing import List, Set, Dict, Optional, Any, Union
import json
from copy import deepcopy

from core.components.card import Card
from games.terraformingmars import TMGameState
from games.terraformingmars import TMTypes
from games.terraformingmars.actions import ModifyPlayerResource
from games.terraformingmars.actions import TMAction
from games.terraformingmars.rules import Discount
from games.terraformingmars.rules.effects import Effect
from games.terraformingmars.rules.effects import GlobalParameterEffect
from games.terraformingmars.rules.effects import PayForActionEffect
from games.terraformingmars.rules.effects import PlaceTileEffect
from games.terraformingmars.rules.effects import PlayCardEffect
from games.terraformingmars.rules.requirements import ActionTypeRequirement
from games.terraformingmars.rules.requirements import CounterRequirement
from games.terraformingmars.rules.requirements import Requirement
from games.terraformingmars.rules.requirements import ResourceIncGenRequirement
from games.terraformingmars.rules.requirements import ResourcesOnCardsRequirement
from games.terraformingmars.rules.requirements import TagOnCardRequirement
from games.terraformingmars.rules.requirements import TagsPlayedRequirement
from games.terraformingmars.rules.requirements import TilePlacedRequirement
from utilities import Utils


class TMCard(Card):
    def __init__(self, name: str = None, component_id: int = None):
        super().__init__(name, component_id)
        
        self.number: int = 0
        self.annotation: Optional[str] = None
        self.card_type: Optional[TMTypes.CardType] = None
        self.cost: int = 0
        self.requirements: Set[Requirement] = set()
        self.tags: List[TMTypes.Tag] = []

        self.discount_effects: List[Discount] = []
        self.resource_mappings: Set[TMGameState.ResourceMapping] = set()

        self.persisting_effects: List[Effect] = []
        self.first_action: Optional[TMAction] = None
        self.first_action_executed: bool = False
        self.action_played: bool = False
        self.actions: List[TMAction] = []
        self.immediate_effects: List[TMAction] = []

        self.map_tile_id_tile_placed: int = -1

        self.n_points: float = 0.0
        self.points_resource: Optional[TMTypes.Resource] = None
        self.points_threshold: Optional[int] = None
        self.points_tag: Optional[TMTypes.Tag] = None
        self.points_tile: Optional[TMTypes.Tile] = None
        self.points_tile_adjacent: bool = False

        self.resource_on_card: Optional[TMTypes.Resource] = None
        self.n_resources_on_card: int = 0
        self.can_resources_be_removed: bool = True

    def meets_requirements(self, gs: TMGameState) -> bool:
        """Check if all requirements for this card are met."""
        for requirement in self.requirements:
            if not requirement.test_condition(gs):
                return False
        return True

    def should_save_card(self) -> bool:
        """Determine if this card should be saved based on its properties."""
        return (self.points_resource is not None or 
                self.points_tag is not None or 
                self.points_tile is not None or 
                self.resource_on_card is not None or 
                len(self.persisting_effects) > 0 or 
                len(self.actions) > 0 or 
                len(self.discount_effects) > 0 or 
                len(self.resource_mappings) > 0)

    @staticmethod
    def load_corporation(card_def: Dict[str, Any]) -> 'TMCard':
        """Load a corporation card from JSON definition."""
        card = TMCard()
        card.card_type = TMTypes.CardType.CORPORATION
        card.number = int(card_def["id"])
        card.annotation = card_def.get("annotation")
        card.set_component_name(card_def["name"])

        start = card_def["start"]
        start_resources = start[0]
        split = start_resources.split(",")
        immediate_effects = []
        
        for s in split:
            s = s.strip()
            split2 = s.split(" ")
            # First is amount
            amount = int(split2[0])
            # Second is what resource
            res_string = split2[1].split("prod")[0]
            res = Utils.search_enum(TMTypes.Resource, res_string)
            immediate_effects.append(
                ModifyPlayerResource(-1, amount, res, "prod" in split2[1])
            )

        for i in range(1, len(start)):
            other = start[i]
            effect_type = other.get("type")
            if effect_type and effect_type.lower() == "first":
                # First action in action phase for the player is decided, not free
                action = other["action"]
                card.first_action = TMAction.parse_action_on_card(action, card, False)

        if card_def.get("tags"):
            ts = card_def["tags"]
            card.tags = []
            for tag_name in ts:
                tag = Utils.search_enum(TMTypes.Tag, tag_name)
                card.tags.append(tag)

        effects = card_def["effect"]
        actions = []
        persisting_effects = []
        
        for effect in effects:
            effect_type = effect.get("type")
            if effect_type and effect_type.lower() == "action":
                # Parse actions
                action = effect["action"]
                cost_resource = None
                cost = 0
                if effect.get("cost"):
                    cost_str = effect["cost"].split("/")
                    cost_resource = getattr(TMTypes.Resource, cost_str[0])
                    cost = int(cost_str[1])
                
                a = TMAction.parse_action_on_card(action, card, False)
                if a:
                    actions.append(a)
                    a.action_type = TMTypes.ActionType.ACTIVE_ACTION
                    if cost != 0:
                        a.set_action_cost(cost_resource, cost, -1)
                    a.set_card_id(card.get_component_id())
                    if effect.get("if"):
                        # Requirement
                        req_str = effect["if"]
                        if "incgen" in req_str:
                            res = Utils.search_enum(TMTypes.Resource, req_str.split("-")[1])
                            if res:
                                a.requirements.add(ResourceIncGenRequirement(res))
                                
            elif effect_type and effect_type.lower() == "discount":
                # Parse discounts
                amount = int(effect["amount"])
                
                if effect.get("counter"):
                    # A discount for CounterRequirement
                    for counter in effect["counter"]:
                        r = CounterRequirement(counter, -1, True)
                        existing_discount = next((d for d in card.discount_effects if d.a == r), None)
                        if existing_discount:
                            disc = existing_discount.b
                            card.discount_effects.append(Discount(r, disc + amount))
                        else:
                            card.discount_effects.append(Discount(r, amount))
                            
                elif effect.get("tag"):
                    # A discount for tag requirements
                    tag = getattr(TMTypes.Tag, effect["tag"])
                    r = TagsPlayedRequirement([tag], [1])
                    existing_discount = next((d for d in card.discount_effects if d.a == r), None)
                    if existing_discount:
                        disc = existing_discount.b
                        card.discount_effects.append(Discount(r, disc + amount))
                    else:
                        card.discount_effects.append(Discount(r, amount))
                        
                elif effect.get("standardproject"):
                    # A discount for buying standard projects
                    sp = getattr(TMTypes.StandardProject, effect["standardproject"])
                    r = ActionTypeRequirement(TMTypes.ActionType.STANDARD_PROJECT, sp)
                    existing_discount = next((d for d in card.discount_effects if d.a == r), None)
                    if existing_discount:
                        disc = existing_discount.b
                        card.discount_effects.append(Discount(r, disc + amount))
                    else:
                        card.discount_effects.append(Discount(r, amount))
                        
            elif effect_type and effect_type.lower() == "resourcemapping":
                from_res = getattr(TMTypes.Resource, effect["from"])
                to_res = getattr(TMTypes.Resource, effect["to"])
                rate = float(effect["rate"])
                card.resource_mappings.add(TMGameState.ResourceMapping(from_res, to_res, rate, None))
                
            elif effect_type and effect_type.lower() == "effect":
                condition = effect["if"]
                result = effect["then"]
                persisting_effects.append(
                    TMCard._parse_effect(condition, TMAction.parse_action_on_card(result, card, True))
                )

        card.immediate_effects = immediate_effects
        card.actions = actions
        card.persisting_effects = persisting_effects

        return card

    @staticmethod
    def load_card_json(card_def: Dict[str, Any]) -> Optional['TMCard']:
        """Load card from JSON using serialization (simplified version)."""
        try:
            # This would need proper JSON deserialization implementation
            # For now, return a basic card structure
            card = TMCard()
            # Implementation would depend on your JSON structure
            return card
        except Exception as e:
            print(f"Error loading card from JSON: {e}")
            return None

    @staticmethod
    def load_card_html(card_def: Dict[str, Any]) -> 'TMCard':
        """Load card from HTML-like JSON structure."""
        card = TMCard()
        class_def = card_def.get("@class", "")
        card.card_type = Utils.search_enum(TMTypes.CardType, class_def.split(" ")[1].strip())
        card.annotation = card_def.get("annotation")
        
        div1 = card_def.get("div", [])
        temp_tags = []
        immediate_effects = []
        actions = []
        persisting_effects = []

        for ob in div1:
            info = ob.get("@class", "")
            
            if "title" in info:
                # Card type
                split = info.split("-")
                card.card_type = Utils.search_enum(TMTypes.CardType, split[-1].strip())
                # Name of card
                card.set_component_name(ob.get("#text"))
                
            elif "price" in info:
                # Cost of card
                card.cost = int(ob.get("#text"))
                
            elif "tag" in info:
                # A tag
                tag = Utils.search_enum(TMTypes.Tag, info.split("-")[1].strip())
                if tag:
                    temp_tags.append(tag)
                    
            elif "number" in info:
                # Card number
                import re
                card.number = int(re.sub(r'[^\d.]', '', ob.get("#text")))
                
            elif "content" in info:
                if isinstance(ob.get("div"), list):
                    div2 = ob["div"]
                    for ob2 in div2:
                        info2 = ob2.get("@class", "")
                        
                        if info2 and "points" in info2:
                            # Points for this card
                            ps = ob2.get("#text")
                            card.n_points = float(ps.split("/")[0])
                            
                        elif info2 and "requirements" in info2:
                            req_str = ob2.get("#text").split("*")
                            for s in req_str:
                                s = s.strip()
                                if "tags" in s:
                                    # Tag requirement
                                    s = s.replace("tags", "").strip()
                                    split = s.split(" ")
                                    tag_count = {}
                                    for s2 in split:
                                        tag = getattr(TMTypes.Tag, s2)
                                        tag_count[tag] = tag_count.get(tag, 0) + 1
                                    
                                    tags = list(tag_count.keys())
                                    min_counts = list(tag_count.values())
                                    r = TagsPlayedRequirement(tags, min_counts)
                                    card.requirements.add(r)
                                    
                                # Additional requirement parsing would go here...
                                
                        elif info2 and "description" in info2:
                            ps = ob2.get("#text")
                            split = ps.split("*")  # * separates multiple effects
                            
                            for s in split:
                                if "Requires" in s or "must" in s:
                                    continue  # Already parsed requirements
                                    
                                s = s.strip()
                                if "Action:" in s:
                                    a = TMAction.parse_action_on_card(s.split(":")[1], card, False)
                                    if a:
                                        a.action_type = TMTypes.ActionType.ACTIVE_ACTION
                                        actions.append(a)
                                        
                                elif "stays on this card" in s:
                                    card.can_resources_be_removed = False
                                    
                                elif "Effect:" in s:
                                    if "discount" in s:
                                        # Discount effects
                                        split2 = s.split("-")
                                        amount = int(split2[1])
                                        reqs = TMCard._parse_discount(split2)
                                        for r in reqs:
                                            existing_discount = next((d for d in card.discount_effects if d.a == r), None)
                                            if existing_discount:
                                                disc = existing_discount.b
                                                card.discount_effects.append(Discount(r, disc + amount))
                                            else:
                                                card.discount_effects.append(Discount(r, amount))
                                                
                                    elif "resourcemapping" in s:
                                        # Resource mappings
                                        split2 = s.split("-")
                                        from_res = getattr(TMTypes.Resource, split2[1])
                                        to_res = getattr(TMTypes.Resource, split2[2])
                                        rate = float(split2[3])
                                        card.resource_mappings.add(TMGameState.ResourceMapping(from_res, to_res, rate, None))
                                        
                                    else:
                                        # Persisting effects
                                        s2 = s.split(":")[1].strip()
                                        when = s2.split(" / ")[0].strip()
                                        then = s2.split(" / ")[1].strip()
                                        effect = TMCard._parse_effect(when, TMAction.parse_action_on_card(then, card, True))
                                        persisting_effects.append(effect)
                                        
                                elif "VP" in s:
                                    # Victory points rules
                                    point_condition = s.split(" ")
                                    n_vp = int(point_condition[0])
                                    n_other = int(point_condition[3])
                                    card.n_points = float(n_vp) / n_other
                                    other = point_condition[4]
                                    
                                    if "tag" in s:
                                        card.points_tag = getattr(TMTypes.Tag, other)
                                    else:
                                        # Maybe a resource?
                                        r = Utils.search_enum(TMTypes.Resource, other)
                                        if r:
                                            card.points_resource = r
                                            card.resource_on_card = r
                                        else:
                                            # A tile
                                            card.points_tile = getattr(TMTypes.Tile, other)
                                            if len(point_condition) > 5 and point_condition[5].lower() == "adjacent":
                                                card.points_tile_adjacent = True
                                                
                                    if point_condition[2].lower() == "if":
                                        card.points_threshold = n_other
                                        
                                else:
                                    # Parse immediate effects
                                    a = TMAction.parse_action_on_card(s, card, True)
                                    if a:
                                        immediate_effects.append(a)

        card.actions = actions
        card.persisting_effects = persisting_effects
        card.immediate_effects = immediate_effects
        card.tags = temp_tags

        return card

    @staticmethod
    def _parse_discount(split2: List[str]) -> Set[Requirement]:
        """Parse discount requirements from string split."""
        reqs = set()
        if len(split2) > 2:
            if split2[2].lower() == "global":
                # Global parameter effect
                for gp in TMTypes.GlobalParameter:
                    reqs.add(CounterRequirement(gp.name, -1, True))
            else:
                # A tag discount?
                tag_def = split2[2].split(",")
                tags = []
                for tag_name in tag_def:
                    tag = Utils.search_enum(TMTypes.Tag, tag_name)
                    if tag:
                        tags.append(tag)
                    else:
                        tags = None
                        break
                        
                if tags:
                    reqs.add(TagOnCardRequirement(tags))
        else:
            reqs.add(TagOnCardRequirement(None))
            
        return reqs

    @staticmethod
    def _parse_effect(when: str, then: TMAction) -> Optional[Effect]:
        """Parse effect conditions and create appropriate Effect object."""
        ss = when.split("(")
        action_type_condition = ss[0].strip()
        content = ss[1].replace(")", "")
        must_be_current_player = "any" not in when

        if action_type_condition.lower() == "placetile":
            # Place tile effect
            split2 = content.split(",")
            tile = None
            resources_gained = None
            
            if len(split2) > 1:
                tile = getattr(TMTypes.Tile, split2[0])
            else:
                if "gain" in split2[0]:
                    split3 = split2[0].replace("gain ", "").split("/")
                    resources_gained = []
                    for resource_name in split3:
                        resources_gained.append(getattr(TMTypes.Resource, resource_name))
                        
            return PlaceTileEffect(must_be_current_player, then, "onMars" in when, tile, resources_gained)
            
        elif action_type_condition.lower() == "playcard":
            # Play card effect
            tag_def = content.split("-")[1].split(",")
            tags = set()
            for tag_name in tag_def:
                tags.add(getattr(TMTypes.Tag, tag_name))
            return PlayCardEffect(must_be_current_player, then, tags)
            
        elif action_type_condition.lower() == "payforaction":
            # Pay for action effect
            at = Utils.search_enum(TMTypes.ActionType, content)
            if at:
                return PayForActionEffect(must_be_current_player, then, at)
            else:
                min_cost = int(content)
                return PayForActionEffect(must_be_current_player, then, min_cost)
                
        elif action_type_condition == "globalparameter":
            # Increase parameter effect
            param = Utils.search_enum(TMTypes.GlobalParameter, content.split("-")[0])
            if param:
                return GlobalParameterEffect(must_be_current_player, then, param)

        return None

    def __eq__(self, other) -> bool:
        """Check equality with another TMCard."""
        if not isinstance(other, TMCard):
            return False
        if not super().__eq__(other):
            return False
            
        return (self.number == other.number and
                self.cost == other.cost and
                self.first_action_executed == other.first_action_executed and
                self.action_played == other.action_played and
                self.map_tile_id_tile_placed == other.map_tile_id_tile_placed and
                abs(self.n_points - other.n_points) < 1e-9 and
                self.points_tile_adjacent == other.points_tile_adjacent and
                self.n_resources_on_card == other.n_resources_on_card and
                self.can_resources_be_removed == other.can_resources_be_removed and
                self.annotation == other.annotation and
                self.card_type == other.card_type and
                self.requirements == other.requirements and
                self.tags == other.tags and
                self.discount_effects == other.discount_effects and
                self.resource_mappings == other.resource_mappings and
                self.persisting_effects == other.persisting_effects and
                self.first_action == other.first_action and
                self.actions == other.actions and
                self.immediate_effects == other.immediate_effects and
                self.points_resource == other.points_resource and
                self.points_threshold == other.points_threshold and
                self.points_tag == other.points_tag and
                self.points_tile == other.points_tile and
                self.resource_on_card == other.resource_on_card)

    def __hash__(self) -> int:
        """Generate hash for the TMCard."""
        return hash((
            super().__hash__(), self.number, self.annotation, self.card_type, self.cost,
            frozenset(self.requirements), tuple(self.tags), tuple(self.discount_effects),
            frozenset(self.resource_mappings), self.first_action, self.first_action_executed,
            self.action_played, self.map_tile_id_tile_placed, self.n_points,
            self.points_resource, self.points_threshold, self.points_tag, self.points_tile,
            self.points_tile_adjacent, self.resource_on_card, self.n_resources_on_card,
            self.can_resources_be_removed
        ))

    def copy(self) -> 'TMCard':
        """Create a deep copy of the TMCard."""
        copy_card = TMCard(self.component_name, self.component_id)
        copy_card.number = self.number
        copy_card.card_type = self.card_type
        copy_card.first_action_executed = self.first_action_executed
        copy_card.action_played = self.action_played
        copy_card.annotation = self.annotation
        copy_card.cost = self.cost
        
        if self.requirements:
            copy_card.requirements = {r.copy() for r in self.requirements}
            
        copy_card.tags = self.tags.copy()
        
        if self.discount_effects:
            copy_card.discount_effects = [
                Discount(discount.a.copy(), discount.b) 
                for discount in self.discount_effects
            ]
            
        if self.resource_mappings:
            copy_card.resource_mappings = {rm.copy() for rm in self.resource_mappings}
            
        if self.persisting_effects:
            copy_card.persisting_effects = [
                effect.copy() if effect else None 
                for effect in self.persisting_effects
            ]
            
        if self.first_action:
            copy_card.first_action = self.first_action.copy()
            
        if self.actions:
            copy_card.actions = [action.copy() for action in self.actions]
            
        if self.immediate_effects:
            copy_card.immediate_effects = [effect.copy() for effect in self.immediate_effects]
            
        copy_card.map_tile_id_tile_placed = self.map_tile_id_tile_placed
        copy_card.n_points = self.n_points
        copy_card.points_resource = self.points_resource
        copy_card.points_threshold = self.points_threshold
        copy_card.points_tag = self.points_tag
        copy_card.points_tile = self.points_tile
        copy_card.points_tile_adjacent = self.points_tile_adjacent
        copy_card.resource_on_card = self.resource_on_card
        copy_card.n_resources_on_card = self.n_resources_on_card
        copy_card.can_resources_be_removed = self.can_resources_be_removed
        
        self.copy_component_to(copy_card)
        return copy_card

    def copy_serializable(self) -> 'TMCard':
        """Create a serializable copy of the TMCard."""
        copy_card = TMCard(self.component_name, self.component_id)
        copy_card.number = self.number
        copy_card.card_type = self.card_type
        copy_card.annotation = self.annotation
        copy_card.cost = self.cost
        
        if self.requirements:
            copy_card.requirements = {r.copy_serializable() for r in self.requirements}
        else:
            copy_card.requirements = set()
            
        if self.tags:
            copy_card.tags = self.tags.copy()
        else:
            copy_card.tags = []
            
        if self.discount_effects:
            copy_card.discount_effects = [
                Discount(discount.a.copy_serializable(), discount.b) 
                for discount in self.discount_effects
            ]
        else:
            copy_card.discount_effects = []
            
        if self.resource_mappings:
            copy_card.resource_mappings = {rm.copy() for rm in self.resource_mappings}
        else:
            copy_card.resource_mappings = set()
            
        if self.persisting_effects:
            copy_card.persisting_effects = [
                effect.copy_serializable() if effect else None 
                for effect in self.persisting_effects
            ]
        else:
            copy_card.persisting_effects = []
            
        if self.first_action:
            copy_card.first_action = self.first_action.copy_serializable()
        else:
            copy_card.first_action = None
            
        if self.actions:
            copy_card.actions = [action.copy_serializable() for action in self.actions]
        else:
            copy_card.actions = []
            
        if self.immediate_effects:
            copy_card.immediate_effects = [effect.copy_serializable() for effect in self.immediate_effects]
        else:
            copy_card.immediate_effects = []
            
        copy_card.map_tile_id_tile_placed = self.map_tile_id_tile_placed
        copy_card.n_points = self.n_points
        copy_card.points_resource = self.points_resource
        copy_card.points_threshold = self.points_threshold
        copy_card.points_tag = self.points_tag
        copy_card.points_tile = self.points_tile
        copy_card.points_tile_adjacent = self.points_tile_adjacent
        copy_card.resource_on_card = self.resource_on_card
        copy_card.n_resources_on_card = self.n_resources_on_card
        copy_card.can_resources_be_removed = self.can_resources_be_removed
        
        self.copy_component_to(copy_card)
        
        # Clean up properties if empty (similar to Java version)
        if not copy_card.properties:
            copy_card.properties = {}
            
        return copy_card