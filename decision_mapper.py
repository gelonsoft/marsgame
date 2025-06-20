import json
from typing import Dict, List, Any, Union, Optional, Tuple
from dataclasses import dataclass
from itertools import product
import logging
  # Set the logging level to

@dataclass
class DecisionContext:
    """Context for making decisions - can be extended with game state, AI logic, etc."""
    player_resources: Dict[str, int] = None
    available_cards: List[str] = None
    game_state: Dict[str, Any] = None
    strategy_preference: str = "balanced"  # "aggressive", "conservative", "balanced"

class TerraformingMarsDecisionMapper:
    """Maps PlayerInputModel objects to InputResponse objects for Terraforming Mars game decisions."""
    
    def __init__(self, context: Optional[DecisionContext] = None):
        self.context = context or DecisionContext()
    
    def map_decision(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point to map a PlayerInputModel to an InputResponse.
        
        Args:
            player_input: Dictionary representing a PlayerInputModel
            
        Returns:
            Dictionary representing an InputResponse
        """
        input_type = player_input.get("type")
        
        mapping_functions = {
            "and": self._map_and_options,
            "or": self._map_or_options,
            "initialCards": self._map_select_initial_cards,
            "option": self._map_select_option,
            "projectCard": self._map_select_project_card,
            "card": self._map_select_card,
            "amount": self._map_select_amount,
            "colony": self._map_select_colony,
            "delegate": self._map_select_delegate,
            "party": self._map_select_party,
            "payment": self._map_select_payment,
            "player": self._map_select_player,
            "productionToLose": self._map_select_production_to_lose,
            "space": self._map_select_space,
            "aresGlobalParameters": self._map_shift_ares_global_parameters,
            "globalEvent": self._map_select_global_event,
            "resource": self._map_select_resource,
            "resources": self._map_select_resources
        }
        
        if input_type not in mapping_functions:
            raise ValueError(f"Unknown input type: {input_type}")
        
        return mapping_functions[input_type](player_input)
    
    def generate_action_space(self, player_input: Dict[str, Any],player_state: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        
        if player_input is None:
            return {}
        """
        Generates all possible action combinations for a given PlayerInputModel.
        
        Args:
            player_input: Dictionary representing a PlayerInputModel
            
        Returns:
            Dictionary mapping action numbers to InputResponse objects
        """
        action_space = {}
        input_type = player_input.get("type")
        
        if input_type == "and":
            # For AND options, we need to generate combinations of all sub-options
            options = player_input.get("options", [])
            if not options:
                return {0: {"type": "and", "responses": []}}
            
            # Generate action spaces for each sub-option
            sub_action_spaces = [self.generate_action_space(opt,player_state) for opt in options]
            
            # Generate all combinations of sub-action indices
            action_indices = product(*[range(len(space)) for space in sub_action_spaces])
            
            for i, indices in enumerate(action_indices):
                responses = [sub_action_spaces[j][idx]["responses"][0] if "responses" in sub_action_spaces[j][idx] 
                        else sub_action_spaces[j][idx] 
                        for j, idx in enumerate(indices)]
                action_space[i] = {"type": "and", "responses": responses}
        
        elif input_type == "or":
            # For OR options, each option is a separate action
            options = player_input.get("options", [])
            i=0

            for idx, option in enumerate(options):
                sub_action_spaces = self.generate_action_space(option,player_state)
                for sub_option_idx in sub_action_spaces:
                    action_space[i] = {"type":"or",
                                       "index":idx,
                                       "response":sub_action_spaces[sub_option_idx]}
                    i+=1
        
        elif input_type == "initialCards":
            # For initial cards selection, we need to handle multiple card selections
            options = player_input.get("options", [])
            if not options:
                return {0: {"type": "initialCards", "responses": []}}
            
            # Generate action spaces for each card selection option
            sub_action_spaces = [self.generate_action_space(opt,player_state) for opt in options]
            
            # Generate all combinations of card selections
            action_indices = product(*[range(len(space)) for space in sub_action_spaces])
            
            for i, indices in enumerate(action_indices):
                responses = []
                for j, idx in enumerate(indices):
                    response = sub_action_spaces[j][idx]
                    if "cards" in response:
                        # For card selection responses
                        responses.append({
                            "type": "card",
                            "cards": response["cards"]
                        })
                    else:
                        # For other types of responses
                        responses.append(response)
                
                action_space[i] = {
                    "type": "initialCards",
                    "responses": responses
                }
        
        elif input_type == "card":
            # For card selection, generate all combinations within min/max constraints
            original_cards = player_input.get("cards", [])
            cards=[card for card in original_cards if not card.get('isDisabled',False)]
            #print(f"Original cards: \n{original_cards}")
            #print(f"Not disabled cards: \n{cards}")
            
            
            min_cards = player_input.get("min", 0)
            max_cards = player_input.get("max", len(cards))
            
            # Generate all possible combinations of cards
            action_num = 0
            for n in range(min_cards, max_cards + 1):
                from itertools import combinations
                for combo in combinations(cards, n):
                    action_space[action_num] = {
                        "type": "card",
                        "cards": [card["name"] for card in combo]
                    }
                    action_num += 1
        
        elif input_type == "amount":
            # For amount selection, each possible amount is a separate action
            min_amount = player_input.get("min", 0)
            max_amount = player_input.get("max", 0)
            
            for amount in range(min_amount, max_amount + 1):
                action_space[amount - min_amount] = {
                    "type": "amount",
                    "amount": amount
                }
        
        elif input_type in ["projectCard", "colony", "delegate", "party", 
                        "player", "globalEvent", "resource"]:
            # For simple selection types, each option is a separate action
            items = []
            if input_type == "projectCard":
                items = player_input.get("cards", [])
            elif input_type == "colony":
                items = player_input.get("coloniesModel", [])
            elif input_type == "delegate":
                items = player_input.get("players", [])
            elif input_type == "party":
                items = player_input.get("parties", [])
            elif input_type == "player":
                items = player_input.get("players", [])
            elif input_type == "globalEvent":
                items = player_input.get("globalEventNames", [])
            elif input_type == "resource":
                items = player_input.get("include", [])
            
            j=0
            for i, item in enumerate(items):
                if input_type == "projectCard":
                    payments = self._create_payment_from_input(player_input, item["name"], items,player_state)
                    for payment in payments:
                        action_space[j] = {
                            "type": "projectCard",
                            "card": item["name"],
                            "payment": payment
                        }
                        j+=1
                elif input_type == "colony":
                    action_space[i] = {
                        "type": "colony",
                        "colonyName": item["name"]
                    }
                elif input_type == "party":
                    action_space[i] = {
                        "type": "party",
                        "partyName": item
                    }
                else:
                    action_space[i] = {
                        "type": input_type,
                        input_type: item if isinstance(item, str) else item["name"] if "name" in item else item
                    }
        
        elif input_type == "productionToLose":
            # For production to lose, we need to generate all possible unit combinations
            # This is complex, so we'll just return the default for now
            action_space[0] = self._map_select_production_to_lose(player_input)
        
        elif input_type == "space":
            # For space selection, each space is a separate action
            spaces = player_input.get("spaces", [])
            for i, space in enumerate(spaces):
                action_space[i] = {
                    "type": "space",
                    "spaceId": space
                }
        
        elif input_type == "resources":
            # For resource selection, we'll just return the default for now
            action_space[0] = self._map_select_resources(player_input)
        
        else:
            # For other types, just return the default mapped decision
            action_space[0] = self.map_decision(player_input)
        
        return action_space
    
    def _map_and_options(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map AndOptionsModel to AndOptionsResponse."""
        options = player_input.get("options", [])
        responses = []
        
        for option in options:
            responses.append(self.map_decision(option))
        
        return {
            "type": "and",
            "responses": responses
        }
    
    def _map_or_options(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map OrOptionsModel to OrOptionsResponse."""
        options = player_input.get("options", [])
        initial_idx = player_input.get("initialIdx", 0)
        
        # Default to the initial index or first option
        selected_index = min(initial_idx, len(options) - 1) if options else 0
        
        # You could add logic here to choose based on context/strategy
        if self.context and hasattr(self.context, 'strategy_preference'):
            selected_index = self._choose_or_option_strategically(options, selected_index)
        
        selected_option = options[selected_index] if options else {}
        
        return {
            "type": "or",
            "index": selected_index,
            "response": self.map_decision(selected_option) if selected_option else {"type": "option"}
        }
    
    def _map_select_initial_cards(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectInitialCardsModel to SelectInitialCardsResponse."""
        options = player_input.get("options", [])
        responses = []
        
        for option in options:
            responses.append(self.map_decision(option))
        
        return {
            "type": "initialCards",
            "responses": responses
        }
    
    def _map_select_option(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectOptionModel to SelectOptionResponse."""
        return {
            "type": "option"
        }
    
    def _map_select_project_card(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectProjectCardToPlayModel to SelectProjectCardToPlayResponse."""
        cards = player_input.get("cards", [])
        
        # Choose first available card by default
        selected_card = cards[0]["name"] if cards else ""
        
        # Create payment based on available resources in the input
        payment = self._create_payment_from_input(player_input, selected_card, cards)
        
        return {
            "type": "projectCard",
            "card": selected_card,
            "payment": payment
        }
    
    def _map_select_card(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectCardModel to SelectCardResponse."""
        cards = player_input.get("cards", [])
        min_cards = player_input.get("min", 0)
        max_cards = player_input.get("max", len(cards))
        
        # Select cards up to the minimum required, or maximum allowed
        num_to_select = min(max_cards, max(min_cards, 1))
        selected_cards = [card["name"] for card in cards[:num_to_select]]
        
        return {
            "type": "card",
            "cards": selected_cards
        }
    
    def _map_select_amount(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectAmountModel to SelectAmountResponse."""
        min_amount = player_input.get("min", 0)
        max_amount = player_input.get("max", 0)
        max_by_default = player_input.get("maxByDefault", False)
        
        # Choose amount based on strategy
        if max_by_default:
            amount = max_amount
        else:
            amount = min_amount
        
        return {
            "type": "amount",
            "amount": amount
        }
    
    def _map_select_colony(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectColonyModel to SelectColonyResponse."""
        colonies = player_input.get("coloniesModel", [])
        
        # Select first available colony
        colony_name = colonies[0]["name"] if colonies else ""
        
        return {
            "type": "colony",
            "colonyName": colony_name
        }
    
    def _map_select_delegate(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectDelegateModel to SelectDelegateResponse."""
        players = player_input.get("players", [])
        
        # Select first available player/neutral
        selected_player = players[0] if players else "NEUTRAL"
        
        return {
            "type": "delegate",
            "player": selected_player
        }
    
    def _map_select_party(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectPartyModel to SelectPartyResponse."""
        parties = player_input.get("parties", [])
        
        # Select first available party
        party_name = parties[0] if parties else ""
        
        return {
            "type": "party",
            "partyName": party_name
        }
    
    def _map_select_payment(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectPaymentModel to SelectPaymentResponse."""
        amount = player_input.get("amount", 0)
        
        # Create basic payment with megacredits
        payment = self._create_basic_payment(amount)
        
        return {
            "type": "payment",
            "payment": payment
        }
    
    def _map_select_player(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectPlayerModel to SelectPlayerResponse."""
        players = player_input.get("players", [])
        
        # Select first available player
        selected_player = players[0] if players else ""
        
        return {
            "type": "player",
            "player": selected_player
        }
    
    def _map_select_production_to_lose(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectProductionToLoseModel to SelectProductionToLoseResponse."""
        pay_production = player_input.get("payProduction", {})
        units = pay_production.get("units", {})
        
        # Use the units from the input, or create empty units
        response_units = {
            "megacredits": units.get("megacredits", 0),
            "steel": units.get("steel", 0),
            "titanium": units.get("titanium", 0),
            "plants": units.get("plants", 0),
            "energy": units.get("energy", 0),
            "heat": units.get("heat", 0)
        }
        
        return {
            "type": "productionToLose",
            "units": response_units
        }
    
    def _map_select_space(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectSpaceModel to SelectSpaceResponse."""
        spaces = player_input.get("spaces", [])
        
        # Select first available space
        space_id = spaces[0] if spaces else ""
        
        return {
            "type": "space",
            "spaceId": space_id
        }
    
    def _map_shift_ares_global_parameters(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map ShiftAresGlobalParametersModel to ShiftAresGlobalParametersResponse."""
        # Default to no changes
        ares_response = {
            "lowOceanDelta": 0,
            "highOceanDelta": 0,
            "temperatureDelta": 0,
            "oxygenDelta": 0
        }
        
        return {
            "type": "aresGlobalParameters",
            "response": ares_response
        }
    
    def _map_select_global_event(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectGlobalEventModel to SelectGlobalEventResponse."""
        global_events = player_input.get("globalEventNames", [])
        
        # Select first available global event
        selected_event = global_events[0] if global_events else ""
        
        return {
            "type": "globalEvent",
            "globalEventName": selected_event
        }
    
    def _map_select_resource(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectResourceModel to SelectResourceResponse."""
        resources = player_input.get("include", [])
        
        # Select first available resource
        selected_resource = resources[0] if resources else ""
        
        return {
            "type": "resource",
            "resource": selected_resource
        }
    
    def _map_select_resources(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectResourcesModel to SelectResourcesResponse."""
        count = player_input.get("count", 0)
        
        # Distribute resources (simple strategy: prefer megacredits)
        units = {
            "megacredits": count,
            "steel": 0,
            "titanium": 0,
            "plants": 0,
            "energy": 0,
            "heat": 0
        }
        
        return {
            "type": "resources",
            "units": units
        }
    
    def _create_payment_from_input(self, player_input: Dict, selected_card: str, cards: List[Dict], player_state: Dict) -> List[Dict]:
        """Generate all valid payment combinations for card playment based on available resources."""
        # Find the selected card to get its cost
        card_cost = 0
        for card in cards:
            if card.get("name") == selected_card:
                card_cost = card.get("calculatedCost", 0)
                break
        
        if card_cost <= 0:
            return []
        
        #print(f"Player input: {player_input}")
        # Get player's available resources
        player_resources = {
            "megaCredits": player_state["thisPlayer"].get("megaCredits", 0),
            "heat": player_state["thisPlayer"].get("heat", 0),
            "steel": player_state["thisPlayer"].get("steel", 0),
            "titanium": player_state["thisPlayer"].get("titanium", 0),
            "plants": player_state["thisPlayer"].get("plants", 0),
            "microbes": player_input.get("microbes", 0),
            "floaters": player_input.get("floaters", 0),
            "lunaArchivesScience": player_input.get("lunaArchivesScience", 0),
            "spireScience": player_input.get("spireScience", 0),
            "seeds": player_input.get("seeds", 0),
            "auroraiData": player_input.get("auroraiData", 0),
            "graphene": player_input.get("graphene", 0),
            "kuiperAsteroids": player_input.get("kuiperAsteroids", 0),
            "corruption": player_input.get("corruption", 0)
        }
        
        # Get resource values and payment restrictions
        steel_value = player_state["thisPlayer"].get("steelValue", 1)
        titanium_value = player_state["thisPlayer"].get("titaniumValue", 1)
        #print(f"Steel value {steel_value}, titanium value={titanium_value}")
        payment_options = player_input.get("paymentOptions", {})
        
        # Check if certain payment methods are restricted
        can_use_heat = payment_options.get("heat", True)
        can_use_steel = True
        can_use_titanium = True
        can_use_plants = payment_options.get("plants", True)
        
        # Calculate maximum possible units for each resource type without overpaying
        def max_resource(resource: str, value: int = 1) -> int:
            max_possible = min(
                player_resources[resource],
                (card_cost + value - 1) // value  # Ceiling division to prevent overpayment
            )
            return max(0, max_possible)
        
        max_mc = max_resource("megaCredits")
        max_heat = max_resource("heat") if can_use_heat else 0
        max_steel = max_resource("steel", steel_value) if can_use_steel else 0
        max_titanium = max_resource("titanium", titanium_value) if can_use_titanium else 0
        max_plants = max_resource("plants") if can_use_plants else 0
        
        # Generate minimal payment combinations
        minimal_payments = []
        
        # Generate base resource combinations (ordered by efficiency)
        for mc in range(max_mc, -1, -1):
            remaining_after_mc = card_cost - mc
            if remaining_after_mc <= 0:
                # Pure MC payment
                payment = self._create_base_payment(mc, 0, 0, 0, 0)
                minimal_payments.append(payment)
                continue
            
            for ti in range(min(max_titanium, (remaining_after_mc + titanium_value - 1) // titanium_value), -1, -1):
                remaining_after_ti = remaining_after_mc - ti * titanium_value
                if remaining_after_ti <= 0:
                    payment = self._create_base_payment(mc, 0, ti, 0, 0)
                    minimal_payments.append(payment)
                    continue
                
                for st in range(min(max_steel, (remaining_after_ti + steel_value - 1) // steel_value), -1, -1):
                    remaining_after_st = remaining_after_ti - st * steel_value
                    if remaining_after_st <= 0:
                        payment = self._create_base_payment(mc, st, ti, 0, 0)
                        minimal_payments.append(payment)
                        continue
                    
                    for he in range(min(max_heat, remaining_after_st), -1, -1):
                        remaining_after_he = remaining_after_st - he
                        if remaining_after_he <= 0:
                            payment = self._create_base_payment(mc, st, ti, he, 0)
                            minimal_payments.append(payment)
                            continue
                        
                        for pl in range(min(max_plants, remaining_after_he), -1, -1):
                            if mc + (ti * titanium_value) + (st * steel_value) + he + pl >= card_cost:
                                payment = self._create_base_payment(mc, st, ti, he, pl)
                                minimal_payments.append(payment)
        
        # Add special resource combinations
        extended_payments = []
        for payment in minimal_payments:
            extended_payments.append(payment.copy())
            
            # Replace MC with special resources where possible
            remaining_mc = payment["megaCredits"]
            for resource in ["microbes", "floaters", "lunaArchivesScience", 
                            "spireScience", "seeds", "auroraiData", 
                            "graphene", "kuiperAsteroids", "corruption"]:
                if player_resources[resource] > 0:
                    max_replace = min(player_resources[resource], remaining_mc)
                    if max_replace > 0:
                        new_payment = payment.copy()
                        new_payment["megaCredits"] -= max_replace
                        new_payment[resource] = max_replace
                        extended_payments.append(new_payment)
        
        # Filter valid payments (no overpayment and within resource limits)
        valid_payments = []
        for payment in extended_payments:
            total_paid = (
                payment["megaCredits"] +
                payment["steel"] * steel_value +
                payment["titanium"] * titanium_value +
                payment["heat"] +
                payment["plants"] +
                sum(payment[res] for res in [
                    "microbes", "floaters", "lunaArchivesScience",
                    "spireScience", "seeds", "auroraiData",
                    "graphene", "kuiperAsteroids", "corruption"
                ])
            )
            
            # Check for exact payment and resource limits
            if (total_paid == card_cost and
                all(0 <= payment[res] <= player_resources[res] for res in player_resources)):
                valid_payments.append(payment)
        
        # Remove duplicates and sort by efficiency
        unique_payments = []
        seen = set()
        
        for payment in sorted(valid_payments, key=lambda p: (
            p["megaCredits"],  # Prefer fewer MC
            -p["titanium"],    # Prefer more titanium (higher value)
            -p["steel"],       # Prefer more steel
            p["heat"],        # Prefer less heat
            p["plants"]       # Prefer less plants
        )):
            payment_tuple = tuple(sorted(payment.items()))
            if payment_tuple not in seen:
                seen.add(payment_tuple)
                unique_payments.append(payment)
        
        return unique_payments
    def _create_base_payment(self, mc: int, st: int, ti: int, he: int, pl: int) -> Dict:
        """Create a base payment dictionary with special resources set to 0."""
        return {
            "megaCredits": mc,
            "steel": st,
            "titanium": ti,
            "heat": he,
            "plants": pl,
            "microbes": 0,
            "floaters": 0,
            "lunaArchivesScience": 0,
            "spireScience": 0,
            "seeds": 0,
            "auroraiData": 0,
            "graphene": 0,
            "kuiperAsteroids": 0,
            "corruption": 0
        }
    
    def _create_basic_payment(self, amount: int) -> Dict[str, Any]:
        """Create a basic payment object with the specified amount in megacredits."""
        return {
            "megaCredits": amount,
            "heat": 0,
            "steel": 0,
            "titanium": 0,
            "plants": 0,
            "microbes": 0,
            "floaters": 0,
            "lunaArchivesScience": 0,
            "spireScience": 0,
            "seeds": 0,
            "auroraiData": 0,
            "graphene": 0,
            "kuiperAsteroids": 0,
            "corruption": 0
        }
    
    def _choose_or_option_strategically(self, options: List[Dict], default_index: int) -> int:
        """Choose an option based on strategy preference. Override this method for custom logic."""
        # Simple example: could be extended with game state analysis
        if self.context.strategy_preference == "aggressive":
            # Prefer higher risk/reward options (later in list)
            return min(len(options) - 1, default_index + 1)
        elif self.context.strategy_preference == "conservative":
            # Prefer safer options (earlier in list)
            return max(0, default_index - 1)
        else:
            # Balanced: use default
            return default_index

# Example usage and testing
def example_usage():
    """Demonstrate how to use the decision mapper and action space generator."""
    
    # Example PlayerInputModel for selecting cards
    select_cards_input = {
        "type": "card",
        "title": "Select cards to keep",
        "buttonLabel": "Select",
        "cards": [
            {"name": "Solar Power", "calculatedCost": 11},
            {"name": "Research", "calculatedCost": 11},
            {"name": "Invention", "calculatedCost": 5}
        ],
        "min": 2,
        "max": 2,
        "selectBlueCardAction": False,
        "showOnlyInLearnerMode": False,
        "showOwner": False
    }
    
    # Create mapper with context
    context = DecisionContext(strategy_preference="balanced")
    mapper = TerraformingMarsDecisionMapper(context)
    
    # Generate action space
    action_space = mapper.generate_action_space(select_cards_input,{})
    logging.debug("Action Space for Card Selection:")
    for action_num, response in action_space.items():
        logging.debug(f"Action {action_num}: {json.dumps(response, indent=2)}")
    
    # Example with OR options
    or_options_input = {
        "type": "or",
        "title": "Choose an action",
        "buttonLabel": "Choose",
        "options": [
            {"type": "option", "title": "Gain 2 plants", "buttonLabel": "Gain plants"},
            {"type": "option", "title": "Gain 1 steel", "buttonLabel": "Gain steel"}
        ],
        "initialIdx": 0
    }
    
    or_action_space = mapper.generate_action_space(or_options_input,{})
    logging.debug("\nAction Space for OR Options:")
    for action_num, response in or_action_space.items():
        logging.debug(f"Action {action_num}: {json.dumps(response, indent=2)}")

if __name__ == "__main__":
    example_usage()