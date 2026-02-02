import json
from typing import Dict, List, Any, Union, Optional, Tuple
from dataclasses import dataclass
from itertools import product
import logging
from uuid import uuid4
  # Set the logging level to
from myconfig import ALL_CARDS
from myutils import find_first_with_nested_attr
from payment_options_calculator import PaymentOptionsCalculator

   


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
        self.payment_options_calculator=PaymentOptionsCalculator()
    
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

            "option": self._map_select_option,
            "player": self._map_select_player,
            "aresGlobalParameters": self._map_shift_ares_global_parameters,
            "globalEvent": self._map_select_global_event,
            "resources": self._map_select_resources
        }
        
        if input_type not in mapping_functions:
            raise ValueError(f"Unknown input type: {input_type}")
        
        return mapping_functions[input_type](player_input)
    
    def generate_action_space(self, player_input: Dict[str, Any],player_state: Dict[str, Any], is_main=True,original_player_input:Union[Dict[str,Any],None]=None) -> Dict[int, Dict[str, Any]]:
        
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
        cards_in_hand=[]
        if player_state and 'cardsInHand' in player_state:
            cards_in_hand=[c['name'] for c in player_state['cardsInHand']]
        if original_player_input is not None:
            i=0
            first_deffered_action=find_first_with_nested_attr(original_player_input,"__deferred_action")
            if first_deffered_action is not None:
                parent,deffered=first_deffered_action
                if deffered['xtype']=="xcard_choose":
                    selected_cards=deffered.get('selected',[])
                    if len(selected_cards)>=deffered['xmin']:
                        action_space[i]={
                                "type":"deffered",
                                "xid":deffered["xid"],
                                "xtype":'xconfirm_card_choose'
                        }
                        i=i+1
                    if len(selected_cards)>1: 
                        action_space[i]={
                                "type":"deffered",
                                "xid":deffered["xid"],
                                "xtype":'xremove_first_card_choose'
                        }
                        i=i+1
                    for c in deffered['xoptions']:
                        if c['name'] not in selected_cards:
                            action_space[i]={
                                "type":"deffered",
                                "xid":deffered["xid"],
                                "xtype":deffered["xtype"],
                                "xoption":c
                            }
                            i=i+1
                elif deffered['xtype']=="xpayment":
                    payment_resources_map=self.payment_options_calculator.create_basic_payment(0)
                    choosen_payment=deffered.get('choosen_payment',{})
                    payment_resources_min_max_map={}
                    new_x_options=[]
                    for p in deffered['xoptions']:
                        skip_option=False
                        for r in choosen_payment.keys():
                            if choosen_payment[r]!=p[r]:
                                skip_option=True
                        if skip_option:
                            continue
                        new_x_options.append(p)
                        for r in payment_resources_map.keys():
                            v=p[r]
                            if v>0:
                                if r not in payment_resources_min_max_map:
                                    payment_resources_min_max_map[r]={}
                                payment_resources_min_max_map[r][v]=True
                    deffered['xoptions']=new_x_options
                    if len(new_x_options)==1:
                        action_space[i]={
                                "type":"deffered",
                                "xid":deffered["xid"],
                                "xtype":"xpayment_confirm",
                                "xoption":new_x_options[0]
                            }
                        i=i+1
                    else:
                        selected_payment_resource=deffered.get('selected',None)
                        if selected_payment_resource is None:
                            for p in payment_resources_min_max_map.keys():
                                if p in choosen_payment:
                                    continue
                                action_space[i]={
                                    "type":"deffered",
                                    "xid":deffered["xid"],
                                    "xtype":"xpayment_choose_resource",
                                    "xoption":p
                                }
                                i=i+1
                        else:
                            for iii in payment_resources_min_max_map[selected_payment_resource].keys():
                                action_space[i]={
                                    "type":"deffered",
                                    "xid":deffered["xid"],
                                    "xtype":"xpayment_choose_amount",
                                    "xres":selected_payment_resource,
                                    "xoption":int(iii)
                                }
                                i=i+1
                elif deffered['xtype'] in ("xspace_choose_level0", "xspace_choose_level1", "xspace_choose_level2"):
                    # --- helpers ------------------------------------------------
                    # Level-0 buckets:  [0-19] [20-39] [40-59] [60-79] [80-99]
                    # Level-1 buckets within a level-0 bucket of size 20:
                    #   sub 0: [base+0 .. base+3]
                    #   sub 1: [base+4 .. base+7]
                    #   sub 2: [base+8 .. base+11]
                    #   sub 3: [base+12 .. base+15]
                    #   sub 4: [base+16 .. base+19]
                    # Level-2: exact space_id selection within a sub-bucket of 4.

                    valid_spaces = set(deffered['xoptions'])  # original full set

                    def spaces_in_range(lo, hi):
                        """Return sorted list of valid space_ids in [lo, hi)."""
                        return sorted(s for s in valid_spaces if lo <= int(s) < hi)

                    if deffered['xtype'] == "xspace_choose_level0":
                        # Present up to 5 level-0 bucket choices (only those
                        # containing at least one valid space_id).
                        for bucket_idx in range(5):
                            lo = bucket_idx * 20
                            hi = lo + 20
                            if spaces_in_range(lo, hi):
                                action_space[i] = {
                                    "type": "deffered",
                                    "xid": deffered["xid"],
                                    "xtype": "xspace_choose_level0",
                                    "xoption": bucket_idx  # which 20-wide bucket
                                }
                                i += 1

                    elif deffered['xtype'] == "xspace_choose_level1":
                        # A level-0 bucket was already chosen; present up to 5
                        # sub-bucket choices within it.
                        bucket_idx = deffered['xselected_level0']
                        base = bucket_idx * 20
                        for sub_idx in range(5):
                            lo = base + sub_idx * 4
                            hi = lo + 4
                            if spaces_in_range(lo, hi):
                                action_space[i] = {
                                    "type": "deffered",
                                    "xid": deffered["xid"],
                                    "xtype": "xspace_choose_level1",
                                    "xoption": sub_idx  # which 4-wide sub-bucket
                                }
                                i += 1

                    elif deffered['xtype'] == "xspace_choose_level2":
                        # Both bucket and sub-bucket chosen; present the exact
                        # valid space_ids (up to 4).
                        bucket_idx = deffered['xselected_level0']
                        sub_idx   = deffered['xselected_level1']
                        base = bucket_idx * 20 + sub_idx * 4
                        for sid in spaces_in_range(base, base + 4):
                            action_space[i] = {
                                "type": "deffered",
                                "xid": deffered["xid"],
                                "xtype": "xspace_choose_level2",
                                "xoption": sid  # the actual space_id
                            }
                            i += 1
                return action_space
            else:
                raise Exception("Bad first_deffered_action")


        input_type = player_input.get("type")
        if input_type == "and":
            options = player_input.get("options", [])
            if not options:
                return {0: {"type": "and", "responses": []}}
            sub_action_spaces = [self.generate_action_space(opt,player_state,False) for opt in options]
            action_indices = product(*[range(len(space)) for space in sub_action_spaces])
            is_limited_amount=len(options)>0
            for opt in options:
                is_limited_amount &=((opt.get('buttonLabel') in ["Select"]) or ('Spend' in opt.get('buttonLabel'))) and opt.get('type')=="amount"
            k=0
            for i, indices in enumerate(action_indices):
                responses=[]
                for j, idx in enumerate(indices):
                    r=None
                    if  "responses" in sub_action_spaces[j][idx]:
                        r=sub_action_spaces[j][idx]["responses"][0] 
                    else:
                        r=sub_action_spaces[j][idx]
                    responses.append(r.copy())               
                
                value=0
                
                for rr in responses:
                    if rr.get('type',"")=="amount" and isinstance(rr,dict) and rr.get('value',-1)>=0:
                        value+=rr.get('value',0)
                        del rr['value']
                if is_limited_amount:
                    limit_value=int(player_input.get('title',{}).get('data',[{}])[0].get('value'))
                    if isinstance(player_input.get('title',''),dict) and  player_input.get('title',{}).get('message','')=="Select how to spend ${0} heat":
                        if value==limit_value:
                            action_space[k] = {"type": "and", "responses": responses}
                            k=k+1
                    else:
                        if value==limit_value:
                            action_space[k] = {"type": "and", "responses": responses}
                            k=k+1
                else:
                    action_space[k] = {"type": "and", "responses": responses}
                    k=k+1
                

        elif input_type == "or":
            # For OR options, each option is a separate action
            options = player_input.get("options", [])
            i=0

            for idx, option in enumerate(options):
                sub_action_spaces = self.generate_action_space(option,player_state,False)
                for sub_option_idx in sub_action_spaces:
                    r=sub_action_spaces[sub_option_idx]
                    if isinstance(r,dict) and r.get('type','') == 'amount' and r.get('value',-1)>=0:
                        del r['value']
                    action_space[i] = {"type":"or",
                                       "index":idx,
                                       "response":sub_action_spaces[sub_option_idx]}
                    i+=1
            for zz in action_space.keys():
                if 'response' in action_space[zz] and 'type' in action_space[zz]['response'] and action_space[zz]['response']['type']=='space':
                    has_space=True
                    break               
                
        
        elif input_type == "initialCards":
            # For initial cards selection, we need to handle multiple card selections
            options = player_input.get("options", [])
            if not options:
                return {0: {"type": "initialCards", "responses": []}}
            
            # Generate action spaces for each card selection option
            sub_action_spaces = [self.generate_action_space(opt,player_state,False) for opt in options]
            
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
            min_cards = player_input.get("min", 0)
            max_cards = player_input.get("max", len(cards))
            #print(f"max_cards={max_cards} min_cards={min_cards} cards={cards}")
            if max_cards<1:
                print(f"BAD MAX CARDS player_input={player_input}")
                print(f"player_input={player_input}\r\nplayer_state={player_state}")
            #print(f"cards in card: {[card['name'] for card in cards]}")
            if len(cards)>0 and max_cards>0:
                action_space[0]={
                    "type": "card",
                    "cards":{"__deferred_action": {
                                    "xid":str(uuid4()),
                                    "xtype": "xcard_choose",
                                    "xoptions":cards,
                                    "xmin":min_cards,
                                    "xmax":max_cards
                                }
                    }
                }

        
        elif input_type == "amount":
            # For amount selection, each possible amount is a separate action
            min_amount = player_input.get("min", 0)
            
            value_coeff=1
            if player_input.get('title',"")=="Stormcraft Incorporated Floaters (2 heat each)":
                value_coeff=2
            max_amount = player_input.get("max", 0)
            j=0
            for amount in range(min_amount, max_amount + 1):
                action_space[j] = {
                    "type": "amount",
                    "amount": amount
                }                 
                if not is_main:
                    action_space[j]["value"]=amount * value_coeff
                j=j+1
        
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
                    #if item['name'] in cards_in_hand:
                    #    continue
                    payments = self.payment_options_calculator.create_payment_from_input(player_input, item["name"], items,player_state)
                    
                    #for payment in payments:
                    action_space[j] = {
                        "type": "projectCard",
                        "card": item["name"],
                        "payment": {
                            "__deferred_action": {
                                "xid":str(uuid4()),
                                "xtype": "xpayment",
                                "xoptions":payments,
                                "xoriginaloptions":payments
                            }
                        }
                    }
                    j+=1
                elif input_type == "colony":
                    action_space[i] = {
                        "type": input_type,
                        "colonyName": item["name"]
                    }
                elif input_type == "party":
                    action_space[i] = {
                        "type": input_type,
                        "partyName": item
                    }
                elif input_type == "delegate":
                    action_space[i] = {
                        "type": input_type,
                        "player": item
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
            if len(spaces) > 0:
                action_space[0] = {
                    "type": "space",
                    "spaceId": {
                        "__deferred_action": {
                            "xid": str(uuid4()),
                            "xtype": "xspace_choose_level0",
                            "xoptions": spaces  # full list of valid space_ids
                        }
                    }
                }
        
        elif input_type == "resources":
            # For resource selection, we'll just return the default for now
            action_space[0] = self._map_select_resources(player_input)
        elif input_type=="payment":
            selected_card=None
            cards=[]
            try:
                if 'standard project' in player_input.get('title',{}).get('message','') and player_input.get('title',{}).get('data',[{}])[0].get('type',0)==3:
                    selected_card=player_input.get('title',{}).get('data',[{}])[0].get('value','')
            except:
                pass

            payments=self.payment_options_calculator.create_payment_from_input(player_input,selected_card,[],player_state=player_state)

            #for i,payment in enumerate(payments):
            action_space[0] = {
                "type":"payment",
                "payment": {
                        "__deferred_action": {
                            "xid":str(uuid4()),
                            "xtype": "xpayment",
                            "xoptions":payments,
                            "xoriginaloptions":payments
                        }
                    }
            }

        else:
            # For other types, just return the default mapped decision
            action_space[0] = self.map_decision(player_input)
        
        return action_space
    

    
    def _map_select_option(self, player_input: Dict[str, Any]) -> Dict[str, Any]:
        """Map SelectOptionModel to SelectOptionResponse."""
        return {
            "type": "option"
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
        