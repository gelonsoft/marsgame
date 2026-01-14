from itertools import product
import re
from typing import Any, Dict, List
from myconfig import ALL_CARDS


PAYMENT_RESOURCES=["megaCredits","heat","steel","titanium","plants","microbes","floaters","lunaArchivesScience","spireScience","seeds","auroraiData","graphene","kuiperAsteroids","corruption"]

class PaymentOptionsCalculator:
    
    def __init__(self):
        
        self.resource_mapper_functions={
            "megaCredits": self._get_megaCredits,
            "heat": self._get_heat,
            "steel": self._get_steel,
            "titanium": self._get_titanium,
            "plants": self._get_plants,
            "microbes": self._get_microbes,
            "floaters": self._get_floaters,
            "lunaArchivesScience": self._get_lunaArchivesScience,
            "spireScience": self._get_spireScience,
            "seeds": self._get_seeds,
            "auroraiData": self._get_auroraiData,
            "graphene": self._get_graphene,
            "kuiperAsteroids": self._get_kuiperAsteroids,
            "corruption": self._get_corruption
        }
    
    def _get_megaCredits(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        player_have=player_state["thisPlayer"].get(resource, 0)
        if player_have>0:
            max_value=min(card_cost,player_have)
            current_payment_options.append({"resource":resource,"max":max_value,"value":1})
        return current_payment_options
    def _get_heat(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        if "Helion" in [n.get('name') for n in player_state.get('thisPlayer',{}).get('tableau',[])]:
            player_have=player_state["thisPlayer"].get(resource, 0)
            if player_have>0:
                max_amount=min(card_cost,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":1})
        return current_payment_options
    def _get_steel(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        if "building" in card_metadata.get("tags",[]):
            player_have=player_state["thisPlayer"].get(resource, 0)
            if player_have>0:
                value=player_state["thisPlayer"].get("steelValue", 1)
                max_amount=min(card_cost//value+card_cost%value,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        elif player_input.get('type',"")=="payment" and player_input.get('paymentOptions',{}).get(resource):
            player_have=player_state["thisPlayer"].get(resource, 0)
            if player_have>0:
                value=player_state["thisPlayer"].get(resource+"Value", 1)
                max_amount=min(card_cost//value+card_cost%value,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_titanium(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        #print(f"Player input: {player_input} {'title' in player_input} {player_input.get('title',"a") is dict}")
        if "space" in card_metadata.get("tags",[]):
            player_have=player_state["thisPlayer"].get(resource, 0)
            if player_have>0:
                value=player_state["thisPlayer"].get("titaniumValue", 1)
                max_amount=min(card_cost//value+card_cost%value,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        elif "Luna Trade Federation" in [n.get('name') for n in player_state.get('thisPlayer',{}).get('tableau',[])]:
            player_have=player_state["thisPlayer"].get(resource, 0)
            if player_have>0:
                value=2
                max_amount=min(card_cost//value+card_cost%value,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        elif (('title' in player_input) and (isinstance(player_input.get('title',"a"), dict)) and ("Directed Impactors"==player_input.get('title',{}).get('data',[{"value":None}])[0].get('value',None))):
            player_have=player_state["thisPlayer"].get(resource, 0)
            if player_have>0:
                value=player_state["thisPlayer"].get(resource+"Value", 1)
                max_amount=min(card_cost//value+card_cost%value,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        elif player_input.get('type',"")=="payment" and player_input.get('paymentOptions',{}).get(resource):
            player_have=player_state["thisPlayer"].get(resource, 0)
            if player_have>0:
                value=player_state["thisPlayer"].get(resource+"Value", 1)
                max_amount=min(card_cost//value+card_cost%value,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_plants(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        for played_card in player_state["thisPlayer"].get("tableau",[]):
            if played_card.get("name")=="Martian Lumber Corp" and "building" in card_metadata.get("tags",[]):
                player_have=player_state["thisPlayer"].get(resource, 0)
                if player_have>0:
                    value=3
                    max_amount=min(card_cost//value+card_cost%value,player_have)
                    current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_microbes(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        for played_card in player_state["thisPlayer"].get("tableau",[]):
            if played_card.get("name")=="Psychrophiles" and "plant" in card_metadata.get("tags",[]):
                player_have=played_card.get("resources",0)
                if player_have>0:
                    value=2
                    max_amount=min(card_cost//value+card_cost%value,player_have)
                    current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_floaters(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        for played_card in player_state["thisPlayer"].get("tableau",[]):
            if played_card.get("name")=="Dirigibles" and "venus" in card_metadata.get("tags",[]):
                player_have=played_card.get("resources",0)
                if player_have>0:
                    value=3
                    max_amount=min(card_cost//value+card_cost%value,player_have)
                    current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_lunaArchivesScience(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        for played_card in player_state["thisPlayer"].get("tableau",[]):
            if played_card.get("name")=="Luna Archives" and "moon" in card_metadata.get("tags",[]):
                player_have=played_card.get("resources",0)
                if player_have>0:
                    max_amount=min(card_cost,player_have)
                    current_payment_options.append({"resource":resource,"max":max_amount,"value":1})
        return current_payment_options
    def _get_spireScience(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        if "Spire" in [n.get('name') for n in player_state.get('thisPlayer',{}).get('tableau',[])]:
            if card_metadata.get('type','')=="standard_project":
                player_have=player_input.get(resource, 0)
                if player_have>0:
                    value=2
                    max_amount=min(card_cost//value+card_cost%value,player_have)
                    current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
            elif player_input.get('type',"")=="payment" and player_input.get('paymentOptions',{}).get(resource):
                    for t in player_state.get('thisPlayer',{}).get('tableau',[]):
                        if t.get('name')=="Spire":
                            player_have=t.get('resources',0)
                            if player_have>0:
                                value=2
                                max_amount=min(card_cost//value+card_cost%value,player_have)
                                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_seeds(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        if "Soylent Seedling Systems" in [n.get('name') for n in player_state.get('thisPlayer',{}).get('tableau',[])] and (selected_card=="Greenery" or "plant" in card_metadata.get("tags",[])):
            player_have=player_input.get(resource, 0)
            if player_have>0:
                value=5
                max_amount=min(card_cost//value+card_cost%value,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_auroraiData(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        if "Aurorai" in [n.get('name') for n in player_state.get('thisPlayer',{}).get('tableau',[])] and card_metadata.get('type','')=="standard_project":
            player_have=player_input.get(resource, 0)
            if player_have>0:
                value=3
                max_amount=min(card_cost//value+card_cost%value,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_graphene(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        for played_card in player_state["thisPlayer"].get("tableau",[]):
            if played_card.get("name")=="Carbon Nanosystems" and ("space" in card_metadata.get("tags",[]) or "city" in card_metadata.get("tags",[])):
                player_have=played_card.get("resources",0)
                if player_have>0:
                    value=4
                    max_amount=min(card_cost//value+card_cost%value,player_have)
                    current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_kuiperAsteroids(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        #z=player_input.get('title',{}).get('data',[{'value':''}])
        #print(f"Checking Kuiper Cooperative for {selected_card} with {z}")
        #print(f"player_input={player_input}")
        if "Kuiper Cooperative" in [n.get('name') for n in player_state.get('thisPlayer',{}).get('tableau',[])] and (
            selected_card=="Aquifer" or selected_card=="Asteroid:SP" or (
                'title' in player_input and isinstance(player_input.get('title',""),dict) and (player_input.get('title',{}).get('data',[{'value':''}])[0].get('value','') in ["Asteroid:SP","Aquifer"])
                )
            ):
            #print("koiper")
            player_have=player_input.get(resource, 0)
            if player_have>0:
                max_amount=min(card_cost,player_have)
                current_payment_options.append({"resource":resource,"max":max_amount,"value":1}) 
        return current_payment_options
    def _get_corruption(self,resource,selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict,current_payment_options:List[Dict]) -> List[Dict]:
        for played_card in player_state["thisPlayer"].get("tableau",[]):
            if played_card.get("name")=="Friends in High Places" and "earth" in card_metadata.get("tags",[]):
                player_have=played_card.get("resources",0)
                if player_have>0:
                    value=10
                    max_amount=min(card_cost//value+card_cost%value,player_have)
                    current_payment_options.append({"resource":resource,"max":max_amount,"value":value})
        return current_payment_options
    def _get_resources_payment_options(self, selected_card: str,card_cost: int, card_metadata: Dict,player_input: Dict, player_state: Dict) -> List[Dict]:
        payment_options=[]
        for r in PAYMENT_RESOURCES:
            payment_options=self.resource_mapper_functions[r](r,selected_card,card_cost, card_metadata,player_input, player_state,payment_options)
            
        return payment_options

    
    def create_payment_from_input(self, player_input: Dict, selected_card: str, cards: List[Dict], player_state: Dict) -> List[Dict]:
        """Generate all valid payment combinations for card playment based on available resources."""
        # Find the selected card to get its cost
        card_cost = -1
        card_metadata={}
        if selected_card:
            if cards:
                for card in cards:
                    if card.get("name") == selected_card:
                        card_cost = card.get("calculatedCost", 0)
                        break
            if card_cost<0:
                card_cost=player_input.get('amount',0)
            card_metadata=ALL_CARDS[selected_card]
            
        else:
            card_cost=player_input.get('amount',0)
            
        if card_cost <= 0:
            return [self.create_basic_payment(0)]
        
        resource_availability=self._get_resources_payment_options(selected_card,card_cost,card_metadata,player_input,player_state)
        #print(f"resource_availability={resource_availability}")
        return self.get_payment_combinations(card_cost,resource_availability)
        
    def get_payment_combinations(self,card_cost, resource_availability):
        # Group resources by their type and sort by valueOfOne in descending order
        resource_groups = {}
        
        for item in resource_availability:
            resource = item["resource"]+"|"+str(item["value"])
            max_available = item["max"]
            value = item["value"]
            
            if resource not in resource_groups:
                resource_groups[resource] = []
            resource_groups[resource].append((max_available, value))
        
        # Sort each resource's entries by valueOfOne in descending order
        for resource in resource_groups:
            resource_groups[resource].sort(key=lambda x: -x[1])
        # Prepare all possible (units, contribution) for each resource, prioritizing higher valueOfOne
        resource_contributions = {}
        for resource in resource_groups:
            possible_units = []
            for max_available, value in resource_groups[resource]:
                for units in range(0, max_available + 1):
                    contribution = units * value
                    possible_units.append((units, contribution))
            resource_contributions[resource] = possible_units
        
        # Generate all combinations of resource contributions
        resources = list(resource_contributions.keys())
        value_combinations = product(*[resource_contributions[r] for r in resources])
        
        
        #print("All combinations:")
        #for combo in value_combinations:
        #    payment = {resource: 0 for resource in PAYMENT_RESOURCES}
        #    for i, resource in enumerate(resources):
        #        units, _ = combo[i]
        #        payment[re.sub(r"\|.*","",resource)] += units
        #        z={k:v for k,v in payment.items() if payment[k]>0}
        #        print(z)   
        #value_combinations = product(*[resource_contributions[r] for r in resources])
        #jj=0
        valid_combinations = []
        for combo in value_combinations:
            total = sum(contribution for (units, contribution) in combo)
            
            #print(f"Compare: {total}>={card_cost} is {total >= card_cost} for combo {combo}")
            
            if total >= card_cost:
                payment = {resource: 0 for resource in PAYMENT_RESOURCES}
                for i, resource in enumerate(resources):
                    units, _ = combo[i]
                    payment[re.sub(r"\|.*","",resource)] += units
                valid_combinations.append(payment)
            #jj+=1
        #print("Value combinations:")
        #for i in valid_combinations:
        #    print({k:v for k,v in i.items() if i[k]>0})
            
        # Filter out redundant combinations (supersets)
        minimal_combinations = []
        for combo in valid_combinations:
            is_minimal = True
            for other in valid_combinations:
                if combo == other:
                    continue
                # Check if 'other' is a subset of 'combo' (i.e., combo uses more resources)
                is_superset = all(combo[r] >= other[r] for r in PAYMENT_RESOURCES)
                is_strict_superset = any(combo[r] > other[r] for r in PAYMENT_RESOURCES)
                if is_superset and is_strict_superset:
                    is_minimal = False
                    break
            if is_minimal:
                minimal_combinations.append(combo)
        # Remove duplicate combinations
        unique_combinations = []
        seen = set()
        for combo in minimal_combinations:
            # Convert the combination to a tuple of tuples for hashing
            combo_tuple = tuple((r, combo[r]) for r in PAYMENT_RESOURCES if combo[r] > 0)
            combo_tuple = tuple(sorted(combo_tuple))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_combinations.append(combo)
                
        return minimal_combinations
    
    def create_basic_payment(self, amount: int) -> Dict[str, Any]:
        """Create a basic payment object with the specified amount in megacredits."""
        return {
            "megaCredits": amount,
            "steel": 0,
            "titanium": 0,
            "heat": 0,
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