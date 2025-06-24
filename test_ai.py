from itertools import product
import re

PAYMENT_RESOURCES = ["megaCredits", "heat", "steel", "titanium", "plants", "microbes", "floaters", 
                    "lunaArchivesScience", "spireScience", "seeds", "auroraiData", "graphene", 
                    "kuiperAsteroids", "corruption"]

def get_payment_combinations(card_cost, resource_availability):
    # Group resources by their type and sort by valueOfOne in descending order
    resource_groups = {}
    
    for item in resource_availability:
        resource = item["resource"]+"|"+str(item["valueOfOne"])
        max_available = item["max"]
        value = item["valueOfOne"]
        
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

# Example usage
card_cost = 6
resource_availability = [
    {"resource": "megaCredits", "max": 4, "valueOfOne": 1},
    {"resource": "steel", "max": 4, "valueOfOne": 1},
    {"resource": "steel", "max": 2, "valueOfOne": 2}
]

combinations = get_payment_combinations(card_cost, resource_availability)
print("Result:")
for i in combinations:
    print({k:v for k,v in i.items() if i[k]>0})