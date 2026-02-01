import numpy as np
import json
from deepdiff import DeepDiff

def reward_function(player_state_before: dict, player_state_after: dict) -> float:
    """
    IMPROVED: Dense shaped reward for Terraforming Mars RL agent.
    Better balanced weights and reward shaping.
    """
    player_before = player_state_before['thisPlayer']
    game_before = player_state_before['game']
    player_after = player_state_after['thisPlayer']
    game_after = player_state_after['game']
    reward_desc = {}
    reward = 0.0
    
    # Clean timer data for comparison
    for d in [player_before, player_after]:
        d['timer'] = {}
        d['needsToDraft'] = False
    
    # Quick check for no changes
    if hash(json.dumps(player_before, sort_keys=True)) == hash(json.dumps(player_after, sort_keys=True)):
        return 0.0

    # Get generation for scaling
    generation = game_after.get("generation", 1)
    max_generation = game_after.get("lastSoloGeneration", 14)
    game_progress = generation / max_generation  # 0 to 1
    
    # ============================
    # 1. Victory Points (Primary Goal)
    # ============================
    vp_before = player_before["victoryPointsBreakdown"]["total"]
    vp_after = player_after["victoryPointsBreakdown"]["total"]
    vp_delta = vp_after - vp_before

    # IMPROVED: Progressive VP importance scaling
    early_game_weight = 1.5  # Generations 1-5
    mid_game_weight = 2.0    # Generations 6-10
    late_game_weight = 3.0   # Generations 11+
    
    if generation <= 5:
        vp_weight = early_game_weight
    elif generation <= 10:
        vp_weight = mid_game_weight
    else:
        vp_weight = late_game_weight
    
    reward_desc['vp_delta'] = vp_delta * vp_weight
    reward += vp_delta * vp_weight

    # ============================
    # 2. Terraforming Rating
    # ============================
    tr_before = player_before["terraformRating"]
    tr_after = player_after["terraformRating"]
    tr_delta = tr_after - tr_before
    
    # IMPROVED: TR is more valuable early (generates income)
    tr_weight = 2.0 - (game_progress * 0.5)  # 2.0 early, 1.5 late
    reward_desc['tr_delta'] = tr_delta * tr_weight
    reward += tr_delta * tr_weight

    # ============================
    # 3. Global Parameter Progress
    # ============================
    if game_before and game_after:
        # IMPROVED: Weight based on how close we are to completion
        oxygen_progress = game_after["oxygenLevel"] / 14.0
        temp_progress = (game_after["temperature"] + 30) / 82.0
        ocean_progress = game_after["oceans"] / 9.0
        
        # Bonus for contributing when parameters are not maxed
        oxygen_delta = game_after["oxygenLevel"] - game_before["oxygenLevel"]
        if oxygen_delta > 0 and oxygen_progress < 1.0:
            reward_desc['oxygenLevel'] = oxygen_delta * 0.8
            reward += oxygen_delta * 0.8
        
        ocean_delta = game_after["oceans"] - game_before["oceans"]
        if ocean_delta > 0 and ocean_progress < 1.0:
            reward_desc['oceans'] = ocean_delta * 0.8
            reward += ocean_delta * 0.8
        
        temp_delta = game_after["temperature"] - game_before["temperature"]
        if temp_delta > 0 and temp_progress < 1.0:
            reward_desc['temperature'] = temp_delta * 0.06
            reward += temp_delta * 0.06

    # ============================
    # 4. Production Engine (Critical)
    # ============================
    prod_keys = [
        "megaCreditProduction", "steelProduction", "titaniumProduction",
        "plantProduction", "energyProduction", "heatProduction"
    ]

    prod_gain = 0
    for key in prod_keys:
        delta = player_after[key] - player_before[key]
        prod_gain += max(0, delta)

    # IMPROVED: Production is VERY valuable early, less so late
    prod_weight = 1.2 - (game_progress * 0.4)  # 1.2 early, 0.8 late
    reward_desc['prod_gain'] = prod_gain * prod_weight
    reward += prod_gain * prod_weight

    # Engine diversity bonus - encourages balanced production
    diversified = sum(player_after[k] > 0 for k in prod_keys)
    reward_desc['diversified'] = diversified * 0.1
    reward += diversified * 0.1

    # NEW: Penalize production loss
    prod_loss = 0
    for key in prod_keys:
        delta = player_after[key] - player_before[key]
        if delta < 0:
            prod_loss += abs(delta)
    if prod_loss > 0:
        reward_desc['prod_loss'] = -prod_loss * 0.8
        reward -= prod_loss * 0.8

    # ============================
    # 5. Economy Efficiency
    # ============================
    mc_before = player_before["megaCredits"]
    mc_after = player_after["megaCredits"]

    # IMPROVED: Context-aware spending rewards
    mc_spent = max(0, mc_before - mc_after)
    
    # Reward spending if we got something valuable
    if mc_spent > 0:
        # Check if we gained production, cards, or VP
        got_value = (
            prod_gain > 0 or 
            vp_delta > 0 or 
            len(player_after["tableau"]) > len(player_before["tableau"])
        )
        
        if got_value:
            # Scale reward by amount spent (diminishing returns)
            spend_reward = min(mc_spent * 0.04, 2.0)
            reward_desc['productive_spending'] = spend_reward
            reward += spend_reward

    # IMPROVED: Dynamic hoarding penalty based on game phase
    if generation >= 8:  # Late game
        hoarding_threshold = 15
    else:  # Early/mid game
        hoarding_threshold = 30
    
    if mc_after > hoarding_threshold:
        penalty = (mc_after - hoarding_threshold) * 0.008
        reward_desc['hoarding'] = -penalty
        reward -= penalty

    # NEW: Bankruptcy penalty
    if mc_after < 0:
        reward_desc['bankruptcy'] = -1.0
        reward -= 1.0

    # ============================
    # 6. Resource Conversion (Strategic)
    # ============================
    # Heat conversion - check if global temperature not maxed
    can_raise_temp = game_after.get("temperature", 0) < 8
    
    if player_before["heat"] >= 8 and can_raise_temp:
        heat_converted = player_before["heat"] - player_after["heat"]
        if heat_converted >= 8:
            reward_desc['heat_conversion'] = 1.0
            reward += 1.0
        elif heat_converted == 0:
            # Missed opportunity
            reward_desc['heat_conversion_miss'] = -0.3
            reward -= 0.3

    # Plant conversion - check if oxygen not maxed
    can_raise_oxygen = game_after.get("oxygenLevel", 0) < 14
    
    if player_before["plants"] >= 8 and can_raise_oxygen:
        plants_converted = player_before["plants"] - player_after["plants"]
        if plants_converted >= 8:
            reward_desc['plants_conversion'] = 1.0
            reward += 1.0
        elif plants_converted == 0:
            # Missed opportunity
            reward_desc['plants_conversion_miss'] = -0.3
            reward -= 0.3

    # ============================
    # 7. Board Control
    # ============================
    cities_delta = player_after["citiesCount"] - player_before["citiesCount"]
    colonies_delta = player_after["coloniesCount"] - player_before["coloniesCount"]
    
    reward_desc['citiesCount'] = cities_delta * 1.2
    reward += cities_delta * 1.2
    
    reward_desc['coloniesCount'] = colonies_delta * 1.0
    reward += colonies_delta * 1.0

    # ============================
    # 8. Card Engine Development
    # ============================
    tableau_before = len(player_before["tableau"])
    tableau_after = len(player_after["tableau"])
    tableau_delta = tableau_after - tableau_before
    
    # IMPROVED: Card value depends on cost efficiency
    if tableau_delta > 0:
        # Check if we overpaid (simple heuristic)
        avg_card_cost = mc_spent / tableau_delta if tableau_delta > 0 else 0
        
        if avg_card_cost <= 15:  # Good deal
            reward_desc['tableau_delta'] = tableau_delta * 1.2
            reward += tableau_delta * 1.2
        else:  # Expensive cards
            reward_desc['tableau_delta'] = tableau_delta * 0.8
            reward += tableau_delta * 0.8
    elif tableau_delta < 0:
        # Losing cards is bad (unless it's a beneficial trade)
        reward_desc['tableau_loss'] = tableau_delta * 0.5
        reward += tableau_delta * 0.5

    # ============================
    # 9. Action Tempo
    # ============================
    actions_delta = player_after["actionsTakenThisRound"] - player_before["actionsTakenThisRound"]
    reward_desc['actionsTakenThisRound'] = actions_delta * 0.05
    reward += actions_delta * 0.05

    # ============================
    # 10. Political Power (Turmoil)
    # ============================
    influence_delta = player_after["influence"] - player_before["influence"]
    reward_desc['influence'] = influence_delta * 0.4
    reward += influence_delta * 0.4

    # ============================
    # 11. Corruption Penalty
    # ============================
    corruption_delta = player_after["corruption"] - player_before["corruption"]
    reward_desc['corruption_penalty'] = -corruption_delta * 2.5
    reward -= corruption_delta * 2.5

    # Corruption existence penalty (scales with amount)
    if player_after["corruption"] > 0:
        reward_desc['corruption_total'] = -player_after["corruption"] * 0.3
        reward -= player_after["corruption"] * 0.3

    # ============================
    # 12. Trade Efficiency
    # ============================
    trades_delta = player_after["tradesThisGeneration"] - player_before["tradesThisGeneration"]
    reward_desc['trades_delta'] = trades_delta * 0.5
    reward += trades_delta * 0.5

    # ============================
    # 13. Blue Card Actions
    # ============================
    blue_actions_delta = player_after["availableBlueCardActionCount"] - player_before["availableBlueCardActionCount"]
    reward_desc['availableBlueCardActionCount'] = blue_actions_delta * 0.25
    reward += blue_actions_delta * 0.25

    # ============================
    # 14. Hand Management
    # ============================
    hand_after = player_after["cardsInHandNbr"]
    hand_before = player_before["cardsInHandNbr"]

    # IMPROVED: Optimal hand size depends on game phase
    if generation <= 5:
        optimal_hand = 8
    elif generation <= 10:
        optimal_hand = 6
    else:
        optimal_hand = 4  # Late game, play your cards
    
    hand_deviation = abs(hand_after - optimal_hand)
    reward_desc['hand_management'] = -hand_deviation * 0.05
    reward -= hand_deviation * 0.05

    # ============================
    # 15. Card Discounts and Efficiency
    # ============================
    discount_delta = player_after["cardDiscount"] - player_before["cardDiscount"]
    reward_desc['cardDiscount'] = discount_delta * 0.5
    reward += discount_delta * 0.5

    steel_value_delta = player_after["steelValue"] - player_before["steelValue"]
    reward_desc['steelValue'] = steel_value_delta * 0.3
    reward += steel_value_delta * 0.3

    titanium_value_delta = player_after["titaniumValue"] - player_before["titaniumValue"]
    reward_desc['titaniumValue'] = titanium_value_delta * 0.3
    reward += titanium_value_delta * 0.3

    # ============================
    # 16. Tag Strategy
    # ============================
    tags_before = player_before["tags"]
    tags_after = player_after["tags"]

    # Tag diversity (more important early)
    diversity_before = sum(v > 0 for v in tags_before.values())
    diversity_after = sum(v > 0 for v in tags_after.values())
    diversity_delta = diversity_after - diversity_before
    
    diversity_weight = 0.3 - (game_progress * 0.1)
    reward_desc['diversity_tags'] = diversity_delta * diversity_weight
    reward += diversity_delta * diversity_delta

    # Specific valuable tags
    jovian_delta = tags_after.get("jovian", 0) - tags_before.get("jovian", 0)
    reward_desc['jovian'] = jovian_delta * 0.6
    reward += jovian_delta * 0.6

    science_delta = tags_after.get("science", 0) - tags_before.get("science", 0)
    reward_desc['science'] = science_delta * 0.4
    reward += science_delta * 0.4

    # ============================
    # 17. Strategic Passing
    # ============================
    if player_after.get("isActive") is False and player_before.get("isActive") is True:
        # Passing is strategic if we've done productive work
        if player_before["actionsTakenThisRound"] >= 2:
            reward_desc['strategic_passing'] = 0.3
            reward += 0.3

    # ============================
    # 18. Stagnation Penalty
    # ============================
    cards_in_hand_delta = player_after.get('cardsInHandNbr', 0) - player_before.get('cardsInHandNbr', 0)
    
    is_stagnant = (
        vp_delta == 0 and
        tr_delta == 0 and
        prod_gain == 0 and
        tableau_after == tableau_before and
        cards_in_hand_delta == 0
    )
    
    if is_stagnant:
        reward_desc['stagnation_penalty'] = -0.2
        reward -= 0.2

    # ============================
    # 19. VP Momentum
    # ============================
    vp_hist = player_after.get("victoryPointsByGeneration", [])
    if len(vp_hist) >= 2:
        momentum = vp_hist[-1] - vp_hist[-2]
        reward_desc['vp_momentum'] = momentum * 0.3
        reward += momentum * 0.3

    # ============================
    # 20. Synergy Bonuses
    # ============================
    # City + Greenery synergy (cities next to greeneries are valuable)
    if cities_delta > 0 and tr_delta > 0:
        reward_desc['city_greenery_synergy'] = 0.4
        reward += 0.4

    # ============================
    # 21. Fleet Utilization
    # ============================
    if player_after["fleetSize"] > 0:
        utilization = player_after["tradesThisGeneration"] / player_after["fleetSize"]
        reward_desc['fleet_utilization'] = utilization * 0.25
        reward += utilization * 0.25

    # ============================
    # 22. Production Balance
    # ============================
    # IMPROVED: Only penalize severe imbalance
    prod_vals = [player_after[k] for k in prod_keys]
    if len(prod_vals) > 0 and max(prod_vals) > 0:
        imbalance = np.std(prod_vals) / (np.mean(prod_vals) + 1e-8)
        if imbalance > 2.0:  # Only penalize severe imbalance
            reward_desc['production_imbalance'] = -imbalance * 0.05
            reward -= imbalance * 0.05

    # ============================
    # 23. Final Reward Normalization
    # ============================
    # IMPROVED: Wider range for better learning signal
    reward = float(np.clip(reward, -15.0, 15.0))
    reward_desc['total'] = reward
    
    # Optional detailed logging (uncomment for debugging)
    # reward_desc_filtered = {k: v for k, v in sorted(reward_desc.items(), key=lambda item: abs(item[1]), reverse=True) if abs(v) > 0.01}
    # print(f"Rewards={reward:.3f} Components={reward_desc_filtered}")
    
    return float(reward)