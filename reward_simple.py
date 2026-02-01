import numpy as np
import json

def reward_function(player_state_before: dict, player_state_after: dict) -> float:
    """
    Simplified reward function with fewer components for easier learning.
    Focus on the most important metrics that correlate with winning.
    """
    player_before = player_state_before['thisPlayer']
    player_after = player_state_after['thisPlayer']
    game_after = player_state_after.get('game', {})
    game_before = player_state_before.get('game', {})
    
    # Clean timer data
    for d in [player_before, player_after]:
        d['timer'] = {}
        d['needsToDraft'] = False
    
    # Quick check for no changes
    if hash(json.dumps(player_before, sort_keys=True)) == hash(json.dumps(player_after, sort_keys=True)):
        return 0.0
    
    reward = 0.0
    generation = game_after.get("generation", 1)
    
    # ========================================
    # COMPONENT 1: Victory Points (PRIMARY)
    # ========================================
    vp_before = player_before["victoryPointsBreakdown"]["total"]
    vp_after = player_after["victoryPointsBreakdown"]["total"]
    vp_delta = vp_after - vp_before
    
    # VPs are THE winning condition - weight heavily
    reward += vp_delta * 4.0
    
    # ========================================
    # COMPONENT 2: Terraforming Rating
    # ========================================
    tr_before = player_before["terraformRating"]
    tr_after = player_after["terraformRating"]
    tr_delta = tr_after - tr_before
    
    # TR gives income and VPs at end - important
    reward += tr_delta * 2.5
    
    # ========================================
    # COMPONENT 3: Production Engine
    # ========================================
    prod_keys = [
        "megaCreditProduction", "steelProduction", "titaniumProduction",
        "plantProduction", "energyProduction", "heatProduction"
    ]
    
    prod_delta = sum(player_after[k] - player_before[k] for k in prod_keys)
    
    # Production is engine - weight highly early game
    if generation <= 8:
        reward += prod_delta * 2.0
    else:
        reward += prod_delta * 1.0
    
    # ========================================
    # COMPONENT 4: Card Engine (Tableau)
    # ========================================
    tableau_before = len(player_before["tableau"])
    tableau_after = len(player_after["tableau"])
    tableau_delta = tableau_after - tableau_before
    
    # Playing cards = building strategy
    reward += tableau_delta * 1.5
    
    # ========================================
    # COMPONENT 5: Global Parameters
    # ========================================
    if game_before and game_after:
        oxygen_delta = game_after.get("oxygenLevel", 0) - game_before.get("oxygenLevel", 0)
        temp_delta = game_after.get("temperature", 0) - game_before.get("temperature", 0)
        ocean_delta = game_after.get("oceans", 0) - game_before.get("oceans", 0)
        
        # Bonus for contributing to global parameters (gives TR + VPs)
        reward += oxygen_delta * 1.0
        reward += temp_delta * 0.1  # Temperature steps are -2 each
        reward += ocean_delta * 1.0
    
    # ========================================
    # COMPONENT 6: Late Game Spending
    # ========================================
    mc_after = player_after["megaCredits"]
    
    # Discourage hoarding in late game
    if generation >= 10:
        if mc_after > 25:
            reward -= (mc_after - 25) * 0.1
    
    # ========================================
    # COMPONENT 7: Stagnation Penalty
    # ========================================
    # Penalize doing nothing productive
    is_stagnant = (
        vp_delta == 0 and
        tr_delta == 0 and
        prod_delta == 0 and
        tableau_delta == 0
    )
    
    if is_stagnant:
        reward -= 0.5
    
    # ========================================
    # Final scaling and clipping
    # ========================================
    # Clip to reasonable range for stable learning
    reward = float(np.clip(reward, -10.0, 10.0))
    
    return reward