import numpy as np
import json
from deepdiff import DeepDiff

def reward_function(player_state_before: dict, player_state_after: dict) -> float:
    """
    Dense shaped reward for Terraforming Mars RL agent.
    Uses player state before and after action.
    Optionally uses game state for endgame scaling (recommended).
    """
    player_before=player_state_before['thisPlayer']
    game_before=player_state_before['game']
    player_after=player_state_after['thisPlayer']
    game_after=player_state_after['game']
    reward_desc={}
    reward = 0.0
    for d in [player_before,player_after]:
        d['timer']={}
        d['needsToDraft']=False
    if hash(json.dumps(player_before,sort_keys=True))==hash(json.dumps(player_after,sort_keys=True)):
        print("Rewards=no_changes=0.1")
        return 0.0

    # ============================
    # 1. Victory Progress (Primary)
    # ============================
    vp_before = player_before["victoryPointsBreakdown"]["total"]
    vp_after = player_after["victoryPointsBreakdown"]["total"]
    vp_delta = vp_after - vp_before

    reward_desc['vp_delta']=vp_delta * 2.5
    reward += vp_delta * 2.5

    # Endgame VP urgency
    if game_after:
        generation = game_after.get("generation", 1)
        reward_desc['end_game_vp_delta']=vp_delta * (generation / 10.0)
        reward += vp_delta * (generation / 10.0)

    # ============================
    # 2. Terraforming Rating
    # ============================
    tr_before = player_before["terraformRating"]
    tr_after = player_after["terraformRating"]
    tr_delta = tr_after - tr_before
    reward_desc['tr_delta']= tr_delta * 1.8
    reward += tr_delta * 1.8

    # ============================
    # 3. Global Parameter Progress
    # ============================
    if game_before and game_after:
        reward_desc['oxygenLevel']=(game_after["oxygenLevel"] - game_before["oxygenLevel"]) * 0.6
        reward += (game_after["oxygenLevel"] - game_before["oxygenLevel"]) * 0.6
        reward_desc['oceans']=(game_after["oceans"] - game_before["oceans"]) * 0.6
        reward += (game_after["oceans"] - game_before["oceans"]) * 0.6
        reward_desc['temperature']= (game_after["temperature"] - game_before["temperature"]) * 0.05
        reward += (game_after["temperature"] - game_before["temperature"]) * 0.05

    # ============================
    # 4. Production Engine Growth
    # ============================
    prod_keys = [
        "megaCreditProduction", "steelProduction", "titaniumProduction",
        "plantProduction", "energyProduction", "heatProduction"
    ]

    prod_gain = 0
    for key in prod_keys:
        prod_gain += max(0, player_after[key] - player_before[key])

    reward_desc['prod_gain']=prod_gain * 0.8
    reward += prod_gain * 0.8

    # Engine diversity bonus
    diversified = sum(player_after[k] > 0 for k in prod_keys)
    reward_desc['diversified']=diversified * 0.15
    reward += diversified * 0.15

    # ============================
    # 5. Economy Efficiency
    # ============================
    mc_before = player_before["megaCredits"]
    mc_after = player_after["megaCredits"]

    # Reward productive spending
    if mc_after < mc_before:
        reward_desc['mc_delta']=(mc_before - mc_after) * 0.03
        reward += (mc_before - mc_after) * 0.03

    # Penalize hoarding
    if mc_after > 25:
        reward_desc['mc_after']=-(mc_after - 25) * 0.005
        reward -= (mc_after - 25) * 0.005

    # ============================
    # 6. Resource Conversion
    # ============================
    # Heat conversion
    if player_before["heat"] >= 8 and player_after["heat"] < player_before["heat"]:
        reward += 0.7
        reward_desc['heat_conversion']=0.7
    elif player_before["heat"] >= 8 and player_after["heat"] >= player_before["heat"]:
        reward_desc['heat_conversion']=-0.2
        reward -= 0.2

    # Plant conversion
    if player_before["plants"] >= 8 and player_after["plants"] < player_before["plants"]:
        reward_desc['plants_conversion']=0.7
        reward += 0.7
    elif player_before["plants"] >= 8 and player_after["plants"] >= player_before["plants"]:
        reward_desc['plants_conversion']=-0.2
        reward -= 0.2

    # ============================
    # 7. Board Control
    # ============================
    reward += (player_after["citiesCount"] - player_before["citiesCount"]) * 1.4
    reward_desc['citiesCount']=(player_after["citiesCount"] - player_before["citiesCount"]) * 1.4
    reward += (player_after["coloniesCount"] - player_before["coloniesCount"]) * 1.2
    reward_desc['coloniesCount']=(player_after["coloniesCount"] - player_before["coloniesCount"]) * 1.2

    # ============================
    # 8. Card Engine Development
    # ============================
    tableau_before = len(player_before["tableau"])
    tableau_after = len(player_after["tableau"])
    reward_desc['tableau_delta']=(tableau_after - tableau_before) * 1.0
    reward += (tableau_after - tableau_before) * 1.0

    # ============================
    # 9. Action Tempo
    # ============================
    reward += (player_after["actionsTakenThisRound"] - player_before["actionsTakenThisRound"]) * 0.08
    reward_desc['actionsTakenThisRound']=(player_after["actionsTakenThisRound"] - player_before["actionsTakenThisRound"]) * 0.08
    # ============================
    # 10. Political Power
    # ============================
    reward += (player_after["influence"] - player_before["influence"]) * 0.5
    reward_desc['influence']=(player_after["influence"] - player_before["influence"]) * 0.5
    # ============================
    # 11. Corruption Penalty
    # ============================
    corruption_delta = player_after["corruption"] - player_before["corruption"]
    reward -= corruption_delta * 2.0
    reward_desc['corruption_penalty']=-corruption_delta * 2.0

    # ============================
    # 12. Trade Efficiency
    # ============================
    reward += (player_after["tradesThisGeneration"] - player_before["tradesThisGeneration"]) * 0.6
    reward_desc['trades_delta']=(player_after["tradesThisGeneration"] - player_before["tradesThisGeneration"]) * 0.6
    # ============================
    # 13. Blue Action Utilization
    # ============================
    reward += (player_after["availableBlueCardActionCount"] - player_before["availableBlueCardActionCount"]) * 0.3
    reward_desc['availableBlueCardActionCount']=(player_after["availableBlueCardActionCount"] - player_before["availableBlueCardActionCount"]) * 0.3
    # ============================
    # 14. Hand Management
    # ============================
    hand_after = player_after["cardsInHandNbr"]

    if hand_after > 10:
        reward -= (hand_after - 10) * 0.08
        reward_desc['hand_after_delta']=-(hand_after - 10) * 0.08

    # ============================
    # 15. Discount Utilization
    # ============================
    reward += (player_after["cardDiscount"] - player_before["cardDiscount"]) * 0.6
    reward_desc['cardDiscount']=(player_after["cardDiscount"] - player_before["cardDiscount"]) * 0.6
    # ============================
    # 16. Steel / Titanium Efficiency
    # ============================
    reward += (player_after["steelValue"] - player_before["steelValue"]) * 0.4
    reward_desc['steelValue']=(player_after["steelValue"] - player_before["steelValue"]) * 0.4
    reward += (player_after["titaniumValue"] - player_before["titaniumValue"]) * 0.4
    reward_desc['titaniumValue']=(player_after["titaniumValue"] - player_before["titaniumValue"]) * 0.4
    # ============================
    # 17. Jovian Strategy
    # ============================
    jovian_before = player_before["tags"].get("jovian", 0)
    jovian_after = player_after["tags"].get("jovian", 0)
    reward += (jovian_after - jovian_before) * 0.8
    reward_desc['jovian']=(jovian_after - jovian_before) * 0.8

    # ============================
    # 18. Engine Flexibility
    # ============================
    tags_before = player_before["tags"]
    tags_after = player_after["tags"]

    diversity_before = sum(v > 0 for v in tags_before.values())
    diversity_after = sum(v > 0 for v in tags_after.values())
    reward += (diversity_after - diversity_before) * 0.4
    reward_desc['diversity_tags']=(diversity_after - diversity_before) * 0.4

    # ============================
    # 19. Passing Discipline
    # ============================
    if player_after.get("isActive") is False and player_before.get("isActive") is True:
        reward += 0.5
        reward_desc['passing_discipline']=0.5

    # ============================
    # 20. Stagnation Penalty
    # ============================
    cards_in_hand_delta=player_after.get('cardsInHandNbr',0)-player_before.get('cardsInHandNbr',0)
    if (
        vp_delta == 0 and
        tr_delta == 0 and
        prod_gain == 0 and
        tableau_after == tableau_before and
        cards_in_hand_delta==0
    ):
        reward_desc['stagnation_penalty']=-0.1
        reward -= 0.1

    # ============================
    # 21. VP Momentum
    # ============================
    vp_hist = player_after.get("victoryPointsByGeneration", [])
    if len(vp_hist) >= 2:
        reward += (vp_hist[-1] - vp_hist[-2]) * 0.4
        reward_desc['vp_momentum']=-0.8

    # ============================
    # 22. Endgame Urgency
    # ============================
    if game_after:
        if game_after.get("generation", 1) >= 10:
            reward_desc['end_game_vp_delta']=vp_delta * 1.2
            reward += vp_delta * 1.2

    # ============================
    # 23. City + Greenery Synergy
    # ============================
    reward += (player_after["citiesCount"] - player_before["citiesCount"]) * \
              (player_after["terraformRating"] - player_before["terraformRating"]) * 0.3

    reward_desc['city_and_greenery']=(player_after["citiesCount"] - player_before["citiesCount"]) * \
              (player_after["terraformRating"] - player_before["terraformRating"]) * 0.3
    # ============================
    # 24. Fleet Utilization
    # ============================
    if player_after["fleetSize"] > 0:
        util = player_after["tradesThisGeneration"] / player_after["fleetSize"]
        reward_desc['fleet_utilization']=util * 0.3
        reward += util * 0.3

    # ============================
    # 25. Over-saving Penalty
    # ============================
    if mc_after > 40 and prod_gain == 0:
        reward_desc['oversaving_penalty']=-0.03
        reward -= 0.03

    # ============================
    # 26. Conversion Miss Penalty
    # ============================
    if player_after["heat"] >= 8:
        reward_desc['heat_conversion_miss']=-0.1
        reward -= 0.1
    if player_after["plants"] >= 8:
        reward_desc['plant_conversion_miss']=-0.1
        reward -= 0.1

    # ============================
    # 27. Production Imbalance Penalty
    # ============================
    prod_vals = [player_after[k] for k in prod_keys]
    reward -= float(np.std(prod_vals) * 0.01)
    reward_desc['production_imbalance']=-float(np.std(prod_vals) * 0.01)
    # ============================
    # 28. Card Spam Penalty
    # ============================
    if tableau_after - tableau_before > 3:
        reward -= 0.3
        reward_desc['card_spam_penalty']=-0.3
        

    # ============================
    # 29. Corruption Risk Scaling
    # ============================
    if player_after["corruption"] > 3:
        reward -= player_after["corruption"] * 0.4
        reward_desc['corruption_risk_scaling']=-player_after["corruption"] * 0.4

    # ============================
    # 30. Reward Normalization
    # ============================
    reward_desc['total']=reward
    reward = float(np.clip(reward, -12.0, 12.0))
    reward_desc['total_clipped']=float(reward)
    # diff_player_states = DeepDiff(player_before, player_after, ignore_order=True)
    # diff_game_states = DeepDiff(game_before, game_after, ignore_order=True)
    # reward_desc={k: v for k, v in sorted(reward_desc.items(), key=lambda item: item[1]) if abs(v)>0.00000001}
    # print(f"Rewards={reward}={reward_desc}")
    # print(f"Diff player_states={diff_player_states}")
    # print(f"Diff game_states={diff_game_states}")
    # if reward<-0.0000001:
    #     pass
    return float(reward)