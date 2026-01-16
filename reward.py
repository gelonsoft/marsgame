import numpy as np

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

    reward = 0.0

    # ============================
    # 1. Victory Progress (Primary)
    # ============================
    vp_before = player_before["victoryPointsBreakdown"]["total"]
    vp_after = player_after["victoryPointsBreakdown"]["total"]
    vp_delta = vp_after - vp_before

    reward += vp_delta * 2.5

    # Endgame VP urgency
    if game_after:
        generation = game_after.get("generation", 1)
        reward += vp_delta * (generation / 10.0)

    # ============================
    # 2. Terraforming Rating
    # ============================
    tr_before = player_before["terraformRating"]
    tr_after = player_after["terraformRating"]
    tr_delta = tr_after - tr_before

    reward += tr_delta * 1.8

    # ============================
    # 3. Global Parameter Progress
    # ============================
    if game_before and game_after:
        reward += (game_after["oxygenLevel"] - game_before["oxygenLevel"]) * 0.6
        reward += (game_after["oceans"] - game_before["oceans"]) * 0.6
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

    reward += prod_gain * 0.8

    # Engine diversity bonus
    diversified = sum(player_after[k] > 0 for k in prod_keys)
    reward += diversified * 0.15

    # ============================
    # 5. Economy Efficiency
    # ============================
    mc_before = player_before["megaCredits"]
    mc_after = player_after["megaCredits"]

    # Reward productive spending
    if mc_after < mc_before:
        reward += (mc_before - mc_after) * 0.03

    # Penalize hoarding
    if mc_after > 25:
        reward -= (mc_after - 25) * 0.02

    # ============================
    # 6. Resource Conversion
    # ============================
    # Heat conversion
    if player_before["heat"] >= 8 and player_after["heat"] < player_before["heat"]:
        reward += 0.7
    elif player_before["heat"] >= 8 and player_after["heat"] >= player_before["heat"]:
        reward -= 0.2

    # Plant conversion
    if player_before["plants"] >= 8 and player_after["plants"] < player_before["plants"]:
        reward += 0.7
    elif player_before["plants"] >= 8 and player_after["plants"] >= player_before["plants"]:
        reward -= 0.2

    # ============================
    # 7. Board Control
    # ============================
    reward += (player_after["citiesCount"] - player_before["citiesCount"]) * 1.4
    reward += (player_after["coloniesCount"] - player_before["coloniesCount"]) * 1.2

    # ============================
    # 8. Card Engine Development
    # ============================
    tableau_before = len(player_before["tableau"])
    tableau_after = len(player_after["tableau"])

    reward += (tableau_after - tableau_before) * 1.0

    # ============================
    # 9. Action Tempo
    # ============================
    reward += (player_after["actionsTakenThisRound"] - player_before["actionsTakenThisRound"]) * 0.08

    # ============================
    # 10. Political Power
    # ============================
    reward += (player_after["influence"] - player_before["influence"]) * 0.5

    # ============================
    # 11. Corruption Penalty
    # ============================
    corruption_delta = player_after["corruption"] - player_before["corruption"]
    reward -= corruption_delta * 2.0

    # ============================
    # 12. Trade Efficiency
    # ============================
    reward += (player_after["tradesThisGeneration"] - player_before["tradesThisGeneration"]) * 0.6

    # ============================
    # 13. Blue Action Utilization
    # ============================
    reward += (player_after["availableBlueCardActionCount"] - player_before["availableBlueCardActionCount"]) * 0.3

    # ============================
    # 14. Hand Management
    # ============================
    hand_after = player_after["cardsInHandNbr"]

    if hand_after > 10:
        reward -= (hand_after - 10) * 0.08

    # ============================
    # 15. Discount Utilization
    # ============================
    reward += (player_after["cardDiscount"] - player_before["cardDiscount"]) * 0.6

    # ============================
    # 16. Steel / Titanium Efficiency
    # ============================
    reward += (player_after["steelValue"] - player_before["steelValue"]) * 0.4
    reward += (player_after["titaniumValue"] - player_before["titaniumValue"]) * 0.4

    # ============================
    # 17. Jovian Strategy
    # ============================
    jovian_before = player_before["tags"].get("jovian", 0)
    jovian_after = player_after["tags"].get("jovian", 0)
    reward += (jovian_after - jovian_before) * 0.8

    # ============================
    # 18. Engine Flexibility
    # ============================
    tags_before = player_before["tags"]
    tags_after = player_after["tags"]

    diversity_before = sum(v > 0 for v in tags_before.values())
    diversity_after = sum(v > 0 for v in tags_after.values())
    reward += (diversity_after - diversity_before) * 0.4

    # ============================
    # 19. Passing Discipline
    # ============================
    if player_after.get("isActive") is False and player_before.get("isActive") is True:
        reward += 0.5

    # ============================
    # 20. Stagnation Penalty
    # ============================
    if (
        vp_delta == 0 and
        tr_delta == 0 and
        prod_gain == 0 and
        tableau_after == tableau_before
    ):
        reward -= 0.8

    # ============================
    # 21. VP Momentum
    # ============================
    vp_hist = player_after.get("victoryPointsByGeneration", [])
    if len(vp_hist) >= 2:
        reward += (vp_hist[-1] - vp_hist[-2]) * 0.4

    # ============================
    # 22. Endgame Urgency
    # ============================
    if game_after:
        if game_after.get("generation", 1) >= 10:
            reward += vp_delta * 1.2

    # ============================
    # 23. City + Greenery Synergy
    # ============================
    reward += (player_after["citiesCount"] - player_before["citiesCount"]) * \
              (player_after["terraformRating"] - player_before["terraformRating"]) * 0.3

    # ============================
    # 24. Fleet Utilization
    # ============================
    if player_after["fleetSize"] > 0:
        util = player_after["tradesThisGeneration"] / player_after["fleetSize"]
        reward += util * 0.3

    # ============================
    # 25. Over-saving Penalty
    # ============================
    if mc_after > 40 and prod_gain == 0:
        reward -= 0.6

    # ============================
    # 26. Conversion Miss Penalty
    # ============================
    if player_after["heat"] >= 8:
        reward -= 0.1
    if player_after["plants"] >= 8:
        reward -= 0.1

    # ============================
    # 27. Production Imbalance Penalty
    # ============================
    prod_vals = [player_after[k] for k in prod_keys]
    reward -= np.std(prod_vals) * 0.05

    # ============================
    # 28. Card Spam Penalty
    # ============================
    if tableau_after - tableau_before > 3:
        reward -= 0.3

    # ============================
    # 29. Corruption Risk Scaling
    # ============================
    if player_after["corruption"] > 3:
        reward -= player_after["corruption"] * 0.4

    # ============================
    # 30. Reward Normalization
    # ============================
    reward = np.clip(reward, -12.0, 12.0)

    return float(reward)