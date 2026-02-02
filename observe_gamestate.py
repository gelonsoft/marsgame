import numpy as np
from typing import Dict, Any, List, Union
from collections import defaultdict
import json
from jsonpath_ng.ext import parse
from myconfig import MAX_ACTIONS, MAX_GAME_FEATURES_SIZE
from factored_actions import ACTION_TYPE_DIM, OBJECT_ID_DIM, PAYMENT_ID_DIM, EXTRA_PARAM_DIM
from myutils import create_fixed_size_array

with open("gs.schema.formatted.json", 'rb') as f:
    json_schema = json.loads(f.read())

MAX_PLAYERS = 6
ALL_CARD_NAMES = parse('$.definitions..CardName.enum').find(json_schema)[0].value

# Helper functions
def normalize(value, min_val, max_val):
    """IMPROVED: Better normalization with safety checks"""
    if max_val <= min_val:
        return 0.0
    result = (value - min_val) / (max_val - min_val)
    return float(np.clip(result, 0.0, 1.0))

def normalize_signed(value, min_val, max_val):
    """NEW: Normalization that preserves sign"""
    if max_val <= min_val:
        return 0.0
    result = (value - min_val) / (max_val - min_val)
    return float(np.clip(result, -1.0, 1.0))

def one_hot_encode(value, categories):
    encoding = [0] * len(categories)
    if value in categories:
        encoding[categories.index(value)] = 1
    return encoding

def safe_divide(numerator, denominator, default=0.0):
    """NEW: Safe division with default value"""
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator

def observe(player_view_model: Dict[str, Any], factored_legal: Union[Dict[int, tuple],None] = None) -> np.ndarray:
    """
    IMPROVED: More efficient observation with better feature engineering
    """
    features = []
    
    game = player_view_model["game"]
    player = player_view_model["thisPlayer"]
    players = player_view_model.get("players", [])
    
    generation = game["generation"]
    max_generation = game.get("lastSoloGeneration", 14)
    game_progress = generation / max_generation
    
    # ============================
    # 1. Core Game State (Compact)
    # ============================
    features.extend([
        normalize(generation, 1, max_generation),
        game_progress,
        *one_hot_encode(game["phase"], ["action", "research", "production", "solar"]),
        normalize(game["oxygenLevel"], 0, 14),
        normalize_signed(game["temperature"], -30, 8),
        normalize(game["oceans"], 0, 9),
        normalize(game.get("venusScaleLevel", 0), 0, 30),
        float(game.get("isTerraformed", False)),
    ])
    
    # Remaining global parameters
    features.extend([
        normalize(14 - game["oxygenLevel"], 0, 14),
        normalize(8 - game["temperature"], 0, 38),
        normalize(9 - game["oceans"], 0, 9),
    ])
    
    # Game metadata
    features.extend([
        normalize(len(game.get("passedPlayers", [])), 0, 5),
        float(player["color"] in game.get("passedPlayers", [])),
        normalize(game.get("undoCount", 0), 0, 10),
    ])
    
    # ============================
    # 2. Expansions (Compact Binary)
    # ============================
    expansions = game["gameOptions"]["expansions"]
    expansion_list = [
        "ares", "ceo", "colonies", "community", "corpera", "moon",
        "pathfinders", "prelude", "prelude2", "promo", "starwars",
        "turmoil", "underworld", "venus"
    ]
    features.extend([float(expansions.get(exp, False)) for exp in expansion_list])
    
    # ============================
    # 3. Player State (Enhanced)
    # ============================
    # Resources
    resource_info = [
        ("megaCredits", 0, 200),
        ("steel", 0, 100),
        ("titanium", 0, 100),
        ("plants", 0, 50),
        ("energy", 0, 20),
        ("heat", 0, 100),
    ]
    
    for res, min_v, max_v in resource_info:
        features.append(normalize(player[res], min_v, max_v))
    
    # Production
    production_info = [
        ("megaCreditProduction", -5, 30),
        ("steelProduction", 0, 20),
        ("titaniumProduction", 0, 20),
        ("plantProduction", 0, 15),
        ("energyProduction", 0, 15),
        ("heatProduction", 0, 25),
    ]
    
    for prod, min_v, max_v in production_info:
        features.append(normalize(player[prod], min_v, max_v))
    
    # Production statistics
    prod_vals = [player[p[0]] for p in production_info]
    total_prod = sum(prod_vals)
    features.extend([
        normalize(total_prod, 0, 50),
        normalize(np.std(prod_vals), 0, 10) if prod_vals else 0.0,
        normalize(np.mean(prod_vals), 0, 10) if prod_vals else 0.0,
    ])
    
    # Resource values
    features.extend([
        normalize(player["steelValue"], 1, 5),
        normalize(player["titaniumValue"], 1, 5),
        float(player["heat"] >= 8),
        float(player["plants"] >= 8),
    ])
    
    # ============================
    # 4. Victory and Progress
    # ============================
    vp = player["victoryPointsBreakdown"]["total"]
    features.extend([
        normalize(vp, 0, 100),
        normalize(player["terraformRating"], 20, 60),
    ])
    
    # VP momentum
    vp_hist = player.get("victoryPointsByGeneration", [])
    if len(vp_hist) >= 2:
        vp_trend = normalize_signed(vp_hist[-1] - vp_hist[-2], -5, 10)
        features.append(vp_trend)
    else:
        features.append(0.0)
    
    # Relative VP position
    if len(players) > 1:
        all_vps = [p["victoryPointsBreakdown"]["total"] for p in players]
        vp_lead = vp - max(vp for vp in all_vps if vp != player["victoryPointsBreakdown"]["total"] or len([v for v in all_vps if v == vp]) > 1)
        features.append(normalize_signed(vp_lead, -30, 30))
        
        # Rank
        sorted_vps = sorted(all_vps, reverse=True)
        rank = sorted_vps.index(vp) + 1
        features.append(normalize(rank, 1, len(players)))
    else:
        features.extend([0.0, 0.0])
    
    # ============================
    # 5. Card Engine
    # ============================
    tableau = player["tableau"]
    hand = player_view_model.get("cardsInHand", [])
    
    features.extend([
        normalize(len(tableau), 0, 40),
        normalize(len(hand), 0, 15),
        normalize(player["cardsInHandNbr"], 0, 15),
        normalize(player.get("cardCost", 3), 0, 20),
        normalize(player.get("cardDiscount", 0), 0, 10),
    ])
    
    # Hand cost statistics
    if hand:
        hand_costs = [c.get("calculatedCost", 0) for c in hand]
        features.extend([
            normalize(np.mean(hand_costs), 0, 30),
            normalize(np.min(hand_costs), 0, 30),
            normalize(np.max(hand_costs), 0, 30),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # ============================
    # 6. Tags (Enhanced)
    # ============================
    tags = player["tags"]
    tag_categories = [
        "building", "space", "science", "power", "earth",
        "jovian", "plant", "microbe", "animal", "city",
        "moon", "mars", "venus", "wild", "event", "clone"
    ]
    
    total_tags = max(1, sum(tags.values()))
    for tag in tag_categories:
        features.append(normalize(tags.get(tag, 0), 0, 15))
    
    # Tag diversity
    tag_diversity = sum(v > 0 for v in tags.values())
    features.append(normalize(tag_diversity, 0, len(tag_categories)))
    
    # ============================
    # 7. Board Control
    # ============================
    features.extend([
        normalize(player["citiesCount"], 0, 10),
        normalize(player.get("coloniesCount", 0), 0, 8),
        normalize(player.get("fleetSize", 0), 0, 10),
        normalize(player.get("tradesThisGeneration", 0), 0, 5),
    ])
    
    # Fleet efficiency
    if player.get("fleetSize", 0) > 0:
        efficiency = player.get("tradesThisGeneration", 0) / player["fleetSize"]
        features.append(normalize(efficiency, 0, 1))
    else:
        features.append(0.0)
    
    # ============================
    # 8. Actions and Tempo
    # ============================
    features.extend([
        normalize(player.get("actionsTakenThisRound", 0), 0, 15),
        normalize(player.get("actionsTakenThisGame", 0), 0, 300),
        normalize(player.get("availableBlueCardActionCount", 0), 0, 10),
        float(player.get("isActive", True)),
    ])
    
    # ============================
    # 9. Special Mechanics
    # ============================
    # Colonies
    if "colonies" in game:
        active_colonies = sum(1 for c in game["colonies"] if c.get("isActive", False))
        features.append(normalize(active_colonies, 0, 10))
        
        owned_colonies = sum(1 for c in game["colonies"] if player["color"] in c.get("colonies", []))
        features.append(normalize(owned_colonies, 0, 5))
    else:
        features.extend([0.0, 0.0])
    
    # Turmoil
    if "turmoil" in game:
        turmoil = game["turmoil"]
        features.extend([
            normalize(player.get("influence", 0), 0, 20),
            float(turmoil.get("chairman") == player["color"]),
            normalize(len(turmoil.get("lobby", [])), 0, 10),
        ])
        
        ruling_party = turmoil.get("ruling")
        party_bonus = {
            "marsFirst": 1, "scientists": 2, "unity": 3,
            "greens": 4, "reds": 5, "kelvinists": 6
        }
        features.append(normalize(party_bonus.get(ruling_party, 0), 0, 6))
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Moon
    if "moon" in game:
        moon = game["moon"]
        features.extend([
            normalize(moon.get("miningRate", 0), 0, 8),
            normalize(moon.get("habitatRate", 0), 0, 8),
            normalize(moon.get("logisticsRate", 0), 0, 8),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # ============================
    # 10. Negative Factors
    # ============================
    features.extend([
        normalize(player.get("corruption", 0), 0, 10),
        float(player.get("corruption", 0) > 0),
        normalize(player.get("excavations", 0), 0, 10),
    ])
    
    # ============================
    # 11. Milestones and Awards
    # ============================
    player_color = player["color"]
    
    owned_milestones = sum(
        1 for m in game.get("milestones", [])
        if any(s.get("playerColor") == player_color for s in m.get("scores", []))
    )
    
    owned_awards = sum(
        1 for a in game.get("awards", [])
        if any(s.get("playerColor") == player_color for s in a.get("scores", []))
    )
    
    features.extend([
        normalize(len(game.get("milestones", [])), 0, 5),
        normalize(len(game.get("awards", [])), 0, 5),
        normalize(owned_milestones, 0, 3),
        normalize(owned_awards, 0, 3),
    ])
    
    # ============================
    # 12. Opponent Summary (Compact)
    # ============================
    if len(players) > 1:
        opponents = [p for p in players if p["color"] != player["color"]]
        
        if opponents:
            # Average opponent metrics
            avg_opp_mc = np.mean([p["megaCredits"] for p in opponents])
            avg_opp_tr = np.mean([p["terraformRating"] for p in opponents])
            avg_opp_prod = np.mean([sum([
                p["megaCreditProduction"], p["steelProduction"],
                p["titaniumProduction"], p["plantProduction"],
                p["energyProduction"], p["heatProduction"]
            ]) for p in opponents])
            
            features.extend([
                normalize_signed(player["megaCredits"] - avg_opp_mc, -100, 100),
                normalize_signed(player["terraformRating"] - avg_opp_tr, -20, 20),
                normalize_signed(total_prod - avg_opp_prod, -20, 20),
            ])
            
            # Strongest opponent
            strongest = max(opponents, key=lambda p: p["victoryPointsBreakdown"]["total"])
            features.extend([
                normalize_signed(vp - strongest["victoryPointsBreakdown"]["total"], -30, 30),
                normalize_signed(player["terraformRating"] - strongest["terraformRating"], -20, 20),
            ])
        else:
            features.extend([0.0] * 5)
    else:
        features.extend([0.0] * 5)
    
    # ============================
    # 13. Board State
    # ============================
    spaces = game.get("spaces", [])
    if spaces:
        total_tiles = max(1, sum(1 for s in spaces if "tileType" in s))
        
        features.extend([
            normalize(sum(1 for s in spaces if s.get("tileType") == "city"), 0, total_tiles),
            normalize(sum(1 for s in spaces if s.get("tileType") == "greenery"), 0, total_tiles),
            normalize(sum(1 for s in spaces if s.get("tileType") == "ocean"), 0, total_tiles),
            normalize(sum(1 for s in spaces if s.get("color") == player["color"]), 0, total_tiles),
        ])
        
        # Available spots
        empty_land = sum(1 for s in spaces if s.get("spaceType") == "land" and "tileType" not in s)
        features.append(normalize(empty_land, 0, 50))
    else:
        features.extend([0.0] * 5)
    
    # ============================
    # 14. Game Options
    # ============================
    game_opts = game.get("gameOptions", {})
    features.extend([
        float(game_opts.get("draftVariant", False)),
        float(game_opts.get("initialDraftVariant", False)),
        float(game_opts.get("preludeDraftVariant", False)),
    ])
    
    # ============================
    # Create fixed-size observation
    # ============================
    # --- Action-context features (Change 3: MCTS prediction helpers) ------
    # These lightweight summary statistics of the legal action set let the
    # dynamics network predict which actions will be available after a step,
    # dramatically improving lookahead quality without bloating obs size.
    #
    # Layout (appended before create_fixed_size_array so they share the 512 budget):
    #   [0]      num_legal / MAX_ACTIONS            — action-set density
    #   [1..16]  action_type histogram (16 bins)    — what *kinds* of decisions are available
    #   [17]     mean object_id / OBJECT_ID_DIM     — avg complexity of targets
    #   [18]     mean payment_id / PAYMENT_ID_DIM   — avg payment complexity
    #   [19..23] waiting_for type one-hot (5 bins)  — which sub-prompt generated this set
    #            (or, card, and, payment, deferred)
    # -----------------------------------------------------------------------
    action_ctx = []
    if factored_legal and len(factored_legal) > 0:
        num_legal = len(factored_legal)
        action_ctx.append(normalize(num_legal, 0, MAX_ACTIONS))

        # Type histogram
        type_hist = np.zeros(ACTION_TYPE_DIM, dtype=np.float32)
        obj_sum = 0.0
        pay_sum = 0.0
        for slot, (t, o, p, e) in factored_legal.items():
            type_hist[t] += 1.0
            obj_sum += o
            pay_sum += p
        type_hist = type_hist / num_legal  # normalise to probability
        action_ctx.extend(type_hist.tolist())

        action_ctx.append(obj_sum / num_legal / max(OBJECT_ID_DIM, 1))
        action_ctx.append(pay_sum / num_legal / max(PAYMENT_ID_DIM, 1))
    else:
        action_ctx.append(0.0)
        action_ctx.extend([0.0] * ACTION_TYPE_DIM)
        action_ctx.extend([0.0, 0.0])

    # waiting_for type one-hot (5 bins: or, card, and, payment, deferred)
    waiting_for = player_view_model.get("waitingFor", {})
    wf_type = waiting_for.get("type", "") if isinstance(waiting_for, dict) else ""
    wf_categories = ["or", "card", "and", "payment", "deffered"]
    action_ctx.extend(one_hot_encode(wf_type, wf_categories))

    features.extend(action_ctx)

    observation = create_fixed_size_array(data_list=features, fixed_size=512)

    # Final processing
    observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
    observation = np.clip(observation, -10, 10)
    
    return observation