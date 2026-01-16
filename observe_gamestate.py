import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
import json
from jsonpath_ng.ext import parse
from myconfig import MAX_ACTIONS,ONE_ACTION_ARRAY_SIZE,MAX_GAME_FEATURES_SIZE, TOTAL_ACTIONS
from myutils import create_fixed_size_array

with open("gs.schema.formatted.json",'rb') as f:
#with open("gs.response.schema.lite.json",'rb') as f:
    json_schema=json.loads(f.read())

MAX_PLAYERS=6
ALL_CARD_NAMES=parse('$.definitions..CardName.enum').find(json_schema)[0].value


standard_projects = [
    "Power Plant:SP", "Asteroid:SP", "Air Scrapping",
    "Colony", "Aquifer", "Greenery", "City"
]

# Helper functions
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0

def one_hot_encode(value, categories):
    encoding = [0] * len(categories)
    if value in categories:
        encoding[categories.index(value)] = 1
    return encoding
    
def encode_action(action: Dict[str, Any]) -> np.ndarray:
    """Encodes any action into a fixed 1024-dimension feature vector with categorical card encoding"""
    action_vec = np.zeros(ONE_ACTION_ARRAY_SIZE, dtype=np.float32)
    
    if not action:
        return action_vec
    
    action_type = action.get("xtype", action.get("type",""))
    
    # 1. Categorical encoding for action type (first dimension)
    action_types = [
        "and", "or", "initialCards", "amount", "card",
        "colony", "delegate", "option", "party", "payment",
        "player", "productionToLose", "projectCard", "space",
        "aresGlobalParameters", "globalEvent", "policy",
        "resource", "resources","xcard_choose","xconfirm_card_choose","xpayment"
    ]
    if action_type in action_types:
        action_vec[0] = action_types.index(action_type) / len(action_types)  # Normalized categorical
    
    # Action-specific features
    if action_type == "and":
        # Encode nested actions (positions 1-512)
        for i, resp in enumerate(action.get("responses", [])[:8]):  # Max 8 nested
            action_vec[1+i*64:1+(i+1)*64] = encode_action(resp)[:64]
    
    elif action_type == "or":
        action_vec[1] = action.get("index", 0) / 10.0  # Normalized index
        # Encode selected response (positions 2-65)
        if "response" in action:
            action_vec[2:66] = encode_action(action["response"])[:64]
    
    elif action_type == "initialCards":
        # Encode all initial card choices (positions 1-512)
        for i, resp in enumerate(action.get("responses", [])[:8]):  # Max 8 choices
            action_vec[1+i*64:1+(i+1)*64] = encode_action(resp)[:64]
    
    elif action_type == "amount":
        action_vec[1] = action.get("amount", 0) / 50.0
        action_vec[2] = action.get("min", 0) / 50.0
        action_vec[3] = action.get("max", 0) / 50.0
    
    elif action_type == "card":
        # 2. Categorical encoding for cards (positions 1-256)
        cards = action.get("cards", [])
        if isinstance(cards,dict):
            action_vec[1:1+64]=encode_action(cards)[:64]
        else:
            for i, card in enumerate(cards[:256]):  # Max 256 cards
                if card["name"] in ALL_CARD_NAMES:
                    action_vec[1+i] = (ALL_CARD_NAMES.index(card["name"]) + 1) / len(ALL_CARD_NAMES)  # Normalized index
            
            # Additional card features (positions 257-320)
            for i, card in enumerate(cards[:64]):
                action_vec[257+i] = card.get("calculatedCost", 0) / 50.0
                action_vec[321+i] = float("science" in card.get("tags", {}))
                action_vec[385+i] = float("building" in card.get("tags", {}))
    elif action_type == "xcard_choose":
        card = action.get("xoption", {})
        action_vec[1] = (ALL_CARD_NAMES.index(card["name"]) + 1) / len(ALL_CARD_NAMES)  # Normalized index

    elif action_type == "xconfirm_card_choose":
        action_vec[1]=1.0

    elif action_type == "colony":
        action_vec[1] = float("colonyName" in action)
    
    elif action_type == "delegate":
        action_vec[1] = float(action.get("player") != "NEUTRAL")
    
    elif action_type == "party":
        action_vec[1] = float("partyName" in action)
    
    elif action_type == "payment":
        payment = action.get("payment", {})
        action_vec[1] = payment.get("megaCredits", 0) / 100.0
        action_vec[2] = payment.get("steel", 0) / 20.0
        action_vec[3] = payment.get("titanium", 0) / 20.0
        action_vec[4] = payment.get("heat", 0) / 20.0
        action_vec[5] = payment.get("plants", 0) / 20.0
        action_vec[6] = payment.get("microbes", 0) / 20.0
        action_vec[7] = payment.get("floaters", 0) / 20.0
        action_vec[8] = payment.get("lunaArchivesScience", 0) / 20.0
        action_vec[9] = payment.get("spireScience", 0) / 20.0
        action_vec[10] = payment.get("seeds", 0) / 20.0
        action_vec[11] = payment.get("auroraiData", 0) / 20.0
        action_vec[12] = payment.get("graphene", 0) / 20.0
        action_vec[13] = payment.get("kuiperAsteroids", 0) / 20.0
        action_vec[14] = payment.get("corruption", 0) / 20.0
    
    elif action_type == "xpayment":
        payment = action.get("xoption", {})
        action_vec[1] = payment.get("megaCredits", 0) / 100.0
        action_vec[2] = payment.get("steel", 0) / 20.0
        action_vec[3] = payment.get("titanium", 0) / 20.0
        action_vec[4] = payment.get("heat", 0) / 20.0
        action_vec[5] = payment.get("plants", 0) / 20.0
        action_vec[6] = payment.get("microbes", 0) / 20.0
        action_vec[7] = payment.get("floaters", 0) / 20.0
        action_vec[8] = payment.get("lunaArchivesScience", 0) / 20.0
        action_vec[9] = payment.get("spireScience", 0) / 20.0
        action_vec[10] = payment.get("seeds", 0) / 20.0
        action_vec[11] = payment.get("auroraiData", 0) / 20.0
        action_vec[12] = payment.get("graphene", 0) / 20.0
        action_vec[13] = payment.get("kuiperAsteroids", 0) / 20.0
        action_vec[14] = payment.get("corruption", 0) / 20.0

    elif action_type == "player":
        action_vec[1] = float(action.get("player") != "NEUTRAL")
    
    elif action_type == "productionToLose":
        units = action.get("units", {})
        action_vec[1] = units.get("energy", 0) / 5.0
        action_vec[2] = units.get("heat", 0) / 5.0
        action_vec[3] = units.get("steel", 0) / 5.0
    
    elif action_type == "projectCard":
        action_vec[1] = float("card" in action)
        # Categorical encoding for project cards (positions 2-257)
        cards = action.get("cards", [])
        for i, card in enumerate(cards[:256]):
            if card["name"] in ALL_CARD_NAMES:
                action_vec[2+i] = (ALL_CARD_NAMES.index(card["name"]) + 1) / len(ALL_CARD_NAMES)
        
        # Payment details (positions 258-262)
        if "payment" in action:
            payment = action["payment"]
            action_vec[258] = payment.get("megaCredits", 0) / 100.0
            action_vec[259] = payment.get("steel", 0) / 20.0
            action_vec[260] = payment.get("titanium", 0) / 20.0
    
    elif action_type == "space":
        # Categorical encoding for spaces (positions 1-512)
        spaces = action.get("spaces", [])
        for i, space in enumerate(spaces[:512]):
            #action_vec[1+i] = hash(space.get("id", "")) % 1000 / 1000  # Simple categorical encoding
            sid = space.get("id", "0")
            action_vec[1+i] = int(sid) / 100.0 if sid.isdigit() else 0.0
        
        # Space features (positions 513-576)
        for i, space in enumerate(spaces[:64]):
            action_vec[513+i] = float("city" in str(space.get("tileType", "")))
            action_vec[577+i] = len(space.get("bonus", [])) / 5.0
    
    elif action_type == "aresGlobalParameters":
        params = action.get("response", {})
        action_vec[1] = params.get("lowOceanDelta", 0) / 2.0
        action_vec[2] = params.get("highOceanDelta", 0) / 2.0
        action_vec[3] = params.get("temperatureDelta", 0) / 2.0
    
    elif action_type == "globalEvent":
        action_vec[1] = float("globalEventName" in action)
    
    elif action_type == "policy":
        action_vec[1] = float("policyId" in action)
    
    elif action_type == "resource":
        action_vec[1] = float("resource" in action)
    
    elif action_type == "resources":
        units = action.get("units", {})
        action_vec[1] = units.get("megacredits", 0) / 20.0
        action_vec[2] = units.get("steel", 0) / 10.0
        action_vec[3] = units.get("titanium", 0) / 10.0
    
    return action_vec
    
def observe(player_view_model: Dict[str, Any],action_slot_map: Dict[str,Any]) -> np.ndarray:
    """
    Comprehensive observation function that includes:
    - All game state features
    - Available player actions (InputResponse)
    
    Args:
        player_view_model: The complete PlayerViewModel dictionary
        available_actions: The current InputResponse object representing available actions
        
    Returns:
        A numpy array containing all observable features including available actions
    """
    # Initialize feature containers


    features = []
    action_features = []
    
    # Initialize feature containers
    features = []
       
    # 1. Game State Features
    game = player_view_model["game"]
    player = player_view_model["thisPlayer"]
    players = player_view_model["players"]
    
    # Basic game state
    features.extend([
        game["generation"] / 14.0,  # Current generation
        *one_hot_encode(game["phase"], ["action", "research", "production", "solar"]),
        game["oxygenLevel"] / 14.0,  # Normalized oxygen
        (game["temperature"] + 30) / 82.0,  # Normalized temperature
        game["oceans"] / 9.0,  # Ocean tiles
        game["venusScaleLevel"] / 30.0,  # Venus scale
        len(game["passedPlayers"]) / 5.0,  # Players passed
        float(player["color"] in game["passedPlayers"]),
        game["undoCount"] / 10.0,  # Normalized undo count
        float(game["isTerraformed"]),  # Terraforming complete
        game["oxygenLevel"] / 14.0,
        (game["temperature"] + 30) / 82.0,
        game["oceans"] / 9.0,
    ])
    features.append((14 - game["oxygenLevel"]) / 14.0)
    features.append((8 - game["temperature"]) / 20.0)
    features.append(len(player_view_model.get("draftedCards", [])) / 10.0)

    max_generations = game.get("lastSoloGeneration", 14)
    features.append((max_generations - game["generation"]) / max_generations)
    # Expansions
    expansions = game["gameOptions"]["expansions"]
    features.extend([
        float(expansions["ares"]),
        float(expansions["ceo"]),
        float(expansions["colonies"]),
        float(expansions["community"]),
        float(expansions["corpera"]),
        float(expansions["moon"]),
        float(expansions["pathfinders"]),
        float(expansions["prelude"]),
        float(expansions["prelude2"]),
        float(expansions["promo"]),
        float(expansions["starwars"]),
        float(expansions["turmoil"]),
        float(expansions["underworld"]),
        float(expansions["venus"]),
    ])
    

    if "colonies" in game:
        tradable = sum(1 for c in game["colonies"] if c.get("isActive"))
        features.append(tradable / 10.0)
        for c in game["colonies"][:5]:
            features.append(c.get("trackPosition", 0) / 10.0)
        for c in game["colonies"][:5]:
            features.append(float("visitor" in c))
        owned = sum(player["color"] in c.get("colonies", []) for c in game["colonies"])
        features.append(owned / 5.0)
        active = sum(c.get("isActive", False) for c in game["colonies"])
        features.append(active / 10.0)
    else:
        features.append(0.0)
        features.extend([0.0]*5)
        features.extend([0.0]*5)
        features.append(0.0)
        features.append(0.0)

    if player["fleetSize"] > 0:
        features.append(player["tradesThisGeneration"] / player["fleetSize"])
    else:
        features.append(0.0)

    # Milestones and Awards
    player_color = player["color"]

    owned_milestones = sum(
        1 for m in game["milestones"]
        if any(s["playerColor"] == player_color for s in m["scores"])
    )

    owned_awards = sum(
        1 for a in game["awards"]
        if any(s["playerColor"] == player_color for s in a.get("scores", []))
    )

    features.extend([
        len(game["milestones"]) / 5.0,
        len(game["awards"]) / 5.0,
        owned_milestones / 5.0,
        owned_awards / 5.0,
    ])
    
    # Moon expansion
    if "moon" in game:
        moon = game["moon"]
        features.extend([
            moon["miningRate"] / 8.0,
            moon["habitatRate"] / 8.0,
            moon["logisticsRate"] / 8.0,
        ])
    else:
        features.extend([0.0, 0.0, 0.0])
    

    if "turmoil" in game:
        turmoil = game["turmoil"]
        features.extend([
            len(turmoil["lobby"]) / 7.0,  # Lobby delegates
            *one_hot_encode(turmoil.get("ruling"), ["marsFirst", "scientists", "unity", "greens", "reds", "kelvinists"]),
        ])
        features.append(float(game["turmoil"].get("chairman") == player["color"]))
        features.append(len(game["turmoil"]["parties"]) / 10.0)
        lobby = game["turmoil"].get("lobby", [])
        features.append(len(lobby) / 10.0)
        dominant = game["turmoil"].get("dominant")
        features.extend(one_hot_encode(dominant, ["marsFirst","scientists","unity","greens","reds","kelvinists"]))
    else:
        features.extend([0.0] * 7)  # lobby + 6 parties
        features.append(0.0)
        features.append(0.0)
        features.append(0.0)
        features.extend([0.0]*6)

    

    ruling = turmoil.get("ruling")
    party_bonus = {
        "marsFirst": 1,
        "scientists": 2,
        "unity": 3,
        "greens": 4,
        "reds": 5,
        "kelvinists": 6
    }
    features.append(party_bonus.get(ruling, 0) / 6.0)
    
    # 2. Player State Features

    
    # Resources and production
    resources = [
        ("megaCredits", 0, 200),
        ("steel", 0, 100),
        ("titanium", 0, 100),
        ("plants", 0, 20),
        ("energy", 0, 20),
        ("heat", 0, 50),
    ]
    
    production = [
        ("megaCreditProduction", -5, 20),
        ("steelProduction", 0, 15),
        ("titaniumProduction", 0, 15),
        ("plantProduction", 0, 10),
        ("energyProduction", 0, 10),
        ("heatProduction", 0, 15),
    ]
    
    for res, min_val, max_val in resources:
        features.append(normalize(player[res], min_val, max_val))
    
    for prod, min_val, max_val in production:
        features.append(normalize(player[prod], min_val, max_val))

    prod_vals = [player[p[0]] for p in production]
    features.append(np.std(prod_vals) / 10.0)
    
    features.append(player["steelValue"] / 5.0)
    features.append(player["titaniumValue"] / 5.0)
    features.append(float(player["heat"] >= 8))
    features.append(float(player["plants"] >= 8))
    # Player stats
    features.extend([
        player["terraformRating"] / 30.0,
        player["corruption"] / 10.0,
        player["citiesCount"] / 10.0,
        player["coloniesCount"] / 5.0,
        player["fleetSize"] / 10.0,
        player["tradesThisGeneration"] / 3.0,
        player.get("handicap",0) / 10.0,
        player["excavations"] / 10.0,
        player["availableBlueCardActionCount"] / 5.0,
        player["actionsTakenThisRound"] / 10.0,
        player["actionsTakenThisGame"] / 200.0,
        player["cardCost"] / 10.0,
        player["cardDiscount"] / 10.0,
        game["deckSize"] / 200.0,
    ])

    features.append(float(player["needsToResearch"]))
    features.append(player["cardsInHandNbr"] / 10.0)

    opp_hands = [p["cardsInHandNbr"] for p in players if p["color"] != player["color"]]
    if opp_hands:
        features.append((player["cardsInHandNbr"] - np.mean(opp_hands)) / 10.0)
    else:
        features.append(0.0)
    features.append(float(game["gameOptions"].get("draftVariant", False)))
    features.append(float(game["gameOptions"].get("initialDraftVariant", False)))
    features.append(float(game["gameOptions"].get("preludeDraftVariant", False)))
    features.append(float(game["gameOptions"].get("politicalAgendasExtension") is not None))

    last_card = player.get("lastCardPlayed")
    features.append(float(last_card is not None))

    ruling_party = game.get("turmoil", {}).get("ruling")
    is_ruling = float(ruling_party and player["color"] == game["turmoil"].get("chairman"))
    features.append(player["influence"] / 20.0)
    features.append(is_ruling)
    
    # Tags (normalized by total)
    tags = player["tags"]
    total_tags = max(1, sum(tags.values()))
    tag_categories = [
        "building", "space", "science", "power", "earth", 
        "jovian", "plant", "microbe", "animal", "city",
        "moon", "mars", "venus", "wild", "event", "clone"
    ]
    features.extend([tags.get(tag, 0) / total_tags for tag in tag_categories])
    features.append(len([t for t in tags.values() if t > 0]) / len(tag_categories))
    features.append(tags.get("jovian", 0) / 10.0)
    
    # Protected resources and production
    protections = ["megacredits", "steel", "titanium", "plants", "energy", "heat"]
    for res in protections:
        features.append(float(player["protectedResources"][res] != "none"))
        features.append(float(player["protectedProduction"][res] != "none"))
    
    # 3. Card Features
    # Tableau cards
    tableau = player["tableau"]
    tableau_features = [
        len(tableau) / 30.0,
        sum(1 for c in tableau if "event" in c.get("tags", {})) / 10.0,
        sum(c.get("resources", 0) for c in tableau) / 20.0,
    ]
    
    # Count tags in tableau
    tableau_tags = defaultdict(int)
    for card in tableau:
        for tag in card.get("tags", {}):
            tableau_tags[tag] += 1
    total_tableau_tags = max(1, sum(tableau_tags.values()))
    tableau_features.extend([tableau_tags.get(tag, 0) / total_tableau_tags for tag in tag_categories])
    
    features.extend(tableau_features)

    vp_history = player.get("victoryPointsByGeneration", [])
    if len(vp_history) >= 2:
        vp_trend = (vp_history[-1] - vp_history[-2]) / 10.0
    else:
        vp_trend = 0.0

    features.extend([
        player["victoryPointsBreakdown"]["total"] / 100.0,
        vp_trend,
    ])
    player_vp = player["victoryPointsBreakdown"]["total"]
    opp_vps = [p["victoryPointsBreakdown"]["total"] for p in players if p["color"] != player["color"]]
    vp_lead = player_vp - max(opp_vps) if opp_vps else 0
    features.append(vp_lead / 20.0)

    vps = sorted([p["victoryPointsBreakdown"]["total"] for p in players], reverse=True)
    rank = vps.index(player["victoryPointsBreakdown"]["total"]) + 1
    features.append(rank / 5.0)
    
    claimed = sum(len(m["scores"]) > 0 for m in game["milestones"])
    features.append(claimed / 5.0)

    funded = sum(len(a.get("scores", [])) > 0 for a in game["awards"])
    features.append(funded / 5.0)
    # Hand cards
    hand = player_view_model["cardsInHand"]
    hand_costs = [c.get("calculatedCost", 0) for c in hand]

    hand_features = [
        len(hand) / 10.0,
        np.mean(hand_costs) / 25.0 if hand_costs else 0.0,
        np.min(hand_costs) / 25.0 if hand_costs else 0.0,
        np.max(hand_costs) / 25.0 if hand_costs else 0.0,
    ]
    
    # Count tags in hand
    hand_tags = defaultdict(int)
    for card in hand:
        for tag in card.get("tags", {}):
            hand_tags[tag] += 1
    total_hand_tags = max(1, sum(hand_tags.values()))
    hand_features.extend([hand_tags.get(tag, 0) / total_hand_tags for tag in tag_categories])
    
    features.extend(hand_features)
    
    # 4. Opponent Features (relative to current player)
    for pn in range(6):
        if pn>=len(players):
            features.extend([0.0]*4)
        else:
            opp=players[pn]
            if opp["color"] == player["color"]:
                continue
            
            # Relative resource comparison
            features.extend([
                (opp["megaCredits"] - player["megaCredits"]) / 100.0,
                (opp["terraformRating"] - player["terraformRating"]) / 30.0,
                (opp["citiesCount"] - player["citiesCount"]) / 10.0,
                (opp["tags"].get("science", 0) - tags.get("science", 0)) / 10.0,
            ])
    
    # 5. Board State Features
    spaces = game["spaces"]
    total_tiles = max(1, sum(1 for s in spaces if "tileType" in s))

    board_features = [
        sum(1 for s in spaces if s.get("tileType") == "city") / total_tiles,
        sum(1 for s in spaces if s.get("tileType") == "greenery") / total_tiles,
        sum(1 for s in spaces if s.get("tileType") == "ocean") / total_tiles,
        sum(1 for s in spaces if s.get("color") == player["color"]) / total_tiles,
    ]

    greenery_spots = sum(1 for s in spaces if s.get("spaceType") == "land" and "tileType" not in s)
    features.append(greenery_spots / 50.0)
    city_spots = sum(1 for s in spaces if s.get("spaceType") == "land" and s.get("tileType") is None)
    features.append(city_spots / 50.0)
    features.append((9 - game["oceans"]) / 9.0)
    
    # Bonus counts
    bonus_counts = defaultdict(int)
    for space in spaces:
        for bonus in space.get("bonus", []):
            bonus_counts[bonus] += 1
    total_bonuses = max(1, sum(bonus_counts.values()))
    board_features.extend([bonus_counts.get(b, 0) / total_bonuses for b in range(1, 10)])  # Assuming bonus types 1-9
    
    features.extend(board_features)
    features.append(float(player_view_model.get("autopass", False)))
    timer = player.get("timer", {})
    features.append(timer.get("sumElapsed", 0) / 60000.0)
    features.append(float(timer.get("running", False)))

    features.append(player["corruption"] / 10.0)
    features.append(float(player["corruption"] > 0))

    # 2. Available Action Features

    action_features = np.zeros((MAX_ACTIONS, ONE_ACTION_ARRAY_SIZE), dtype=np.float32)
    action_features[:,0] = 0.0
    # Encode the current available action
    if action_slot_map:
            # Encode primary action
            #action_features[:ONE_ACTION_ARRAY_SIZE] = encode_action(available_actions)
            for slot, action in action_slot_map.items():
                feats = encode_action(action)   # must return size ONE_ACTION_ARRAY_SIZE
                action_features[int(slot)] = feats
                
                
    
    # Convert to numpy array
    observation = create_fixed_size_array(data_list=features,fixed_size=512)
    observation = np.concatenate([
        observation,
        action_features.flatten()
    ])
    # Final processing
    observation = np.nan_to_num(observation)  # Handle any NaN values
    observation = np.clip(observation, -10, 10)  # Clip extreme values
    
    return observation

def get_actions_shape():
    return np.zeros(TOTAL_ACTIONS, dtype=np.float32).shape
