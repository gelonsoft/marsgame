import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
import json
from jsonpath_ng.ext import parse
from myconfig import MAX_ACTIONS,ONE_ACTION_ARRAY_SIZE,MAX_GAME_FEATURES_SIZE, TOTAL_ACTIONS

with open("gs.schema.formatted.json",'rb') as f:
#with open("gs.response.schema.lite.json",'rb') as f:
    json_schema=json.loads(f.read())

ALL_CARD_NAMES=parse('$.definitions..CardName.enum').find(json_schema)[0].value


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
    action_vec = np.zeros(1024, dtype=np.float32)
    
    if not action:
        return action_vec
    
    action_type = action.get("type", "")
    
    # 1. Categorical encoding for action type (first dimension)
    action_types = [
        "and", "or", "initialCards", "amount", "card",
        "colony", "delegate", "option", "party", "payment",
        "player", "productionToLose", "projectCard", "space",
        "aresGlobalParameters", "globalEvent", "policy",
        "resource", "resources"
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
        for i, card in enumerate(cards[:256]):  # Max 256 cards
            if card["name"] in ALL_CARD_NAMES:
                action_vec[1+i] = (ALL_CARD_NAMES.index(card["name"]) + 1) / len(ALL_CARD_NAMES)  # Normalized index
        
        # Additional card features (positions 257-320)
        for i, card in enumerate(cards[:64]):
            action_vec[257+i] = card.get("calculatedCost", 0) / 50.0
            action_vec[321+i] = float("science" in card.get("tags", {}))
            action_vec[385+i] = float("building" in card.get("tags", {}))
    
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
            action_vec[1+i] = hash(space.get("id", "")) % 1000 / 1000  # Simple categorical encoding
        
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
    
def observe(player_view_model: Dict[str, Any],available_actions: Dict[str,Any]) -> np.ndarray:
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
    
    # Basic game state
    features.extend([
        game["generation"],  # Current generation
        float(game["phase"] == "action"),  # Phase indicators
        float(game["phase"] == "research"),
        float(game["phase"] == "production"),
        game["oxygenLevel"] / 14.0,  # Normalized oxygen
        (game["temperature"] + 30) / 82.0,  # Normalized temperature
        game["oceans"] / 9.0,  # Ocean tiles
        game["venusScaleLevel"] / 30.0,  # Venus scale
        len(game["passedPlayers"]) / 5.0,  # Players passed
        game["undoCount"] / 10.0,  # Normalized undo count
        float(game["isTerraformed"]),  # Terraforming complete
    ])
    
    # Expansions
    expansions = game["gameOptions"]["expansions"]
    features.extend([
        float(expansions["moon"]),
        float(expansions["venus"]),
        float(expansions["turmoil"]),
        float(expansions["colonies"]),
    ])
    
    # Milestones and Awards
    features.extend([
        len(game["milestones"]),  # Claimed milestones
        len(game["awards"]),  # Funded awards
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
    
    # Turmoil
    if "turmoil" in game:
        turmoil = game["turmoil"]
        features.extend([
            len(turmoil["lobby"]) / 7.0,  # Lobby delegates
            *one_hot_encode(turmoil.get("ruling"), ["marsFirst", "scientists", "unity", "greens", "reds", "kelvinists"]),
        ])
    else:
        features.extend([0.0] * 7)  # lobby + 6 parties
    
    # 2. Player State Features
    player = player_view_model["thisPlayer"]
    players = player_view_model["players"]
    
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
    
    # Player stats
    features.extend([
        player["terraformRating"] / 30.0,
        player["influence"] / 20.0,
        player["corruption"] / 10.0,
        player["citiesCount"] / 10.0,
        player["coloniesCount"] / 5.0,
        player["fleetSize"] / 10.0,
        player.get("handicap",0) / 10.0,
        player["excavations"] / 10.0,
    ])
    
    # Tags (normalized by total)
    tags = player["tags"]
    total_tags = max(1, sum(tags.values()))
    tag_categories = [
        "building", "space", "science", "power", "earth", 
        "jovian", "plant", "microbe", "animal", "city",
        "moon", "mars", "venus", "wild", "event", "clone"
    ]
    features.extend([tags.get(tag, 0) / total_tags for tag in tag_categories])
    
    # Protected resources and production
    protections = ["megacredits", "steel", "titanium", "plants", "energy", "heat"]
    for res in protections:
        features.append(float(player["protectedResources"][res] != "none"))
        features.append(float(player["protectedProduction"][res] != "none"))
    
    # 3. Card Features
    # Tableau cards
    tableau = player["tableau"]
    tableau_features = [
        len(tableau),  # Total cards
        sum(1 for c in tableau if "event" in c.get("tags", {})),  # Event cards
        sum(c.get("resources", 0) for c in tableau) / max(1, len(tableau)),  # Avg resources
    ]
    
    # Count tags in tableau
    tableau_tags = defaultdict(int)
    for card in tableau:
        for tag in card.get("tags", {}):
            tableau_tags[tag] += 1
    total_tableau_tags = max(1, sum(tableau_tags.values()))
    tableau_features.extend([tableau_tags.get(tag, 0) / total_tableau_tags for tag in tag_categories])
    
    features.extend(tableau_features)
    
    # Hand cards
    hand = player_view_model["cardsInHand"]
    hand_features = [
        len(hand),  # Cards in hand
        sum(c.get("calculatedCost", 0) for c in hand) / max(1, len(hand)),  # Avg cost
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
    for opp in players:
        if opp["color"] == player["color"]:
            continue
        
        # Relative resource comparison
        features.extend([
            opp["megaCredits"] - player["megaCredits"],
            opp["terraformRating"] - player["terraformRating"],
            opp["citiesCount"] - player["citiesCount"],
            opp["tags"].get("science", 0) - tags.get("science", 0),
        ])
    
    # Pad for max players (assuming max 5 players total)
    num_opponents = len(players) - 1
    if num_opponents < 4:  # Assuming 1 current player + 4 max opponents
        features.extend([0.0] * (4 - num_opponents) * 4)  # 4 features per opponent
    
    # 5. Board State Features
    spaces = game["spaces"]
    board_features = [
        sum(1 for s in spaces if s.get("tileType") == "city"),  # Total cities
        sum(1 for s in spaces if s.get("tileType") == "greenery"),  # Total greenery
        sum(1 for s in spaces if s.get("tileType") == "ocean"),  # Total oceans
        sum(1 for s in spaces if s.get("color") == player["color"]),  # Player tiles
    ]
    
    # Bonus counts
    bonus_counts = defaultdict(int)
    for space in spaces:
        for bonus in space.get("bonus", []):
            bonus_counts[bonus] += 1
    total_bonuses = max(1, sum(bonus_counts.values()))
    board_features.extend([bonus_counts.get(b, 0) / total_bonuses for b in range(1, 10)])  # Assuming bonus types 1-9
    
    features.extend(board_features)
    
    # 2. Available Action Features

    action_features = np.zeros(MAX_ACTIONS*ONE_ACTION_ARRAY_SIZE, dtype=np.float32)
    # Encode the current available action
    if available_actions:
            # Encode primary action
            action_features[:ONE_ACTION_ARRAY_SIZE] = encode_action(available_actions)
            
            # Encode nested actions for AND/OR types
            if available_actions.get("type") in ["and", "or"]:
                nested_actions = available_actions.get("responses", [])[:MAX_ACTIONS-1]
                for i, nested_action in enumerate(nested_actions):
                    start_idx = (i+1) * ONE_ACTION_ARRAY_SIZE
                    action_features[start_idx:start_idx+ONE_ACTION_ARRAY_SIZE] = encode_action(nested_action)
    
    # Combine all features
    #features.extend(action_features)
    
    # Convert to numpy array
    observation = np.array(features, dtype=np.float32)
    observation=observation[:MAX_GAME_FEATURES_SIZE]
    observation = np.concatenate([
        observation,
        action_features
    ])
    # Final processing
    observation = np.nan_to_num(observation)  # Handle any NaN values
    observation = np.clip(observation, -10, 10)  # Clip extreme values
    
    return observation

def get_actions_shape():
    return np.zeros(TOTAL_ACTIONS, dtype=np.float32).shape
