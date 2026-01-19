import numpy as np
from typing import Dict, Any, List
import json
from myconfig import MAX_ACTIONS,ONE_ACTION_ARRAY_SIZE,MAX_GAME_FEATURES_SIZE, TOTAL_ACTIONS
from jsonpath_ng.ext import parse

with open("gs.schema.formatted.json",'rb') as f:
#with open("gs.response.schema.lite.json",'rb') as f:
    json_schema=json.loads(f.read())

ALL_CARD_NAMES=parse('$.definitions..CardName.enum').find(json_schema)[0].value

def decode_observation(observation: np.ndarray) -> Dict[str, Any]:
    """
    Decodes an observation array back into human-readable JSON format.
    
    Args:
        observation: The numpy array output from the observe() function
        
    Returns:
        A dictionary containing decoded game state and action information
    """
    GAME_FEATURE_SIZE=observation.shape[0]-MAX_ACTIONS*ONE_ACTION_ARRAY_SIZE
    
    # Split the observation into game features and action features
    game_features = observation[:GAME_FEATURE_SIZE]
    action_features_flat = observation[GAME_FEATURE_SIZE:]
    
    # Reshape action features to 2D array
    action_features = action_features_flat.reshape(MAX_ACTIONS, ONE_ACTION_ARRAY_SIZE)
    
    decoded = {
        "game_state": {},
        "actions": []
    }
    
    # Decode game features (reverse engineering from observe())
    # This is simplified - you'll need to adjust based on your exact feature encoding
    idx = 0
    
    # Basic game state features
    decoded["game_state"]["generation"] = int(game_features[idx])
    idx += 1
    
    # Phase indicators
    phase_mapping = ["action", "research", "production"]
    phase_idx = np.argmax(game_features[idx:idx+3])
    decoded["game_state"]["phase"] = phase_mapping[phase_idx]
    idx += 3
    
    # Environment levels (scaled back)
    decoded["game_state"]["oxygen_level"] = float(game_features[idx] * 14.0)
    idx += 1
    
    decoded["game_state"]["temperature"] = float(game_features[idx] * 82.0 - 30)
    idx += 1
    
    decoded["game_state"]["oceans"] = int(game_features[idx] * 9.0)
    idx += 1
    
    decoded["game_state"]["venus_scale"] = float(game_features[idx] * 30.0)
    idx += 1
    
    # Player counts
    decoded["game_state"]["passed_players"] = int(game_features[idx] * 5.0)
    idx += 1
    
    decoded["game_state"]["undo_count"] = int(game_features[idx] * 10.0)
    idx += 1
    
    decoded["game_state"]["is_terraformed"] = bool(game_features[idx] > 0.5)
    idx += 1
    
    # Expansions
    expansions = ["moon", "venus", "turmoil", "colonies"]
    decoded["game_state"]["expansions"] = {}
    for exp in expansions:
        decoded["game_state"]["expansions"][exp] = bool(game_features[idx] > 0.5)
        idx += 1
    
    # Milestones and Awards
    decoded["game_state"]["milestones"] = int(game_features[idx])
    idx += 1
    
    decoded["game_state"]["awards"] = int(game_features[idx])
    idx += 1
    
    # Moon expansion features
    if decoded["game_state"]["expansions"]["moon"]:
        decoded["game_state"]["moon"] = {
            "mining_rate": float(game_features[idx] * 8.0),
            "habitat_rate": float(game_features[idx+1] * 8.0),
            "logistics_rate": float(game_features[idx+2] * 8.0)
        }
        idx += 3
    else:
        idx += 3  # Skip the zeroed features
    
    # Turmoil features
    if decoded["game_state"]["expansions"]["turmoil"]:
        parties = ["marsFirst", "scientists", "unity", "greens", "reds", "kelvinists"]
        party_probs = game_features[idx+1:idx+7]
        ruling_idx = np.argmax(party_probs)
        
        decoded["game_state"]["turmoil"] = {
            "lobby_delegates": int(game_features[idx] * 7.0),
            "ruling_party": parties[ruling_idx] if np.max(party_probs) > 0.5 else "none"
        }
        idx += 7
    else:
        idx += 7  # Skip the zeroed features
    
    # Decode actions
    action_types = [
        "and", "or", "initialCards", "amount", "card",
        "colony", "delegate", "option", "party", "payment",
        "player", "productionToLose", "projectCard", "space",
        "aresGlobalParameters", "globalEvent", "policy",
        "resource", "resources", "xcard_choose", "xconfirm_card_choose","xremove_first_card_choose", "xpayment"
    ]
    
    for i in range(MAX_ACTIONS):
        action_vec = action_features[i]
        
        # Skip empty actions (all zeros)
        if np.all(action_vec == 0):
            continue
        
        # Decode action type
        action_type_val = action_vec[0]
        action_type_idx = int(action_type_val * len(action_types))
        
        if 0 <= action_type_idx < len(action_types):
            action_type = action_types[action_type_idx]
        else:
            action_type = "unknown"
        
        action_decoded = {
            "type": action_type,
            "slot": i,
            "details": {}
        }
        
        # Decode action-specific features
        if action_type == "amount":
            action_decoded["details"]["amount"] = float(action_vec[1] * 50.0)
            action_decoded["details"]["min"] = float(action_vec[2] * 50.0)
            action_decoded["details"]["max"] = float(action_vec[3] * 50.0)
        
        
        elif action_type in ["payment", "xpayment"]:
            resources = [
                "megaCredits", "steel", "titanium", "heat", "plants",
                "microbes", "floaters", "lunaArchivesScience", "spireScience",
                "seeds", "auroraiData", "graphene", "kuiperAsteroids", "corruption"
            ]
            payment = {}
            for j, res in enumerate(resources):
                payment[res] = float(action_vec[j+1] * (100.0 if j == 0 else 20.0))
            action_decoded["details"]["payment"] = payment
        
        elif action_type == "xcard_choose":
            # Decode card name from normalized index
            card_idx_normalized = action_vec[1]
            if card_idx_normalized > 0 and ALL_CARD_NAMES:
                # Reverse the encoding: (index + 1) / len(ALL_CARD_NAMES)
                card_idx = int(card_idx_normalized * len(ALL_CARD_NAMES)) - 1
                if 0 <= card_idx < len(ALL_CARD_NAMES):
                    action_decoded["details"]["card_name"] = ALL_CARD_NAMES[card_idx]
                else:
                    action_decoded["details"]["card_name"] = f"Unknown (index: {card_idx})"
            else:
                action_decoded["details"]["card_name"] = "No card selected"
            
            action_decoded["details"]["card_index"] = float(card_idx_normalized)

        elif action_type == "player":
            action_decoded["details"]["player"] = "NOT_NEUTRAL" if action_vec[1] > 0.5 else "NEUTRAL"
        
        elif action_type == "productionToLose":
            action_decoded["details"]["units"] = {
                "energy": float(action_vec[1] * 5.0),
                "heat": float(action_vec[2] * 5.0),
                "steel": float(action_vec[3] * 5.0)
            }
        
        # Add confidence scores
        action_decoded["confidence"] = {
            "type_confidence": float(action_vec[0]),
            "non_zero_features": int(np.sum(action_vec > 0.01))
        }
        
        decoded["actions"].append(action_decoded)
    
    # Add metadata
    decoded["metadata"] = {
        "total_features": len(observation),
        "game_features_count": MAX_GAME_FEATURES_SIZE,
        "action_features_count": MAX_ACTIONS * ONE_ACTION_ARRAY_SIZE,
        "non_zero_game_features": int(np.sum(game_features > 0.01)),
        "active_actions": len(decoded["actions"])
    }
    
    return decoded


def decode_observation_to_json(observation: np.ndarray, pretty: bool = True) -> str:
    """
    Decodes observation and returns as JSON string.
    
    Args:
        observation: The numpy array output from observe()
        pretty: Whether to format the JSON with indentation
        
    Returns:
        JSON string of decoded observation
    """
    decoded = decode_observation(observation)
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    if pretty:
        return json.dumps(decoded, indent=2, cls=NumpyEncoder)
    else:
        return json.dumps(decoded, cls=NumpyEncoder)


# Example usage function
def decode_and_save(observation: np.ndarray, output_file: str = "decoded_observation.json"):
    """
    Decodes an observation and saves it to a JSON file.
    
    Args:
        observation: The numpy array output from observe()
        output_file: Path to save the decoded JSON
    """
    decoded_json = decode_observation_to_json(observation, pretty=True)
    
    with open(output_file, 'w') as f:
        f.write(decoded_json)
    
    print(f"Decoded observation saved to {output_file}")
    
    # Also print a summary
    decoded = decode_observation(observation)
    print(f"\nSummary:")
    print(f"  - Game Generation: {decoded['game_state'].get('generation', 'N/A')}")
    print(f"  - Phase: {decoded['game_state'].get('phase', 'N/A')}")
    print(f"  - Active Actions: {decoded['metadata']['active_actions']}")
    print(f"  - Action Types: {[a['type'] for a in decoded['actions']]}")