import numpy as np
import json
from pettingzoo import AECEnv

# Define a helper function to safely get nested values with a default
def get_nested(d, keys, default=0):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if isinstance(d, (int, float, str)) else default

# Normalize numeric values to [0, 1] range with assumed game-specific max values
NORMALIZATION_FACTORS = {
    "megaCredits": 100,
    "steel": 50,
    "titanium": 30,
    "plants": 50,
    "energy": 50,
    "heat": 100,
    "megaCreditProduction": 15,
    "steelProduction": 10,
    "titaniumProduction": 10,
    "plantProduction": 10,
    "energyProduction": 10,
    "heatProduction": 10,
    "temperature": 100,
    "oxygenLevel": 14,
    "oceans": 9,
    "generation": 20,
    "venusScaleLevel": 30
}

# Categorical encodings
COLOR_ENCODING = {
    "red": 0,
    "green": 1,
    "yellow": 2,
    "blue": 3,
    "black": 4,
    "purple": 5,
    "orange": 6,
    "pink": 7,
    "neutral": 8,
    "bronze": 9
}

PARTY_ENCODING = {
    "Mars First": 0,
    "Scientists": 1,
    "Unity": 2,
    "Kelvinists": 3,
    "Reds": 4,
    "Greens": 5
}

# Maximum number of available actions to encode (pad or truncate to this size)
MAX_ACTIONS = 20

FEATURE_KEYS = [
    (["thisPlayer", "megaCredits"], "megaCredits"),
    (["thisPlayer", "steel"], "steel"),
    (["thisPlayer", "titanium"], "titanium"),
    (["thisPlayer", "plants"], "plants"),
    (["thisPlayer", "energy"], "energy"),
    (["thisPlayer", "heat"], "heat"),
    (["thisPlayer", "megaCreditProduction"], "megaCreditProduction"),
    (["thisPlayer", "steelProduction"], "steelProduction"),
    (["thisPlayer", "titaniumProduction"], "titaniumProduction"),
    (["thisPlayer", "plantProduction"], "plantProduction"),
    (["thisPlayer", "energyProduction"], "energyProduction"),
    (["thisPlayer", "heatProduction"], "heatProduction"),
    (["game", "temperature"], "temperature"),
    (["game", "oxygenLevel"], "oxygenLevel"),
    (["game", "oceans"], "oceans"),
    (["game", "generation"], "generation"),
    (["game", "venusScaleLevel"], "venusScaleLevel")
]

CATEGORICAL_KEYS = [
    (["thisPlayer", "color"], COLOR_ENCODING, 10),
    (["game", "turmoil", "dominant"], PARTY_ENCODING, 6)
]

# Define the observation function
def observe(env: AECEnv, player_view_json: str) -> np.ndarray:
    try:
        player_view = json.loads(player_view_json)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON input")

    features = []

    # Process numeric fields with normalization
    for keys, norm_key in FEATURE_KEYS:
        value = get_nested(player_view, keys)
        max_value = NORMALIZATION_FACTORS.get(norm_key, 1)
        features.append(min(value / max_value, 1.0))

    # Process categorical fields as one-hot
    for keys, encoding_map, depth in CATEGORICAL_KEYS:
        category = get_nested(player_view, keys, default=None)
        one_hot = [0.0] * depth
        if category in encoding_map:
            one_hot[encoding_map[category]] = 1.0
        features.extend(one_hot)

    # Encode legal actions as one-hot padded vector
    legal_actions = []
    waiting_input = player_view.get("waitingFor")
    if isinstance(waiting_input, dict) and "options" in waiting_input:
        for option in waiting_input["options"]:
            encoded = hash(json.dumps(option, sort_keys=True)) % (10**6)
            legal_actions.append(encoded)
    elif isinstance(waiting_input, dict):
        encoded = hash(json.dumps(waiting_input, sort_keys=True)) % (10**6)
        legal_actions.append(encoded)

    # Convert legal actions to one-hot fixed size array
    padded = [0.0] * MAX_ACTIONS
    for i in range(min(MAX_ACTIONS, len(legal_actions))):
        padded[i] = legal_actions[i] / 10**6  # Normalize to [0, 1]

    features.extend(padded)

    return np.array(features, dtype=np.float32)

# Example usage (assuming env is a PettingZoo AECEnv object):
# obs = observe(env, player_view_json)
# print(obs.shape)
