from typing import Any, Dict

import numpy as np

from myconfig import ONE_ACTION_ARRAY_SIZE
from observe_gamestate import ALL_CARD_NAMES, normalize

def _encode_payment(action_vec, payment, offset):
    pass

def encode_action(action: Dict[str, Any], is_main=True) -> np.ndarray:
    """
    IMPROVED: More efficient action encoding with better feature extraction
    """
    action_vec = np.zeros(ONE_ACTION_ARRAY_SIZE, dtype=np.float32)
    
    if not action:
        return action_vec
    
    action_type = action.get("xtype", action.get("type", ""))
    
    # Action type encoding (categorical)
    action_types = [
        "and", "or", "initialCards", "amount", "card",
        "colony", "delegate", "option", "party", "payment",
        "player", "productionToLose", "projectCard", "space",
        "aresGlobalParameters", "globalEvent", "policy",
        "resource", "resources", "xcard_choose", "xconfirm_card_choose",
        "xremove_first_card_choose", "xpayment"
    ]
    
    if action_type in action_types:
        action_vec[0] = action_types.index(action_type) / len(action_types)
    
    # Type-specific encoding (more compact)
    if action_type == "and":
        for i, resp in enumerate(action.get("responses", [])[:8]):
            action_vec[1 + i * 8:1 + (i + 1) * 8] = encode_action(resp, False)[:8]
    
    elif action_type == "or":
        action_vec[1] = action.get("index", 0) / 10.0
        if "response" in action:
            action_vec[2:34] = encode_action(action["response"], False)[:32]
    
    elif action_type == "amount":
        action_vec[1] = normalize(action.get("amount", 0), 0, 50)
        action_vec[2] = normalize(action.get("min", 0), 0, 50)
        action_vec[3] = normalize(action.get("max", 0), 0, 50)
    
    elif action_type == "card":
        cards = action.get("cards", [])
        if isinstance(cards, dict):
            action_vec[1:33] = encode_action(cards, False)[:32]
        else:
            for i, card in enumerate(cards[:32]):
                if card.get("name") in ALL_CARD_NAMES:
                    action_vec[1 + i] = (hash(card['name']) % 1000) / 1000.0
    
    elif action_type in ["xcard_choose", "xpayment", "xconfirm_card_choose", "xremove_first_card_choose"]:
        if action_type == "xcard_choose":
            card = action.get("xoption", {})
            action_vec[1] = (hash(card.get('name', '')) % 1000) / 1000.0
        elif action_type in ["xconfirm_card_choose", "xremove_first_card_choose"]:
            action_vec[1] = 1.0
        elif action_type == "xpayment":
            payment = action.get("xoption", {})
            _encode_payment(action_vec, payment, offset=1)
    
    elif action_type == "payment":
        payment = action.get("payment", {})
        _encode_payment(action_vec, payment, offset=1)
    
    elif action_type == "projectCard":
        action_vec[1] = (hash(str(action.get('card', ''))) % 1000) / 1000.0
        cards = action.get("cards", [])
        for i, card in enumerate(cards[:24]):
            if card.get("name") in ALL_CARD_NAMES:
                action_vec[2 + i] = (hash(card["name"]) % 1000) / 1000.0
        if "payment" in action:
            _encode_payment(action_vec, action["payment"], offset=26)
    
    elif action_type == "space":
        pass
        #TODO: space encode
        # sid = action.get("spaceId", "0")
        # action_vec[1] = int(sid) / 100.0 if sid.isdigit() else 0.0
        # spaces = action.get("spaces", [])
        # for i, space in enumerate(spaces[:32]):
        #     sid = space.get("id", "0")
        #     action_vec[2 + i] = int(sid) / 100.0 if sid.isdigit() else 0.0
    
    return action_vec