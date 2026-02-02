"""
Factored (Hierarchical) Action Encoding for Terraforming Mars.

Every legal action is decomposed into exactly 4 factors:

    (action_type, object_id, payment_id, extra_param)

This gives the policy 4 stationary heads whose output dimensions never
change regardless of how many concrete actions are legal right now.
Learning transfers across game states because the same "play projectCard
with card_index=5" pattern always maps to the same (type, obj, pay, extra)
regardless of which other cards are available.

Factor semantics
----------------
Head 0 – action_type  (16 classes)
    Maps every possible top-level decision type to a fixed index.
    Index 0 is reserved for PASS / no-op.

Head 1 – object_id  (32 slots)
    Identifies *what* the action targets.  Card indices, standard-project
    IDs, colony names, space buckets, etc. are all hashed into [0, 32).
    When an action has no target (e.g. PASS), object_id = 0.

Head 2 – payment_id  (16 slots)
    Identifies *how* the action is paid for.  Payment option indices from
    the PaymentOptionsCalculator are hashed into [0, 16).
    Actions with no payment component use payment_id = 0.

Head 3 – extra_param  (16 slots)
    Captures any remaining sub-choice: confirm vs. cancel, amount in a
    range, sub-bucket selection in space narrowing, etc.
    Default = 0 when unused.

Conditional masking
-------------------
The env computes 4 nested mask dicts after every action-space generation:

    head_masks[0]                          → which action_types exist
    cond_masks_obj_by_type[t]              → which object_ids exist for type t
    cond_masks_pay_by_obj[(t, o)]          → which payment_ids exist for (t, o)
    cond_masks_extra_by_pay[(t, o, p)]     → which extras exist for (t, o, p)

MCTS and the policy apply these masks sequentially so that only
combinations that correspond to an actual legal action can be sampled.
"""

from typing import Dict, Any, Tuple

# --------------- Public constants ------------------------------------------------
ACTION_TYPE_DIM   = 16
OBJECT_ID_DIM     = 32
PAYMENT_ID_DIM    = 16
EXTRA_PARAM_DIM   = 16
NUM_HEADS         = 4
FACTORED_ACTION_DIMS = (ACTION_TYPE_DIM, OBJECT_ID_DIM, PAYMENT_ID_DIM, EXTRA_PARAM_DIM)

# --------------- Action-type registry --------------------------------------------
# Every xtype / type string that can appear in an action dict is mapped to a
# fixed index.  New types can be appended (up to ACTION_TYPE_DIM - 1); index 0
# is PASS.
_ACTION_TYPE_TO_IDX: Dict[str, int] = {
    "PASS":                         0,
    "or":                           1,
    "and":                          2,
    "card":                         3,
    "projectCard":                  4,
    "payment":                      5,
    "space":                        6,
    "amount":                       7,
    "option":                       8,
    "colony":                       9,
    "delegate":                    10,
    "party":                       11,
    "resource":                    12,
    "resources":                   13,
    "deffered":                    14,   # covers all xtype sub-variants
    "initialCards":                15,
}

# --------------- object_id sub-registries ----------------------------------------
# Standard projects have well-known names; we give them stable slots so the
# network can learn "OCEAN always means raise oceans" across games.
_STANDARD_PROJECT_SLOTS: Dict[str, int] = {
    "OCEAN":         1,
    "TEMPERATURE":   2,
    "OXYGEN":        3,
    "ASTEROID":      4,
    "SOLAR_REWARD":  5,
    "GREENERY":      6,
    "CITY":          7,
}

# Deferred sub-types that need distinct object_ids so the network can
# distinguish "pick a card" from "pick a payment resource" etc.
_DEFERRED_SUBTYPE_SLOTS: Dict[str, int] = {
    "xcard_choose":              16,
    "xconfirm_card_choose":      17,
    "xremove_first_card_choose": 18,
    "xpayment_confirm":          19,
    "xpayment_choose_resource":  20,
    "xpayment_choose_amount":    21,
    "xspace_choose_level0":      22,
    "xspace_choose_level1":      23,
    "xspace_choose_level2":      24,
}


class FactoredActionEncoder:
    """Stateless encoder/decoder between action dicts and (t, o, p, e) tuples."""

    # ------------------------------------------------------------------
    # encode  –  action_dict  →  (type_idx, obj_idx, pay_idx, extra_idx)
    # ------------------------------------------------------------------
    def encode(self, action: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Deterministic encoding of any legal-action dict produced by
        TerraformingMarsDecisionMapper.generate_action_space().
        """
        if action is None or action == {}:
            return (0, 0, 0, 0)   # PASS

        action_type = action.get("type", "")
        xtype = action.get("xtype", "")

        # --- Head 0: action_type -------------------------------------------
        if action_type == "deffered":
            t = _ACTION_TYPE_TO_IDX.get("deffered", 14)
        else:
            t = _ACTION_TYPE_TO_IDX.get(action_type, 0)

        # --- Head 1: object_id ---------------------------------------------
        o = self._encode_object_id(action, action_type, xtype)

        # --- Head 2: payment_id --------------------------------------------
        p = self._encode_payment_id(action, action_type)

        # --- Head 3: extra_param -------------------------------------------
        e = self._encode_extra(action, action_type, xtype)

        return (t, o, p, e)

    def encode_slot(self, slot: int) -> Tuple[int, int, int, int]:
        """
        Fallback encoder for a flat slot index (used when the replay buffer
        does not yet store the full factored tuple).  Deterministic spread
        across the factor space so gradients still flow to all heads.
        """
        t = slot % ACTION_TYPE_DIM
        o = (slot * 7 + 3) % OBJECT_ID_DIM
        p = (slot * 13 + 5) % PAYMENT_ID_DIM
        e = (slot * 11 + 2) % EXTRA_PARAM_DIM
        return (t, o, p, e)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _encode_object_id(self, action: Dict, action_type: str, xtype: str) -> int:
        """Map the action's target to a stable object_id in [0, OBJECT_ID_DIM)."""

        if action_type == "projectCard":
            # Card name → stable hash slot in [1, 16)
            card_name = action.get("card", "")
            return 1 + (hash(card_name) % 15)

        if action_type == "payment":
            # Payment-only actions (standard projects) – look for project name in title
            title = action.get("title", {})
            if isinstance(title, dict):
                msg = title.get("message", "")
                for proj, slot in _STANDARD_PROJECT_SLOTS.items():
                    if proj.lower() in msg.lower():
                        return slot
            return 0

        if action_type == "card":
            # Card-selection: hash the deferred xid for uniqueness, or use 0
            cards = action.get("cards", {})
            if isinstance(cards, dict) and "__deferred_action" in cards:
                return 1 + (hash(cards["__deferred_action"].get("xid", "")) % 15)
            return 0

        if action_type == "space":
            # Space narrowing starts at level0; the spaceId/bucket is in the deferred
            spaceId = action.get("spaceId", {})
            if isinstance(spaceId, dict) and "__deferred_action" in spaceId:
                return 1 + (hash(spaceId["__deferred_action"].get("xid", "")) % 15)
            return 0

        if action_type == "colony":
            return 1 + (hash(action.get("colonyName", "")) % 15)

        if action_type == "party":
            return 1 + (hash(action.get("partyName", "")) % 15)

        if action_type == "delegate":
            return 1 + (hash(str(action.get("player", ""))) % 15)

        if action_type == "deffered":
            # Use the deferred sub-type to pick a stable slot
            return _DEFERRED_SUBTYPE_SLOTS.get(xtype, 0)

        if action_type == "or":
            # OR index distinguishes siblings
            return action.get("index", 0) % OBJECT_ID_DIM

        if action_type == "amount":
            # The amount value itself is the object
            return action.get("amount", 0) % OBJECT_ID_DIM

        if action_type == "option":
            return 0   # single-option confirm

        if action_type == "initialCards":
            return 0

        return 0

    def _encode_payment_id(self, action: Dict, action_type: str) -> int:
        """Map payment options to a stable payment_id in [0, PAYMENT_ID_DIM)."""
        # projectCard actions have a deferred payment; hash the xid
        if action_type == "projectCard":
            payment = action.get("payment", {})
            if isinstance(payment, dict) and "__deferred_action" in payment:
                xid = payment["__deferred_action"].get("xid", "")
                return hash(xid) % PAYMENT_ID_DIM
            return 0

        # payment actions likewise
        if action_type == "payment":
            payment = action.get("payment", {})
            if isinstance(payment, dict) and "__deferred_action" in payment:
                xid = payment["__deferred_action"].get("xid", "")
                return hash(xid) % PAYMENT_ID_DIM
            return 0

        # Deferred payment sub-steps: use xoption hash
        if action_type == "deffered":
            xtype = action.get("xtype", "")
            if xtype in ("xpayment_confirm", "xpayment_choose_resource", "xpayment_choose_amount"):
                xopt = action.get("xoption", "")
                return hash(str(xopt)) % PAYMENT_ID_DIM

        return 0

    def _encode_extra(self, action: Dict, action_type: str, xtype: str) -> int:
        """Map sub-choice / amount / confirmation flags to extra_param in [0, EXTRA_PARAM_DIM)."""

        if action_type == "deffered":
            if xtype in ("xconfirm_card_choose",):
                return 1   # confirm = 1
            if xtype in ("xremove_first_card_choose",):
                return 2   # remove = 2
            if xtype == "xpayment_choose_amount":
                # The amount itself
                return action.get("xoption", 0) % EXTRA_PARAM_DIM
            if xtype in ("xspace_choose_level0", "xspace_choose_level1", "xspace_choose_level2"):
                # Bucket / sub-bucket / space index
                return int(action.get("xoption", 0)) % EXTRA_PARAM_DIM
            if xtype == "xcard_choose":
                # Card position among options
                xoptions = action.get("xoptions", [])
                card_name = action.get("xoption", {}).get("name", "")
                for i, c in enumerate(xoptions):
                    if c.get("name") == card_name:
                        return i % EXTRA_PARAM_DIM
                return 0

        if action_type == "amount":
            return action.get("amount", 0) % EXTRA_PARAM_DIM

        if action_type == "or":
            # sub-option index within the chosen OR branch
            resp = action.get("response", {})
            if isinstance(resp, dict) and resp.get("type") == "amount":
                return resp.get("amount", 0) % EXTRA_PARAM_DIM
            return 0

        return 0