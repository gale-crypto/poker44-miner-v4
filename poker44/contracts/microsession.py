"""Strict schema-v4.1 micro-session validation shared by validators and miners."""

from __future__ import annotations

from typing import Any

V4_SESSION_KEYS = {"schema_version", "item_id", "window_id", "decisions"}
STRATEGIC_DECISION_KEYS = {
    "decision_number",
    "phase",
    "position_group",
    "pressure",
    "action_type",
    "size_bucket",
    "is_all_in",
}
FORBIDDEN_KEYS = {
    "actor_group",
    "bot_family",
    "ground_truth",
    "is_bot",
    "is_human",
    "label",
}
PHASES = {"preflop", "flop", "turn", "river"}
POSITION_GROUPS = {"early", "late", "blinds"}
PRESSURES = {"facing_bet", "no_call"}
ACTION_TYPES = {"fold", "check", "call", "bet", "raise", "all_in"}
SIZE_BUCKETS = {
    "not_applicable",
    "unknown",
    "third_pot_or_less",
    "half_pot",
    "three_quarter_pot",
    "pot",
    "overbet",
    "all_in",
}


def find_forbidden(value: Any, path: str = "session") -> list[str]:
    leaked: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_KEYS:
                leaked.append(child_path)
            leaked.extend(find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaked.extend(find_forbidden(child, f"{path}[{index}]"))
    return leaked


def validate_v4_micro_session(session: dict[str, Any], index: int) -> None:
    prefix = f"sessions[{index}]"
    if set(session) != V4_SESSION_KEYS:
        raise ValueError(f"{prefix} does not match the micro-session contract")
    if str(session.get("schema_version") or "") != "4.1":
        raise ValueError(f"{prefix} has an unsupported micro-session schema")
    if not str(session.get("item_id") or "").strip():
        raise ValueError(f"{prefix} has no item_id")
    if not str(session.get("window_id") or "").strip():
        raise ValueError(f"{prefix} has no window_id")
    decisions = session.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 4:
        raise ValueError(f"{prefix} must contain exactly 4 strategic decisions")
    postflop = 0
    for decision_index, decision in enumerate(decisions):
        path = f"{prefix}.decisions[{decision_index}]"
        if not isinstance(decision, dict) or set(decision) != STRATEGIC_DECISION_KEYS:
            raise ValueError(f"{path} does not match the strategic contract")
        if not isinstance(decision["decision_number"], int):
            raise ValueError(f"{path}.decision_number must be an integer")
        if not isinstance(decision["is_all_in"], bool):
            raise ValueError(f"{path}.is_all_in must be boolean")
        enum_fields = {
            "phase": PHASES,
            "position_group": POSITION_GROUPS,
            "pressure": PRESSURES,
            "action_type": ACTION_TYPES,
            "size_bucket": SIZE_BUCKETS,
        }
        for key, allowed in enum_fields.items():
            if decision[key] not in allowed:
                raise ValueError(f"{path}.{key} is invalid")
        if decision["phase"] in {"flop", "turn", "river"}:
            postflop += 1
    if postflop < 1:
        raise ValueError(f"{prefix} must contain at least 1 postflop decision")
