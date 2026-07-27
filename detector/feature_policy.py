"""Drop feature families that are unstable between benchmark and live payloads.

Measured against real validator payloads, these families sit 3-35 sigma outside
the benchmark distribution, which means every tree split on them routes the whole
live batch down one branch and contributes nothing to the ranking:

    hero_*            hero_seat is a per-hand alias the validator re-derives in
                      order of first action, and the 5-8 action window often does
                      not contain hero at all.
    stack_*           benchmark stacks average ~240bb and vary; live is pinned to
                      a 100bb buy-in (measured 100.04 / 100.02 / 99.999).
    showdown*         payload_view hardcodes "showdown": False, so this is a
                      constant at inference time.
    player_count      benchmark is a constant 6 seats; live reaches 9.
    seat_utilization  same cause.
    button_*          payload_view hardcodes "button_seat": 0.
    hand_count*       benchmark chunks are 30-40 hands, live 80-100.

This mirrors the policy in the highest-scoring miner on the subnet, which moved
from 0.527 to 0.6904 live across the commits that introduced it.

Set POKER44_DRIFT_POLICY=0 to keep every column, which restores the pre-fix
behaviour for an A/B.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Dict

DRIFT_PRONE_TOKENS = (
    "hero_",
    "player_count",
    "seat_utilization",
    "showdown",
    "stack_",
    "button_",
)
DRIFT_PRONE_EXACT = {"hand_count", "hand_count_log", "log_hand_count"}

ENABLED = os.environ.get("POKER44_DRIFT_POLICY", "1") != "0"


def stable_features(names: Sequence[str]) -> list[str]:
    """Deterministic allowlist of columns that survive the benchmark->live shift."""
    if not ENABLED:
        return sorted(names)
    return [
        name
        for name in sorted(names)
        if name not in DRIFT_PRONE_EXACT
        and not any(token in name for token in DRIFT_PRONE_TOKENS)
    ]


def policy_report(names: Sequence[str], kept: Sequence[str]) -> Dict[str, Any]:
    kept_set = set(kept)
    dropped = sorted(name for name in names if name not in kept_set)
    return {
        "policy": "poker44-drift-v1",
        "enabled": ENABLED,
        "total": len(names),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_features": dropped,
    }
