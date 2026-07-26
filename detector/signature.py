"""Action-sequence signature member — a zero-parameter per-chunk score in [0, 1].

Computed over the sanitised (miner-visible) hand only (action_type, street,
coarse bb bucket) — never seat/hero/button identity, which the validator
re-aliases. Contract: ``signature_scores(chunks) -> list[float]`` in [0, 1],
higher == more bot-like.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

_ACT = {
    "fold": "F", "check": "K", "call": "C",
    "bet": "B", "raise": "R", "allin": "A", "all_in": "A",
}
_STREET = {"preflop": "p", "flop": "f", "turn": "t", "river": "r"}


def _num(v: Any, d: float = 0.0) -> float:
    try:
        f = float(v)
        return d if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return d


def _size_bin(bb: float) -> str:
    # Coarse sizing ladder; deliberately different granularity from the validator
    # grid so identical raw lines still collapse but near-misses do not.
    if bb <= 0.0:
        return "0"
    if bb <= 1.5:
        return "a"
    if bb <= 3.5:
        return "b"
    if bb <= 7.0:
        return "c"
    if bb <= 15.0:
        return "d"
    return "e"


def _hand_signature(hand: Dict[str, Any]) -> str:
    """Full ordered line: street+action+size token per action, plus street shape."""
    tokens: List[str] = []
    for a in hand.get("actions") or []:
        if not isinstance(a, dict):
            continue
        st = _STREET.get(str(a.get("street", "")).lower(), "?")
        ac = _ACT.get(str(a.get("action_type", "")).lower(), "?")
        sz = _size_bin(_num(a.get("normalized_amount_bb"), _num(a.get("amount"))))
        tokens.append(st + ac + sz)
    shape = "".join(
        _STREET.get(str(s.get("street", "")).lower(), "?")
        for s in (hand.get("streets") or [])
        if isinstance(s, dict)
    )
    return shape + "#" + ".".join(tokens)


def _action_bigrams(hand: Dict[str, Any]) -> List[str]:
    seq = [
        _ACT.get(str(a.get("action_type", "")).lower(), "?")
        for a in (hand.get("actions") or [])
        if isinstance(a, dict)
    ]
    return [seq[i] + seq[i + 1] for i in range(len(seq) - 1)]


def _norm_entropy(counts: List[int]) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    if len(ps) <= 1:
        return 0.0
    return -sum(p * math.log(p) for p in ps) / math.log(len(ps))


def signature_score(chunk: List[Dict[str, Any]]) -> float:
    """One concentration score in [0, 1] for a chunk (higher == more replayed)."""
    hands = [h for h in (chunk or []) if isinstance(h, dict)]
    n = len(hands)
    if n == 0:
        return 0.5

    sigs = [_hand_signature(h) for h in hands]
    sc = Counter(sigs)
    top_share = max(sc.values()) / n                       # biggest replayed template
    unique_share = len(sc) / n                             # diversity (human -> high)
    repeat_mass = sum(c for c in sc.values() if c >= 2) / n  # mass inside templates

    # Cross-hand action-bigram entropy: humans mix transitions, scripts don't.
    bg = Counter()
    for h in hands:
        bg.update(_action_bigrams(h))
    bigram_uni = _norm_entropy(list(bg.values()))          # high == human-like

    # Street-shape concentration (how uniformly hands reach the same streets).
    shapes = Counter(s.split("#", 1)[0] for s in sigs)
    shape_top = max(shapes.values()) / n

    concentration = (
        0.40 * top_share
        + 0.30 * repeat_mass
        + 0.15 * (1.0 - unique_share)
        + 0.15 * shape_top
    )
    # Blend in the inverse of bigram diversity (bot -> low entropy -> high risk).
    raw = 0.80 * concentration + 0.20 * (1.0 - bigram_uni)
    return 0.0 if raw < 0.0 else 1.0 if raw > 1.0 else raw


def signature_scores(chunks: List[List[Dict[str, Any]]]) -> List[float]:
    return [signature_score(list(c or [])) for c in (chunks or [])]
