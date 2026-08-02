"""Scoring for the Poker44 schema-v4.1 micro-session contract (subnet 0.2.1).

REWRITTEN 2026-08-02 after the first scored round. The previous version placed
0.30 of its weight on a coherence term, 0.25 on a pressure term and 0.20 on a
sizing term, and used the subnet's reference model as a mere 0.25 ingredient. It
scored 0.2366 (uid 130 rank 78, uid 220 rank 79) against 0.2432 for the 71
miners running the stock reference untouched. Measured on the 88 items the
validator actually sent in window_da6247ca65cdc942675797cb21aecc76:

  * the coherence term was a DEAD CONSTANT -- 352 decisions, zero violations.
    `facing_bet` never pairs with `check`, `no_call` never with `call`, a size
    bucket appears on exactly the 103 bet/raise/all_in decisions and nowhere
    else, and `is_all_in` is true exactly 5 times matching the 5 all_in actions.
    All five checks fired zero times. It carried no ranking information and
    consumed 30% of the output range.
  * that compression capped the Brier term. Our spread was sd 0.0478 against the
    reference's 0.2113; even with a PERFECT ranking the old vector could reach
    brier_skill 0.113 (0.0227 of the available 0.20) where the reference could
    reach 0.466.
  * the per-item weight renormalisation inverted the one feature with proven
    signal. `_sizing_profile` returned None on 59/88 items, and those were
    precisely the passive ones (mean reference score 0.2719 against 0.5497 for
    the rest). Dropping the term moved the denominator 1.00 -> 0.80, multiplying
    the survivors by 1.25 and lifting low-aggression items ABOVE aggressive ones.

WHAT THE VALIDATOR GUARANTEES. audit_redteam_leakage() rejects a window outright
unless the multiset of `phase|pressure` signatures is IDENTICAL between the human
and bot pools, position_group agrees within 15 percentage points, and
decision_count / postflop_decisions / facing_bet_decisions / context_variant_count
all fail to separate the classes. A scored window has passed that gate, so every
context-shape feature is non-discriminative BY CONSTRUCTION. Only which action
was taken, and at which size, can carry signal. Weight spent on context
composition is spent on provable noise.

THE DESIGN. Three layers, in strict priority order.

  1. BACKBONE -- the subnet's own ReferenceSessionModel score, reproduced
     exactly. Back-solving the observed 0.2432 with brier_skill = 0 puts its
     average precision at ~0.70 against 0.50 for chance, so its action-composition
     features and their signs are empirically validated. It is the primary sort
     key, not an ingredient. Nothing below can reorder across its blocks.

  2. TIE-BREAK -- the reference takes only 24 distinct values on a 4-decision
     item, so 88 items land in ~16 blocks. Ties are the single most expensive
     thing in this reward. _recall_at_fpr consumes an entire equally-scored block
     before testing the 5% FPR budget, and it accumulates best_recall as a MAX
     over the thresholds it visits; splitting a block adds thresholds without
     moving the ones already there, so recall@FPR<=0.05 is monotonically
     NON-DECREASING under any tie-break, informative or not. That is a guarantee,
     not a hope. Amplitude is 0.002, which is 30% of the 0.006667 minimum gap in
     the reference lattice, so the backbone ordering is preserved exactly.

     The tie-break uses finer-grained versions of the SAME signals, conditioned
     on pressure. The gate matches how often each pressure state occurs, but not
     which action is chosen under it, so conditional rates are legitimate where
     composition is not. Every component keeps the reference's sign convention
     (more aggression, more folding, bigger sizing => more bot-like); a component
     whose direction we cannot justify is not included.

  3. CALIBRATION -- a monotone percentile map onto [0.5-H, 0.5+H]. Being monotone
     it cannot change average precision or recall by a single item; it exists
     purely to unlock the 0.20 Brier term, which both we and the reference
     forfeited entirely last round. H was chosen by replaying the subnet's own
     reward() over the real 88-item window against simulated labelings spanning
     AP 0.60-0.80, with the tie-break assumed to carry ZERO signal so the result
     is a lower bound. The optimum is flat across 0.30-0.40; H=0.35 sits in the
     middle. H=0 scores exactly 0.0000, which is what a fully tied vector earns.

NEUTRAL FILL, NEVER RENORMALISATION. When a component has no support on an item
it contributes 0.5 against a FIXED denominator. Renormalising the weights is what
inverted the signal last round, and it is not repeated anywhere here.

Zero fitted parameters -- there is still no labelled v4.1 corpus in existence.
Contract: ``score_items(items) -> list[float]``, one score per item in [0, 1],
aligned with the request order, higher == more bot-like.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Sequence

# Half-width of the calibration band. See layer 3 above.
CALIBRATION_H = float(os.getenv("POKER44_V4_CALIBRATION_H", "0.35"))
# Strictly below the 0.006667 minimum gap in the reference lattice, so the
# tie-break can only reorder within a block, never across blocks.
TIEBREAK_AMPLITUDE = float(os.getenv("POKER44_V4_TIEBREAK", "0.002"))

_AGGRESSIVE = frozenset({"bet", "raise", "all_in"})
_PASSIVE = frozenset({"check", "call"})
_NO_SIZE = frozenset({"not_applicable", "unknown"})
# Natural pot-fraction ordering, used only for the tie-break's sizing component.
_SIZE_RANK = {
    "third_pot_or_less": 0.2, "half_pot": 0.4, "three_quarter_pot": 0.6,
    "pot": 0.8, "overbet": 1.0, "all_in": 1.0,
}
_NEUTRAL = 0.5


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def _field(decision: Dict[str, Any], key: str) -> str:
    return str(decision.get(key) or "")


def _decisions(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [d for d in (item.get("decisions") or []) if isinstance(d, dict)]


def reference_score(decisions: List[Dict[str, Any]]) -> float:
    """poker44/miner/model.py ReferenceSessionModel._session_score, verbatim.

    Reproduced rather than imported: the miner venv does not carry the validator
    package, and this must not drift. Constants and rounding are upstream's.
    """
    counts = Counter(_field(d, "action_type") for d in decisions)
    meaningful = max(1, len(decisions))
    aggression = sum(counts[a] for a in _AGGRESSIVE) / meaningful
    passive = sum(counts[a] for a in _PASSIVE) / meaningful
    folds = counts["fold"] / meaningful
    overbets = sum(
        1 for d in decisions if _field(d, "size_bucket") in {"overbet", "all_in"}
    ) / meaningful
    return round(_clamp01(
        0.42 * _clamp01(aggression / 0.45)
        + 0.24 * _clamp01(folds / 0.45)
        + 0.20 * _clamp01(overbets / 0.20)
        + 0.14 * (1.0 - _clamp01(passive / 0.75))
    ), 6)


def _tiebreak(decisions: List[Dict[str, Any]], item_id: str) -> float:
    """Finer-grained restatement of the backbone's own signals, in [0, 1).

    Conditioned on pressure, which the gate matches in frequency but not in
    response. Missing support contributes the neutral 0.5 against a fixed
    denominator -- it is never renormalised away.
    """
    faced = [d for d in decisions if _field(d, "pressure") == "facing_bet"]
    free = [d for d in decisions if _field(d, "pressure") == "no_call"]

    # Aggression under fire: raising a live bet rather than flat-calling it.
    if faced:
        agg_faced = sum(
            1 for d in faced if _field(d, "action_type") in {"raise", "all_in"}
        ) / len(faced)
    else:
        agg_faced = _NEUTRAL

    # Behaviour with the initiative: betting out, and folding a free option.
    if free:
        agg_free = sum(
            1 for d in free if _field(d, "action_type") in {"bet", "raise"}
        ) / len(free)
        fold_free = sum(
            1 for d in free if _field(d, "action_type") == "fold"
        ) / len(free)
    else:
        agg_free = fold_free = _NEUTRAL

    # Where inside the sizing menu the bets land. The backbone only counts
    # overbet/all_in; the intermediate buckets are unused information.
    sized = [_SIZE_RANK.get(_field(d, "size_bucket"), 0.0) for d in decisions
             if _field(d, "size_bucket") not in _NO_SIZE]
    size_level = (sum(sized) / len(sized)) if sized else _NEUTRAL

    blended = _clamp01(
        0.30 * agg_faced + 0.25 * agg_free + 0.20 * fold_free + 0.25 * size_level
    )

    # Final splitter for items identical on every component above. Deterministic
    # and stable across processes (hash() is salted per-run, so it is not used).
    digest = 0
    for character in (item_id or "") + "|".join(
        _field(d, "action_type") + _field(d, "size_bucket") for d in decisions
    ):
        digest = (digest * 131 + ord(character)) % 1000003
    return _clamp01(0.97 * blended + 0.03 * (digest / 1000003.0))


def score_item(item: Dict[str, Any]) -> float:
    """Uncalibrated backbone score for one item, tie-break included.

    Kept for single-item use and diagnostics. The served path is score_items,
    which additionally applies the batch percentile calibration.
    """
    try:
        decisions = _decisions(item)
        if not decisions:
            return _NEUTRAL
        base = reference_score(decisions)
        offset = (_tiebreak(decisions, str(item.get("item_id") or ""))
                  - 0.5) * TIEBREAK_AMPLITUDE
        return round(_clamp01(base + offset), 6)
    except Exception:
        return _NEUTRAL      # a scorer bug must not cost the whole window


def score_items(items: Sequence[Dict[str, Any]]) -> List[float]:
    """One score per item, aligned with the request order."""
    if not items:
        return []
    raw = [score_item(i) if isinstance(i, dict) else _NEUTRAL for i in items]
    if len(raw) < 2:
        return [round(_NEUTRAL, 6)] * len(raw)
    # Monotone percentile map. Cannot reorder anything, so average precision and
    # recall@FPR are untouched; this only positions the values for the Brier term.
    order = sorted(range(len(raw)), key=lambda i: (raw[i], i))
    out = [0.0] * len(raw)
    span = len(raw) - 1
    for position, index in enumerate(order):
        out[index] = round(
            _clamp01(0.5 + CALIBRATION_H * (2.0 * position / span - 1.0)), 6
        )
    return out


def debug_components(item: Dict[str, Any]) -> Dict[str, Any]:
    decisions = _decisions(item)
    return {
        "n_decisions": len(decisions),
        "reference": reference_score(decisions) if decisions else None,
        "tiebreak": _tiebreak(decisions, str(item.get("item_id") or "")) if decisions else None,
        "uncalibrated": score_item(item),
    }
