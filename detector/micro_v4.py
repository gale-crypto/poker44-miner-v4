"""Scoring for the Poker44 schema-v4.1 micro-session contract (subnet 0.2.1).

WHY THIS REPLACES session_v3.py. The shipped v3.0 contract is not the one the
dev branch advertised. An item is exactly FOUR strategic decisions, not 12-128,
and arrives as `items` on a MicroSessionDetectionSynapse rather than `sessions`.
Every term in session_v3.py needs more decisions than now exist -- template
repetition wanted >=6, size determinism >=4 sized bets, policy determinism >=4
support across repeated contexts -- so on a 4-decision item they all fold to
None and the weight collapses onto the reference prior. This module is built for
the size of item that actually arrives.

WHAT SIGNAL IS AVAILABLE. Four decisions, seven categorical fields each, at
least one postflop. No amounts, no cards, no timing, no identity. Context
(phase/position_group/pressure) is matched across items by the validator's
redteam gate, so context frequency carries nothing; what is left is WHICH action
was chosen in a given context, at which size bucket.

  * coherence -- combinations a real table would not produce: checking while
    facing a bet, calling with nothing to call, a sized bucket on a fold, an
    is_all_in flag disagreeing with the action or bucket. A generator that
    assembles decisions field-by-field leaks here; a real capture cannot. Costs
    nothing when both sides are clean.
  * pressure response -- how the subject splits fold/call/raise when facing a
    bet, and how it continues when not. A fixed policy concentrates; a person
    spreads.
  * sizing profile -- which buckets appear at all, and whether they cluster on
    the one or two a script tends to reuse.
  * reference prior -- the subnet's own ReferenceSessionModel mix, so we cannot
    rank below the baseline every miner gets for free.

GRANULARITY IS A DESIGN CONSTRAINT HERE, NOT A DETAIL. With four categorical
decisions the natural score space is tiny -- plain action fractions can only take
the values 0, .25, .5, .75, 1 -- and the v3 reward punishes ties twice over:
average_precision_score degrades on them, and _recall_at_fpr consumes an entire
equally-scored block before testing the 5% FPR budget, so one tied block mixing
bots and humans can spend the whole budget at once. The dashboard's eight-way tie
at 0.0770 is what that looks like in production. Every term here therefore
carries a fine-grained ordinal component chosen to break ties that the coarse
fractions alone would create.

CALIBRATION. The reward is 0.50*AP_skill + 0.30*recall@FPR<=0.05 +
0.20*brier_skill, and brier_skill reads the probabilities directly against a
constant-prevalence baseline. With no labelled v4.1 corpus in existence there is
nothing to calibrate against, so output is anchored into a moderate band: that
forfeits at most the 0.20 brier term (it is floored at zero) and cannot go
negative, whereas confident-and-wrong forfeits it anyway. Replace this the moment
a labelled corpus ships.

Zero fitted parameters. Contract: ``score_items(items) -> list[float]``, one
score per item in [0, 1], higher == more bot-like.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

W_COHERENCE = float(os.getenv("POKER44_V4_W_COHERENCE", "0.30"))
W_PRESSURE = float(os.getenv("POKER44_V4_W_PRESSURE", "0.25"))
W_SIZING = float(os.getenv("POKER44_V4_W_SIZING", "0.20"))
W_REFERENCE = float(os.getenv("POKER44_V4_W_REFERENCE", "0.25"))

LOW_OUT = float(os.getenv("POKER44_V4_LOW_OUT", "0.15"))
HIGH_OUT = float(os.getenv("POKER44_V4_HIGH_OUT", "0.85"))

_AGGRESSIVE = frozenset({"bet", "raise", "all_in"})
_PASSIVE = frozenset({"check", "call"})
_NO_SIZE = frozenset({"not_applicable", "unknown"})
# Ordinal ladder over size buckets. Used only to give the sizing term a
# continuous component; the ordering is the natural pot-fraction ordering.
_SIZE_RANK = {
    "not_applicable": 0.0, "unknown": 0.0, "third_pot_or_less": 0.2,
    "half_pot": 0.4, "three_quarter_pot": 0.6, "pot": 0.8,
    "overbet": 1.0, "all_in": 1.0,
}


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def _field(decision: Dict[str, Any], key: str) -> str:
    return str(decision.get(key) or "")


def _coherence_violations(decisions: List[Dict[str, Any]]) -> float:
    """Fraction of decisions containing an internally impossible combination.

    These are not style judgements -- each is a state a real table cannot
    produce. A pipeline that samples fields independently will emit them; a
    faithful capture will not.
    """
    bad = 0.0
    for decision in decisions:
        action = _field(decision, "action_type")
        bucket = _field(decision, "size_bucket")
        pressure = _field(decision, "pressure")
        all_in = bool(decision.get("is_all_in"))
        hits = 0
        if pressure == "facing_bet" and action == "check":
            hits += 1                      # cannot check into a live bet
        if pressure == "no_call" and action == "call":
            hits += 1                      # nothing to call
        if action in {"fold", "check"} and bucket not in _NO_SIZE:
            hits += 1                      # a fold/check has no sizing
        if all_in and action != "all_in" and bucket != "all_in":
            hits += 1                      # flag disagrees with both fields
        if action == "all_in" and bucket not in {"all_in", "unknown"}:
            hits += 1
        bad += min(1.0, hits / 2.0)        # two independent flaws saturate
    return _clamp01(bad / max(1, len(decisions)))


def _pressure_response(decisions: List[Dict[str, Any]]) -> Optional[float]:
    """Concentration of the response taken under each pressure state.

    A rule answers `facing_bet` the same way every time; a person mixes. With at
    most four decisions this is coarse, so a size-rank spread is folded in to
    keep the output from collapsing onto a handful of values.
    """
    faced = [d for d in decisions if _field(d, "pressure") == "facing_bet"]
    free = [d for d in decisions if _field(d, "pressure") == "no_call"]
    parts: List[float] = []
    for group in (faced, free):
        if len(group) < 2:
            continue
        actions = Counter(_field(d, "action_type") for d in group)
        parts.append(max(actions.values()) / len(group))
    if not parts:
        return None
    concentration = sum(parts) / len(parts)
    ranks = [_SIZE_RANK.get(_field(d, "size_bucket"), 0.0) for d in decisions]
    spread = (max(ranks) - min(ranks)) if ranks else 0.0
    # Concentrated actions AND flat sizing is the scripted signature; the spread
    # term also supplies the continuous component that breaks ties.
    return _clamp01(0.80 * concentration + 0.20 * (1.0 - spread))


def _sizing_profile(decisions: List[Dict[str, Any]]) -> Optional[float]:
    """How narrow the sizing menu is across the decisions that carry one."""
    sized = [_field(d, "size_bucket") for d in decisions
             if _field(d, "size_bucket") not in _NO_SIZE]
    if len(sized) < 2:
        return None
    counts = Counter(sized)
    top_share = max(counts.values()) / len(sized)
    distinct_share = len(counts) / len(sized)
    mean_rank = sum(_SIZE_RANK.get(s, 0.0) for s in sized) / len(sized)
    return _clamp01(0.55 * top_share + 0.30 * (1.0 - distinct_share)
                    + 0.15 * mean_rank)


def _reference_prior(decisions: List[Dict[str, Any]]) -> float:
    """poker44/miner/model.py ReferenceSessionModel, constants upstream's."""
    n = max(1, len(decisions))
    counts = Counter(_field(d, "action_type") for d in decisions)
    aggression = sum(counts[a] for a in _AGGRESSIVE) / n
    passive = sum(counts[a] for a in _PASSIVE) / n
    folds = counts["fold"] / n
    overbets = sum(1 for d in decisions
                   if _field(d, "size_bucket") in {"overbet", "all_in"}) / n
    return _clamp01(
        0.42 * _clamp01(aggression / 0.45)
        + 0.24 * _clamp01(folds / 0.45)
        + 0.20 * _clamp01(overbets / 0.20)
        + 0.14 * (1.0 - _clamp01(passive / 0.75))
    )


def _tiebreak(decisions: List[Dict[str, Any]]) -> float:
    """Tiny deterministic offset in [0, 1) from the exact decision tuple.

    Two items with identical coarse statistics but different decision ORDER are
    different observations, and leaving them exactly equal wastes both AP and the
    FPR budget (see the module docstring). This is not signal and is scaled to
    stay far below any real difference; it only splits what would otherwise be a
    hard tie.
    """
    total = 0.0
    for i, d in enumerate(decisions):
        total += (i + 1) * (
            _SIZE_RANK.get(_field(d, "size_bucket"), 0.0)
            + 0.37 * len(_field(d, "action_type"))
            + 0.11 * len(_field(d, "phase"))
        )
    return (total % 1.0)


def score_item(item: Dict[str, Any]) -> float:
    """Score one schema-v4.1 micro-session. Never raises."""
    try:
        decisions = [d for d in (item.get("decisions") or []) if isinstance(d, dict)]
        if not decisions:
            return 0.5
        terms = (
            (W_COHERENCE, _coherence_violations(decisions)),
            (W_PRESSURE, _pressure_response(decisions)),
            (W_SIZING, _sizing_profile(decisions)),
            (W_REFERENCE, _reference_prior(decisions)),
        )
        usable = [(w, v) for w, v in terms if v is not None]
        total = sum(w for w, _ in usable)
        if total <= 0:
            return 0.5
        raw = sum(w * v for w, v in usable) / total
        span = HIGH_OUT - LOW_OUT
        out = LOW_OUT + span * raw
        # Sub-milli offset: splits exact ties without reordering real differences.
        out += (_tiebreak(decisions) - 0.5) * 0.0008
        return round(_clamp01(out), 6)
    except Exception:
        return 0.5      # a scorer bug must not cost the whole window


def score_items(items: Sequence[Dict[str, Any]]) -> List[float]:
    """One score per item, aligned with the request order."""
    return [score_item(i) if isinstance(i, dict) else 0.5 for i in (items or [])]


def debug_components(item: Dict[str, Any]) -> Dict[str, Any]:
    decisions = [d for d in (item.get("decisions") or []) if isinstance(d, dict)]
    return {
        "n_decisions": len(decisions),
        "coherence_violations": _coherence_violations(decisions) if decisions else None,
        "pressure_response": _pressure_response(decisions) if decisions else None,
        "sizing_profile": _sizing_profile(decisions) if decisions else None,
        "reference_prior": _reference_prior(decisions) if decisions else None,
        "score": score_item(item),
    }
