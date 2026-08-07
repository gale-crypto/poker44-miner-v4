"""Scoring for the Poker44 schema-v4.1 micro-session contract.

HISTORY, measured on live windows -- read this before changing weights.

  round 2  ours 0.2366  vs the 71-miner stock tie 0.2432   -0.0066  rank 78
  round 3  ours 0.1300  vs the 77-miner stock tie 0.0830   +0.0470  rank 35
  round 4  all four of our arms at brier_skill 0.000, accuracy 0.45-0.52

WHAT DID NOT SURVIVE CONTACT. Round 3 ran a controlled A/B -- identical code,
weights the only difference -- and `conc 0.40` beat `conc 0.15` by 0.1964 to
0.1100, which looked like proof that policy CONCENTRATION beats MARGINAL action
composition. Round 4 reversed it exactly: the same conc 0.40 config fell to
0.0189 (rank 177) while the marginal-heavy arm came in above it at 0.0674, and
an arm pushed to conc 0.55 scored 0.0032 with AP_skill 0.000. With ~70 items and
discriminators sitting near chance, one window cannot separate these configs.
Weights are therefore parked in the middle and no further weight A/B is worth
running until some discriminator clears the noise floor.

The subnet owner has since stated the opposing agents are "not bots based on
static decisions: they are agents with dynamic logic and multiple behavioral
profiles". That is the likely reason every repetition-based feature has failed:
concentration and conditional determinism were built to catch a scripted seat,
and there is no scripted seat to catch.

WHAT THE VALIDATOR GUARANTEES, AND WHAT IT COSTS. audit_redteam_leakage()
rejects a window unless `Counter(phase|pressure signature)` is IDENTICAL between
the human and bot pools, position_group agrees within 15pp, and decision_count /
postflop / facing_bet / context-variety all fail to separate the classes. So
every context-shape feature is non-discriminative BY CONSTRUCTION, and -- because
Counter equality forces each signature to appear equally often in each class --
EVERY context group is exactly half bot. Verified across all captured windows:
every group even-sized, which chance would not produce.

THE THREE LAYERS.

  1. RAW SIGNAL -- marginal action composition (the subnet's own reference model,
     the only feature family with any demonstrated skill), plus tuple
     concentration, conditional action entropy and size rigidity. Every term
     contributes against a FIXED denominator with an absent term falling back to
     the neutral 0.5. Renormalising onto the terms that happen to be defined is
     what inverted the round-2 signal, and the reference _v41 scorer still has
     that defect.

  2. TIE-BREAK -- content-derived, never item_id (see _tiebreak). Ties are the
     most expensive thing in this reward: a fully tied vector scores exactly
     0.0000. Because `_recall_at_fpr` accumulates best_recall as a MAX over the
     thresholds it visits, splitting a tied block only ADDS thresholds, so
     recall@FPR<=0.05 is monotonically non-decreasing under any tie-break. That
     is a guarantee. Note it is doing heavy lifting: ~43 of 70 items typically
     share a raw score, so the discriminator's resolution, not the tie-break, is
     the real limitation.

  3. CALIBRATION -- default `group`: rank within the context group, then spread
     smoothly across the whole window. Between-group differences are provably
     label-irrelevant (layer 0 above) so ranking within the group deletes noise,
     and spreading keeps values distinct. It also pins the median to 0.5, which
     matters because `accuracy` is scored as `(score >= 0.5) == truth` at
     prevalence 0.5 -- the previous `anchor` default put 74.3% of scores above
     0.5 and thereby forced accuracy to chance and brier_skill to 0. Against
     labels respecting the group constraint this beats a plain global percentile
     at every signal level tested and is neutral at chance. `percentile` and
     `anchor` remain available via POKER44_V4_CALIBRATION.

Zero fitted parameters: no labelled v4.1 corpus exists yet (every benchmark and
training route returns ROUTE_NOT_FOUND). The owner has said ground truth for
used evaluation datasets will be published, at which point this file should be
refitted rather than reasoned about.

Contract: ``score_items(items) -> list[float]``, one score per item in [0, 1],
aligned with the request order, higher == more bot-like.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict  # noqa: F401  (defaultdict: group mode)
from typing import Any, Dict, List, Optional, Sequence

# Round 3 made conc 0.40 look decisively better than conc 0.15 (0.1964 vs
# 0.1100). Round 4 REVERSED it -- the same conc 0.40 config fell to 0.0189 while
# the marginal-heavy arm came in above it at 0.0674, and an arm pushed to
# conc 0.55 scored 0.0032 with AP_skill 0.000. With ~70 items and discriminators
# near chance, one window cannot separate these; the two rounds average to a
# tie. Defaults therefore sit in the middle rather than at either round's
# apparent winner, and no further weight A/B is worth running until there is a
# discriminator that clears noise.
W_MARGINAL = float(os.getenv("POKER44_V4_W_MARGINAL", "0.55"))
W_CONCENTRATION = float(os.getenv("POKER44_V4_W_CONCENTRATION", "0.30"))
W_DETERMINISM = float(os.getenv("POKER44_V4_W_DETERMINISM", "0.10"))
W_RIGIDITY = float(os.getenv("POKER44_V4_W_RIGIDITY", "0.05"))

CALIBRATION = os.getenv("POKER44_V4_CALIBRATION", "group")  # group | percentile | anchor
PERCENTILE_H = float(os.getenv("POKER44_V4_PERCENTILE_H", "0.35"))
LOW_ANCHOR = float(os.getenv("POKER44_V4_LOW_ANCHOR", "0.30"))
HIGH_ANCHOR = float(os.getenv("POKER44_V4_HIGH_ANCHOR", "0.85"))
FLOOR = float(os.getenv("POKER44_V4_FLOOR", "0.05"))
CEILING = float(os.getenv("POKER44_V4_CEILING", "0.98"))

# Kept well below the smallest gap the blended raw signal produces, so the
# tie-break only ever reorders within a block.
TIEBREAK_AMPLITUDE = float(os.getenv("POKER44_V4_TIEBREAK", "0.0015"))

_VOLUNTARY = frozenset({"bet", "raise", "all_in"})
_REAL_SIZE = frozenset({"third_pot_or_less", "half_pot", "three_quarter_pot",
                        "pot", "overbet", "all_in"})
_NEUTRAL = 0.5


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def _field(decision: Dict[str, Any], key: str) -> str:
    return str(decision.get(key) or "")


def _decisions(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [d for d in (item.get("decisions") or []) if isinstance(d, dict)]


def _norm_entropy(counts: List[int]) -> float:
    """Shannon entropy of a count vector, normalised to [0, 1] by its support."""
    total = sum(counts)
    k = len(counts)
    if total <= 0 or k <= 1:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return _clamp01(h / math.log(k))


def _concentration(decisions: List[Dict[str, Any]]) -> float:
    """Repetition of whole decision tuples. A script reuses them; a person does not."""
    n = len(decisions)
    tuples = Counter(
        (_field(d, "phase"), _field(d, "position_group"), _field(d, "pressure"),
         _field(d, "action_type"), _field(d, "size_bucket"), bool(d.get("is_all_in")))
        for d in decisions
    )
    ordered = sorted(tuples.values(), reverse=True)
    top_share = ordered[0] / n
    top2_share = sum(ordered[:2]) / n
    unique_share = len(tuples) / n
    repeat_mass = sum(c for c in ordered if c >= 2) / n
    return _clamp01(0.30 * top_share + 0.15 * top2_share
                    + 0.35 * repeat_mass + 0.20 * (1.0 - unique_share))


def _determinism(decisions: List[Dict[str, Any]]) -> Optional[float]:
    """1 - action entropy within each matched (phase, position, pressure) context.

    The gate matches how often each context OCCURS but not how the subject
    answers it, so this reads signal where context composition cannot.
    """
    contexts: Dict[tuple, Counter] = defaultdict(Counter)
    for d in decisions:
        contexts[(_field(d, "phase"), _field(d, "position_group"),
                  _field(d, "pressure"))][_field(d, "action_type")] += 1
    weighted, total = 0.0, 0
    for counter in contexts.values():
        m = sum(counter.values())
        if m >= 2:
            weighted += m * _norm_entropy(list(counter.values()))
            total += m
    return (1.0 - weighted / total) if total else None


def _rigidity(decisions: List[Dict[str, Any]]) -> Optional[float]:
    """1 - entropy of the size buckets actually chosen on voluntary bets."""
    sizes = [_field(d, "size_bucket") for d in decisions
             if _field(d, "action_type") in _VOLUNTARY
             and _field(d, "size_bucket") in _REAL_SIZE]
    if len(sizes) < 3:
        return None
    return 1.0 - _norm_entropy(list(Counter(sizes).values()))


def _marginal(decisions: List[Dict[str, Any]]) -> float:
    """The subnet's own ReferenceSessionModel mix. Constants are upstream's.

    Round 2 established this carries real signal (AP ~0.70 against 0.50 for
    chance); round 3 established it should not carry the whole load.
    """
    n = max(1, len(decisions))
    acts = Counter(_field(d, "action_type") for d in decisions)
    aggression = _clamp01((acts["bet"] + acts["raise"] + acts["all_in"]) / n / 0.45)
    folds = _clamp01(acts["fold"] / n / 0.45)
    passive = _clamp01((acts["check"] + acts["call"]) / n / 0.75)
    overbets = _clamp01(sum(
        1 for d in decisions if _field(d, "size_bucket") in {"overbet", "all_in"}
    ) / n / 0.20)
    return _clamp01(0.42 * aggression + 0.24 * folds
                    + 0.20 * overbets + 0.14 * (1.0 - passive))


def raw_score(decisions: List[Dict[str, Any]]) -> float:
    """Blended policy signal in [0, 1]. FIXED denominator, neutral fill."""
    determinism = _determinism(decisions)
    rigidity = _rigidity(decisions)
    total = W_MARGINAL + W_CONCENTRATION + W_DETERMINISM + W_RIGIDITY
    if total <= 0:
        return _NEUTRAL
    blended = (
        W_MARGINAL * _marginal(decisions)
        + W_CONCENTRATION * _concentration(decisions)
        + W_DETERMINISM * (determinism if determinism is not None else _NEUTRAL)
        + W_RIGIDITY * (rigidity if rigidity is not None else _NEUTRAL)
    ) / total
    return _clamp01(blended)


def _tiebreak(decisions: List[Dict[str, Any]]) -> float:
    """Deterministic splitter in [0, 1) for items identical on every term above.

    CONTENT ONLY -- item_id is deliberately excluded. On 2026-08-07 the validator
    sent the same 70 sessions four times (`..._r5` through `..._r8`) with 70/70
    identical decision content but ZERO item_id overlap: ids are regenerated per
    window and carry no information. Feeding them in made us score identical
    sessions differently across those replays -- only 50-58 of 66 matched
    sessions scored the same, drifting up to 0.28 -- because the tie-break
    decides within-group rank and rank drives the final spread.

    That drift was pure noise, and it also blocked learning: with ground truth
    for used datasets due to be published, identical input must give identical
    output or a replayed window cannot be compared against its own label.

    Not signal either way; the ordering it imposes on raw-score ties is arbitrary
    whichever key is used. The point is that it is now reproducible. Stable
    across processes -- hash() is salted per run, so a rolling polynomial is used.
    """
    digest = 0
    material = "|".join(
        _field(d, "action_type") + _field(d, "size_bucket") + _field(d, "phase")
        + _field(d, "position_group") + _field(d, "pressure")
        for d in decisions
    )
    for character in material:
        digest = (digest * 131 + ord(character)) % 1000003
    return digest / 1000003.0


def _anchor_calibrate(raw: float) -> float:
    """Piecewise-linear map that preserves the raw signal's shape.

    Never emits maximal confidence: Brier skill punishes 1.0-when-wrong far more
    than the ranking terms gain from it.
    """
    if raw <= LOW_ANCHOR:
        out = FLOOR + (0.5 - FLOOR) * (raw / max(LOW_ANCHOR, 1e-6))
    elif raw >= HIGH_ANCHOR:
        out = CEILING
    else:
        out = 0.5 + (CEILING - 0.5) * (raw - LOW_ANCHOR) / max(HIGH_ANCHOR - LOW_ANCHOR, 1e-6)
    return min(CEILING, _clamp01(out))


def score_item(item: Dict[str, Any]) -> float:
    """Score one schema-v4.1 micro-session. Never raises."""
    try:
        decisions = _decisions(item)
        if not decisions:
            return _NEUTRAL
        raw = raw_score(decisions)
        out = _anchor_calibrate(raw)
        out += (_tiebreak(decisions) - 0.5) * TIEBREAK_AMPLITUDE
        return round(_clamp01(out), 6)
    except Exception:
        return _NEUTRAL      # a scorer bug must not cost the whole window


def context_signature(item: Dict[str, Any]) -> tuple:
    """The exact key audit_redteam_leakage matches between the two pools.

    See poker44/validator/evaluation/redteam_gate.py::_context_signature.
    """
    return tuple(sorted(
        f"{_field(d, 'phase')}|{_field(d, 'pressure')}" for d in _decisions(item)))


def _spread(order: List[int], n: int) -> List[float]:
    """Lay an ordering out evenly across [0.5-H, 0.5+H]. Mean and median land on
    0.5, which is what `accuracy` (`score >= 0.5`) and the Brier baseline expect
    at the fixed 50% prevalence."""
    out = [0.0] * n
    span = max(1, n - 1)
    for position, index in enumerate(order):
        out[index] = round(
            _clamp01(0.5 + PERCENTILE_H * (2.0 * position / span - 1.0)), 6)
    return out


def score_items(items: Sequence[Dict[str, Any]]) -> List[float]:
    """One score per item, aligned with the request order.

    THE STRUCTURAL POINT. audit_redteam_leakage rejects a window unless
    `Counter(phase|pressure signature)` is IDENTICAL between the human and bot
    pools. Counter equality means each signature occurs the same number of times
    in each class, so EVERY context group is exactly half bot -- and a group of
    size two is one bot and one human with identical context. Verified on all
    three captured windows: 62 groups, every one even-sized (chance would be
    2^-62). So between-group score differences carry ZERO label information,
    while within-group order carries all of it.

    Ranking within the group therefore deletes provably irrelevant variance, and
    spreading the result keeps every value distinct so neither AP nor the 5% FPR
    budget is spent on tied blocks. Against labels that respect the group
    constraint this beats a plain global percentile at every signal level tested
    (+29% to +46% for within-group accuracy 0.55-0.80) and is neutral at chance.

    A fully confident group-binary output -- which is what uid 88's signature
    implies (accuracy 0.846, brier_skill 0.466, recall exactly 0.000) -- was also
    tested and is WORSE than this below ~0.62 accuracy, because the tied bands
    forfeit the whole 30% recall term.
    """
    if not items:
        return []
    scored = [score_item(i) if isinstance(i, dict) else _NEUTRAL for i in items]
    n = len(scored)
    if n < 2 or CALIBRATION == "anchor":
        return scored
    if CALIBRATION == "percentile":
        return _spread(sorted(range(n), key=lambda i: (scored[i], i)), n)

    # group mode: percentile within the context group, then spread globally.
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for index, item in enumerate(items):
        key = context_signature(item) if isinstance(item, dict) else ()
        groups[key].append(index)
    position: List[float] = [0.5] * n
    for member_indices in groups.values():
        ordered = sorted(member_indices, key=lambda i: (scored[i], i))
        size = len(ordered)
        for rank, index in enumerate(ordered):
            position[index] = (rank + 0.5) / size
    # Residual ties broken by the raw score, so items never collide.
    return _spread(sorted(range(n), key=lambda i: (position[i], scored[i], i)), n)


def debug_components(item: Dict[str, Any]) -> Dict[str, Any]:
    decisions = _decisions(item)
    if not decisions:
        return {"n_decisions": 0, "score": _NEUTRAL}
    return {
        "n_decisions": len(decisions),
        "marginal": _marginal(decisions),
        "concentration": _concentration(decisions),
        "determinism": _determinism(decisions),
        "rigidity": _rigidity(decisions),
        "raw": raw_score(decisions),
        "score": score_item(item),
    }
