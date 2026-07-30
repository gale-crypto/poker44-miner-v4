"""Scoring for the Poker44 v3 "strategic subject session" contract.

WHY THIS IS A SEPARATE FILE. v3 replaces the payload wholesale. There are no
hands, no streets and no bet amounts -- a session is a flat list of 12-128
decisions, each carrying only phase, position_group, pressure, action_type, an
8-way pot-relative size_bucket and an is_all_in flag. Every feature in
features.py, signature.py and luck.py reads fields that do not exist here, so
none of them can be reused; this module is a ground-up scorer for the new
contract and leaves the v2 path untouched.

WHAT SIGNAL IS LEFT. The validator's redteam gate fails any window whose
sessions do not all share an identical sorted multiset of
``phase|position_group|pressure`` (redteam_gate.py, "strategic_context_
distribution_differs"), and separately audits decision_count,
postflop_decisions, facing_bet_decisions and context_variant_count for leakage.
So context is matched by construction and carries nothing. What remains is the
POLICY: which action was chosen in each context, at which size, and how the
choices are arranged. That is what this module measures.

  * policy determinism -- a rule-driven agent returns the same action every time
    the same context recurs, so the entropy of P(action | context) collapses.
    Humans mix. This is the primary term.
  * size determinism -- given a voluntary bet, a script reuses one or two size
    buckets; the entropy of the size-bucket distribution collapses.
  * template repetition -- consecutive identical (context, action, size) tokens,
    the same concentration idea the S1-RW luck member uses, rebuilt on decisions.
  * a light strategic-profile prior, matching the subnet's own reference model
    in poker44/miner/model.py so this cannot rank far below the shipped baseline.

Terms that cannot be estimated on a given session (too few voluntary sizes, no
context seen twice) return None and fold their weight back onto the terms that
can, rather than contributing a misleading zero.

CALIBRATION, AND WHY IT IS DELIBERATELY TIMID. The v3 reward is
``0.50*AP_skill + 0.30*recall@FPR<=0.05 + 0.20*brier_skill``. brier_skill scores
the PROBABILITIES, not their ranking, against a baseline that predicts the
prevalence constant -- so an overconfident wrong score is punished in a way the
v2 reward never punished it. With no labelled v3 data yet there is nothing to
calibrate against, so output is anchored into a moderate band around 0.5. That
concedes some brier_skill in exchange for not being confidently wrong; it should
be replaced with a fitted calibration the moment labelled sessions exist.

NOTE none of the v2 post-processing (threshold remap, batch safety budget,
positive floor) is applied here. All three existed to protect
threshold_sanity_quality, which the v3 reward removes, and all three overwrite
score magnitudes -- which brier_skill now reads directly.

Contract: ``score_sessions(sessions) -> list[float]``, one score per session,
each in [0, 1], higher == more bot-like. Zero fitted parameters.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

# Weights on the four terms. Not fitted -- there is no labelled v3 data. Ordered
# by how directly each measures "same input, same output", which is the thing
# that separates a policy from a person.
W_POLICY = float(os.getenv("POKER44_V3_W_POLICY", "0.45"))
W_SIZE = float(os.getenv("POKER44_V3_W_SIZE", "0.20"))
W_REPEAT = float(os.getenv("POKER44_V3_W_REPEAT", "0.20"))
W_PROFILE = float(os.getenv("POKER44_V3_W_PROFILE", "0.15"))

# Output band. Deliberately short of [0, 1] -- see the calibration note above.
LOW_OUT = float(os.getenv("POKER44_V3_LOW_OUT", "0.12"))
HIGH_OUT = float(os.getenv("POKER44_V3_HIGH_OUT", "0.88"))
# Raw-score anchors mapped onto the output band.
LOW_ANCHOR = float(os.getenv("POKER44_V3_LOW_ANCHOR", "0.25"))
HIGH_ANCHOR = float(os.getenv("POKER44_V3_HIGH_ANCHOR", "0.80"))

_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
_AGGRESSIVE = frozenset({"bet", "raise", "all_in"})
_PASSIVE = frozenset({"check", "call"})
# Buckets that describe an actual sizing choice. "not_applicable" is what a fold
# or check carries; "unknown" means the validator could not resolve it.
_REAL_BUCKETS = frozenset({
    "third_pot_or_less", "half_pot", "three_quarter_pot", "pot", "overbet", "all_in",
})


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            out -= p * math.log(p)
    return out


def _normalised_entropy(counter: Counter, options: int) -> float:
    """Entropy scaled by the most it could be given how many draws we saw.

    Normalising by log(len(alphabet)) would penalise short contexts unfairly: a
    context seen twice cannot exhibit more than log(2) of entropy no matter how
    mixed the underlying policy is.
    """
    total = sum(counter.values())
    ceiling = min(options, total)
    if ceiling <= 1:
        return 0.0
    return _clamp01(_entropy(list(counter.values())) / math.log(ceiling))


def _context(decision: Dict[str, Any]) -> str:
    return (f"{decision.get('phase')}|{decision.get('position_group')}"
            f"|{decision.get('pressure')}")


def _token(decision: Dict[str, Any]) -> str:
    return (f"{_context(decision)}>{decision.get('action_type')}"
            f":{decision.get('size_bucket')}"
            f"{'!' if decision.get('is_all_in') else ''}")


def _policy_determinism(decisions: List[Dict[str, Any]]) -> Optional[float]:
    """1 - mean normalised entropy of P(action | context), weighted by support.

    Contexts seen once are skipped: a single observation cannot distinguish a
    deterministic rule from a coin flip, and counting it as zero entropy would
    score every short session as a bot.
    """
    by_context: Dict[str, Counter] = defaultdict(Counter)
    for decision in decisions:
        by_context[_context(decision)][str(decision.get("action_type") or "")] += 1
    weighted, support = 0.0, 0
    for counter in by_context.values():
        n = sum(counter.values())
        if n < 2:
            continue
        weighted += n * _normalised_entropy(counter, len(_ACTIONS))
        support += n
    if support < 4:
        return None
    return _clamp01(1.0 - weighted / support)


def _size_determinism(decisions: List[Dict[str, Any]]) -> Optional[float]:
    """1 - normalised entropy of the size buckets actually chosen."""
    buckets = Counter(
        str(d.get("size_bucket"))
        for d in decisions
        if str(d.get("size_bucket")) in _REAL_BUCKETS
    )
    if sum(buckets.values()) < 4:
        return None
    return _clamp01(1.0 - _normalised_entropy(buckets, len(_REAL_BUCKETS)))


def _template_repetition(decisions: List[Dict[str, Any]]) -> Optional[float]:
    """Concentration of (context, action, size) tokens -- the S1-RW idea on v3.

    Contexts are matched across sessions, so the context half of each token is
    shared by every session in the window and cannot discriminate on its own;
    what varies is how often a session reuses the SAME response in the same
    place, which is what the concentration mix reads.
    """
    n = len(decisions)
    if n < 6:
        return None
    counts = sorted(Counter(_token(d) for d in decisions).values(), reverse=True)
    top_share = counts[0] / n
    top2_share = sum(counts[:2]) / n
    unique_share = len(counts) / n
    repeat_mass = sum(c for c in counts if c >= 2) / n
    return _clamp01(0.30 * top_share + 0.15 * top2_share
                    + 0.35 * repeat_mass + 0.20 * (1.0 - unique_share))


def _strategic_profile(decisions: List[Dict[str, Any]]) -> float:
    """The subnet's own reference mix (poker44/miner/model.py, v3 branch).

    Kept as a weak term so this scorer cannot rank far below the baseline the
    subnet ships. Its constants are upstream's, not fitted here.
    """
    n = max(1, len(decisions))
    counts = Counter(str(d.get("action_type") or "") for d in decisions)
    aggression = sum(counts[a] for a in _AGGRESSIVE) / n
    passivity = sum(counts[a] for a in _PASSIVE) / n
    folds = counts["fold"] / n
    overbets = sum(
        1 for d in decisions if str(d.get("size_bucket")) in {"overbet", "all_in"}
    ) / n
    return _clamp01(
        0.42 * _clamp01(aggression / 0.45)
        + 0.24 * _clamp01(folds / 0.45)
        + 0.20 * _clamp01(overbets / 0.20)
        + 0.14 * (1.0 - _clamp01(passivity / 0.75))
    )


def _calibrate(raw: float) -> float:
    """Piecewise-linear anchors onto a deliberately moderate output band."""
    mid = 0.5 * (LOW_OUT + HIGH_OUT)
    if raw <= LOW_ANCHOR:
        return LOW_OUT + (mid - LOW_OUT) * (raw / max(LOW_ANCHOR, 1e-6))
    if raw >= HIGH_ANCHOR:
        return HIGH_OUT
    span = max(HIGH_ANCHOR - LOW_ANCHOR, 1e-6)
    return mid + (HIGH_OUT - mid) * (raw - LOW_ANCHOR) / span


def score_strategic_session(session: Dict[str, Any]) -> float:
    """Score one schema_version=3 session. Never raises."""
    try:
        decisions = [d for d in (session.get("decisions") or []) if isinstance(d, dict)]
        if not decisions:
            return 0.5
        terms = (
            (W_POLICY, _policy_determinism(decisions)),
            (W_SIZE, _size_determinism(decisions)),
            (W_REPEAT, _template_repetition(decisions)),
            (W_PROFILE, _strategic_profile(decisions)),
        )
        usable = [(w, v) for w, v in terms if v is not None]
        total = sum(w for w, _ in usable)
        if total <= 0:
            return 0.5
        raw = sum(w * v for w, v in usable) / total
        return round(_clamp01(_calibrate(raw)), 6)
    except Exception:
        return 0.5      # a scorer bug must not cost the whole window


def debug_components(session: Dict[str, Any]) -> Dict[str, Any]:
    decisions = [d for d in (session.get("decisions") or []) if isinstance(d, dict)]
    return {
        "n_decisions": len(decisions),
        "policy_determinism": _policy_determinism(decisions),
        "size_determinism": _size_determinism(decisions),
        "template_repetition": _template_repetition(decisions),
        "strategic_profile": _strategic_profile(decisions) if decisions else None,
        "score": score_strategic_session(session),
    }


def score_sessions(
    sessions: Sequence[Dict[str, Any]],
    *,
    legacy_batch_scorer=None,
) -> List[float]:
    """One score per session, dispatching on schema_version.

    schema_version 3 -> the strategic scorer above.
    schema_version 1/2 -> the session carries ``hands``, which is exactly the v2
    chunk shape, so the trained model still applies. Those are scored as ONE
    batch rather than one call per session, because the v2 serving path is
    batch-relative and per-session calls would give each session its own budget.
    """
    out: List[Optional[float]] = [None] * len(sessions)
    legacy_index: List[int] = []
    legacy_chunks: List[Any] = []

    for i, session in enumerate(sessions):
        if not isinstance(session, dict):
            out[i] = 0.5
            continue
        if str(session.get("schema_version") or "") == "3":
            out[i] = score_strategic_session(session)
            continue
        hands = session.get("hands")
        if legacy_batch_scorer is not None and isinstance(hands, list) and hands:
            legacy_index.append(i)
            legacy_chunks.append(hands)
        else:
            out[i] = 0.5

    if legacy_chunks:
        try:
            scored = legacy_batch_scorer(legacy_chunks)
        except Exception:
            scored = [0.5] * len(legacy_chunks)
        for i, value in zip(legacy_index, scored):
            try:
                out[i] = _clamp01(float(value))
            except (TypeError, ValueError):
                out[i] = 0.5

    return [0.5 if v is None else float(v) for v in out]
