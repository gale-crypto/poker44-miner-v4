"""Likelihood-ratio scoring from the offline-fitted 2-component mixture.

WHY THIS EXISTS. Every hand-built feature so far is an aggregate over four
decisions -- counts, shares, entropies -- which is why ~43 of 70 items share a
single raw score and our accuracy sits at chance (0.495) while the top miners
reach 0.85. The optimal scorer is the likelihood ratio P(item|bot)/P(item|human),
which is density estimation rather than feature engineering.

HOW IT WAS FITTED WITHOUT LABELS. audit_redteam_leakage rejects a window unless
Counter(phase|pressure) is identical between the pools, so every context group is
exactly half bot -- verified on every captured window. That balanced-assignment
constraint anchors an EM that would otherwise be unidentifiable.

v2 CORPUS. v1 claimed 438 "de-duplicated" items but actually counted 575 rows
that were only 328 unique sessions: r5/r9/r10 are byte-identical replays of one
70-item corpus, so that block carried 3x weight and the tables were tuned to the
very window being scored. v2 fits 296 items over 4 pairwise-disjoint windows
(<0.5 content overlap), and coarsens context from phase|position_group|pressure
to phase|pressure -- the finest split the redteam gate still guarantees is
balanced. Cells 134 -> 68, observations per cell 9.8 -> 17.4.

WHAT IT IS NOT -- READ THIS BEFORE TRUSTING IT. Leave-one-window-out says the
tables transfer (Spearman +0.756 against the tables each held-out window fits for
itself) but on *every* window they correlate more strongly with plain aggression
(+0.817 mean) than with that self-fit. Agreement with a pure aggression split is
0.740. So the part that generalises is close to "count the aggressive actions",
and the extra structure EM finds is window-specific. Since the feature arm
already encodes aggression and scores at chance, expect this to land near chance
too. It is enabled per-miner by env, never by default, and the paired
normal/inverted arms exist to measure sign and magnitude rather than assume them.

ORIENTATION. The more aggressive component is labelled bot, matching the sign of
the subnet's own reference model. If that is inverted, the leaderboard will show
accuracy consistently BELOW 0.500 -- visible, and a one-line flip.

Fitting is offline (seconds); this module only evaluates, which is a few dict
lookups per decision and keeps the serving path in single-digit milliseconds.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ARTIFACT = os.getenv(
    "POKER44_V4_MIXTURE_PATH",
    str(Path(__file__).resolve().parent / "artifacts" / "mixture_v2.json"),
)

# Which component is "bot". The fit orients component 1 as the more aggressive one
# because that is the sign the subnet's own reference model uses, but nothing in an
# unlabelled fit can confirm it. `inverted` negates the ratio so a paired arm can
# settle the question live: whichever arm reports accuracy above 0.500 is correct,
# and if both sit at 0.500 the signal is absent rather than mis-signed.
ORIENT = os.getenv("POKER44_V4_MIXTURE_ORIENT", "normal").strip().lower()
_SIGN = -1.0 if ORIENT == "inverted" else 1.0

_MODEL: Optional[Dict[str, Any]] = None
_LOADED = False


def _load() -> Optional[Dict[str, Any]]:
    """Read the fitted tables once. A missing or broken artifact is not fatal --
    the caller falls back to the feature blend rather than failing the window."""
    global _MODEL, _LOADED
    if _LOADED:
        return _MODEL
    _LOADED = True
    try:
        with open(ARTIFACT, encoding="utf-8") as handle:
            raw = json.load(handle)
        model = {
            "alpha": float(raw["alpha"]),
            "n_actions": len(raw["actions"]),
            "n_sizes": len(raw["sizes"]),
            "act": [
                {k: (dict(v), float(sum(v.values()))) for k, v in raw["action_given_context"][c].items()}
                for c in (0, 1)
            ],
            "siz": [
                {k: (dict(v), float(sum(v.values()))) for k, v in raw["size_given_action"][c].items()}
                for c in (0, 1)
            ],
            "version": raw.get("version", "?"),
            "n_items": raw.get("n_items", 0),
            # v1 keyed contexts on phase|position_group|pressure; v2 coarsens to
            # phase|pressure. The artifact carries its own spec so serving cannot
            # silently key on something the tables were never fitted for.
            "context_spec": str(raw.get("context_spec")
                                or "phase|position_group|pressure"),
        }
        _MODEL = model
    except Exception:
        _MODEL = None
    return _MODEL


def available() -> bool:
    return _load() is not None


def describe() -> str:
    model = _load()
    if model is None:
        return f"mixture unavailable ({ARTIFACT})"
    return (f"mixture {model['version']} fitted on {model['n_items']} items")


def _logp(table: Dict[str, Any], key: str, value: str, support: int, alpha: float) -> float:
    counts, total = table.get(key, (None, 0.0))
    hit = counts.get(value, 0.0) if counts else 0.0
    return math.log((hit + alpha) / (total + alpha * support))


def score_raw(decisions: List[Dict[str, Any]]) -> Optional[float]:
    """Mean per-decision log-likelihood ratio, higher == more bot-like.

    Returns None when the artifact is absent so the caller can fall back. The
    per-decision mean (rather than the sum) keeps the scale independent of how
    many decisions an item carries, in case that ever stops being exactly four.
    """
    model = _load()
    if model is None or not decisions:
        return None
    alpha = model["alpha"]
    na, ns = model["n_actions"], model["n_sizes"]
    fields = model["context_spec"].split("|")
    total = 0.0
    for d in decisions:
        context = "|".join(str(d.get(f) or "") for f in fields)
        action = str(d.get("action_type") or "")
        size = str(d.get("size_bucket") or "")
        for c in (0, 1):
            value = (_logp(model["act"][c], context, action, na, alpha)
                     + _logp(model["siz"][c], action, size, ns, alpha))
            total += value if c == 1 else -value
    return _SIGN * total / len(decisions)
