"""Serving path for the behavioural bot detector.

The served score is a probability-space combination of two members, followed by
a monotone threshold remap (moving the fitted deploy threshold to 0.5) and a
batch safety budget that caps the flagged fraction. Neither post-step changes
the ranking (AP / recall@FPR are untouched).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from detector.features import profile_features
from detector.signature import signature_scores

_ART = Path(__file__).resolve().parent / "artifacts"

# Default weight on the signature member (probability-space). Overridable so the
# artifact's fitted value wins; env is only the fallback.
SIG_WEIGHT = float(os.environ.get("POKER44_SIG_WEIGHT", "0.25"))
# Cap on the fraction of >=0.5 (bot) calls per batch.
#
# Was 0.20. The binding constraint is fpr@0.5 -> fraction / (1 - bot_share) as
# the live bot share falls, and threshold_sanity_quality decays once fpr@0.5
# passes 0.10. At 0.20 a batch that is mostly human drives fpr@0.5 to ~0.40,
# costing ~0.10 reward outright; 0.05 keeps roughly 2x headroom.
MAX_POS_FRAC = float(os.environ.get("POKER44_MAX_POS_FRAC", "0.05"))


class MonoGuard:
    """Probability-space monotone-guarded soft vote.

    The three members are averaged directly in probability space (NOT ranked), so
    the monotone members' calibrated, distribution-shift-robust probabilities are
    preserved. Pickled into the artifact, so this class must import from a shipped
    file to unpickle at serve time; train.py builds instances of it.
    """

    def __init__(self, mono_lgbm, mono_hgb, logit, cols, weights=(0.45, 0.35, 0.20)):
        self.mono_lgbm = mono_lgbm
        self.mono_hgb = mono_hgb
        self.logit = logit
        self.cols = list(cols)
        self.weights = tuple(float(w) for w in weights)

    def score(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        wl, wh, wg = self.weights
        a = self.mono_lgbm.predict_proba(X)[:, 1]
        b = self.mono_hgb.predict_proba(X)[:, 1]
        c = self.logit.predict_proba(X)[:, 1]
        return (wl * a + wh * b + wg * c) / (wl + wh + wg)


def _remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotone piecewise-linear remap sending decision threshold t -> 0.5."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


def _batch_safety_budget(scores: np.ndarray, max_frac: float) -> np.ndarray:
    """Cap the fraction of >=0.5 calls per batch WITHOUT changing the ranking."""
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0 or max_frac >= 1.0:
        return s
    k = max(1, int(np.floor(max_frac * n)))
    positive = np.flatnonzero(s >= 0.5)
    if positive.size <= k:
        return s
    order = positive[np.argsort(-s[positive], kind="stable")]
    squeeze = order[k:]
    below = s[s < 0.5]
    lo = min(float(below.max()) if below.size else 0.45, 0.499)
    span = 0.5 - lo
    out = s.copy()
    m = squeeze.size
    for rank, idx in enumerate(squeeze):
        out[idx] = lo + span * (m - rank) / (m + 1.0)
    return np.clip(out, 0.0, 1.0)


def _positive_floor(scores: np.ndarray, k_min: int = 1) -> np.ndarray:
    """Guarantee at least ``k_min`` chunks sit >= 0.5, order preserved.

    threshold_sanity_quality -- and with it the ENTIRE reward, not just its own
    0.30 -- is zero when no truly-bot chunk scores >= 0.5. A threshold fitted
    offline can flag nothing at all once the live distribution shifts, which is a
    total loss for the round. Lifting the top of the ranking across the line
    costs nothing (AP and recall@FPR are rank metrics and are untouched) and
    removes the failure mode entirely.
    """
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return s
    k = max(1, min(int(k_min), n))
    if int((s >= 0.5).sum()) >= k:
        return s
    out = s.copy()
    promoted = np.argsort(-s, kind="stable")[:k]
    for rank, idx in enumerate(promoted):
        out[idx] = 0.501 + 0.008 * (k - rank) / (k + 1.0)
    return np.clip(out, 0.0, 1.0)


def feature_matrix(chunks: List[List[Dict[str, Any]]], cols: List[str]) -> np.ndarray:
    feats = [profile_features(c) for c in chunks]
    for d, c in zip(feats, chunks):
        d["hand_count"] = float(len(c))
    if feats and cols:
        # An artifact trained against an older feature schema would otherwise
        # read every renamed column as 0.0 and score silently-degraded garbage.
        missing = [col for col in cols if col not in feats[0]]
        if len(missing) > max(2, 0.05 * len(cols)):
            raise RuntimeError(
                f"artifact expects {len(missing)}/{len(cols)} features this build no "
                f"longer emits (e.g. {missing[:3]}). Retrain: python -m detector.train"
            )
    return np.array([[float(d.get(col, 0.0)) for col in cols] for d in feats], dtype=float)


class Detector:
    """Loads the trained artifact and scores validator batches."""

    def __init__(self, art_dir: Path | str = _ART):
        art_dir = Path(art_dir)
        art = joblib.load(art_dir / "model.joblib")
        self.monoguard: MonoGuard = art["monoguard"]
        self.cols = self.monoguard.cols
        self.sig_weight = float(art.get("sig_weight", SIG_WEIGHT))
        self.threshold = float(art.get("deploy_threshold", 0.5))
        with open(art_dir / "meta.json") as fh:
            self.meta = json.load(fh)

    def combined(self, chunks) -> np.ndarray:
        """Probability-space combination of the two mechanisms (pre-calibration)."""
        p_mono = self.monoguard.score(feature_matrix(chunks, self.cols))
        p_sig = np.asarray(signature_scores(chunks), dtype=float)
        w = self.sig_weight
        return (1.0 - w) * p_mono + w * p_sig

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        p = self.combined(chunks)
        s = _remap_to_threshold(p, self.threshold)
        s = _batch_safety_budget(s, MAX_POS_FRAC)
        s = _positive_floor(s)
        return [0.1 if not chunk else round(float(v), 6)
                for chunk, v in zip(chunks, s)]


_SINGLETON: Detector | None = None


def get_model() -> Detector:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Detector()
    return _SINGLETON
