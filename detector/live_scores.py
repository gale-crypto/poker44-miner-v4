"""Live competition scores, bound to the exact artifact that produced them.

WHY THIS FILE EXISTS. A miner gets no per-example feedback: the validator sends
unlabelled chunks and never says which were bots. The only signal that ever comes
back from live data is one aggregate leaderboard number per competition round.
That is roughly one float per cycle, so every one of them is worth recording
properly -- and it is worth nothing at all if you cannot say which build produced
it. Hence the artifact hash on every row.

HOW TO READ A SCORE. The reward is

    0.35*AP + 0.30*recall@fpr<=0.05 + 0.20*tsq + 0.10*tsq + 0.05*latency

so a model with NO discriminative power at all -- AP 0.5, recall 0, but a sanely
placed 0.5 line -- still scores 0.525. That is the floor, not zero. The only
number that carries information is the excess above it:

    0.5289  ->  +0.004   no measurable live signal
    0.6904  ->  +0.165   implied AP ~0.70, a working detector

A 0.0 is not a low score, it is a different failure: threshold_sanity_quality
went to zero because no truly-bot chunk reached 0.5, which zeroes the entire
reward regardless of ranking quality. That is why `zeroed` is its own field.

WHAT THIS IS NOT. These scores are context for design decisions. They are never a
per-example training label, and nothing in the training path reads them.

Record a round:

    python -m detector.live_scores --round "Competition 7 R2" --score 0.58
    python -m detector.live_scores --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "config" / "live_scores.json"
META = ROOT / "detector" / "artifacts" / "meta.json"

# reward with tsq=1, latency=1, AP=0.5 (random ranking), recall@fpr=0.
NO_SIGNAL_FLOOR = 0.35 * 0.5 + 0.20 + 0.10 + 0.05

REWARD_FORMULA = "0.35*AP + 0.30*recall@fpr<=0.05 + 0.20*tsq + 0.10*tsq + 0.05"


def load(path: Path = LEDGER) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    return records if isinstance(records, list) else []


def latest(path: Path = LEDGER) -> Optional[Dict[str, Any]]:
    records = load(path)
    return records[-1] if records else None


def implied_ap(score: float, recall: float = 0.0) -> float:
    """AP consistent with a score, given an assumed recall@fpr. Diagnostic only.

    Reward trades AP against recall (0.35*AP + 0.30*recall), so the value at
    recall=0 is the LARGEST AP consistent with the score, not an estimate of it.
    A real detector carries nonzero recall, so its true AP is lower: 0.6904 reads
    as AP<=0.973 here but is nearer 0.70 once recall@fpr~0.3 is accounted for.
    """
    return (float(score) - 0.35 - 0.30 * float(recall)) / 0.35


def _meta(path: Path = META) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record(
    competition_round: str,
    live_score: float,
    *,
    recorded_at: str,
    zeroed: bool = False,
    note: str = "",
    source: str = "user-reported Poker44 leaderboard",
    path: Path = LEDGER,
    meta_path: Path = META,
) -> Dict[str, Any]:
    """Append one round, auto-filling model identity from the served meta.json.

    Identity is read from the artifact rather than typed in, because the one
    mistake this ledger exists to prevent is attributing a score to the wrong
    build.
    """
    meta = _meta(meta_path)
    if not meta.get("artifact_sha256"):
        raise RuntimeError(
            f"{meta_path} carries no artifact_sha256, so this score cannot be "
            "bound to a build. Retrain first: python -m detector.train"
        )
    entry = {
        "competition_round": competition_round,
        "recorded_at": recorded_at,
        "live_score": float(live_score),
        "excess_over_floor": round(float(live_score) - NO_SIGNAL_FLOOR, 4),
        "zeroed": bool(zeroed),
        "variant": meta.get("variant", "unknown"),
        "model_version": meta.get("model_version", "unknown"),
        "artifact_sha256": meta["artifact_sha256"],
        "feature_schema_sha256": meta.get("feature_schema_sha256", "unknown"),
        "config": {
            "scale_norm": meta.get("scale_norm"),
            "drift_policy": (meta.get("feature_policy") or {}).get("enabled"),
            "amount_bucket_fixed": True,
            "positive_floor": True,
            "max_pos_frac": meta.get("max_pos_frac"),
            "positive_fraction": meta.get("positive_fraction"),
            "sig_weight": meta.get("sig_weight"),
        },
        "offline_reward": (
            (meta.get("offline_holdout") or meta.get("walk_forward") or {}).get("reward")
        ),
        "reward_formula_version": REWARD_FORMULA,
        "source": source,
        "diagnosis": note or (
            "Zero true positives at 0.5 -- threshold_sanity_quality gate, not a low "
            "ranking. Check the miner's positive-fraction logs for that round."
            if zeroed else
            f"Excess over the {NO_SIGNAL_FLOOR:.3f} no-signal floor is "
            f"{float(live_score) - NO_SIGNAL_FLOOR:+.4f}. Context only - never a "
            "per-example training label."
        ),
    }
    records = load(path)
    records.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return entry


def summary(path: Path = LEDGER) -> str:
    records = load(path)
    if not records:
        return "no rounds recorded"
    lines = [
        f"no-signal floor = {NO_SIGNAL_FLOOR:.3f}  |  APmax = largest AP consistent "
        f"with the score (at recall=0); true AP is lower",
        "",
        f"{'round':<24} {'score':>7} {'excess':>8} {'APmax':>6} {'version':<18} artifact",
        "-" * 96,
    ]
    for r in records:
        score = float(r.get("live_score", 0.0))
        ap = "n/a" if r.get("zeroed") else f"{implied_ap(score):.3f}"
        lines.append(
            f"{str(r.get('competition_round'))[:24]:<24} {score:>7.4f} "
            f"{score - NO_SIGNAL_FLOOR:>+8.4f} {ap:>6} "
            f"{str(r.get('model_version'))[:18]:<18} {str(r.get('artifact_sha256'))[:12]}"
        )
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--round", dest="competition_round")
    parser.add_argument("--score", type=float)
    parser.add_argument("--date", dest="recorded_at",
                        help="YYYY-MM-DD the score was observed")
    parser.add_argument("--zeroed", action="store_true",
                        help="reward was 0.0 (tsq gate), not merely low")
    parser.add_argument("--note", default="")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.show or args.score is None:
        print(summary())
        return 0
    if not args.competition_round or not args.recorded_at:
        parser.error("--round and --date are required when recording a score")
    entry = record(
        args.competition_round, args.score,
        recorded_at=args.recorded_at, zeroed=args.zeroed, note=args.note,
    )
    print(json.dumps(entry, indent=2))
    print()
    print(summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
