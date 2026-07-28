"""Poker44 miner entrypoint (subnet 126).

Serves one bot-risk score per chunk from the trained detector.

Run:
    python neurons/miner.py --netuid 126 \
        --wallet.name <cold> --wallet.hotkey <hot> \
        --subtensor.network finney --axon.port 8091
"""

# NOTE: do NOT `from __future__ import annotations` here. bittensor's axon.attach
# introspects the real type of forward()'s `synapse` parameter via issubclass();
# stringised (PEP 563) annotations break that with "issubclass() arg 1 must be a
# class". The reference miner omits the future-import for the same reason.

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (build_local_model_manifest,
                                          evaluate_manifest_compliance,
                                          manifest_digest)
from poker44.validator.synapse import DetectionSynapse

from detector import live_capture
from detector.inference import get_model

MODEL_NAME = os.environ.get("POKER44_MODEL_NAME", "poker44-miner-v4")
MODEL_VERSION = os.environ.get("POKER44_MODEL_VERSION", "3.0.0")
ARTIFACT = ROOT / "detector" / "artifacts" / "model.joblib"


def _artifact_sha256(artifact_path: Path) -> str:
    """Fingerprint of the served weights.

    The weights are distributed out-of-band, so this is what lets a reader tell
    whether the artifact behind a given score is the one they were told about.
    """
    env_hash = os.environ.get("POKER44_MODEL_ARTIFACT_SHA256", "").strip()
    if env_hash:
        return env_hash
    if not artifact_path.exists():
        return ""
    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _repo_commit(repo_root: Path) -> str:
    """The commit being served. Falls back to git when the env var is unset —
    an empty repo_commit is a manifest policy violation, not a blank field.
    """
    env_commit = os.environ.get("POKER44_MODEL_REPO_COMMIT", "").strip()
    if env_commit:
        return env_commit
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


# NOTE: weights are shipped out of band and are NOT in the repo, so there is no
# raw-at-commit URL to publish. artifact_url therefore defaults to "" (the same
# default every scoring miner on the subnet uses) and is opt-in via
# POKER44_MODEL_ARTIFACT_URL once the exact bytes are uploaded as a release
# asset. Advertising a URL that 404s, or that serves different bytes than
# artifact_sha256 describes, is a false transparency claim -- worse than
# publishing nothing. artifact_sha256 is still emitted: it identifies the running
# weights even when they are not downloadable.


# Fraction of a degraded batch placed above 0.5. Matches the served
# MAX_POS_FRAC / POSITIVE_FRACTION so a degraded round behaves like a normal one.
_FALLBACK_POSITIVE_FRACTION = 0.05


def _fallback_scores(chunks) -> list:
    """Rank-preserving scores for when the model path fails.

    A CONSTANT fallback loses the round outright, in both directions: every chunk
    at exactly 0.5 drives fpr@0.5 to 1.0, which zeroes threshold_sanity_quality
    and with it the ENTIRE reward; any constant below 0.5 flags nothing, which
    zeroes it the same way. So a degraded response must still rank, and must still
    put a few chunks over the line.

    Ranks by duplicate-signature share -- a scripted policy replays action
    sequences, a human does not -- computed from the raw payload only. No model,
    no artifact, no feature module, so it stays available when those are exactly
    what failed. Ranking quality is poor by design; the point is that a degraded
    round still collects its 0.35 structural floor instead of scoring zero.
    """
    total = len(chunks)
    if total == 0:
        return []

    concentration = []
    for chunk in chunks:
        try:
            signatures = [
                tuple(
                    (str(a.get("action_type") or ""), str(a.get("street") or ""))
                    for a in ((hand or {}).get("actions") or [])
                    if isinstance(a, dict)
                )
                for hand in (chunk or [])
                if isinstance(hand, dict)
            ]
            concentration.append(
                1.0 - len(set(signatures)) / len(signatures) if signatures else 0.0
            )
        except Exception:
            concentration.append(0.0)

    k = max(1, min(total, int(total * _FALLBACK_POSITIVE_FRACTION)))
    order = sorted(range(total), key=lambda i: (-concentration[i], i))
    positives, negatives = order[:k], order[k:]
    scores = [0.0] * total
    for rank, index in enumerate(positives):
        share = 1.0 if len(positives) <= 1 else 1.0 - rank / (len(positives) - 1)
        scores[index] = round(0.501 + share * 0.008, 6)
    for rank, index in enumerate(negatives):
        share = 1.0 if len(negatives) <= 1 else 1.0 - rank / (len(negatives) - 1)
        scores[index] = round(0.010 + share * 0.480, 6)
    return scores


def _policy_enabled(meta) -> object:
    """Drift-policy state, whichever meta.json layout this build wrote.

    v4 records one policy report; v5 records one per feature view. Read both
    rather than assuming, so a schema change downgrades the log line instead of
    silently reporting None.
    """
    policy = meta.get("feature_policy") or {}
    if "enabled" in policy:
        return policy["enabled"]
    for value in policy.values():
        if isinstance(value, dict) and "enabled" in value:
            return value["enabled"]
    return None


def _offline(meta, key: str) -> float:
    """Offline holdout metric. Benchmark-only -- it does NOT predict live reward
    (this family measured 0.93 offline against 0.53 live), so it is logged for
    provenance, never as a health signal.
    """
    block = meta.get("offline_holdout") or meta.get("walk_forward") or {}
    try:
        return float(block.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


class Miner(BaseMinerNeuron):
    """Poker44 bot-detection miner."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.detector = get_model()
        meta = self.detector.meta
        repo_url = os.environ.get("POKER44_MODEL_REPO_URL", "")
        repo_commit = _repo_commit(ROOT)
        self.model_manifest = build_local_model_manifest(
            repo_root=ROOT,
            # Every entry must be present in the published repo, so the manifest
            # never names a file a reader cannot open.
            implementation_files=[
                ROOT / "neurons" / "miner.py",
                ROOT / "detector" / "inference.py",
                ROOT / "detector" / "features.py",
                ROOT / "detector" / "signature.py",
                ROOT / "detector" / "live_capture.py",
                ROOT / "detector" / "artifacts" / "meta.json",
            ],
            defaults={
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "framework": "scikit-learn",
                "license": "MIT",
                "repo_url": repo_url,
                "repo_commit": repo_commit,
                "artifact_sha256": _artifact_sha256(ARTIFACT),
                "artifact_url": "",
                # Scope the open_source claim rather than leaving it to be read
                # as broader than it is. Every file the manifest NAMES is in the
                # public repo; the training pipeline and the weights are not, so
                # say that plainly. artifact_sha256 is what lets a reader verify
                # the served weights even though they are distributed
                # out-of-band. A precise true claim is worth more than a broad
                # one that invites an argument.
                "notes": (
                    "Behavioural bot detector. Inference path, feature pipeline "
                    "and model metadata are published; the training pipeline is "
                    "private and the weights are distributed out-of-band, "
                    "identified by artifact_sha256."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": "Trained only on the public Poker44 benchmark.",
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": "No validator-only data is used.",
                "data_attestation": "Features use miner-visible behaviour only.",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        bt.logging.info(
            f"Poker44 miner ready | version={meta.get('model_version', '?')} "
            f"artifact={str(meta.get('artifact_sha256', ''))[:12]} "
            f"features={meta.get('feature_count')} "
            f"scale_norm={meta.get('scale_norm')} "
            f"drift_policy={_policy_enabled(meta)} | "
            f"offline reward={_offline(meta, 'reward'):.4f} "
            f"ap={_offline(meta, 'ap'):.4f} (benchmark-only; does not predict live)")
        bt.logging.info(
            f"Manifest transparency: {self.manifest_compliance['status']} "
            f"(missing={self.manifest_compliance['missing_fields']}) "
            f"digest={manifest_digest(self.model_manifest)}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        started = time.perf_counter()
        degraded = False
        try:
            scores = self.detector.score_chunks(chunks)
        except Exception as exc:  # never crash on a malformed request
            bt.logging.error(f"SCORING FAILED ({exc}); serving the model-free "
                             f"fallback ranking. Investigate and restart.")
            degraded = True
            scores = _fallback_scores(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        # Per-request diagnostics. positive_fraction is the one that matters: a
        # 0.0 leaderboard result is threshold_sanity_quality going to zero, and
        # the only way to tell "flagged nothing" from "flagged plenty but all
        # wrong" after the fact is to have logged this at the time.
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if scores:
            ordered = sorted(scores)
            n = len(ordered)
            positives = sum(1 for s in scores if s >= 0.5)
            hand_counts = [len(c) for c in chunks] or [0]
            bt.logging.info(
                f"Scored {n} chunks | positive_fraction={positives / n:.4f} "
                f"({positives}/{n}) | min={ordered[0]:.4f} med={ordered[n // 2]:.4f} "
                f"max={ordered[-1]:.4f} mean={sum(scores) / n:.4f} | "
                f"hands/chunk {min(hand_counts)}-{max(hand_counts)} | "
                f"latency={elapsed_ms:.0f}ms{' | DEGRADED' if degraded else ''}"
            )
            if positives == 0:
                bt.logging.error(
                    "positive_fraction=0 -- no chunk reached 0.5, so "
                    "threshold_sanity_quality is 0 and this request contributes a "
                    "ZERO reward regardless of ranking quality."
                )
        else:
            bt.logging.info(f"Scored 0 chunks | latency={elapsed_ms:.0f}ms")

        # Diagnostic capture of the live (unlabeled) input distribution. OFF
        # unless POKER44_CAPTURE / POKER44_CAPTURE_BATCH are set. Both helpers
        # swallow every error internally; wrapped again here so capture can never
        # affect the response that has already been built above.
        try:
            validator_hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
            live_capture.capture(chunks, scores, self.uid, validator_hotkey)
            live_capture.capture_batch(chunks, scores, self.uid, validator_hotkey)
        except Exception:
            pass
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running...")
        while True:
            try:
                bt.logging.info(
                    f"UID {miner.uid} | incentive {miner.metagraph.I[miner.uid]:.6f}")
            except Exception:
                pass
            time.sleep(5 * 60)
