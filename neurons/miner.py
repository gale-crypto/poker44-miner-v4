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


def _artifact_url(repo_url: str, commit: str) -> str:
    """Permalink to the served weights, pinned to the commit rather than a branch
    so it keeps resolving to the exact bytes artifact_sha256 describes.
    """
    if not repo_url or not commit:
        return ""
    base = repo_url.strip().rstrip("/")
    if base.endswith(".git"):
        base = base[:-4]
    return f"{base}/raw/{commit}/detector/artifacts/model.joblib"


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
                "artifact_url": _artifact_url(repo_url, repo_commit),
                "notes": "Behavioural bot detector.",
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
            f"Poker44 miner ready | reward={meta.get('cv_reward', 0):.4f} "
            f"ap={meta.get('cv_ap', 0):.4f}")
        bt.logging.info(
            f"Manifest transparency: {self.manifest_compliance['status']} "
            f"(missing={self.manifest_compliance['missing_fields']}) "
            f"digest={manifest_digest(self.model_manifest)}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        try:
            scores = self.detector.score_chunks(chunks)
        except Exception as exc:  # never crash on a malformed request
            bt.logging.warning(f"scoring failed ({exc}); falling back to 0.5")
            scores = [0.5] * len(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Scored {len(chunks)} chunks | "
                        f"flagged={sum(1 for p in synapse.predictions if p)} | "
                        f"mean={sum(scores)/max(len(scores), 1):.3f}")
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
