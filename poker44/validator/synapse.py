"""Synapse definitions for Poker44 miners and validators."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

import bittensor as bt
from pydantic import ConfigDict, Field

class DetectionSynapse(bt.Synapse):
    """
    Carries multiple chunks (batches) of poker hands to a miner and returns bot-risk scores.
    Each chunk gets one risk score/prediction.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # List of chunks, where each chunk is a list of hands
    # required_hash_fields forces this to be sent in body, not headers
    chunks: List[List[dict]] = Field(default_factory=list)
    risk_scores: Optional[List[float]] = None  # One score per chunk
    predictions: Optional[List[bool]] = None    # One prediction per chunk
    model_manifest: Optional[Dict[str, Any]] = None

    # Tell Bittensor to send chunks in the body, not headers
    required_hash_fields: ClassVar[List[str]] = ["chunks"]

    def deserialize(self) -> "DetectionSynapse":
        """Deserialize chunks back into HandHistory objects if needed."""
        # Chunks arrive as list of lists of dicts
        # You can keep them as dicts or convert back to HandHistory
        return self


# Poker44 v3. Mirrors poker44/protocol.py on the subnet's dev branch. Kept as a
# SEPARATE class rather than the upstream `DetectionSynapse = SessionDetection-
# Synapse` alias: bittensor dispatches on the class NAME, so aliasing would
# unregister the v2 handler and stop us answering validators that have not
# migrated yet. Both are attached; whichever arrives is served.
SUPPORTED_PROTOCOL_VERSIONS = frozenset({"1", "2"})


class SessionDetectionSynapse(bt.Synapse):
    """Classify miner-visible subject sessions as human or bot.

    One risk score per session, near zero for human and near one for bot.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    protocol_version: str = "2"
    window_id: str = ""
    dataset_hash: str = ""
    sessions: List[Dict[str, Any]] = Field(default_factory=list)

    risk_scores: Optional[List[float]] = None
    predictions: Optional[List[bool]] = None
    model_version: Optional[str] = None

    required_hash_fields: ClassVar[List[str]] = [
        "protocol_version",
        "window_id",
        "dataset_hash",
        "sessions",
    ]

    def deserialize(self) -> "SessionDetectionSynapse":
        return self


def validate_session_request(synapse: "SessionDetectionSynapse") -> None:
    """Reject unsupported or incomplete requests before inference."""
    if synapse.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError(
            f"Unsupported Poker44 protocol version: {synapse.protocol_version}"
        )
    if not synapse.window_id.strip():
        raise ValueError("window_id is required")
    if synapse.protocol_version == "2":
        dataset_hash = synapse.dataset_hash.strip().lower()
        if len(dataset_hash) != 64 or any(
            character not in "0123456789abcdef" for character in dataset_hash
        ):
            raise ValueError("dataset_hash must be a 64-character SHA-256 hex digest")

# ---------------------------------------------------------------------------
# Poker44 v3.0 PRODUCTION contract (subnet release 0.2.1, main @ 8f13026).
#
# This is what validators actually send. It is NOT the SessionDetectionSynapse
# the dev branch advertised: the class name differs (bittensor routes on the
# class name, so the wrong one is simply never called), the payload field is
# `items` not `sessions`, the envelope carries contract_version and query_id,
# and each item is schema 4.1 with exactly four decisions.
#
# Copied verbatim from poker44/protocol.py upstream rather than retyped, so the
# field names and required_hash_fields cannot drift from the sender's.
# ---------------------------------------------------------------------------
from typing import Literal

from poker44.contracts import find_forbidden, validate_v4_micro_session

class MicroSessionDetectionSynapse(bt.Synapse):
    """Classify audited schema-v4.1 tournament micro-sessions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    contract_version: Literal["microsession-v1"] = "microsession-v1"
    window_id: str = ""
    dataset_hash: str = ""
    query_id: str = ""
    items: list[dict[str, Any]] = Field(default_factory=list)
    risk_scores: list[float] | None = None
    predictions: list[bool] | None = None
    model_version: str | None = None

    required_hash_fields: ClassVar[list[str]] = [
        "contract_version",
        "window_id",
        "dataset_hash",
        "query_id",
        "items",
    ]

    def deserialize(self) -> "MicroSessionDetectionSynapse":
        return self


def _validate_envelope(window_id: str, dataset_hash: str, query_id: str) -> None:
    if not window_id.strip():
        raise ValueError("window_id is required")
    digest = dataset_hash.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("dataset_hash must be a 64-character SHA-256 hex digest")
    if not query_id.strip():
        raise ValueError("query_id is required")


def validate_micro_session_request(synapse: MicroSessionDetectionSynapse) -> None:
    _validate_envelope(synapse.window_id, synapse.dataset_hash, synapse.query_id)
    if not synapse.items:
        raise ValueError("micro-session request contains no items")
    for index, item in enumerate(synapse.items):
        if not isinstance(item, dict) or str(item.get("schema_version")) != "4.1":
            raise ValueError(f"items[{index}] must use schema 4.1")
        leaked = find_forbidden(item, f"items[{index}]")
        if leaked:
            raise ValueError(f"items[{index}] contains forbidden fields: {sorted(leaked)}")
        validate_v4_micro_session(item, index)
