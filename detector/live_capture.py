"""Live validator-query capture (operational, local-only, gitignored).

Derived from poker44_ml/live_capture.py in Travis861-Poker44_v2 (MIT), the same
origin as super_poker/live_capture.py in super_poker_3. Adapted for this repo:
capture directory defaults to <repo_root>/live_capture, plus status() and a
summarize CLI.

WHY. The public benchmark is not the live distribution. Measured on real captured
payloads: benchmark chunks run 30-40 hands at ~240bb stacks and ~110bb pot
growth; live runs 80-100 hands at a pinned 100bb buy-in with ~3-6bb pots, up to 9
seats, and a passive share of 0.60 against the benchmark's 0.18. A classifier
separates the two domains at AUC 1.0000. You cannot see any of that without
holding real live payloads, and the leaderboard will not tell you -- it reports
one aggregate number per round.

WHAT IS PERSISTED. INPUTS ONLY, plus this miner's own score. A live query carries
no ground-truth bot/human label, so nothing written here can serve as a
supervised training label. The payloads are already sanitized by the validator
(no hole cards, no board, no outcomes, no real seats, amounts quantized).

SAFETY CONTRACT
  * OFF by default. POKER44_CAPTURE=1 (per-chunk), POKER44_CAPTURE_BATCH=1
    (whole-query snapshots).
  * Deduped by content hash: validators re-send the same daily snapshot on every
    query, so without this the size cap fills with duplicates within hours.
  * Size-capped (POKER44_CAPTURE_MAX_BYTES, default 250MB). status() reports when
    the cap has latched, so a silent stop is visible in the miner log rather than
    being discovered later as a gap in the data.
  * Thread-safe and FAIL-SAFE: every path is wrapped, so a capture error can
    never affect scoring or the response.
  * Output is gitignored and never leaves the box.

ATTESTATION. While these captures are used only for diagnosis they do NOT change
your training-data statement. The moment you feed them into training -- even
unlabeled, for domain adaptation -- update POKER44_MODEL_TRAINING_DATA_STATEMENT
and POKER44_MODEL_PRIVATE_DATA_ATTESTATION truthfully before serving again.

Inspect what you have collected:

    python -m detector.live_capture --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

_LOCK = threading.Lock()
# This file lives at <repo_root>/detector/live_capture.py, so parents[1] is root.
_DIR = Path(
    os.getenv("POKER44_CAPTURE_DIR")
    or Path(__file__).resolve().parents[1] / "live_capture"
)
_MAX_BYTES = int(os.getenv("POKER44_CAPTURE_MAX_BYTES", str(250 * 1024 * 1024)))

_state: Dict[str, Any] = {"path": None, "full": False, "seen": None, "written": 0}
_batch: Dict[str, Any] = {"path": None, "full": False, "seen": None, "written": 0}


def enabled() -> bool:
    return os.getenv("POKER44_CAPTURE", "0") == "1"


def batch_enabled() -> bool:
    return os.getenv("POKER44_CAPTURE_BATCH", "0") == "1"


def _key(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_seen(path: Path, field: str) -> set:
    seen: set = set()
    try:
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        seen.add(_key(json.loads(line).get(field) or []))
                    except Exception:
                        continue
    except Exception:
        pass
    return seen


def _append(state: Dict[str, Any], path: Path, payload: str) -> None:
    with _LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)
    state["written"] += payload.count("\n")


def capture(
    chunks: Sequence[Sequence[dict]],
    scores: Sequence[float],
    miner_id: Any,
    validator: Any,
) -> None:
    """Append one JSONL record per NEW chunk: {t, v, uid, n, score, chunk}.

    Input-only (no labels). Never raises -- capture must not affect serving.
    """
    if not enabled() or _state["full"] or not chunks:
        return
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        if _state["path"] is None:
            _state["path"] = _DIR / f"capture_{str(miner_id)[:16]}.jsonl"
        path: Path = _state["path"]
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            _state["full"] = True
            return
        if _state["seen"] is None:
            _state["seen"] = _load_seen(path, "chunk")
        seen: set = _state["seen"]

        ts, vtag, uid = round(time.time(), 2), str(validator or "")[:8], str(miner_id)
        lines: List[str] = []
        for chunk, score in zip(chunks, scores):
            key = _key(chunk)
            if key in seen:
                continue          # same snapshot re-sent; already on disk
            seen.add(key)
            try:
                value = round(float(score), 6)
            except (TypeError, ValueError):
                value = None
            lines.append(json.dumps(
                {"t": ts, "v": vtag, "uid": uid, "n": len(chunk),
                 "score": value, "chunk": chunk},
                separators=(",", ":"), default=str,
            ))
        if lines:
            _append(_state, path, "\n".join(lines) + "\n")
    except Exception:
        pass          # capture must NEVER affect serving


def capture_batch(
    chunks: Sequence[Sequence[dict]],
    scores: Sequence[float],
    miner_id: Any,
    validator: Any,
) -> None:
    """Append the whole query as ONE record to batch_<uid>.jsonl. Never raises.

    Kept separate from the per-chunk file because batch composition -- how many
    chunks a validator sends and their size spread -- is itself a thing worth
    measuring, and it is lost once chunks are flattened.
    """
    if not batch_enabled() or _batch["full"] or not chunks:
        return
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        if _batch["path"] is None:
            _batch["path"] = _DIR / f"batch_{str(miner_id)[:16]}.jsonl"
        path: Path = _batch["path"]
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            _batch["full"] = True
            return
        if _batch["seen"] is None:
            _batch["seen"] = _load_seen(path, "chunks")
        key = _key(list(chunks))
        if key in _batch["seen"]:
            return
        _batch["seen"].add(key)

        out: List[Any] = []
        for score in scores:
            try:
                out.append(round(float(score), 6))
            except (TypeError, ValueError):
                out.append(None)
        record = {
            "t": round(time.time(), 2),
            "v": str(validator or "")[:8],
            "uid": str(miner_id),
            "n_chunks": len(chunks),
            "sizes": [len(c) for c in chunks],
            "positive_fraction": (
                round(sum(1 for s in out if s is not None and s >= 0.5) / len(out), 6)
                if out else None
            ),
            "scores": out,
            "chunks": list(chunks),
        }
        _append(_batch, path, json.dumps(record, separators=(",", ":"), default=str) + "\n")
    except Exception:
        pass


def status() -> Dict[str, Any]:
    """One-line state for the miner log. A latched size cap must be visible."""
    return {
        "chunk_capture": enabled(),
        "batch_capture": batch_enabled(),
        "dir": str(_DIR),
        "chunk_records_written": _state["written"],
        "batch_records_written": _batch["written"],
        "capped": bool(_state["full"] or _batch["full"]),
        "max_bytes": _MAX_BYTES,
    }


# --------------------------------------------------------------------------- #
# inspection
# --------------------------------------------------------------------------- #

def summarize(directory: Path = _DIR) -> str:
    if not directory.exists():
        return f"no capture directory at {directory} (set POKER44_CAPTURE=1 to collect)"
    lines = [f"capture dir: {directory}"]
    for path in sorted(directory.glob("*.jsonl")):
        sizes: List[int] = []
        records = 0
        validators, uids = set(), set()
        first = last = None
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    records += 1
                    stamp = rec.get("t")
                    if isinstance(stamp, (int, float)):
                        first = stamp if first is None else min(first, stamp)
                        last = stamp if last is None else max(last, stamp)
                    validators.add(rec.get("v"))
                    uids.add(rec.get("uid"))
                    if "sizes" in rec:
                        sizes.extend(int(s) for s in (rec.get("sizes") or []))
                    elif "n" in rec:
                        sizes.append(int(rec.get("n") or 0))
        except OSError as exc:
            lines.append(f"  {path.name}: unreadable ({exc})")
            continue
        mb = path.stat().st_size / (1024 * 1024)
        span = ""
        if first is not None and last is not None:
            hours = (last - first) / 3600.0
            span = (f" | {time.strftime('%Y-%m-%d', time.gmtime(first))}"
                    f"..{time.strftime('%Y-%m-%d', time.gmtime(last))} ({hours:.1f}h)")
        lines.append(f"  {path.name}: {records} records, {mb:.1f} MB{span}")
        lines.append(f"    validators={len(validators)} uids={sorted(uids)}")
        if sizes:
            ordered = sorted(sizes)
            n = len(ordered)
            lines.append(
                f"    hands/chunk: min={ordered[0]} q10={ordered[n // 10]} "
                f"med={ordered[n // 2]} q90={ordered[(9 * n) // 10]} max={ordered[-1]} "
                f"(n={n})"
            )
    if len(lines) == 1:
        lines.append("  (no .jsonl files yet)")
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dir", default=None)
    args = parser.parse_args()
    directory = Path(args.dir) if args.dir else _DIR
    if args.summarize:
        print(summarize(directory))
    else:
        print(json.dumps(status(), indent=2))
        print()
        print(summarize(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
