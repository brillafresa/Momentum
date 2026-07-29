"""Ground-truth manifest for FMS recalibration runs.

Locks a single completed session, its snapshot, ranking hash, and audit split
so downstream feature building and refitting stay reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from calibration.session import (
    SESSION_ROOT_DIR,
    SNAPSHOT_ROOT_DIR,
    latest_completed_session,
    load_session,
)


MANIFEST_PATH = "fms_recalib_manifest.json"
AUDIT_FRACTION = 0.20
AUDIT_SEED = 20260729


@dataclass(frozen=True)
class RecalibManifest:
    session_id: str
    snapshot_id: str
    saved_at: str
    chart_period: str
    n_symbols: int
    ranking: List[str]
    ranking_hash: str
    snapshot_prices_hash: str
    inconsistencies: List[Dict[str, Any]]
    audit_symbols: List[str]
    development_symbols: List[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ranking_hash(ranking: List[str]) -> str:
    payload = "\n".join(ranking)
    return _sha256_text(payload)


def _prices_hash(prices: pd.DataFrame, symbols: List[str]) -> str:
    cols = [s for s in symbols if s in prices.columns]
    subset = prices[cols].copy()
    subset = subset.sort_index()
    csv_bytes = subset.to_csv().encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _split_audit_symbols(ranking: List[str], *, fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    n = len(ranking)
    audit_n = max(1, int(round(n * fraction)))
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n + 1, dtype=float)
    # Stratified pick: sample evenly across rank regions.
    order = np.argsort(ranks)
    bucket_size = max(1, n // audit_n)
    audit: List[str] = []
    for start in range(0, n, bucket_size):
        bucket = [ranking[i] for i in order[start : start + bucket_size]]
        if bucket:
            audit.append(str(rng.choice(bucket)))
        if len(audit) >= audit_n:
            break
    audit = sorted(set(audit))
    if len(audit) < audit_n:
        remaining = [s for s in ranking if s not in audit]
        rng.shuffle(remaining)
        audit.extend(remaining[: audit_n - len(audit)])
        audit = sorted(set(audit))
    development = [s for s in ranking if s not in audit]
    return audit, development


def build_manifest(
    *,
    session_id: Optional[str] = None,
    audit_fraction: float = AUDIT_FRACTION,
    audit_seed: int = AUDIT_SEED,
) -> RecalibManifest:
    """Build and persist the latest completed-session manifest."""
    if session_id is None:
        session_id, session = latest_completed_session()
    else:
        session = load_session(session_id)

    ranking = [str(s) for s in session.get("final_ranking") or []]
    if not ranking:
        raise RuntimeError(f"session {session_id} has no final_ranking")

    snapshot_id = str(session.get("snapshot_id") or session_id.replace("cal_", ""))
    snap_path = os.path.join(SNAPSHOT_ROOT_DIR, snapshot_id, "prices_krw.pkl")
    if not os.path.exists(snap_path):
        raise FileNotFoundError(f"snapshot not found: {snap_path}")

    prices = pd.read_pickle(snap_path)
    missing = [s for s in ranking if s not in prices.columns]
    if missing:
        raise RuntimeError(f"ranking symbols missing from snapshot: {missing[:5]}")

    meta = session.get("meta") or {}
    chart_period = str(meta.get("chart_period") or "3M")
    audit_symbols, development_symbols = _split_audit_symbols(
        ranking, fraction=audit_fraction, seed=audit_seed
    )

    manifest = RecalibManifest(
        session_id=session_id,
        snapshot_id=snapshot_id,
        saved_at=str(session.get("saved_at") or ""),
        chart_period=chart_period,
        n_symbols=len(ranking),
        ranking=ranking,
        ranking_hash=_ranking_hash(ranking),
        snapshot_prices_hash=_prices_hash(prices, ranking),
        inconsistencies=list(session.get("inconsistencies") or []),
        audit_symbols=audit_symbols,
        development_symbols=development_symbols,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
    return manifest


def load_manifest(path: str = MANIFEST_PATH) -> RecalibManifest:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return RecalibManifest(**payload)


def assert_manifest_fresh(
    manifest: RecalibManifest,
    *,
    ranking: List[str],
    prices: pd.DataFrame,
) -> None:
    """Raise when session ranking or snapshot drifts from the locked manifest."""
    if _ranking_hash(ranking) != manifest.ranking_hash:
        raise RuntimeError("ranking hash mismatch — rebuild manifest before refitting")
    if _prices_hash(prices, ranking) != manifest.snapshot_prices_hash:
        raise RuntimeError("snapshot prices hash mismatch — rebuild manifest before refitting")
    if len(ranking) != manifest.n_symbols:
        raise RuntimeError(
            f"symbol count mismatch: expected {manifest.n_symbols}, got {len(ranking)}"
        )
