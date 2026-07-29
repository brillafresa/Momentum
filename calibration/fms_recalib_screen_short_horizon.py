"""Legacy incremental short-horizon screen; not the current refit path.

The full-set result is descriptive. Repeated stratified holdouts tune the
candidate weight on training symbols and report unseen-symbol metric deltas.
Use ``fms_recalib_refit.py`` for the zero-based scratch workflow.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from calibration.fms_recalib_tune_weights_and_transitions import (
    Metrics,
    compute_metrics,
    dominates,
)
from calibration.session import SNAPSHOT_ROOT_DIR, latest_completed_session
from calibration.short_horizon_features import compute_short_horizon_candidates
from core.fms import score_legacy_fms_from_feature_frame


FEATURE_CSV = "fms_recalib_features.csv"
OUT_CSV = "fms_recalib_short_horizon_screen.csv"
ALPHAS = np.linspace(-1.5, 1.5, 81)


def _standardize(series: pd.Series) -> pd.Series:
    """Median-fill and standardize an experimental feature."""
    values = series.astype(float).replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / std


def _metric_key(metrics: Metrics) -> tuple[float, float, float]:
    """Return the optimization key used by existing FMS tuning scripts."""
    return metrics.inv, -metrics.rho, metrics.pair_err


def _compute_metrics_fast(true_rank: pd.Series, score: pd.Series) -> Metrics:
    """Compute the existing three rank metrics with vectorized pair operations."""
    common = pd.concat([true_rank, score], axis=1).dropna()
    common.columns = ["true_rank", "score"]
    common = common.sort_values("true_rank")
    return _compute_metrics_arrays(
        common["true_rank"].to_numpy(dtype=float),
        common["score"].to_numpy(dtype=float),
    )


def _compute_metrics_arrays(ranks: np.ndarray, scores: np.ndarray) -> Metrics:
    """Compute ranking metrics from finite arrays ordered by true rank."""
    n = len(ranks)
    if n <= 1:
        return Metrics(inv=0.0, rho=1.0, pair_err=0.0)

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    inversions = int(np.count_nonzero((scores[:, None] < scores[None, :]) & upper))
    inversion_rate = inversions / int(upper.sum())

    descending = np.argsort(-scores, kind="stable")
    model_rank = np.empty(n, dtype=float)
    model_rank[descending] = np.arange(1, n + 1, dtype=float)
    rho = float(np.corrcoef(ranks, model_rank)[0, 1])

    true_delta = ranks[None, :] - ranks[:, None]
    model_delta = model_rank[None, :] - model_rank[:, None]
    pair_error = float(np.abs(true_delta - model_delta)[upper].mean())
    return Metrics(inv=inversion_rate, rho=rho, pair_err=pair_error)


def _best_alpha(
    frame: pd.DataFrame,
    baseline_score: pd.Series,
    feature_z: pd.Series,
    indices: Iterable[str],
) -> tuple[float, Metrics]:
    """Choose an additive weight using only the supplied symbols."""
    idx = list(indices)
    ranks = frame.loc[idx, "rank"].to_numpy(dtype=float)
    base_values = baseline_score.loc[idx].to_numpy(dtype=float)
    feature_values = feature_z.loc[idx].to_numpy(dtype=float)
    order = np.argsort(ranks, kind="stable")
    ranks = ranks[order]
    base_values = base_values[order]
    feature_values = feature_values[order]
    best_alpha = 0.0
    best_metrics = _compute_metrics_arrays(ranks, base_values)
    for alpha in ALPHAS:
        metrics = _compute_metrics_arrays(
            ranks, base_values + float(alpha) * feature_values
        )
        if _metric_key(metrics) < _metric_key(best_metrics):
            best_alpha = float(alpha)
            best_metrics = metrics
    return best_alpha, best_metrics


def _stratified_folds(rank: pd.Series, rng: np.random.Generator, k: int = 5) -> list[list[str]]:
    """Create rank-stratified folds with every rank region represented."""
    ordered = rank.sort_values().index.to_numpy(dtype=str)
    buckets = np.array_split(ordered, k)
    folds: list[list[str]] = [[] for _ in range(k)]
    for bucket in buckets:
        shuffled = bucket.copy()
        rng.shuffle(shuffled)
        for i, symbol in enumerate(shuffled):
            folds[i % k].append(str(symbol))
    return folds


def _cross_validate(
    frame: pd.DataFrame,
    baseline_score: pd.Series,
    feature_z: pd.Series,
    *,
    repeats: int = 20,
) -> dict[str, float]:
    """Tune on four folds and summarize deltas on the held-out fold."""
    rng = np.random.default_rng(20260729)
    inv_delta: list[float] = []
    rho_delta: list[float] = []
    err_delta: list[float] = []
    alphas: list[float] = []
    strict_wins = 0
    splits = 0
    all_symbols = set(frame.index.astype(str))

    for _ in range(repeats):
        for test_symbols in _stratified_folds(frame["rank"], rng):
            train_symbols = sorted(all_symbols.difference(test_symbols))
            alpha, _ = _best_alpha(frame, baseline_score, feature_z, train_symbols)
            base = _compute_metrics_fast(
                frame.loc[test_symbols, "rank"], baseline_score.loc[test_symbols]
            )
            candidate = _compute_metrics_fast(
                frame.loc[test_symbols, "rank"],
                baseline_score.loc[test_symbols] + alpha * feature_z.loc[test_symbols],
            )
            inv_delta.append(candidate.inv - base.inv)
            rho_delta.append(candidate.rho - base.rho)
            err_delta.append(candidate.pair_err - base.pair_err)
            alphas.append(alpha)
            strict_wins += int(dominates(candidate, base))
            splits += 1

    return {
        "cv_alpha_median": float(np.median(alphas)),
        "cv_inv_delta_median": float(np.median(inv_delta)),
        "cv_rho_delta_median": float(np.median(rho_delta)),
        "cv_pair_err_delta_median": float(np.median(err_delta)),
        "cv_strict_win_rate": strict_wins / splits,
    }


def main() -> None:
    """Run additive screening and write a compact evidence table."""
    session_id, session = latest_completed_session()
    frame = pd.read_csv(FEATURE_CSV, index_col=0)
    frame.index = frame.index.astype(str)
    expected = [str(s) for s in session["final_ranking"]]
    if frame.index.tolist() != expected:
        raise RuntimeError(
            f"{FEATURE_CSV} does not match latest completed session {session_id}; "
            "run python fms_recalib_build_features.py first"
        )

    snapshot_id = str(session["snapshot_id"])
    prices_path = os.path.join(SNAPSHOT_ROOT_DIR, snapshot_id, "prices_krw.pkl")
    prices = pd.read_pickle(prices_path)
    candidates = compute_short_horizon_candidates(prices).reindex(frame.index)
    baseline_score = score_legacy_fms_from_feature_frame(frame)
    baseline_metrics = compute_metrics(frame, baseline_score)

    rows: list[dict[str, float | str]] = []
    for name in candidates.columns:
        feature_z = _standardize(candidates[name])
        alpha, full_metrics = _best_alpha(frame, baseline_score, feature_z, frame.index)
        row: dict[str, float | str] = {
            "feature": name,
            "rank_spearman": float(candidates[name].corr(frame["rank"], method="spearman")),
            "full_alpha": alpha,
            "full_inversion_rate": full_metrics.inv,
            "full_spearman_rho": full_metrics.rho,
            "full_pair_delta_error": full_metrics.pair_err,
            "full_strictly_dominates": float(dominates(full_metrics, baseline_metrics)),
        }
        row.update(_cross_validate(frame, baseline_score, feature_z))
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["cv_strict_win_rate", "full_inversion_rate"],
        ascending=[False, True],
    )
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("session:", session_id)
    print("baseline:", json.dumps(asdict(baseline_metrics), ensure_ascii=False))
    print(result.to_string(index=False))
    print("Wrote", OUT_CSV)


if __name__ == "__main__":
    main()
