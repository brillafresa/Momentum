"""Legacy production-plus-addon experiment; not the current refit path.

The production FMS body remains unchanged. This script evaluates two
interpretable additive candidates against its score:

* TE15: signed 15-day trend efficiency (reward)
* STALE_AGE: days since the 21-day high × prior visible daily spike (penalty)

Weights are selected on training symbols in repeated rank-stratified folds.
Use ``fms_recalib_refit.py`` for the current zero-based scratch workflow.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from calibration.fms_recalib_screen_short_horizon import (
    Metrics,
    _compute_metrics_fast,
    _standardize,
    _stratified_folds,
)
from calibration.session import SNAPSHOT_ROOT_DIR, latest_completed_session
from calibration.short_horizon_features import compute_short_horizon_candidates
from core.fms import production_fms_score_params, score_legacy_fms_from_feature_frame


FEATURE_CSV = "fms_recalib_features.csv"
OUT_JSON = "fms_recalib_latest_fit.json"
TE_WEIGHTS = np.arange(0.0, 1.21, 0.1)
STALE_WEIGHTS = np.arange(0.0, 1.01, 0.1)


def _key(metrics: Metrics) -> tuple[float, float, float]:
    """Order candidates by inversion, then Spearman, then pair error."""
    return metrics.inv, -metrics.rho, metrics.pair_err


def _strictly_improves(candidate: Metrics, baseline: Metrics) -> bool:
    """Return whether all three ranking metrics improve strictly."""
    return (
        candidate.inv < baseline.inv
        and candidate.rho > baseline.rho
        and candidate.pair_err < baseline.pair_err
    )


def _curvature_sign_correction(frame: pd.DataFrame) -> pd.Series:
    """Return the delta that makes positive EMA20 curvature favorable.

    Production currently rewards negative curvature and penalizes positive
    curvature. The correction reverses those two subterms while preserving the
    existing EMA-shape outer weight and slope contribution.
    """
    params = production_fms_score_params()
    curvature = frame["EMA20_CURV_20D"].astype(float)
    positive = _standardize(curvature.clip(lower=0.0))
    negative = _standardize((-curvature).clip(lower=0.0))
    inner_delta = (
        params.w_ema_curv_reward_base + params.w_ema_curv_penalty_base
    ) * (positive - negative)
    return params.w_ema_shape * inner_delta


def _best_weights(
    frame: pd.DataFrame,
    baseline: pd.Series,
    te15: pd.Series,
    stale_age: pd.Series,
    symbols: list[str],
    *,
    allow_curvature_fix: bool = False,
) -> tuple[bool, float, float, Metrics]:
    """Fit the two additive weights on the supplied symbols only."""
    ranks = frame.loc[symbols, "rank"]
    curvature_delta = _curvature_sign_correction(frame).loc[symbols]
    best = (
        False,
        0.0,
        0.0,
        _compute_metrics_fast(ranks, baseline.loc[symbols]),
    )
    curvature_options = (False, True) if allow_curvature_fix else (False,)
    for fix_curvature_sign in curvature_options:
        corrected_base = baseline.loc[symbols]
        if fix_curvature_sign:
            corrected_base = corrected_base + curvature_delta
        for te_weight in TE_WEIGHTS:
            for stale_weight in STALE_WEIGHTS:
                score = (
                    corrected_base
                    + float(te_weight) * te15.loc[symbols]
                    - float(stale_weight) * stale_age.loc[symbols]
                )
                metrics = _compute_metrics_fast(ranks, score)
                if _key(metrics) < _key(best[3]):
                    best = (
                        fix_curvature_sign,
                        float(te_weight),
                        float(stale_weight),
                        metrics,
                    )
    return best


def _nested_cv(
    frame: pd.DataFrame,
    baseline: pd.Series,
    te15: pd.Series,
    stale_age: pd.Series,
    *,
    allow_curvature_fix: bool = False,
    repeats: int = 40,
) -> dict[str, object]:
    """Tune on four folds and evaluate the selected weights on held-out symbols."""
    rng = np.random.default_rng(20260729)
    symbols = set(frame.index.astype(str))
    curvature_delta = _curvature_sign_correction(frame)
    rows = []
    for _ in range(repeats):
        for test_symbols in _stratified_folds(frame["rank"], rng):
            train_symbols = sorted(symbols.difference(test_symbols))
            fix_curvature_sign, te_weight, stale_weight, _ = _best_weights(
                frame,
                baseline,
                te15,
                stale_age,
                train_symbols,
                allow_curvature_fix=allow_curvature_fix,
            )
            candidate_score = baseline.loc[test_symbols]
            if fix_curvature_sign:
                candidate_score = candidate_score + curvature_delta.loc[test_symbols]
            base_metrics = _compute_metrics_fast(
                frame.loc[test_symbols, "rank"], baseline.loc[test_symbols]
            )
            candidate_metrics = _compute_metrics_fast(
                frame.loc[test_symbols, "rank"],
                candidate_score
                + te_weight * te15.loc[test_symbols]
                - stale_weight * stale_age.loc[test_symbols],
            )
            rows.append(
                {
                    "fix_curvature_sign": fix_curvature_sign,
                    "te_weight": te_weight,
                    "stale_weight": stale_weight,
                    "inv_delta": candidate_metrics.inv - base_metrics.inv,
                    "rho_delta": candidate_metrics.rho - base_metrics.rho,
                    "pair_err_delta": candidate_metrics.pair_err - base_metrics.pair_err,
                    "strict_win": _strictly_improves(candidate_metrics, base_metrics),
                }
            )
    result = pd.DataFrame(rows)
    return {
        "splits": len(result),
        "strict_win_rate": float(result["strict_win"].mean()),
        "curvature_sign_fix_rate": float(result["fix_curvature_sign"].mean()),
        "te_weight_median": float(result["te_weight"].median()),
        "stale_weight_median": float(result["stale_weight"].median()),
        "inv_delta_median": float(result["inv_delta"].median()),
        "rho_delta_median": float(result["rho_delta"].median()),
        "pair_err_delta_median": float(result["pair_err_delta"].median()),
        "inv_delta_p10_p90": [
            float(result["inv_delta"].quantile(0.10)),
            float(result["inv_delta"].quantile(0.90)),
        ],
        "rho_delta_p10_p90": [
            float(result["rho_delta"].quantile(0.10)),
            float(result["rho_delta"].quantile(0.90)),
        ],
        "pair_err_delta_p10_p90": [
            float(result["pair_err_delta"].quantile(0.10)),
            float(result["pair_err_delta"].quantile(0.90)),
        ],
    }


def _stratified_subsample_validation(
    frame: pd.DataFrame,
    baseline: pd.Series,
    candidate: pd.Series,
    repeats: int = 500,
) -> dict[str, object]:
    """Measure fixed-model deltas on repeated 80% rank-stratified symbol subsets."""
    rng = np.random.default_rng(20260730)
    quintiles = np.array_split(frame.sort_values("rank").index.to_numpy(dtype=str), 5)
    rows = []
    for _ in range(repeats):
        sample = []
        for quintile in quintiles:
            count = max(2, int(round(len(quintile) * 0.8)))
            sample.extend(rng.choice(quintile, size=count, replace=False).tolist())
        base_metrics = _compute_metrics_fast(frame.loc[sample, "rank"], baseline.loc[sample])
        candidate_metrics = _compute_metrics_fast(
            frame.loc[sample, "rank"], candidate.loc[sample]
        )
        rows.append(
            {
                "inv_delta": candidate_metrics.inv - base_metrics.inv,
                "rho_delta": candidate_metrics.rho - base_metrics.rho,
                "pair_err_delta": candidate_metrics.pair_err - base_metrics.pair_err,
                "strict_win": _strictly_improves(candidate_metrics, base_metrics),
            }
        )
    result = pd.DataFrame(rows)
    return {
        "subsamples": len(result),
        "strict_win_rate": float(result["strict_win"].mean()),
        "inv_delta_95pct": [
            float(result["inv_delta"].quantile(0.025)),
            float(result["inv_delta"].quantile(0.975)),
        ],
        "rho_delta_95pct": [
            float(result["rho_delta"].quantile(0.025)),
            float(result["rho_delta"].quantile(0.975)),
        ],
        "pair_err_delta_95pct": [
            float(result["pair_err_delta"].quantile(0.025)),
            float(result["pair_err_delta"].quantile(0.975)),
        ],
    }


def _label_uncertainty(
    session: dict,
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, object]:
    """Re-evaluate all combinations of the adjacent review reversals."""
    ranking = [str(symbol) for symbol in session["final_ranking"]]
    pairs = [
        (str(item["a"]), str(item["b"]))
        for item in session.get("inconsistencies") or []
    ]
    variants = 0
    strict_wins = 0
    for bits in itertools.product([False, True], repeat=len(pairs)):
        order = ranking.copy()
        for reverse, (left, right) in zip(bits, pairs):
            if not reverse or left not in order or right not in order:
                continue
            i, j = order.index(left), order.index(right)
            order[i], order[j] = order[j], order[i]
        rank = pd.Series({symbol: i + 1 for i, symbol in enumerate(order)})
        base_metrics = _compute_metrics_fast(rank, baseline.reindex(rank.index))
        candidate_metrics = _compute_metrics_fast(rank, candidate.reindex(rank.index))
        strict_wins += int(_strictly_improves(candidate_metrics, base_metrics))
        variants += 1
    return {
        "review_inconsistencies": len(pairs),
        "ranking_variants": variants,
        "strict_wins": strict_wins,
    }


def _comparison_accuracy(session: dict, score: pd.Series) -> float:
    """Return descriptive accuracy on the user's recorded A/B choices."""
    records = (session.get("history") or []) + (session.get("review_history") or [])
    correct = 0
    valid = 0
    for item in records:
        left, right, choice = str(item["a"]), str(item["b"]), str(item["choice"])
        if left not in score.index or right not in score.index:
            continue
        predicted = left if score.loc[left] >= score.loc[right] else right
        correct += int(predicted == choice)
        valid += 1
    return correct / valid if valid else float("nan")


def main() -> None:
    """Fit latest-session candidates and persist the evidence report."""
    session_id, session = latest_completed_session()
    frame = pd.read_csv(FEATURE_CSV, index_col=0)
    frame.index = frame.index.astype(str)
    ranking = [str(symbol) for symbol in session["final_ranking"]]
    if frame.index.tolist() != ranking:
        raise RuntimeError(
            f"{FEATURE_CSV} does not match latest completed session {session_id}"
        )

    prices = pd.read_pickle(
        os.path.join(SNAPSHOT_ROOT_DIR, str(session["snapshot_id"]), "prices_krw.pkl")
    )
    features = compute_short_horizon_candidates(prices).reindex(frame.index)
    te15 = _standardize(features["TREND_EFFICIENCY_15D"])
    stale_age = _standardize(features["STALE_AGE"])
    baseline = score_legacy_fms_from_feature_frame(frame)

    fix_curvature_sign, te_weight, stale_weight, candidate_metrics = _best_weights(
        frame,
        baseline,
        te15,
        stale_age,
        frame.index.tolist(),
        allow_curvature_fix=True,
    )
    candidate = baseline.copy()
    if fix_curvature_sign:
        candidate = candidate + _curvature_sign_correction(frame)
    candidate = candidate + te_weight * te15 - stale_weight * stale_age
    baseline_metrics = _compute_metrics_fast(frame["rank"], baseline)

    true_top = set(frame.nsmallest(max(1, len(frame) // 5), "rank").index)
    model_top = set(candidate.nlargest(len(true_top)).index)
    baseline_top = set(baseline.nlargest(len(true_top)).index)

    report = {
        "policy": "latest completed session only",
        "session_id": session_id,
        "snapshot_id": session["snapshot_id"],
        "n_symbols": len(frame),
        "formula": (
            "production_fms"
            + (" + curvature_sign_correction" if fix_curvature_sign else "")
            + " + te_weight*Z(TE15) - stale_weight*Z(STALE_AGE)"
        ),
        "definitions": {
            "TE15": "log(P_t/P_t-15) / sum(abs(daily_log_returns_15d))",
            "STALE_AGE": "days_since_oldest_21d_high * max_daily_log_gain(t-21:t-3)",
        },
        "weights": {
            "fix_ema20_curvature_sign": fix_curvature_sign,
            "te_weight": te_weight,
            "stale_weight": stale_weight,
        },
        "baseline_metrics": asdict(baseline_metrics),
        "candidate_metrics": asdict(candidate_metrics),
        "curvature_sign_ablation": {
            "metrics_if_fixed_alone": asdict(
                _compute_metrics_fast(
                    frame["rank"], baseline + _curvature_sign_correction(frame)
                )
            ),
            "selected_in_final_model": fix_curvature_sign,
        },
        "nested_symbol_holdout": _nested_cv(
            frame,
            baseline,
            te15,
            stale_age,
            allow_curvature_fix=fix_curvature_sign,
        ),
        "stratified_subsamples": _stratified_subsample_validation(
            frame, baseline, candidate
        ),
        "label_uncertainty": _label_uncertainty(session, baseline, candidate),
        "comparison_accuracy": {
            "baseline": _comparison_accuracy(session, baseline),
            "candidate": _comparison_accuracy(session, candidate),
        },
        "top_quintile_recall": {
            "baseline": len(true_top & baseline_top) / len(true_top),
            "candidate": len(true_top & model_top) / len(true_top),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
