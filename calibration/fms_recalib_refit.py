"""True from-scratch interpretable FMS refit.

Production FMS is evaluated only as a benchmark. Every candidate score starts
at zero and is learned exclusively from visible-window price features.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from calibration.manifest import load_manifest
from calibration.ranking_metrics import (
    Metrics,
    compute_metrics,
    strictly_improves,
    stratified_folds,
    top_quintile_recall,
)
from calibration.session import SNAPSHOT_ROOT_DIR, latest_completed_session
from core.fms import compute_fms_snapshot
from core.fms_features import FEATURE_DIRECTION, candidate_feature_columns

FEATURE_CSV = "fms_recalib_features.csv"
OUT_JSON = "fms_recalib_scratch_candidate.json"
SCORES_CSV = "fms_recalib_scratch_scores.csv"
RESIDUAL_CSV = "fms_recalib_scratch_residual_pairs.csv"
RANDOM_SEED = 20260729
L1_GRID = (0.01, 0.03, 0.07, 0.15)
MAX_TERMS = 10
OUTER_REPEATS = 4
BOOTSTRAPS = 120


@dataclass(frozen=True)
class FittedModel:
    family: str
    columns: List[str]
    weights: List[float]
    medians: Dict[str, float]
    means: Dict[str, float]
    scales: Dict[str, float]
    l1: float


def _load_inputs() -> tuple[pd.DataFrame, List[str], List[str], dict]:
    if not os.path.exists(FEATURE_CSV):
        raise FileNotFoundError(f"{FEATURE_CSV} missing")
    manifest = load_manifest()
    frame = pd.read_csv(FEATURE_CSV, index_col=0)
    frame = frame.loc[manifest.ranking].copy()
    frame["rank"] = np.arange(1, len(frame) + 1)
    _, session = latest_completed_session()
    return frame, manifest.development_symbols, manifest.audit_symbols, session


def _base_columns(frame: pd.DataFrame) -> List[str]:
    """Select visible-window candidates; production R_4M is benchmark-only."""
    return [
        c for c in candidate_feature_columns(frame)
        if c not in {"R_4M", "R_6M"} and frame[c].notna().sum() >= 40
    ]


def _ordinal_rank(
    frame: pd.DataFrame, symbols: Sequence[str]
) -> pd.Series:
    """Return contiguous 1..n truth ranks for a symbol subset."""
    original = frame.loc[list(symbols), "rank"].sort_values()
    return pd.Series(
        np.arange(1, len(original) + 1, dtype=float),
        index=original.index,
        name="rank",
    ).reindex(list(symbols))


def _fit_normalizer(
    frame: pd.DataFrame, symbols: Sequence[str], columns: Sequence[str]
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    medians: Dict[str, float] = {}
    means: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    for col in columns:
        values = frame.loc[list(symbols), col].astype(float).replace(
            [np.inf, -np.inf], np.nan
        )
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        medians[col] = median
        means[col] = float(filled.mean())
        scale = float(filled.std(ddof=0))
        scales[col] = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    return medians, means, scales


def _normalized_base(
    frame: pd.DataFrame,
    symbols: Sequence[str],
    columns: Sequence[str],
    medians: Dict[str, float],
    means: Dict[str, float],
    scales: Dict[str, float],
) -> pd.DataFrame:
    out = pd.DataFrame(index=list(symbols))
    for col in columns:
        values = frame.loc[list(symbols), col].astype(float).replace(
            [np.inf, -np.inf], np.nan
        )
        z = (values.fillna(medians[col]) - means[col]) / scales[col]
        out[col] = z * float(FEATURE_DIRECTION.get(col, 1))
    return out.clip(-4.0, 4.0)


def _prune_correlated(
    x: pd.DataFrame, rank: pd.Series, threshold: float = 0.94
) -> List[str]:
    quality = {}
    for col in x.columns:
        if x[col].nunique(dropna=True) <= 1:
            quality[col] = 0.0
            continue
        rho, _ = spearmanr(x[col], -rank.loc[x.index])
        quality[col] = abs(float(rho)) if np.isfinite(rho) else 0.0
    ordered = sorted(x.columns, key=lambda c: quality[c], reverse=True)
    kept: List[str] = []
    for col in ordered:
        if x[col].nunique(dropna=True) <= 1:
            continue
        if all(
            (
                not np.isfinite(float(x[col].corr(x[k])))
                or abs(float(x[col].corr(x[k]))) < threshold
            )
            for k in kept
        ):
            kept.append(col)
    return kept


def _design_for_family(base: pd.DataFrame, family: str) -> pd.DataFrame:
    if family == "sparse_linear":
        return base.copy()
    if family == "monotone_gam":
        out = {}
        for col in base.columns:
            values = base[col]
            out[f"{col}::linear"] = values
            out[f"{col}::tanh"] = np.tanh(values)
            out[f"{col}::softplus"] = np.logaddexp(values - 0.5, 0.0)
        return pd.DataFrame(out, index=base.index)
    if family == "limited_interactions":
        out = base.copy()
        pairs = (
            ("R_3D_LOG", "LOG_SLOPE_15D"),
            ("LOG_SLOPE_15D", "TREND_R2_15D"),
            ("RECOVERY_3D_VS_PRIOR7", "EMA20_ACCEL_3D_VS_10D"),
            ("R_21D_LOG", "TREND_EFFICIENCY_15D"),
            ("DD_RECOVERY", "DOWNSIDE_RMS_10D"),
            ("STALE_AGE", "LOG_SLOPE_21D"),
        )
        for left, right in pairs:
            if left in base and right in base:
                # Continuous confirmation: high only when both signed-good axes agree.
                out[f"CONFIRM({left},{right})"] = np.minimum(
                    base[left], base[right]
                )
        return out
    raise ValueError(f"unknown family: {family}")


def _pair_differences(x: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    order = np.argsort(ranks)
    ordered = x[order]
    i, j = np.triu_indices(len(order), k=1)
    return ordered[i] - ordered[j]


def _fit_nonnegative_pairwise(
    x: pd.DataFrame, ranks: pd.Series, *, l1: float
) -> np.ndarray:
    diffs = _pair_differences(
        x.to_numpy(dtype=float), ranks.loc[x.index].to_numpy(dtype=float)
    )
    n_features = x.shape[1]

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        margin = diffs @ weights
        loss = float(np.logaddexp(0.0, -margin).mean())
        sigmoid_neg = 1.0 / (1.0 + np.exp(np.clip(margin, -40.0, 40.0)))
        grad = -(diffs.T @ sigmoid_neg) / len(diffs)
        loss += l1 * float(weights.sum()) + 0.002 * float(weights @ weights)
        grad += l1 + 0.004 * weights
        return loss, grad

    result = minimize(
        lambda w: objective(w),
        np.zeros(n_features, dtype=float),
        jac=True,
        bounds=[(0.0, 6.0)] * n_features,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    weights = result.x
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("non-finite pairwise fit")
    return weights


def _select_terms(
    columns: Sequence[str], weights: np.ndarray, max_terms: int = MAX_TERMS
) -> tuple[List[str], np.ndarray]:
    order = np.argsort(-weights)
    selected = [i for i in order if weights[i] > 1e-4][:max_terms]
    if not selected:
        selected = [int(order[0])]
    return [columns[i] for i in selected], weights[selected]


def _choose_l1(
    frame: pd.DataFrame,
    symbols: List[str],
    base_columns: List[str],
    family: str,
    rng: np.random.Generator,
) -> float:
    folds = stratified_folds(frame.loc[symbols, "rank"], rng, k=4)
    rows = []
    for l1 in L1_GRID:
        fold_metrics = []
        term_counts = []
        for valid in folds:
            train = sorted(set(symbols) - set(valid))
            med, mean, scale = _fit_normalizer(frame, train, base_columns)
            train_base = _normalized_base(
                frame, train, base_columns, med, mean, scale
            )
            valid_base = _normalized_base(
                frame, valid, base_columns, med, mean, scale
            )
            kept = _prune_correlated(
                train_base, frame.loc[train, "rank"]
            )
            train_design = _design_for_family(train_base[kept], family)
            valid_design = _design_for_family(valid_base[kept], family)
            weights = _fit_nonnegative_pairwise(
                train_design, frame.loc[train, "rank"], l1=l1
            )
            cols, selected_weights = _select_terms(
                list(train_design.columns), weights
            )
            score = valid_design[cols].to_numpy() @ selected_weights
            fold_metrics.append(
                compute_metrics(
                    _ordinal_rank(frame, valid),
                    pd.Series(score, index=valid),
                )
            )
            term_counts.append(len(cols))
        rows.append(
            (
                l1,
                float(np.median([m.inv for m in fold_metrics])),
                -float(np.median([m.rho for m in fold_metrics])),
                float(np.median([m.pair_err for m in fold_metrics])),
                float(np.median(term_counts)),
            )
        )
    # One-standard-error spirit: metrics first, then fewer terms / larger penalty.
    return min(rows, key=lambda row: (row[1], row[2], row[3], row[4], -row[0]))[0]


def _fit_model(
    frame: pd.DataFrame,
    train_symbols: List[str],
    base_columns: List[str],
    family: str,
    *,
    l1: float,
) -> FittedModel:
    med, mean, scale = _fit_normalizer(frame, train_symbols, base_columns)
    base = _normalized_base(
        frame, train_symbols, base_columns, med, mean, scale
    )
    kept = _prune_correlated(base, frame.loc[train_symbols, "rank"])
    design = _design_for_family(base[kept], family)
    weights = _fit_nonnegative_pairwise(
        design, frame.loc[train_symbols, "rank"], l1=l1
    )
    cols, selected_weights = _select_terms(list(design.columns), weights)
    return FittedModel(
        family=family,
        columns=cols,
        weights=[float(v) for v in selected_weights],
        medians=med,
        means=mean,
        scales=scale,
        l1=float(l1),
    )


def _score_model(
    model: FittedModel,
    frame: pd.DataFrame,
    symbols: Sequence[str],
    base_columns: List[str],
) -> pd.Series:
    base = _normalized_base(
        frame,
        symbols,
        base_columns,
        model.medians,
        model.means,
        model.scales,
    )
    design = _design_for_family(base, model.family)
    missing = [col for col in model.columns if col not in design]
    if missing:
        raise RuntimeError(f"model columns unavailable: {missing}")
    values = design[model.columns].to_numpy() @ np.asarray(model.weights)
    return pd.Series(values, index=list(symbols), name="scratch_fms")


def _nested_validate_family(
    frame: pd.DataFrame,
    dev_symbols: List[str],
    base_columns: List[str],
    family: str,
) -> dict:
    family_seed = {
        "sparse_linear": 101,
        "monotone_gam": 211,
        "limited_interactions": 307,
    }[family]
    rng = np.random.default_rng(RANDOM_SEED + family_seed)
    rows = []
    for _ in range(OUTER_REPEATS):
        for test in stratified_folds(frame.loc[dev_symbols, "rank"], rng, k=5):
            train = sorted(set(dev_symbols) - set(test))
            l1 = _choose_l1(frame, train, base_columns, family, rng)
            model = _fit_model(
                frame, train, base_columns, family, l1=l1
            )
            score = _score_model(model, frame, test, base_columns)
            metrics = compute_metrics(_ordinal_rank(frame, test), score)
            rows.append(
                {
                    "inv": metrics.inv,
                    "rho": metrics.rho,
                    "pair_err": metrics.pair_err,
                    "terms": len(model.columns),
                    "l1": l1,
                }
            )
    result = pd.DataFrame(rows)
    return {
        "splits": len(result),
        "inv_median": float(result["inv"].median()),
        "inv_standard_error": float(
            result["inv"].std(ddof=1) / np.sqrt(len(result))
        ),
        "rho_median": float(result["rho"].median()),
        "pair_err_median": float(result["pair_err"].median()),
        "terms_median": float(result["terms"].median()),
        "l1_mode": float(result["l1"].mode().iloc[0]),
        "raw": rows,
    }


def _bootstrap_stability(
    frame: pd.DataFrame,
    dev_symbols: List[str],
    base_columns: List[str],
    family: str,
    l1: float,
) -> dict:
    rng = np.random.default_rng(RANDOM_SEED + 31)
    selections: Dict[str, int] = {}
    weights: Dict[str, List[float]] = {}
    completed = 0
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(dev_symbols, size=len(dev_symbols), replace=True).tolist()
        bootstrap_frame = frame.loc[sample].copy()
        bootstrap_frame.index = [
            f"{symbol}__bootstrap_{i}" for i, symbol in enumerate(sample)
        ]
        bootstrap_symbols = bootstrap_frame.index.astype(str).tolist()
        model = _fit_model(
            bootstrap_frame,
            bootstrap_symbols,
            base_columns,
            family,
            l1=l1,
        )
        completed += 1
        for col, weight in zip(model.columns, model.weights):
            selections[col] = selections.get(col, 0) + 1
            weights.setdefault(col, []).append(weight)
    completed = max(1, completed)
    return {
        col: {
            "selection_rate": count / completed,
            "weight_median": float(np.median(weights[col])),
            "weight_p10": float(np.percentile(weights[col], 10)),
            "weight_p90": float(np.percentile(weights[col], 90)),
        }
        for col, count in sorted(
            selections.items(), key=lambda item: item[1], reverse=True
        )
    }


def _leave_one_out(
    frame: pd.DataFrame,
    dev_symbols: List[str],
    base_columns: List[str],
    family: str,
    l1: float,
) -> dict:
    full_model = _fit_model(
        frame, dev_symbols, base_columns, family, l1=l1
    )
    full_score = _score_model(
        full_model, frame, dev_symbols, base_columns
    )
    full_metrics = compute_metrics(
        _ordinal_rank(frame, dev_symbols), full_score
    )
    inv = []
    rho = []
    pair_err = []
    for omitted in dev_symbols:
        train = [s for s in dev_symbols if s != omitted]
        model = _fit_model(frame, train, base_columns, family, l1=l1)
        score = _score_model(model, frame, train, base_columns)
        metrics = compute_metrics(_ordinal_rank(frame, train), score)
        inv.append(metrics.inv)
        rho.append(metrics.rho)
        pair_err.append(metrics.pair_err)
    return {
        "full": asdict(full_metrics),
        "inv_range": [float(min(inv)), float(max(inv))],
        "rho_range": [float(min(rho)), float(max(rho))],
        "pair_err_range": [float(min(pair_err)), float(max(pair_err))],
    }


def _label_variants(session: dict) -> List[List[str]]:
    """Return all 2^k combinations for k inconsistent review pairs."""
    ranking = [str(s) for s in session.get("final_ranking") or []]
    inconsistencies = session.get("inconsistencies") or []
    variants: List[List[str]] = []
    for bits in itertools.product((0, 1), repeat=len(inconsistencies)):
        variant = ranking.copy()
        for bit, inc in zip(bits, inconsistencies):
            a, b = str(inc["a"]), str(inc["b"])
            winner = str(
                inc["first_choice"] if bit == 0 else inc["second_choice"]
            )
            ia, ib = variant.index(a), variant.index(b)
            if winner == b and ia < ib:
                variant[ia], variant[ib] = variant[ib], variant[ia]
            elif winner == a and ib < ia:
                variant[ib], variant[ia] = variant[ia], variant[ib]
        variants.append(variant)
    return variants


def _residual_pairs(
    frame: pd.DataFrame,
    score: pd.Series,
    symbols: Sequence[str],
    top_n: int = 30,
) -> pd.DataFrame:
    true_rank = frame.loc[list(symbols), "rank"]
    model_rank = pd.Series(
        {
            sym: i + 1
            for i, sym in enumerate(score.sort_values(ascending=False).index)
        }
    )
    rows = []
    ordered = list(symbols)
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            if (true_rank[left] < true_rank[right]) == (
                model_rank[left] < model_rank[right]
            ):
                continue
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "true_rank_left": int(true_rank[left]),
                    "true_rank_right": int(true_rank[right]),
                    "model_rank_left": int(model_rank[left]),
                    "model_rank_right": int(model_rank[right]),
                    "rank_gap": abs(
                        int(true_rank[left]) - int(true_rank[right])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "rank_gap", ascending=False
    ).head(top_n)


def _metric_dict(
    frame: pd.DataFrame, score: pd.Series, symbols: Sequence[str]
) -> dict:
    return asdict(
        compute_metrics(
            _ordinal_rank(frame, symbols), score.loc[list(symbols)]
        )
    )


def main() -> None:
    frame, dev_symbols, audit_symbols, session = _load_inputs()
    base_columns = _base_columns(frame)
    manifest = load_manifest()
    prices_path = os.path.join(
        SNAPSHOT_ROOT_DIR, manifest.snapshot_id, "prices_krw.pkl"
    )
    prices = pd.read_pickle(prices_path)
    production = compute_fms_snapshot(
        prices[manifest.ranking],
        reference_prices_krw=prices[manifest.ranking],
        ohlc_data=None,
        symbols=manifest.ranking,
    )["FMS"].reindex(frame.index)
    families = ("sparse_linear", "monotone_gam", "limited_interactions")

    nested = {
        family: _nested_validate_family(
            frame, dev_symbols, base_columns, family
        )
        for family in families
    }
    best_family = min(
        families, key=lambda family: nested[family]["inv_median"]
    )
    one_se_limit = (
        nested[best_family]["inv_median"]
        + nested[best_family]["inv_standard_error"]
    )
    complexity = {
        "sparse_linear": 0,
        "monotone_gam": 1,
        "limited_interactions": 2,
    }
    eligible = [
        family
        for family in families
        if nested[family]["inv_median"] <= one_se_limit
    ]
    winner = min(
        eligible,
        key=lambda family: (
            complexity[family],
            nested[family]["terms_median"],
            -nested[family]["rho_median"],
        ),
    )
    winner_l1 = float(nested[winner]["l1_mode"])
    model = _fit_model(
        frame, dev_symbols, base_columns, winner, l1=winner_l1
    )

    # Audit symbols are opened exactly once, after family/hyperparameter freeze.
    all_symbols = frame.index.astype(str).tolist()
    candidate = _score_model(model, frame, all_symbols, base_columns)
    residuals = _residual_pairs(frame, candidate, dev_symbols)
    residuals.to_csv(RESIDUAL_CSV, index=False, encoding="utf-8-sig")

    variants = _label_variants(session)
    variant_rows = []
    for variant in variants:
        variant_frame = frame.loc[variant].copy()
        variant_frame["rank"] = np.arange(1, len(variant_frame) + 1)
        variant_rows.append(
            {
                "baseline": asdict(
                    compute_metrics(variant_frame["rank"], production.loc[variant])
                ),
                "candidate": asdict(
                    compute_metrics(variant_frame["rank"], candidate.loc[variant])
                ),
            }
        )

    bootstrap = _bootstrap_stability(
        frame, dev_symbols, base_columns, winner, winner_l1
    )
    loo = _leave_one_out(
        frame, dev_symbols, base_columns, winner, winner_l1
    )

    score_frame = pd.DataFrame(
        {
            "true_rank": frame["rank"],
            "production_score": production,
            "scratch_score": candidate,
            "split": [
                "audit" if s in audit_symbols else "development"
                for s in frame.index
            ],
        }
    )
    score_frame["production_rank"] = production.rank(
        ascending=False, method="first"
    ).astype(int)
    score_frame["scratch_rank"] = candidate.rank(
        ascending=False, method="first"
    ).astype(int)
    score_frame.to_csv(SCORES_CSV, encoding="utf-8-sig")

    payload = {
        "status": "candidate_only_not_promoted",
        "policy": "true scratch score starts at zero; production is benchmark only",
        "session_id": load_manifest().session_id,
        "snapshot_id": load_manifest().snapshot_id,
        "n_symbols": len(frame),
        "families": {
            family: {
                key: value
                for key, value in result.items()
                if key != "raw"
            }
            for family, result in nested.items()
        },
        "winner": winner,
        "one_standard_error": {
            "best_raw_family": best_family,
            "inversion_limit": one_se_limit,
            "eligible_families": eligible,
            "selected_simplest": winner,
        },
        "model": asdict(model),
        "formula": " ".join(
            (
                "+" if FEATURE_DIRECTION.get(column.split("::")[0], 1) > 0
                else "-"
            )
            + f" {weight:.6f}*Z({column})"
            for column, weight in zip(model.columns, model.weights)
        ).lstrip("+ "),
        "effective_terms": [
            {
                "feature": column,
                "direction": FEATURE_DIRECTION.get(
                    column.split("::")[0], 1
                ),
                "weight": weight,
            }
            for column, weight in zip(model.columns, model.weights)
        ],
        "metrics": {
            "production": {
                "development": _metric_dict(
                    frame, production, dev_symbols
                ),
                "audit": _metric_dict(frame, production, audit_symbols),
                "full": _metric_dict(frame, production, all_symbols),
            },
            "scratch": {
                "development": _metric_dict(
                    frame, candidate, dev_symbols
                ),
                "audit": _metric_dict(frame, candidate, audit_symbols),
                "full": _metric_dict(frame, candidate, all_symbols),
            },
        },
        "top_quintile_recall": {
            "production": top_quintile_recall(frame["rank"], production),
            "scratch": top_quintile_recall(frame["rank"], candidate),
        },
        "bootstrap_stability": bootstrap,
        "leave_one_symbol_out": loo,
        "label_uncertainty": {
            "variants": len(variants),
            "strict_wins_vs_production": sum(
                strictly_improves(
                    Metrics(**row["candidate"]), Metrics(**row["baseline"])
                )
                for row in variant_rows
            ),
            "results": variant_rows,
        },
        "residual_pairs_csv": RESIDUAL_CSV,
        "scores_csv": SCORES_CSV,
        "production_modified": False,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "winner": winner,
        "production_full": payload["metrics"]["production"]["full"],
        "scratch_full": payload["metrics"]["scratch"]["full"],
        "audit_production": payload["metrics"]["production"]["audit"],
        "audit_scratch": payload["metrics"]["scratch"]["audit"],
        "label_variants": payload["label_uncertainty"]["variants"],
        "production_modified": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
