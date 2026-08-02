# -*- coding: utf-8 -*-
"""Monte Carlo nonlinear scratch FMS refit (primary path since 2026-08-02).

Workflow:
1. Load latest completed session features/manifest (production = benchmark only).
2. Score competing nonlinear formula families with Monte Carlo parameter search.
3. Nested holdout + label-variant checks on development; audit once at the end.
4. Write candidate JSON/CSV. Do NOT promote to production here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from calibration.manifest import load_manifest
from calibration.nonlinear_formulas import FORMULA_FAMILIES, FormulaFamily, family_by_name
from calibration.ranking_metrics import (
    Metrics,
    compute_metrics,
    compute_metrics_fast,
    stratified_folds,
    strictly_improves,
    top_quintile_recall,
)
from calibration.session import latest_completed_session
from core.fms import score_fms_from_feature_frame

FEATURE_CSV = "fms_recalib_features.csv"
OUT_JSON = "fms_recalib_scratch_candidate.json"
SCORES_CSV = "fms_recalib_scratch_scores.csv"
RESIDUAL_CSV = "fms_recalib_scratch_residual_pairs.csv"
NL_RULES_JSON = "fms_recalib_natural_language_rules.json"
RANDOM_SEED = 20260802
MC_SAMPLES_PER_FAMILY = 2500
OUTER_REPEATS = 3
LOCAL_REFINE = 250
NESTED_REFIT_SAMPLES = 180
MULTI_SEEDS = (20260802, 20260803, 20260811)


@dataclass(frozen=True)
class CandidateResult:
    family: str
    params: Dict[str, float]
    metrics_full: Metrics
    metrics_dev: Metrics
    metrics_audit: Metrics
    nested_inv_mean: float
    nested_inv_std: float
    label_variant_wins: int
    label_variant_total: int


def _load_inputs() -> tuple[pd.DataFrame, List[str], List[str], dict]:
    if not os.path.exists(FEATURE_CSV):
        raise FileNotFoundError(
            f"{FEATURE_CSV} missing — run `python fms_recalib_build_features.py` first"
        )
    manifest = load_manifest()
    frame = pd.read_csv(FEATURE_CSV, index_col=0)
    frame = frame.loc[manifest.ranking].copy()
    frame["rank"] = np.arange(1, len(frame) + 1)
    _, session = latest_completed_session()
    return frame, list(manifest.development_symbols), list(manifest.audit_symbols), session


def _ordinal_rank(frame: pd.DataFrame, symbols: Sequence[str]) -> pd.Series:
    original = frame.loc[list(symbols), "rank"].sort_values()
    return pd.Series(
        np.arange(1, len(original) + 1, dtype=float),
        index=original.index,
        name="rank",
    ).reindex(list(symbols))


def _score_family(
    family: FormulaFamily, frame: pd.DataFrame, params: Dict[str, float]
) -> pd.Series:
    raw = family.score(frame, params)
    return raw.reindex(frame.index).astype(float).fillna(raw.median())


def _utility(m: Metrics) -> float:
    """Scalar utility for MC search: lower inversion, higher rho, lower pair err."""
    return (-m.inv) * 10.0 + m.rho * 3.0 - m.pair_err * 0.05


def _search_family(
    family: FormulaFamily,
    frame: pd.DataFrame,
    symbols: Sequence[str],
    *,
    rng: np.random.Generator,
    n_samples: int,
    local_refine: Optional[int] = None,
) -> tuple[Dict[str, float], Metrics, float]:
    truth = _ordinal_rank(frame, symbols)
    best_params: Optional[Dict[str, float]] = None
    best_metrics: Optional[Metrics] = None
    best_u = -1e18
    sub = frame.loc[list(symbols)]
    refine_n = LOCAL_REFINE if local_refine is None else int(local_refine)
    for _ in range(n_samples):
        params = family.sample_params(rng)
        scores = _score_family(family, sub, params)
        metrics = compute_metrics_fast(truth, scores)
        u = _utility(metrics)
        if u > best_u:
            best_u = u
            best_params = params
            best_metrics = metrics
    assert best_params is not None and best_metrics is not None

    # Local refine: jitter around best.
    for _ in range(refine_n):
        jittered = {}
        for key, value in best_params.items():
            scale = 0.15 if abs(value) > 1e-8 else 0.05
            jittered[key] = float(value * float(np.exp(rng.normal(0.0, scale))))
            if key.endswith("floor") or "edge" in key or key == "eps":
                jittered[key] = float(np.clip(jittered[key], 1e-4, 0.5))
        scores = _score_family(family, sub, jittered)
        metrics = compute_metrics_fast(truth, scores)
        u = _utility(metrics)
        if u > best_u:
            best_u = u
            best_params = jittered
            best_metrics = metrics
    return best_params, best_metrics, best_u


def _nested_inv(
    family: FormulaFamily,
    frame: pd.DataFrame,
    symbols: Sequence[str],
    params: Dict[str, float],
    *,
    rng: np.random.Generator,
    refit: bool = True,
) -> tuple[float, float]:
    """Nested holdout inversion; optionally re-fit params inside each train fold."""
    invs: List[float] = []
    for _ in range(OUTER_REPEATS):
        for test in stratified_folds(frame.loc[list(symbols), "rank"], rng, k=5):
            train = [s for s in symbols if s not in set(test)]
            if refit and len(train) >= 20:
                fold_params, _, _ = _search_family(
                    family,
                    frame,
                    train,
                    rng=rng,
                    n_samples=NESTED_REFIT_SAMPLES,
                    local_refine=40,
                )
            else:
                fold_params = params
            truth = _ordinal_rank(frame, test)
            scores = _score_family(family, frame.loc[test], fold_params)
            invs.append(compute_metrics_fast(truth, scores).inv)
    arr = np.asarray(invs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _label_variants(
    session: dict, ranking: Sequence[str]
) -> List[List[str]]:
    """Rebuild rankings for each inconsistency first/second choice combination."""
    inconsistencies = list(session.get("inconsistencies") or [])
    if not inconsistencies:
        return [list(ranking)]
    # Cap at 5 inconsistencies → 32 variants.
    inconsistencies = inconsistencies[:5]
    base = list(ranking)
    variants = [base]
    for mask in range(1, 2 ** len(inconsistencies)):
        order = list(base)
        for bit, item in enumerate(inconsistencies):
            use_second = bool(mask & (1 << bit))
            preferred = item.get("second_choice" if use_second else "first_choice")
            a = item.get("a")
            b = item.get("b")
            if preferred not in (a, b) or a not in order or b not in order:
                continue
            ia, ib = order.index(a), order.index(b)
            winner, loser = (a, b) if preferred == a else (b, a)
            # Keep relative order: winner should appear before loser (better rank).
            if order.index(winner) > order.index(loser):
                i_w, i_l = order.index(winner), order.index(loser)
                order[i_w], order[i_l] = order[i_l], order[i_w]
        variants.append(order)
    return variants


def _metrics_on_ranking(
    family: FormulaFamily,
    frame: pd.DataFrame,
    ranking: Sequence[str],
    params: Dict[str, float],
    symbols: Sequence[str],
) -> Metrics:
    rank_map = {sym: i + 1 for i, sym in enumerate(ranking)}
    truth = pd.Series({s: float(rank_map[s]) for s in symbols if s in rank_map})
    # Re-sequence subset to 1..n
    truth = truth.sort_values()
    truth = pd.Series(
        np.arange(1, len(truth) + 1, dtype=float), index=truth.index
    )
    scores = _score_family(family, frame.loc[truth.index], params)
    return compute_metrics_fast(truth, scores)


def _largest_residual_pairs(
    true_rank: pd.Series, score: pd.Series, *, top_n: int = 25
) -> pd.DataFrame:
    model_rank = score.rank(ascending=False, method="average")
    gap = (model_rank - true_rank).abs().sort_values(ascending=False)
    rows = []
    for sym in gap.head(top_n).index:
        rows.append(
            {
                "symbol": sym,
                "true_rank": int(true_rank.loc[sym]),
                "model_rank": float(model_rank.loc[sym]),
                "score": float(score.loc[sym]),
                "abs_rank_gap": float(gap.loc[sym]),
            }
        )
    return pd.DataFrame(rows)


def run_refit(
    *,
    mc_samples: int = MC_SAMPLES_PER_FAMILY,
    seed: int = RANDOM_SEED,
    seeds: Sequence[int] = MULTI_SEEDS,
    nested_refit: bool = True,
) -> dict:
    frame, dev_symbols, audit_symbols, session = _load_inputs()
    rng = np.random.default_rng(seed)

    baseline = score_fms_from_feature_frame(frame)
    base_full = compute_metrics(frame["rank"], baseline)
    base_dev = compute_metrics(_ordinal_rank(frame, dev_symbols), baseline.loc[dev_symbols])
    base_audit = compute_metrics(
        _ordinal_rank(frame, audit_symbols), baseline.loc[audit_symbols]
    )

    # Multi-seed search: keep best params per family by development utility.
    family_best: Dict[str, tuple[Dict[str, float], Metrics, float]] = {}
    for family in FORMULA_FAMILIES:
        best_local = None
        for s in seeds:
            # Deterministic family offset (avoid PYTHONHASHSEED-unstable hash()).
            fam_off = sum(ord(ch) for ch in family.name) * 97
            local_rng = np.random.default_rng(int(s) + fam_off)
            params, dev_metrics, u = _search_family(
                family, frame, dev_symbols, rng=local_rng, n_samples=mc_samples
            )
            if best_local is None or u > best_local[2]:
                best_local = (params, dev_metrics, u)
        assert best_local is not None
        family_best[family.name] = best_local
        print(
            f"[mc] {family.name}: best_dev_inv={best_local[1].inv:.4f} "
            f"rho={best_local[1].rho:.4f} utility={best_local[2]:.4f}"
        )

    family_reports = []
    candidates: List[CandidateResult] = []

    for family in FORMULA_FAMILIES:
        params, _, _ = family_best[family.name]
        full_scores = _score_family(family, frame, params)
        metrics_full = compute_metrics(frame["rank"], full_scores)
        metrics_dev = compute_metrics(
            _ordinal_rank(frame, dev_symbols), full_scores.loc[dev_symbols]
        )
        metrics_audit = compute_metrics(
            _ordinal_rank(frame, audit_symbols), full_scores.loc[audit_symbols]
        )
        nested_mean, nested_std = _nested_inv(
            family,
            frame,
            dev_symbols,
            params,
            rng=rng,
            refit=nested_refit,
        )

        variants = _label_variants(session, list(frame.index))
        wins = 0
        for ranking in variants:
            vm = _metrics_on_ranking(family, frame, ranking, params, dev_symbols)
            rank_map = {sym: i + 1 for i, sym in enumerate(ranking)}
            truth = pd.Series(
                {s: float(rank_map[s]) for s in dev_symbols if s in rank_map}
            ).sort_values()
            truth = pd.Series(np.arange(1, len(truth) + 1, dtype=float), index=truth.index)
            bm = compute_metrics_fast(truth, baseline.loc[truth.index])
            if strictly_improves(vm, bm):
                wins += 1

        # Residual health: mean |gap| on true top quintile should not explode.
        model_rank = full_scores.rank(ascending=False, method="average")
        top_mask = frame["rank"] <= max(1, int(round(len(frame) * 0.2)))
        top_gap = float((model_rank[top_mask] - frame.loc[top_mask, "rank"]).abs().mean())

        cand = CandidateResult(
            family=family.name,
            params=params,
            metrics_full=metrics_full,
            metrics_dev=metrics_dev,
            metrics_audit=metrics_audit,
            nested_inv_mean=nested_mean,
            nested_inv_std=nested_std,
            label_variant_wins=wins,
            label_variant_total=len(variants),
        )
        candidates.append(cand)
        family_reports.append(
            {
                "family": family.name,
                "natural_language": family.natural_language,
                "params": params,
                "metrics_full": asdict(metrics_full),
                "metrics_dev": asdict(metrics_dev),
                "metrics_audit": asdict(metrics_audit),
                "nested_inv_mean": nested_mean,
                "nested_inv_std": nested_std,
                "nested_refit": nested_refit,
                "label_variant_wins": wins,
                "label_variant_total": len(variants),
                "top_quintile_mean_abs_rank_gap": top_gap,
                "improves_vs_production_full": strictly_improves(metrics_full, base_full),
                "improves_vs_production_dev": strictly_improves(metrics_dev, base_dev),
                "improves_vs_production_audit": strictly_improves(metrics_audit, base_audit),
                "top_quintile_recall_full": top_quintile_recall(frame["rank"], full_scores),
            }
        )

    # Prefer lower full inversion within a soft nested band (0.05).
    # Nested CV on ~120 symbols is noisy; band avoids vetoing strong full/dev fits.
    best_nested = min(c.nested_inv_mean for c in candidates)
    nested_band = 0.05
    eligible = [c for c in candidates if c.nested_inv_mean <= best_nested + nested_band]
    best = min(
        eligible,
        key=lambda c: (c.metrics_full.inv, -c.metrics_full.rho, c.nested_inv_mean),
    )
    winner_family = family_by_name(best.family)
    winner_scores = _score_family(winner_family, frame, best.params)
    residuals = _largest_residual_pairs(frame["rank"], winner_scores)

    nl_rules = []
    if os.path.exists(NL_RULES_JSON):
        with open(NL_RULES_JSON, "r", encoding="utf-8") as f:
            nl_rules = json.load(f)

    # Optional note: full sparse L-BFGS compare is a separate CLI.
    legacy_compare = {
        "status": "deferred",
        "command": "python fms_recalib_refit.py",
        "reason": "legacy sparse path is comparison-only; run separately after MC winner",
    }

    payload = {
        "status": "candidate_only_not_promoted",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "nonlinear_monte_carlo_round2",
        "session_id": session.get("session_id"),
        "snapshot_id": session.get("snapshot_id"),
        "n_symbols": int(len(frame)),
        "seed": seed,
        "seeds": list(seeds),
        "mc_samples_per_family": mc_samples,
        "nested_refit_samples": NESTED_REFIT_SAMPLES if nested_refit else 0,
        "natural_language_rules_path": NL_RULES_JSON if nl_rules else None,
        "natural_language_rules": nl_rules,
        "production_benchmark": {
            "full": asdict(base_full),
            "development": asdict(base_dev),
            "audit": asdict(base_audit),
        },
        "legacy_sparse_compare": legacy_compare,
        "selected": {
            "family": best.family,
            "natural_language": winner_family.natural_language,
            "params": best.params,
            "metrics_full": asdict(best.metrics_full),
            "metrics_dev": asdict(best.metrics_dev),
            "metrics_audit": asdict(best.metrics_audit),
            "nested_inv_mean": best.nested_inv_mean,
            "nested_inv_std": best.nested_inv_std,
            "label_variant_wins": best.label_variant_wins,
            "label_variant_total": best.label_variant_total,
            "formula_note": (
                "Round-2 nonlinear MC with residual features "
                "(MID_DIP_RECOVERY, STALE_AFTER_RUN, RECENT_JUMP_SHARE_5D), "
                "multi-seed search, nested re-fit. Not production."
            ),
        },
        "all_families": family_reports,
        "notes": [
            "Round 2: residual-driven features/families + deeper MC + nested refit.",
            "Production FMS was benchmark-only; no asset-class exception rules.",
            "Do not promote until user approval.",
        ],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    out_scores = pd.DataFrame(
        {
            "rank": frame["rank"],
            "production_fms": baseline,
            "candidate_score": winner_scores,
        },
        index=frame.index,
    )
    out_scores.to_csv(SCORES_CSV, encoding="utf-8-sig")
    residuals.to_csv(RESIDUAL_CSV, index=False, encoding="utf-8-sig")
    print("Wrote", OUT_JSON)
    print("Wrote", SCORES_CSV)
    print("Wrote", RESIDUAL_CSV)
    print(
        "selected",
        best.family,
        "full",
        asdict(best.metrics_full),
        "nested",
        best.nested_inv_mean,
        "vs production",
        asdict(base_full),
    )
    return payload


def main() -> None:
    run_refit()


if __name__ == "__main__":
    main()
