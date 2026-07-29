"""Shared ranking metrics for FMS recalibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class Metrics:
    inv: float
    rho: float
    pair_err: float


def score_to_model_rank(score: pd.Series) -> pd.Series:
    order = score.sort_values(ascending=False).index.to_list()
    rank_map = {sym: i + 1 for i, sym in enumerate(order)}
    return pd.Series({sym: rank_map.get(sym, np.nan) for sym in score.index})


def compute_pairwise_rank_delta_error(true_rank: pd.Series, model_rank: pd.Series) -> float:
    df = pd.concat([true_rank, model_rank], axis=1).dropna()
    df.columns = ["true_rank", "model_rank"]
    n = len(df)
    if n <= 1:
        return 0.0
    df_sorted = df.sort_values("true_rank", ascending=True)
    r_true = df_sorted["true_rank"].to_numpy()
    r_model = df_sorted["model_rank"].to_numpy()
    total_err = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_err += abs((r_true[j] - r_true[i]) - (r_model[j] - r_model[i]))
            pairs += 1
    return total_err / pairs if pairs else 0.0


def pairwise_inversion_rate(true_rank: pd.Series, score: pd.Series) -> float:
    df = pd.concat([true_rank, score], axis=1).dropna()
    df.columns = ["true_rank", "score"]
    n = len(df)
    if n <= 1:
        return 0.0
    df_sorted = df.sort_values("true_rank", ascending=True)
    scores = df_sorted["score"].to_numpy()
    inv = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if scores[i] < scores[j]:
                inv += 1
    return inv / total if total else 0.0


def compute_metrics(true_rank: pd.Series, score: pd.Series) -> Metrics:
    inv = pairwise_inversion_rate(true_rank, score)
    model_rank = score_to_model_rank(score)
    common = pd.concat([true_rank, model_rank], axis=1).dropna()
    rho, _ = spearmanr(common.iloc[:, 0], common.iloc[:, 1])
    pair_err = compute_pairwise_rank_delta_error(true_rank, model_rank)
    return Metrics(inv=float(inv), rho=float(rho), pair_err=float(pair_err))


def compute_metrics_fast(true_rank: pd.Series, score: pd.Series) -> Metrics:
    common = pd.concat([true_rank, score], axis=1).dropna()
    common.columns = ["true_rank", "score"]
    common = common.sort_values("true_rank")
    return compute_metrics_arrays(
        common["true_rank"].to_numpy(dtype=float),
        common["score"].to_numpy(dtype=float),
    )


def compute_metrics_arrays(ranks: np.ndarray, scores: np.ndarray) -> Metrics:
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


def metric_key(metrics: Metrics) -> tuple[float, float, float]:
    return metrics.inv, -metrics.rho, metrics.pair_err


def strictly_improves(candidate: Metrics, baseline: Metrics) -> bool:
    return (
        candidate.inv < baseline.inv
        and candidate.rho > baseline.rho
        and candidate.pair_err < baseline.pair_err
    )


def stratified_folds(rank: pd.Series, rng: np.random.Generator, k: int = 5) -> list[list[str]]:
    ordered = rank.sort_values().index.to_numpy(dtype=str)
    buckets = np.array_split(ordered, k)
    folds: list[list[str]] = [[] for _ in range(k)]
    for bucket in buckets:
        shuffled = bucket.copy()
        rng.shuffle(shuffled)
        for i, symbol in enumerate(shuffled):
            folds[i % k].append(str(symbol))
    return folds


def top_quintile_recall(true_rank: pd.Series, score: pd.Series, *, quintile: float = 0.2) -> float:
    n = len(true_rank)
    k = max(1, int(round(n * quintile)))
    true_top = set(true_rank.sort_values().head(k).index.astype(str))
    pred_top = set(score.sort_values(ascending=False).head(k).index.astype(str))
    return len(true_top.intersection(pred_top)) / float(k)
