from __future__ import annotations

import numpy as np


def _finite_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[ok], y_pred[ok]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y, yhat = _finite_pair(y_true, y_pred)
    n = int(y.size)
    if n == 0:
        return {"n": 0, "r2": np.nan, "rmse": np.nan, "rrmse": np.nan, "mae": np.nan, "bias": np.nan}
    resid = yhat - y
    sse = float(np.sum(resid ** 2))
    rmse = float(np.sqrt(sse / n))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / denom) if denom > 0 else np.nan
    mean_obs = float(np.mean(y))
    rrmse = float(rmse / abs(mean_obs) * 100.0) if mean_obs != 0 else np.nan
    return {"n": n, "r2": r2, "rmse": rmse, "rrmse": rrmse, "mae": mae, "bias": bias}


def optimism_gap(lodo: dict, grouped_random: dict) -> dict:
    """Positive RMSE gap means leave-one-date-out is harder than grouped random CV."""
    return {
        "rmse": float(lodo["rmse"] - grouped_random["rmse"]),
        "r2": float(grouped_random["r2"] - lodo["r2"]),
    }


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alphas: tuple[float, float] = (2.5, 97.5),
) -> dict:
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat, groups = y[ok], yhat[ok], groups[ok]
    point = regression_metrics(y, yhat)
    if y.size == 0:
        return {**point, "r2_ci95": [np.nan, np.nan], "rmse_ci95": [np.nan, np.nan]}
    uniq = np.unique(groups)
    buckets = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    r2s = []
    rmses = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([buckets[g] for g in chosen])
        stats = regression_metrics(y[idx], yhat[idx])
        if np.isfinite(stats["r2"]):
            r2s.append(stats["r2"])
        if np.isfinite(stats["rmse"]):
            rmses.append(stats["rmse"])
    def ci(values: list[float]) -> list[float]:
        if len(values) < 20:
            return [np.nan, np.nan]
        lo, hi = np.percentile(values, alphas)
        return [float(lo), float(hi)]
    point["r2_ci95"] = ci(r2s)
    point["rmse_ci95"] = ci(rmses)
    point["n_boot_r2"] = len(r2s)
    return point


def paired_rmse_difference(
    y_true: np.ndarray,
    pred_reference: np.ndarray,
    pred_alternative: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """RMSE(alternative) minus RMSE(reference) on the same rows.

    A spatial-block bootstrap resamples clusters with replacement. The interval
    is conditional on the dates present in the rows.
    """
    y = np.asarray(y_true, dtype=float)
    reference = np.asarray(pred_reference, dtype=float)
    alternative = np.asarray(pred_alternative, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(y) & np.isfinite(reference) & np.isfinite(alternative)
    y, reference, alternative, groups = y[ok], reference[ok], alternative[ok], groups[ok]

    def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        if actual.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))

    point = _rmse(y, alternative) - _rmse(y, reference)
    if y.size == 0:
        return {"mean": point, "ci95": [float("nan"), float("nan")], "n": 0, "n_clusters": 0}
    uniq = np.unique(groups)
    buckets = {group: np.flatnonzero(groups == group) for group in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([buckets[group] for group in chosen])
        draws.append(_rmse(y[idx], alternative[idx]) - _rmse(y[idx], reference[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": point,
        "ci95": [float(lo), float(hi)],
        "n": int(y.size),
        "n_clusters": int(uniq.size),
        "rmse_reference": _rmse(y, reference),
        "rmse_alternative": _rmse(y, alternative),
    }


def paired_error_gap(delta: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Mean of |error_lodo| - |error_grouped|. Positive means date hold-out is worse."""
    delta = np.asarray(delta, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(delta)
    delta, groups = delta[ok], groups[ok]
    point = float(np.mean(delta)) if delta.size else float("nan")
    if delta.size == 0:
        return {"mean": point, "ci95": [float("nan"), float("nan")], "n": 0}
    uniq = np.unique(groups)
    buckets = {group: np.flatnonzero(groups == group) for group in uniq}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([buckets[group] for group in chosen])
        means.append(float(np.mean(delta[idx])))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": point, "ci95": [float(lo), float(hi)], "n": int(delta.size), "n_clusters": int(uniq.size)}


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    ok = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(ok):
        return float("nan")
    y, lower, upper = y[ok], lower[ok], upper[ok]
    width = upper - lower
    below = (lower - y) * (y < lower)
    above = (y - upper) * (y > upper)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


def coverage_and_width(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    ok = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(ok):
        return {"n": 0, "picp": np.nan, "mpiw": np.nan}
    y, lower, upper = y[ok], lower[ok], upper[ok]
    inside = (y >= lower) & (y <= upper)
    return {"n": int(y.size), "picp": float(np.mean(inside)), "mpiw": float(np.mean(upper - lower))}
