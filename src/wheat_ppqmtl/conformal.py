from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal quantile. `coverage` is 0.90 or 0.95."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = int(scores.size)
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * coverage))
    rank = min(max(rank, 1), n)
    ordered = np.sort(scores)
    return float(ordered[rank - 1])


def prediction_intervals(
    y_pred: np.ndarray,
    calibration_abs_residual: np.ndarray,
    sigma_test: np.ndarray | None = None,
    sigma_calibration: np.ndarray | None = None,
    levels: tuple[float, ...] = (0.90, 0.95),
) -> dict[str, np.ndarray]:
    """Absolute residual conformal, or normalized conformal when sigma is available."""
    y_pred = np.asarray(y_pred, dtype=float)
    out: dict[str, np.ndarray] = {}
    use_sigma = sigma_test is not None and sigma_calibration is not None
    if use_sigma:
        sigma_calibration = np.clip(np.asarray(sigma_calibration, dtype=float), 1e-6, None)
        sigma_test = np.clip(np.asarray(sigma_test, dtype=float), 1e-6, None)
        scores = np.asarray(calibration_abs_residual, dtype=float) / sigma_calibration
    else:
        scores = np.asarray(calibration_abs_residual, dtype=float)
    for level in levels:
        q = conformal_quantile(scores, level)
        tag = str(int(round(level * 100)))
        if use_sigma:
            half = q * sigma_test
        else:
            half = np.full_like(y_pred, q, dtype=float)
        out[f"y_lo{tag}"] = y_pred - half
        out[f"y_hi{tag}"] = y_pred + half
        out[f"q{tag}"] = np.full(y_pred.shape, q, dtype=float)
    return out
