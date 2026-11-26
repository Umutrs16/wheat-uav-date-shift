"""Applicability, selective prediction, and a small target-date intercept.

Dissimilarity follows Meyer and Pebesma (2021): Euclidean distance in the
training-standardized predictor space, divided by the mean within-training
distance. The threshold is the upper Tukey fence of training DI values,
capped at the maximum training DI. Predictors are the five reflectance bands
with equal weight. Day of year is omitted here, because a new date would
otherwise fall outside the support by construction. A bands-plus-indices
version is reported beside it and is not used to choose the primary rule.

Weights and thresholds use only the training and calibration rows of each
frozen outer fold.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import regression_metrics
from wheat_ppqmtl.paths import configure_stdio, configs


BANDS = ["B", "G", "R", "RE", "NIR"]
BANDS_VI = BANDS + ["ndvi", "ndre", "gndvi", "cire", "rvi", "osavi", "evi2"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _ensemble(path: pd.DataFrame, role: str) -> pd.DataFrame:
    part = path[(path["role"] == role) & (path["target"] == "lcc")].copy()
    return part.groupby(["outer_fold", "sample_id"], as_index=False).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        y_std=("y_pred", "std"),
        date_key=("date_key", "first"),
    )


def _distances(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    # query (m, p), reference (n, p) -> (m, n)
    return np.sqrt(np.maximum(0.0, ((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)))


def _aoa_fold(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> dict:
    scaler = StandardScaler().fit(train[columns].to_numpy(float))
    x_train = scaler.transform(train[columns].to_numpy(float))
    x_test = scaler.transform(test[columns].to_numpy(float))
    dates = train["date_key"].to_numpy()
    full = _distances(x_train, x_train)
    np.fill_diagonal(full, np.nan)
    denom = float(np.nanmean(full))
    train_di = []
    for i, day in enumerate(dates):
        others = full[i, dates != day]
        train_di.append(np.nanmin(others) / denom if others.size else np.nan)
    train_di = np.asarray(train_di, float)
    fence = float(np.nanquantile(train_di, 0.75) + 1.5 * (np.nanquantile(train_di, 0.75) - np.nanquantile(train_di, 0.25)))
    threshold = float(min(fence, np.nanmax(train_di)))
    nearest = _distances(x_train, x_test).min(axis=1) / denom
    return {"di": nearest, "threshold": threshold, "inside": nearest <= threshold}


def _retained_rmse(y, pred, refuse_first: np.ndarray, fraction: float) -> float:
    n_keep = int(round(len(y) * fraction))
    n_keep = min(max(n_keep, 1), len(y))
    order = np.argsort(refuse_first)
    keep = order[:n_keep]
    return regression_metrics(y[keep], pred[keep])["rmse"]


def _error_captured(y, pred, refuse_first: np.ndarray, fraction: float) -> float:
    error = np.abs(y - pred)
    n_take = max(1, int(round(len(y) * fraction)))
    chosen = np.argsort(refuse_first)[-n_take:]
    return float(error[chosen].sum() / error.sum())


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(values[order][min(np.searchsorted(cumulative, quantile, side="left"), len(values) - 1)])


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")
    splits = read_table(root / "splits" / "lodo.csv")
    predictions = pd.read_csv(root / "outputs" / "predictions" / "spectral__lodo.csv")
    test = _ensemble(predictions, "test").merge(features, on="sample_id", suffixes=("", "_f"))
    calibration = _ensemble(predictions, "calibration").merge(features, on="sample_id", suffixes=("", "_f"))
    rng = np.random.default_rng(42)
    fold_rows = []
    coverage_rows = []
    few_rows = []
    for fold, held in test.groupby("outer_fold"):
        columns_keep = ["sample_id"] + BANDS_VI
        train = splits[(splits["outer_fold"] == fold) & (splits["role"] == "train")].merge(
            features[columns_keep], on="sample_id"
        )
        cal = calibration[calibration["outer_fold"] == fold]
        y = held["y_true"].to_numpy(float)
        pred = held["y_pred"].to_numpy(float)
        report = {"date": str(held["date_key"].iloc[0]), "n": int(len(held)), "rmse_all": regression_metrics(y, pred)["rmse"]}
        for name, columns in (("bands", BANDS), ("bands_vi", BANDS_VI)):
            aoa = _aoa_fold(train, held, columns)
            inside = aoa["inside"]
            report[name] = {
                "threshold": aoa["threshold"],
                "fraction_inside": float(inside.mean()),
                "rmse_inside": None if inside.sum() < 5 else regression_metrics(y[inside], pred[inside])["rmse"],
                "rmse_outside": None if (~inside).sum() < 5 else regression_metrics(y[~inside], pred[~inside])["rmse"],
                "retain_80_rmse": _retained_rmse(y, pred, aoa["di"], 0.8),
                "error_captured_at_20": _error_captured(y, pred, aoa["di"], 0.2),
            }
            if name == "bands":
                di = aoa["di"]
        sigma = held["y_std"].to_numpy(float)
        ndvi = -held["ndvi"].to_numpy(float)
        random_capture = []
        random_rmse = []
        for _ in range(200):
            rank = rng.random(len(held))
            random_capture.append(_error_captured(y, pred, rank, 0.2))
            random_rmse.append(_retained_rmse(y, pred, rank, 0.8))
        report["sigma_error_captured_at_20"] = _error_captured(y, pred, sigma, 0.2)
        report["low_ndvi_error_captured_at_20"] = _error_captured(y, pred, ndvi, 0.2)
        report["random_error_captured_at_20_mean"] = float(np.mean(random_capture))
        report["sigma_retain_80_rmse"] = _retained_rmse(y, pred, sigma, 0.8)
        report["random_retain_80_rmse_mean"] = float(np.mean(random_rmse))
        fold_rows.append(report)

        # Locally weighted absolute-residual interval. Bandwidth is the median
        # nearest-neighbour distance in the training bands, fixed before the test date.
        scaler = StandardScaler().fit(train[BANDS].to_numpy(float))
        x_train = scaler.transform(train[BANDS].to_numpy(float))
        train_dist = _distances(x_train, x_train)
        np.fill_diagonal(train_dist, np.nan)
        bandwidth = float(np.nanmedian(np.nanmin(train_dist, axis=1)))
        bandwidth = max(bandwidth, 1e-6)
        x_cal = scaler.transform(cal[BANDS].to_numpy(float))
        x_test = scaler.transform(held[BANDS].to_numpy(float))
        cal_score = np.abs(cal["y_true"].to_numpy(float) - cal["y_pred"].to_numpy(float))
        standard_q = float(np.quantile(cal_score, 0.9))
        dist = _distances(x_cal, x_test)
        weighted_q = np.array([
            _weighted_quantile(cal_score, np.exp(-(row ** 2) / (2 * bandwidth ** 2)), 0.9)
            for row in dist
        ])
        coverage_rows.append({
            "date": str(held["date_key"].iloc[0]),
            "n": int(len(held)),
            "standard_coverage": float(np.mean(np.abs(y - pred) <= standard_q)),
            "standard_width": float(2 * standard_q),
            "weighted_coverage": float(np.mean(np.abs(y - pred) <= weighted_q)),
            "weighted_width": float(np.mean(2 * weighted_q)),
        })

        # Intercept shift from a few plots of the held-out date. Scored on the other plots.
        for k in (5, 10, 20):
            if len(held) <= k + 5:
                continue
            scores = []
            base = []
            for draw in range(50):
                pick = rng.choice(len(held), size=k, replace=False)
                rest = np.setdiff1d(np.arange(len(held)), pick)
                bias = float(np.mean(y[pick] - pred[pick]))
                scores.append(regression_metrics(y[rest], pred[rest] + bias)["rmse"])
                base.append(regression_metrics(y[rest], pred[rest])["rmse"])
            few_rows.append({
                "date": str(held["date_key"].iloc[0]),
                "k": k,
                "n_evaluated": int(len(held) - k),
                "rmse_uncorrected_mean": float(np.mean(base)),
                "rmse_intercept_mean": float(np.mean(scores)),
            })

    pooled_y = test["y_true"].to_numpy(float)
    pooled_p = test["y_pred"].to_numpy(float)
    payload = {
        "note": (
            "AOA and weights are computed inside each leave-one-date-out training fold. "
            "Refusing the highest dissimilarity is compared with ensemble spread, low NDVI and random refusal. "
            "The few-plot intercept uses labels from the held-out date and is not a zero-shot result."
        ),
        "by_date": fold_rows,
        "conformal_absolute": coverage_rows,
        "few_shot_intercept": few_rows,
        "pooled_rmse": regression_metrics(pooled_y, pooled_p)["rmse"],
    }
    dest = root / "outputs" / "metrics" / "aoa_risk.json"
    dest.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
