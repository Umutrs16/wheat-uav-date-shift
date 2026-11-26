"""Equal-weight risk from spectral distance, PROSAIL residual and ensemble spread.

The three scores are standardized with the training rows of each leave-one-date-out
fold. Weights are fixed at one third and are not chosen from the held-out date.
The PROSAIL residual is the Euclidean distance to the existing literature-range
lookup table, not a new inversion.
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


def _min_distance(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    distances = np.sqrt(np.maximum(0.0, ((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)))
    return distances.min(axis=1)


def _retained(y, pred, score, fraction: float) -> float:
    keep = max(1, int(round(len(y) * fraction)))
    chosen = np.argsort(score)[:keep]
    return regression_metrics(y[chosen], pred[chosen])["rmse"]


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")
    splits = read_table(root / "splits" / "lodo.csv")
    predictions = pd.read_csv(root / "outputs" / "predictions" / "spectral__lodo.csv")
    test = predictions[(predictions["role"] == "test") & (predictions["target"] == "lcc")]
    ensemble = test.groupby(["outer_fold", "sample_id"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"), y_std=("y_pred", "std"), date_key=("date_key", "first")
    )
    library = pd.read_csv(root / "data" / "synthetic" / "prosail.csv", usecols=BANDS)
    if len(library) > 4000:
        library = library.sample(n=4000, random_state=42)
    library_x = library.to_numpy(float)
    rng = np.random.default_rng(42)
    rows = []
    for fold, held in ensemble.groupby("outer_fold"):
        train = splits[(splits["outer_fold"] == fold) & (splits["role"] == "train")].merge(
            features[["sample_id", *BANDS]], on="sample_id"
        )
        scaler = StandardScaler().fit(train[BANDS].to_numpy(float))
        x_train = scaler.transform(train[BANDS].to_numpy(float))
        x_test = scaler.transform(held.merge(features[["sample_id", *BANDS]], on="sample_id")[BANDS].to_numpy(float))
        x_lib = scaler.transform(library_x)
        train_full = np.sqrt(np.maximum(0.0, ((x_train[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)))
        np.fill_diagonal(train_full, np.nan)
        denom = float(np.nanmean(train_full))
        dates = train["date_key"].to_numpy() if "date_key" in train.columns else splits.loc[train.index, "date_key"].to_numpy()
        # date_key comes from the split side of the merge.
        if "date_key" not in train.columns:
            raise RuntimeError("date key missing")
        train_di = []
        for i, day in enumerate(train["date_key"].to_numpy()):
            others = train_full[i, train["date_key"].to_numpy() != day]
            train_di.append(np.nanmin(others) / denom)
        train_di = np.asarray(train_di, float)
        test_di = _min_distance(x_train, x_test) / denom
        train_e = _min_distance(x_lib, x_train)
        test_e = _min_distance(x_lib, x_test)
        train_pred = predictions[(predictions["outer_fold"] == fold) & (predictions["role"] == "calibration") & (predictions["target"] == "lcc")]
        train_u = train_pred.groupby("sample_id")["y_pred"].std().to_numpy(float)
        test_u = held["y_std"].to_numpy(float)
        pieces = []
        for train_values, test_values in ((train_di, test_di), (train_e, test_e), (train_u, test_u)):
            mu = np.nanmean(train_values)
            sd = np.nanstd(train_values)
            sd = sd if sd > 1e-8 else 1.0
            pieces.append((test_values - mu) / sd)
        risk = np.nanmean(np.vstack(pieces), axis=0)
        y = held["y_true"].to_numpy(float)
        pred = held["y_pred"].to_numpy(float)
        random_scores = [ _retained(y, pred, rng.random(len(y)), 0.8) for _ in range(200)]
        rows.append({
            "date": str(held["date_key"].iloc[0]),
            "n": int(len(held)),
            "rmse_all": regression_metrics(y, pred)["rmse"],
            "rmse_lowest_risk_80": _retained(y, pred, risk, 0.8),
            "rmse_lowest_spread_80": _retained(y, pred, test_u, 0.8),
            "rmse_random_80_mean": float(np.mean(random_scores)),
            "risk_error_spearman": float(pd.Series(risk).corr(pd.Series(np.abs(y - pred)), method="spearman")),
        })
    dest = root / "outputs" / "metrics" / "risk_score.json"
    dest.write_text(json.dumps(_jsonable({
        "note": "Risk is the mean of training-standardized dissimilarity, PROSAIL nearest-spectrum distance and ensemble standard deviation. Weights were not fit on the held-out date.",
        "by_date": rows,
    }), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
