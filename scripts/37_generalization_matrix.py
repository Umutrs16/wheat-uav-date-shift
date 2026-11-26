"""Model by validation-design matrix with plot-bootstrap intervals.

Every cell uses saved out-of-fold predictions; seeds are averaged per plot and
fold before scoring. The DOY-only and training-date-mean references are refitted
on the existing chronological (forward) splits, which were not regenerated.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.paths import configure_stdio, configs

SCHEMES = ["grouped_random", "spatial_block", "forward", "lodo"]
LEARNED = {
    "Spectral ensemble": "spectral",
    "PLSR": "plsr__bands_vi_doy",
    "GPR": "gpr__bands_vi_doy",
    "SVR": "svr__bands_vi_doy",
    "RF": "rf__bands_vi_doy",
    "XGBoost": "xgb__bands_vi_doy",
    "MLP": "mlp__bands_vi_doy",
    "Trend-residual": "trend_residual",
}
DRAWS = 2000


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


def _learned(root, stem: str, scheme: str, target: str) -> pd.DataFrame | None:
    path = root / "outputs" / "predictions" / f"{stem}__{scheme}.csv"
    if not path.exists():
        return None
    frame = read_table(path)
    frame = frame[frame["target"] == target]
    if "role" in frame.columns:
        frame = frame[frame["role"] == "test"]
    if target == "lai" and "lai_qc" in frame.columns:
        frame = frame[np.isclose(pd.to_numeric(frame["lai_qc"], errors="coerce"), 1.0)]
    return (frame.groupby(["outer_fold", "sample_id", "date_key"], as_index=False)
            .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))


def _refit_seasonal(root, features: pd.DataFrame, scheme: str) -> pd.DataFrame:
    splits = read_table(root / "splits" / f"{scheme}.csv")
    columns = ["doy_sin", "doy_cos"]
    rows = []
    for target in ("lcc", "lai"):
        for fold in sorted(splits["outer_fold"].unique()):
            merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_feat"))
            if target == "lai":
                merged = merged[np.isclose(pd.to_numeric(merged["lai_qc"], errors="coerce"), 1.0)]
            train = merged[merged["role"] == "train"].dropna(subset=columns + [target])
            test = merged[merged["role"] == "test"].dropna(subset=columns + [target])
            if len(train) < 8 or test.empty:
                continue
            if train["date_key"].nunique() < 2:
                doy = np.full(len(test), float(train[target].mean()))
            else:
                model = LinearRegression().fit(train[columns].to_numpy(float), train[target].to_numpy(float))
                doy = model.predict(test[columns].to_numpy(float))
            means = train.groupby("date_key")[target].mean()
            date_mean = test["date_key"].map(means).fillna(float(train[target].mean())).to_numpy(float)
            rows.append(pd.DataFrame({
                "scheme": scheme, "target": target, "outer_fold": int(fold), "sample_id": test["sample_id"].to_numpy(),
                "date_key": test["date_key"].to_numpy(), "y_true": test[target].to_numpy(float),
                "doy_only": doy, "training_date_mean": date_mean,
            }))
    return pd.concat(rows, ignore_index=True)


def _seasonal(root, features: pd.DataFrame, scheme: str, target: str) -> dict[str, pd.DataFrame]:
    cached = root / "outputs" / "predictions" / f"diagnostic_doy__{scheme}.csv"
    if not cached.exists():
        _refit_seasonal(root, features, scheme).to_csv(cached, index=False)
    frame = read_table(cached)
    frame = frame[frame["target"] == target]
    return {
        "DOY-only": frame.rename(columns={"doy_only": "y_pred"})[["outer_fold", "sample_id", "date_key", "y_true", "y_pred"]],
        "Training-date mean": frame.rename(columns={"training_date_mean": "y_pred"})[["outer_fold", "sample_id", "date_key", "y_true", "y_pred"]],
    }


def _score(frame: pd.DataFrame, rng) -> dict:
    y = frame["y_true"].to_numpy(float)
    pred = frame["y_pred"].to_numpy(float)
    resid = pred - y
    index = rng.integers(0, len(y), size=(DRAWS, len(y)))
    boot = np.sqrt(np.mean(resid[index] ** 2, axis=1))
    by_date = frame.assign(sq=resid ** 2).groupby("date_key")["sq"].mean().pow(0.5)
    date_bias = frame.assign(r=resid).groupby("date_key")["r"].transform("mean").to_numpy(float)
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "rmse_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "r2": float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)),
        "bias": float(resid.mean()),
        "within_date_rmse": float(np.sqrt(np.mean((resid - date_bias) ** 2))),
        "date_level_rmse": float(np.sqrt(np.mean(date_bias ** 2))),
        "macro_rmse": float(by_date.mean()),
        "by_date_rmse": {str(k): float(v) for k, v in by_date.items()},
        "by_date_bias": {str(k): float(v) for k, v in frame.assign(r=resid).groupby("date_key")["r"].mean().items()},
    }


def _paired(a: pd.DataFrame, b: pd.DataFrame, rng) -> dict:
    keys = ["outer_fold", "sample_id"]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    ea = (merged["y_pred_a"] - merged["y_true_a"]).to_numpy(float) ** 2
    eb = (merged["y_pred_b"] - merged["y_true_b"]).to_numpy(float) ** 2
    index = rng.integers(0, len(ea), size=(DRAWS, len(ea)))
    diff = np.sqrt(ea[index].mean(axis=1)) - np.sqrt(eb[index].mean(axis=1))
    return {
        "n": int(len(ea)),
        "delta_rmse": float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())),
        "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
    }


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")
    rng = np.random.default_rng(42)
    payload = {"note": __doc__.strip(), "draws": DRAWS, "targets": {}}
    for target in ("lcc", "lai"):
        cells = {}
        frames = {}
        for scheme in SCHEMES:
            members = {name: _learned(root, stem, scheme, target) for name, stem in LEARNED.items()}
            members.update(_seasonal(root, features, scheme, target))
            for name, frame in members.items():
                if frame is None or frame.empty:
                    continue
                frames[(name, scheme)] = frame
                cells.setdefault(name, {})[scheme] = _score(frame, rng)
        paired = {}
        for scheme in SCHEMES:
            base = frames.get(("Spectral ensemble", scheme))
            if base is None:
                continue
            for other in ("DOY-only", "PLSR", "Training-date mean", "Trend-residual"):
                if (other, scheme) in frames:
                    paired[f"spectral_minus_{other}_{scheme}"] = _paired(base, frames[(other, scheme)], rng)
        payload["targets"][target] = {"cells": cells, "paired": paired}
    dest = root / "outputs" / "metrics" / "generalization_matrix.json"
    dest.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    for target, block in payload["targets"].items():
        print(target)
        for name, row in block["cells"].items():
            print("  ", name.ljust(20), " ".join(f"{s}:{row[s]['rmse']:.2f}" if s in row else f"{s}:--" for s in SCHEMES))
        for key, value in block["paired"].items():
            print("   ", key, f"{value['delta_rmse']:.2f} [{value['ci95'][0]:.2f}, {value['ci95'][1]:.2f}] n={value['n']}")
    print(dest)


if __name__ == "__main__":
    main()
