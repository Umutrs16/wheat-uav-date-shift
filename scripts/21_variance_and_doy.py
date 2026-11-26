"""Diagnostic baselines that do not replace the primary spectral model.

Date-mean RMSE uses the observed date mean, including the plots being scored.
It is a description of within-date scatter, not a prediction that can be deployed.
DOY-only and training-date-mean use only the training rows of each frozen fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.paths import configure_stdio, configs


SCHEMES = ("grouped_random", "spatial_block", "lodo")


def _variance(values: pd.Series, dates: pd.Series) -> dict:
    y = values.to_numpy(float)
    groups = dates.to_numpy()
    grand = float(y.mean())
    sst = float(np.sum((y - grand) ** 2))
    rows = []
    ssb = 0.0
    ssw = 0.0
    for day in pd.unique(groups):
        part = y[groups == day]
        mean = float(part.mean())
        ssb += len(part) * (mean - grand) ** 2
        ssw += float(np.sum((part - mean) ** 2))
        rows.append({
            "date": str(day),
            "n": int(len(part)),
            "mean": mean,
            "sd": float(part.std(ddof=1)) if len(part) > 1 else 0.0,
            "min": float(part.min()),
            "max": float(part.max()),
        })
    return {
        "n": int(len(y)),
        "grand_mean": grand,
        "sst": sst,
        "ss_between": float(ssb),
        "ss_within": float(ssw),
        "between_share": float(ssb / sst) if sst else None,
        "within_share": float(ssw / sst) if sst else None,
        "date_mean_descriptive_rmse": float(np.sqrt(ssw / len(y))),
        "by_date": rows,
    }


def _rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _r2(y, pred) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _ensemble(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["role"] == "test"]
    if "lai_qc" in frame.columns:
        keep = frame["target"].ne("lai") | np.isclose(pd.to_numeric(frame["lai_qc"], errors="coerce"), 1.0)
        frame = frame[keep]
    frame = frame[np.isfinite(frame["y_true"]) & np.isfinite(frame["y_pred"])]
    return frame.groupby(["sample_id", "target"], as_index=False).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        date_key=("date_key", "first"),
    )


def _within_report(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(float)
    pred = frame["y_pred"].to_numpy(float)
    dates = frame["date_key"].to_numpy()
    y_dm = np.empty_like(y)
    p_dm = np.empty_like(pred)
    by_date = []
    date_sse = 0.0
    for day in pd.unique(dates):
        mask = dates == day
        y_dm[mask] = y[mask] - y[mask].mean()
        p_dm[mask] = pred[mask] - pred[mask].mean()
        date_sse += len(y[mask]) * (y[mask].mean() - pred[mask].mean()) ** 2
        ss = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        rho = float("nan")
        if len(y[mask]) > 2 and np.nanstd(pred[mask]) > 1e-8 and np.nanstd(y[mask]) > 1e-8:
            rho = float(spearmanr(y[mask], pred[mask]).statistic)
        by_date.append({
            "date": str(day),
            "n": int(mask.sum()),
            "rmse": _rmse(y[mask], pred[mask]),
            "bias": float(pred[mask].mean() - y[mask].mean()),
            "within_rmse": _rmse(y_dm[mask], p_dm[mask]),
            "within_r2": None if ss <= 1e-8 else _r2(y[mask], pred[mask]),
            "spearman": None if not np.isfinite(rho) else float(rho),
            "observed_range": float(y[mask].max() - y[mask].min()),
        })
    return {
        "n": int(len(y)),
        "rmse": _rmse(y, pred),
        "r2": _r2(y, pred),
        "within_rmse": _rmse(y_dm, p_dm),
        "date_level_rmse": float(np.sqrt(date_sse / len(y))),
        "by_date": by_date,
    }


def _doy_and_date_mean(root: Path, features: pd.DataFrame) -> dict:
    out = {}
    columns = ["doy_sin", "doy_cos"]
    for scheme in SCHEMES:
        splits = read_table(root / "splits" / f"{scheme}.csv")
        scheme_rows = []
        for target in ("lcc", "lai"):
            for fold in sorted(splits["outer_fold"].unique()):
                fold_df = splits[splits["outer_fold"] == fold]
                merged = fold_df.merge(features, on="sample_id", suffixes=("", "_feat"))
                if target == "lai":
                    qc = pd.to_numeric(merged["lai_qc"], errors="coerce")
                    merged = merged[np.isclose(qc, 1.0)]
                train = merged[merged["role"] == "train"].dropna(subset=columns + [target])
                test = merged[merged["role"] == "test"].dropna(subset=columns + [target])
                if len(train) < 8 or len(test) == 0:
                    continue
                model = LinearRegression().fit(train[columns].to_numpy(float), train[target].to_numpy(float))
                pred = model.predict(test[columns].to_numpy(float))
                date_means = train.groupby("date_key")[target].mean()
                fallback = float(train[target].mean())
                date_pred = test["date_key"].map(date_means).fillna(fallback).to_numpy(float)
                for sample_id, date_key, y_true, y_doy, y_date in zip(
                    test["sample_id"], test["date_key"], test[target], pred, date_pred
                ):
                    scheme_rows.append({
                        "scheme": scheme,
                        "target": target,
                        "outer_fold": int(fold),
                        "sample_id": sample_id,
                        "date_key": date_key,
                        "y_true": float(y_true),
                        "doy_only": float(y_doy),
                        "training_date_mean": float(y_date),
                    })
        frame = pd.DataFrame(scheme_rows)
        summary = {}
        for target in ("lcc", "lai"):
            part = frame[frame["target"] == target]
            if part.empty:
                continue
            summary[target] = {
                "doy_only": _within_report(part.rename(columns={"doy_only": "y_pred"})),
                "training_date_mean": _within_report(part.rename(columns={"training_date_mean": "y_pred"})),
            }
        out[scheme] = summary
        dest = root / "outputs" / "predictions" / f"diagnostic_doy__{scheme}.csv"
        frame.to_csv(dest, index=False)
    return out


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    obs = read_table(root / "data" / "interim" / "observations_plot.csv")
    lcc = obs.dropna(subset=["lcc"])
    lai = obs[np.isclose(pd.to_numeric(obs["lai_qc"], errors="coerce"), 1.0)].dropna(subset=["lai"])
    payload = {
        "note": (
            "between_share is SS_between/SS_total from the raw observations. "
            "date_mean_descriptive_rmse uses each date's own observed mean and is not a deployable baseline. "
            "DOY-only is an ordinary linear regression on doy_sin and doy_cos, fit on the frozen training rows only. "
            "It does not replace the spectral model."
        ),
        "lcc": _variance(lcc["lcc"], lcc["date_key"]),
        "lai": _variance(lai["lai"], lai["date_key"]),
        "spectral_ensemble": {},
        "diagnostics": {},
    }
    pred_dir = root / "outputs" / "predictions"
    for scheme in SCHEMES:
        path = pred_dir / f"spectral__{scheme}.csv"
        if not path.exists():
            continue
        ens = _ensemble(path)
        payload["spectral_ensemble"][scheme] = {
            target: _within_report(ens[ens["target"] == target])
            for target in ("lcc", "lai")
        }
    features = read_table(root / "data" / "features" / "features.csv")
    payload["diagnostics"] = _doy_and_date_mean(root, features)
    dest = root / "outputs" / "metrics" / "variance_doy_diagnostic.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
