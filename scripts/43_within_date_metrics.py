"""Within-date agreement and spatially clustered intervals for the Gaocheng designs.

Within-date RMSE measures error after removing each date's mean bias; it does
not measure whether plots are ranked correctly. This script adds within-date
R2 (bias removed), Spearman rho, the ratio of predicted to observed SD, recall
of the lowest 20% of plots and overlap of the highest 20%.

Intervals resample 40 m spatial blocks with replacement inside each date, so
plots that share a block stay together. Method differences use the same
resampled blocks for both methods. 6 June is reported but excluded from the
within-date means because its LCC range is about 2 ug cm-2.

The random-order reference band for the risk-coverage curves (script 35) is
also computed here from repeated random retention orders.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.multilevel import (
    agreement,
    chance_overlap_band,
    grouped_agreement,
    jsonable,
    percentile_interval,
    stratified_cluster_draws,
)
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
REFERENCES = ("DOY-only", "Training-date mean")
PAIRED = ("DOY-only", "PLSR", "Training-date mean", "Trend-residual", "RF")
EXCLUDED = "2024.6.6"
DRAWS = 2000
RANDOM_ORDERS = 1000
COVERAGE = np.round(np.arange(1.0, 0.499, -0.05), 2)


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


def _seasonal(root, scheme: str, target: str) -> dict[str, pd.DataFrame]:
    frame = read_table(root / "outputs" / "predictions" / f"diagnostic_doy__{scheme}.csv")
    frame = frame[frame["target"] == target]
    cols = ["outer_fold", "sample_id", "date_key", "y_true", "y_pred"]
    return {
        "DOY-only": frame.rename(columns={"doy_only": "y_pred"})[cols],
        "Training-date mean": frame.rename(columns={"training_date_mean": "y_pred"})[cols],
    }


def _fast_stats(y, p, day, green_days):
    """RMSE parts and green-date means of within-date R2, rho and low-tail recall."""
    err = p - y
    out = {"rmse": float(np.sqrt(np.mean(err ** 2)))}
    date_bias = pd.Series(err).groupby(day).transform("mean").to_numpy()
    out["within_date_rmse"] = float(np.sqrt(np.mean((err - date_bias) ** 2)))
    out["date_level_rmse"] = float(np.sqrt(np.mean(date_bias ** 2)))
    r2, rho, recall = [], [], []
    for key in green_days:
        mask = day == key
        if mask.sum() < 5:
            continue
        yy, pp = y[mask], p[mask]
        yc = yy - yy.mean()
        ec = (pp - yy) - (pp - yy).mean()
        sst = float((yc ** 2).sum())
        r2.append(1.0 - float((ec ** 2).sum()) / sst if sst > 0 else np.nan)
        if pp.std() < 1e-12:
            rho.append(np.nan)
            recall.append(np.nan)
            continue
        ry, rp = rankdata(yy), rankdata(pp)
        rho.append(float(np.corrcoef(ry, rp)[0, 1]))
        m = max(1, int(round(0.2 * yy.size)))
        recall.append(len(set(np.argsort(yy, kind="stable")[:m]) & set(np.argsort(pp, kind="stable")[:m])) / m)
    out["r2_within_green"] = float(np.nanmean(r2)) if r2 else np.nan
    out["spearman_green"] = float(np.nanmean(rho)) if rho and np.isfinite(rho).any() else np.nan
    out["low_recall_green"] = float(np.nanmean(recall)) if recall and np.isfinite(recall).any() else np.nan
    return out


def _cell(payload: dict) -> tuple[tuple, dict]:
    frame = payload["frame"]
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    day = frame["date_key"].to_numpy()
    block = (frame["date_key"] + "|" + frame["block_id"].astype(str)).to_numpy()
    green_days = [d for d in pd.unique(day) if d != EXCLUDED]
    point = _fast_stats(y, p, day, green_days)
    grouped = grouped_agreement(y, p, day, min_n=5, exclude=(EXCLUDED,))
    boot = {key: [] for key in point}
    others = {name: {"frame": other, "diff": {k: [] for k in ("rmse", "within_date_rmse", "r2_within_green", "spearman_green")}}
              for name, other in payload["others"].items()}
    aligned = {}
    for name, other in payload["others"].items():
        merged = frame[["outer_fold", "sample_id"]].reset_index().merge(
            other[["outer_fold", "sample_id", "y_pred"]], on=["outer_fold", "sample_id"], how="left")
        aligned[name] = merged.sort_values("index")["y_pred"].to_numpy(float)
    for idx in stratified_cluster_draws(day, block, DRAWS, payload["seed"]):
        stats = _fast_stats(y[idx], p[idx], day[idx], green_days)
        for key, value in stats.items():
            boot[key].append(value)
        for name, other_pred in aligned.items():
            ok = np.isfinite(other_pred[idx])
            if ok.mean() < 0.99:
                continue
            other_stats = _fast_stats(y[idx], other_pred[idx], day[idx], green_days)
            for key in others[name]["diff"]:
                others[name]["diff"][key].append(stats[key] - other_stats[key])
    result = {
        "n": int(len(frame)),
        "point": point,
        "block_ci95": {key: percentile_interval(values) for key, values in boot.items()},
        "by_date": grouped["by_group"],
        "macro_green": grouped["macro"],
        "pooled_within_date": agreement(y - pd.Series(y).groupby(day).transform("mean").to_numpy(),
                                        p - pd.Series(p).groupby(day).transform("mean").to_numpy()),
        "paired": {},
    }
    for name, other_pred in aligned.items():
        diffs = others[name]["diff"]
        if not diffs["rmse"]:
            continue
        ok = np.isfinite(other_pred)
        other_point = _fast_stats(y[ok], other_pred[ok], day[ok], green_days)
        result["paired"][name] = {
            "n": int(ok.sum()),
            **{f"delta_{key}": point[key] - other_point[key] for key in diffs},
            **{f"delta_{key}_ci95": percentile_interval(values) for key, values in diffs.items()},
        }
    return payload["key"], result


def _random_band(frame: pd.DataFrame, group: str, y: str, pred: str, seed: int = 42) -> dict:
    data = frame.dropna(subset=[y, pred])
    rng = np.random.default_rng(seed)
    groups = [part[y].to_numpy(float) - part[pred].to_numpy(float) for _, part in data.groupby(group)]
    lo, hi, mid = [], [], []
    for c in COVERAGE:
        values = np.empty(RANDOM_ORDERS)
        for r in range(RANDOM_ORDERS):
            total_sq, total_n = 0.0, 0
            for err in groups:
                keep = max(1, int(round(len(err) * c)))
                chosen = rng.permutation(len(err))[:keep]
                total_sq += float(np.sum(err[chosen] ** 2))
                total_n += keep
            values[r] = np.sqrt(total_sq / total_n)
        lo.append(float(np.quantile(values, 0.025)))
        hi.append(float(np.quantile(values, 0.975)))
        mid.append(float(values.mean()))
    return {"coverage": COVERAGE.tolist(), "mean": mid, "lo95": lo, "hi95": hi, "orders": RANDOM_ORDERS}


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")[["sample_id", "block_id"]]
    tasks = []
    for target in ("lcc", "lai"):
        for scheme in SCHEMES:
            members = {name: _learned(root, stem, scheme, target) for name, stem in LEARNED.items()}
            members.update(_seasonal(root, scheme, target))
            members = {k: v.merge(features, on="sample_id") for k, v in members.items() if v is not None and not v.empty}
            for seed_offset, (name, frame) in enumerate(members.items()):
                others = {o: members[o] for o in PAIRED if name == "Spectral ensemble" and o in members}
                tasks.append({"key": (target, scheme, name), "frame": frame.reset_index(drop=True), "others": others,
                              "seed": 42})
    workers = max(2, min(14, (os.cpu_count() or 4) - 6))
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key, value in pool.map(_cell, tasks, chunksize=1):
            results[key] = value
            print(f"{key}: rmse {value['point']['rmse']:.3f} within {value['point']['within_date_rmse']:.3f} "
                  f"R2w {value['point']['r2_within_green']:.3f} rho {value['point']['spearman_green']:.3f} "
                  f"recall {value['point']['low_recall_green']:.3f}", flush=True)
    payload = {
        "note": " ".join(__doc__.split()),
        "draws": DRAWS,
        "excluded_from_within_date_means": EXCLUDED,
        "chance_low_recall_band": {},
        "targets": {},
        "risk_coverage_random_band": {},
    }
    for (target, scheme, name), value in results.items():
        payload["targets"].setdefault(target, {}).setdefault(scheme, {})[name] = value
    gao = pd.read_csv(root / "outputs" / "predictions" / "gaocheng_lodo_plot_diagnostics.csv")
    for day, part in gao.groupby("date_key"):
        payload["chance_low_recall_band"][str(day)] = {"n_lcc": int(len(part)), "lcc": chance_overlap_band(len(part)),
                                                       "n_lai": int(part["lai_observed"].sum()),
                                                       "lai": chance_overlap_band(int(part["lai_observed"].sum())) if part["lai_observed"].sum() >= 5 else None}
    bg = pd.read_csv(root / "outputs" / "predictions" / "bulgaria_plot_diagnostics.csv")
    payload["risk_coverage_random_band"]["gaocheng_lodo"] = {
        "lcc": _random_band(gao, "date_key", "y_true_lcc", "y_pred_lcc"),
        "lai": _random_band(gao[gao["lai_observed"]], "date_key", "y_true_lai", "y_pred_lai"),
    }
    payload["risk_coverage_random_band"]["knezha_zero_shot"] = {
        "lcc": _random_band(bg, "Campaign", "lcc", "y_pred_lcc"),
        "lai": _random_band(bg, "Campaign", "lai", "y_pred_lai"),
    }
    dest = root / "outputs" / "metrics" / "within_date_metrics.json"
    dest.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
