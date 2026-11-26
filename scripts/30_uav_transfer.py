"""Cross-campaign and cross-sensor tests on extracted Sequoia plot reflectance.

Gaocheng predictors are the green, red, red-edge and near-infrared camera bands.
Bulgarian predictors are the Sequoia bands extracted from the mosaics. The band
centers differ. PLS components are chosen on source tune rows only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _path  # noqa: F401
from wheat_ppqmtl.baselines import _fit_classical, _predict
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import regression_metrics
from wheat_ppqmtl.paths import configure_stdio, configs

COLUMNS = ["G", "R", "RE", "NIR"]
EXP = {"baselines": {"plsr_components": [1, 2, 3]}}


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


def _within(y, pred, groups) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    groups = np.asarray(groups)
    stats = regression_metrics(y, pred)
    y_dm = np.empty_like(y)
    p_dm = np.empty_like(pred)
    scored = np.zeros(len(y), dtype=bool)
    by = []
    for key in pd.unique(groups):
        mask = groups == key
        if mask.sum() < 5:
            continue
        scored[mask] = True
        y_dm[mask] = y[mask] - y[mask].mean()
        p_dm[mask] = pred[mask] - pred[mask].mean()
        by.append({
            "group": str(key),
            "n": int(mask.sum()),
            **regression_metrics(y[mask], pred[mask]),
            "within_rmse": float(np.sqrt(np.mean((y_dm[mask] - p_dm[mask]) ** 2))),
        })
    stats["within_rmse"] = float(np.sqrt(np.mean((y_dm[scored] - p_dm[scored]) ** 2))) if scored.any() else None
    stats["by_group"] = by
    return stats


def _predict_external(train, tune, test, target):
    model, bundle = _fit_classical(
        "plsr", train[COLUMNS].to_numpy(float), train[target].to_numpy(float),
        tune[COLUMNS].to_numpy(float), tune[target].to_numpy(float), 42, EXP,
    )
    pred, _ = _predict("plsr", model, bundle, test[COLUMNS].to_numpy(float))
    return pred


def _holdout(frame: pd.DataFrame, group_col: str, target: str) -> dict:
    preds = np.full(len(frame), np.nan)
    rows = []
    for key, index in frame.groupby(group_col).groups.items():
        test_i = np.array(list(index))
        train_i = np.setdiff1d(np.arange(len(frame)), test_i)
        train = frame.iloc[train_i].dropna(subset=COLUMNS + [target])
        test = frame.iloc[test_i].dropna(subset=COLUMNS + [target])
        if len(train) < 12 or len(test) < 5:
            continue
        tune = train.sample(n=max(5, int(round(0.15 * len(train)))), random_state=42)
        fit = train.drop(index=tune.index)
        pred = _predict_external(fit, tune, test, target)
        preds[frame.index.get_indexer(test.index)] = pred
        rows.append({"group": str(key), "n": int(len(test)), **regression_metrics(test[target], pred)})
    ok = np.isfinite(preds)
    pooled = _within(frame.loc[frame.index[ok], target], preds[ok], frame.loc[frame.index[ok], group_col].to_numpy()) if ok.any() else {}
    if ok.any():
        pooled["rmse_ci95"] = _bootstrap_rmse(frame.loc[frame.index[ok], target].to_numpy(float), preds[ok])
    return {"by_group": rows, "pooled": pooled, "_pred": preds}


def _bootstrap_rmse(y, pred, draws: int = 2000) -> list[float]:
    rng = np.random.default_rng(42)
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    values = []
    for _ in range(draws):
        index = rng.integers(0, len(y), len(y))
        values.append(np.sqrt(np.mean((y[index] - pred[index]) ** 2)))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _gaocheng_ensemble(root, features, bulgaria, target):
    splits = read_table(root / "splits" / "grouped_random.csv")
    column = target
    preds = []
    for fold in sorted(splits["outer_fold"].unique()):
        merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
        if target == "lai":
            merged = merged[np.isclose(pd.to_numeric(merged["lai_qc"], errors="coerce"), 1.0)]
        train = merged[merged["role"] == "train"].dropna(subset=COLUMNS + [column])
        tune = merged[merged["role"] == "tune"].dropna(subset=COLUMNS + [column])
        preds.append(_predict_external(train, tune, bulgaria, target))
    return np.mean(np.vstack(preds), axis=0)


def _few_shot(y, pred, groups):
    rng = np.random.default_rng(42)
    rows = []
    for key in pd.unique(groups):
        mask = np.flatnonzero(np.asarray(groups) == key)
        for k in (5, 10):
            if len(mask) <= k + 5:
                continue
            base, fixed = [], []
            for _ in range(30):
                pick = rng.choice(mask, size=k, replace=False)
                rest = np.setdiff1d(mask, pick)
                bias = float(np.mean(y[pick] - pred[pick]))
                base.append(regression_metrics(y[rest], pred[rest])["rmse"])
                fixed.append(regression_metrics(y[rest], pred[rest] + bias)["rmse"])
            rows.append({
                "group": str(key), "k": k, "n_evaluated": int(len(mask) - k),
                "rmse_uncorrected_mean": float(np.mean(base)),
                "rmse_intercept_mean": float(np.mean(fixed)),
            })
    return rows


def _variance(values, groups) -> dict:
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    grand = values.mean()
    sst = float(np.sum((values - grand) ** 2))
    ssb = sum(len(values[groups == key]) * (values[groups == key].mean() - grand) ** 2 for key in pd.unique(groups))
    return {"n": int(len(values)), "between_share": float(ssb / sst) if sst else None}


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    uav = pd.read_csv(root / "data" / "external" / "bulgaria" / "uav_esu_reflectance.csv")
    scale = float(np.nanmedian(uav["NIR"]))
    factor = 1.0
    if scale > 100:
        factor = 10000.0
    elif scale > 2:
        factor = 255.0
    for column in COLUMNS:
        uav[column] = pd.to_numeric(uav[column], errors="coerce") / factor
    uav["year"] = np.where(uav["Campaign"].isin(["November_2017", "April_2018"]), "2017-2018", "2016-2017")
    features = read_table(root / "data" / "features" / "features.csv")
    payload = {
        "note": (
            "Bulgarian bands are 20 m means from Sequoia mosaics. Gaocheng bands are the "
            "DJI camera. Reflectance was divided by the recorded scale factor when the "
            "mosaic values were not already on a 0-1 scale."
        ),
        "n": int(len(uav)),
        "reflectance_divisor": factor,
        "band_median": {column: float(uav[column].median()) for column in COLUMNS},
        "campaigns": uav.groupby("Campaign").size().astype(int).to_dict(),
        "targets": {},
    }
    saved = []
    for target in ("lcc", "lai"):
        part = uav.dropna(subset=COLUMNS + [target]).reset_index(drop=True)
        if len(part) < 20:
            payload["targets"][target] = {"n": int(len(part))}
            continue
        pred = _gaocheng_ensemble(root, features, part, target)
        zero = _within(part[target], pred, part["Campaign"].to_numpy())
        zero["rmse_ci95"] = _bootstrap_rmse(part[target].to_numpy(float), pred)
        designs = {
            "leave_campaign_out": _holdout(part, "Campaign", target),
            "leave_field_out": _holdout(part, "field_id", target),
            "leave_year_out": _holdout(part, "year", target),
        }
        for name, values in [("zero_shot_from_gaocheng", pred)] + [(key, value.pop("_pred")) for key, value in designs.items()]:
            saved.append(pd.DataFrame({
                "design": name, "target": target, "ESU_ID": part["ESU_ID"], "Campaign": part["Campaign"],
                "field_id": part["field_id"], "year": part["year"], "y_true": part[target], "y_pred": values,
            }))
        payload["targets"][target] = {
            "n": int(len(part)),
            "between_campaign_share": _variance(part[target], part["Campaign"]),
            "zero_shot_from_gaocheng": zero,
            "few_shot_intercept": _few_shot(part[target].to_numpy(float), pred, part["Campaign"].to_numpy()),
            **designs,
        }
    pd.concat(saved, ignore_index=True).to_csv(root / "outputs" / "predictions" / "bulgaria_uav_predictions.csv", index=False)
    dest = root / "outputs" / "metrics" / "bulgaria_uav_transfer.json"
    dest.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
