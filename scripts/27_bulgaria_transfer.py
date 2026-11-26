"""External test on the Bulgarian field spectra.

The spectra are ASD FieldSpec reflectances, integrated with the nominal
Gaussian spectral response of the Gaocheng camera (centers and FWHM from the
data paper). They are not Sequoia orthomosaic pixels. Chlorophyll is CCM-300
leaf chlorophyll in μg cm-2. Components of the partial least squares model are
chosen on Gaocheng tune rows only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from wheat_ppqmtl.baselines import _fit_classical, _predict
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import regression_metrics
from wheat_ppqmtl.paths import configure_stdio, configs


BANDS = [("G", 560.0, 20.0), ("R", 668.0, 10.0), ("RE", 717.0, 10.0), ("NIR", 840.0, 40.0)]
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


def _convolve(spectra: pd.DataFrame) -> pd.DataFrame:
    names = [c for c in spectra.columns if c.startswith("p") and c.endswith("nm")]
    waves = np.array([float(c[1:-2]) for c in names])
    values = spectra[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    out = spectra[["ESU_ID", "Campaign", "Measurement_date"]].copy()
    for name, center, fwhm in BANDS:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        weight = np.exp(-0.5 * ((waves - center) / sigma) ** 2)
        weight /= weight.sum()
        out[name] = values @ weight
    return out


def _season(campaign: str) -> str:
    year = campaign.split("_")[-1]
    return "S16-17" if year in {"2016", "2017"} and campaign != "November_2017" else "S17-18"


def _within(y, pred, groups) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    groups = np.asarray(groups)
    stats = regression_metrics(y, pred)
    y_dm = np.empty_like(y)
    p_dm = np.empty_like(pred)
    by = []
    for key in pd.unique(groups):
        mask = groups == key
        if mask.sum() < 3:
            continue
        y_dm[mask] = y[mask] - y[mask].mean()
        p_dm[mask] = pred[mask] - pred[mask].mean()
        by.append({"group": str(key), "n": int(mask.sum()), **regression_metrics(y[mask], pred[mask]),
                    "within_rmse": float(np.sqrt(np.mean((y_dm[mask] - p_dm[mask]) ** 2)))})
    scored = np.zeros(len(y), dtype=bool)
    for key in pd.unique(groups):
        mask = groups == key
        if mask.sum() >= 3:
            scored[mask] = True
            y_dm[mask] = y[mask] - y[mask].mean()
            p_dm[mask] = pred[mask] - pred[mask].mean()
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


def _gaocheng_ensemble(root: Path, features: pd.DataFrame, bulgaria: pd.DataFrame, target: str) -> np.ndarray:
    splits = read_table(root / "splits" / "grouped_random.csv")
    columns = {"lcc": "lcc", "lai": "lai"}[target]
    preds = []
    for fold in sorted(splits["outer_fold"].unique()):
        merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
        if target == "lai":
            qc = np.isclose(pd.to_numeric(merged["lai_qc"], errors="coerce"), 1.0)
            merged = merged[qc]
        train = merged[merged["role"] == "train"].dropna(subset=COLUMNS + [columns])
        tune = merged[merged["role"] == "tune"].dropna(subset=COLUMNS + [columns])
        renamed = train.rename(columns={columns: target})
        tune = tune.rename(columns={columns: target})
        preds.append(_predict_external(renamed, tune, bulgaria, target))
    return np.mean(np.vstack(preds), axis=0)


def _local_holdout(frame: pd.DataFrame, group_col: str, target: str) -> dict:
    rows = []
    preds = np.full(len(frame), np.nan)
    for key, part_index in frame.groupby(group_col).groups.items():
        test_i = np.array(list(part_index))
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
    pooled = _within(frame.loc[frame.index[ok], target], preds[ok], frame.loc[frame.index[ok], group_col]) if ok.any() else {}
    return {"by_group": rows, "pooled": pooled}


def _few_shot(y, pred, groups) -> list[dict]:
    rng = np.random.default_rng(42)
    rows = []
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    for key in pd.unique(groups):
        mask = np.flatnonzero(groups == key)
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
            rows.append({"group": str(key), "k": k, "n_evaluated": int(len(mask) - k),
                         "rmse_uncorrected_mean": float(np.mean(base)), "rmse_intercept_mean": float(np.mean(fixed))})
    return rows


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    table = root / "data" / "external" / "bulgaria"
    spectra = pd.read_csv(table / "ESU_Field_spectral_data.txt", sep=r"\s+")
    bio = pd.read_csv(table / "ESU_Biophysical_and_biochemical_crop_data.txt", sep="\t")
    bands = _convolve(spectra)
    use = bands.merge(bio[["ESU_ID", "CC_mean", "LAI_mean"]], on="ESU_ID", how="left")
    use["field_id"] = [
        f"{_season(c)}_U{e.split('_')[1]}" for c, e in zip(use["Campaign"], use["ESU_ID"])
    ]
    use = use.replace({"NA": np.nan})
    use["lcc"] = pd.to_numeric(use["CC_mean"], errors="coerce")
    use["lai"] = pd.to_numeric(use["LAI_mean"], errors="coerce")
    features = read_table(root / "data" / "features" / "features.csv")
    payload = {
        "note": (
            "Bulgarian reflectance is the field spectrometer integrated with nominal Gaussian responses "
            "at the Gaocheng camera centers and FWHM. UAV mosaics were not used. "
            "PLS components were selected on Gaocheng tune rows or, for Bulgaria-only hold-outs, "
            "on 15 percent of the source groups with seed 42."
        ),
        "n_spectra": int(len(spectra)),
        "n_with_chlorophyll": int(use["lcc"].notna().sum()),
        "n_with_lai": int(use["lai"].notna().sum()),
        "targets": {},
    }
    for target in ("lcc", "lai"):
        part = use.dropna(subset=COLUMNS + [target]).reset_index(drop=True)
        pred = _gaocheng_ensemble(root, features, part, target)
        payload["targets"][target] = {
            "n": int(len(part)),
            "zero_shot_from_gaocheng": _within(part[target], pred, part["Campaign"].to_numpy()),
            "few_shot_intercept": _few_shot(part[target], pred, part["Campaign"].to_numpy()),
            "bulgaria_leave_campaign_out": _local_holdout(part, "Campaign", target),
            "bulgaria_leave_field_out": _local_holdout(part, "field_id", target),
        }
    dest = root / "outputs" / "metrics" / "bulgaria_transfer.json"
    dest.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
