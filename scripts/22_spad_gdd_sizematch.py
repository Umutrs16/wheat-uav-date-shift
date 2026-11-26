"""SPAD-scale sensitivity, external GDD baseline, and a size-matched hold-out.

Reconstructed SPAD is the algebraic inverse of the date-specific equations in
the data paper. It is not the original meter file. March uses the literature
equation the data authors adopted, not a fit to this campaign.

GDD uses the Open-Meteo historical archive at the field coordinate. It is not
a logger on the experimental plots. Base temperature 0 °C and the sowing date
2023-10-10 are fixed before looking at test error.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from wheat_ppqmtl.baselines import _fit_classical, _predict
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import cluster_bootstrap_ci, regression_metrics
from wheat_ppqmtl.paths import configure_stdio, configs


EQUATIONS = {
    "2024.3.31": ("quadratic", 0.0102, 0.3015),
    "2024.4.12": ("linear", 1.1339, -5.5071),
    "2024.4.25": ("linear", 3.3983, -112.95),
    "2024.5.18": ("linear", 0.9774, -5.5896),
    "2024.6.6": ("linear", 0.7433, -0.5729),
}
COLUMNS = ["B", "G", "R", "RE", "NIR", "ndvi", "ndre", "gndvi", "cire", "rvi", "osavi", "evi2", "doy_sin", "doy_cos"]


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


def _spad_one(date_key: str, lcc: float) -> float:
    kind = EQUATIONS[date_key]
    if kind[0] == "linear":
        slope, intercept = kind[1], kind[2]
        return (lcc - intercept) / slope
    a, b = kind[1], kind[2]
    disc = b * b + 4.0 * a * lcc
    return (-b + np.sqrt(disc)) / (2.0 * a)


def _variance(values: np.ndarray, dates: np.ndarray) -> dict:
    grand = float(values.mean())
    sst = float(np.sum((values - grand) ** 2))
    ssb = 0.0
    rows = []
    for day in pd.unique(dates):
        part = values[dates == day]
        mean = float(part.mean())
        ssb += len(part) * (mean - grand) ** 2
        rows.append({
            "date": str(day),
            "n": int(len(part)),
            "mean": mean,
            "sd": float(part.std(ddof=1)),
            "min": float(part.min()),
            "max": float(part.max()),
        })
    return {
        "n": int(len(values)),
        "between_share": float(ssb / sst),
        "within_share": float(1.0 - ssb / sst),
        "by_date": rows,
    }


def _within(y, pred, dates) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    y_dm = np.empty_like(y)
    p_dm = np.empty_like(pred)
    by_date = []
    for day in pd.unique(dates):
        mask = dates == day
        y_dm[mask] = y[mask] - y[mask].mean()
        p_dm[mask] = pred[mask] - pred[mask].mean()
        by_date.append({
            "date": str(day),
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))),
            "within_rmse": float(np.sqrt(np.mean((y_dm[mask] - p_dm[mask]) ** 2))),
        })
    return {
        **regression_metrics(y, pred),
        "within_rmse": float(np.sqrt(np.mean((y_dm - p_dm) ** 2))),
        "by_date": by_date,
    }


def _fit_predict(train, tune, test, target, exp) -> np.ndarray:
    x_train = train[COLUMNS].to_numpy(float)
    y_train = train[target].to_numpy(float)
    x_tune = tune[COLUMNS].to_numpy(float)
    y_tune = tune[target].to_numpy(float)
    model, bundle = _fit_classical("plsr", x_train, y_train, x_tune, y_tune, 42, exp)
    pred, _ = _predict("plsr", model, bundle, test[COLUMNS].to_numpy(float))
    return pred


def _pls_schemes(root: Path, features: pd.DataFrame, exp: dict) -> dict:
    out = {}
    for scheme in ("grouped_random", "spatial_block", "lodo"):
        splits = read_table(root / "splits" / f"{scheme}.csv")
        out[scheme] = {}
        for target in ("lcc", "spad_star"):
            parts = []
            for fold in sorted(splits["outer_fold"].unique()):
                merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
                train = merged[merged["role"] == "train"].dropna(subset=COLUMNS + [target])
                tune = merged[merged["role"] == "tune"].dropna(subset=COLUMNS + [target])
                test = merged[merged["role"] == "test"].dropna(subset=COLUMNS + [target])
                pred = _fit_predict(train, tune, test, target, exp)
                part = test[["sample_id", "date_key", "block_id", target]].copy()
                part["y_pred"] = pred
                parts.append(part)
            frame = pd.concat(parts, ignore_index=True)
            stats = _within(frame[target], frame["y_pred"], frame["date_key"].to_numpy())
            ci = cluster_bootstrap_ci(frame[target], frame["y_pred"], frame["block_id"].to_numpy(), n_boot=1000, seed=42)
            stats["rmse_ci95"] = ci["rmse_ci95"]
            out[scheme][target] = stats
    return out


def _size_matched(root: Path, features: pd.DataFrame, exp: dict) -> dict:
    """Same test size as each left-out date, but the held-out plots are random.

    Training size is capped at the leave-one-date-out training count. Twenty
    draws, seed 42. This asks whether the date gap is only a smaller sample.
    """
    splits = read_table(root / "splits" / "lodo.csv")
    rng = np.random.default_rng(42)
    draws = []
    date_rows = []
    for fold in sorted(splits["outer_fold"].unique()):
        merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
        test = merged[merged["role"] == "test"].dropna(subset=COLUMNS + ["lcc"])
        train_n = int((merged["role"] == "train").sum())
        complement = merged[merged["role"] != "test"].dropna(subset=COLUMNS + ["lcc"])
        date_pred = _fit_predict(
            merged[merged["role"] == "train"].dropna(subset=COLUMNS + ["lcc"]),
            merged[merged["role"] == "tune"].dropna(subset=COLUMNS + ["lcc"]),
            test,
            "lcc",
            exp,
        )
        date_rows.append({"date": str(test["date_key"].iloc[0]), "n_test": int(len(test)), "n_train": train_n,
                          **regression_metrics(test["lcc"], date_pred)})
        fold_scores = []
        ids = features.dropna(subset=COLUMNS + ["lcc"])["sample_id"].to_numpy()
        for _ in range(20):
            chosen = set(rng.choice(ids, size=len(test), replace=False))
            pool = features[features["sample_id"].isin(ids) & ~features["sample_id"].isin(chosen)].copy()
            pool = pool.sample(n=min(train_n, len(pool)), random_state=int(rng.integers(0, 1_000_000)))
            tune_n = max(8, int(round(0.15 * len(pool))))
            tune = pool.iloc[:tune_n]
            train = pool.iloc[tune_n:]
            held = features[features["sample_id"].isin(chosen)]
            pred = _fit_predict(train, tune, held, "lcc", exp)
            fold_scores.append(regression_metrics(held["lcc"], pred)["rmse"])
        draws.append({"date_matched_n": int(len(test)), "random_holdout_rmse_mean": float(np.mean(fold_scores)),
                      "random_holdout_rmse_sd": float(np.std(fold_scores, ddof=1))})
    return {"leave_one_date": date_rows, "size_matched_random_holdout": draws}


def _gdd(features: pd.DataFrame) -> dict:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?latitude=37.881194&longitude=114.853997"
        "&start_date=2023-10-10&end_date=2024-06-06&daily=temperature_2m_max,temperature_2m_min&timezone=Asia%2FShanghai"
    )
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    daily = payload["daily"]
    dates = pd.to_datetime(pd.Series(daily["time"]))
    temp = (np.asarray(daily["temperature_2m_max"], float) + np.asarray(daily["temperature_2m_min"], float)) / 2.0
    gdd = np.cumsum(np.maximum(0.0, temp))
    by_day = {d.date().isoformat(): float(v) for d, v in zip(dates, gdd)}
    features = features.copy()
    features["gdd"] = features["date_iso"].map(by_day)
    if features["gdd"].isna().any():
        return {"available": False, "error": "one or more flight dates were absent from the archive"}
    out = {"available": True, "source": "Open-Meteo archive, 2 m air temperature, base 0 C, cumulative from 2023-10-10",
           "gdd_by_flight": features.groupby("date_key")["gdd"].first().astype(float).to_dict(),
           "schemes": {}}
    root_note = out["schemes"]
    # The caller fills schemes. Return the column through features mutation by assigning back.
    features.attrs["gdd_meta"] = out
    return features


def _gdd_models(root: Path, features: pd.DataFrame) -> dict:
    from sklearn.linear_model import LinearRegression

    schemes = {}
    for scheme in ("grouped_random", "spatial_block", "lodo"):
        splits = read_table(root / "splits" / f"{scheme}.csv")
        rows = []
        for fold in sorted(splits["outer_fold"].unique()):
            merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
            train = merged[merged["role"] == "train"]
            test = merged[merged["role"] == "test"]
            model = LinearRegression().fit(train[["gdd"]].to_numpy(float), train["lcc"].to_numpy(float))
            pred = model.predict(test[["gdd"]].to_numpy(float))
            part = test[["sample_id", "date_key", "lcc"]].copy()
            part["y_pred"] = pred
            rows.append(part)
        frame = pd.concat(rows, ignore_index=True)
        schemes[scheme] = _within(frame["lcc"], frame["y_pred"], frame["date_key"].to_numpy())
    return schemes


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")
    features["spad_star"] = [_spad_one(d, y) for d, y in zip(features["date_key"], features["lcc"])]
    recovered = []
    for date_key, lcc, spad in zip(features["date_key"], features["lcc"], features["spad_star"]):
        kind = EQUATIONS[date_key]
        if kind[0] == "linear":
            recovered.append(kind[1] * spad + kind[2])
        else:
            recovered.append(kind[1] * spad ** 2 + kind[2] * spad)
    payload = {
        "spad_note": (
            "spad_star inverts the data-paper Table 3 equations. March is the literature equation "
            "cited there (Bannari et al. 2007) because extraction samples were limited; its R2 was not printed. "
            "The quadratic term is read as x squared. These values are not original SPAD readings."
        ),
        "roundtrip_max_abs_lcc_error": float(np.max(np.abs(np.asarray(recovered) - features["lcc"].to_numpy(float)))),
        "spad_outside_0_80": int(((features["spad_star"] < 0) | (features["spad_star"] > 80)).sum()),
        "lcc_variance": _variance(features["lcc"].to_numpy(float), features["date_key"].to_numpy()),
        "spad_star_variance": _variance(features["spad_star"].to_numpy(float), features["date_key"].to_numpy()),
        "pls_bands_vi_doy": _pls_schemes(root, features, cfg["experiment"]),
        "size_matched": _size_matched(root, features, cfg["experiment"]),
    }
    gdd_features = _gdd(features)
    if isinstance(gdd_features, pd.DataFrame):
        meta = gdd_features.attrs["gdd_meta"]
        meta["schemes"] = _gdd_models(root, gdd_features)
        payload["gdd"] = meta
    else:
        payload["gdd"] = gdd_features
    dest = root / "outputs" / "metrics" / "spad_gdd_sizematch.json"
    dest.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
