"""Ground-calibration budget for a new flight or campaign.

For each held-out Gaocheng date and each Bulgarian campaign, plots are split at
random into a sampling pool and an evaluation half (200 repeats). From the pool,
k plots are labelled by one of three rules fixed before any score was computed:
random, representative (the plot nearest each k-means centroid in standardized
band space) or highest pre-specified risk. Two estimators use the k labels: the
model prediction shifted by the mean residual of the k plots, and the mean of
the k labels alone. Both are scored on the evaluation half, which is never
used for selection or correction.
"""
from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from wheat_ppqmtl.paths import configure_stdio, configs

BUDGETS = [0, 1, 2, 3, 5, 8, 10, 15, 20]
REPEATS = 200
STRATEGIES = ["random", "representative", "high_risk"]


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


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _representative(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k >= len(x):
        return np.arange(len(x))
    z = StandardScaler().fit_transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centers = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z).cluster_centers_
    chosen: list[int] = []
    for center in centers:
        order = np.argsort(((z - center) ** 2).sum(axis=1))
        for index in order:
            if index not in chosen:
                chosen.append(int(index))
                break
    return np.asarray(chosen, dtype=int)


def _select(strategy: str, pool: np.ndarray, k: int, x: np.ndarray, risk: np.ndarray, rng, seed: int) -> np.ndarray:
    if strategy == "random":
        return rng.choice(pool, size=k, replace=False)
    if strategy == "representative":
        return pool[_representative(x[pool], k, seed)]
    order = np.argsort(-risk[pool], kind="stable")
    return pool[order[:k]]


def _budget(frame: pd.DataFrame, group: str, y_col: str, pred_col: str, risk_col: str, bands: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {"groups": [], "macro": {}}
    records = []
    for key, part in frame.groupby(group):
        y = part[y_col].to_numpy(float)
        pred = part[pred_col].to_numpy(float)
        x = part[bands].to_numpy(float)
        risk = part[risk_col].to_numpy(float)
        n = len(part)
        pool_n = n // 2
        for repeat in range(REPEATS):
            perm = rng.permutation(n)
            pool, evaluation = perm[:pool_n], perm[pool_n:]
            oracle = _rmse(y[evaluation], np.full(len(evaluation), y[evaluation].mean()))
            uncorrected = _rmse(y[evaluation], pred[evaluation])
            for strategy in STRATEGIES:
                for k in BUDGETS:
                    if k > pool_n:
                        continue
                    if k == 0:
                        records.append((str(key), strategy, k, repeat, uncorrected, np.nan, oracle))
                        continue
                    chosen = _select(strategy, pool, k, x, risk, rng, seed + repeat)
                    shift = float(np.mean(y[chosen] - pred[chosen]))
                    records.append((
                        str(key), strategy, k, repeat,
                        _rmse(y[evaluation], pred[evaluation] + shift),
                        _rmse(y[evaluation], np.full(len(evaluation), y[chosen].mean())),
                        oracle,
                    ))
        print(f"  {key}: n={n}", flush=True)
    table = pd.DataFrame(records, columns=["group", "strategy", "k", "repeat", "model_shift", "label_mean", "oracle"])
    for key, part in table.groupby("group"):
        n = int((frame[group].astype(str) == key).sum())
        rows = []
        for (strategy, k), cell in part.groupby(["strategy", "k"]):
            paired = cell["label_mean"] - cell["model_shift"]
            rows.append({
                "strategy": strategy, "k": int(k),
                "model_shift_mean": float(cell["model_shift"].mean()),
                "model_shift_p025": float(cell["model_shift"].quantile(0.025)),
                "model_shift_p975": float(cell["model_shift"].quantile(0.975)),
                "label_mean_mean": float(cell["label_mean"].mean()) if cell["label_mean"].notna().any() else None,
                "label_mean_p025": float(cell["label_mean"].quantile(0.025)) if cell["label_mean"].notna().any() else None,
                "label_mean_p975": float(cell["label_mean"].quantile(0.975)) if cell["label_mean"].notna().any() else None,
                "label_mean_minus_model_shift": float(paired.mean()) if paired.notna().any() else None,
                "share_model_shift_better": float((paired > 0).mean()) if paired.notna().any() else None,
            })
        out["groups"].append({
            "group": key, "n": n, "pool": n // 2,
            "oracle_rmse_mean": float(part["oracle"].mean()),
            "curves": rows,
        })
    for (strategy, k), cell in table.groupby(["strategy", "k"]):
        per_group = cell.groupby("group")[["model_shift", "label_mean", "oracle"]].mean()
        paired = (cell["label_mean"] - cell["model_shift"]).groupby(cell["group"]).mean()
        out["macro"].setdefault(strategy, []).append({
            "k": int(k), "groups": sorted(per_group.index.tolist()), "n_groups": int(len(per_group)),
            "model_shift_mean": float(per_group["model_shift"].mean()),
            "label_mean_mean": float(per_group["label_mean"].mean()) if per_group["label_mean"].notna().any() else None,
            "oracle_mean": float(per_group["oracle"].mean()),
            "label_mean_minus_model_shift": float(paired.mean()) if paired.notna().any() else None,
        })
    return out


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    gao = pd.read_csv(root / "outputs" / "predictions" / "gaocheng_lodo_plot_diagnostics.csv")
    bg = pd.read_csv(root / "outputs" / "predictions" / "bulgaria_plot_diagnostics.csv")
    gao_bands = ["B", "G", "R", "RE", "NIR"]
    bg_bands = ["G", "R", "RE", "NIR"]
    payload = {
        "design": {
            "repeats": REPEATS, "budgets": BUDGETS, "strategies": STRATEGIES,
            "split": "half of each group is the sampling pool, the other half is scored",
            "risk": "risk_rtm: mean of training-standardized dissimilarity, PROSAIL reconstruction residual and ensemble spread",
            "estimators": {
                "model_shift": "prediction plus the mean residual of the k labelled plots",
                "label_mean": "mean of the k labelled plots, no reflectance",
            },
        },
        "gaocheng_lodo": {
            "lcc": _budget(gao.dropna(subset=["y_true_lcc", "y_pred_lcc"]), "date_key", "y_true_lcc", "y_pred_lcc", "risk_rtm_lcc", gao_bands, 42),
            "lai": _budget(gao[gao["lai_observed"]].dropna(subset=["y_true_lai", "y_pred_lai"]), "date_key", "y_true_lai", "y_pred_lai", "risk_rtm_lai", gao_bands, 42),
        },
        "bulgaria_zero_shot": {
            "lcc": _budget(bg.dropna(subset=["lcc", "y_pred_lcc"]), "Campaign", "lcc", "y_pred_lcc", "risk_rtm_lcc", bg_bands, 42),
            "lai": _budget(bg.dropna(subset=["lai", "y_pred_lai"]), "Campaign", "lai", "y_pred_lai", "risk_rtm_lai", bg_bands, 42),
        },
    }
    dest = root / "outputs" / "metrics" / "sampling_budget.json"
    dest.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    for site in ("gaocheng_lodo", "bulgaria_zero_shot"):
        for target in ("lcc", "lai"):
            macro = payload[site][target]["macro"]
            print(site, target)
            for strategy in STRATEGIES:
                line = " ".join(
                    f"k{row['k']}:{row['model_shift_mean']:.2f}/{(row['label_mean_mean'] or float('nan')):.2f}[{row['n_groups']}]"
                    for row in macro[strategy]
                )
                print(f"  {strategy}: {line}")
    print(dest)


if __name__ == "__main__":
    main()
