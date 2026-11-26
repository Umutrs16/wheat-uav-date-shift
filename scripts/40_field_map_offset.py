"""Map-level check of the 25 April date hold-out.

The spectral ensemble trained without 25 April has been applied to the whole
orthomosaic (scripts/12_map_unseen_date.py). Map cells are compared with the
training reflectance through the dissimilarity index and threshold used for the
plots. Five plots nearest the k-means centroids of standardized band space give
one offset, which is scored on the remaining plots of that date.
"""
from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.paths import configure_stdio, configs

DAY = "2024.4.25"
BANDS = ["B", "G", "R", "RE", "NIR"]
FACTOR = 34
K = 5


def main() -> None:
    configure_stdio()
    root = configs()["root"]
    diag = pd.read_csv(root / "outputs" / "predictions" / "gaocheng_lodo_plot_diagnostics.csv")
    held = diag[diag["date_key"] == DAY].copy()
    fold = int(held["outer_fold"].iloc[0])
    feats = read_table(root / "data" / "features" / "features.csv")
    splits = read_table(root / "splits" / "lodo.csv")
    train = splits[(splits["outer_fold"] == fold) & (splits["role"] == "train")].merge(feats[["sample_id", *BANDS]], on="sample_id")
    scaler = StandardScaler().fit(train[BANDS].to_numpy(float))
    x_train = scaler.transform(train[BANDS].to_numpy(float))
    pair = np.sqrt(np.maximum(0.0, ((x_train[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)))
    np.fill_diagonal(pair, np.nan)
    denom = float(np.nanmean(pair))
    threshold = float(held["di_threshold"].iloc[0])

    with rasterio.open(root / "outputs" / "maps" / f"{DAY}_spectral_lcc_lai_sigma.tif") as src:
        shape = (src.height // FACTOR, src.width // FACTOR)
        lcc_map = src.read(1, out_shape=shape, resampling=Resampling.average).astype(float)
        cell_m = float(abs(src.transform.a) * FACTOR)
    with rasterio.open(root / "data" / "raw" / "UAV_image" / f"{DAY}_UAV.tif") as src:
        refl = src.read(out_shape=(src.count, *shape), resampling=Resampling.average).astype(float)
    valid = np.isfinite(lcc_map) & np.all(refl > 0, axis=0)
    cells = refl.reshape(len(BANDS), -1).T
    ok = valid.reshape(-1)
    di = cKDTree(x_train).query(scaler.transform(cells[ok]), k=1)[0] / denom

    plots = held.dropna(subset=["y_true_lcc", "y_pred_lcc"]).reset_index(drop=True)
    z = StandardScaler().fit_transform(plots[BANDS].to_numpy(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centers = KMeans(n_clusters=K, n_init=10, random_state=42).fit(z).cluster_centers_
    chosen: list[int] = []
    for c in centers:
        for idx in np.argsort(((z - c) ** 2).sum(axis=1)):
            if idx not in chosen:
                chosen.append(int(idx))
                break
    offset = float(np.mean(plots.loc[chosen, "y_true_lcc"] - plots.loc[chosen, "y_pred_lcc"]))
    rest = plots.drop(index=chosen)
    payload = {
        "date": DAY, "outer_fold": fold, "n_plots": int(len(plots)), "k": K, "offset": offset,
        "rmse_other_plots_before": float(np.sqrt(np.mean((rest["y_true_lcc"] - rest["y_pred_lcc"]) ** 2))),
        "rmse_other_plots_after": float(np.sqrt(np.mean((rest["y_true_lcc"] - rest["y_pred_lcc"] - offset) ** 2))),
        "n_other_plots": int(len(rest)),
        "di_threshold": threshold, "map_cell_m": cell_m,
        "map_inside_aoa_fraction": float(np.mean(di <= threshold)),
        "plot_inside_aoa_fraction": float(held["inside_aoa"].mean()),
        "plot_bias_pred_minus_obs": float((plots["y_pred_lcc"] - plots["y_true_lcc"]).mean()),
    }
    dest = root / "outputs" / "metrics" / "field_map_offset.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
