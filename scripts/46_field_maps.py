"""Field-map diagnostics for Fig. 8: Gaocheng 25 April and one Knezha spring LAI unit.

Gaocheng (25 April held out):
  crop mask NDVI >= 0.25 inside the field extent; predicted and observed SD;
  within-date R2 of the raw LODO predictions; residuals at plots not used for
  the five-plot offset; maps saved as a compressed npz for the figure script.

Knezha (April 2018, unit U2):
  extract the Sequoia mosaic with 7-Zip; average it to about 2.25 m cells and
  take 20 m moving-window means of each band (the ESU footprint), so that the
  Gaocheng four-band PLSR zero-shot model is applied at the support of the ESU
  reflectance it was evaluated on; correct with an offset and with ASRC from
  five spectral-cluster ESUs of the whole campaign (none on U2, whose ESUs are
  only used for evaluation); save the LAI stack, the U2 mask and ESU scores.

Run after 35, 40 and 42. Does not retrain the primary spectral ensemble.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.multilevel import agreement, jsonable
from wheat_ppqmtl.paths import configure_stdio, configs

DAY = "2024.4.25"
BANDS5 = ["B", "G", "R", "RE", "NIR"]
FOUR = ["G", "R", "RE", "NIR"]
FACTOR = 34
K = 5
NDVI_MIN = 0.25
M_PSEUDO = 5.0
KNEZHA_ZIP = "S17-18_U2_2018-04-04.zip"
KNEZHA_CAMPAIGN = "April_2018"
KNEZHA_FIELD = "S17-18_U2"
MAP_FACTOR = 9
SUPPORT_M = 20.0
MIN_VALID = 0.5


def _seven_zip(root: Path) -> str | None:
    bundled = root / "tools" / "7zip" / "extra" / "x64" / "7za.exe"
    if bundled.exists():
        return str(bundled)
    return shutil.which("7za") or shutil.which("7z")


def _band_name(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    if "rededge" in stem or "red_edge" in stem:
        return "RE"
    if stem.endswith("_nir") or stem.endswith("-nir"):
        return "NIR"
    if stem.endswith("_green") or stem.endswith("-green"):
        return "G"
    if stem.endswith("_red") or stem.endswith("-red"):
        return "R"
    return None


def _extract_mosaic(root: Path, zip_name: str, work: Path) -> dict[str, Path]:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    path = root / "data" / "external" / "bulgaria" / "uav" / zip_name
    try:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".tif")]
            archive.extractall(work, members)
    except NotImplementedError:
        seven = _seven_zip(root)
        if seven is None:
            raise RuntimeError("Deflate64 requires 7-Zip")
        completed = subprocess.run([seven, "x", "-y", f"-o{work}", str(path)], capture_output=True, text=True)
        if completed.returncode != 0:
            raise OSError(completed.stderr or completed.stdout or "7za failed")
    paths = {}
    for member in work.rglob("*.tif"):
        band = _band_name(str(member))
        if band:
            paths[band] = member
    missing = [b for b in FOUR if b not in paths]
    if missing:
        raise RuntimeError(f"missing bands in {zip_name}: {missing}")
    return paths


def _nearest_unique(points: np.ndarray, centers: np.ndarray) -> list[int]:
    distance = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    chosen, used = [], set()
    for c in np.argsort(distance.min(axis=0)):
        for i in np.argsort(distance[:, c]):
            if int(i) not in used:
                chosen.append(int(i))
                used.add(int(i))
                break
    return chosen


def _asrc(y_s, f_s, f_eval, lam0, v_f, m=M_PSEUDO) -> tuple[np.ndarray, float, float]:
    ybar, fbar = float(np.mean(y_s)), float(np.mean(f_s))
    yc, fc = y_s - ybar, f_s - fbar
    s_ff = float((fc ** 2).sum())
    s_fy = float((fc * yc).sum())
    lam = float(np.clip((s_fy + m * v_f * lam0) / (s_ff + m * v_f), 0.0, 1.0)) if v_f > 0 else lam0
    return ybar + lam * (f_eval - fbar), lam, ybar + 1.0 * (f_eval - fbar)  # asrc, lam, offset


def _gaocheng(root: Path) -> dict:
    diag = pd.read_csv(root / "outputs" / "predictions" / "gaocheng_lodo_plot_diagnostics.csv")
    held = diag[diag["date_key"] == DAY].copy()
    fold = int(held["outer_fold"].iloc[0])
    feats = read_table(root / "data" / "features" / "features.csv")
    splits = read_table(root / "splits" / "lodo.csv")
    train = splits[(splits["outer_fold"] == fold) & (splits["role"] == "train")].merge(
        feats[["sample_id", *BANDS5]], on="sample_id")
    scaler = StandardScaler().fit(train[BANDS5].to_numpy(float))
    x_train = scaler.transform(train[BANDS5].to_numpy(float))
    pair = np.sqrt(np.maximum(0.0, ((x_train[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)))
    np.fill_diagonal(pair, np.nan)
    denom = float(np.nanmean(pair))
    threshold = float(held["di_threshold"].iloc[0])

    with rasterio.open(root / "outputs" / "maps" / f"{DAY}_spectral_lcc_lai_sigma.tif") as src:
        shape = (src.height // FACTOR, src.width // FACTOR)
        lcc_map = src.read(1, out_shape=shape, resampling=Resampling.average).astype(float)
        transform = src.transform
        cell_m = float(abs(transform.a) * FACTOR)
        extent = [transform.c, transform.c + src.width * transform.a,
                  transform.f - src.height * abs(transform.e), transform.f]
    with rasterio.open(root / "data" / "raw" / "UAV_image" / f"{DAY}_UAV.tif") as src:
        refl = src.read(out_shape=(src.count, *shape), resampling=Resampling.average).astype(float)
        nir = src.read(5, out_shape=(src.height // 10, src.width // 10), resampling=Resampling.average).astype(float)

    valid = np.isfinite(lcc_map) & np.all(refl > 0, axis=0)
    ndvi = (refl[4] - refl[2]) / np.clip(refl[4] + refl[2], 1e-6, None)
    crop = valid & (ndvi >= NDVI_MIN)
    lcc_crop = np.where(crop, lcc_map, np.nan)
    cells = refl.reshape(len(BANDS5), -1).T
    di = np.full(cells.shape[0], np.nan)
    ok = valid.reshape(-1)
    di[ok] = cKDTree(x_train).query(scaler.transform(cells[ok]), k=1)[0] / denom
    di_map = di.reshape(shape)

    plots = held.dropna(subset=["y_true_lcc", "y_pred_lcc"]).reset_index(drop=True)
    z = StandardScaler().fit_transform(plots[BANDS5].to_numpy(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centers = KMeans(n_clusters=K, n_init=10, random_state=42).fit(z).cluster_centers_
    chosen = _nearest_unique(z, centers)
    rest = plots.drop(index=chosen)
    y_s = plots.loc[chosen, "y_true_lcc"].to_numpy(float)
    f_s = plots.loc[chosen, "y_pred_lcc"].to_numpy(float)
    f_rest = rest["y_pred_lcc"].to_numpy(float)
    y_rest = rest["y_true_lcc"].to_numpy(float)
    # lambda0: within-date slope on LODO calibration rows of the training dates
    cal = splits[(splits["outer_fold"] == fold) & (splits["role"] == "calibration")][["sample_id", "date_key"]].merge(
        diag[["sample_id", "y_true_lcc", "y_pred_lcc"]], on="sample_id")
    cal = cal[cal["date_key"] != DAY].dropna(subset=["y_true_lcc", "y_pred_lcc"])
    yc = cal["y_true_lcc"] - cal.groupby("date_key")["y_true_lcc"].transform("mean")
    fc = cal["y_pred_lcc"] - cal.groupby("date_key")["y_pred_lcc"].transform("mean")
    sff = float((fc ** 2).sum())
    lam0 = float(np.clip((fc * yc).sum() / sff, 0.0, 1.0)) if sff > 0 else 0.0
    v_f = float(np.var(plots["y_pred_lcc"].to_numpy(float), ddof=0))
    asrc_rest, lam, offset_rest = _asrc(y_s, f_s, f_rest, lam0, v_f)
    offset = float(np.mean(y_s - f_s))
    agr_raw = agreement(y_rest, f_rest)
    agr_off = agreement(y_rest, offset_rest)
    agr_asrc = agreement(y_rest, asrc_rest)
    agr_all = agreement(plots["y_true_lcc"].to_numpy(float), plots["y_pred_lcc"].to_numpy(float))

    residual_map = np.full(shape, np.nan)
    for row in rest.itertuples(index=False):
        c = int((row.x - extent[0]) / cell_m)
        r = int((extent[3] - row.y) / cell_m)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            residual_map[r, c] = float(row.y_true_lcc - (row.y_pred_lcc + offset))

    stack_path = root / "outputs" / "maps" / "fig8_gaocheng_25april.npz"
    np.savez_compressed(
        stack_path,
        lcc_raw=lcc_map.astype(np.float32),
        lcc_crop=lcc_crop.astype(np.float32),
        lcc_offset=(lcc_crop + offset).astype(np.float32),
        ndvi=ndvi.astype(np.float32),
        crop=crop.astype(np.uint8),
        di=di_map.astype(np.float32),
        residual_eval=residual_map.astype(np.float32),
        nir_preview=np.where(nir <= 0, np.nan, nir).astype(np.float32),
        extent=np.asarray(extent, float),
        cell_m=np.asarray([cell_m], float),
        plot_x=plots["x"].to_numpy(float),
        plot_y=plots["y"].to_numpy(float),
        plot_obs=plots["y_true_lcc"].to_numpy(float),
        plot_pred=plots["y_pred_lcc"].to_numpy(float),
        chosen=np.asarray(chosen, int),
        rest_index=rest.index.to_numpy(int),
    )
    payload = {
        "date": DAY, "outer_fold": fold, "n_plots": int(len(plots)), "k": K,
        "ndvi_min": NDVI_MIN, "crop_fraction": float(crop.mean()),
        "map_cell_m": cell_m, "extent": extent,
        "obs_sd_plots": float(plots["y_true_lcc"].std(ddof=0)),
        "pred_sd_plots": float(plots["y_pred_lcc"].std(ddof=0)),
        "pred_sd_map_crop": float(np.nanstd(lcc_crop)),
        "sd_ratio_plots": float(plots["y_pred_lcc"].std(ddof=0) / plots["y_true_lcc"].std(ddof=0)),
        "within_date_r2_all_plots": agr_all["r2_within"],
        "spearman_all_plots": agr_all["spearman"],
        "offset": offset, "lambda0": lam0, "lambda_asrc": lam,
        "n_other_plots": int(len(rest)),
        "rmse_other_raw": agr_raw["rmse"], "rmse_other_offset": agr_off["rmse"],
        "rmse_other_asrc": agr_asrc["rmse"],
        "r2_within_other_raw": agr_raw["r2_within"],
        "r2_within_other_offset": agr_off["r2_within"],
        "r2_within_other_asrc": agr_asrc["r2_within"],
        "di_threshold": threshold,
        "map_inside_aoa_fraction": float(np.nanmean(di_map[np.isfinite(di_map)] <= threshold)),
        "map_inside_aoa_fraction_crop": float(np.nanmean(di_map[crop] <= threshold)),
        "plot_inside_aoa_fraction": float(held["inside_aoa"].mean()),
        "plot_bias_pred_minus_obs": float((plots["y_pred_lcc"] - plots["y_true_lcc"]).mean()),
        "npz": str(stack_path.relative_to(root)).replace("\\", "/"),
    }
    return payload


def _fit_zero_shot_plsr(root: Path, target: str = "lai"):
    features = read_table(root / "data" / "features" / "features.csv")
    features["lai_ok"] = np.isclose(pd.to_numeric(features["lai_qc"], errors="coerce"), 1.0)
    splits = read_table(root / "splits" / "grouped_random.csv")
    models = []
    for fold in sorted(splits["outer_fold"].unique()):
        merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
        train = merged[merged["role"] == "train"].dropna(subset=FOUR + [target])
        tune = merged[merged["role"] == "tune"].dropna(subset=FOUR + [target])
        if target == "lai":
            train = train[train["lai_ok"]]
            tune = tune[tune["lai_ok"]]
        if len(train) < 12 or len(tune) < 3:
            continue
        x_tr, y_tr = train[FOUR].to_numpy(float), train[target].to_numpy(float)
        x_tu, y_tu = tune[FOUR].to_numpy(float), tune[target].to_numpy(float)
        scaler = StandardScaler().fit(x_tr)
        xt, xv = scaler.transform(x_tr), scaler.transform(x_tu)
        best, best_score = None, float("inf")
        for n_comp in (1, 2, 3):
            if n_comp >= x_tr.shape[1] or n_comp >= len(x_tr):
                continue
            model = PLSRegression(n_components=n_comp).fit(xt, y_tr)
            pred = model.predict(xv).reshape(-1)
            score = float(np.sqrt(np.mean((pred - y_tu) ** 2)))
            if score < best_score:
                best, best_score = (model, scaler), score
        if best is not None:
            models.append(best)
    if not models:
        raise RuntimeError("no PLSR fold models")

    def predict(x: np.ndarray) -> np.ndarray:
        preds = [m.predict(s.transform(x)).reshape(-1) for m, s in models]
        return np.mean(preds, axis=0)

    return predict, len(models)


def _knezha(root: Path) -> dict:
    predict, n_folds = _fit_zero_shot_plsr(root, "lai")
    uav = pd.read_csv(root / "data" / "external" / "bulgaria" / "uav_esu_reflectance.csv")
    bio = pd.read_csv(root / "data" / "external" / "bulgaria" / "ESU_Biophysical_and_biochemical_crop_data.txt", sep="\t")
    camp = uav[uav["Campaign"] == KNEZHA_CAMPAIGN].copy()
    camp = camp.merge(bio[["ESU_ID", "Point_X_UTM34", "Point_Y_UTM34"]], on="ESU_ID", how="left")
    camp["y_pred"] = predict(camp[FOUR].to_numpy(float))
    camp = camp.dropna(subset=["lai", "y_pred"]).reset_index(drop=True)

    # lambda0 from Gaocheng OOF four-band PLSR (same target)
    features = read_table(root / "data" / "features" / "features.csv")
    features["lai_ok"] = np.isclose(pd.to_numeric(features["lai_qc"], errors="coerce"), 1.0)
    splits = read_table(root / "splits" / "grouped_random.csv")
    oof = []
    for fold in sorted(splits["outer_fold"].unique()):
        merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
        train = merged[merged["role"] == "train"].dropna(subset=FOUR + ["lai"])
        tune = merged[merged["role"] == "tune"].dropna(subset=FOUR + ["lai"])
        test = merged[merged["role"] == "test"].dropna(subset=FOUR + ["lai"])
        train, tune, test = train[train["lai_ok"]], tune[tune["lai_ok"]], test[test["lai_ok"]]
        if len(train) < 12 or len(test) < 3:
            continue
        x_tr, y_tr = train[FOUR].to_numpy(float), train["lai"].to_numpy(float)
        x_tu, y_tu = tune[FOUR].to_numpy(float), tune["lai"].to_numpy(float)
        scaler = StandardScaler().fit(x_tr)
        xt, xv = scaler.transform(x_tr), scaler.transform(x_tu)
        best = None
        best_score = float("inf")
        for n_comp in (1, 2, 3):
            if n_comp >= x_tr.shape[1] or n_comp >= len(x_tr):
                continue
            model = PLSRegression(n_components=n_comp).fit(xt, y_tr)
            score = float(np.sqrt(np.mean((model.predict(xv).reshape(-1) - y_tu) ** 2))) if len(tune) else 0.0
            if score < best_score:
                best, best_score = model, score
        pred = best.predict(scaler.transform(test[FOUR].to_numpy(float))).reshape(-1)
        part = test[["sample_id", "date_key", "lai"]].copy()
        part["y_pred"] = pred
        oof.append(part)
    oof = pd.concat(oof, ignore_index=True)
    yc = oof["lai"] - oof.groupby("date_key")["lai"].transform("mean")
    fc = oof["y_pred"] - oof.groupby("date_key")["y_pred"].transform("mean")
    sff = float((fc ** 2).sum())
    lam0 = float(np.clip((fc * yc).sum() / sff, 0.0, 1.0)) if sff > 0 else 0.0

    z = StandardScaler().fit_transform(camp[FOUR].to_numpy(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centers = KMeans(n_clusters=K, n_init=10, random_state=42).fit(z).cluster_centers_
    chosen = _nearest_unique(z, centers)
    # Evaluation ESUs on the mapped field that were not used for calibration
    field = camp[camp["field_id"] == KNEZHA_FIELD].copy()
    cal_ids = set(camp.loc[chosen, "ESU_ID"])
    eval_field = field[~field["ESU_ID"].isin(cal_ids)].copy()
    y_s = camp.loc[chosen, "lai"].to_numpy(float)
    f_s = camp.loc[chosen, "y_pred"].to_numpy(float)
    v_f = float(np.var(camp["y_pred"].to_numpy(float), ddof=0))
    asrc_all, lam, offset_all = _asrc(y_s, f_s, camp["y_pred"].to_numpy(float), lam0, v_f)
    camp["y_asrc"] = asrc_all
    camp["y_offset"] = offset_all
    offset = float(np.mean(y_s - f_s))
    agr_raw = agreement(eval_field["lai"].to_numpy(float), eval_field["y_pred"].to_numpy(float)) if len(eval_field) else {}
    agr_off = agreement(eval_field["lai"].to_numpy(float),
                        eval_field["ESU_ID"].map(dict(zip(camp["ESU_ID"], camp["y_offset"]))).to_numpy(float)) if len(eval_field) else {}
    agr_asrc = agreement(eval_field["lai"].to_numpy(float),
                         eval_field["ESU_ID"].map(dict(zip(camp["ESU_ID"], camp["y_asrc"]))).to_numpy(float)) if len(eval_field) else {}
    agr_camp = agreement(camp["lai"].to_numpy(float), camp["y_pred"].to_numpy(float))

    work = root / "data" / "external" / "bulgaria" / "_fig8_uav_tmp"
    paths = _extract_mosaic(root, KNEZHA_ZIP, work)
    with rasterio.open(paths["G"]) as ref:
        height, width = ref.height // MAP_FACTOR, ref.width // MAP_FACTOR
        transform = ref.transform * Affine.scale(ref.width / width, ref.height / height)
        extent = [transform.c, transform.c + width * transform.a,
                  transform.f + height * transform.e, transform.f]
        cell_m = float(abs(transform.a))
        crs = str(ref.crs)
    bands = {}
    for band, path in paths.items():
        with rasterio.open(path) as src:
            bands[band] = src.read(1, out_shape=(height, width), resampling=Resampling.average).astype(float)
    stack = np.stack([bands[b] for b in FOUR], axis=0)
    valid = np.all(np.isfinite(stack) & (stack > 0), axis=0)
    window = max(1, int(round(SUPPORT_M / cell_m)))
    if window % 2 == 0:
        window += 1
    fraction = uniform_filter(valid.astype(np.float32), size=window, mode="constant")
    focal = np.stack([uniform_filter(np.where(valid, stack[b], 0.0).astype(np.float32), size=window, mode="constant")
                      / np.maximum(fraction, 1e-6) for b in range(len(FOUR))], axis=0)
    units = gpd.read_file(f"zip://{(root / 'data' / 'external' / 'bulgaria' / 'Shapefile_Units.zip').as_posix()}!Units.shp")
    polygon = units[units["Unit_ID"] == KNEZHA_FIELD].to_crs(crs)
    inside = geometry_mask(polygon.geometry, out_shape=(height, width), transform=transform, invert=True)
    ndvi = (bands["NIR"] - bands["R"]) / np.clip(bands["NIR"] + bands["R"], 1e-6, None)
    crop = valid & (ndvi >= NDVI_MIN) & (fraction >= MIN_VALID) & inside
    flat = focal.reshape(4, -1).T
    pred = np.full(flat.shape[0], np.nan)
    ok = crop.reshape(-1)
    pred[ok] = predict(flat[ok])
    pred_map = pred.reshape(height, width)
    asrc_map = np.where(crop, float(np.mean(y_s)) + lam * (pred_map - float(np.mean(f_s))), np.nan)
    offset_map = np.where(crop, pred_map + offset, np.nan)

    # Map values at the U2 ESU centres against the ESU-level predictions used in the scores.
    inverse = ~transform
    check = []
    for row in field.itertuples(index=False):
        col_f, row_f = inverse * (float(row.Point_X_UTM34), float(row.Point_Y_UTM34))
        r_i, c_i = int(np.floor(row_f)), int(np.floor(col_f))
        value = float(pred_map[r_i, c_i]) if 0 <= r_i < height and 0 <= c_i < width else np.nan
        check.append({"ESU_ID": row.ESU_ID, "esu_pred": float(row.y_pred), "map_pred": value})
    diffs = [abs(c["map_pred"] - c["esu_pred"]) for c in check if np.isfinite(c["map_pred"])]

    stack_path = root / "outputs" / "maps" / "fig8_knezha_april2018_u2.npz"
    np.savez_compressed(
        stack_path,
        lai_raw=pred_map.astype(np.float32),
        lai_offset=offset_map.astype(np.float32),
        lai_asrc=asrc_map.astype(np.float32),
        ndvi=ndvi.astype(np.float32),
        crop=crop.astype(np.uint8),
        inside=inside.astype(np.uint8),
        nir=np.where(bands["NIR"] <= 0, np.nan, bands["NIR"]).astype(np.float32),
        extent=np.asarray(extent, float),
        cell_m=np.asarray([cell_m], float),
        support_m=np.asarray([window * cell_m], float),
        field_bounds=np.asarray(polygon.total_bounds, float),
        esu_id=camp["ESU_ID"].astype(str).to_numpy(),
        esu_x=camp["Point_X_UTM34"].to_numpy(float),
        esu_y=camp["Point_Y_UTM34"].to_numpy(float),
        esu_obs=camp["lai"].to_numpy(float),
        esu_pred=camp["y_pred"].to_numpy(float),
        esu_offset=camp["y_offset"].to_numpy(float),
        esu_asrc=camp["y_asrc"].to_numpy(float),
        esu_field=camp["field_id"].astype(str).to_numpy(),
        chosen=np.asarray(chosen, int),
        field_mask=(camp["field_id"] == KNEZHA_FIELD).to_numpy(),
    )
    shutil.rmtree(work, ignore_errors=True)
    return {
        "campaign": KNEZHA_CAMPAIGN, "field_id": KNEZHA_FIELD, "image_zip": KNEZHA_ZIP,
        "n_folds": n_folds, "n_campaign_esus": int(len(camp)), "k": K,
        "n_eval_field": int(len(eval_field)), "ndvi_min": NDVI_MIN,
        "crop_fraction": float(crop[inside].mean()), "map_cell_m": cell_m, "support_m": float(window * cell_m),
        "window_cells": window, "min_valid_fraction": MIN_VALID, "extent": extent, "crs": crs,
        "field_bounds": [float(v) for v in polygon.total_bounds],
        "chosen_fields": sorted(set(camp.loc[chosen, "field_id"].astype(str))),
        "map_vs_esu": check, "map_vs_esu_max_abs_diff": float(max(diffs)) if diffs else None,
        "map_vs_esu_mean_abs_diff": float(np.mean(diffs)) if diffs else None,
        "obs_sd_campaign": float(camp["lai"].std(ddof=0)),
        "pred_sd_campaign": float(camp["y_pred"].std(ddof=0)),
        "pred_sd_map_crop": float(np.nanstd(pred_map)),
        "sd_ratio_campaign": float(camp["y_pred"].std(ddof=0) / camp["lai"].std(ddof=0)),
        "within_campaign_r2": agr_camp["r2_within"], "spearman_campaign": agr_camp["spearman"],
        "offset": offset, "lambda0": lam0, "lambda_asrc": lam,
        "rmse_eval_raw": agr_raw.get("rmse"), "rmse_eval_offset": agr_off.get("rmse"),
        "rmse_eval_asrc": agr_asrc.get("rmse"),
        "r2_within_eval_raw": agr_raw.get("r2_within"),
        "r2_within_eval_offset": agr_off.get("r2_within"),
        "r2_within_eval_asrc": agr_asrc.get("r2_within"),
        "spearman_eval_raw": agr_raw.get("spearman"),
        "spearman_eval_offset": agr_off.get("spearman"),
        "spearman_eval_asrc": agr_asrc.get("spearman"),
        "chosen_esus": camp.loc[chosen, "ESU_ID"].tolist(),
        "npz": str(stack_path.relative_to(root)).replace("\\", "/"),
    }


def main() -> None:
    configure_stdio()
    root = configs()["root"]
    (root / "outputs" / "maps").mkdir(parents=True, exist_ok=True)
    gao = _gaocheng(root)
    print("gaocheng", json.dumps({k: gao[k] for k in gao if k != "extent"}, indent=2), flush=True)
    knz = _knezha(root)
    print("knezha", json.dumps({k: knz[k] for k in knz if k != "extent"}, indent=2), flush=True)
    numbers_path = root / "outputs" / "metrics" / "figure_numbers.json"
    numbers = json.loads(numbers_path.read_text(encoding="utf-8")) if numbers_path.exists() else {}
    numbers["fig8_25april"] = {
        "outer_fold": gao["outer_fold"], "n_plots": gao["n_plots"], "k": gao["k"],
        "offset": gao["offset"], "rmse_other_plots_before": gao["rmse_other_raw"],
        "rmse_other_plots_after": gao["rmse_other_offset"], "n_other_plots": gao["n_other_plots"],
        "di_threshold": gao["di_threshold"], "map_cell_m": gao["map_cell_m"],
        "map_inside_aoa_fraction": gao["map_inside_aoa_fraction"],
        "plot_inside_aoa_fraction": gao["plot_inside_aoa_fraction"],
        "plot_bias_pred_minus_obs": gao["plot_bias_pred_minus_obs"],
        "crop_fraction": gao["crop_fraction"], "obs_sd_plots": gao["obs_sd_plots"],
        "pred_sd_plots": gao["pred_sd_plots"], "sd_ratio_plots": gao["sd_ratio_plots"],
        "within_date_r2_all_plots": gao["within_date_r2_all_plots"],
        "rmse_other_asrc": gao["rmse_other_asrc"], "lambda_asrc": gao["lambda_asrc"],
    }
    numbers["fig8_knezha_lai"] = {k: knz[k] for k in knz if k != "extent"}
    numbers_path.write_text(json.dumps(jsonable(numbers), indent=2), encoding="utf-8")
    dest = root / "outputs" / "metrics" / "field_maps.json"
    dest.write_text(json.dumps(jsonable({"gaocheng_25april": gao, "knezha_april2018_u2": knz}), indent=2), encoding="utf-8")
    print(dest, flush=True)


if __name__ == "__main__":
    main()
