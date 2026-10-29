from __future__ import annotations

import argparse
import math
from datetime import date

import numpy as np
import pandas as pd

from wheat_ppqmtl.indices import vegetation_indices
from wheat_ppqmtl.io_utils import read_table, write_json, write_table
from wheat_ppqmtl.paths import configure_stdio, configs
from wheat_ppqmtl.solar import assumed_utc, solar_zenith_deg


def add_phenology(frame: pd.DataFrame, solar_cfg: dict) -> pd.DataFrame:
    out = frame.copy()
    doy = out["doy"].to_numpy(float)
    out["doy_sin"] = np.sin(2.0 * math.pi * doy / 365.0)
    out["doy_cos"] = np.cos(2.0 * math.pi * doy / 365.0)
    zenith = []
    for _, row in out.iterrows():
        day = date.fromisoformat(row["date_iso"])
        moment = assumed_utc(day, int(solar_cfg["local_hour"]), int(solar_cfg["local_minute"]), float(solar_cfg["utc_offset_hours"]))
        zenith.append(solar_zenith_deg(float(row["lat"]), float(row["lon"]), moment))
    out["sza_deg"] = zenith
    out["cos_sza"] = np.cos(np.radians(out["sza_deg"]))
    return out


def add_ccc(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = frame.copy()
    both = out["lcc"].notna() & out["lai"].notna()
    out["ccc"] = np.where(both, factor * out["lcc"] * out["lai"], np.nan)
    return out


def _patch_index(root) -> pd.DataFrame | None:
    path = root / "data" / "patches" / "patch_index.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _stats_from_patches(index: pd.DataFrame, threshold: float) -> pd.DataFrame:
    from skimage.feature import graycomatrix, graycoprops

    rows = []
    for rec in index.itertuples(index=False):
        payload = np.load(rec.path)
        image = payload["reflectance"].astype(float)  # C,H,W
        valid = payload["valid"].astype(bool)
        if valid.ndim == 3:
            valid_hw = valid[0]
        else:
            valid_hw = valid
        row = {"sample_id": rec.sample_id, "patch_valid_fraction": float(valid_hw.mean())}
        names = ["B", "G", "R", "RE", "NIR"]
        for band, name in enumerate(names):
            values = image[band][valid_hw]
            if values.size == 0:
                row[name] = np.nan
                row[f"{name}_median"] = np.nan
                row[f"{name}_p10"] = np.nan
                row[f"{name}_p90"] = np.nan
                continue
            row[name] = float(np.mean(values))
            row[f"{name}_median"] = float(np.median(values))
            row[f"{name}_p10"] = float(np.percentile(values, 10))
            row[f"{name}_p90"] = float(np.percentile(values, 90))
        red = image[2]
        nir = image[4]
        ndvi = (nir - red) / np.where(np.abs(nir + red) < 1e-6, np.nan, nir + red)
        row["veg_fraction"] = float(np.nanmean(ndvi > threshold))
        gray = nir.copy()
        finite = np.isfinite(gray)
        if finite.any():
            lo, hi = np.nanpercentile(gray, [1, 99])
            scaled = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)
            scaled = (scaled * 15).astype(np.uint8)
            matrix = graycomatrix(scaled, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
            row["glcm_contrast"] = float(graycoprops(matrix, "contrast")[0, 0])
            row["glcm_homogeneity"] = float(graycoprops(matrix, "homogeneity")[0, 0])
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(frame: pd.DataFrame, patch_stats: pd.DataFrame | None, ccc_factor: float, solar_cfg: dict, ndvi_threshold: float) -> tuple[pd.DataFrame, dict]:
    published = frame.copy()
    for band in ("B", "G", "R", "RE", "NIR"):
        published[f"{band}_published"] = published[band]
    source = "published_plot_mean"
    if patch_stats is not None and len(patch_stats):
        keep_published = published.drop(columns=["B", "G", "R", "RE", "NIR"])
        merged = keep_published.merge(patch_stats, on="sample_id", how="left")
        source = "inner_core"
        # Rows without a patch keep the published means so the table stays complete,
        # but those rows are flagged and excluded from CNN training.
        for band in ("B", "G", "R", "RE", "NIR"):
            merged[band] = merged[band].fillna(merged[f"{band}_published"])
        frame = merged
    else:
        frame = published
    frame = vegetation_indices(frame)
    frame = add_phenology(frame, solar_cfg)
    frame = add_ccc(frame, ccc_factor)
    meta = {
        "spectral_source": source,
        "ccc_factor": ccc_factor,
        "solar": solar_cfg,
        "ndvi_threshold_for_vegetation_fraction": ndvi_threshold,
        "n_rows": int(len(frame)),
        "n_with_patch": int(frame["patch_valid_fraction"].notna().sum()) if "patch_valid_fraction" in frame else 0,
        "solar_time_is_assumed": True,
    }
    return frame, meta


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Build plot-level predictors from the table and any patches.").parse_args()
    cfg = configs()
    root = cfg["root"]
    plot_path = root / "data" / "interim" / "observations_plot.csv"
    table_path = plot_path if plot_path.exists() else root / "data" / "interim" / "observations.csv"
    frame = read_table(table_path)
    patches = _patch_index(root)
    patch_stats = _stats_from_patches(patches, float(cfg["data"]["vegetation_fraction"]["ndvi_threshold"])) if patches is not None else None
    features, meta = build_features(
        frame,
        patch_stats,
        float(cfg["data"]["ccc"]["factor"]),
        cfg["data"]["solar"],
        float(cfg["data"]["vegetation_fraction"]["ndvi_threshold"]),
    )
    feature_dir = root / "data" / "features"
    write_table(features, feature_dir / "features.csv")
    write_json(feature_dir / "feature_meta.json", meta)
    bad = features[["ndvi", "ndre", "gndvi", "cire", "rvi", "osavi", "evi2"]].replace([np.inf, -np.inf], np.nan)
    print(f"features={len(features)} source={meta['spectral_source']} nonfinite_vi={int(bad.isna().any(axis=1).sum())}")


if __name__ == "__main__":
    main()
