from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import torch

from wheat_ppqmtl.indices import safe_div
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.model import PPQMTL
from wheat_ppqmtl.observations import image_name
from wheat_ppqmtl.paths import configure_stdio, configs
from wheat_ppqmtl.scaling import RobustScale
from wheat_ppqmtl.solar import assumed_utc, solar_zenith_deg


def _checkpoints_for(root: Path, date_key: str, model_id: str) -> list[Path]:
    splits = read_table(root / "splits" / "lodo.csv")
    test = splits[splits["role"] == "test"]
    matches = test[test["date_key"] == date_key]
    if matches.empty:
        return []
    fold = int(matches["outer_fold"].iloc[0])
    return sorted((root / "models" / "checkpoints").glob(f"{model_id}__lodo__fold{fold}__seed*__lcc-lai.pt"))


def map_date(date_key: str, model_id: str = "ppq_tabular") -> Path | None:
    import rasterio

    cfg = configs()
    root = cfg["root"]
    ckpt_paths = _checkpoints_for(root, date_key, model_id)
    if not ckpt_paths:
        print(f"No {model_id} LODO checkpoint for {date_key}")
        return None
    year, month, day = (int(part) for part in date_key.split("."))
    image_paths = list((root / "data" / "raw").rglob(image_name(date(year, month, day))))
    if not image_paths:
        print(f"No GeoTIFF for {date_key}; mapping is implemented and waiting for the image.")
        return None
    saved = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    if saved["use_cnn"]:
        print("CNN mapping needs a sliding window. This draft maps the tabular PPQ checkpoint.")
        return None
    models = []
    for ckpt_path in ckpt_paths:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = PPQMTL(pheno_dim=len(state["pheno_cols"]), use_cnn=False, use_phenology=state["pheno_scaler"] is not None)
        model.load_state_dict(state["model"])
        model.eval()
        models.append(model)
    print(f"mapping ensemble n={len(models)}")
    phys_scaler = RobustScale.from_dict(saved["phys_scaler"])
    pheno_scaler = None if saved["pheno_scaler"] is None else RobustScale.from_dict(saved["pheno_scaler"])
    target = saved["target_scalers"]
    observations = read_table(root / "data" / "interim" / "observations_plot.csv")
    sample = observations[observations["date_key"] == date_key].iloc[0]
    moment = assumed_utc(date(year, month, day), int(cfg["data"]["solar"]["local_hour"]), int(cfg["data"]["solar"]["local_minute"]), float(cfg["data"]["solar"]["utc_offset_hours"]))
    cos_sza = float(np.cos(np.radians(solar_zenith_deg(float(sample["lat"]), float(sample["lon"]), moment))))
    doy = int(date(year, month, day).timetuple().tm_yday)
    doy_sin = np.sin(2 * np.pi * doy / 365.0)
    doy_cos = np.cos(2 * np.pi * doy / 365.0)
    threshold = float(cfg["data"]["vegetation_fraction"]["ndvi_threshold"])

    def _window_means(image: np.ndarray, pixel_m: float) -> tuple[np.ndarray, int]:
        from scipy.ndimage import uniform_filter

        window_m = float(cfg["data"]["patch"]["inner_m"])
        k = max(1, int(round(window_m / pixel_m)))
        if k % 2 == 0:
            k += 1
        valid = np.isfinite(image).all(axis=0) & (image >= 0).all(axis=0) & (image <= 1.5).all(axis=0)
        fraction = uniform_filter(valid.astype(np.float32), size=k, mode="constant")
        means = np.empty_like(image, dtype=np.float32)
        for band in range(image.shape[0]):
            filled = np.where(valid, image[band], 0).astype(np.float32)
            means[band] = uniform_filter(filled, size=k, mode="constant") / np.maximum(fraction, 1e-6)
        means[:, fraction < 0.8] = np.nan
        return means, k

    def predict_block(image: np.ndarray, pixel_m: float) -> np.ndarray:
        image, _window_px = _window_means(image, pixel_m)
        b, g, r, re, nir = image
        ndvi = safe_div(nir - r, nir + r)
        flat = image.reshape(5, -1).T
        if list(saved["phys_cols"]) != ["B", "G", "R", "RE", "NIR"]:
            raise RuntimeError("Mapping expects a five-band physics encoder")
        phys_s = phys_scaler.transform(flat).astype(np.float32)
        pheno_s = None
        if pheno_scaler is not None:
            vi = np.column_stack([
                ndvi.reshape(-1),
                safe_div(nir - re, nir + re).reshape(-1),
                safe_div(nir - g, nir + g).reshape(-1),
                (safe_div(nir, re) - 1).reshape(-1),
                safe_div(nir, r).reshape(-1),
                (1.16 * safe_div(nir - r, nir + r + 0.16)).reshape(-1),
                (2.5 * safe_div(nir - r, nir + 2.4 * r + 1)).reshape(-1),
                np.full(flat.shape[0], doy_sin),
                np.full(flat.shape[0], doy_cos),
            ])
            # Keep the column order stored in the checkpoint.
            frame = {name: vi[:, i] for i, name in enumerate(["ndvi", "ndre", "gndvi", "cire", "rvi", "osavi", "evi2", "doy_sin", "doy_cos"])}
            ordered = np.column_stack([frame[name] for name in saved["pheno_cols"] if name in frame])
            pheno_s = pheno_scaler.transform(ordered).astype(np.float32)
        out = np.full((4, flat.shape[0]), np.nan, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, flat.shape[0], 200000):
                stop = min(start + 200000, flat.shape[0])
                mus = {name: [] for name in ("lcc", "lai")}
                vars_ = {name: [] for name in ("lcc", "lai")}
                for model in models:
                    pred = model(
                        torch.tensor(phys_s[start:stop]),
                        None if pheno_s is None else torch.tensor(pheno_s[start:stop]),
                        None,
                    )
                    for name, mu_key, var_key in (("lcc", "mu_lcc", "logvar_lcc"), ("lai", "mu_lai", "logvar_lai")):
                        center = target[name]["center"][0]
                        scale = target[name]["scale"][0]
                        mu = pred[mu_key].numpy() * scale + center
                        sigma = np.exp(0.5 * pred[var_key].numpy()) * scale
                        mus[name].append(mu)
                        vars_[name].append(sigma ** 2)
                for band, name in ((0, "lcc"), (1, "lai")):
                    stacked = np.stack(mus[name], axis=0)
                    mu_bar = stacked.mean(axis=0)
                    total = np.mean(np.stack(vars_[name], axis=0) + stacked ** 2, axis=0) - mu_bar ** 2
                    out[band, start:stop] = mu_bar
                    out[band + 2, start:stop] = np.sqrt(np.maximum(total, 0))
        valid = (ndvi.reshape(-1) >= threshold) & np.isfinite(flat).all(axis=1)
        out[:, ~valid] = np.nan
        return out.reshape(4, image.shape[1], image.shape[2])

    src_path = image_paths[0]
    dest = root / "outputs" / "maps" / f"{date_key}_{model_id}_lcc_lai_sigma.tif"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=4, dtype="float32", nodata=np.nan, compress="deflate")
        pixel_m = abs(float(src.transform.a))
        image = src.read(out_dtype="float32")[:5]
        mapped = predict_block(image, pixel_m)
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(mapped)
            dst.update_tags(support_window_m=str(cfg["data"]["patch"]["inner_m"]), pixel_m=f"{pixel_m:.4f}")
    print(f"wrote {dest}")
    return dest


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Map LCC, LAI and their predictive standard deviations for one flight date.")
    parser.add_argument("--date", default="2024.4.25")
    parser.add_argument("--model", default="spectral")
    args = parser.parse_args()
    map_date(args.date, args.model)


if __name__ == "__main__":
    main()
