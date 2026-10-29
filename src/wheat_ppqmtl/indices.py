from __future__ import annotations

import numpy as np
import pandas as pd


SPECTRAL = ["B", "G", "R", "RE", "NIR"]
VI_COLUMNS = ["ndvi", "ndre", "gndvi", "cire", "rvi", "osavi", "evi2"]
PHYSICS_COLUMNS = SPECTRAL + ["cos_sza"]
PHENOLOGY_COLUMNS = VI_COLUMNS + ["doy_sin", "doy_cos"]
BASELINE_COLUMNS = SPECTRAL + VI_COLUMNS + ["cos_sza"]
B0_COLUMNS = SPECTRAL + ["cos_sza"]


def safe_div(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan, dtype=float)
    ok = np.abs(denominator) >= eps
    out[ok] = numerator[ok] / denominator[ok]
    return out


def vegetation_indices(df: pd.DataFrame) -> pd.DataFrame:
    b = df["B"].to_numpy(float)
    g = df["G"].to_numpy(float)
    r = df["R"].to_numpy(float)
    re = df["RE"].to_numpy(float)
    nir = df["NIR"].to_numpy(float)
    out = df.copy()
    out["ndvi"] = safe_div(nir - r, nir + r)
    out["ndre"] = safe_div(nir - re, nir + re)
    out["gndvi"] = safe_div(nir - g, nir + g)
    out["cire"] = safe_div(nir, re) - 1.0
    out["rvi"] = safe_div(nir, r)
    out["osavi"] = 1.16 * safe_div(nir - r, nir + r + 0.16)
    out["evi2"] = 2.5 * safe_div(nir - r, nir + 2.4 * r + 1.0)
    # Keep a column used only as a mask diagnostic, not as a ground-truth cover.
    out["ndvi_raw"] = out["ndvi"]
    _ = b  # blue is retained as a spectral predictor; it is not in these indices.
    return out


def gaussian_srf(wavelengths: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    weights = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    total = weights.sum()
    if total <= 0:
        raise ValueError("SRF weights summed to zero")
    return weights / total


def integrate_bands(spectrum: np.ndarray, wavelengths: np.ndarray, centers: dict, fwhm: dict, order: list[str]) -> np.ndarray:
    values = []
    for name in order:
        weights = gaussian_srf(wavelengths, centers[name], fwhm[name])
        values.append(float(np.dot(spectrum, weights)))
    return np.asarray(values, dtype=float)
