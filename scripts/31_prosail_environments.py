"""Four pre-specified PROSAIL environments, kept separate from the main lookup table.

Each environment overrides the ranges listed in ENVIRONMENTS. Parameters that
are not listed keep the literature bounds in configs/prosail.yaml. The sensor
comparison integrates the same canopy spectra with the Gaocheng Gaussian
responses and with nominal Sequoia centers and bandwidths. Those Sequoia
bandwidths are not a measured spectral response from the mosaics.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import prosail

import _path  # noqa: F401
from wheat_ppqmtl.indices import integrate_bands
from wheat_ppqmtl.paths import configure_stdio, configs

WAVELENGTHS = np.arange(400, 2501)
N_PER_ENV = 250
ENVIRONMENTS = {
    "early_low_cover": {"lai": (0.2, 1.5), "cab": (20.0, 55.0), "cbrown": (0.0, 0.3), "psoil": (0.5, 1.0), "rsoil": (0.8, 1.5)},
    "peak_growth": {"lai": (2.0, 6.0), "cab": (40.0, 80.0), "cbrown": (0.0, 0.4)},
    "high_lai": {"lai": (6.0, 10.5), "cab": (35.0, 70.0), "ala": (45.0, 70.0)},
    "senescence": {"lai": (1.0, 4.5), "cab": (5.0, 15.0), "cbrown": (0.6, 1.5), "cw": (0.005, 0.015)},
}
SEQUOIA = {
    "order": ["G", "R", "RE", "NIR"],
    "centers": {"G": 550.0, "R": 660.0, "RE": 735.0, "NIR": 790.0},
    "fwhm": {"G": 40.0, "R": 40.0, "RE": 10.0, "NIR": 40.0},
}


def _draw(ranges: dict, overrides: dict, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = {}
    for name, (lo, hi) in ranges.items():
        if name in overrides:
            lo, hi = overrides[name]
        if name == "hspot":
            draws[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        else:
            draws[name] = rng.uniform(lo, hi, n)
    return draws


def _simulate(draws: dict, anthocyanin: float) -> np.ndarray:
    spectra = np.zeros((len(draws["n"]), WAVELENGTHS.size), dtype=float)
    for i in range(len(draws["n"])):
        spectrum = prosail.run_prosail(
            draws["n"][i], draws["cab"][i], draws["car"][i], draws["cbrown"][i],
            draws["cw"][i], draws["cm"][i], draws["lai"][i], draws["ala"][i],
            draws["hspot"][i], draws["sza"][i], draws["vza"][i], draws["raa"][i],
            ant=anthocyanin, prospect_version="D", typelidf=2,
            rsoil=float(draws["rsoil"][i]), psoil=float(draws["psoil"][i]),
        )
        spectra[i] = np.asarray(spectrum, dtype=float).reshape(-1)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(draws['n'])}", flush=True)
    return spectra


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    ranges = {name: tuple(bounds) for name, bounds in cfg["prosail"]["ranges"].items()}
    centers = cfg["data"]["bands"]["centers_nm"]
    fwhm = cfg["data"]["bands"]["fwhm_nm"]
    order = cfg["data"]["bands"]["order"]
    rows = []
    sensor_rows = []
    for index, (name, overrides) in enumerate(ENVIRONMENTS.items()):
        print(name, flush=True)
        draws = _draw(ranges, overrides, N_PER_ENV, 42 + index)
        spectra = _simulate(draws, float(cfg["prosail"]["anthocyanin"]))
        bands = np.vstack([
            integrate_bands(spectra[i], WAVELENGTHS, centers, fwhm, order) for i in range(len(spectra))
        ])
        for i in range(len(spectra)):
            row = {"environment": name, "cab": float(draws["cab"][i]), "lai": float(draws["lai"][i])}
            for column, value in zip(order, bands[i]):
                row[column] = float(value)
            rows.append(row)
        if name == "peak_growth":
            gaocheng = np.vstack([
                integrate_bands(spectra[i], WAVELENGTHS, centers, fwhm, SEQUOIA["order"]) for i in range(len(spectra))
            ])
            sequoia = np.vstack([
                integrate_bands(spectra[i], WAVELENGTHS, SEQUOIA["centers"], SEQUOIA["fwhm"], SEQUOIA["order"])
                for i in range(len(spectra))
            ])
            for column, g_values, s_values in zip(SEQUOIA["order"], gaocheng.T, sequoia.T):
                sensor_rows.append({
                    "band": column,
                    "gaocheng_center_nm": centers[column],
                    "sequoia_center_nm": SEQUOIA["centers"][column],
                    "mean_absolute_difference": float(np.mean(np.abs(g_values - s_values))),
                    "mean_gaocheng": float(np.mean(g_values)),
                    "mean_sequoia": float(np.mean(s_values)),
                })
    table = pd.DataFrame(rows)
    dest = root / "data" / "synthetic" / "prosail_environments.csv"
    table.to_csv(dest, index=False)
    observed = pd.read_csv(root / "data" / "features" / "features.csv")
    library = table[order].to_numpy(float)
    center = library.mean(axis=0)
    scale = library.std(axis=0)
    scale[scale == 0] = 1.0
    lib_z = (library - center) / scale
    by_date = []
    for date, part in observed.groupby("date_key"):
        obs = (part[order].to_numpy(float) - center) / scale
        distances = np.sqrt(((obs[:, None, :] - lib_z[None, :, :]) ** 2).sum(axis=2))
        dist = distances.min(axis=1)
        env = table["environment"].to_numpy()[distances.argmin(axis=1)]
        counts = pd.Series(env).value_counts(normalize=True).to_dict()
        by_date.append({
            "date": str(date),
            "n": int(len(part)),
            "median_nearest_distance": float(np.median(dist)),
            "nearest_environment_share": {str(k): float(v) for k, v in counts.items()},
        })
    payload = {
        "n_per_environment": N_PER_ENV,
        "note": (
            "Environments use fixed literature-style ranges and were not chosen from the test dates. "
            "Nearest-environment shares describe spectral proximity, not a successful inversion. "
            "Sequoia bandwidths are nominal and were not measured from the downloaded mosaics."
        ),
        "sensor_shift_on_peak_growth": sensor_rows,
        "observed_nearest_environment": by_date,
        "library_band_mean": {column: float(table[column].mean()) for column in order},
        "library_band_sd": {column: float(table[column].std()) for column in order},
    }
    out = root / "outputs" / "metrics" / "prosail_environments.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out, flush=True)


if __name__ == "__main__":
    main()
