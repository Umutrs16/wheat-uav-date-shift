from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import qmc

from wheat_ppqmtl.indices import integrate_bands
from wheat_ppqmtl.io_utils import write_json, write_table
from wheat_ppqmtl.paths import configure_stdio, configs


WAVELENGTHS = np.arange(400, 2501)


def _map_unit(u: np.ndarray, lo: float, hi: float, log: bool = False) -> np.ndarray:
    if log:
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    return lo + u * (hi - lo)


def _simulate_chunk(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    import prosail

    indices = payload["indices"]
    draws = payload["draws"]
    bands = np.zeros((len(indices), len(payload["order"])), dtype=float)
    for index in range(len(indices)):
        spectrum = prosail.run_prosail(
            draws["n"][index], draws["cab"][index], draws["car"][index], draws["cbrown"][index],
            draws["cw"][index], draws["cm"][index], draws["lai"][index], draws["ala"][index],
            draws["hspot"][index], draws["sza"][index], draws["vza"][index], draws["raa"][index],
            ant=float(payload["anthocyanin"]),
            prospect_version=payload["prospect_version"],
            typelidf=2,
            rsoil=float(draws["rsoil"][index]),
            psoil=float(draws["psoil"][index]),
        )
        spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
        if spectrum.size != WAVELENGTHS.size:
            raise RuntimeError(f"PROSAIL returned {spectrum.size} wavelengths, expected {WAVELENGTHS.size}")
        values = integrate_bands(spectrum, WAVELENGTHS, payload["centers"], payload["fwhm"], payload["order"])
        bands[index] = np.clip(values * payload["gain"][index] + payload["noise"][index], 0.0, 1.0)
    return np.asarray(indices), bands


def simulate(n_samples: int, seed: int, observations: pd.DataFrame | None = None) -> pd.DataFrame:
    import os
    from concurrent.futures import ProcessPoolExecutor

    del observations  # Field labels are not allowed to reshape the lookup table.
    cfg = configs()
    data_cfg = cfg["data"]
    prosail_cfg = cfg["prosail"]
    ranges = prosail_cfg["ranges"]
    names = ["n", "cab", "car", "cbrown", "cw", "cm", "lai", "ala", "hspot", "sza", "vza", "raa", "rsoil", "psoil"]
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n_samples)
    draws = {}
    for i, name in enumerate(names):
        lo, hi = ranges[name]
        draws[name] = _map_unit(unit[:, i], lo, hi, log=(name == "hspot"))
    rho = float(prosail_cfg["car_cab_correlation"])
    z_cab = (draws["cab"] - draws["cab"].mean()) / draws["cab"].std()
    z_car = (draws["car"] - draws["car"].mean()) / draws["car"].std()
    mixed = rho * z_cab + np.sqrt(max(0.0, 1.0 - rho ** 2)) * z_car
    draws["car"] = np.clip(mixed * draws["car"].std() + draws["car"].mean(), ranges["car"][0], ranges["car"][1])

    centers = data_cfg["bands"]["centers_nm"]
    fwhm = data_cfg["bands"]["fwhm_nm"]
    order = data_cfg["bands"]["order"]
    gain_lo, gain_hi = prosail_cfg["domain_randomization"]["gain"]
    sig_lo, sig_hi = prosail_cfg["domain_randomization"]["noise_sigma"]
    rng = np.random.default_rng(seed + 2)
    gain = rng.uniform(gain_lo, gain_hi, size=(n_samples, len(order)))
    sigma = rng.uniform(sig_lo, sig_hi, size=n_samples)
    noise = rng.normal(0.0, 1.0, size=(n_samples, len(order))) * sigma[:, None]
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    chunks = np.array_split(np.arange(n_samples), workers * 4)
    payload = {
        "draws": draws,
        "centers": centers,
        "fwhm": fwhm,
        "order": order,
        "anthocyanin": float(prosail_cfg["anthocyanin"]),
        "prospect_version": prosail_cfg["prospect_version"],
        "gain": gain,
        "noise": noise,
    }
    bands = np.zeros((n_samples, len(order)), dtype=float)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            item = dict(payload)
            item["indices"] = chunk.astype(int)
            item["draws"] = {name: values[chunk] for name, values in draws.items()}
            item["gain"] = gain[chunk]
            item["noise"] = noise[chunk]
            futures.append(pool.submit(_simulate_chunk, item))
        for future in futures:
            indices, values = future.result()
            bands[indices] = values
            done += len(indices)
            print(f"prosail {done}/{n_samples}", flush=True)
    factor = float(data_cfg["ccc"]["factor"])
    table = pd.DataFrame({name: draws[name] for name in names})
    for column, values in zip(order, bands.T):
        table[column] = values
    table["cos_sza"] = np.cos(np.radians(table["sza"]))
    table["ccc"] = factor * table["cab"] * table["lai"]
    val_n = int(round(len(table) * float(prosail_cfg["pretrain"]["val_fraction"])))
    split = np.array(["train"] * len(table))
    split[:val_n] = "val"
    rng.shuffle(split)
    table["sim_split"] = split
    return table


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Latin-hypercube PROSAIL simulation at the UAV band responses.")
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()
    cfg = configs()
    n_samples = int(args.n_samples or cfg["prosail"]["draft_samples"])
    table = simulate(n_samples, int(cfg["prosail"]["seed"]), None)
    dest = write_table(table, cfg["root"] / "data" / "synthetic" / "prosail.csv")
    write_json(cfg["root"] / "data" / "synthetic" / "prosail_meta.json", {
        "n_samples": int(len(table)),
        "paper_samples": int(cfg["prosail"]["paper_samples"]),
        "path": str(dest),
        "prospect_version": cfg["prosail"]["prospect_version"],
        "srf": "Gaussian, dataset-readme FWHM",
        "sensor_response_note": "Integrated with Gaussian spectral response functions, not the camera's measured SRF.",
        "observed_range_rejection": False,
        "range_source": "literature bounds in configs/prosail.yaml; field labels were not used",
    })
    print(f"wrote {dest} n={len(table)}")


if __name__ == "__main__":
    main()
