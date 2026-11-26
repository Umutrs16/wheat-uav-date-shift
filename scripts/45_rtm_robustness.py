"""Robustness of the PROSAIL reconstruction residual and identifiability of leaf chlorophyll.

1. Convergence: residual (script 35) with 256, 1024 and 4096 Latin-hypercube
   draws of the nuisance parameters. 256 draws with the script-35 seed must
   reproduce the saved values.
2. Solar zenith: residual with the assumed acquisition time moved by -1.5 h and +1.5 h.
3. Range: distance to the nearest lookup-table spectrum (e_nn) for a library with
   the lower bounds of the literature ranges (Cab 5, LAI 0.2) and for one with the
   lower bounds extended to Cab 1 and LAI 0.02, below the smallest observed values.
4. Identifiability: for each plot with observed LAI, the best reconstruction over
   a grid of leaf chlorophyll values, with LAI fixed at the observed value or
   left free (drawn with the nuisance parameters). A flat profile means the
   bands do not constrain chlorophyll for that plot.
Run after 35.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc, spearmanr
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from wheat_ppqmtl.indices import gaussian_srf
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.multilevel import jsonable
from wheat_ppqmtl.paths import configure_stdio, configs
from wheat_ppqmtl.solar import assumed_utc, solar_zenith_deg

WAVELENGTHS = np.arange(400, 2501)
GAO = ["B", "G", "R", "RE", "NIR"]
FOUR = ["G", "R", "RE", "NIR"]
SEQUOIA = {"centers": {"G": 550, "R": 660, "RE": 735, "NIR": 790}, "fwhm": {"G": 40, "R": 40, "RE": 10, "NIR": 40}}
NUISANCE = ["n", "car", "cbrown", "cw", "cm", "ala", "hspot", "rsoil", "psoil"]
DRAW_SIZES = [256, 1024, 4096]
CAB_GRID = np.array([2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], float)
PROFILE_DRAWS_FIXED = 256
PROFILE_DRAWS_FREE = 512
TOLERANCES = [0.002, 0.005]
LAI_FREE = (0.02, 10.5)
EXTENDED = {"cab": (1.0, 80.0), "lai": (0.02, 10.5)}
SHIFT_H = 1.5
KNEZHA = (43.495, 24.081)
EEST = {
    2016: (date(2016, 3, 27), date(2016, 10, 30)),
    2017: (date(2017, 3, 26), date(2017, 10, 29)),
    2018: (date(2018, 3, 25), date(2018, 10, 28)),
}


def _weights(centers: dict, fwhm: dict, order: list[str]) -> np.ndarray:
    return np.vstack([gaussian_srf(WAVELENGTHS, centers[b], fwhm[b]) for b in order])


def _draws(ranges: dict, n: int, seed: int, names=NUISANCE, extra: dict | None = None) -> dict:
    names = list(names) + list(extra or {})
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    out = {}
    for i, name in enumerate(names):
        lo, hi = (extra or {}).get(name, ranges.get(name, (None, None)))
        if name == "hspot":
            out[name] = np.exp(np.log(lo) + unit[:, i] * (np.log(hi) - np.log(lo)))
        else:
            out[name] = lo + unit[:, i] * (hi - lo)
    return out


def _spectra(draws: dict, cab, lai, sza, j_range) -> np.ndarray:
    import prosail

    rows = []
    for j in j_range:
        lai_j = draws["lai"][j] if lai is None else lai
        rows.append(np.asarray(prosail.run_prosail(
            draws["n"][j], cab, draws["car"][j], draws["cbrown"][j], draws["cw"][j], draws["cm"][j],
            lai_j, draws["ala"][j], draws["hspot"][j], sza, 0.0, 0.0,
            ant=0.0, prospect_version="D", typelidf=2,
            rsoil=float(draws["rsoil"][j]), psoil=float(draws["psoil"][j]),
        ), dtype=float).reshape(-1))
    return np.vstack(rows)


def _residual_chunk(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    draws, weights = payload["draws"], payload["weights"]
    count = len(next(iter(draws.values())))
    best = np.full(len(payload["cab"]), np.nan)
    for i in range(len(payload["cab"])):
        cab, lai, sza, obs = payload["cab"][i], payload["lai"][i], payload["sza"][i], payload["obs"][i]
        if not (np.isfinite(cab) and np.isfinite(lai) and np.isfinite(sza) and np.isfinite(obs).all()):
            continue
        sims = _spectra(draws, float(cab), float(lai), float(sza), range(count)) @ weights.T
        best[i] = float(np.sqrt(np.mean((sims - obs[None, :]) ** 2, axis=1)).min())
    return payload["index"], best


def _profile_chunk(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = payload["weights"]
    fixed_draws, free_draws = payload["fixed"], payload["free"]
    n_fixed = len(fixed_draws["n"])
    n_free = len(free_draws["n"])
    out_fixed = np.full((len(payload["lai"]), len(CAB_GRID)), np.nan)
    out_free = np.full((len(payload["lai"]), len(CAB_GRID)), np.nan)
    for i in range(len(payload["lai"])):
        lai, sza, obs = payload["lai"][i], payload["sza"][i], payload["obs"][i]
        if not (np.isfinite(sza) and np.isfinite(obs).all()):
            continue
        for g, cab in enumerate(CAB_GRID):
            if np.isfinite(lai):
                sims = _spectra(fixed_draws, float(cab), float(lai), float(sza), range(n_fixed)) @ weights.T
                out_fixed[i, g] = float(np.sqrt(np.mean((sims - obs[None, :]) ** 2, axis=1)).min())
            sims = _spectra(free_draws, float(cab), None, float(sza), range(n_free)) @ weights.T
            out_free[i, g] = float(np.sqrt(np.mean((sims - obs[None, :]) ** 2, axis=1)).min())
    return payload["index"], out_fixed, out_free


def _parallel(fn, payloads: list[dict], workers: int):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, payloads, chunksize=1))


def _residual(frame: pd.DataFrame, cab, lai, sza, obs, weights, draws, workers, label) -> np.ndarray:
    cab = np.clip(np.asarray(cab, float), 0.5, 100.0)
    lai = np.clip(np.asarray(lai, float), 0.05, 10.5)
    out = np.full(len(frame), np.nan)
    chunks = [c for c in np.array_split(np.arange(len(frame)), workers * 4) if len(c)]
    payloads = [{"index": c, "cab": cab[c], "lai": lai[c], "sza": np.asarray(sza, float)[c], "obs": obs[c],
                 "weights": weights, "draws": draws} for c in chunks]
    for index, values in _parallel(_residual_chunk, payloads, workers):
        out[index] = values
    print(f"  {label}: median {np.nanmedian(out):.4f}", flush=True)
    return out


def _profile(obs, lai, sza, weights, fixed, free, workers) -> tuple[np.ndarray, np.ndarray]:
    n = len(obs)
    out_fixed = np.full((n, len(CAB_GRID)), np.nan)
    out_free = np.full((n, len(CAB_GRID)), np.nan)
    chunks = [c for c in np.array_split(np.arange(n), workers * 4) if len(c)]
    payloads = [{"index": c, "lai": np.asarray(lai, float)[c], "sza": np.asarray(sza, float)[c], "obs": obs[c],
                 "weights": weights, "fixed": fixed, "free": free} for c in chunks]
    for index, a, b in _parallel(_profile_chunk, payloads, workers):
        out_fixed[index] = a
        out_free[index] = b
    return out_fixed, out_free


def _profile_metrics(profile: np.ndarray, observed_cab: np.ndarray) -> dict:
    out = {}
    minimum = np.nanmin(profile, axis=1)
    out["argmin_cab"] = CAB_GRID[np.nanargmin(np.where(np.isfinite(profile), profile, np.inf), axis=1)]
    out["e_min"] = minimum
    for tol in TOLERANCES:
        inside = profile <= (minimum[:, None] + tol)
        lo = np.array([CAB_GRID[row].min() if row.any() else np.nan for row in inside])
        hi = np.array([CAB_GRID[row].max() if row.any() else np.nan for row in inside])
        out[f"width_{tol:g}"] = hi - lo
        out[f"covers_obs_{tol:g}"] = (observed_cab >= lo - 2.5) & (observed_cab <= hi + 2.5)
    out["relative_range"] = (np.nanmax(profile, axis=1) - minimum) / np.maximum(minimum, 1e-9)
    return out


def _sza_shifted(frame: pd.DataFrame, site: str, shift_h: float) -> np.ndarray:
    values = []
    for _, row in frame.iterrows():
        if site == "gaocheng":
            day = date(*map(int, str(row["date_key"]).split(".")))
            minutes = int(round((11 + shift_h) * 60))
            moment = assumed_utc(day, minutes // 60, minutes % 60, 8.0)
            values.append(solar_zenith_deg(float(row["lat"]), float(row["lon"]), moment))
        else:
            day = date.fromisoformat(str(row["image_date"]))
            start, end = EEST[day.year]
            minutes = int(round((11 + shift_h) * 60))
            moment = assumed_utc(day, minutes // 60, minutes % 60, 3.0 if start <= day < end else 2.0)
            values.append(solar_zenith_deg(KNEZHA[0], KNEZHA[1], moment))
    return np.array(values, float)


def _library(cfg: dict, ranges: dict, n: int, seed: int, centers: dict, fwhm: dict, order: list[str], workers: int) -> np.ndarray:
    from wheat_ppqmtl.prosail_sim import _simulate_chunk

    prosail_cfg = cfg["prosail"]
    names = ["n", "cab", "car", "cbrown", "cw", "cm", "lai", "ala", "hspot", "sza", "vza", "raa", "rsoil", "psoil"]
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    draws = {}
    for i, name in enumerate(names):
        lo, hi = ranges[name]
        draws[name] = np.exp(np.log(lo) + unit[:, i] * (np.log(hi) - np.log(lo))) if name == "hspot" else lo + unit[:, i] * (hi - lo)
    rho = float(prosail_cfg["car_cab_correlation"])
    z_cab = (draws["cab"] - draws["cab"].mean()) / draws["cab"].std()
    z_car = (draws["car"] - draws["car"].mean()) / draws["car"].std()
    mixed = rho * z_cab + np.sqrt(max(0.0, 1.0 - rho ** 2)) * z_car
    draws["car"] = np.clip(mixed * draws["car"].std() + draws["car"].mean(), ranges["car"][0], ranges["car"][1])
    rng = np.random.default_rng(seed + 2)
    gain = rng.uniform(*prosail_cfg["domain_randomization"]["gain"], size=(n, len(order)))
    sigma = rng.uniform(*prosail_cfg["domain_randomization"]["noise_sigma"], size=n)
    noise = rng.normal(0.0, 1.0, size=(n, len(order))) * sigma[:, None]
    bands = np.zeros((n, len(order)))
    payloads = [{"indices": c.astype(int), "draws": {k: v[c] for k, v in draws.items()}, "centers": centers, "fwhm": fwhm,
                 "order": order, "anthocyanin": float(prosail_cfg["anthocyanin"]),
                 "prospect_version": prosail_cfg["prospect_version"], "gain": gain[c], "noise": noise[c]}
                for c in np.array_split(np.arange(n), workers * 2)]
    for index, values in _parallel(_simulate_chunk, payloads, workers):
        bands[index] = values
    return bands


def _within_spearman(score, err, groups) -> float:
    rows = []
    for key in pd.unique(groups):
        mask = (groups == key) & np.isfinite(score) & np.isfinite(err)
        if mask.sum() >= 5 and np.std(score[mask]) > 1e-12:
            rows.append(spearmanr(score[mask], err[mask]).statistic)
    return float(np.mean(rows)) if rows else np.nan


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    workers = max(2, min(20, (os.cpu_count() or 4) - 4))
    ranges = cfg["prosail"]["ranges"]
    seed = int(cfg["prosail"]["seed"])
    features = read_table(root / "data" / "features" / "features.csv")[["sample_id", "lat", "lon"]]
    gao = pd.read_csv(root / "outputs" / "predictions" / "gaocheng_lodo_plot_diagnostics.csv").merge(features, on="sample_id", how="left")
    kz = pd.read_csv(root / "outputs" / "predictions" / "bulgaria_plot_diagnostics.csv")
    w_gao = _weights(cfg["data"]["bands"]["centers_nm"], cfg["data"]["bands"]["fwhm_nm"], GAO)
    w_seq = _weights(SEQUOIA["centers"], SEQUOIA["fwhm"], FOUR)
    sites = {
        "gaocheng": {"frame": gao, "obs": gao[GAO].to_numpy(float), "weights": w_gao, "group": "date_key",
                     "pred": (gao["y_pred_lcc"], gao["y_pred_lai"]),
                     "true": (gao["y_true_lcc"], np.where(gao["lai_observed"], gao["y_true_lai"], np.nan)),
                     "saved": ("e_rtm", "e_rtm_observed"), "lcc": gao["y_true_lcc"], "lai_err": gao["y_true_lai"] - gao["y_pred_lai"]},
        "knezha": {"frame": kz, "obs": kz[FOUR].to_numpy(float), "weights": w_seq, "group": "Campaign",
                   "pred": (kz["y_pred_lcc"], kz["y_pred_lai"]), "true": (kz["lcc"], kz["lai"]),
                   "saved": ("e_rtm", "e_rtm_observed"), "lcc": kz["lcc"], "lai_err": kz["lai"] - kz["y_pred_lai"]},
    }
    plot_rows = []
    summary = {"convergence": {}, "solar_zenith": {}, "range": {}, "identifiability": {}}
    for site, s in sites.items():
        frame = s["frame"]
        groups = frame[s["group"]].astype(str).to_numpy()
        table = pd.DataFrame({"site": site, "unit_id": frame["sample_id"] if site == "gaocheng" else frame["ESU_ID"],
                              "group": groups})
        print(site, flush=True)
        for traits, (cab, lai), saved in (("predicted", s["pred"], s["saved"][0]), ("observed", s["true"], s["saved"][1])):
            for n_draws in DRAW_SIZES:
                draws = _draws(ranges, n_draws, seed)
                table[f"e_{traits}_{n_draws}"] = _residual(frame, cab, lai, frame["sza_deg"], s["obs"], s["weights"], draws, workers,
                                                          f"{traits} traits, {n_draws} draws")
            reproduced = table[f"e_{traits}_256"].to_numpy(float)
            reference = frame[saved].to_numpy(float)
            ok = np.isfinite(reproduced) & np.isfinite(reference)
            conv = {"max_abs_difference_vs_saved_256": float(np.max(np.abs(reproduced[ok] - reference[ok]))) if ok.any() else None}
            for n_draws in DRAW_SIZES[1:]:
                a, b = table[f"e_{traits}_256"].to_numpy(float), table[f"e_{traits}_{n_draws}"].to_numpy(float)
                m = np.isfinite(a) & np.isfinite(b)
                conv[f"spearman_256_vs_{n_draws}"] = float(spearmanr(a[m], b[m]).statistic)
                conv[f"median_ratio_{n_draws}_to_256"] = float(np.median(b[m] / a[m]))
            conv["median_by_group"] = {
                str(n): {g: float(np.nanmedian(table.loc[table["group"] == g, f"e_{traits}_{n}"])) for g in pd.unique(groups)}
                for n in DRAW_SIZES}
            summary["convergence"].setdefault(site, {})[traits] = conv
        err_lcc = np.abs(s["lcc"].to_numpy(float) - s["pred"][0].to_numpy(float))
        err_lai = np.abs(s["lai_err"].to_numpy(float))
        summary["convergence"][site]["within_group_spearman_with_abs_error"] = {
            str(n): {"lcc": _within_spearman(table[f"e_predicted_{n}"].to_numpy(float), err_lcc, groups),
                     "lai": _within_spearman(table[f"e_predicted_{n}"].to_numpy(float), err_lai, groups)}
            for n in DRAW_SIZES}

        draws = _draws(ranges, 256, seed)
        for label, shift in (("minus", -SHIFT_H), ("plus", SHIFT_H)):
            sza = _sza_shifted(frame, site, shift)
            table[f"sza_{label}"] = sza
            table[f"e_predicted_sza_{label}"] = _residual(frame, s["pred"][0], s["pred"][1], sza, s["obs"], s["weights"], draws,
                                                         workers, f"predicted traits, time {shift:+.1f} h")
        base = table["e_predicted_256"].to_numpy(float)
        summary["solar_zenith"][site] = {
            "sza_range_deg": {"assumed": [float(frame["sza_deg"].min()), float(frame["sza_deg"].max())],
                              "minus": [float(table["sza_minus"].min()), float(table["sza_minus"].max())],
                              "plus": [float(table["sza_plus"].min()), float(table["sza_plus"].max())]},
            **{f"spearman_vs_{label}": float(spearmanr(base, table[f"e_predicted_sza_{label}"], nan_policy="omit").statistic)
               for label in ("minus", "plus")},
            "median_by_group": {label: {g: float(np.nanmedian(table.loc[table["group"] == g, col])) for g in pd.unique(groups)}
                                for label, col in (("assumed", "e_predicted_256"), ("minus", "e_predicted_sza_minus"),
                                                   ("plus", "e_predicted_sza_plus"))},
        }

        observed_lai = s["true"][1]
        lcc_obs = s["lcc"].to_numpy(float)
        use = np.isfinite(np.asarray(observed_lai, float))
        if site == "gaocheng":
            use &= (frame["date_key"] != "2024.6.6").to_numpy()
        idx = np.flatnonzero(use)
        fixed = _draws(ranges, PROFILE_DRAWS_FIXED, seed + 101)
        free = _draws(ranges, PROFILE_DRAWS_FREE, seed + 202, extra={"lai": LAI_FREE})
        print(f"  identifiability profiles for {len(idx)} plots", flush=True)
        prof_fixed, prof_free = _profile(s["obs"][idx], np.asarray(observed_lai, float)[idx], frame["sza_deg"].to_numpy(float)[idx],
                                         s["weights"], fixed, free, workers)
        ident = {}
        for mode, profile in (("lai_fixed", prof_fixed), ("lai_free", prof_free)):
            metrics = _profile_metrics(profile, lcc_obs[idx])
            for key, values in metrics.items():
                col = np.full(len(frame), np.nan)
                col[idx] = values
                table[f"{mode}_{key}"] = col
            g = groups[idx]
            ident[mode] = {
                "n": int(len(idx)),
                **{f"median_width_{tol:g}": float(np.nanmedian(metrics[f"width_{tol:g}"])) for tol in TOLERANCES},
                **{f"share_covering_observed_{tol:g}": float(np.mean(metrics[f"covers_obs_{tol:g}"])) for tol in TOLERANCES},
                "median_relative_range": float(np.nanmedian(metrics["relative_range"])),
                "within_group_spearman_argmin_vs_observed": _within_spearman(metrics["argmin_cab"], lcc_obs[idx], g),
                "pooled_spearman_argmin_vs_observed": float(spearmanr(metrics["argmin_cab"], lcc_obs[idx]).statistic),
                "by_group": {key: {"n": int((g == key).sum()),
                                   **{f"median_width_{tol:g}": float(np.nanmedian(metrics[f"width_{tol:g}"][g == key])) for tol in TOLERANCES},
                                   "median_argmin": float(np.nanmedian(metrics["argmin_cab"][g == key])),
                                   "median_observed": float(np.nanmedian(lcc_obs[idx][g == key]))}
                             for key in pd.unique(g)},
            }
            ident[mode]["mean_profile_relative"] = np.nanmean(profile / np.nanmin(profile, axis=1, keepdims=True), axis=0).tolist()
        ident["cab_grid"] = CAB_GRID.tolist()
        summary["identifiability"][site] = ident
        plot_rows.append(table)

    # Range: lookup-table distance with the literature and the extended lower bounds.
    standard = dict(ranges)
    extended = dict(ranges)
    extended["cab"] = list(EXTENDED["cab"])
    extended["lai"] = list(EXTENDED["lai"])
    for site, centers, fwhm, order in (("gaocheng", cfg["data"]["bands"]["centers_nm"], cfg["data"]["bands"]["fwhm_nm"], GAO),
                                       ("knezha", SEQUOIA["centers"], SEQUOIA["fwhm"], FOUR)):
        lib_std = _library(cfg, standard, 4000, seed + 31, centers, fwhm, order, workers)
        lib_ext = _library(cfg, extended, 4000, seed + 31, centers, fwhm, order, workers)
        frame = sites[site]["frame"]
        x = frame[order].to_numpy(float)
        scaler = StandardScaler().fit(x)
        e_std = cKDTree(scaler.transform(lib_std)).query(scaler.transform(x), k=1)[0]
        e_ext = cKDTree(scaler.transform(lib_ext)).query(scaler.transform(x), k=1)[0]
        table = next(t for t in plot_rows if t["site"].iloc[0] == site)
        table["e_nn_standard"] = e_std
        table["e_nn_extended"] = e_ext
        groups = frame[sites[site]["group"]].astype(str).to_numpy()
        err_lcc = np.abs(sites[site]["lcc"].to_numpy(float) - sites[site]["pred"][0].to_numpy(float))
        summary["range"][site] = {
            "extended_bounds": EXTENDED,
            "spearman_standard_vs_extended": float(spearmanr(e_std, e_ext).statistic),
            "median_standard": float(np.median(e_std)), "median_extended": float(np.median(e_ext)),
            "share_closer_with_extended": float(np.mean(e_ext < e_std - 1e-12)),
            "median_by_group": {g: {"standard": float(np.median(e_std[groups == g])), "extended": float(np.median(e_ext[groups == g]))}
                                for g in pd.unique(groups)},
            "within_group_spearman_with_abs_lcc_error": {"standard": _within_spearman(e_std, err_lcc, groups),
                                                         "extended": _within_spearman(e_ext, err_lcc, groups)},
        }
    pd.concat(plot_rows, ignore_index=True).to_csv(root / "outputs" / "predictions" / "rtm_robustness_plots.csv", index=False)
    dest = root / "outputs" / "metrics" / "rtm_robustness.json"
    dest.write_text(json.dumps(jsonable({
        "note": " ".join(__doc__.split()),
        "draw_sizes": DRAW_SIZES, "cab_grid": CAB_GRID.tolist(), "tolerances": TOLERANCES,
        "profile_draws": {"lai_fixed": PROFILE_DRAWS_FIXED, "lai_free": PROFILE_DRAWS_FREE, "lai_free_range": LAI_FREE},
        "time_shift_h": SHIFT_H,
        **summary,
    }), indent=1), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
