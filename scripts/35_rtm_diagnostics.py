"""Per-plot applicability diagnostics, including a PROSAIL reconstruction residual.

For each plot, PROSAIL is run at the predicted chlorophyll and leaf area while the
other leaf, canopy and soil parameters are drawn from the literature ranges of
the lookup table. The reconstruction residual is the smallest band-RMS
difference between the observed plot reflectance and those simulations. The
same residual is computed at the observed traits where both labels exist.

Scores are standardized with rows that the model did not use for gradient
updates on the held-out date: training rows for distances and calibration rows
for ensemble spread and reconstruction residual. Weights are fixed at one third.
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
from wheat_ppqmtl.baselines import _fit_classical, _predict
from wheat_ppqmtl.indices import gaussian_srf
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import regression_metrics
from wheat_ppqmtl.paths import configure_stdio, configs
from wheat_ppqmtl.solar import assumed_utc, solar_zenith_deg

WAVELENGTHS = np.arange(400, 2501)
GAO = ["B", "G", "R", "RE", "NIR"]
FOUR = ["G", "R", "RE", "NIR"]
SEQUOIA = {"centers": {"G": 550, "R": 660, "RE": 735, "NIR": 790}, "fwhm": {"G": 40, "R": 40, "RE": 10, "NIR": 40}}
NUISANCE = ["n", "car", "cbrown", "cw", "cm", "ala", "hspot", "rsoil", "psoil"]
N_DRAWS = 256
KNEZHA = (43.495, 24.081)
EEST = {
    2016: (date(2016, 3, 27), date(2016, 10, 30)),
    2017: (date(2017, 3, 26), date(2017, 10, 29)),
    2018: (date(2018, 3, 25), date(2018, 10, 28)),
}
COVERAGE = np.round(np.arange(1.0, 0.499, -0.05), 2)
EXP = {"baselines": {"plsr_components": [1, 2, 3]}}


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
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _weights(centers: dict, fwhm: dict, order: list[str]) -> np.ndarray:
    return np.vstack([gaussian_srf(WAVELENGTHS, centers[name], fwhm[name]) for name in order])


def _nuisance_draws(ranges: dict, seed: int) -> dict:
    unit = qmc.LatinHypercube(d=len(NUISANCE), seed=seed).random(N_DRAWS)
    draws = {}
    for i, name in enumerate(NUISANCE):
        lo, hi = ranges[name]
        if name == "hspot":
            draws[name] = np.exp(np.log(lo) + unit[:, i] * (np.log(hi) - np.log(lo)))
        else:
            draws[name] = lo + unit[:, i] * (hi - lo)
    return draws


def _reconstruct_chunk(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    import prosail

    draws = payload["draws"]
    weights = payload["weights"]
    best = np.full(len(payload["cab"]), np.nan)
    for i in range(len(payload["cab"])):
        cab, lai, sza = float(payload["cab"][i]), float(payload["lai"][i]), float(payload["sza"][i])
        obs = payload["obs"][i]
        if not (np.isfinite(cab) and np.isfinite(lai) and np.isfinite(obs).all()):
            continue
        sims = np.empty((N_DRAWS, weights.shape[0]))
        for j in range(N_DRAWS):
            spectrum = prosail.run_prosail(
                draws["n"][j], cab, draws["car"][j], draws["cbrown"][j], draws["cw"][j], draws["cm"][j],
                lai, draws["ala"][j], draws["hspot"][j], sza, 0.0, 0.0,
                ant=0.0, prospect_version="D", typelidf=2,
                rsoil=float(draws["rsoil"][j]), psoil=float(draws["psoil"][j]),
            )
            sims[j] = weights @ np.asarray(spectrum, dtype=float).reshape(-1)
        best[i] = float(np.sqrt(np.mean((sims - obs[None, :]) ** 2, axis=1)).min())
    return payload["index"], best


def _reconstruct(tasks: pd.DataFrame, obs: np.ndarray, weights: np.ndarray, draws: dict, label: str) -> np.ndarray:
    """Return the minimum band-RMS residual for each task row."""
    cab = np.clip(tasks["cab"].to_numpy(float), 0.5, 100.0)
    lai = np.clip(tasks["lai"].to_numpy(float), 0.05, 10.5)
    sza = tasks["sza"].to_numpy(float)
    out = np.full(len(tasks), np.nan)
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    chunks = [chunk for chunk in np.array_split(np.arange(len(tasks)), workers * 4) if len(chunk)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_reconstruct_chunk, {
                "index": chunk, "cab": cab[chunk], "lai": lai[chunk], "sza": sza[chunk], "obs": obs[chunk],
                "weights": weights, "draws": draws,
            })
            for chunk in chunks
        ]
        for future in futures:
            index, values = future.result()
            out[index] = values
    print(f"reconstructed {label}: n={len(tasks)} median={np.nanmedian(out):.4f}", flush=True)
    return out


def _sequoia_library(cfg: dict, n: int, seed: int) -> pd.DataFrame:
    from wheat_ppqmtl.prosail_sim import _simulate_chunk

    prosail_cfg = cfg["prosail"]
    ranges = prosail_cfg["ranges"]
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
    gain = rng.uniform(*prosail_cfg["domain_randomization"]["gain"], size=(n, len(FOUR)))
    sigma = rng.uniform(*prosail_cfg["domain_randomization"]["noise_sigma"], size=n)
    noise = rng.normal(0.0, 1.0, size=(n, len(FOUR))) * sigma[:, None]
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    bands = np.zeros((n, len(FOUR)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for chunk in np.array_split(np.arange(n), workers * 2):
            futures.append(pool.submit(_simulate_chunk, {
                "indices": chunk.astype(int),
                "draws": {name: values[chunk] for name, values in draws.items()},
                "centers": SEQUOIA["centers"], "fwhm": SEQUOIA["fwhm"], "order": FOUR,
                "anthocyanin": float(prosail_cfg["anthocyanin"]),
                "prospect_version": prosail_cfg["prospect_version"],
                "gain": gain[chunk], "noise": noise[chunk],
            }))
        for future in futures:
            index, values = future.result()
            bands[index] = values
    return pd.DataFrame(bands, columns=FOUR)


def _di_threshold(train_di: np.ndarray) -> float:
    q1, q3 = np.nanquantile(train_di, [0.25, 0.75])
    return float(min(q3 + 1.5 * (q3 - q1), np.nanmax(train_di)))


def _zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mu = np.nanmean(reference)
    sd = np.nanstd(reference)
    return (values - mu) / (sd if sd > 1e-8 else 1.0)


def _ensemble(frame: pd.DataFrame, role: str, target: str) -> pd.DataFrame:
    part = frame[(frame["role"] == role) & (frame["target"] == target)]
    return part.groupby(["outer_fold", "sample_id"], as_index=False).agg(
        date_key=("date_key", "first"), y_true=("y_true", "first"), lai_qc=("lai_qc", "first"),
        y_pred=("y_pred", "mean"), y_std=("y_pred", "std"),
    )


def _risk_coverage(frame: pd.DataFrame, group: str, y: str, pred: str, scores: list[str], rng) -> dict:
    data = frame.dropna(subset=[y, pred]).copy()
    data["abs_err"] = (data[y] - data[pred]).abs()
    curves = {}
    for score in scores + ["random"]:
        within, across = [], []
        for c in COVERAGE:
            kept_sq = []
            for _, part in data.groupby(group):
                keep = max(1, int(round(len(part) * c)))
                if score == "random":
                    sq = []
                    for _ in range(200):
                        idx = rng.permutation(len(part))[:keep]
                        sq.append(np.mean(part["abs_err"].to_numpy()[idx] ** 2))
                    kept_sq.append((keep, float(np.mean(sq))))
                else:
                    order = np.argsort(part[score].to_numpy(float), kind="stable")[:keep]
                    kept_sq.append((keep, float(np.mean(part["abs_err"].to_numpy()[order] ** 2))))
            total = sum(k for k, _ in kept_sq)
            within.append(float(np.sqrt(sum(k * v for k, v in kept_sq) / total)))
            if score == "random":
                across.append(within[-1])
            else:
                keep = max(1, int(round(len(data) * c)))
                order = np.argsort(data[score].to_numpy(float), kind="stable")[:keep]
                across.append(float(np.sqrt(np.mean(data["abs_err"].to_numpy()[order] ** 2))))
        curves[score] = {"coverage": COVERAGE.tolist(), "rmse_within_group": within, "rmse_across_groups": across,
                         "aurc_within_group": float(np.mean(within)), "aurc_across_groups": float(np.mean(across))}
    return curves


def _correlations(frame: pd.DataFrame, group: str, y: str, pred: str, scores: list[str]) -> dict:
    data = frame.dropna(subset=[y, pred]).copy()
    data["abs_err"] = (data[y] - data[pred]).abs()
    out = {"pooled": {}, "by_group": []}
    for score in scores:
        ok = data[score].notna()
        out["pooled"][score] = float(spearmanr(data.loc[ok, score], data.loc[ok, "abs_err"]).correlation)
    for key, part in data.groupby(group):
        row = {"group": str(key), "n": int(len(part)), **regression_metrics(part[y], part[pred])}
        for score in scores:
            ok = part[score].notna()
            if ok.sum() >= 5 and part.loc[ok, score].std() > 1e-12:
                row[f"spearman_{score}"] = float(spearmanr(part.loc[ok, score], part.loc[ok, "abs_err"]).correlation)
        out["by_group"].append(row)
    return out


def gaocheng(cfg: dict, draws: dict, library: np.ndarray) -> tuple[pd.DataFrame, dict]:
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")
    splits = read_table(root / "splits" / "lodo.csv")
    predictions = pd.read_csv(root / "outputs" / "predictions" / "spectral__lodo.csv")
    test = _ensemble(predictions, "test", "lcc").merge(
        _ensemble(predictions, "test", "lai")[["outer_fold", "sample_id", "y_true", "y_pred", "y_std"]],
        on=["outer_fold", "sample_id"], suffixes=("_lcc", "_lai"),
    )
    cal = _ensemble(predictions, "calibration", "lcc").merge(
        _ensemble(predictions, "calibration", "lai")[["outer_fold", "sample_id", "y_pred", "y_std"]],
        on=["outer_fold", "sample_id"], suffixes=("_lcc", "_lai"),
    )
    columns = ["sample_id", "x", "y", "sza_deg", *GAO]
    test = test.merge(features[columns], on="sample_id")
    cal = cal.merge(features[columns], on="sample_id")
    weights = _weights(cfg["data"]["bands"]["centers_nm"], cfg["data"]["bands"]["fwhm_nm"], GAO)

    tasks_pred = pd.DataFrame({"cab": test["y_pred_lcc"], "lai": test["y_pred_lai"], "sza": test["sza_deg"]})
    test["e_rtm"] = _reconstruct(tasks_pred, test[GAO].to_numpy(float), weights, draws, "Gaocheng held-out, predicted traits")
    observed = np.isclose(pd.to_numeric(test["lai_qc"], errors="coerce").to_numpy(), 1.0) & test["y_true_lai"].notna().to_numpy()
    tasks_true = pd.DataFrame({
        "cab": test["y_true_lcc"], "lai": np.where(observed, test["y_true_lai"], np.nan), "sza": test["sza_deg"],
    })
    test["e_rtm_observed"] = _reconstruct(tasks_true, test[GAO].to_numpy(float), weights, draws, "Gaocheng held-out, observed traits")
    tasks_cal = pd.DataFrame({"cab": cal["y_pred_lcc"], "lai": cal["y_pred_lai"], "sza": cal["sza_deg"]})
    cal["e_rtm"] = _reconstruct(tasks_cal, cal[GAO].to_numpy(float), weights, draws, "Gaocheng calibration rows")

    parts = []
    for fold, held in test.groupby("outer_fold"):
        train = splits[(splits["outer_fold"] == fold) & (splits["role"] == "train")].merge(features[["sample_id", *GAO]], on="sample_id")
        scaler = StandardScaler().fit(train[GAO].to_numpy(float))
        x_train = scaler.transform(train[GAO].to_numpy(float))
        x_test = scaler.transform(held[GAO].to_numpy(float))
        pair = np.sqrt(np.maximum(0.0, ((x_train[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)))
        np.fill_diagonal(pair, np.nan)
        denom = float(np.nanmean(pair))
        days = train["date_key"].to_numpy()
        train_di = np.array([np.nanmin(pair[i, days != day]) for i, day in enumerate(days)]) / denom
        test_di = cKDTree(x_train).query(x_test, k=1)[0] / denom
        lib_tree = cKDTree(scaler.transform(library))
        train_e = lib_tree.query(x_train, k=1)[0]
        test_e = lib_tree.query(x_test, k=1)[0]
        cal_fold = cal[cal["outer_fold"] == fold]
        held = held.copy()
        held["di"] = test_di
        held["di_threshold"] = _di_threshold(train_di)
        held["inside_aoa"] = test_di <= held["di_threshold"]
        held["e_nn"] = test_e
        z_d = _zscore(test_di, train_di)
        z_e = _zscore(test_e, train_e)
        z_r = _zscore(held["e_rtm"].to_numpy(float), cal_fold["e_rtm"].to_numpy(float))
        for target in ("lcc", "lai"):
            z_u = _zscore(held[f"y_std_{target}"].to_numpy(float), cal_fold[f"y_std_{target}"].to_numpy(float))
            held[f"risk_nn_{target}"] = np.nanmean(np.vstack([z_d, z_e, z_u]), axis=0)
            held[f"risk_rtm_{target}"] = np.nanmean(np.vstack([z_d, z_r, z_u]), axis=0)
        parts.append(held)
    frame = pd.concat(parts, ignore_index=True)
    frame["lai_observed"] = np.isclose(pd.to_numeric(frame["lai_qc"], errors="coerce"), 1.0) & frame["y_true_lai"].notna()
    rng = np.random.default_rng(42)
    summary = {"n": int(len(frame)), "targets": {}, "by_date": []}
    for target in ("lcc", "lai"):
        data = frame if target == "lcc" else frame[frame["lai_observed"]]
        scores = ["di", "e_nn", "e_rtm", f"y_std_{target}", f"risk_nn_{target}", f"risk_rtm_{target}"]
        summary["targets"][target] = {
            "correlations": _correlations(data, "date_key", f"y_true_{target}", f"y_pred_{target}", scores),
            "risk_coverage": _risk_coverage(data, "date_key", f"y_true_{target}", f"y_pred_{target}", scores, rng),
        }
    for day, part in frame.groupby("date_key"):
        summary["by_date"].append({
            "date": str(day), "n": int(len(part)),
            "inside_aoa_fraction": float(part["inside_aoa"].mean()),
            "e_rtm_predicted_median": float(part["e_rtm"].median()),
            "e_rtm_observed_median": float(part["e_rtm_observed"].median()) if part["e_rtm_observed"].notna().any() else None,
            "n_observed_traits": int(part["e_rtm_observed"].notna().sum()),
            "e_nn_median": float(part["e_nn"].median()),
        })
    return frame, summary


def bulgaria(cfg: dict, draws: dict, library_gao: np.ndarray) -> tuple[pd.DataFrame, dict]:
    root = cfg["root"]
    features = read_table(root / "data" / "features" / "features.csv")
    splits = read_table(root / "splits" / "grouped_random.csv")
    uav = pd.read_csv(root / "data" / "external" / "bulgaria" / "uav_esu_reflectance.csv")
    uav["year"] = np.where(uav["Campaign"].isin(["November_2017", "April_2018"]), "2017-2018", "2016-2017")
    x_bg = uav[FOUR].to_numpy(float)
    oof = {target: np.full(len(features), np.nan) for target in ("lcc", "lai")}
    stacks = {target: [] for target in ("lcc", "lai")}
    gao_stacks = {target: [] for target in ("lcc", "lai")}
    fold_of = {}
    for fold in sorted(splits["outer_fold"].unique()):
        merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
        for sid in merged.loc[merged["role"] == "test", "sample_id"]:
            fold_of[sid] = fold
        for target in ("lcc", "lai"):
            part = merged
            if target == "lai":
                part = merged[np.isclose(pd.to_numeric(merged["lai_qc"], errors="coerce"), 1.0)]
            train = part[part["role"] == "train"].dropna(subset=FOUR + [target])
            tune = part[part["role"] == "tune"].dropna(subset=FOUR + [target])
            model, bundle = _fit_classical(
                "plsr", train[FOUR].to_numpy(float), train[target].to_numpy(float),
                tune[FOUR].to_numpy(float), tune[target].to_numpy(float), 42, EXP,
            )
            stacks[target].append(_predict("plsr", model, bundle, x_bg)[0])
            gao_stacks[target].append(_predict("plsr", model, bundle, features[FOUR].to_numpy(float))[0])
            test_ids = merged.loc[merged["role"] == "test", "sample_id"]
            index = features.index[features["sample_id"].isin(test_ids)]
            oof[target][index] = _predict("plsr", model, bundle, features.loc[index, FOUR].to_numpy(float))[0]
    for target in ("lcc", "lai"):
        uav[f"y_pred_{target}"] = np.mean(np.vstack(stacks[target]), axis=0)
        uav[f"y_std_{target}"] = np.std(np.vstack(stacks[target]), axis=0, ddof=1)
    gao_std = {target: np.std(np.vstack(gao_stacks[target]), axis=0, ddof=1) for target in ("lcc", "lai")}

    def _sza(day: str) -> float:
        when = date.fromisoformat(day)
        start, end = EEST[when.year]
        moment = assumed_utc(when, 11, 0, 3.0 if start <= when < end else 2.0)
        return float(solar_zenith_deg(KNEZHA[0], KNEZHA[1], moment))

    uav["sza_deg"] = uav["image_date"].map(_sza)
    seq_weights = _weights(SEQUOIA["centers"], SEQUOIA["fwhm"], FOUR)
    gao_weights = _weights(cfg["data"]["bands"]["centers_nm"], cfg["data"]["bands"]["fwhm_nm"], FOUR)
    uav["e_rtm"] = _reconstruct(
        pd.DataFrame({"cab": uav["y_pred_lcc"], "lai": uav["y_pred_lai"], "sza": uav["sza_deg"]}),
        x_bg, seq_weights, draws, "Bulgaria, predicted traits",
    )
    uav["e_rtm_observed"] = _reconstruct(
        pd.DataFrame({"cab": uav["lcc"], "lai": uav["lai"], "sza": uav["sza_deg"]}),
        x_bg, seq_weights, draws, "Bulgaria, observed traits",
    )
    gao_ref = features[["sample_id", "sza_deg", *FOUR]].copy()
    gao_ref["e_rtm"] = _reconstruct(
        pd.DataFrame({"cab": oof["lcc"], "lai": oof["lai"], "sza": gao_ref["sza_deg"]}),
        gao_ref[FOUR].to_numpy(float), gao_weights, draws, "Gaocheng out-of-fold, four bands",
    )

    scaler = StandardScaler().fit(features[FOUR].to_numpy(float))
    x_gao = scaler.transform(features[FOUR].to_numpy(float))
    x_query = scaler.transform(x_bg)
    pair = np.sqrt(np.maximum(0.0, ((x_gao[:, None, :] - x_gao[None, :, :]) ** 2).sum(axis=2)))
    np.fill_diagonal(pair, np.nan)
    denom = float(np.nanmean(pair))
    folds = features["sample_id"].map(fold_of).to_numpy()
    train_di = np.array([np.nanmin(pair[i, folds != folds[i]]) for i in range(len(features))]) / denom
    uav["di"] = cKDTree(x_gao).query(x_query, k=1)[0] / denom
    uav["di_threshold"] = _di_threshold(train_di)
    uav["inside_aoa"] = uav["di"] <= uav["di_threshold"]
    seq_library = _sequoia_library(cfg, 4000, int(cfg["prosail"]["seed"]))
    uav["e_nn"] = cKDTree(scaler.transform(seq_library[FOUR].to_numpy(float))).query(x_query, k=1)[0]
    gao_e = cKDTree(scaler.transform(library_gao[:, 1:])).query(x_gao, k=1)[0]
    z_d = _zscore(uav["di"].to_numpy(float), train_di)
    z_e = _zscore(uav["e_nn"].to_numpy(float), gao_e)
    z_r = _zscore(uav["e_rtm"].to_numpy(float), gao_ref["e_rtm"].to_numpy(float))
    for target in ("lcc", "lai"):
        z_u = _zscore(uav[f"y_std_{target}"].to_numpy(float), gao_std[target])
        uav[f"risk_nn_{target}"] = np.nanmean(np.vstack([z_d, z_e, z_u]), axis=0)
        uav[f"risk_rtm_{target}"] = np.nanmean(np.vstack([z_d, z_r, z_u]), axis=0)
    rng = np.random.default_rng(42)
    summary = {
        "n": int(len(uav)), "di_threshold": float(uav["di_threshold"].iloc[0]),
        "inside_aoa_fraction": float(uav["inside_aoa"].mean()),
        "gaocheng_reference_e_rtm_median": float(np.nanmedian(gao_ref["e_rtm"])),
        "targets": {}, "by_campaign": [],
    }
    for target in ("lcc", "lai"):
        scores = ["di", "e_nn", "e_rtm", f"y_std_{target}", f"risk_nn_{target}", f"risk_rtm_{target}"]
        summary["targets"][target] = {
            "correlations": _correlations(uav.rename(columns={target: f"y_true_{target}"}), "Campaign", f"y_true_{target}", f"y_pred_{target}", scores),
            "risk_coverage": _risk_coverage(uav.rename(columns={target: f"y_true_{target}"}), "Campaign", f"y_true_{target}", f"y_pred_{target}", scores, rng),
        }
    for campaign, part in uav.groupby("Campaign"):
        summary["by_campaign"].append({
            "campaign": str(campaign), "n": int(len(part)),
            "inside_aoa_fraction": float(part["inside_aoa"].mean()),
            "di_median": float(part["di"].median()),
            "e_rtm_predicted_median": float(part["e_rtm"].median()),
            "e_rtm_observed_median": float(part["e_rtm_observed"].median()),
        })
    return uav, summary


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    draws = _nuisance_draws(cfg["prosail"]["ranges"], int(cfg["prosail"]["seed"]))
    library = pd.read_csv(root / "data" / "synthetic" / "prosail.csv", usecols=GAO)
    if len(library) > 4000:
        library = library.sample(n=4000, random_state=42)
    library = library[GAO].to_numpy(float)
    gao_frame, gao_summary = gaocheng(cfg, draws, library)
    gao_frame.to_csv(root / "outputs" / "predictions" / "gaocheng_lodo_plot_diagnostics.csv", index=False)
    bg_frame, bg_summary = bulgaria(cfg, draws, library)
    bg_frame.to_csv(root / "outputs" / "predictions" / "bulgaria_plot_diagnostics.csv", index=False)
    dest = root / "outputs" / "metrics" / "rtm_diagnostics.json"
    dest.write_text(json.dumps(_jsonable({
        "note": (
            "Reconstruction residual: minimum band-RMS difference between observed reflectance and PROSAIL "
            f"spectra at the predicted (or observed) chlorophyll and LAI, over {N_DRAWS} Latin-hypercube draws of "
            "the remaining parameters from the lookup-table ranges. View zenith 0. Solar zenith is the assumed "
            "11:00 local time. Risk scores average training-standardized dissimilarity, a residual and ensemble spread."
        ),
        "n_nuisance_draws": N_DRAWS,
        "gaocheng_lodo": gao_summary,
        "bulgaria_zero_shot": bg_summary,
    }), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
