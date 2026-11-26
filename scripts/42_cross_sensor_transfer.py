"""Cross-sensor transfer between the Gaocheng camera bands and the Knezha Sequoia bands.

T1  raw four-band transfer from Gaocheng to Knezha.
T2  the T1 models applied to Knezha bands after PROSAIL band harmonization.
T3  training restricted to Gaocheng dates whose growth stage overlaps the Knezha campaign.
T4  T3 models applied to harmonized Knezha bands.
Few-label calibration of the T2 predictions (T5) is in script 44.
Reverse transfer trains on Knezha (raw or harmonized bands) and predicts Gaocheng.

Band harmonization maps Sequoia band values to Gaocheng-equivalent bands with
linear models fitted on PROSAIL canopy spectra integrated with both sets of
spectral response functions. The primary map is a separate linear adjustment
per band, as in bandpass adjustment of Landsat and Sentinel-2; a four-band
linear map (suffix m) is a sensitivity variant. Harmonization addresses
band-pass differences only, not differences in ground instruments, plot
support, illumination or radiometric processing.

Models are fitted on the grouped-random training rows of each Gaocheng outer
fold, tuned on the matching tune rows, and averaged over the five fold models.
No DOY features are used because day of year does not transfer between sites.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from wheat_ppqmtl.indices import gaussian_srf, vegetation_indices
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import regression_metrics
from wheat_ppqmtl.multilevel import (
    between_inner_agreement,
    error_decomposition,
    grouped_agreement,
    jsonable,
    percentile_interval,
    stratified_cluster_draws,
    within_inner_agreement,
)
from wheat_ppqmtl.paths import configure_stdio, configs

WAVELENGTHS = np.arange(400, 2501)
FOUR = ["G", "R", "RE", "NIR"]
VI = ["ndvi", "ndre", "gndvi", "cire", "rvi", "osavi", "evi2"]
FEATURE_SETS = {"bands": FOUR, "bands_vi": FOUR + VI}
SEQUOIA = {"centers": {"G": 550, "R": 660, "RE": 735, "NIR": 790}, "fwhm": {"G": 40, "R": 40, "RE": 10, "NIR": 40}}
MODELS = ["plsr", "ridge", "gpr", "svr", "rf", "xgb", "mlp"]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
N_PAIRS = 20000
DRAWS = 2000
TARGETS = ("lcc", "lai")
PARAM = {"lcc": "cab", "lai": "lai"}
# Gaocheng Feekes stages expressed as Zadoks ranges; Knezha ranges come from the ESU phenology table.
GAO_ZADOKS = {
    "2024.3.31": (30, 31),
    "2024.4.12": (32, 45),
    "2024.4.25": (61, 69),
    "2024.5.18": (75, 85),
    "2024.6.6": (87, 92),
}
REVERSE_FOLDS = 4
REVERSE_NAMES = {
    ("raw", "all"): "R1_raw", ("harm", "all"): "R2_harmonized", ("harm_m", "all"): "R2m_harmonized_fourband",
    ("raw", "matched"): "R3_matched_raw", ("harm", "matched"): "R4_matched_harmonized",
    ("harm_m", "matched"): "R4m_matched_harmonized_fourband",
}
FORWARD_ALL = ("T1_raw", "T2_harmonized", "T2m_harmonized_fourband", "source_mean", "prosail_hybrid")
REVERSE_ALL = ("R1_raw", "R2_harmonized", "R2m_harmonized_fourband", "source_mean")
SPRING = ("March_2017", "April_2017", "May_2017", "April_2018")


def _init_worker() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")


def _knezha_zadoks(path) -> dict[str, tuple[int, int]]:
    lines = path.read_bytes().decode("utf-8-sig").splitlines()[1:]
    pattern = re.compile(r"\s(?P<campaign>[A-Za-z]+_\d{4})\s\d{4}-\d{2}-\d{2}\s\S+\s(?P<stage>Z\d{2}(?:\s*to\s*Z\d{2})?)")
    codes = defaultdict(list)
    for line in lines:
        match = pattern.search(line)
        if match:
            codes[match["campaign"]].extend(int(v) for v in re.findall(r"Z(\d{2})", match["stage"]))
    return {key: (min(values), max(values)) for key, values in codes.items()}


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _with_vi(frame: pd.DataFrame) -> pd.DataFrame:
    out = vegetation_indices(frame.assign(B=np.nan))
    return out


def _matrix(frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    return frame[FEATURE_SETS[feature_set]].to_numpy(float)


def _design(cfg: dict, n: int, seed: int) -> dict:
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
    return draws


def _pairs_chunk(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import prosail

    draws = payload["draws"]
    count = len(payload["index"])
    gao = np.zeros((count, len(FOUR)))
    seq = np.zeros((count, len(FOUR)))
    for i in range(count):
        spectrum = prosail.run_prosail(
            draws["n"][i], draws["cab"][i], draws["car"][i], draws["cbrown"][i], draws["cw"][i], draws["cm"][i],
            draws["lai"][i], draws["ala"][i], draws["hspot"][i], draws["sza"][i], draws["vza"][i], draws["raa"][i],
            ant=payload["ant"], prospect_version=payload["version"], typelidf=2,
            rsoil=float(draws["rsoil"][i]), psoil=float(draws["psoil"][i]),
        )
        values = np.asarray(spectrum, dtype=float).reshape(-1)
        gao[i] = payload["w_gao"] @ values
        seq[i] = payload["w_seq"] @ values
    return payload["index"], gao, seq


def _simulate_pairs(cfg: dict, workers: int) -> pd.DataFrame:
    draws = _design(cfg, N_PAIRS, int(cfg["prosail"]["seed"]) + 7)
    bands = cfg["data"]["bands"]
    w_gao = np.vstack([gaussian_srf(WAVELENGTHS, bands["centers_nm"][b], bands["fwhm_nm"][b]) for b in FOUR])
    w_seq = np.vstack([gaussian_srf(WAVELENGTHS, SEQUOIA["centers"][b], SEQUOIA["fwhm"][b]) for b in FOUR])
    gao = np.zeros((N_PAIRS, len(FOUR)))
    seq = np.zeros((N_PAIRS, len(FOUR)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_pairs_chunk, {
                "index": chunk, "draws": {k: v[chunk] for k, v in draws.items()},
                "w_gao": w_gao, "w_seq": w_seq,
                "ant": float(cfg["prosail"]["anthocyanin"]), "version": cfg["prosail"]["prospect_version"],
            })
            for chunk in np.array_split(np.arange(N_PAIRS), workers * 4)
        ]
        for future in futures:
            index, g, s = future.result()
            gao[index] = g
            seq[index] = s
    frame = pd.DataFrame(draws)
    for i, band in enumerate(FOUR):
        frame[f"gao_{band}"] = gao[:, i]
        frame[f"seq_{band}"] = seq[:, i]
    return frame


class _BandMap:
    """Separate linear adjustment for each band."""

    def __init__(self, seq: np.ndarray, gao: np.ndarray):
        self.models = [LinearRegression().fit(seq[:, [j]], gao[:, j]) for j in range(seq.shape[1])]

    def predict(self, seq: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(seq[:, [j]]) for j, m in enumerate(self.models)])


def _harmonization(pairs: pd.DataFrame) -> tuple[_BandMap, LinearRegression, dict]:
    seq = pairs[[f"seq_{b}" for b in FOUR]].to_numpy(float)
    gao = pairs[[f"gao_{b}" for b in FOUR]].to_numpy(float)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(pairs))
    cut = int(0.75 * len(pairs))
    fit_i, test_i = order[:cut], order[cut:]
    multi = LinearRegression().fit(seq[fit_i], gao[fit_i]).predict(seq[test_i])
    single = _BandMap(seq[fit_i], gao[fit_i]).predict(seq[test_i])
    validation = {}
    for j, band in enumerate(FOUR):
        truth = gao[test_i, j]
        validation[band] = {
            "rmse_identity": float(np.sqrt(np.mean((seq[test_i, j] - truth) ** 2))),
            "rmse_same_band_linear": float(np.sqrt(np.mean((single[:, j] - truth) ** 2))),
            "rmse_four_band_linear": float(np.sqrt(np.mean((multi[:, j] - truth) ** 2))),
            "r2_same_band_linear": float(regression_metrics(truth, single[:, j])["r2"]),
            "r2_four_band_linear": float(regression_metrics(truth, multi[:, j])["r2"]),
            "median_gaocheng": float(np.median(truth)),
            "median_sequoia": float(np.median(seq[test_i, j])),
        }
    band_map = _BandMap(seq, gao)
    multi_map = LinearRegression().fit(seq, gao)
    coefficients = {
        "same_band": {band: {"intercept": float(m.intercept_), "slope": float(m.coef_[0])}
                      for band, m in zip(FOUR, band_map.models)},
        "four_band": {band: {"intercept": float(multi_map.intercept_[j]),
                             **{f"seq_{src}": float(multi_map.coef_[j, k]) for k, src in enumerate(FOUR)}}
                      for j, band in enumerate(FOUR)},
    }
    return band_map, multi_map, {"n_pairs": int(len(pairs)), "held_out_fraction": 0.25, "validation": validation,
                                 "coefficients": coefficients}


def _fit_predict(task: dict) -> tuple[tuple, dict]:
    from wheat_ppqmtl.baselines import _fit_classical, _fit_mlp, _predict, _predict_mlp

    name = task["model"]
    x_train, y_train, x_tune, y_tune = task["x_train"], task["y_train"], task["x_tune"], task["y_tune"]
    if name == "mlp":
        model, bundle = _fit_mlp(x_train, y_train, x_tune, y_tune, 42, task["model_cfg"])

        def predict(x):
            return _predict_mlp(model, bundle, x)[0]
    elif name == "ridge":
        scaler = StandardScaler().fit(x_train)
        xt, xv = scaler.transform(x_train), scaler.transform(x_tune)
        scores = [np.sqrt(np.mean((Ridge(alpha=a).fit(xt, y_train).predict(xv) - y_tune) ** 2)) for a in RIDGE_ALPHAS]
        model = Ridge(alpha=float(RIDGE_ALPHAS[int(np.argmin(scores))])).fit(xt, y_train)

        def predict(x):
            return model.predict(scaler.transform(x))
    else:
        model, bundle = _fit_classical(name, x_train, y_train, x_tune, y_tune, 42, task["exp"])

        def predict(x):
            return _predict(name, model, bundle, x)[0]
    return task["key"], {label: np.asarray(predict(x), float) for label, x in task["apply"].items()}


def _exp_for(feature_set: str, exp: dict) -> dict:
    out = json.loads(json.dumps(exp))
    if feature_set == "bands":
        out["baselines"]["plsr_components"] = [1, 2, 3]
    return out


def _score_knezha(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    campaign = frame["group"].to_numpy()
    cf = (frame["group"] + "|" + frame["field"]).to_numpy()
    wc = grouped_agreement(y, p, campaign, min_n=5)
    autumn = [c for c in pd.unique(campaign) if c not in SPRING]
    spring = [c for c in pd.unique(campaign) if c in SPRING]
    out = {**regression_metrics(y, p), "errors": error_decomposition(y, p, campaign, cf),
           "within_campaign": wc["macro"], "by_campaign": wc["by_group"],
           "within_campaign_spring": grouped_agreement(y, p, campaign, min_n=5, exclude=autumn)["macro"],
           "within_campaign_autumn": grouped_agreement(y, p, campaign, min_n=5, exclude=spring)["macro"] if autumn else {},
           "between_fields": between_inner_agreement(y, p, campaign, cf),
           "within_fields": within_inner_agreement(y, p, cf, min_n=3)}
    fields = frame["field"].to_numpy()
    keys = pd.unique(fields)
    members = {key: np.flatnonzero(fields == key) for key in keys}
    rng = np.random.default_rng(42)
    values = []
    for _ in range(DRAWS):
        idx = np.concatenate([members[key] for key in rng.choice(keys, size=keys.size, replace=True)])
        values.append(np.sqrt(np.mean((p[idx] - y[idx]) ** 2)))
    out["rmse_ci95"] = percentile_interval(values)
    return out


def _score_gaocheng(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    day = frame["group"].to_numpy()
    block = (frame["group"] + "|" + frame["field"]).to_numpy()
    green = frame["group"] != "2024.6.6"
    wc = grouped_agreement(y, p, day, min_n=5, exclude=("2024.6.6",))
    out = {**regression_metrics(y, p), "errors": error_decomposition(y, p, day, block),
           "within_date_green": wc["macro"], "by_date": wc["by_group"],
           "green_dates": regression_metrics(y[green.to_numpy()], p[green.to_numpy()])}
    values = [np.sqrt(np.mean((p[idx] - y[idx]) ** 2)) for idx in stratified_cluster_draws(day, block, DRAWS, 42)]
    out["rmse_ci95"] = percentile_interval(values)
    return out


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    workers = max(2, min(16, (os.cpu_count() or 4) - 4))
    base = root / "data" / "external" / "bulgaria"

    knezha = pd.read_csv(base / "uav_esu_reflectance.csv")
    knezha["year"] = np.where(knezha["Campaign"].isin(["November_2017", "April_2018"]), "2017-2018", "2016-2017")
    features = read_table(root / "data" / "features" / "features.csv")
    features["lai_ok"] = np.isclose(pd.to_numeric(features["lai_qc"], errors="coerce"), 1.0)
    splits = read_table(root / "splits" / "grouped_random.csv")

    zadoks = _knezha_zadoks(base / "ESU_Phenological_data.txt")
    matched = {c: [d for d, z in GAO_ZADOKS.items() if _overlap(z, zadoks[c])]
               for c in sorted(knezha["Campaign"].unique()) if c in zadoks}
    matched = {c: d for c, d in matched.items() if d}
    reverse_matched = defaultdict(list)
    for campaign, days in matched.items():
        for day in days:
            reverse_matched[day].append(campaign)
    print("stage matches:", matched, flush=True)

    pairs_path = root / "data" / "synthetic" / "prosail_sensor_pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
    else:
        pairs = _simulate_pairs(cfg, workers)
        pairs.to_csv(pairs_path, index=False)
    band_map, multi_map, harmonization = _harmonization(pairs)
    harmonized = knezha.copy()
    harmonized[FOUR] = band_map.predict(knezha[FOUR].to_numpy(float))
    harmonized_m = knezha.copy()
    harmonized_m[FOUR] = multi_map.predict(knezha[FOUR].to_numpy(float))
    versions = {"raw": _with_vi(knezha), "harm": _with_vi(harmonized), "harm_m": _with_vi(harmonized_m)}
    band_table = knezha[["ESU_ID", "Campaign", "field_id", *FOUR]].copy()
    for band in FOUR:
        band_table[f"{band}_harmonized"] = harmonized[band]
        band_table[f"{band}_harmonized_fourband"] = harmonized_m[band]
    band_table.to_csv(root / "outputs" / "predictions" / "knezha_harmonized_bands.csv", index=False)
    harmonization["band_medians"] = {
        "gaocheng_all_dates": {b: float(features[b].median()) for b in FOUR},
        "knezha_raw": {b: float(knezha[b].median()) for b in FOUR},
        "knezha_harmonized": {b: float(harmonized[b].median()) for b in FOUR},
        "knezha_harmonized_fourband": {b: float(harmonized_m[b].median()) for b in FOUR},
        "by_campaign": {
            c: {"raw": {b: float(g[b].median()) for b in FOUR},
                "harmonized": {b: float(harmonized.loc[g.index, b].median()) for b in FOUR},
                "harmonized_fourband": {b: float(harmonized_m.loc[g.index, b].median()) for b in FOUR}}
            for c, g in knezha.groupby("Campaign")
        },
        "gaocheng_by_date": {d: {b: float(g[b].median()) for b in FOUR} for d, g in features.groupby("date_key")},
    }
    print("harmonization (held-out PROSAIL RMSE identity -> same-band -> four-band):",
          {b: (round(v["rmse_identity"], 4), round(v["rmse_same_band_linear"], 4), round(v["rmse_four_band_linear"], 4))
           for b, v in harmonization["validation"].items()}, flush=True)

    # Knezha four-band harmonized to Gaocheng scale for reverse transfer
    gao_all = features.copy()
    tasks = []
    exp = cfg["experiment"]
    model_cfg = cfg["model"]

    # Forward: Gaocheng -> Knezha
    for target in TARGETS:
        kz_mask = knezha[target].notna().to_numpy()
        for fold in sorted(splits["outer_fold"].unique()):
            merged = splits[splits["outer_fold"] == fold].merge(features, on="sample_id", suffixes=("", "_f"))
            merged = merged.dropna(subset=FOUR + VI + [target])
            if target == "lai":
                merged = merged[merged["lai_ok"]]
            train_sets = [("all", None)] + [(c, days) for c, days in matched.items()]
            for set_name, days in train_sets:
                tr = merged[merged["role"] == "train"]
                tu = merged[merged["role"] == "tune"]
                if days is not None:
                    tr = tr[tr["date_key"].isin(days)]
                    tu = tu[tu["date_key"].isin(days)]
                if len(tr) < 12 or len(tu) < 3:
                    continue
                te = merged[merged["role"] == "test"]
                if set_name == "all":
                    rows = kz_mask
                else:
                    rows = kz_mask & (knezha["Campaign"] == set_name).to_numpy()
                for fs in FEATURE_SETS:
                    apply = {f"kz_{name}": _matrix(frame.loc[rows], fs) for name, frame in versions.items()}
                    if set_name == "all":
                        apply["gao_test"] = _matrix(te, fs)
                    for model in MODELS:
                        tasks.append({
                            "key": ("forward", set_name, target, fs, model, int(fold)),
                            "model": model, "exp": _exp_for(fs, exp), "model_cfg": model_cfg,
                            "x_train": _matrix(tr, fs), "y_train": tr[target].to_numpy(float),
                            "x_tune": _matrix(tu, fs), "y_tune": tu[target].to_numpy(float),
                            "apply": apply,
                            "rows": np.flatnonzero(rows), "test_ids": te["sample_id"].tolist() if set_name == "all" else [],
                        })

    # Reverse: Knezha -> Gaocheng
    fields = sorted(knezha["field_id"].unique())
    perm = np.random.default_rng(42).permutation(len(fields))
    field_fold = {fields[i]: int(rank % REVERSE_FOLDS) for rank, i in enumerate(perm)}
    knezha["rev_fold"] = knezha["field_id"].map(field_fold)
    gao_raw = gao_all.dropna(subset=FOUR + VI).reset_index(drop=True)
    for target in TARGETS:
        for treatment, source in versions.items():
            source = source.assign(rev_fold=knezha["rev_fold"]).dropna(subset=FOUR + VI + [target])
            train_sets = [("all", None)] + [(day, camps) for day, camps in reverse_matched.items()]
            for set_name, camps in train_sets:
                sub = source if camps is None else source[source["Campaign"].isin(camps)]
                gao_rows = np.ones(len(gao_raw), dtype=bool) if camps is None else (gao_raw["date_key"] == set_name).to_numpy()
                for k in range(REVERSE_FOLDS):
                    tr = sub[sub["rev_fold"] != k]
                    tu = sub[sub["rev_fold"] == k]
                    if len(tr) < 12 or len(tu) < 3:
                        continue
                    for fs in FEATURE_SETS:
                        for model in MODELS:
                            tasks.append({
                                "key": ("reverse_" + treatment, set_name, target, fs, model, k),
                                "model": model, "exp": _exp_for(fs, exp), "model_cfg": model_cfg,
                                "x_train": _matrix(tr, fs), "y_train": tr[target].to_numpy(float),
                                "x_tune": _matrix(tu, fs), "y_tune": tu[target].to_numpy(float),
                                "apply": {"gao": _matrix(gao_raw.loc[gao_rows], fs)},
                                "rows": np.flatnonzero(gao_rows), "test_ids": [],
                            })
    print(f"fitting {len(tasks)} models on {workers} workers", flush=True)
    results = {}
    meta = {task["key"]: {"rows": task["rows"], "test_ids": task["test_ids"]} for task in tasks}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        stripped = [{k: v for k, v in task.items() if k not in ("rows", "test_ids")} for task in tasks]
        for i, (key, outputs) in enumerate(pool.map(_fit_predict, stripped, chunksize=1)):
            results[key] = outputs
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)}", flush=True)

    # PROSAIL hybrid models: one random forest per sensor, trained on simulations with that sensor's bands.
    rng = np.random.default_rng(int(cfg["prosail"]["seed"]) + 9)
    dr = cfg["prosail"]["domain_randomization"]
    hybrid = {}
    for sensor in ("seq", "gao"):
        bands = pairs[[f"{sensor}_{b}" for b in FOUR]].to_numpy(float)
        gain = rng.uniform(*dr["gain"], size=bands.shape)
        sigma = rng.uniform(*dr["noise_sigma"], size=len(bands))
        noisy = np.clip(bands * gain + rng.normal(0.0, 1.0, size=bands.shape) * sigma[:, None], 0.0, 1.0)
        sim = _with_vi(pd.DataFrame(noisy, columns=FOUR))
        for target in TARGETS:
            forest = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                           random_state=42, n_jobs=workers)
            forest.fit(_matrix(sim, "bands_vi"), pairs[PARAM[target]].to_numpy(float))
            if sensor == "seq":
                hybrid[("seq", target)] = forest.predict(_matrix(versions["raw"], "bands_vi"))
            else:
                hybrid[("gao", target)] = forest.predict(_matrix(gao_raw, "bands_vi"))

    # Assemble predictions
    saved = []
    gao_mean = {t: float(features.loc[features["lai_ok"] if t == "lai" else features.index, t].mean()) for t in TARGETS}
    stage_mean = {
        t: {c: float(features.loc[(features["date_key"].isin(days)) & (features["lai_ok"] if t == "lai" else True), t].mean())
            for c, days in matched.items()}
        for t in TARGETS
    }

    def knezha_frame(values, rows, scenario, model, fs, target):
        part = knezha.iloc[rows]
        return pd.DataFrame({
            "direction": "gaocheng_to_knezha", "scenario": scenario, "model": model, "features": fs, "target": target,
            "unit_id": part["ESU_ID"].to_numpy(), "group": part["Campaign"].to_numpy(), "field": part["field_id"].to_numpy(),
            "year": part["year"].to_numpy(), "y_true": part[target].to_numpy(float), "y_pred": values,
        })

    for target in TARGETS:
        for fs in FEATURE_SETS:
            for model in MODELS:
                folds = [k for k in results if k[0] == "forward" and k[1] == "all" and k[2] == target and k[3] == fs and k[4] == model]
                if folds:
                    rows = meta[folds[0]]["rows"]
                    for label, scenario in (("kz_raw", "T1_raw"), ("kz_harm", "T2_harmonized"),
                                            ("kz_harm_m", "T2m_harmonized_fourband")):
                        values = np.mean([results[k][label] for k in folds], axis=0)
                        saved.append(knezha_frame(values, rows, scenario, model, fs, target))
                    oof = []
                    for k in folds:
                        ids = meta[k]["test_ids"]
                        oof.append(pd.DataFrame({"sample_id": ids, "outer_fold": k[5], "y_pred": results[k]["gao_test"]}))
                    oof = pd.concat(oof).merge(features[["sample_id", "date_key", "block_id", target]], on="sample_id")
                    saved.append(pd.DataFrame({
                        "direction": "gaocheng_out_of_fold", "scenario": "grouped_cv", "model": model, "features": fs,
                        "target": target, "unit_id": oof["sample_id"], "group": oof["date_key"],
                        "field": oof["block_id"].astype(str), "year": "2024", "y_true": oof[target], "y_pred": oof["y_pred"],
                    }))
                for campaign in matched:
                    folds = [k for k in results if k[0] == "forward" and k[1] == campaign and k[2] == target and k[3] == fs and k[4] == model]
                    if not folds:
                        continue
                    rows = meta[folds[0]]["rows"]
                    for label, scenario in (("kz_raw", "T3_matched_raw"), ("kz_harm", "T4_matched_harmonized"),
                                            ("kz_harm_m", "T4m_matched_harmonized_fourband")):
                        values = np.mean([results[k][label] for k in folds], axis=0)
                        saved.append(knezha_frame(values, rows, scenario, model, fs, target))
        rows = np.flatnonzero(knezha[target].notna().to_numpy())
        saved.append(knezha_frame(np.full(rows.size, gao_mean[target]), rows, "source_mean", "constant", "none", target))
        stage_rows = np.flatnonzero(knezha[target].notna().to_numpy() & knezha["Campaign"].isin(matched).to_numpy())
        saved.append(knezha_frame(knezha["Campaign"].iloc[stage_rows].map(stage_mean[target]).to_numpy(float),
                                  stage_rows, "stage_source_mean", "constant", "none", target))
        saved.append(knezha_frame(hybrid[("seq", target)][rows], rows, "prosail_hybrid", "rf", "bands_vi", target))
        # reverse
        for treatment in versions:
            for fs in FEATURE_SETS:
                for model in MODELS:
                    for set_name in ["all"] + list(reverse_matched):
                        folds = [k for k in results if k[0] == "reverse_" + treatment and k[1] == set_name and k[2] == target and k[3] == fs and k[4] == model]
                        if not folds:
                            continue
                        rows = meta[folds[0]]["rows"]
                        part = gao_raw.iloc[rows]
                        values = np.mean([results[k]["gao"] for k in folds], axis=0)
                        scenario = REVERSE_NAMES[(treatment, "all" if set_name == "all" else "matched")]
                        saved.append(pd.DataFrame({
                            "direction": "knezha_to_gaocheng", "scenario": scenario, "model": model, "features": fs,
                            "target": target, "unit_id": part["sample_id"].to_numpy(), "group": part["date_key"].to_numpy(),
                            "field": part["block_id"].astype(str).to_numpy(), "year": "2024",
                            "y_true": part[target].to_numpy(float), "y_pred": values,
                        }))
        kz_mean = float(knezha[target].mean())
        saved.append(pd.DataFrame({
            "direction": "knezha_to_gaocheng", "scenario": "source_mean", "model": "constant", "features": "none",
            "target": target, "unit_id": gao_raw["sample_id"], "group": gao_raw["date_key"],
            "field": gao_raw["block_id"].astype(str), "year": "2024", "y_true": gao_raw[target], "y_pred": kz_mean,
        }))
        saved.append(pd.DataFrame({
            "direction": "gaocheng_in_domain", "scenario": "prosail_hybrid", "model": "rf", "features": "bands_vi",
            "target": target, "unit_id": gao_raw["sample_id"], "group": gao_raw["date_key"],
            "field": gao_raw["block_id"].astype(str), "year": "2024", "y_true": gao_raw[target],
            "y_pred": hybrid[("gao", target)],
        }))
    predictions = pd.concat(saved, ignore_index=True)
    lai_ok = features.set_index("sample_id")["lai_ok"]
    on_gao = predictions["direction"].isin(["knezha_to_gaocheng", "gaocheng_in_domain", "gaocheng_out_of_fold"])
    drop = on_gao & (predictions["target"] == "lai") & ~predictions["unit_id"].map(lai_ok).eq(True)
    predictions = predictions[~drop].dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    predictions.to_csv(root / "outputs" / "predictions" / "cross_sensor_predictions.csv", index=False)

    # Score
    summary = {"forward": {}, "reverse": {}, "gaocheng_reference": {}}
    for target in TARGETS:
        fwd = predictions[(predictions["direction"] == "gaocheng_to_knezha") & (predictions["target"] == target)]
        matched_campaigns = sorted(fwd.loc[fwd["scenario"] == "T3_matched_raw", "group"].unique())
        summary["forward"][target] = {"matched_campaigns": matched_campaigns, "all": {}, "matched": {}}
        for (scenario, model, fs), part in fwd.groupby(["scenario", "model", "features"]):
            name = f"{scenario}|{model}|{fs}"
            if scenario in FORWARD_ALL:
                summary["forward"][target]["all"][name] = _score_knezha(part)
            sub = part[part["group"].isin(matched_campaigns)]
            if len(sub):
                summary["forward"][target]["matched"][name] = _score_knezha(sub)
        oracle = fwd[fwd["scenario"] == "source_mean"].copy()
        oracle["y_pred"] = oracle.groupby("group")["y_true"].transform("mean")
        summary["forward"][target]["all"]["campaign_mean_oracle|constant|none"] = _score_knezha(oracle)
        summary["forward"][target]["matched"]["campaign_mean_oracle|constant|none"] = _score_knezha(oracle[oracle["group"].isin(matched_campaigns)])

        rev = predictions[(predictions["direction"] == "knezha_to_gaocheng") & (predictions["target"] == target)]
        summary["reverse"][target] = {"all": {}, "matched": {}}
        matched_days = sorted(rev.loc[rev["scenario"] == "R3_matched_raw", "group"].unique())
        summary["reverse"][target]["matched_dates"] = matched_days
        for (scenario, model, fs), part in rev.groupby(["scenario", "model", "features"]):
            name = f"{scenario}|{model}|{fs}"
            if scenario in REVERSE_ALL:
                summary["reverse"][target]["all"][name] = _score_gaocheng(part)
            sub = part[part["group"].isin(matched_days)]
            if len(sub) and sub["group"].nunique() >= 1:
                summary["reverse"][target]["matched"][name] = _score_gaocheng(sub)
        ref = predictions[predictions["direction"].isin(["gaocheng_out_of_fold", "gaocheng_in_domain"]) & (predictions["target"] == target)]
        summary["gaocheng_reference"][target] = {
            f"{s}|{m}|{f}": _score_gaocheng(part) for (s, m, f), part in ref.groupby(["scenario", "model", "features"])
        }

    payload = {
        "note": (
            "Gaocheng-to-Knezha transfer with four common bands. T1 raw Sequoia bands; T2 Sequoia bands mapped to "
            "Gaocheng-equivalent bands with a four-band linear model fitted on PROSAIL spectra; T3 training restricted "
            "to Gaocheng dates whose Zadoks range overlaps the campaign; T4 = T3 with harmonized bands. Predictions "
            "average five grouped-fold models. Reverse transfer averages four field-grouped Knezha folds. Knezha CIs "
            "resample field units; Gaocheng CIs resample 40 m blocks within dates."
        ),
        "gaocheng_zadoks": GAO_ZADOKS,
        "knezha_zadoks": zadoks,
        "stage_matches": matched,
        "reverse_stage_matches": dict(reverse_matched),
        "source_means": {"gaocheng": gao_mean, "gaocheng_stage": stage_mean,
                         "knezha": {t: float(knezha[t].mean()) for t in TARGETS}},
        "harmonization": harmonization,
        "reverse_field_folds": field_fold,
        **summary,
    }
    dest = root / "outputs" / "metrics" / "cross_sensor_transfer.json"
    dest.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")

    for target in TARGETS:
        print(f"== {target} forward (all campaigns) ==", flush=True)
        for name, s in sorted(summary["forward"][target]["all"].items(), key=lambda kv: kv[1]["rmse"]):
            print(f"  {name:45s} rmse {s['rmse']:.3f} bias {s['bias']:+.3f} campaign {s['errors']['outer_rmse']:.3f} "
                  f"within {s['errors']['within_outer_rmse']:.3f} rho_wc {s['within_campaign'].get('spearman', np.nan):+.2f}", flush=True)
        print(f"== {target} forward (matched campaigns {summary['forward'][target]['matched_campaigns']}) ==", flush=True)
        for name, s in sorted(summary["forward"][target]["matched"].items(), key=lambda kv: kv[1]["rmse"]):
            print(f"  {name:45s} rmse {s['rmse']:.3f} bias {s['bias']:+.3f}", flush=True)
        print(f"== {target} reverse (all dates) ==", flush=True)
        for name, s in sorted(summary["reverse"][target]["all"].items(), key=lambda kv: kv[1]["rmse"]):
            print(f"  {name:45s} rmse {s['rmse']:.3f} bias {s['bias']:+.3f} within-date(green) {s['within_date_green'].get('within_rmse', np.nan):.3f}", flush=True)
    print(dest)


if __name__ == "__main__":
    main()
