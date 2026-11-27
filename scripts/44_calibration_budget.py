"""Limited ground calibration on a held-out date or campaign, with random and spatially separated splits.

For each group (a held-out Gaocheng date or a Knezha campaign) the plots are
split into a sampling pool and an evaluation set. Two splits are used:
  random_half   a random half of the plots forms the pool (as in script 36);
  spatial_half  a random half of the 40 m blocks (Gaocheng) or fields (Knezha)
                forms the pool; Gaocheng evaluation plots within 10 m of a pool
                plot are dropped.
k plots are chosen from the pool by one of six strategies and their labels
feed one of six estimators, all of the form  y_hat = ybar_S + lambda (f - fbar_S):
  label_mean      lambda = 0
  weighted_mean   lambda = 0, cluster means weighted by the share of plots in each cluster
  offset          lambda = 1
  asrc_prior      lambda = lambda0
  asrc            lambda = clip((S_fy + m v_f lambda0) / (S_ff + m v_f), 0, 1)
  ridge_affine    lambda = S_fy / (S_ff + m v_f)
S_fy and S_ff are the centred cross- and sum of squares of the k labels and
predictions, v_f is the prediction variance over all plots of the group (no
labels) and m = 5 is fixed in advance. With k = 1 both sums are zero, so asrc
returns lambda0. lambda0 is the within-date slope of the observed on the
predicted values in data the model did not train on and that are not from the
evaluated group: Gaocheng calibration rows of the training dates of the same
leave-one-date-out fold, and for Knezha the Gaocheng out-of-fold predictions
of the same four-band model.
Two release rules choose between the map (Mode B) and the label mean (Mode C)
from the k labels alone:
  mode_rule        k < 3: label mean; k >= 3: asrc if its leave-one-out squared
                   error over the k labels is below that of the label mean
  mode_rule_ridge  the same with ridge_affine in place of asrc
The held-out evaluation plots are only used to score the choice.
No estimator uses evaluation labels. Monte Carlo bands describe split and
selection randomness only; spread across groups is reported separately.
Run after 35 and 42.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.multilevel import jsonable
from wheat_ppqmtl.paths import configure_stdio, configs

BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20]
REPEATS = 200
STRATEGIES = ["random", "spatial_strata", "spectral_cluster", "spatial_spectral", "systematic", "high_risk"]
ESTIMATORS = ["label_mean", "weighted_mean", "offset", "asrc_prior", "asrc", "ridge_affine", "mode_rule", "mode_rule_ridge"]
RULES = ["mode_rule", "mode_rule_ridge"]
K_RULE_MIN = 3
DESIGNS = ["random_half", "spatial_half"]
M_PSEUDO = 5.0
M_SENSITIVITY = [1.0, 2.0, 5.0, 10.0, 20.0]
BUFFER_M = 10.0
TAU = {"lcc": [3.0, 4.0, 5.0, 6.0], "lai": [0.3, 0.5, 0.75, 1.0]}
SPRING = ("March_2017", "April_2017", "May_2017", "April_2018")
GAO_BANDS = ["B", "G", "R", "RE", "NIR"]
FOUR = ["G", "R", "RE", "NIR"]


def _within_slope(y, f, groups) -> float:
    frame = pd.DataFrame({"y": y, "f": f, "g": groups}).dropna()
    yc = frame["y"] - frame.groupby("g")["y"].transform("mean")
    fc = frame["f"] - frame.groupby("g")["f"].transform("mean")
    sff = float((fc ** 2).sum())
    return float(np.clip((fc * yc).sum() / sff, 0.0, 1.0)) if sff > 0 else 0.0


def _split(design: str, coords, clusters, rng) -> tuple[np.ndarray, np.ndarray]:
    n = len(coords)
    if design == "random_half":
        order = rng.permutation(n)
        return np.sort(order[: n // 2]), np.sort(order[n // 2:])
    keys = np.array(pd.unique(clusters), dtype=object)
    rng.shuffle(keys)
    n_pool = max(1, len(keys) // 2)
    in_pool = np.isin(clusters, keys[:n_pool])
    pool = np.flatnonzero(in_pool)
    evaluation = np.flatnonzero(~in_pool)
    if BUFFER_M > 0 and len(pool) and len(evaluation):
        near = cKDTree(coords[pool]).query(coords[evaluation], k=1)[0] < BUFFER_M
        evaluation = evaluation[~near]
    return pool, evaluation


def _nearest_unique(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest point to each center (in center order), without repeats."""
    distance = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    chosen = np.full(len(centers), -1)
    used = set()
    for c in np.argsort(distance.min(axis=0)):
        for i in np.argsort(distance[:, c]):
            if int(i) not in used:
                chosen[c] = int(i)
                used.add(int(i))
                break
    return chosen


def _serpentine(coords: np.ndarray, k: int) -> np.ndarray:
    strips = max(1, int(round(np.sqrt(k))))
    rank = np.argsort(np.argsort(coords[:, 1], kind="stable"), kind="stable")
    strip = np.minimum(strips - 1, (rank * strips) // len(coords))
    order = []
    for s in range(strips):
        members = np.flatnonzero(strip == s)
        members = members[np.argsort(coords[members, 0] * (1 if s % 2 == 0 else -1), kind="stable")]
        order.extend(members.tolist())
    return np.array(order)


def _select(strategy: str, k: int, pool: np.ndarray, frame_idx: np.ndarray, data: dict, rng, seed: int):
    """Return selected frame indices and the share of the group's plots each one represents."""
    coords, bands, risk = data["coords"], data["bands"], data["risk"]
    uniform = np.full(k, 1.0 / k)
    if strategy == "random":
        return rng.choice(pool, size=k, replace=False), uniform
    if strategy == "high_risk":
        order = pool[np.argsort(-risk[pool], kind="stable")]
        return order[:k], uniform
    if strategy == "systematic":
        order = pool[_serpentine(coords[pool], k)]
        step = len(order) / k
        start = rng.uniform(0, step)
        picks = order[np.minimum(len(order) - 1, np.floor(start + step * np.arange(k)).astype(int))]
        return picks, uniform
    if strategy == "spatial_strata":
        model = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(coords[pool])
        labels = model.labels_
        picks, used = [], set()
        for c in range(k):
            members = [int(i) for i in pool[labels == c] if int(i) not in used]
            if not members:
                members = [int(i) for i in pool if int(i) not in used]
            pick = int(rng.choice(members))
            picks.append(pick)
            used.add(pick)
        share = np.bincount(model.predict(coords[frame_idx]), minlength=k) / len(frame_idx)
        return np.array(picks), share / share.sum()
    scaler = StandardScaler().fit(bands[pool])
    zb = scaler.transform(bands)
    if strategy == "spectral_cluster":
        space = zb
    else:
        cs = StandardScaler().fit(coords[pool])
        space = np.hstack([zb / np.sqrt(zb.shape[1]), cs.transform(coords) / np.sqrt(2.0)])
    model = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(space[pool])
    picks = pool[_nearest_unique(space[pool], model.cluster_centers_)]
    share = np.bincount(model.predict(space[frame_idx]), minlength=k) / len(frame_idx)
    if share.sum() <= 0:
        return picks, uniform
    return picks, share / share.sum()


def _route(coords: np.ndarray, entry: np.ndarray) -> float:
    left = list(range(len(coords)))
    here = entry
    total = 0.0
    while left:
        d = np.sqrt(((coords[left] - here) ** 2).sum(axis=1))
        j = int(np.argmin(d))
        total += float(d[j])
        here = coords[left[j]]
        left.pop(j)
    return total


def _loo_prefers_map(y_s, f_s, lam0, v_f, m, bounded: bool) -> bool:
    """True if the slope estimator beats the label mean in leave-one-out error over the k labels."""
    k = len(y_s)
    if k < K_RULE_MIN or v_f <= 0:
        return False
    dy, df = y_s - y_s.mean(), f_s - f_s.mean()
    c = k / (k - 1.0)
    s_fy = float((df * dy).sum()) - c * df * dy
    s_ff = float((df ** 2).sum()) - c * df ** 2
    if bounded:
        lam = np.clip((s_fy + m * v_f * lam0) / (s_ff + m * v_f), 0.0, 1.0)
    else:
        lam = s_fy / (s_ff + m * v_f)
    return bool(np.sum((dy - lam * df) ** 2) < np.sum(dy ** 2))


def _estimate(y_s, f_s, w, f_eval, lam0, v_f, m=M_PSEUDO) -> dict:
    ybar, fbar = y_s.mean(), f_s.mean()
    s_fy = float(((f_s - fbar) * (y_s - ybar)).sum())
    s_ff = float(((f_s - fbar) ** 2).sum())
    lam_asrc = float(np.clip((s_fy + m * v_f * lam0) / (s_ff + m * v_f), 0.0, 1.0)) if v_f > 0 else lam0
    b_ridge = s_fy / (s_ff + m * v_f) if v_f > 0 else 0.0
    out = {
        "label_mean": np.full_like(f_eval, ybar),
        "weighted_mean": np.full_like(f_eval, float((w * y_s).sum())),
        "offset": f_eval + (ybar - fbar),
        "asrc_prior": ybar + lam0 * (f_eval - fbar),
        "asrc": ybar + lam_asrc * (f_eval - fbar),
        "ridge_affine": ybar + b_ridge * (f_eval - fbar),
        "_lam_asrc": lam_asrc,
        "_s": (ybar, fbar, s_fy, s_ff),
    }
    pick_asrc = _loo_prefers_map(y_s, f_s, lam0, v_f, m, bounded=True)
    pick_ridge = _loo_prefers_map(y_s, f_s, lam0, v_f, m, bounded=False)
    out["mode_rule"] = out["asrc"] if pick_asrc else out["label_mean"]
    out["mode_rule_ridge"] = out["ridge_affine"] if pick_ridge else out["label_mean"]
    out["_mode_b"] = (pick_asrc, pick_ridge)
    return out


def _run_group(payload: dict) -> dict:
    data = payload["data"]
    y = data["y"]
    sources = data["predictions"]
    n = len(y)
    frame_idx = np.arange(n)
    shape = (REPEATS, len(STRATEGIES), len(BUDGETS))
    out = {
        "key": payload["key"],
        "n": n,
        "rmse": {s: np.full(shape + (len(ESTIMATORS),), np.nan) for s in sources},
        "uncalibrated": {s: np.full(REPEATS, np.nan) for s in sources},
        "lam": {s: np.full(shape, np.nan) for s in sources},
        "mode_b": {s: np.full(shape + (len(RULES),), np.nan) for s in sources},
        "m_sensitivity": {s: np.full((REPEATS, 2, len(BUDGETS), len(M_SENSITIVITY)), np.nan) for s in sources},
        "route": np.full(shape, np.nan),
        "clusters_sampled": np.full(shape, np.nan),
        "pool_size": np.zeros(REPEATS, dtype=int),
        "eval_size": np.zeros(REPEATS, dtype=int),
    }
    entry = data["coords"].min(axis=0)
    base_seed = payload["seed"]
    for r in range(REPEATS):
        rng = np.random.default_rng([base_seed, r])
        pool, evaluation = _split(payload["design"], data["coords"], data["clusters"], rng)
        out["pool_size"][r] = len(pool)
        out["eval_size"][r] = len(evaluation)
        if len(evaluation) < 3 or len(pool) < 1:
            continue
        for s, f in sources.items():
            out["uncalibrated"][s][r] = float(np.sqrt(np.mean((f[evaluation] - y[evaluation]) ** 2)))
        for si, strategy in enumerate(STRATEGIES):
            for ki, k in enumerate(BUDGETS):
                if k > len(pool):
                    continue
                sel_rng = np.random.default_rng([base_seed, r, si, k])
                picks, share = _select(strategy, k, pool, frame_idx, data, sel_rng, int(base_seed % 100000 + r))
                if payload["route"]:
                    out["route"][r, si, ki] = _route(data["coords"][picks], entry)
                out["clusters_sampled"][r, si, ki] = len(set(data["clusters"][picks].tolist()))
                for s, f in sources.items():
                    est = _estimate(y[picks], f[picks], share, f[evaluation], data["lambda0"][s], data["v_f"][s])
                    for ei, name in enumerate(ESTIMATORS):
                        out["rmse"][s][r, si, ki, ei] = float(np.sqrt(np.mean((est[name] - y[evaluation]) ** 2)))
                    out["lam"][s][r, si, ki] = est["_lam_asrc"]
                    out["mode_b"][s][r, si, ki] = est["_mode_b"]
                    if strategy in ("random", "spectral_cluster"):
                        ybar, fbar, s_fy, s_ff = est["_s"]
                        v_f, lam0 = data["v_f"][s], data["lambda0"][s]
                        for mi, m in enumerate(M_SENSITIVITY):
                            lam = float(np.clip((s_fy + m * v_f * lam0) / (s_ff + m * v_f), 0.0, 1.0)) if v_f > 0 else lam0
                            pred = ybar + lam * (f[evaluation] - fbar)
                            out["m_sensitivity"][s][r, 0 if strategy == "random" else 1, ki, mi] = float(
                                np.sqrt(np.mean((pred - y[evaluation]) ** 2)))
    return out


def _gaocheng_groups(root) -> list[dict]:
    diag = pd.read_csv(root / "outputs" / "predictions" / "gaocheng_lodo_plot_diagnostics.csv")
    features = read_table(root / "data" / "features" / "features.csv")[["sample_id", "block_id"]]
    diag = diag.merge(features, on="sample_id", how="left")
    preds = pd.read_csv(root / "outputs" / "predictions" / "spectral__lodo.csv")
    cal = preds[preds["role"] == "calibration"]
    cal = cal.groupby(["outer_fold", "sample_id", "target"], as_index=False).agg(
        date_key=("date_key", "first"), lai_qc=("lai_qc", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    groups = []
    for target in ("lcc", "lai"):
        for day, part in diag.groupby("date_key"):
            if target == "lai":
                part = part[part["lai_observed"]]
            part = part.dropna(subset=[f"y_true_{target}", f"y_pred_{target}"]).reset_index(drop=True)
            if len(part) < 10:
                continue
            fold = int(part["outer_fold"].iloc[0])
            c = cal[(cal["outer_fold"] == fold) & (cal["target"] == target)]
            if target == "lai":
                c = c[np.isclose(pd.to_numeric(c["lai_qc"], errors="coerce"), 1.0)]
            lam0 = _within_slope(c["y_true"].to_numpy(float), c["y_pred"].to_numpy(float), c["date_key"].to_numpy())
            f = part[f"y_pred_{target}"].to_numpy(float)
            groups.append({
                "case": "gaocheng_lodo", "target": target, "group": str(day),
                "data": {
                    "y": part[f"y_true_{target}"].to_numpy(float),
                    "predictions": {"spectral": f},
                    "coords": part[["x", "y"]].to_numpy(float),
                    "bands": part[GAO_BANDS].to_numpy(float),
                    "risk": part[f"risk_rtm_{target}"].to_numpy(float),
                    "clusters": part["block_id"].astype(str).to_numpy(),
                    "lambda0": {"spectral": lam0},
                    "v_f": {"spectral": float(np.var(f))},
                    "slope_eval": {"spectral": _within_slope(part[f"y_true_{target}"].to_numpy(float), f, np.zeros(len(f)))},
                    "n_calibration_rows": int(len(c)),
                },
                "route": True,
            })
    return groups


def _knezha_groups(root) -> list[dict]:
    diag = pd.read_csv(root / "outputs" / "predictions" / "bulgaria_plot_diagnostics.csv")
    bio = pd.read_csv(root / "data" / "external" / "bulgaria" / "ESU_Biophysical_and_biochemical_crop_data.txt", sep="\t")
    diag = diag.merge(bio[["ESU_ID", "Point_X_UTM34", "Point_Y_UTM34"]], on="ESU_ID", how="left")
    cross = pd.read_csv(root / "outputs" / "predictions" / "cross_sensor_predictions.csv")
    plsr = cross[(cross["model"] == "plsr") & (cross["features"] == "bands")]
    groups = []
    for target in ("lcc", "lai"):
        harm = plsr[(plsr["scenario"] == "T2_harmonized") & (plsr["target"] == target)].set_index("unit_id")["y_pred"]
        oof = plsr[(plsr["direction"] == "gaocheng_out_of_fold") & (plsr["target"] == target)]
        lam0 = _within_slope(oof["y_true"].to_numpy(float), oof["y_pred"].to_numpy(float), oof["group"].to_numpy())
        for campaign, part in diag.groupby("Campaign"):
            part = part.dropna(subset=[target, f"y_pred_{target}"]).reset_index(drop=True)
            if len(part) < 10:
                continue
            raw = part[f"y_pred_{target}"].to_numpy(float)
            harmonized = part["ESU_ID"].map(harm).to_numpy(float)
            sources = {"zero_shot_raw": raw}
            if np.isfinite(harmonized).all():
                sources["zero_shot_harmonized"] = harmonized
            groups.append({
                "case": "knezha", "target": target, "group": str(campaign),
                "data": {
                    "y": part[target].to_numpy(float),
                    "predictions": sources,
                    "coords": part[["Point_X_UTM34", "Point_Y_UTM34"]].to_numpy(float),
                    "bands": part[FOUR].to_numpy(float),
                    "risk": part[f"risk_rtm_{target}"].to_numpy(float),
                    "clusters": part["field_id"].astype(str).to_numpy(),
                    "lambda0": {s: lam0 for s in sources},
                    "v_f": {s: float(np.var(v)) for s, v in sources.items()},
                    "slope_eval": {s: _within_slope(part[target].to_numpy(float), v, np.zeros(len(v))) for s, v in sources.items()},
                },
                "route": False,
            })
    return groups


PAIRS = {
    "asrc_minus_label_mean": ("asrc", "label_mean"), "asrc_minus_offset": ("asrc", "offset"),
    "offset_minus_label_mean": ("offset", "label_mean"), "asrc_minus_ridge_affine": ("asrc", "ridge_affine"),
    "asrc_minus_asrc_prior": ("asrc", "asrc_prior"), "weighted_minus_label_mean": ("weighted_mean", "label_mean"),
    "ridge_affine_minus_label_mean": ("ridge_affine", "label_mean"),
    "ridge_affine_minus_offset": ("ridge_affine", "offset"),
    "asrc_prior_minus_label_mean": ("asrc_prior", "label_mean"),
    "mode_rule_minus_label_mean": ("mode_rule", "label_mean"), "mode_rule_minus_asrc": ("mode_rule", "asrc"),
    "mode_rule_ridge_minus_label_mean": ("mode_rule_ridge", "label_mean"),
    "mode_rule_ridge_minus_ridge_affine": ("mode_rule_ridge", "ridge_affine"),
}


def _quiet(func, *args, **kwargs):
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _nanmean(values, axis=0):
    return _quiet(np.nanmean, values, axis=axis)


def _nanquantile(values, q, axis=0):
    return _quiet(np.nanquantile, values, q, axis=axis)


def _summ(values: np.ndarray, axis=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _nanmean(values, axis=axis), _nanquantile(values, 0.025, axis=axis), _nanquantile(values, 0.975, axis=axis)


def _aggregate(results: list[dict], groups: list[dict], source: str, label_groups: list[str]) -> dict:
    """Macro averages over groups per repeat, Monte Carlo bands, paired differences and k*.

    A macro value needs every group in that repeat; budgets for which fewer than half
    of the repeats have all groups are not reported, so the group mix never changes along a curve.
    """
    stack = np.stack([res["rmse"][source] for res in results])          # G x R x S x K x E
    unc = np.stack([res["uncalibrated"][source] for res in results])     # G x R
    lam = np.stack([res["lam"][source] for res in results])               # G x R x S x K
    macro = stack.mean(axis=0)                                            # R x S x K x E, NaN if a group is missing
    valid = np.isfinite(macro).mean(axis=0) >= 0.5                       # S x K x E
    macro = np.where(valid[None], macro, np.nan)
    n_groups = np.full(valid.shape, len(results))
    mean, lo, hi = _summ(macro, axis=0)
    out = {"groups": label_groups, "budgets": BUDGETS, "strategies": STRATEGIES, "estimators": ESTIMATORS,
           "uncalibrated": {"macro_mean": float(np.nanmean(np.nanmean(unc, axis=0))),
                            "by_group": {g: float(np.nanmean(u)) for g, u in zip(label_groups, unc)}},
           "macro": {}, "paired": {}, "by_group": {}, "lambda_asrc_mean": {}, "k_star": {}, "group_spread": {}}
    for si, strategy in enumerate(STRATEGIES):
        out["macro"][strategy] = {}
        out["lambda_asrc_mean"][strategy] = _nanmean(_nanmean(lam[:, :, si, :], axis=1), axis=0).tolist()
        for ei, est in enumerate(ESTIMATORS):
            out["macro"][strategy][est] = {"mean": mean[si, :, ei].tolist(), "lo95": lo[si, :, ei].tolist(),
                                           "hi95": hi[si, :, ei].tolist(), "n_groups": n_groups[si, :, ei].tolist(),
                                           "share_repeats_complete": np.isfinite(macro[:, si, :, ei]).mean(axis=0).tolist()}
            group_means = _nanmean(stack[:, :, si, :, ei], axis=1)       # G x K
            out["group_spread"].setdefault(strategy, {})[est] = {
                "sd_of_group_means": _quiet(np.nanstd, group_means, axis=0, ddof=1).tolist() if len(results) > 1 else None,
                "mean_monte_carlo_sd": _nanmean(_quiet(np.nanstd, stack[:, :, si, :, ei], axis=1, ddof=1), axis=0).tolist(),
            }
        out["paired"][strategy] = {}
        for name, (a, b) in PAIRS.items():
            diff = macro[:, si, :, ESTIMATORS.index(a)] - macro[:, si, :, ESTIMATORS.index(b)]
            d_mean, d_lo, d_hi = _summ(diff, axis=0)
            ga = _nanmean(stack[:, :, si, :, ESTIMATORS.index(a)], axis=1)
            gb = _nanmean(stack[:, :, si, :, ESTIMATORS.index(b)], axis=1)
            share = _nanmean(np.where(np.isfinite(diff), diff < 0, np.nan), axis=0)
            out["paired"][strategy][name] = {"mean": d_mean.tolist(), "lo95": d_lo.tolist(), "hi95": d_hi.tolist(),
                                             "share_negative": share.tolist(),
                                             "groups_better": (ga < gb).sum(axis=0).tolist(),
                                             "groups_compared": (np.isfinite(ga) & np.isfinite(gb)).sum(axis=0).tolist()}
    base = STRATEGIES.index("random")
    out["strategy_paired"] = {}
    for si, strategy in enumerate(STRATEGIES):
        if si == base:
            continue
        out["strategy_paired"][strategy] = {}
        for est in ("label_mean", "weighted_mean", "asrc", "ridge_affine", "mode_rule"):
            ei = ESTIMATORS.index(est)
            diff = macro[:, si, :, ei] - macro[:, base, :, ei]
            d_mean, d_lo, d_hi = _summ(diff, axis=0)
            share = _nanmean(np.where(np.isfinite(diff), diff < 0, np.nan), axis=0)
            out["strategy_paired"][strategy][est] = {"mean": d_mean.tolist(), "lo95": d_lo.tolist(),
                                                     "hi95": d_hi.tolist(), "share_negative": share.tolist()}
    mode_b = np.stack([res["mode_b"][source] for res in results])          # G x R x S x K x rules
    i_label = ESTIMATORS.index("label_mean")
    out["mode_b_share"], out["rule_correct_share"], out["mode_b_share_by_group"] = {}, {}, {}
    for si, strategy in enumerate(STRATEGIES):
        out["mode_b_share"][strategy], out["rule_correct_share"][strategy] = {}, {}
        for ri, rule in enumerate(RULES):
            chosen = mode_b[:, :, si, :, ri]
            alt = stack[:, :, si, :, ESTIMATORS.index("asrc" if rule == "mode_rule" else "ridge_affine")]
            lab = stack[:, :, si, :, i_label]
            correct = np.where(chosen == 1, alt <= lab, lab <= alt).astype(float)
            correct[~np.isfinite(chosen) | ~np.isfinite(alt)] = np.nan
            out["mode_b_share"][strategy][rule] = _nanmean(chosen.reshape(-1, len(BUDGETS)), axis=0).tolist()
            out["rule_correct_share"][strategy][rule] = _nanmean(correct.reshape(-1, len(BUDGETS)), axis=0).tolist()
    for gi, g in enumerate(label_groups):
        out["mode_b_share_by_group"][g] = {strategy: _nanmean(mode_b[gi, :, si, :, 0], axis=0).tolist()
                                           for si, strategy in enumerate(STRATEGIES)}
    out["lambda_by_group"] = {}
    for gi, g in enumerate(label_groups):
        out["lambda_by_group"][g] = {}
        for strategy in ("random", "spectral_cluster"):
            vals = lam[gi, :, STRATEGIES.index(strategy), :]
            out["lambda_by_group"][g][strategy] = {
                "median": _nanquantile(vals, 0.5, axis=0).tolist(), "q25": _nanquantile(vals, 0.25, axis=0).tolist(),
                "q75": _nanquantile(vals, 0.75, axis=0).tolist(), "share_at_zero": _nanmean(np.where(np.isfinite(vals), vals <= 1e-9, np.nan), axis=0).tolist(),
                "share_at_one": _nanmean(np.where(np.isfinite(vals), vals >= 1 - 1e-9, np.nan), axis=0).tolist()}
    clusters = np.stack([res["clusters_sampled"] for res in results])     # G x R x S x K
    out["clusters_sampled_median"] = {strategy: _nanquantile(clusters[:, :, si, :].reshape(-1, len(BUDGETS)), 0.5, axis=0).tolist()
                                      for si, strategy in enumerate(STRATEGIES)}
    for gi, g in enumerate(label_groups):
        out["by_group"][g] = {}
        for si, strategy in enumerate(STRATEGIES):
            m, l, h = _summ(stack[gi, :, si, :, :], axis=0)
            out["by_group"][g][strategy] = {est: {"mean": m[:, ei].tolist(), "lo95": l[:, ei].tolist(), "hi95": h[:, ei].tolist()}
                                            for ei, est in enumerate(ESTIMATORS)}
    target = groups[0]["target"]
    for tau in TAU[target]:
        key = f"{tau:g}"
        out["k_star"][key] = {}
        for si, strategy in enumerate(STRATEGIES):
            out["k_star"][key][strategy] = {}
            for ei, est in enumerate(ESTIMATORS):
                curve = mean[si, :, ei]
                ok = np.flatnonzero(np.isfinite(curve) & (curve <= tau))
                group_curves = _nanmean(stack[:, :, si, :, ei], axis=1)
                per_group = []
                for gc in group_curves:
                    hit = np.flatnonzero(np.isfinite(gc) & (gc <= tau))
                    per_group.append(BUDGETS[int(hit[0])] if hit.size else None)
                out["k_star"][key][strategy][est] = {"macro": BUDGETS[int(ok[0])] if ok.size else None, "by_group": per_group}
        out["k_star"][key]["direct_map_meets_tau_by_group"] = [bool(np.nanmean(u) <= tau) for u in unc]
    return out


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    groups = _gaocheng_groups(root) + _knezha_groups(root)
    tasks = []
    for gi, group in enumerate(groups):
        for di, design in enumerate(DESIGNS):
            tasks.append({"key": (group["case"], group["target"], group["group"], design), "design": design,
                          "data": group["data"], "route": group["route"], "seed": 1000 * gi + 17 * di + 42})
    workers = max(2, min(20, (os.cpu_count() or 4) - 4))
    print(f"{len(tasks)} group-design runs on {workers} workers", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_run_group, tasks, chunksize=1):
            results[res["key"]] = res
            print("  done", res["key"], flush=True)

    payload = {
        "note": " ".join(__doc__.split()),
        "settings": {"budgets": BUDGETS, "repeats": REPEATS, "strategies": STRATEGIES, "estimators": ESTIMATORS,
                     "rules": RULES, "k_rule_min": K_RULE_MIN,
                     "m_pseudo": M_PSEUDO, "m_sensitivity": M_SENSITIVITY, "buffer_m": BUFFER_M, "tau": TAU},
        "lambda0": {}, "slope_eval": {}, "cases": {},
    }
    rows = []
    for case in ("gaocheng_lodo", "knezha"):
        for target in ("lcc", "lai"):
            members = [g for g in groups if g["case"] == case and g["target"] == target]
            if not members:
                continue
            sources = list(members[0]["data"]["predictions"])
            for source in sources:
                payload["lambda0"].setdefault(case, {}).setdefault(target, {})[source] = {
                    g["group"]: g["data"]["lambda0"][source] for g in members if source in g["data"]["lambda0"]}
                payload["slope_eval"].setdefault(case, {}).setdefault(target, {})[source] = {
                    g["group"]: g["data"]["slope_eval"][source] for g in members if source in g["data"]["slope_eval"]}
                for design in DESIGNS:
                    use = [g for g in members if source in g["data"]["predictions"]]
                    res = [results[(case, target, g["group"], design)] for g in use]
                    labels = [g["group"] for g in use]
                    block = _aggregate(res, use, source, labels)
                    block["pool_size_median"] = {g: float(np.median(r["pool_size"])) for g, r in zip(labels, res)}
                    block["eval_size_median"] = {g: float(np.median(r["eval_size"])) for g, r in zip(labels, res)}
                    msens = np.stack([r["m_sensitivity"][source] for r in res])
                    macro_m = msens.mean(axis=0)
                    complete = np.isfinite(macro_m).mean(axis=0) >= 0.5
                    curve_m = np.where(complete, _nanmean(macro_m, axis=0), np.nan)
                    block["m_sensitivity"] = {
                        strategy: {f"{m:g}": curve_m[j, :, mi].tolist() for mi, m in enumerate(M_SENSITIVITY)}
                        for j, strategy in enumerate(("random", "spectral_cluster"))}
                    if case == "gaocheng_lodo":
                        route = np.stack([r["route"] for r in res])
                        block["route_m_median"] = {s: _nanquantile(route[:, :, si, :].reshape(-1, len(BUDGETS)), 0.5, axis=0).tolist()
                                                   for si, s in enumerate(STRATEGIES)}
                    if case == "knezha":
                        for season, keep in (("spring", lambda g: g in SPRING), ("autumn", lambda g: g not in SPRING)):
                            idx = [i for i, g in enumerate(labels) if keep(g)]
                            if idx:
                                part = _aggregate([res[i] for i in idx], [use[i] for i in idx], source, [labels[i] for i in idx])
                                block[f"macro_{season}"] = part["macro"]
                                block[f"season_{season}"] = {key: part[key] for key in ("paired", "mode_b_share", "rule_correct_share")}
                    payload["cases"].setdefault(f"{case}|{source}", {}).setdefault(target, {})[design] = block
                    for g, gb in block["by_group"].items():
                        for strategy, ests in gb.items():
                            for est, vals in ests.items():
                                for ki, k in enumerate(BUDGETS):
                                    rows.append({"case": case, "source": source, "target": target, "design": design,
                                                 "group": g, "strategy": strategy, "k": k, "estimator": est,
                                                 "rmse_mean": vals["mean"][ki], "rmse_lo95": vals["lo95"][ki],
                                                 "rmse_hi95": vals["hi95"][ki]})
                    m = block["macro"]
                    print(f"{case}|{source} {target} {design}: uncal {block['uncalibrated']['macro_mean']:.3f} | k=5 random "
                          f"mean {m['random']['label_mean']['mean'][3]:.3f} offset {m['random']['offset']['mean'][3]:.3f} "
                          f"asrc {m['random']['asrc']['mean'][3]:.3f} ridge {m['random']['ridge_affine']['mean'][3]:.3f} | "
                          f"k=5 spectral mean {m['spectral_cluster']['label_mean']['mean'][3]:.3f} "
                          f"weighted {m['spectral_cluster']['weighted_mean']['mean'][3]:.3f} asrc {m['spectral_cluster']['asrc']['mean'][3]:.3f} | "
                          f"k=5 rule {m['random']['mode_rule']['mean'][3]:.3f} (map chosen {block['mode_b_share']['random']['mode_rule'][3]:.2f}, "
                          f"right {block['rule_correct_share']['random']['mode_rule'][3]:.2f})",
                          flush=True)
    pd.DataFrame(rows).to_csv(root / "outputs" / "metrics" / "calibration_budget_groups.csv", index=False)
    dest = root / "outputs" / "metrics" / "calibration_budget.json"
    dest.write_text(json.dumps(jsonable(payload), indent=1), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
