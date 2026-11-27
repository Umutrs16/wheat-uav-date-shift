"""One file for the intervals, paired differences and summary tables quoted in the manuscript.

Interval protocol (2000 percentile draws, seed 42). Intervals are conditional on the dates
and campaigns that were observed; they do not describe new dates, seasons or sites.
  Gaocheng  40 m blocks resampled with replacement inside each date;
  Knezha    the field units resampled with replacement, each with all its ESUs and campaigns.
Paired differences reuse the same draws for both methods.

Sections
  gaocheng_designs   RMSE and within-date agreement of every method and design (script 43), with R2 and bias
  gaocheng_paired    ensemble minus references (script 43); grouped-vs-LODO and forward-vs-LODO
                     absolute-error gaps recomputed with the same date-stratified blocks
  knezha_designs     zero-shot and within-Knezha designs with field-unit intervals
  transfer           every scenario, model and feature set in both directions, on all target
                     plots and on the stage-matched subset
  transfer_paired    paired RMSE differences between transfer scenarios and source means
  field_levels       within-campaign, between-field and within-field agreement at Knezha
  variance           sequential sums of squares and moment (ANOVA) estimates
  calibration        limited-label calibration summaries (script 44, per-group arrays dropped)
  mode_a             leave-one-group-out check of releasing the uncalibrated map
  lai_anchors        forward LAI standardisation anchors and their sensitivity
Run after 41-45.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.metrics import regression_metrics
from wheat_ppqmtl.multilevel import (
    between_inner_agreement,
    grouped_agreement,
    jsonable,
    percentile_interval,
    stratified_cluster_draws,
    within_inner_agreement,
)
from wheat_ppqmtl.paths import configure_stdio, configs

DRAWS = 2000
SEED = 42
SPRING = ("March_2017", "April_2017", "May_2017", "April_2018")
EXCLUDED = "2024.6.6"
TAU = {"lcc": [3.0, 4.0, 5.0, 6.0], "lai": [0.3, 0.5, 0.75, 1.0]}
LAI_ANCHORS = {"spectral": [1.0, 6.0], "anchor_0p5_8": [0.5, 8.0], "anchor_0p2_10p5": [0.2, 10.5]}
FORWARD_PAIRS = {
    "all": [("T1_raw|plsr|bands", "source_mean|constant|none"),
            ("T2_harmonized|plsr|bands", "source_mean|constant|none"),
            ("T2_harmonized|plsr|bands", "T1_raw|plsr|bands"),
            ("T2m_harmonized_fourband|plsr|bands", "T1_raw|plsr|bands"),
            ("T1_raw|gpr|bands", "source_mean|constant|none"),
            ("prosail_hybrid|rf|bands_vi", "source_mean|constant|none")],
    "matched": [("T3_matched_raw|plsr|bands", "T1_raw|plsr|bands"),
                ("T4_matched_harmonized|plsr|bands", "T2_harmonized|plsr|bands"),
                ("T3_matched_raw|plsr|bands", "stage_source_mean|constant|none"),
                ("T4_matched_harmonized|plsr|bands", "stage_source_mean|constant|none"),
                ("T1_raw|plsr|bands", "source_mean|constant|none")],
}
REVERSE_PAIRS = {
    "all": [("R1_raw|plsr|bands", "source_mean|constant|none"),
            ("R2_harmonized|plsr|bands", "source_mean|constant|none"),
            ("R2_harmonized|plsr|bands", "R1_raw|plsr|bands"),
            ("R2m_harmonized_fourband|plsr|bands", "R1_raw|plsr|bands")],
    "matched": [("R3_matched_raw|plsr|bands", "R1_raw|plsr|bands"),
                ("R4_matched_harmonized|plsr|bands", "R2_harmonized|plsr|bands"),
                ("R3_matched_raw|plsr|bands", "source_mean|constant|none"),
                ("R1_raw|plsr|bands", "source_mean|constant|none")],
}
LEVEL_SETS = {
    "T1_raw_plsr": ("transfer", "T1_raw|plsr|bands"),
    "T2_harmonized_plsr": ("transfer", "T2_harmonized|plsr|bands"),
    "T1_raw_gpr": ("transfer", "T1_raw|gpr|bands"),
    "leave_field_out": ("knezha", "leave_field_out"),
    "leave_campaign_out": ("knezha", "leave_campaign_out"),
}
CAL_KEYS = ("uncalibrated", "macro", "paired", "strategy_paired", "k_star", "mode_b_share", "rule_correct_share",
            "mode_b_share_by_group", "lambda_by_group", "lambda_asrc_mean", "clusters_sampled_median", "m_sensitivity",
            "route_m_median", "pool_size_median", "eval_size_median", "group_spread", "groups",
            "macro_spring", "macro_autumn", "season_spring", "season_autumn")


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _nanmean(values) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _field_draws(fields: np.ndarray, n_draws: int = DRAWS, seed: int = SEED):
    """Row indices and the draw position of each resampled field (duplicates stay distinct)."""
    keys = pd.unique(fields)
    members = {key: np.flatnonzero(fields == key) for key in keys}
    rng = np.random.default_rng(seed)
    for _ in range(n_draws):
        picks = rng.choice(keys, size=keys.size, replace=True)
        idx = np.concatenate([members[key] for key in picks])
        slot = np.concatenate([np.full(members[key].size, j) for j, key in enumerate(picks)])
        yield idx, slot


def _gaocheng_draws(day: np.ndarray, block: np.ndarray):
    return list(stratified_cluster_draws(day, np.char.add(np.char.add(day.astype(str), "|"), block.astype(str)), DRAWS, SEED))


def _levels(y, p, campaign, unit) -> dict:
    """Within-campaign agreement (mean over campaigns with >= 5 ESUs), between-field agreement of
    field means after removing campaign means (campaigns with >= 3 fields) and within-field
    agreement after removing field means (fields with >= 3 ESUs)."""
    r2w, rho = [], []
    for c in pd.unique(campaign):
        m = campaign == c
        if m.sum() < 5:
            continue
        yy, pp = y[m], p[m]
        yc = yy - yy.mean()
        ec = (pp - yy) - (pp - yy).mean()
        sst = float((yc ** 2).sum())
        r2w.append(1.0 - float((ec ** 2).sum()) / sst if sst > 0 else np.nan)
        rho.append(_rho(yy, pp))
    out = {"within_campaign_r2": _nanmean(r2w), "within_campaign_rho": _nanmean(rho)}
    _, first, inv = np.unique(unit, return_index=True, return_inverse=True)
    cnt = np.bincount(inv)
    my, mp = np.bincount(inv, y) / cnt, np.bincount(inv, p) / cnt
    unit_campaign = campaign[first]
    yb, pb = [], []
    for c in pd.unique(unit_campaign):
        sel = unit_campaign == c
        if sel.sum() < 3:
            continue
        yb.append(my[sel] - my[sel].mean())
        pb.append(mp[sel] - mp[sel].mean())
    if yb:
        yb, pb = np.concatenate(yb), np.concatenate(pb)
        sst = float((yb ** 2).sum())
        out.update({"between_field_r2": 1.0 - float(((pb - yb) ** 2).sum()) / sst if sst > 0 else np.nan,
                    "between_field_rho": _rho(yb, pb), "n_field_campaigns": int(yb.size)})
    keep = cnt[inv] >= 3
    if keep.any():
        yw, pw = y[keep] - my[inv][keep], p[keep] - mp[inv][keep]
        sst = float((yw ** 2).sum())
        out.update({"within_field_r2": 1.0 - float(((pw - yw) ** 2).sum()) / sst if sst > 0 else np.nan,
                    "within_field_rho": _rho(yw, pw),
                    "within_field_rmse": float(np.sqrt(np.mean((pw - yw) ** 2))),
                    "n_within": int(keep.sum()), "n_within_units": int(np.unique(inv[keep]).size)})
    return out


def _level_block(y, p, campaign, field) -> dict:
    unit = np.char.add(np.char.add(campaign.astype(str), "|"), field.astype(str))
    point = _levels(y, p, campaign, unit)
    check_b = between_inner_agreement(y, p, campaign, unit)
    check_w = within_inner_agreement(y, p, unit, min_n=3)
    assert abs(check_b["r2_within"] - point["between_field_r2"]) < 1e-9
    assert abs(check_w["r2_within"] - point["within_field_r2"]) < 1e-9
    boot = {key: [] for key in point if not key.startswith("n_")}
    for idx, slot in _field_draws(field):
        c = campaign[idx]
        u = np.char.add(np.char.add(c.astype(str), "|"), slot.astype(str))
        stats = _levels(y[idx], p[idx], c, u)
        for key in boot:
            boot[key].append(stats.get(key, np.nan))
    return {"n": int(y.size), "n_fields": int(pd.unique(field).size), "point": point,
            "ci95": {key: percentile_interval(values) for key, values in boot.items()}}


def _gaocheng_designs(root, within: dict, matrix: dict) -> dict:
    out = {}
    for target, schemes in within["targets"].items():
        for scheme, methods in schemes.items():
            for name, cell in methods.items():
                ref = matrix[target]["cells"][name][scheme]
                assert abs(ref["rmse"] - cell["point"]["rmse"]) < 1e-9 and ref["n"] == cell["n"]
                out.setdefault(target, {}).setdefault(scheme, {})[name] = {
                    "n": cell["n"], "rmse": cell["point"]["rmse"], "rmse_ci95": cell["block_ci95"]["rmse"],
                    "within_date_rmse": cell["point"]["within_date_rmse"],
                    "within_date_rmse_ci95": cell["block_ci95"]["within_date_rmse"],
                    "date_level_rmse": cell["point"]["date_level_rmse"],
                    "r2_within_green": cell["point"]["r2_within_green"],
                    "r2_within_green_ci95": cell["block_ci95"]["r2_within_green"],
                    "spearman_green": cell["point"]["spearman_green"],
                    "spearman_green_ci95": cell["block_ci95"]["spearman_green"],
                    "low_recall_green": cell["point"]["low_recall_green"],
                    "r2": ref.get("r2"), "bias": ref.get("bias"), "rrmse": ref.get("rrmse"),
                }
    return out


def _seed_mean(root, stem: str, scheme: str, target: str) -> pd.DataFrame:
    frame = read_table(root / "outputs" / "predictions" / f"{stem}__{scheme}.csv")
    frame = frame[(frame["target"] == target) & (frame["role"] == "test")]
    if target == "lai":
        frame = frame[np.isclose(pd.to_numeric(frame["lai_qc"], errors="coerce"), 1.0)]
    return (frame.groupby(["outer_fold", "sample_id", "date_key"], as_index=False)
            .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))


def _paired_gap(merged: pd.DataFrame, a: str, b: str) -> dict:
    """Mean absolute-error and RMSE differences (a minus b) with date-stratified block intervals."""
    y = merged["y_true"].to_numpy(float)
    ea = np.abs(merged[a].to_numpy(float) - y)
    eb = np.abs(merged[b].to_numpy(float) - y)
    day = merged["date_key"].to_numpy().astype(str)
    block = merged["block_id"].to_numpy().astype(str)
    draws = _gaocheng_draws(day, block)
    d_mae = [float(np.mean(ea[idx] - eb[idx])) for idx in draws]
    d_rmse = [float(np.sqrt(np.mean(ea[idx] ** 2)) - np.sqrt(np.mean(eb[idx] ** 2))) for idx in draws]
    return {"n": int(len(merged)), "n_blocks": int(pd.unique(np.char.add(np.char.add(day, "|"), block)).size),
            "delta_mae": float(np.mean(ea - eb)), "delta_mae_ci95": percentile_interval(d_mae),
            "rmse_a": float(np.sqrt(np.mean(ea ** 2))), "rmse_b": float(np.sqrt(np.mean(eb ** 2))),
            "delta_rmse": float(np.sqrt(np.mean(ea ** 2)) - np.sqrt(np.mean(eb ** 2))),
            "delta_rmse_ci95": percentile_interval(d_rmse)}


def _gaocheng_paired(root, within: dict) -> dict:
    blocks = read_table(root / "data" / "features" / "features.csv")[["sample_id", "block_id"]]
    out = {"ensemble_minus_reference": {}, "lodo_minus_grouped": {}, "forward_minus_lodo": {}}
    for target, schemes in within["targets"].items():
        for scheme, methods in schemes.items():
            paired = methods.get("Spectral ensemble", {}).get("paired", {})
            out["ensemble_minus_reference"].setdefault(target, {})[scheme] = paired
    later = {"2024.4.12", "2024.4.25", "2024.5.18", "2024.6.6"}
    for target in ("lcc", "lai"):
        lodo = _seed_mean(root, "spectral", "lodo", target)
        grouped = _seed_mean(root, "spectral", "grouped_random", target)
        forward = _seed_mean(root, "spectral", "forward", target)
        m1 = lodo.merge(grouped[["sample_id", "y_pred"]], on="sample_id", suffixes=("_lodo", "_grouped")).merge(blocks, on="sample_id")
        out["lodo_minus_grouped"][target] = _paired_gap(m1, "y_pred_lodo", "y_pred_grouped")
        m2 = lodo[lodo["date_key"].isin(later)].merge(forward[["sample_id", "y_pred"]], on="sample_id",
                                                       suffixes=("_lodo", "_forward")).merge(blocks, on="sample_id")
        out["forward_minus_lodo"][target] = _paired_gap(m2, "y_pred_forward", "y_pred_lodo")
    return out


def _knezha_point(y, p, campaign) -> dict:
    point = regression_metrics(y, p)
    wc = grouped_agreement(y, p, campaign, min_n=5)
    autumn = [c for c in pd.unique(campaign) if c not in SPRING]
    spring = [c for c in pd.unique(campaign) if c in SPRING]
    point["within_campaign"] = wc["macro"]
    point["within_campaign_spring"] = grouped_agreement(y, p, campaign, min_n=5, exclude=autumn)["macro"]
    point["within_campaign_autumn"] = grouped_agreement(y, p, campaign, min_n=5, exclude=spring)["macro"]
    return point


def _knezha_designs(root) -> tuple[dict, pd.DataFrame]:
    preds = pd.read_csv(root / "outputs" / "predictions" / "bulgaria_uav_predictions.csv")
    out = {}
    for (target, design), part in preds.groupby(["target", "design"]):
        part = part.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
        y, p = part["y_true"].to_numpy(float), part["y_pred"].to_numpy(float)
        field = part["field_id"].to_numpy().astype(str)
        rmse = [float(np.sqrt(np.mean((p[idx] - y[idx]) ** 2))) for idx, _ in _field_draws(field)]
        cell = _knezha_point(y, p, part["Campaign"].to_numpy().astype(str))
        cell["rmse_ci95"] = percentile_interval(rmse)
        cell["n_fields"] = int(pd.unique(field).size)
        out.setdefault(target, {})[design] = cell
    return out, preds


def _wide(cross: pd.DataFrame, direction: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = cross[(cross["direction"] == direction) & (cross["target"] == target)].copy()
    part["key"] = part["scenario"] + "|" + part["model"] + "|" + part["features"]
    wide = part.pivot_table(index="unit_id", columns="key", values="y_pred", aggfunc="first")
    meta = part.drop_duplicates("unit_id").set_index("unit_id").loc[wide.index, ["group", "field", "y_true"]]
    return wide, meta


def _transfer(root, cross: pd.DataFrame, xfer: dict) -> tuple[dict, dict]:
    out, paired = {}, {}
    for direction, label, pairs in (("gaocheng_to_knezha", "forward", FORWARD_PAIRS),
                                    ("knezha_to_gaocheng", "reverse", REVERSE_PAIRS)):
        for target in ("lcc", "lai"):
            wide, meta = _wide(cross, direction, target)
            if label == "forward":
                wide["campaign_mean_oracle|constant|none"] = meta.groupby("group")["y_true"].transform("mean")
            matched = xfer[label][target]["matched_campaigns" if label == "forward" else "matched_dates"]
            for block in ("all", "matched"):
                rows = np.ones(len(meta), bool) if block == "all" else meta["group"].isin(matched).to_numpy()
                sub, info = wide[rows], meta[rows]
                if label == "forward" and block == "matched":
                    sub = sub.copy()
                    sub["campaign_mean_oracle|constant|none"] = info.groupby("group")["y_true"].transform("mean")
                keys = [k for k in sub.columns if np.isfinite(sub[k].to_numpy(float)).all()]
                y = info["y_true"].to_numpy(float)
                pred = sub[keys].to_numpy(float)
                err = pred - y[:, None]
                groups = info["group"].to_numpy().astype(str)
                field = info["field"].to_numpy().astype(str)
                draws = ([idx for idx, _ in _field_draws(field)] if label == "forward"
                         else _gaocheng_draws(groups, field))
                boot = np.array([np.sqrt(np.mean(err[idx] ** 2, axis=0)) for idx in draws])
                cells = {}
                for j, key in enumerate(keys):
                    point = regression_metrics(y, pred[:, j])
                    excl = ([c for c in pd.unique(groups) if c not in SPRING] if label == "forward" else [EXCLUDED])
                    wc = grouped_agreement(y, pred[:, j], groups, min_n=5, exclude=excl)["macro"]
                    ref = xfer[label][target][block].get(key)
                    if ref is not None:
                        assert abs(ref["rmse"] - point["rmse"]) < 1e-6, (label, target, block, key, ref["rmse"], point["rmse"])
                    cells[key] = {"n": int(point["n"]), "rmse": float(point["rmse"]), "rmse_ci95": percentile_interval(boot[:, j]),
                                  "r2": float(point["r2"]), "bias": float(point["bias"]), "mae": float(point["mae"]),
                                  "within_group_r2": wc["r2_within"], "within_group_rho": wc["spearman"],
                                  "within_group_rmse": wc["within_rmse"]}
                learned = [k for k in keys if not k.split("|")[1] in ("constant",)]
                best = min(learned, key=lambda k: cells[k]["rmse"]) if learned else None
                out.setdefault(label, {}).setdefault(target, {})[block] = {
                    "groups": sorted(pd.unique(groups).tolist()), "n": int(len(y)),
                    "n_clusters": int(pd.unique(field).size) if label == "forward" else int(pd.unique(np.char.add(groups, field)).size),
                    "best_learned": best, "cells": cells}
                block_pairs = list(pairs[block])
                if best is not None:
                    block_pairs.append((best, "source_mean|constant|none"))
                for a, b in block_pairs:
                    if a not in keys or b not in keys:
                        continue
                    ia, ib = keys.index(a), keys.index(b)
                    paired.setdefault(label, {}).setdefault(target, {}).setdefault(block, {})[f"{a} minus {b}"] = {
                        "delta_rmse": cells[a]["rmse"] - cells[b]["rmse"],
                        "delta_rmse_ci95": percentile_interval(boot[:, ia] - boot[:, ib]),
                        "share_draws_negative": float(np.mean(boot[:, ia] - boot[:, ib] < 0))}
    return out, paired


def _field_levels(cross: pd.DataFrame, knezha_preds: pd.DataFrame) -> dict:
    out = {}
    for target in ("lcc", "lai"):
        wide, meta = _wide(cross, "gaocheng_to_knezha", target)
        kn = knezha_preds[knezha_preds["target"] == target]
        for name, (source, key) in LEVEL_SETS.items():
            if source == "transfer":
                frame = pd.DataFrame({"y": meta["y_true"].to_numpy(float), "p": wide[key].to_numpy(float),
                                      "campaign": meta["group"].to_numpy().astype(str), "field": meta["field"].to_numpy().astype(str)})
            else:
                part = kn[kn["design"] == key]
                frame = pd.DataFrame({"y": part["y_true"].to_numpy(float), "p": part["y_pred"].to_numpy(float),
                                      "campaign": part["Campaign"].to_numpy().astype(str), "field": part["field_id"].to_numpy().astype(str)})
            frame = frame.dropna()
            for season, keep in (("all", None), ("spring", True), ("autumn", False)):
                sub = frame if keep is None else frame[frame["campaign"].isin(SPRING) == keep]
                if sub["campaign"].nunique() == 0:
                    continue
                out.setdefault(target, {}).setdefault(name, {})[season] = _level_block(
                    sub["y"].to_numpy(float), sub["p"].to_numpy(float), sub["campaign"].to_numpy(), sub["field"].to_numpy())
                pt = out[target][name][season]["point"]
                print(f"levels {target} {name} {season}: within-campaign R2 {pt['within_campaign_r2']:.2f} rho {pt['within_campaign_rho']:.2f} | "
                      f"between-field R2 {pt.get('between_field_r2', np.nan):.2f} rho {pt.get('between_field_rho', np.nan):.2f} | "
                      f"within-field R2 {pt.get('within_field_r2', np.nan):.2f} rho {pt.get('within_field_rho', np.nan):.2f}", flush=True)
    return out


def _variance(hier: dict) -> dict:
    def ss(block):
        if not block:
            return None
        shares = [level["share"] for level in block["levels"]] + [block["residual"]["share"]]
        return {"n": block["n"], "shares": shares, "n_groups": [level["n_groups"] for level in block["levels"]]}

    out = {}
    for site, outer_key in (("knezha", "within_campaign_components"), ("gaocheng", "within_date_components")):
        for target in ("lcc", "lai"):
            b = hier[site][target]
            mom = b[outer_key]
            out.setdefault(site, {})[target] = {
                "sequential_ss": ss(b["nested"]),
                "sequential_ss_spring": ss(b.get("nested_spring")), "sequential_ss_autumn": ss(b.get("nested_autumn")),
                "sequential_ss_green_dates": ss(b.get("nested_green_dates")),
                "moment_within_outer": {"inner_share": mom["inner_share"], "var_inner": mom["var_inner"],
                                        "var_residual": mom["var_residual"], "n0": mom["n0"],
                                        "n_inner": mom["n_inner"], "n_outer": mom["n_outer"]},
            }
    return out


def _calibration(cal: dict) -> dict:
    out = {"settings": cal["settings"], "lambda0": cal["lambda0"], "slope_eval": cal["slope_eval"], "cases": {}}
    for case, targets in cal["cases"].items():
        for target, designs in targets.items():
            for design, block in designs.items():
                out["cases"].setdefault(case, {}).setdefault(target, {})[design] = {k: block[k] for k in CAL_KEYS if k in block}
    return out


def _mode_a(root) -> dict:
    gao = pd.read_csv(root / "outputs" / "predictions" / "gaocheng_lodo_plot_diagnostics.csv")
    kn = pd.read_csv(root / "outputs" / "predictions" / "bulgaria_plot_diagnostics.csv")
    pick = lambda frame, y, p, g: pd.DataFrame({"y": frame[y].to_numpy(float), "p": frame[p].to_numpy(float),  # noqa: E731
                                                "g": frame[g].astype(str).to_numpy()})
    frames = {
        ("gaocheng_lodo", "lcc"): pick(gao, "y_true_lcc", "y_pred_lcc", "date_key"),
        ("gaocheng_lodo", "lai"): pick(gao[gao["lai_observed"]], "y_true_lai", "y_pred_lai", "date_key"),
        ("knezha", "lcc"): pick(kn, "lcc", "y_pred_lcc", "Campaign"),
        ("knezha", "lai"): pick(kn, "lai", "y_pred_lai", "Campaign"),
    }
    out = {}
    for (case, target), frame in frames.items():
        frame = frame.dropna(subset=["y", "p"])
        per = {str(g): float(np.sqrt(np.mean((part["p"] - part["y"]) ** 2))) for g, part in frame.groupby("g")}
        rows = {}
        for tau in TAU[target]:
            checks = []
            for g, value in per.items():
                dev = float(np.mean([v for h, v in per.items() if h != g]))
                checks.append({"group": g, "development_rmse": dev, "admitted": dev <= tau, "rmse": value, "met": value <= tau})
            rows[f"{tau:g}"] = {"checks": checks,
                                "n_admitted": int(sum(c["admitted"] for c in checks)),
                                "n_admitted_met": int(sum(c["admitted"] and c["met"] for c in checks)),
                                "n_rejected_would_meet": int(sum((not c["admitted"]) and c["met"] for c in checks)),
                                "n_groups": len(checks)}
        out.setdefault(case, {})[target] = {"rmse_by_group": per, "tau": rows}
    return out


def _anchors(summary: dict) -> dict:
    out = {"rule": "Fixed anchors replace the LAI target scaler only when the training dates hold fewer than 8 "
                   "independent LAI labels (the first forward step, trained on 31 March); median and IQR are those of the anchor pair.",
           "variants": {}}
    for model, pair in LAI_ANCHORS.items():
        lo, hi = pair
        entry = {"anchors": pair, "median": (lo + hi) / 2.0, "iqr": (hi - lo) / 2.0, "primary": model == "spectral"}
        for target in ("lcc", "lai"):
            if model == "spectral":
                row = next(r for r in summary["forward_by_date"]
                           if r["model"] == "spectral" and r["target"] == target and r["date_key"] == "2024.4.12")
                entry[target] = {"rmse": row["rmse"], "bias": row["bias"], "n": row["n"]}
            else:
                row = summary["models"][model]["forward"][target]
                entry[target] = {"rmse": row["ensemble_rmse"], "bias": row["ensemble_bias"], "n": row["n"]}
        out["variants"][model] = entry
    return out


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    metrics = root / "outputs" / "metrics"
    load = lambda name: json.loads((metrics / name).read_text(encoding="utf-8"))  # noqa: E731
    within = load("within_date_metrics.json")
    matrix = load("generalization_matrix.json")["targets"]
    xfer = load("cross_sensor_transfer.json")
    cal = load("calibration_budget.json")
    hier = load("knezha_hierarchy.json")
    summary = load("summary.json")
    cross = pd.read_csv(root / "outputs" / "predictions" / "cross_sensor_predictions.csv")

    payload = {"note": " ".join(__doc__.split()), "draws": DRAWS, "seed": SEED,
               "protocol": {"gaocheng": "40 m blocks resampled with replacement within each date",
                            "knezha": "field units resampled with replacement with all their ESUs and campaigns",
                            "interpretation": "conditional on the observed dates and campaigns"}}
    payload["gaocheng_designs"] = _gaocheng_designs(root, within, matrix)
    payload["gaocheng_paired"] = _gaocheng_paired(root, within)
    gap = payload["gaocheng_paired"]["lodo_minus_grouped"]["lcc"]
    print(f"LODO minus grouped LCC: dMAE {gap['delta_mae']:.3f} {gap['delta_mae_ci95']} | "
          f"summary.json {summary['paired_optimism']['spectral']['lcc']['overall']['mean']:.3f}", flush=True)
    payload["knezha_designs"], knezha_preds = _knezha_designs(root)
    payload["transfer"], payload["transfer_paired"] = _transfer(root, cross, xfer)
    for target in ("lcc", "lai"):
        zs = payload["knezha_designs"][target]["zero_shot_from_gaocheng"]
        t1 = payload["transfer"]["forward"][target]["all"]["cells"]["T1_raw|plsr|bands"]
        assert abs(zs["rmse"] - t1["rmse"]) < 1e-6 and zs["n"] == t1["n"], (target, zs["rmse"], t1["rmse"])
        payload["knezha_designs"][target]["zero_shot_from_gaocheng"]["rmse_ci95"] = t1["rmse_ci95"]
    payload["field_levels"] = _field_levels(cross, knezha_preds)
    payload["variance"] = _variance(hier)
    payload["calibration"] = _calibration(cal)
    payload["mode_a"] = _mode_a(root)
    payload["lai_anchors"] = _anchors(summary)
    dest = metrics / "unified_results.json"
    dest.write_text(json.dumps(jsonable(payload), indent=1), encoding="utf-8")
    for label in ("forward", "reverse"):
        for target in ("lcc", "lai"):
            for block, pairs in payload["transfer_paired"][label][target].items():
                for name, d in pairs.items():
                    print(f"{label} {target} {block}: {name}: {d['delta_rmse']:+.3f} {np.round(d['delta_rmse_ci95'], 3).tolist()}")
    print(dest)


if __name__ == "__main__":
    main()
