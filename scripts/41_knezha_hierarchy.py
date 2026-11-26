"""Three-level variance structure and error decomposition for Knezha and Gaocheng.

Knezha: campaign, field within campaign, ESU within field.
Gaocheng: acquisition date, 40 m block within date, plot within block.

The saved Knezha predictions (zero-shot from Gaocheng and the within-Knezha
hold-out designs of script 30) are split into campaign-level, between-field and
within-field error, so that skill within a campaign can be attributed to
differences between fields or to differences between ESUs inside a field.
Run after 30.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _path  # noqa: F401
from wheat_ppqmtl.io_utils import read_table
from wheat_ppqmtl.multilevel import (
    between_inner_agreement,
    error_decomposition,
    grouped_agreement,
    jsonable,
    nested_shares,
    percentile_interval,
    within_inner_agreement,
    within_outer_components,
)
from wheat_ppqmtl.paths import configure_stdio, configs

DRAWS = 2000
SPRING = ["March_2017", "April_2017", "May_2017", "April_2018"]
AUTUMN = ["November_2016", "November_2017"]
DESIGNS = ["zero_shot_from_gaocheng", "leave_campaign_out", "leave_field_out", "leave_year_out"]


def _knezha_structure(uav: pd.DataFrame, bio: pd.DataFrame, target: str) -> dict:
    part = uav.dropna(subset=[target]).copy()
    part["cf"] = part["Campaign"] + "|" + part["field_id"]
    spread = "CC_std" if target == "lcc" else "LAI_std"
    part = part.merge(bio[["ESU_ID", spread]], on="ESU_ID", how="left")
    out = {
        "nested": nested_shares(part[target], [part["Campaign"], part["cf"]]),
        "within_campaign_components": within_outer_components(part, target, "Campaign", "cf"),
        "within_esu_sd_median": float(part[spread].median()),
        "by_campaign": [],
    }
    for campaign, group in part.groupby("Campaign"):
        fields = group.groupby("field_id")[target]
        within = group[target] - fields.transform("mean")
        multi = fields.transform("size") >= 2
        out["by_campaign"].append({
            "campaign": campaign,
            "n": int(len(group)),
            "n_fields": int(group["field_id"].nunique()),
            "mean": float(group[target].mean()),
            "sd": float(group[target].std(ddof=1)),
            "field_mean_sd": float(fields.mean().std(ddof=1)) if group["field_id"].nunique() > 1 else np.nan,
            "within_field_sd": float(np.sqrt(np.sum(within[multi] ** 2) / max(1, multi.sum() - group.loc[multi, "field_id"].nunique()))),
            "within_esu_sd_median": float(group[spread].median()),
            "nested": nested_shares(group[target], [group["field_id"]]),
        })
    for label, campaigns in (("spring", SPRING), ("autumn", AUTUMN)):
        sub = part[part["Campaign"].isin(campaigns)]
        out[f"nested_{label}"] = nested_shares(sub[target], [sub["Campaign"], sub["cf"]])
    return out


def _gaocheng_structure(features: pd.DataFrame, target: str) -> dict:
    part = features.dropna(subset=[target]).copy()
    if target == "lai":
        part = part[np.isclose(pd.to_numeric(part["lai_qc"], errors="coerce"), 1.0)]
    part["db"] = part["date_key"].astype(str) + "|" + part["block_id"].astype(str)
    green = part[part["date_key"] != "2024.6.6"]
    return {
        "nested": nested_shares(part[target], [part["date_key"], part["db"]]),
        "nested_green_dates": nested_shares(green[target], [green["date_key"], green["db"]]),
        "within_date_components": within_outer_components(part, target, "date_key", "db"),
        "blocks_per_date": part.groupby("date_key")["block_id"].nunique().astype(int).to_dict(),
    }


def _design_summary(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    cf = (frame["Campaign"] + "|" + frame["field_id"]).to_numpy()
    out = {
        "n": int(len(frame)),
        "errors": error_decomposition(y, p, frame["Campaign"].to_numpy(), cf),
        "within_campaign": grouped_agreement(y, p, frame["Campaign"].to_numpy(), min_n=5),
        "between_fields": between_inner_agreement(y, p, frame["Campaign"].to_numpy(), cf),
        "within_fields": within_inner_agreement(y, p, cf, min_n=3),
        "by_season": {},
        "by_year": {},
    }
    for label, campaigns in (("spring", SPRING), ("autumn", AUTUMN)):
        sub = frame[frame["Campaign"].isin(campaigns)]
        if len(sub):
            sub_cf = (sub["Campaign"] + "|" + sub["field_id"]).to_numpy()
            out["by_season"][label] = {
                "errors": error_decomposition(sub["y_true"], sub["y_pred"], sub["Campaign"].to_numpy(), sub_cf),
                "within_campaign": grouped_agreement(sub["y_true"], sub["y_pred"], sub["Campaign"].to_numpy(), min_n=5)["macro"],
                "between_fields": between_inner_agreement(sub["y_true"], sub["y_pred"], sub["Campaign"].to_numpy(), sub_cf),
                "within_fields": within_inner_agreement(sub["y_true"], sub["y_pred"], sub_cf, min_n=3),
            }
    for year, sub in frame.groupby("year"):
        sub_cf = (sub["Campaign"] + "|" + sub["field_id"]).to_numpy()
        out["by_year"][str(year)] = {
            "errors": error_decomposition(sub["y_true"], sub["y_pred"], sub["Campaign"].to_numpy(), sub_cf),
            "within_campaign": grouped_agreement(sub["y_true"], sub["y_pred"], sub["Campaign"].to_numpy(), min_n=5)["macro"],
        }
    # Field units are resampled with all their campaigns; each unit is one ground-sampling cluster.
    fields = frame["field_id"].to_numpy()
    keys = pd.unique(fields)
    members = {key: np.flatnonzero(fields == key) for key in keys}
    rng = np.random.default_rng(42)
    rmse, within, between = [], [], []
    for _ in range(DRAWS):
        idx = np.concatenate([members[key] for key in rng.choice(keys, size=keys.size, replace=True)])
        dec = error_decomposition(y[idx], p[idx], frame["Campaign"].to_numpy()[idx], cf[idx])
        rmse.append(dec["rmse"])
        within.append(dec["within_inner_rmse"])
        between.append(dec["between_inner_rmse"])
    out["field_bootstrap_ci95"] = {
        "rmse": percentile_interval(rmse),
        "between_field_rmse": percentile_interval(between),
        "within_field_rmse": percentile_interval(within),
        "n_field_units": int(keys.size),
        "draws": DRAWS,
    }
    return out


def main() -> None:
    configure_stdio()
    cfg = configs()
    root = cfg["root"]
    base = root / "data" / "external" / "bulgaria"
    uav = pd.read_csv(base / "uav_esu_reflectance.csv")
    bio = pd.read_csv(base / "ESU_Biophysical_and_biochemical_crop_data.txt", sep="\t")
    features = read_table(root / "data" / "features" / "features.csv")
    predictions = pd.read_csv(root / "outputs" / "predictions" / "bulgaria_uav_predictions.csv")

    payload = {
        "note": (
            "Sequential sums-of-squares shares for nested levels (outermost first) and ANOVA (moment) variance "
            "components after removing outer-level means. Knezha within-ESU SD is the standard deviation "
            "reported in the ESU table; the number of readings behind it is not given there. "
            "Prediction errors are split into campaign-level, between-field and within-field mean squares, "
            "which add up to the overall mean squared error. Confidence intervals resample field units."
        ),
        "knezha": {}, "gaocheng": {}, "predictions": {},
    }
    for target in ("lcc", "lai"):
        payload["knezha"][target] = _knezha_structure(uav, bio, target)
        payload["gaocheng"][target] = _gaocheng_structure(features, target)
        payload["predictions"][target] = {}
        for design in DESIGNS:
            part = predictions[(predictions["design"] == design) & (predictions["target"] == target)].dropna(subset=["y_true", "y_pred"])
            if len(part):
                payload["predictions"][target][design] = _design_summary(part.reset_index(drop=True))
        k = payload["knezha"][target]
        print(
            f"{target}: campaign share {k['nested']['levels'][0]['share']:.3f}, field|campaign "
            f"{k['nested']['levels'][1]['share']:.3f}, ESU|field {k['nested']['residual']['share']:.3f}; "
            f"field share of within-campaign variance {k['within_campaign_components']['inner_share']:.3f}",
            flush=True,
        )
        for design, summary in payload["predictions"][target].items():
            e = summary["errors"]
            print(
                f"  {design}: rmse {e['rmse']:.3f} campaign {e['outer_rmse']:.3f} between-field "
                f"{e['between_inner_rmse']:.3f} within-field {e['within_inner_rmse']:.3f}; "
                f"between-field rho {summary['between_fields'].get('spearman', np.nan):.2f}, "
                f"within-field rho {summary['within_fields'].get('spearman', np.nan):.2f}",
                flush=True,
            )
    dest = root / "outputs" / "metrics" / "knezha_hierarchy.json"
    dest.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
