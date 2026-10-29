from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wheat_ppqmtl.io_utils import read_table, write_json
from wheat_ppqmtl.metrics import cluster_bootstrap_ci, optimism_gap, paired_error_gap, paired_rmse_difference, regression_metrics
from wheat_ppqmtl.paths import configure_stdio, configs


def _load_predictions(root) -> pd.DataFrame:
    files = sorted((root / "outputs" / "predictions").glob("*.csv"))
    skip = {"with_intervals.csv", "intervals_ensemble.csv"}
    frames = [pd.read_csv(path) for path in files if path.name not in skip]
    if not frames:
        raise FileNotFoundError("No prediction files in outputs/predictions")
    return pd.concat(frames, ignore_index=True)


def _derived_ccc(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    keys = ["model", "scheme", "outer_fold", "seed", "role", "sample_id"]
    lcc = frame[frame["target"] == "lcc"][keys + ["plot_id", "date_key", "y_true", "y_pred"]]
    lai = frame[frame["target"] == "lai"][keys + ["lai_qc", "y_true", "y_pred"]]
    merged = lcc.merge(lai, on=keys, how="inner", suffixes=("_lcc", "_lai"))
    if merged.empty:
        return frame
    y_true = factor * merged["y_true_lcc"] * merged["y_true_lai"]
    y_true = y_true.where(np.isfinite(merged["y_true_lcc"]) & np.isfinite(merged["y_true_lai"]))
    derived = pd.DataFrame({
        "model": merged["model"],
        "scheme": merged["scheme"],
        "outer_fold": merged["outer_fold"],
        "seed": merged["seed"],
        "sample_id": merged["sample_id"],
        "plot_id": merged["plot_id"],
        "date_key": merged["date_key"],
        "lai_qc": merged["lai_qc"],
        "role": merged["role"],
        "target": "ccc_derived",
        "y_true": y_true,
        "y_pred": factor * merged["y_pred_lcc"] * merged["y_pred_lai"],
        "y_sigma": np.nan,
    })
    return pd.concat([frame, derived], ignore_index=True)


def _independent_labels(frame: pd.DataFrame) -> pd.DataFrame:
    qc = pd.to_numeric(frame["lai_qc"], errors="coerce")
    derived = frame["target"].isin(["lai", "ccc", "ccc_derived"])
    return frame.loc[~(derived & ~np.isclose(qc, 1.0))].copy()


def _attach_blocks(frame: pd.DataFrame, root) -> pd.DataFrame:
    path = root / "data" / "interim" / "observations_plot.csv"
    if not path.exists() or "sample_id" not in frame.columns:
        frame = frame.copy()
        frame["block_id"] = frame["plot_id"].astype(str)
        return frame
    blocks = read_table(path)[["sample_id", "block_id"]].drop_duplicates("sample_id")
    return frame.merge(blocks, on="sample_id", how="left")


def _seed_average(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "scheme", "target", "role", "sample_id"]
    return frame.groupby(keys, as_index=False).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        plot_id=("plot_id", "first"),
        date_key=("date_key", "first"),
        lai_qc=("lai_qc", "first"),
        block_id=("block_id", "first"),
        n_seeds=("seed", "nunique"),
    )


def _paired_model_gap(averaged: pd.DataFrame, reference: str, alternative: str, schemes: tuple[str, ...], n_boot: int, seed: int) -> dict:
    """RMSE(alternative) - RMSE(reference) on the same test rows."""
    out = {}
    for scheme in schemes:
        out[scheme] = {}
        for target in ("lcc", "lai"):
            left = averaged[
                (averaged["model"] == reference)
                & (averaged["scheme"] == scheme)
                & (averaged["target"] == target)
                & (averaged["role"] == "test")
            ]
            right = averaged[
                (averaged["model"] == alternative)
                & (averaged["scheme"] == scheme)
                & (averaged["target"] == target)
                & (averaged["role"] == "test")
            ]
            merged = left.merge(right, on=["sample_id", "target"], suffixes=("_ref", "_alt"))
            if merged.empty:
                continue
            groups = merged["block_id_ref"].fillna(merged["plot_id_ref"].astype(str))
            out[scheme][target] = paired_rmse_difference(
                merged["y_true_ref"], merged["y_pred_ref"], merged["y_pred_alt"], groups, n_boot=n_boot, seed=seed,
            )
    return out


def summarize() -> dict:
    cfg = configs()
    root = cfg["root"]
    raw = _attach_blocks(_derived_ccc(_load_predictions(root), float(cfg["data"]["ccc"]["factor"])), root)
    raw_test = raw[raw["role"] == "test"].copy()
    weak_rows = []
    qc = pd.to_numeric(raw_test["lai_qc"], errors="coerce")
    weak = raw_test[(raw_test["target"] == "lai") & np.isclose(qc, 2.0)]
    for keys, part in weak.groupby(["model", "scheme", "date_key", "seed"]):
        weak_rows.append({
            "model": keys[0], "scheme": keys[1], "date_key": keys[2], "seed": int(keys[3]),
            "n": int(len(part)),
            "predicted_mean": float(part["y_pred"].mean()),
            "recorded_date_mean": float(part["y_true"].mean()),
        })
    test = _independent_labels(raw_test)
    n_boot = int(cfg["experiment"]["bootstrap_replicates"])
    per_seed = []
    for keys, part in test.groupby(["model", "scheme", "target", "seed"]):
        stats = cluster_bootstrap_ci(
            part["y_true"], part["y_pred"], part["block_id"].fillna(part["plot_id"].astype(str)),
            n_boot=n_boot, seed=int(cfg["experiment"]["seed"]),
        )
        date_rmses = []
        for _, dated in part.groupby("date_key"):
            if len(dated) >= 8:
                date_rmses.append(regression_metrics(dated["y_true"], dated["y_pred"])["rmse"])
        stats["macro_rmse"] = float(np.mean(date_rmses)) if date_rmses else np.nan
        stats["n_dates"] = int(len(date_rmses))
        per_seed.append({
            "model": keys[0], "scheme": keys[1], "target": keys[2], "seed": int(keys[3]), **stats,
        })
    seed_df = pd.DataFrame(per_seed)
    summary = {"by_seed": seed_df.to_dict(orient="records"), "models": {}}
    for (model, scheme, target), part in seed_df.groupby(["model", "scheme", "target"]):
        entry = {
            "n_seeds": int(len(part)),
            "n": int(part["n"].mean()),
            "r2_mean": float(part["r2"].mean()),
            "r2_sd": float(part["r2"].std(ddof=0)) if len(part) > 1 else None,
            "rmse_mean": float(part["rmse"].mean()),
            "rmse_sd": float(part["rmse"].std(ddof=0)) if len(part) > 1 else None,
            "rrmse_mean": float(part["rrmse"].mean()),
            "mae_mean": float(part["mae"].mean()),
            "bias_mean": float(part["bias"].mean()),
            "macro_rmse_mean": float(part["macro_rmse"].mean()),
            "macro_rmse_sd": float(part["macro_rmse"].std(ddof=0)) if len(part) > 1 else None,
            "n_dates": int(part["n_dates"].mean()),
            "r2_ci95_seed0": part.sort_values("seed").iloc[0]["r2_ci95"],
            "rmse_ci95_seed0": part.sort_values("seed").iloc[0]["rmse_ci95"],
        }
        summary["models"].setdefault(model, {}).setdefault(scheme, {})[target] = entry

    summary["optimism_gap"] = {}

    qc_rows = []
    lodo = test[test["scheme"] == "lodo"]
    lai = lodo[lodo["target"] == "lai"]
    for keys, part in lai.groupby(["model", "lai_qc", "seed"]):
        stats = regression_metrics(part["y_true"], part["y_pred"])
        qc_rows.append({"model": keys[0], "lai_qc": keys[1], "seed": int(keys[2]), **stats})
    summary["date_level_lai"] = weak_rows
    averaged = _seed_average(test)
    for (model, scheme, target), part in averaged.groupby(["model", "scheme", "target"]):
        entry = summary["models"].get(model, {}).get(scheme, {}).get(target)
        if entry is None or str(target).endswith("derived"):
            continue
        stats = cluster_bootstrap_ci(
            part["y_true"], part["y_pred"],
            part["block_id"].fillna(part["plot_id"].astype(str)),
            n_boot=n_boot, seed=int(cfg["experiment"]["seed"]),
        )
        entry["ensemble_rmse"] = float(stats["rmse"])
        entry["ensemble_bias"] = float(stats["bias"])
        entry["ensemble_r2"] = float(stats["r2"])
        entry["rmse_ci95"] = stats["rmse_ci95"]
        entry["r2_ci95"] = stats["r2_ci95"]
    for model, schemes in summary["models"].items():
        if "lodo" in schemes and "grouped_random" in schemes:
            summary["optimism_gap"][model] = {}
            for target in schemes["lodo"]:
                if target in schemes["grouped_random"] and not str(target).endswith("derived"):
                    if "ensemble_rmse" not in schemes["lodo"][target] or "ensemble_rmse" not in schemes["grouped_random"][target]:
                        continue
                    summary["optimism_gap"][model][target] = optimism_gap(
                        {"rmse": schemes["lodo"][target]["ensemble_rmse"], "r2": schemes["lodo"][target]["ensemble_r2"]},
                        {"rmse": schemes["grouped_random"][target]["ensemble_rmse"], "r2": schemes["grouped_random"][target]["ensemble_r2"]},
                    )
    summary["per_date"] = []
    lodo_avg = averaged[averaged["scheme"] == "lodo"]
    for keys, part in lodo_avg.groupby(["model", "target", "date_key"]):
        stats = regression_metrics(part["y_true"], part["y_pred"])
        summary["per_date"].append({
            "model": keys[0], "target": keys[1], "date_key": keys[2],
            "rmse": float(stats["rmse"]), "bias": float(stats["bias"]),
            "r2": float(stats["r2"]), "n": int(stats["n"]),
        })
    for model, schemes in summary["models"].items():
        for target, entry in schemes.get("lodo", {}).items():
            usable = [
                row["rmse"] for row in summary["per_date"]
                if row["model"] == model and row["target"] == target and int(row["n"]) >= 8
            ]
            if usable:
                entry["ensemble_macro"] = float(np.mean(usable))
    summary["forward_by_date"] = []
    forward_avg = averaged[averaged["scheme"] == "forward"]
    for keys, part in forward_avg.groupby(["model", "target", "date_key"]):
        stats = regression_metrics(part["y_true"], part["y_pred"])
        summary["forward_by_date"].append({
            "model": keys[0], "target": keys[1], "date_key": keys[2],
            "rmse": float(stats["rmse"]), "bias": float(stats["bias"]),
            "r2": float(stats["r2"]), "n": int(stats["n"]),
        })
    summary["paired_optimism"] = {}
    for model in averaged["model"].unique():
        summary["paired_optimism"][model] = {}
        for target in ("lcc", "lai"):
            left = averaged[(averaged["model"] == model) & (averaged["scheme"] == "lodo") & (averaged["target"] == target)]
            right = averaged[(averaged["model"] == model) & (averaged["scheme"] == "grouped_random") & (averaged["target"] == target)]
            merged = left.merge(right, on=["sample_id", "target"], suffixes=("_lodo", "_grouped"))
            if merged.empty:
                continue
            delta = np.abs(merged["y_true_lodo"] - merged["y_pred_lodo"]) - np.abs(merged["y_true_grouped"] - merged["y_pred_grouped"])
            groups = merged["block_id_lodo"].fillna(merged["plot_id_lodo"].astype(str))
            gap = paired_error_gap(delta, groups, n_boot=n_boot, seed=int(cfg["experiment"]["seed"]))
            by_date = []
            for day, part in merged.groupby("date_key_lodo"):
                by_date.append({
                    "date_key": day,
                    "n": int(len(part)),
                    "delta_mae": float(np.mean(np.abs(part["y_true_lodo"] - part["y_pred_lodo"]) - np.abs(part["y_true_grouped"] - part["y_pred_grouped"]))),
                })
            summary["paired_optimism"][model][target] = {"overall": gap, "by_date": by_date}
    later_dates = {"2024.4.12", "2024.4.25", "2024.5.18", "2024.6.6"}
    summary["paired_forward"] = {}
    for model in averaged["model"].unique():
        summary["paired_forward"][model] = {}
        for target in ("lcc", "lai"):
            left = averaged[
                (averaged["model"] == model)
                & (averaged["scheme"] == "lodo")
                & (averaged["target"] == target)
                & (averaged["role"] == "test")
                & (averaged["date_key"].isin(later_dates))
            ]
            right = averaged[
                (averaged["model"] == model)
                & (averaged["scheme"] == "forward")
                & (averaged["target"] == target)
                & (averaged["role"] == "test")
            ]
            merged = left.merge(right, on=["sample_id", "target"], suffixes=("_lodo", "_forward"))
            if merged.empty:
                continue
            lodo_stats = regression_metrics(merged["y_true_lodo"], merged["y_pred_lodo"])
            forward_stats = regression_metrics(merged["y_true_forward"], merged["y_pred_forward"])
            delta = np.abs(merged["y_true_forward"] - merged["y_pred_forward"]) - np.abs(merged["y_true_lodo"] - merged["y_pred_lodo"])
            groups = merged["block_id_lodo"].fillna(merged["plot_id_lodo"].astype(str))
            gap = paired_error_gap(delta, groups, n_boot=n_boot, seed=int(cfg["experiment"]["seed"]))
            by_date = []
            for day, part in merged.groupby("date_key_lodo"):
                day_delta = np.abs(part["y_true_forward"] - part["y_pred_forward"]) - np.abs(part["y_true_lodo"] - part["y_pred_lodo"])
                by_date.append({
                    "date_key": day,
                    "n": int(len(part)),
                    "lodo_rmse": float(regression_metrics(part["y_true_lodo"], part["y_pred_lodo"])["rmse"]),
                    "forward_rmse": float(regression_metrics(part["y_true_forward"], part["y_pred_forward"])["rmse"]),
                    "delta_mae": float(np.mean(day_delta)),
                })
            summary["paired_forward"][model][target] = {
                "n": int(len(merged)),
                "lodo_rmse": float(lodo_stats["rmse"]),
                "forward_rmse": float(forward_stats["rmse"]),
                "overall": gap,
                "by_date": by_date,
            }
    summary["paired_initialisation"] = _paired_model_gap(
        averaged, "spectral", "no_prosail", ("grouped_random", "lodo"), n_boot, int(cfg["experiment"]["seed"]),
    )
    summary["lai_by_qc"] = []
    if qc_rows:
        qc_df = pd.DataFrame(qc_rows)
        for keys, part in qc_df.groupby(["model", "lai_qc"]):
            summary["lai_by_qc"].append({
                "model": keys[0], "lai_qc": None if pd.isna(keys[1]) else float(keys[1]),
                "rmse": float(part["rmse"].mean()), "bias": float(part["bias"].mean()), "n": int(part["n"].mean()),
            })
    seed_df.to_csv(root / "outputs" / "metrics" / "metrics_by_seed.csv", index=False)
    write_json(root / "outputs" / "metrics" / "summary.json", summary)
    print(f"models={list(summary['models'])}")
    return summary


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Aggregate frozen-split predictions. No test-set retuning.").parse_args()
    (configs()["root"] / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    summarize()


if __name__ == "__main__":
    main()
