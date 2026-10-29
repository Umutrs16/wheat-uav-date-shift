from __future__ import annotations

import argparse

import pandas as pd

from wheat_ppqmtl.io_utils import read_table, write_json, write_table
from wheat_ppqmtl.paths import configure_stdio, configs
from wheat_ppqmtl.splits import (
    assign_forward_folds,
    assign_grouped_folds,
    assign_lodo_folds,
    assign_spatial_block_folds,
    leakage_report,
)


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Freeze grouped-random, spatial-block, and LODO splits.").parse_args()
    cfg = configs()
    root = cfg["root"]
    frame = read_table(root / "data" / "interim" / "observations_plot.csv")
    model_cfg = cfg["model"]["train"]
    exp = cfg["experiment"]
    seed = int(exp["seed"])
    n_splits = int(exp["n_splits"])
    tune = float(model_cfg["inner_tune_fraction"])
    cal = float(model_cfg["inner_calibration_fraction"])
    spatial_cfg = cfg["data"]["spatial_cv"]
    buffer_m = float(spatial_cfg["buffer_m"])
    grouped = assign_grouped_folds(frame, "plot_id", "grouped_random", n_splits, seed, tune, cal)
    spatial = assign_spatial_block_folds(
        frame, float(spatial_cfg["block_m"]), buffer_m, "spatial_block", n_splits, seed + 17, tune, cal,
    )
    spatial_extra = []
    for offset, block_m in enumerate(spatial_cfg["sensitivity_m"], start=1):
        if float(block_m) == float(spatial_cfg["block_m"]):
            continue
        spatial_extra.append(assign_spatial_block_folds(
            frame, float(block_m), buffer_m, f"spatial_block_{int(block_m)}", n_splits, seed + 17 + offset, tune, cal,
        ))
    lodo = assign_lodo_folds(frame, seed, tune, cal)
    forward = assign_forward_folds(frame, seed, tune, cal)
    splits = pd.concat([grouped, spatial, *spatial_extra, lodo, forward], ignore_index=True)
    problems = leakage_report(splits)
    if problems:
        raise SystemExit("Split leakage:\n" + "\n".join(problems))
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for scheme, part in splits.groupby("scheme"):
        write_table(part, split_dir / f"{scheme}.csv")
        for fold, fold_df in part.groupby("outer_fold"):
            write_table(fold_df, split_dir / f"{scheme}_fold{int(fold)}.csv")
    balance = (
        splits.loc[splits["role"] == "test"]
        .groupby(["scheme", "outer_fold", "date_key"])
        .size()
        .reset_index(name="n_test")
    )
    write_table(balance, split_dir / "split_balance.csv")
    write_json(split_dir / "split_meta.json", {
        "seed": seed,
        "n_splits": n_splits,
        "tune_fraction": tune,
        "calibration_fraction": cal,
        "block_m": cfg["data"]["spatial_cv"]["block_m"],
        "buffer_m": buffer_m,
        "sensitivity_block_m": cfg["data"]["spatial_cv"]["sensitivity_m"],
        "group_random": "ID-grouped: seeded shuffle of the publisher identifier, then greedy assignment to the lightest fold. The identifier is not a stable physical plot.",
        "spatial": "cell of each observation coordinate, with points inside the buffer of a test point removed from train, tune, and calibration",
        "lodo": "each date is an outer test fold; remaining publisher identifiers are split into train, tune, and calibration",
        "forward": "test is the next acquisition date; only earlier dates are split into train, tune, and calibration",
        "leakage_problems": problems,
    })
    print(f"splits written, rows={len(splits)}, leakage_problems=0")


if __name__ == "__main__":
    main()
