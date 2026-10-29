from __future__ import annotations

import numpy as np
import pandas as pd


def _balanced_folds(group_ids: list, weights: dict, n_splits: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    order = list(group_ids)
    rng.shuffle(order)
    loads = [0] * n_splits
    assignment = {}
    for gid in order:
        fold = int(np.argmin(loads))
        assignment[gid] = fold
        loads[fold] += int(weights[gid])
    return assignment


def _inner_roles(group_ids: list, seed: int, tune_fraction: float, cal_fraction: float) -> dict:
    rng = np.random.default_rng(seed)
    ids = list(group_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n < 3:
        raise ValueError(f"Need at least 3 groups to split train/tune/calibration, found {n}")
    n_tune = max(1, int(round(n * tune_fraction)))
    n_cal = max(1, int(round(n * cal_fraction)))
    if n_tune + n_cal >= n:
        n_tune = 1
        n_cal = 1
    roles = {}
    for gid in ids[:n_tune]:
        roles[gid] = "tune"
    for gid in ids[n_tune:n_tune + n_cal]:
        roles[gid] = "calibration"
    for gid in ids[n_tune + n_cal:]:
        roles[gid] = "train"
    return roles


def assign_grouped_folds(
    frame: pd.DataFrame,
    group_col: str,
    scheme: str,
    n_splits: int,
    seed: int,
    tune_fraction: float = 0.15,
    cal_fraction: float = 0.15,
) -> pd.DataFrame:
    weights = frame.groupby(group_col).size().to_dict()
    groups = list(weights)
    if len(groups) < n_splits:
        raise ValueError(f"{scheme} has {len(groups)} groups, fewer than {n_splits} folds")
    outer = _balanced_folds(groups, weights, n_splits, seed)
    rows = []
    for fold in range(n_splits):
        test_groups = {gid for gid, assigned in outer.items() if assigned == fold}
        train_groups = [gid for gid in groups if gid not in test_groups]
        inner = _inner_roles(train_groups, seed + 1000 + fold, tune_fraction, cal_fraction)
        for _, rec in frame.iterrows():
            gid = rec[group_col]
            role = "test" if gid in test_groups else inner[gid]
            rows.append({
                "sample_id": rec["sample_id"],
                "plot_id": int(rec["plot_id"]),
                "date_key": rec["date_key"],
                "scheme": scheme,
                "outer_fold": fold,
                "role": role,
                "group_id": str(gid),
            })
    return pd.DataFrame(rows)


def assign_lodo_folds(
    frame: pd.DataFrame,
    seed: int,
    tune_fraction: float = 0.15,
    cal_fraction: float = 0.15,
) -> pd.DataFrame:
    dates = sorted(frame["date_key"].unique(), key=_date_sort)
    rows = []
    for fold, held_out in enumerate(dates):
        pool = frame.loc[frame["date_key"] != held_out, "plot_id"].astype(int).unique().tolist()
        inner = _inner_roles(pool, seed + 2000 + fold, tune_fraction, cal_fraction)
        for _, rec in frame.iterrows():
            if rec["date_key"] == held_out:
                role = "test"
            else:
                role = inner[int(rec["plot_id"])]
            rows.append({
                "sample_id": rec["sample_id"],
                "plot_id": int(rec["plot_id"]),
                "date_key": rec["date_key"],
                "scheme": "lodo",
                "outer_fold": fold,
                "role": role,
                "group_id": held_out if role == "test" else str(int(rec["plot_id"])),
                "held_out_date": held_out,
            })
    return pd.DataFrame(rows)


def _date_sort(key: str) -> tuple[int, int, int]:
    year, month, day = key.split(".")
    return int(year), int(month), int(day)


def _block_ids(frame: pd.DataFrame, block_m: float) -> pd.Series:
    bx = np.floor(frame["x"].to_numpy(float) / block_m).astype(int)
    by = np.floor(frame["y"].to_numpy(float) / block_m).astype(int)
    return bx.astype(str) + "_" + by.astype(str)


def _distances_to_test(pool_xy: np.ndarray, test_xy: np.ndarray) -> np.ndarray:
    if len(pool_xy) == 0 or len(test_xy) == 0:
        return np.array([])
    delta = pool_xy[:, None, :] - test_xy[None, :, :]
    return np.sqrt(np.sum(delta ** 2, axis=2)).min(axis=1)


def assign_spatial_block_folds(
    frame: pd.DataFrame,
    block_m: float,
    buffer_m: float,
    scheme: str,
    n_splits: int,
    seed: int,
    tune_fraction: float = 0.15,
    cal_fraction: float = 0.15,
) -> pd.DataFrame:
    work = frame.copy()
    work["block_id"] = _block_ids(work, block_m)
    weights = work.groupby("block_id").size().to_dict()
    groups = list(weights)
    if len(groups) < n_splits:
        raise ValueError(f"{scheme} has {len(groups)} blocks, fewer than {n_splits} folds")
    outer = _balanced_folds(groups, weights, n_splits, seed)
    rows = []
    for fold in range(n_splits):
        test_groups = {gid for gid, assigned in outer.items() if assigned == fold}
        test = work[work["block_id"].isin(test_groups)]
        pool = work[~work["block_id"].isin(test_groups)].copy()
        distances = _distances_to_test(pool[["x", "y"]].to_numpy(float), test[["x", "y"]].to_numpy(float))
        buffered = pool.iloc[np.flatnonzero(distances < buffer_m)] if len(distances) else pool.iloc[[]]
        kept = pool.iloc[np.flatnonzero(distances >= buffer_m)] if len(distances) else pool
        inner = _inner_roles(kept["block_id"].astype(str).unique().tolist(), seed + 3000 + fold, tune_fraction, cal_fraction)
        role_by_id = {rec["sample_id"]: "test" for _, rec in test.iterrows()}
        for _, rec in buffered.iterrows():
            role_by_id[rec["sample_id"]] = "buffer"
        for _, rec in kept.iterrows():
            role_by_id[rec["sample_id"]] = inner[str(rec["block_id"])]
        for _, rec in work.iterrows():
            rows.append({
                "sample_id": rec["sample_id"],
                "plot_id": int(rec["plot_id"]),
                "date_key": rec["date_key"],
                "scheme": scheme,
                "outer_fold": fold,
                "role": role_by_id[rec["sample_id"]],
                "group_id": str(rec["block_id"]),
            })
    return pd.DataFrame(rows)


def assign_forward_folds(
    frame: pd.DataFrame,
    seed: int,
    tune_fraction: float = 0.15,
    cal_fraction: float = 0.15,
) -> pd.DataFrame:
    dates = sorted(frame["date_key"].unique(), key=_date_sort)
    if len(dates) < 2:
        raise ValueError("Forward validation needs at least two acquisition dates")
    rows = []
    for fold, test_date in enumerate(dates[1:]):
        past = frame[frame["date_key"].map(lambda key: _date_sort(key) < _date_sort(test_date))]
        future = frame[frame["date_key"].map(lambda key: _date_sort(key) > _date_sort(test_date))]
        groups = past["plot_id"].astype(int).unique().tolist()
        inner = _inner_roles(groups, seed + 4000 + fold, tune_fraction, cal_fraction)
        past_ids = set(past["sample_id"])
        future_ids = set(future["sample_id"])
        for _, rec in frame.iterrows():
            if rec["date_key"] == test_date:
                role = "test"
            elif rec["sample_id"] in past_ids:
                role = inner[int(rec["plot_id"])]
            elif rec["sample_id"] in future_ids:
                role = "excluded"
            else:
                raise RuntimeError("A row was neither past, test, nor future")
            rows.append({
                "sample_id": rec["sample_id"],
                "plot_id": int(rec["plot_id"]),
                "date_key": rec["date_key"],
                "scheme": "forward",
                "outer_fold": fold,
                "role": role,
                "group_id": test_date if role == "test" else str(int(rec["plot_id"])),
                "held_out_date": test_date,
            })
    return pd.DataFrame(rows)


def leakage_report(splits: pd.DataFrame) -> list[str]:
    problems = []
    for scheme, scheme_df in splits.groupby("scheme"):
        for fold, fold_df in scheme_df.groupby("outer_fold"):
            roles = {role: set(part["sample_id"]) for role, part in fold_df.groupby("role")}
            names = list(roles)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    overlap = roles[left] & roles[right]
                    if overlap:
                        problems.append(f"{scheme} fold {fold}: {len(overlap)} sample_ids in both {left} and {right}")
            if scheme == "grouped_random":
                test_plots = set(fold_df.loc[fold_df["role"] == "test", "plot_id"])
                other_plots = set(fold_df.loc[fold_df["role"] != "test", "plot_id"])
                leaked = test_plots & other_plots
                if leaked:
                    problems.append(f"{scheme} fold {fold}: plot_id leakage {sorted(leaked)[:8]}")
            if str(scheme).startswith("spatial_block"):
                test_blocks = set(fold_df.loc[fold_df["role"] == "test", "group_id"])
                used = fold_df["role"].isin(["train", "tune", "calibration"])
                other_blocks = set(fold_df.loc[used, "group_id"])
                leaked_blocks = test_blocks & other_blocks
                if leaked_blocks:
                    problems.append(f"{scheme} fold {fold}: block leakage {sorted(leaked_blocks)[:8]}")
            if scheme == "forward":
                test_dates = set(fold_df.loc[fold_df["role"] == "test", "date_key"])
                if len(test_dates) != 1:
                    problems.append(f"forward fold {fold}: expected one test date, found {test_dates}")
                else:
                    test_key = _date_sort(next(iter(test_dates)))
                    used = fold_df["role"].isin(["train", "tune", "calibration"])
                    for day in set(fold_df.loc[used, "date_key"]):
                        if _date_sort(day) >= test_key:
                            problems.append(f"forward fold {fold}: {day} is not strictly before the test date")
            if scheme == "lodo":
                test_dates = set(fold_df.loc[fold_df["role"] == "test", "date_key"])
                other_dates = set(fold_df.loc[fold_df["role"] != "test", "date_key"])
                if test_dates & other_dates:
                    problems.append(f"lodo fold {fold}: date leakage {test_dates & other_dates}")
                if len(test_dates) != 1:
                    problems.append(f"lodo fold {fold}: expected one test date, found {test_dates}")
    return problems
