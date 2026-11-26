"""Nested variance shares, within-group agreement and stratified cluster resampling.

Groups are nested from the outside in (acquisition date or campaign, then field,
then plot). Inner labels must already be unique across outer groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr


def _arrays(*values):
    return [np.asarray(v) for v in values]


def nested_shares(values, levels: list) -> dict:
    """Sequential sums of squares for nested levels; the residual is the innermost unit."""
    y = np.asarray(values, float)
    grand = float(y.mean())
    sst = float(((y - grand) ** 2).sum())
    previous = np.full_like(y, grand)
    out = {"n": int(y.size), "mean": grand, "sd": float(y.std(ddof=1)) if y.size > 1 else np.nan,
           "ss_total": sst, "levels": []}
    for labels in levels:
        means = pd.Series(y).groupby(np.asarray(labels)).transform("mean").to_numpy()
        ss = float(((means - previous) ** 2).sum())
        out["levels"].append({"ss": ss, "share": ss / sst if sst > 0 else np.nan,
                              "n_groups": int(pd.unique(np.asarray(labels)).size)})
        previous = means
    ss_res = float(((y - previous) ** 2).sum())
    out["residual"] = {"ss": ss_res, "share": ss_res / sst if sst > 0 else np.nan}
    return out


def within_outer_components(frame: pd.DataFrame, value: str, outer: str, inner: str) -> dict:
    """ANOVA (moment) split of the variance left after removing outer-group means.

    Outer groups are fixed; inner groups nested in them are random. For an
    unbalanced design the inner variance is (MS_inner - MS_residual) / n0,
    truncated at zero.
    """
    data = frame[[value, outer, inner]].dropna()
    y = data[value].to_numpy(float)
    o = data[outer].to_numpy()
    i = data[inner].to_numpy()
    inner_means = pd.Series(y).groupby(i).transform("mean").to_numpy()
    outer_means = pd.Series(y).groupby(o).transform("mean").to_numpy()
    n_total = y.size
    n_inner = int(pd.unique(i).size)
    n_outer = int(pd.unique(o).size)
    df_b = n_inner - n_outer
    df_w = n_total - n_inner
    if df_b <= 0 or df_w <= 0:
        return {"n": int(n_total), "n_inner": n_inner}
    ms_b = float(((inner_means - outer_means) ** 2).sum()) / df_b
    ms_w = float(((y - inner_means) ** 2).sum()) / df_w
    sizes = data.groupby([outer, inner]).size()
    outer_sizes = sizes.groupby(level=0).sum()
    squared = (sizes ** 2).groupby(level=0).sum()
    n0 = float((n_total - (squared / outer_sizes).sum()) / df_b)
    var_inner = max(0.0, (ms_b - ms_w) / n0)
    total = var_inner + ms_w
    return {"n": int(n_total), "n_inner": n_inner, "n_outer": n_outer, "n0": n0,
            "ms_inner": ms_b, "ms_residual": ms_w, "var_inner": var_inner, "var_residual": ms_w,
            "inner_share": var_inner / total if total > 0 else np.nan}


def tail_overlap(y, pred, fraction: float = 0.2, lowest: bool = True) -> float:
    y, pred = _arrays(y, pred)
    m = max(1, int(round(fraction * y.size)))
    sign = 1.0 if lowest else -1.0
    obs = set(np.argsort(sign * y.astype(float), kind="stable")[:m].tolist())
    hat = set(np.argsort(sign * pred.astype(float), kind="stable")[:m].tolist())
    return len(obs & hat) / m


def chance_overlap_band(n: int, fraction: float = 0.2, level: float = 0.95) -> list[float]:
    """Interval of the tail overlap expected from a random ranking of n plots."""
    m = max(1, int(round(fraction * n)))
    dist = hypergeom(n, m, m)
    lo, hi = dist.ppf([(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo / m), float(hi / m)]


def agreement(y, pred, fraction: float = 0.2) -> dict:
    """Agreement inside one group. r2_within removes the group bias first."""
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    n = int(y.size)
    if n < 3:
        return {"n": n}
    err = pred - y
    yc = y - y.mean()
    ec = err - err.mean()
    sst = float((yc ** 2).sum())
    sd_obs = float(y.std(ddof=0))
    sd_pred = float(pred.std(ddof=0))
    const = sd_pred < 1e-12 or sd_obs < 1e-12
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "within_rmse": float(np.sqrt(np.mean(ec ** 2))),
        "r2_within": float(1.0 - (ec ** 2).sum() / sst) if sst > 0 else np.nan,
        "pearson": np.nan if const else float(np.corrcoef(y, pred)[0, 1]),
        "spearman": np.nan if const else float(spearmanr(y, pred).statistic),
        "sd_obs": sd_obs,
        "sd_pred": sd_pred,
        "sd_ratio": sd_pred / sd_obs if sd_obs > 0 else np.nan,
        "low_recall": np.nan if const else tail_overlap(y, pred, fraction, lowest=True),
        "high_overlap": np.nan if const else tail_overlap(y, pred, fraction, lowest=False),
    }


AGREEMENT_KEYS = ["rmse", "bias", "within_rmse", "r2_within", "pearson", "spearman",
                  "sd_obs", "sd_pred", "sd_ratio", "low_recall", "high_overlap"]


def grouped_agreement(y, pred, groups, min_n: int = 5, fraction: float = 0.2, exclude=()) -> dict:
    """Per-group agreement and the unweighted mean over groups with at least min_n plots."""
    y, pred, groups = _arrays(y, pred, groups)
    rows = []
    for key in pd.unique(groups):
        mask = groups == key
        if mask.sum() < min_n:
            continue
        row = {"group": str(key), **agreement(y[mask], pred[mask], fraction)}
        row["chance_band"] = chance_overlap_band(int(mask.sum()), fraction)
        rows.append(row)
    used = [row for row in rows if row["group"] not in set(map(str, exclude))]
    macro = {key: _safe_mean([row.get(key, np.nan) for row in used]) for key in AGREEMENT_KEYS}
    return {"by_group": rows, "macro": macro, "groups_in_macro": [row["group"] for row in used]}


def _safe_mean(values) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def error_decomposition(y, pred, outer, inner=None) -> dict:
    """Split squared error into outer-group, inner-within-outer and residual parts.

    The three mean-square terms add up to the overall MSE.
    """
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    err = pred - y
    outer = np.asarray(outer)
    e_outer = pd.Series(err).groupby(outer).transform("mean").to_numpy()
    out = {"n": int(y.size), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "outer_rmse": float(np.sqrt(np.mean(e_outer ** 2)))}
    if inner is None:
        out["within_outer_rmse"] = float(np.sqrt(np.mean((err - e_outer) ** 2)))
        return out
    inner = np.asarray(inner)
    e_inner = pd.Series(err).groupby(inner).transform("mean").to_numpy()
    out["between_inner_rmse"] = float(np.sqrt(np.mean((e_inner - e_outer) ** 2)))
    out["within_inner_rmse"] = float(np.sqrt(np.mean((err - e_inner) ** 2)))
    out["within_outer_rmse"] = float(np.sqrt(np.mean((err - e_outer) ** 2)))
    return out


def between_inner_agreement(y, pred, outer, inner, min_inner: int = 3) -> dict:
    """Agreement of inner-group means after removing outer-group means.

    Only outer groups with at least min_inner inner groups contribute.
    """
    frame = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(pred, float),
                          "outer": np.asarray(outer), "inner": np.asarray(inner)})
    means = frame.groupby(["outer", "inner"], as_index=False)[["y", "p"]].mean()
    counts = means.groupby("outer")["inner"].transform("size")
    means = means[counts >= min_inner].copy()
    if means.empty:
        return {"n_inner": 0}
    means["yc"] = means["y"] - means.groupby("outer")["y"].transform("mean")
    means["pc"] = means["p"] - means.groupby("outer")["p"].transform("mean")
    sst = float((means["yc"] ** 2).sum())
    sse = float(((means["pc"] - means["yc"]) ** 2).sum())
    const = means["pc"].std() < 1e-12
    return {
        "n_inner": int(len(means)), "n_outer": int(means["outer"].nunique()),
        "r2_within": 1.0 - sse / sst if sst > 0 else np.nan,
        "spearman": np.nan if const else float(spearmanr(means["yc"], means["pc"]).statistic),
        "sd_obs": float(means["yc"].std(ddof=0)), "sd_pred": float(means["pc"].std(ddof=0)),
    }


def within_inner_agreement(y, pred, inner, min_n: int = 3) -> dict:
    """Pooled agreement after removing each inner group's own means (plots within fields)."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(pred, float), "inner": np.asarray(inner)})
    counts = frame.groupby("inner")["y"].transform("size")
    frame = frame[counts >= min_n].copy()
    if frame.empty:
        return {"n": 0}
    frame["yc"] = frame["y"] - frame.groupby("inner")["y"].transform("mean")
    frame["pc"] = frame["p"] - frame.groupby("inner")["p"].transform("mean")
    sst = float((frame["yc"] ** 2).sum())
    sse = float(((frame["pc"] - frame["yc"]) ** 2).sum())
    const = frame["pc"].std() < 1e-12
    return {
        "n": int(len(frame)), "n_inner": int(frame["inner"].nunique()),
        "r2_within": 1.0 - sse / sst if sst > 0 else np.nan,
        "spearman": np.nan if const else float(spearmanr(frame["yc"], frame["pc"]).statistic),
        "sd_obs": float(frame["yc"].std(ddof=0)), "sd_pred": float(frame["pc"].std(ddof=0)),
        "within_rmse": float(np.sqrt(np.mean((frame["pc"] - frame["yc"]) ** 2))),
    }


def stratified_cluster_draws(strata, clusters, n_draws: int, seed: int = 42):
    """Yield row indices: clusters resampled with replacement inside each stratum."""
    strata, clusters = _arrays(strata, clusters)
    layout = []
    for stratum in pd.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        keys = clusters[rows]
        members = [rows[keys == key] for key in pd.unique(keys)]
        layout.append(members)
    rng = np.random.default_rng(seed)
    for _ in range(n_draws):
        picked = []
        for members in layout:
            chosen = rng.integers(0, len(members), len(members))
            picked.extend(members[i] for i in chosen)
        yield np.concatenate(picked)


def percentile_interval(values, level: float = 0.95) -> list[float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return [np.nan, np.nan]
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
