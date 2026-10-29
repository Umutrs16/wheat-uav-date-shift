"""Scale-aware Sentinel tests that leave the published UAV scores unchanged.

Satellite values are attached once per plot and scored again after plots that
share a 20 m pixel are treated as one cluster. A nearest earlier scene is never
described as a same-day observation. Sentinel-1 orbits are not mixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from wheat_ppqmtl.indices import vegetation_indices
from wheat_ppqmtl.metrics import paired_rmse_difference, regression_metrics
from wheat_ppqmtl.paths import configure_stdio, project_root

UAV_FEATURES = ["B", "G", "R", "RE", "NIR", "ndvi", "ndre", "gndvi", "cire", "rvi", "osavi", "evi2", "doy_sin", "doy_cos"]
S2_FEATURES = ["s2_ndvi", "s2_ndre", "s2_swir", "s2_lag_days", "s2_within_10d", "s2_n_prior", "s2_ndvi_delta", "s2_missing"]
S1_FEATURES = ["s1_vv", "s1_vh", "s1_ratio", "s1_lag_days", "s1_missing"]
COMPONENTS = (2, 3, 4, 5, 6)
CLEAR_SCL = {4, 5, 7}
FLIGHTS = ["2024-03-31", "2024-04-12", "2024-04-25", "2024-05-18", "2024-06-06"]


def _safe_index(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = a + b
    out = pd.Series(np.nan, index=a.index, dtype=float)
    ok = denom.abs() >= 1e-6
    out.loc[ok] = (a.loc[ok] - b.loc[ok]) / denom.loc[ok]
    return out


def _latest_prior(samples: pd.DataFrame, flight: pd.Timestamp, valid: pd.Series) -> pd.Series | None:
    prior = samples[(samples["acquired"] <= flight) & valid].sort_values("acquired")
    if prior.empty:
        return None
    return prior.iloc[-1]


def _attach_s2(plots: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy()
    samples["acquired"] = pd.to_datetime(samples["date"], utc=True)
    samples["point_clear"] = samples["scl_10m"].isin(CLEAR_SCL) & samples["scl_20m"].isin(CLEAR_SCL)
    samples["s2_ndvi"] = _safe_index(samples["B8"], samples["B4"])
    samples["s2_ndre"] = _safe_index(samples["B8A"], samples["B5"])
    samples["s2_swir"] = _safe_index(samples["B8A"], samples["B11"])
    grouped = {key: frame for key, frame in samples.groupby("sample_id")}
    rows = []
    for plot in plots.itertuples(index=False):
        flight = pd.Timestamp(plot.date_iso, tz="UTC")
        history = grouped.get(plot.sample_id)
        row = {"sample_id": plot.sample_id}
        if history is None:
            rows.append(row)
            continue
        clear = history[history["point_clear"]].sort_values("acquired")
        prior = clear[clear["acquired"] <= flight]
        row["s2_n_prior"] = int(len(prior))
        row["pixel_10m"] = np.nan
        row["pixel_20m"] = np.nan
        if prior.empty:
            row["s2_missing"] = 1
            rows.append(row)
            continue
        latest = prior.iloc[-1]
        previous = prior.iloc[-2] if len(prior) > 1 else None
        lag = (flight - latest["acquired"]).total_seconds() / 86400
        row.update({
            "s2_date": latest["date"],
            "s2_lag_days": lag,
            "s2_within_10d": int(lag <= 10),
            "s2_ndvi": latest["s2_ndvi"],
            "s2_ndre": latest["s2_ndre"],
            "s2_swir": latest["s2_swir"],
            "s2_ndvi_delta": np.nan if previous is None else latest["s2_ndvi"] - previous["s2_ndvi"],
            "s2_missing": 0,
            "pixel_10m": f"{int(round(latest['col_10m']))}_{int(round(latest['row_10m']))}",
            "pixel_20m": f"{int(round(latest['col_20m']))}_{int(round(latest['row_20m']))}",
            "col_10m": int(round(latest["col_10m"])),
            "row_10m": int(round(latest["row_10m"])),
        })
        rows.append(row)
    return plots.merge(pd.DataFrame(rows), on="sample_id", how="left")


def _attach_s1(frame: pd.DataFrame, samples: pd.DataFrame, orbit: int) -> pd.DataFrame:
    use = samples[samples["relative_orbit"] == orbit].copy()
    use["acquired"] = pd.to_datetime(use["date"], utc=True)
    grouped = {key: part for key, part in use.groupby("sample_id")}
    rows = []
    for plot in frame.itertuples(index=False):
        flight = pd.Timestamp(plot.date_iso, tz="UTC")
        history = grouped.get(plot.sample_id)
        row = {"sample_id": plot.sample_id, "s1_orbit": orbit, "s1_missing": 1}
        if history is not None:
            prior = history[history["acquired"] <= flight].sort_values("acquired")
            if len(prior):
                latest = prior.iloc[-1]
                lag = (flight - latest["acquired"]).total_seconds() / 86400
                row.update({
                    "s1_date": latest["date"],
                    "s1_lag_days": lag,
                    "s1_vv": latest["VV"],
                    "s1_vh": latest["VH"],
                    "s1_ratio": latest["VV"] - latest["VH"],
                    "s1_missing": 0,
                })
        rows.append(row)
    return frame.merge(pd.DataFrame(rows), on="sample_id", how="left")


def _choose_orbit(plots: pd.DataFrame, samples: pd.DataFrame) -> int:
    lags = {}
    for orbit, part in samples.groupby("relative_orbit"):
        attached = _attach_s1(plots, samples, int(orbit))
        lags[int(orbit)] = float(attached["s1_lag_days"].median())
    return min(lags, key=lags.get)


def _strict_ten_day(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    late = out["s2_within_10d"] != 1
    for column in ("s2_ndvi", "s2_ndre", "s2_swir", "s2_ndvi_delta"):
        out.loc[late, column] = np.nan
    out.loc[late, "s2_missing"] = 1
    return out


def _components(train: pd.DataFrame, columns: list[str], target: str) -> int:
    """Choose components from training dates only.

    The forward step that trains on 31 March alone cannot leave a training
    date out. It uses two components, the lower end of the pre-specified
    range. The test date is not in this frame.
    """
    dates = [day for day in train["date_iso"].drop_duplicates().tolist() if (train["date_iso"] == day).sum() >= 8]
    if len(dates) < 2:
        return 2
    scores = []
    for n_comp in COMPONENTS:
        if n_comp >= len(columns) or n_comp >= len(train):
            continue
        errors = []
        for held in dates:
            fit = train[train["date_iso"] != held]
            test = train[train["date_iso"] == held]
            if len(fit) <= n_comp or test.empty:
                continue
            pred = _predict(fit, test, columns, target, n_comp)
            errors.append(regression_metrics(test[target], pred)["rmse"])
        if errors:
            scores.append((float(np.mean(errors)), n_comp))
    return min(scores)[1] if scores else 2


def _predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], target: str, n_comp: int) -> np.ndarray:
    medians = train[columns].median()
    x_train = train[columns].fillna(medians).to_numpy(float)
    x_test = test[columns].fillna(medians).to_numpy(float)
    scaler = StandardScaler().fit(x_train)
    n_comp = min(n_comp, x_train.shape[0] - 1, x_train.shape[1])
    n_comp = max(1, n_comp)
    model = PLSRegression(n_components=n_comp)
    model.fit(scaler.transform(x_train), train[target].to_numpy(float))
    return model.predict(scaler.transform(x_test)).reshape(-1)


def _run_design(frame: pd.DataFrame, columns: list[str], target: str, design: str) -> pd.DataFrame:
    use = frame.dropna(subset=[target]).copy()
    if target == "lai":
        use = use[np.isclose(use["lai_qc"], 1.0)]
    rows = []
    dates = [day for day in FLIGHTS if (use["date_iso"] == day).any()]
    for held in dates:
        if design == "lodo":
            train = use[use["date_iso"] != held]
        else:
            train = use[use["date_iso"] < held]
        test = use[use["date_iso"] == held]
        if len(train) < 12 or test.empty:
            continue
        n_comp = _components(train, columns, target)
        pred = _predict(train, test, columns, target, n_comp)
        part = test[["sample_id", "date_iso", "pixel_20m", target]].copy()
        part["y_true"] = part[target]
        part["y_pred"] = pred
        part["n_components"] = n_comp
        part["design"] = design
        part["target"] = target
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _summarize(pred: pd.DataFrame, label: str) -> dict:
    out = {"model": label, "design": pred["design"].iloc[0], "target": pred["target"].iloc[0]}
    metrics = regression_metrics(pred["y_true"], pred["y_pred"])
    out.update({key: metrics[key] for key in ("rmse", "mae", "bias", "r2", "n")})
    date_rows = []
    for day, part in pred.groupby("date_iso"):
        if len(part) < 8:
            continue
        item = regression_metrics(part["y_true"], part["y_pred"])
        date_rows.append({"date": day, "rmse": item["rmse"], "bias": item["bias"], "n": item["n"]})
    out["macro_rmse"] = float(np.mean([item["rmse"] for item in date_rows])) if date_rows else None
    out["by_date"] = date_rows
    pixel = pred.groupby(["date_iso", "pixel_20m"], as_index=False).agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
    pixel_metrics = regression_metrics(pixel["y_true"], pixel["y_pred"])
    out["pixel_rmse"] = pixel_metrics["rmse"]
    out["pixel_n"] = pixel_metrics["n"]
    return out


def _paired(reference: pd.DataFrame, alternative: pd.DataFrame) -> dict:
    merged = reference[["sample_id", "y_true", "y_pred", "pixel_20m"]].merge(
        alternative[["sample_id", "y_pred"]], on="sample_id", suffixes=("_ref", "_alt")
    )
    gap = paired_rmse_difference(
        merged["y_true"].to_numpy(float),
        merged["y_pred_ref"].to_numpy(float),
        merged["y_pred_alt"].to_numpy(float),
        merged["pixel_20m"].astype(str).to_numpy(),
    )
    return gap


def _support_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (day, pixel), part in frame.groupby(["date_iso", "pixel_10m"]):
        lcc = part["lcc"].dropna()
        rows.append({
            "date": day,
            "pixel_10m": pixel,
            "n_plots": int(len(part)),
            "lcc_mean": float(lcc.mean()) if len(lcc) else np.nan,
            "lcc_sd": float(lcc.std(ddof=1)) if len(lcc) > 1 else np.nan,
            "s2_ndvi": float(part["s2_ndvi"].iloc[0]) if part["s2_ndvi"].notna().any() else np.nan,
            "s2_lag_days": float(part["s2_lag_days"].iloc[0]) if part["s2_lag_days"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def _monitoring(root: Path) -> pd.DataFrame:
    intervals = pd.read_csv(root / "outputs" / "predictions" / "intervals_ensemble.csv")
    features = vegetation_indices(pd.read_csv(root / "data" / "features" / "features.csv"))
    held = intervals[
        (intervals["model"] == "spectral")
        & (intervals["scheme"] == "lodo")
        & (intervals["target"] == "lcc")
        & (intervals["role"] == "test")
        & (intervals["estimator"] == "ensemble")
        & (intervals["normalized"] == True)
    ].copy()
    held["abs_error"] = (held["y_true"] - held["y_pred"]).abs()
    held["width"] = held["y_hi90"] - held["y_lo90"]
    held = held.merge(features[["sample_id", "ndvi", "x"]], on="sample_id", how="left")
    rows = []
    rng = np.random.default_rng(42)
    for day, part in held.groupby("date_key"):
        part = part.reset_index(drop=True)
        total = float(part["abs_error"].sum())
        large = part["abs_error"] >= 15
        for fraction in (0.1, 0.2, 0.3):
            k = max(1, int(round(fraction * len(part))))
            uncertain = part.nlargest(k, "width")
            low_ndvi = part.nsmallest(k, "ndvi")
            ordered = part.sort_values("x")
            uniform_index = np.linspace(0, len(ordered) - 1, k).round().astype(int)
            uniform = ordered.iloc[uniform_index]
            random_capture = []
            random_recall = []
            for _ in range(200):
                chosen = part.iloc[rng.choice(len(part), size=k, replace=False)]
                random_capture.append(float(chosen["abs_error"].sum()) / total)
                random_recall.append(float(chosen["abs_error"].ge(15).sum()) / max(1, int(large.sum())))
            for name, chosen in (("uncertainty", uncertain), ("low_ndvi", low_ndvi), ("uniform_easting", uniform)):
                rows.append({
                    "date": day,
                    "budget": fraction,
                    "rule": name,
                    "error_capture": float(chosen["abs_error"].sum()) / total,
                    "large_error_recall": float((chosen["abs_error"] >= 15).sum()) / max(1, int(large.sum())),
                    "n": int(len(part)),
                    "k": k,
                })
            rows.append({
                "date": day,
                "budget": fraction,
                "rule": "random",
                "error_capture": float(np.mean(random_capture)),
                "large_error_recall": float(np.mean(random_recall)),
                "n": int(len(part)),
                "k": k,
            })
    return pd.DataFrame(rows)


def _calendar(out: Path, root: Path) -> None:
    s2 = pd.read_csv(root / "outputs" / "satellite" / "s2_dates.csv")
    s1 = pd.read_csv(root / "outputs" / "satellite" / "s1_scenes.csv")
    predictions = pd.read_csv(root / "outputs" / "satellite" / "scale_predictions.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.1), gridspec_kw={"height_ratios": [1.05, 1.0]})
    ax = axes[0]
    ax.scatter(pd.to_datetime(s2["date"]), np.full(len(s2), 3), s=22, c="#0072B2", label="S2 usable")
    for orbit, color in ((40, "#E69F00"), (142, "#009E73")):
        part = s1[s1["relative_orbit"] == orbit]
        ax.scatter(pd.to_datetime(part["date"]), np.full(len(part), 2 if orbit == 40 else 1), s=22, c=color, label=f"S1 orbit {orbit}")
    flight_time = pd.to_datetime(FLIGHTS)
    ax.scatter(flight_time, np.zeros(len(FLIGHTS)), s=46, c="#D55E00", marker="D", label="UAV flight", zorder=3)
    lags = [9, 1, 4, 7, 11]
    matched = pd.to_datetime(["2024-03-22", "2024-04-11", "2024-04-21", "2024-05-11", "2024-05-26"])
    for flight, scene, lag in zip(flight_time, matched, lags):
        color = "#9B2226" if lag > 10 else "#333333"
        ax.annotate(
            "",
            xy=(flight, 0.15),
            xytext=(scene, 2.85),
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "ls": (0, (2, 2))},
        )
        ax.text(flight, -0.38, f"−{lag} d", ha="center", va="top", fontsize=8, color=color)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["UAV flight", "S1 orbit 142", "S1 orbit 40", "S2 usable"])
    ax.set_ylim(-0.8, 3.55)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title("(a)", loc="left", fontsize=12, fontweight="bold")

    part = predictions[(predictions["design"] == "lodo") & (predictions["target"] == "lcc") & predictions["model"].isin(["uav", "uav_s2"])]
    dates = list(FLIGHTS)
    labels = ["31 Mar", "12 Apr", "25 Apr", "18 May", "6 Jun"]
    bar = axes[1]
    for model, color, name, offset in (
        ("uav", "#0072B2", "UAV predictors", -0.16),
        ("uav_s2", "#D55E00", "UAV + lagged Sentinel-2", 0.16),
    ):
        ys = []
        for day in dates:
            sub = part[(part["model"] == model) & (part["date_iso"] == day)]
            ys.append(float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2))))
        bar.bar([index + offset for index in range(5)], ys, width=0.3, color=color, label=name)
    bar.set_xticks(range(5), labels, fontsize=10)
    bar.set_ylabel("Leave-one-date-out RMSE (μg cm−2)", fontsize=10)
    bar.tick_params(labelsize=9)
    bar.legend(frameon=False, fontsize=9, loc="upper left")
    bar.set_ylim(0, 22)
    bar.set_title("(b)", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out / "fig9_sentinel_panels.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 4)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main() -> None:
    configure_stdio()
    root = project_root()
    satellite = root / "outputs" / "satellite"
    plots = vegetation_indices(pd.read_csv(root / "data" / "interim" / "observations_plot.csv"))
    plots["doy_sin"] = np.sin(2 * np.pi * plots["doy"] / 365)
    plots["doy_cos"] = np.cos(2 * np.pi * plots["doy"] / 365)
    s2 = pd.read_csv(satellite / "s2_plot_samples.csv")
    s1 = pd.read_csv(satellite / "s1_plot_samples.csv")
    frame = _attach_s2(plots, s2)
    orbit = _choose_orbit(plots, s1)
    frame = _attach_s1(frame, s1, orbit)
    missing_pixel = frame["pixel_20m"].isna()
    frame.loc[missing_pixel, "pixel_20m"] = "plot:" + frame.loc[missing_pixel, "sample_id"].astype(str)
    strict = _strict_ten_day(frame)
    support = _support_table(frame.dropna(subset=["pixel_10m"]))
    support.to_csv(satellite / "pixel_support.csv", index=False)
    specs = {
        "uav": (frame, UAV_FEATURES),
        "uav_s2": (frame, UAV_FEATURES + S2_FEATURES),
        "uav_s2_within_10d": (strict, UAV_FEATURES + S2_FEATURES),
        "uav_s2_s1": (frame, UAV_FEATURES + S2_FEATURES + S1_FEATURES),
        "s2_only": (frame, S2_FEATURES),
    }
    summaries = []
    stored = []
    for label, (table, columns) in specs.items():
        for design in ("lodo", "forward"):
            for target in ("lcc", "lai"):
                pred = _run_design(table, columns, target, design)
                if pred.empty:
                    continue
                pred["model"] = label
                stored.append(pred)
                summaries.append(_summarize(pred, label))
                print(label, design, target, round(summaries[-1]["rmse"], 3), flush=True)
    predictions = pd.concat(stored, ignore_index=True)
    predictions.to_csv(satellite / "scale_predictions.csv", index=False)
    reference = predictions[(predictions["model"] == "uav") & (predictions["design"] == "lodo") & (predictions["target"] == "lcc")]
    paired = {}
    for label in ("uav_s2", "uav_s2_within_10d", "uav_s2_s1", "s2_only"):
        alternative = predictions[(predictions["model"] == label) & (predictions["design"] == "lodo") & (predictions["target"] == "lcc")]
        paired[label] = _paired(reference, alternative)
    forward_ref = predictions[(predictions["model"] == "uav") & (predictions["design"] == "forward") & (predictions["target"] == "lcc")]
    forward_alt = predictions[(predictions["model"] == "uav_s2") & (predictions["design"] == "forward") & (predictions["target"] == "lcc")]
    paired["forward_uav_s2"] = _paired(forward_ref, forward_alt)
    monitoring = _monitoring(root)
    monitoring.to_csv(satellite / "monitoring_priority.csv", index=False)
    _calendar(root / "outputs" / "figures", root)
    counts = support.groupby("date")["n_plots"].agg(["count", "median", "max"])
    multi = support[support["n_plots"] > 1]
    result = {
        "s1_orbit_selected_by_median_lag": orbit,
        "s2_lag_by_flight": frame.groupby("date_iso")["s2_lag_days"].median().round(2).to_dict(),
        "s2_within_10d_by_flight": frame.groupby("date_iso")["s2_within_10d"].mean().round(3).to_dict(),
        "pixels_with_more_than_one_plot": int(len(multi)),
        "pixel_dates": int(len(support)),
        "within_pixel_lcc_sd_median": None if multi["lcc_sd"].dropna().empty else round(float(multi["lcc_sd"].median()), 3),
        "plots_per_pixel": counts.reset_index().to_dict("records"),
        "models": summaries,
        "paired_lodo_lcc_rmse_minus_uav": paired,
        "monitoring_macro": monitoring.groupby(["budget", "rule"])[["error_capture", "large_error_recall"]].mean().reset_index().to_dict("records"),
    }
    (satellite / "experiment_summary.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(result)["paired_lodo_lcc_rmse_minus_uav"], indent=2), flush=True)


if __name__ == "__main__":
    main()
