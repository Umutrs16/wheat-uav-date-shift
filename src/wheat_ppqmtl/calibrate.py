from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wheat_ppqmtl.conformal import prediction_intervals
from wheat_ppqmtl.io_utils import write_json
from wheat_ppqmtl.metrics import coverage_and_width, interval_score
from wheat_ppqmtl.paths import configure_stdio, configs


def _seed_ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean prediction and total SD across initialisations.

    Total variance is the mean of sigma^2 + mu^2, minus the squared mean prediction.
    It contains the heteroscedastic term and the between-seed term.
    """
    keys = ["model", "scheme", "outer_fold", "sample_id", "target", "role"]
    rows = []
    for key, part in frame.groupby(keys, sort=False):
        mu = part["y_pred"].to_numpy(float)
        mu_bar = float(np.mean(mu))
        sigma = part["y_sigma"].to_numpy(float)
        if np.isfinite(sigma).all():
            total = float(np.mean(sigma ** 2 + mu ** 2) - mu_bar ** 2)
            sigma_total = float(np.sqrt(max(total, 0.0)))
        elif len(mu) > 1:
            sigma_total = float(np.std(mu, ddof=0))
        else:
            sigma_total = np.nan
        base = part.iloc[0]
        rows.append({
            "model": key[0], "scheme": key[1], "outer_fold": key[2], "sample_id": key[3],
            "target": key[4], "role": key[5], "date_key": base.get("date_key"),
            "y_true": float(base["y_true"]), "y_pred": mu_bar, "y_sigma": sigma_total,
            "lai_qc": base.get("lai_qc"), "seed": -1,
        })
    return pd.DataFrame(rows)


def _interval_rows(frame: pd.DataFrame, levels: tuple[float, ...], estimator: str, force_sigma: bool | None) -> tuple[list[pd.DataFrame], list[dict]]:
    pieces = []
    reports = []
    group_cols = ["model", "scheme", "outer_fold", "target"] if estimator == "ensemble" else ["model", "scheme", "outer_fold", "seed", "target"]
    for keys, part in frame.groupby(group_cols, sort=False):
        key = dict(zip(group_cols, keys))
        if key["target"] in {"lai", "ccc", "ccc_derived"}:
            qc = pd.to_numeric(part["lai_qc"], errors="coerce")
            part = part[np.isclose(qc, 1.0)]
        calibration = part[(part["role"] == "calibration") & np.isfinite(part["y_true"]) & np.isfinite(part["y_pred"])]
        test = part[part["role"] == "test"].copy()
        if len(calibration) == 0 or len(test) == 0:
            continue
        sigma_cal = calibration["y_sigma"].to_numpy(float)
        sigma_test = test["y_sigma"].to_numpy(float)
        use_sigma = np.isfinite(sigma_cal).all() and np.isfinite(sigma_test).all()
        if force_sigma is False:
            use_sigma = False
        if force_sigma is True and not use_sigma:
            continue
        abs_resid = np.abs(calibration["y_true"].to_numpy(float) - calibration["y_pred"].to_numpy(float))
        intervals = prediction_intervals(
            test["y_pred"].to_numpy(float),
            abs_resid,
            sigma_test if use_sigma else None,
            sigma_cal if use_sigma else None,
            levels,
        )
        test = test.reset_index(drop=True)
        for column, values in intervals.items():
            if column.startswith("q"):
                continue
            test[column] = values
        test["estimator"] = estimator
        test["normalized"] = bool(use_sigma)
        pieces.append(test)
        for level in levels:
            tag = str(int(round(level * 100)))
            finite = test[np.isfinite(test["y_true"])]
            stats = coverage_and_width(finite["y_true"], finite[f"y_lo{tag}"], finite[f"y_hi{tag}"])
            reports.append({
                "model": key["model"], "scheme": key["scheme"], "outer_fold": int(key["outer_fold"]),
                "seed": int(key.get("seed", -1)), "target": key["target"], "level": level,
                "normalized": bool(use_sigma), "estimator": estimator,
                "n_calibration": int(len(calibration)), "n_test": int(len(finite)),
                "interval_score": interval_score(finite["y_true"], finite[f"y_lo{tag}"], finite[f"y_hi{tag}"], 1 - level),
                **stats,
            })
    return pieces, reports


def calibrate() -> pd.DataFrame:
    cfg = configs()
    root = cfg["root"]
    skip = {"with_intervals.csv", "intervals_ensemble.csv"}
    files = [path for path in (root / "outputs" / "predictions").glob("*.csv") if path.name not in skip]
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    levels = tuple(float(level) for level in cfg["experiment"]["conformal_levels"])
    pieces, reports = _interval_rows(frame, levels, "seed", None)
    ensemble = _seed_ensemble(frame)
    for force in (True, False):
        extra, extra_reports = _interval_rows(ensemble, levels, "ensemble", force)
        pieces.extend(extra)
        reports.extend(extra_reports)
    predicted = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    seed_rows = predicted[predicted["estimator"] == "seed"] if len(predicted) else predicted
    ensemble_rows = predicted[predicted["estimator"] == "ensemble"] if len(predicted) else predicted
    seed_rows.to_csv(root / "outputs" / "predictions" / "with_intervals.csv", index=False)
    ensemble_rows.to_csv(root / "outputs" / "predictions" / "intervals_ensemble.csv", index=False)
    report = pd.DataFrame(reports)
    report.to_csv(root / "outputs" / "metrics" / "calibration_by_fold.csv", index=False)
    pooled = []
    if len(report):
        for keys, part in report.groupby(["model", "scheme", "target", "level", "normalized", "estimator"]):
            weights = part["n_test"].to_numpy(float) if "n_test" in part else np.ones(len(part))
            pooled.append({
                "model": keys[0], "scheme": keys[1], "target": keys[2], "level": float(keys[3]),
                "normalized": bool(keys[4]), "estimator": keys[5],
                "picp": float(part["picp"].mean()),
                "picp_weighted": float(np.average(part["picp"], weights=weights)),
                "mpiw": float(part["mpiw"].mean()),
                "interval_score": float(part["interval_score"].mean()),
                "n_calibration_mean": float(part["n_calibration"].mean()),
            })
    write_json(root / "outputs" / "metrics" / "calibration_summary.json", {"rows": pooled})
    print(f"intervals={len(seed_rows)} ensemble={len(ensemble_rows)} groups={len(pooled)}")
    return predicted


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Split-conformal intervals from the inner calibration split only.").parse_args()
    (configs()["root"] / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    calibrate()


if __name__ == "__main__":
    main()
