"""Recompute the headline scores from the saved prediction tables.

Run this file from the reproducibility folder, or from scripts/. It reads the
exported tables and does not refit a model.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _folder() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "oof_predictions.csv").exists():
        return here
    return here.parent / "outputs" / "reproducibility"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("sample_id", as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), date_key=("date_key", "first"))


def _check(name: str, value: float, expected: float, digits: int = 2) -> bool:
    shown = f"{value:.{digits}f}"
    target = f"{expected:.{digits}f}"
    ok = shown == target
    print(f"{'OK' if ok else 'MISMATCH'}  {name}: recomputed {shown}  recorded {target}")
    return ok


def main() -> None:
    folder = _folder()
    oof = pd.read_csv(folder / "oof_predictions.csv")
    forward = pd.read_csv(folder / "forward_predictions.csv")
    intervals = pd.read_csv(folder / "conformal_intervals.csv")
    ok = True

    def neural(frame, scheme):
        part = frame[(frame["model"] == "spectral") & (frame["scheme"] == scheme) & (frame["target"] == "lcc") & (frame["role"] == "test")]
        return _ensemble(part)

    grouped = neural(oof, "grouped_random")
    spatial = neural(oof, "spatial_block")
    held = neural(oof, "lodo")
    later = neural(forward, "forward")
    ok &= _check("ID-grouped LCC RMSE", _rmse(grouped.y_true.to_numpy(), grouped.y_pred.to_numpy()), 6.59)
    ok &= _check("40 m spatial LCC RMSE", _rmse(spatial.y_true.to_numpy(), spatial.y_pred.to_numpy()), 6.90)
    ok &= _check("LODO LCC RMSE", _rmse(held.y_true.to_numpy(), held.y_pred.to_numpy()), 12.66)
    ok &= _check("Forward LCC RMSE", _rmse(later.y_true.to_numpy(), later.y_pred.to_numpy()), 14.44)
    ok &= _check("Forward n", float(len(later)), 403, digits=0)

    common = set(later["sample_id"])
    matched = held[held["sample_id"].isin(common)]
    ok &= _check("Matched LODO LCC RMSE", _rmse(matched.y_true.to_numpy(), matched.y_pred.to_numpy()), 12.99)
    ok &= _check("Matched n", float(len(matched)), 403, digits=0)

    cover = intervals[(intervals["model"] == "spectral") & (intervals["scheme"] == "lodo") & (intervals["target"] == "lcc") & (intervals["normalized"] == True) & (intervals["estimator"] == "ensemble")]
    inside = (cover["y_true"] >= cover["y_lo90"]) & (cover["y_true"] <= cover["y_hi90"])
    macro = float(inside.groupby(cover["date_key"]).mean().mean())
    pooled = float(inside.mean())
    ok &= _check("90% macro coverage (%)", 100 * macro, 54.2, digits=1)
    ok &= _check("90% pooled coverage (%)", 100 * pooled, 63.1, digits=1)

    satellite = pd.read_csv(folder / "satellite_pls_predictions.csv")
    for model, expected in (("uav", 11.53), ("uav_s2", 12.92), ("uav_s2_s1", 17.73), ("s2_only", 29.70)):
        part = satellite[(satellite["design"] == "lodo") & (satellite["target"] == "lcc") & (satellite["model"] == model)]
        ok &= _check(f"PLS {model} LODO RMSE", _rmse(part.y_true.to_numpy(), part.y_pred.to_numpy()), expected)
    june = satellite[(satellite["design"] == "lodo") & (satellite["target"] == "lcc") & (satellite["model"] == "uav_s2_s1") & (satellite["date_iso"] == "2024-06-06")]
    ok &= _check("June S1 mean prediction", float(june.y_pred.mean()), -47.82)

    matches = pd.read_csv(folder / "sentinel_plot_matches.csv")
    lags = matches.groupby("flight_date")["s2_lag_days"].first().to_dict()
    expected_lags = {"2024-03-31": 9, "2024-04-12": 1, "2024-04-25": 4, "2024-05-18": 7, "2024-06-06": 11}
    for day, lag in expected_lags.items():
        ok &= _check(f"S2 lag {day}", float(lags[day]), lag, digits=0)
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
