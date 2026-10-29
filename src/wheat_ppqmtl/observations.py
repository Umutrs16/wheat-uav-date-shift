from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _norm(name: str) -> str:
    text = str(name).lower().replace("μ", "u").replace("µ", "u")
    return "".join(ch for ch in text if ch.isascii() and ch.isalnum())


def parse_survey_date(value) -> date:
    text = str(value).strip()
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day") and not isinstance(value, str):
        return date(int(value.year), int(value.month), int(value.day))
    parts = text.replace("/", ".").replace("-", ".").split(".")
    if len(parts) < 3:
        raise ValueError(f"Unrecognised date: {value!r}")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def date_key(day: date) -> str:
    """Match the released GeoTIFF names, which are not zero-padded."""
    return f"{day.year}.{day.month}.{day.day}"


def image_name(day: date) -> str:
    return f"{date_key(day)}_UAV.tif"


def _pick(columns: list[str], *candidates: str) -> str | None:
    pairs = [(_norm(column), column) for column in columns]
    for candidate in candidates:
        for key, original in pairs:
            if key == candidate:
                return original
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        for key, original in pairs:
            if key.startswith(candidate) and key != candidate:
                return original
    return None


def standardize_observations(raw: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw.columns)
    mapping = {
        "plot_id": _pick(columns, "plotid", "plot"),
        "date": _pick(columns, "date"),
        "lat": _pick(columns, "lat", "latitude"),
        "lon": _pick(columns, "lon", "longitude"),
        "B": _pick(columns, "b"),
        "G": _pick(columns, "g"),
        "R": _pick(columns, "r"),
        "RE": _pick(columns, "re", "rededge"),
        "NIR": _pick(columns, "nir"),
        "lcc": _pick(columns, "lcc", "lccugcm2"),
        "lai_qc": _pick(columns, "laiqc"),
        "lai": _pick(columns, "lai"),
    }
    missing = [key for key, value in mapping.items() if value is None and key != "lai_qc"]
    if missing:
        raise ValueError(f"Observation table is missing columns: {missing}; found {columns}")
    frame = pd.DataFrame()
    frame["plot_id"] = raw[mapping["plot_id"]].astype(int)
    parsed = [parse_survey_date(v) for v in raw[mapping["date"]]]
    frame["date_key"] = [date_key(v) for v in parsed]
    frame["date_iso"] = [v.isoformat() for v in parsed]
    frame["doy"] = [v.timetuple().tm_yday for v in parsed]
    frame["lat"] = raw[mapping["lat"]].astype(float)
    frame["lon"] = raw[mapping["lon"]].astype(float)
    for band in ("B", "G", "R", "RE", "NIR"):
        frame[band] = raw[mapping[band]].astype(float)
    frame["lcc"] = raw[mapping["lcc"]].astype(float)
    frame["lai"] = pd.to_numeric(raw[mapping["lai"]], errors="coerce")
    if mapping["lai_qc"] is None:
        frame["lai_qc"] = np.nan
    else:
        frame["lai_qc"] = pd.to_numeric(raw[mapping["lai_qc"]], errors="coerce")
    frame["row_id"] = np.arange(len(frame))
    frame["sample_id"] = [
        f"{int(pid)}_{day}_{int(row)}"
        for pid, day, row in zip(frame["plot_id"], frame["date_key"], frame["row_id"])
    ]
    # A plot measured once on a date keeps a stable id even if row order changes,
    # as long as plot_id and date are unique. Duplicate plot-dates keep the row suffix.
    counts = frame.groupby(["plot_id", "date_key"])["row_id"].transform("size")
    unique_mask = counts == 1
    frame.loc[unique_mask, "sample_id"] = [
        f"{int(pid)}_{day}"
        for pid, day in zip(frame.loc[unique_mask, "plot_id"], frame.loc[unique_mask, "date_key"])
    ]
    return frame
