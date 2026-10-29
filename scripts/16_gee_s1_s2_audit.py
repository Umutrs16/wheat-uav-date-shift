"""Inventory Sentinel-1/2 over the Gaocheng field. This does not score retrievals.

The 250 m buffer rectangle is an inventory window, not a cadastral boundary.
A scene is marked usable only when SCL classes 4, 5 and 7 cover at least 80%
of that window. Future scenes are not counted as observations of a flight.
"""

from __future__ import annotations

import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path

import ee
import pandas as pd

from wheat_ppqmtl.paths import configure_stdio, project_root

PROJECT = os.environ.get("EE_PROJECT", "")
CENTER = (114.853997, 37.881194)
BUFFER_M = 250
START = "2023-10-01"
END = "2024-06-16"
CLEAR_SCL = (4, 5, 7)
CLEAR_MIN = 0.80
UAV_DATES = ("2024-03-31", "2024-04-12", "2024-04-25", "2024-05-18", "2024-06-06")
WINDOWS = ((0, 3), (4, 7), (8, 10))


def _roi() -> ee.Geometry:
    lon, lat = CENTER
    return ee.Geometry.Point([lon, lat]).buffer(BUFFER_M).bounds()


def _s2_features(images: ee.ImageCollection, roi: ee.Geometry) -> list[dict]:
    ids = images.aggregate_array("system:index").getInfo()
    rows = []
    for index in ids:
        image = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{index}")
        scl = image.select("SCL")
        clear = ee.Image.constant(0).rename("clear")
        for code in CLEAR_SCL:
            clear = clear.Or(scl.eq(code))
        clear = clear.rename("clear")
        stats = clear.addBands(scl.eq(4).rename("vegetation")).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=20,
            maxPixels=10_000_000,
        ).getInfo()
        props = image.toDictionary(["PRODUCT_ID", "MGRS_TILE", "CLOUDY_PIXEL_PERCENTAGE", "system:time_start"]).getInfo()
        rows.append({
            "scene_id": props.get("PRODUCT_ID") or index,
            "system_index": index,
            "date": datetime.fromtimestamp(props["system:time_start"] / 1000, timezone.utc).strftime("%Y-%m-%d"),
            "millis": props["system:time_start"],
            "tile": props.get("MGRS_TILE"),
            "cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE"),
            "roi_clear": stats.get("clear"),
            "roi_vegetation": stats.get("vegetation"),
            "usable": stats.get("clear") is not None and float(stats["clear"]) >= CLEAR_MIN,
        })
        print(f"S2 {len(rows)}/{len(ids)} {rows[-1]['date']} clear={rows[-1]['roi_clear']}", flush=True)
    return rows


def _s1_features(images: ee.ImageCollection) -> list[dict]:
    collection = images.map(lambda img: ee.Feature(None, {
        "system_index": img.get("system:index"),
        "millis": img.get("system:time_start"),
        "orbit_pass": img.get("orbitProperties_pass"),
        "relative_orbit": img.get("relativeOrbitNumber_start"),
        "platform": img.get("platform_number"),
    }))
    features = collection.getInfo()["features"]
    rows = []
    for feature in features:
        props = feature["properties"]
        rows.append({
            "scene_id": props["system_index"],
            "date": datetime.fromtimestamp(props["millis"] / 1000, timezone.utc).strftime("%Y-%m-%d"),
            "millis": props["millis"],
            "orbit_pass": props.get("orbit_pass"),
            "relative_orbit": props.get("relative_orbit"),
            "platform": props.get("platform"),
        })
    return rows


def _windows(scenes: pd.DataFrame, usable_only: bool, sensor: str) -> pd.DataFrame:
    rows = []
    table = scenes[scenes["usable"]] if usable_only and "usable" in scenes.columns else scenes
    for flight in UAV_DATES:
        flight_ms = pd.Timestamp(flight, tz="UTC")
        prior = table[pd.to_datetime(table["date"], utc=True) <= flight_ms].copy()
        if len(prior):
            prior["lag_days"] = (flight_ms - pd.to_datetime(prior["date"], utc=True)).dt.total_seconds() / 86400
        for lo, hi in WINDOWS:
            part = prior[(prior["lag_days"] >= lo) & (prior["lag_days"] <= hi)] if len(prior) else prior
            nearest = prior["lag_days"].min() if len(prior) else math.nan
            rows.append({
                "sensor": sensor,
                "flight": flight,
                "window": f"{lo}-{hi}d",
                "n_scenes": int(len(part)),
                "nearest_prior_lag_days": None if pd.isna(nearest) else round(float(nearest), 2),
                "usable_only": usable_only,
            })
    return pd.DataFrame(rows)


def main() -> None:
    configure_stdio()
    if not PROJECT:
        raise SystemExit("Set EE_PROJECT to a Google Earth Engine cloud project before running this script.")
    ee.Initialize(project=PROJECT)
    roi = _roi()
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(START, END)
    )
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(START, END)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    print("S2 scenes", s2.size().getInfo(), "S1 scenes", s1.size().getInfo(), flush=True)
    s2_rows = _s2_features(s2, roi)
    s1_rows = _s1_features(s1)
    out = project_root() / "outputs" / "satellite"
    out.mkdir(parents=True, exist_ok=True)
    s2_frame = pd.DataFrame(s2_rows).sort_values("millis")
    s1_frame = pd.DataFrame(s1_rows).sort_values("millis")
    s2_frame.to_csv(out / "s2_scenes.csv", index=False)
    s1_frame.to_csv(out / "s1_scenes.csv", index=False)
    windows = pd.concat([
        _windows(s2_frame, usable_only=True, sensor="S2"),
        _windows(s1_frame.assign(usable=True), usable_only=False, sensor="S1"),
    ], ignore_index=True)
    windows.to_csv(out / "uav_scene_windows.csv", index=False)
    bounds = roi.bounds().getInfo()["coordinates"][0]
    meta = {
        "project": PROJECT,
        "center_lon": CENTER[0],
        "center_lat": CENTER[1],
        "buffer_m": BUFFER_M,
        "roi_bounds_lonlat": bounds,
        "start": START,
        "end": END,
        "clear_scl": list(CLEAR_SCL),
        "clear_fraction_minimum": CLEAR_MIN,
        "roi_note": "Bounding rectangle of a 250 m buffer. Not an official field boundary.",
        "n_s2": int(len(s2_frame)),
        "n_s2_usable": int(s2_frame["usable"].sum()),
        "n_s1": int(len(s1_frame)),
        "s1_orbits": s1_frame.groupby(["orbit_pass", "relative_orbit"]).size().reset_index(name="n").to_dict("records"),
    }
    (out / "audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
