"""Sample Sentinel-2 and Sentinel-1 at the plot coordinates.

One MGRS tile is kept for each date. Overlapping granules from the same
overpass are not treated as a second look. Reflectance is divided by 10000.
Red-edge and SWIR bands stay on their 20 m grid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ee
import pandas as pd

from wheat_ppqmtl.paths import configure_stdio, project_root

PROJECT = os.environ.get("EE_PROJECT", "")
PREFERRED_TILE = "50SLH"
S2_10M = ["B2", "B3", "B4", "B8"]
S2_20M = ["B5", "B6", "B7", "B8A", "B11", "B12"]


def _points(frame: pd.DataFrame) -> ee.FeatureCollection:
    features = [
        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {"sample_id": str(row.sample_id)})
        for row in frame.itertuples(index=False)
    ]
    return ee.FeatureCollection(features)


def _chosen_s2(scenes: pd.DataFrame) -> pd.DataFrame:
    usable = scenes[scenes["usable"]].copy()
    usable["tile_rank"] = (usable["tile"] != PREFERRED_TILE).astype(int)
    usable = usable.sort_values(["date", "roi_clear", "tile_rank"], ascending=[True, False, True])
    return usable.drop_duplicates("date", keep="first")


def _sample(image: ee.Image, points: ee.FeatureCollection, bands: list[str], scale_band: str, optical: bool) -> list[dict]:
    projection = image.select(scale_band).projection()
    values = image.select(bands)
    if optical:
        values = values.divide(10000)
    stack = values.addBands(ee.Image.pixelCoordinates(projection).rename(["col", "row"]))
    if optical:
        stack = stack.addBands(image.select("SCL"))
    sampled = stack.sampleRegions(collection=points, projection=projection, scale=projection.nominalScale(), geometries=False)
    rows = []
    for feature in sampled.getInfo()["features"]:
        props = feature["properties"]
        if props.get(scale_band) is None and props.get(bands[0]) is None:
            continue
        rows.append(props)
    return rows


def main() -> None:
    configure_stdio()
    if not PROJECT:
        raise SystemExit("Set EE_PROJECT to a Google Earth Engine cloud project before running this script.")
    ee.Initialize(project=PROJECT)
    root = project_root()
    out = root / "outputs" / "satellite"
    plots = pd.read_csv(root / "data" / "interim" / "observations_plot.csv")
    points = _points(plots[["sample_id", "lon", "lat"]])
    scenes = pd.read_csv(out / "s2_scenes.csv")
    chosen = _chosen_s2(scenes)
    chosen.to_csv(out / "s2_dates.csv", index=False)
    print(f"S2 dates {len(chosen)}", flush=True)

    s2_rows = []
    for record in chosen.itertuples(index=False):
        image = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{record.system_index}")
        coarse = _sample(image, points, S2_20M, "B5", optical=True)
        fine = _sample(image, points, S2_10M, "B4", optical=True)
        fine_frame = pd.DataFrame(fine).rename(columns={"col": "col_10m", "row": "row_10m", "SCL": "scl_10m"})
        coarse_frame = pd.DataFrame(coarse).rename(columns={"col": "col_20m", "row": "row_20m", "SCL": "scl_20m"})
        keep = ["sample_id", "col_20m", "row_20m", "scl_20m", *S2_20M]
        merged = fine_frame.merge(coarse_frame[keep], on="sample_id", how="left")
        merged["scene_id"] = record.scene_id
        merged["system_index"] = record.system_index
        merged["date"] = record.date
        merged["tile"] = record.tile
        s2_rows.append(merged)
        print(f"S2 {record.date} {len(merged)}", flush=True)
    s2 = pd.concat(s2_rows, ignore_index=True)
    s2.to_csv(out / "s2_plot_samples.csv", index=False)

    s1_scenes = pd.read_csv(out / "s1_scenes.csv")
    s1_rows = []
    for record in s1_scenes.itertuples(index=False):
        image = ee.Image(f"COPERNICUS/S1_GRD/{record.scene_id}")
        rows = _sample(image, points, ["VV", "VH", "angle"], "VV", optical=False)
        frame = pd.DataFrame(rows).rename(columns={"col": "col_10m", "row": "row_10m"})
        frame["scene_id"] = record.scene_id
        frame["date"] = record.date
        frame["orbit_pass"] = record.orbit_pass
        frame["relative_orbit"] = record.relative_orbit
        s1_rows.append(frame)
        print(f"S1 {record.date} orbit {record.relative_orbit} {len(frame)}", flush=True)
    s1 = pd.concat(s1_rows, ignore_index=True)
    s1.to_csv(out / "s1_plot_samples.csv", index=False)
    meta = {
        "preferred_tile": PREFERRED_TILE,
        "n_s2_dates": int(len(chosen)),
        "n_s2_rows": int(len(s2)),
        "n_s1_rows": int(len(s1)),
        "s2_20m_note": "B5, B6, B7, B8A, B11 and B12 were sampled on the native 20 m grid.",
        "s2_scale": "Surface reflectance divided by 10000.",
    }
    (out / "sample_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
