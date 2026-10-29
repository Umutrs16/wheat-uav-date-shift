"""Check the five flight-matched Sentinel-2 scenes with a stricter clear rule.

Classes 4 and 5 are vegetation and bare soil. Class 7 is unclassified and is
counted separately. Cloud probability is the L2A MSK_CLDPRB band.
"""

from __future__ import annotations

import json
import os

import ee
import pandas as pd

from wheat_ppqmtl.paths import configure_stdio, project_root

PROJECT = os.environ.get("EE_PROJECT", "")
SCENES = [
    "20240322T030521_20240322T031627_T50SLH",
    "20240411T030521_20240411T030632_T50SLH",
    "20240421T030531_20240421T031451_T50SLH",
    "20240511T030551_20240511T030907_T50SLH",
    "20240526T030519_20240526T031702_T50SLH",
]


def main() -> None:
    configure_stdio()
    if not PROJECT:
        raise SystemExit("Set EE_PROJECT to a Google Earth Engine cloud project before running this script.")
    ee.Initialize(project=PROJECT)
    root = project_root()
    plots = pd.read_csv(root / "data" / "interim" / "observations_plot.csv", usecols=["sample_id", "lon", "lat"])
    points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {"sample_id": str(row.sample_id)})
        for row in plots.itertuples(index=False)
    ])
    roi = ee.Geometry.Point([114.853997, 37.881194]).buffer(250).bounds()
    rows = []
    for index in SCENES:
        image = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{index}")
        scl = image.select("SCL")
        clear = scl.eq(4).Or(scl.eq(5)).rename("clear45")
        stack = clear.addBands([
            scl.eq(7).rename("unclassified"),
            scl.eq(3).rename("shadow"),
            image.select("MSK_CLDPRB").rename("cloud_prob"),
        ])
        region = stack.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=20, maxPixels=10_000_000).getInfo()
        sampled = stack.select("cloud_prob").sampleRegions(
            collection=points, projection=image.select("SCL").projection(), scale=20, geometries=False
        ).getInfo()["features"]
        probs = [feature["properties"].get("cloud_prob") for feature in sampled]
        probs = [float(value) for value in probs if value is not None]
        rows.append({
            "system_index": index,
            "roi_clear_4_5": region.get("clear45"),
            "roi_unclassified": region.get("unclassified"),
            "roi_shadow": region.get("shadow"),
            "roi_cloud_prob": region.get("cloud_prob"),
            "plot_cloud_prob_max": max(probs) if probs else None,
            "plot_cloud_prob_mean": sum(probs) / len(probs) if probs else None,
            "plot_cloud_prob_above_20": sum(value > 20 for value in probs) / len(probs) if probs else None,
            "n_plots": len(probs),
        })
        print(rows[-1], flush=True)
    dest = root / "outputs" / "satellite" / "s2_clear_sensitivity.json"
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
