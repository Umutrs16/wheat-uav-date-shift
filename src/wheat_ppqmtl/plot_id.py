from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from wheat_ppqmtl.io_utils import read_table, write_json, write_table
from wheat_ppqmtl.paths import configure_stdio, configs


def project_coordinates(frame: pd.DataFrame, source_crs: str, target_crs: str) -> pd.DataFrame:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(frame["lon"].to_numpy(float), frame["lat"].to_numpy(float))
    out = frame.copy()
    out["x"] = x
    out["y"] = y
    return out


def cluster_labels(x: np.ndarray, y: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    return DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(np.column_stack([x, y]))


def plot_radius_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plot_id, part in frame.groupby("plot_id"):
        coords = part[["x", "y"]].to_numpy(float)
        center = np.median(coords, axis=0)
        radius = np.sqrt(((coords - center) ** 2).sum(axis=1))
        rows.append({
            "plot_id": int(plot_id),
            "n": int(len(part)),
            "n_dates": int(part["date_key"].nunique()),
            "max_radius_m": float(radius.max()),
            "median_x": float(center[0]),
            "median_y": float(center[1]),
        })
    return pd.DataFrame(rows)


def _nearest_other_date_m(frame: pd.DataFrame) -> np.ndarray:
    xs = frame["x"].to_numpy(float)
    ys = frame["y"].to_numpy(float)
    dates = frame["date_key"].to_numpy()
    nearest = np.empty(len(frame), dtype=float)
    for index in range(len(frame)):
        distance = (xs - xs[index]) ** 2 + (ys - ys[index]) ** 2
        distance[dates == dates[index]] = np.inf
        nearest[index] = float(np.sqrt(distance.min()))
    return nearest


def assign_block_id(frame: pd.DataFrame, block_m: float) -> pd.DataFrame:
    """40 m cell of each observation. Published PLOT_ID is not a stable location."""
    out = frame.copy()
    out["block_x"] = np.floor(out["x"].to_numpy(float) / block_m).astype(int)
    out["block_y"] = np.floor(out["y"].to_numpy(float) / block_m).astype(int)
    out["block_id"] = out["block_x"].astype(str) + "_" + out["block_y"].astype(str)
    return out


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Project plots, audit persistent plot ids, assign spatial blocks.").parse_args()
    cfg = configs()
    root = cfg["root"]
    data_cfg = cfg["data"]
    frame = read_table(root / "data" / "interim" / "observations.csv")
    frame = project_coordinates(frame, data_cfg["crs"]["geographic"], data_cfg["crs"]["projected"])
    eps = float(data_cfg["plot_id"]["eps_m"])
    labels = cluster_labels(
        frame["x"].to_numpy(float),
        frame["y"].to_numpy(float),
        eps,
        int(data_cfg["plot_id"]["min_samples"]),
    )
    frame["cluster_id"] = labels.astype(int)
    ari = float(adjusted_rand_score(frame["plot_id"].astype(int), frame["cluster_id"].astype(int)))
    radii = plot_radius_table(frame)
    frame = assign_block_id(frame, float(data_cfg["spatial_cv"]["block_m"]))
    interim = root / "data" / "interim"
    write_table(frame, interim / "observations_plot.csv")
    write_table(radii, interim / "plot_radius.csv")
    sensitive = {}
    for extra in data_cfg["plot_id"]["sensitivity_m"]:
        extra_labels = cluster_labels(frame["x"].to_numpy(float), frame["y"].to_numpy(float), float(extra), 1)
        sensitive[str(extra)] = float(adjusted_rand_score(frame["plot_id"].astype(int), extra_labels))
    nearest = _nearest_other_date_m(frame)
    report = {
        "published_plot_id_used_for_grouped_random": True,
        "published_plot_id_is_stable_location": False,
        "cluster_eps_m": eps,
        "adjusted_rand_vs_published_plot_id": ari,
        "sensitivity_ari": sensitive,
        "plots_with_radius_above_1m": int((radii["max_radius_m"] > 1.0).sum()),
        "plots_with_radius_above_eps": int((radii["max_radius_m"] > eps).sum()),
        "median_of_max_radius_m": float(radii["max_radius_m"].median()),
        "max_within_plot_radius_m": float(radii["max_radius_m"].max()),
        "n_points_other_date_neighbor_within_eps": int((nearest < eps).sum()),
        "median_other_date_neighbor_m": float(np.median(nearest)),
        "n_spatial_blocks": int(frame["block_id"].nunique()),
        "block_m": float(data_cfg["spatial_cv"]["block_m"]),
        "block_rule": "40 m cell of each observation's own projected coordinate",
        "note": (
            "Grouped-random CV keeps the publisher PLOT_ID together. "
            "That identifier is not a revisited location. "
            "Spatial CV uses each point's own 40 m cell, so one PLOT_ID can occupy several blocks."
        ),
    }
    write_json(interim / "plot_id_audit.json", report)
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            frame,
            geometry=[Point(xy) for xy in zip(frame["x"], frame["y"])],
            crs=data_cfg["crs"]["projected"],
        )
        gdf.to_file(interim / "points_with_plotid.gpkg", driver="GPKG")
    except Exception as exc:
        (interim / "geopackage_error.txt").write_text(str(exc), encoding="utf-8")
    print(
        f"ARI={ari:.3f} blocks={report['n_spatial_blocks']} "
        f"plots_radius>1m={report['plots_with_radius_above_1m']}"
    )


if __name__ == "__main__":
    main()
