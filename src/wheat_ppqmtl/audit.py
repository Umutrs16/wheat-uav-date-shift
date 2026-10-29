from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wheat_ppqmtl.io_utils import write_json, write_table
from wheat_ppqmtl.observations import image_name, standardize_observations
from wheat_ppqmtl.paths import configure_stdio, configs


def _find_table(raw_root: Path) -> Path:
    matches = list(raw_root.rglob("observed_data.xlsx"))
    if not matches:
        raise FileNotFoundError("observed_data.xlsx is not in data/raw. Run 00_download.py first.")
    return matches[0]


def _raster_audit(raw_root: Path, observations: pd.DataFrame) -> list[dict]:
    try:
        import rasterio
    except ImportError:
        return []
    from datetime import date

    reports = []
    for day, part in observations.groupby("date_key"):
        day_obj = date.fromisoformat(part["date_iso"].iloc[0])
        matches = list(raw_root.rglob(image_name(day_obj)))
        if not matches:
            reports.append({"date_key": day, "present": False})
            continue
        path = matches[0]
        with rasterio.open(path) as src:
            sample = src.read(out_shape=(src.count, 64, 64), masked=True)
            values = np.asarray(sample.filled(np.nan), dtype=float)
            finite = values[np.isfinite(values)]
            reports.append({
                "date_key": day,
                "present": True,
                "path": str(path),
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtypes": list(src.dtypes),
                "nodata": src.nodata,
                "pixel_size": [abs(src.transform.a), abs(src.transform.e)],
                "bounds": list(src.bounds),
                "sample_min": float(np.min(finite)) if finite.size else None,
                "sample_p50": float(np.median(finite)) if finite.size else None,
                "sample_p99": float(np.percentile(finite, 99)) if finite.size else None,
                "sample_max": float(np.max(finite)) if finite.size else None,
            })
    return reports


def build_audit(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    table_path = _find_table(raw_root)
    raw = pd.read_excel(table_path)
    observations = standardize_observations(raw)
    data_cfg = configs()["data"]
    by_date = []
    for day, part in observations.groupby("date_key"):
        stage = data_cfg["stages"].get(day, {})
        by_date.append({
            "date_key": day,
            "date_iso": part["date_iso"].iloc[0],
            "stage_en": stage.get("stage_en"),
            "feekes": stage.get("feekes"),
            "data_paper_stage_zh": stage.get("data_paper_stage_zh"),
            "n": int(len(part)),
            "n_lcc": int(part["lcc"].notna().sum()),
            "n_lai": int(part["lai"].notna().sum()),
            "n_qc1": int(np.isclose(part["lai_qc"], 1).sum()),
            "n_qc2": int(np.isclose(part["lai_qc"], 2).sum()),
            "lcc_min": float(part["lcc"].min()),
            "lcc_max": float(part["lcc"].max()),
            "lcc_mean": float(part["lcc"].mean()),
            "lai_min": float(part["lai"].min()) if part["lai"].notna().any() else None,
            "lai_max": float(part["lai"].max()) if part["lai"].notna().any() else None,
            "lai_mean": float(part["lai"].mean()) if part["lai"].notna().any() else None,
        })
    spectral = observations[["B", "G", "R", "RE", "NIR"]]
    report = {
        "source_table": str(table_path),
        "n_rows": int(len(observations)),
        "n_plots": int(observations["plot_id"].nunique()),
        "n_dates": int(observations["date_key"].nunique()),
        "dates": sorted(observations["date_key"].unique(), key=lambda s: tuple(int(p) for p in s.split("."))),
        "duplicate_plot_dates": int(observations.duplicated(["plot_id", "date_key"]).sum()),
        "lai_missing": int(observations["lai"].isna().sum()),
        "lai_qc_missing": int(observations["lai_qc"].isna().sum()),
        "reflectance_min": float(spectral.min().min()),
        "reflectance_max": float(spectral.max().max()),
        "nir_p99": float(observations["NIR"].quantile(0.99)),
        "likely_zero_to_one_scale": bool(observations["NIR"].quantile(0.99) < data_cfg["reflectance"]["scale_warn_above"]),
        "published_plot_id_present": True,
        "by_date": by_date,
        "stage_label_note": (
            "On 2024-05-18 the data paper's Chinese stage label is heading, while the Feekes "
            "code printed in that paper is 11.1-11.2 (ripening). Both are retained; the Feekes "
            "code is not silently relabelled."
        ),
        "v4_has_six_dates": int(observations["date_key"].nunique()) > 5,
        "rasters": _raster_audit(raw_root, observations),
    }
    return observations, report


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Audit the released wheat table and any local GeoTIFFs.").parse_args()
    root = configs()["root"]
    observations, report = build_audit(root / "data" / "raw")
    interim = root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    write_table(observations, interim / "observations.csv")
    write_json(interim / "audit_report.json", report)
    write_table(pd.DataFrame(report["by_date"]), interim / "table1_by_date.csv")
    print(
        f"rows={report['n_rows']} plots={report['n_plots']} dates={report['dates']} "
        f"lai_missing={report['lai_missing']} scale_0_1={report['likely_zero_to_one_scale']}"
    )


if __name__ == "__main__":
    main()
