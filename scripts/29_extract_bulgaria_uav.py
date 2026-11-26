"""Extract 20 m mean Sequoia reflectance at Bulgarian ESU centers.

Each mosaic zip holds one GeoTIFF per band. A plot is kept only when its
center falls inside a mosaic for the same unit and campaign. Overlapping
mosaics keep the flight whose unit list is the smaller one. Reflectance is
not rescaled here; the saved table records the raw mosaic values.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "external" / "bulgaria"
UAV = TABLE / "uav"
WORK = TABLE / "_uav_extract_tmp"
DEST = ROOT / "data" / "external" / "bulgaria" / "uav_esu_reflectance.csv"

BAND_KEYS = (
    ("green", "G"),
    ("rededge", "RE"),
    ("rede", "RE"),
    ("red", "R"),
    ("nir", "NIR"),
)


def _units(token: str) -> list[int]:
    body = token[1:]
    parts = [int(part) for part in body.split("-")]
    if len(parts) == 2 and parts[1] - parts[0] > 1:
        return list(range(parts[0], parts[1] + 1))
    return parts


def _zip_meta(path: Path) -> dict:
    stem = path.stem
    season, unit_token, date = stem.split("_")
    return {
        "path": path,
        "season": season,
        "units": _units(unit_token),
        "image_date": pd.Timestamp(date),
        "span": len(_units(unit_token)),
    }


def _band_name(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    if "rededge" in stem or "red_edge" in stem:
        return "RE"
    if stem.endswith("_nir") or stem.endswith("-nir"):
        return "NIR"
    if stem.endswith("_green") or stem.endswith("-green"):
        return "G"
    if stem.endswith("_red") or stem.endswith("-red"):
        return "R"
    return None


def _window_mean(dataset, x: float, y: float) -> float | None:
    if not (dataset.bounds.left <= x <= dataset.bounds.right and dataset.bounds.bottom <= y <= dataset.bounds.top):
        return None
    if dataset.crs is not None and dataset.crs.is_geographic:
        half = 10.0 / (111_320.0 * max(0.2, np.cos(np.radians(y))))
    else:
        half = 10.0
    window = from_bounds(x - half, y - half, x + half, y + half, dataset.transform)
    window = window.round_offsets().round_lengths()
    if window.width < 1 or window.height < 1:
        return None
    values = dataset.read(1, window=window, masked=True)
    valid = np.ma.count(values)
    if valid < 0.5 * values.size:
        return None
    return float(values.mean())


def _seven_zip() -> str | None:
    bundled = ROOT / "tools" / "7zip" / "extra" / "x64" / "7za.exe"
    if bundled.exists():
        return str(bundled)
    return shutil.which("7za") or shutil.which("7z")


def _extract_with_7zip(path: Path) -> None:
    seven = _seven_zip()
    if seven is None:
        raise NotImplementedError("Deflate64 requires 7-Zip (7za or 7z on PATH)")
    completed = subprocess.run(
        [seven, "x", "-y", f"-o{WORK}", str(path)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise OSError(detail[-1] if detail else "7za failed")


def _extract_zip(meta: dict, plots: pd.DataFrame) -> list[dict]:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    rows = []
    try:
        try:
            with zipfile.ZipFile(meta["path"]) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".tif")]
                archive.extractall(WORK, members)
        except NotImplementedError:
            _extract_with_7zip(meta["path"])
    except (zipfile.BadZipFile, NotImplementedError, OSError, EOFError, zlib.error) as exc:
        print(f"skip {meta['path'].name}: {type(exc).__name__}: {exc}", flush=True)
        shutil.rmtree(WORK, ignore_errors=True)
        return rows
    members = [str(path.relative_to(WORK)) for path in WORK.rglob("*.tif")]
    paths = {}
    for member in members:
        band = _band_name(member)
        if band:
            paths[band] = WORK / member
    missing = [band for band in ("G", "R", "RE", "NIR") if band not in paths]
    if missing:
        print(f"missing bands {meta['path'].name}: {missing}", flush=True)
        shutil.rmtree(WORK, ignore_errors=True)
        return rows
    datasets = {band: rasterio.open(paths[band]) for band in paths}
    try:
        reference = datasets["G"]
        print(
            f"{meta['path'].name} crs={reference.crs} size={reference.width}x{reference.height} "
            f"res={reference.res} bounds={tuple(round(v, 1) for v in reference.bounds)}",
            flush=True,
        )
        transformer = None
        if reference.crs is not None and reference.crs.to_epsg() != 32634:
            transformer = Transformer.from_crs(32634, reference.crs, always_xy=True)
        for plot in plots.itertuples(index=False):
            x, y = float(plot.Point_X_UTM34), float(plot.Point_Y_UTM34)
            if transformer is not None:
                x, y = transformer.transform(x, y)
            values = {band: _window_mean(datasets[band], x, y) for band in ("G", "R", "RE", "NIR")}
            if any(value is None for value in values.values()):
                continue
            rows.append({
                "ESU_ID": plot.ESU_ID,
                "Campaign": plot.Campaign,
                "unit": int(plot.unit),
                "field_id": plot.field_id,
                "image_zip": meta["path"].name,
                "image_date": meta["image_date"].date().isoformat(),
                "lp80_date": plot.LP80_date,
                "G": values["G"],
                "R": values["R"],
                "RE": values["RE"],
                "NIR": values["NIR"],
                "lcc": plot.lcc,
                "lai": plot.lai,
                "crs": str(reference.crs),
                "pixel_m": float(abs(reference.res[0])) if reference.crs is not None and not reference.crs.is_geographic else np.nan,
            })
    finally:
        for dataset in datasets.values():
            dataset.close()
        shutil.rmtree(WORK, ignore_errors=True)
    print(f"  matched {len(rows)}", flush=True)
    return rows


def main() -> None:
    bio = pd.read_csv(TABLE / "ESU_Biophysical_and_biochemical_crop_data.txt", sep="\t")
    parsed = bio["ESU_ID"].str.extract(r"([a-z]+)(\d{4})_(\d+)_(\d+)")
    bio["unit"] = pd.to_numeric(parsed[2], errors="coerce")
    bio["season"] = np.where(bio["Campaign"].isin(["November_2017", "April_2018"]), "S17-18", "S16-17")
    bio["field_id"] = bio["season"] + "_U" + bio["unit"].astype("Int64").astype(str)
    bio["lcc"] = pd.to_numeric(bio["CC_mean"], errors="coerce")
    bio["lai"] = pd.to_numeric(bio["LAI_mean"], errors="coerce")
    collected = []
    for path in sorted(UAV.glob("*.zip")):
        meta = _zip_meta(path)
        month = meta["image_date"].month
        year = meta["image_date"].year
        plots = bio[bio["unit"].isin(meta["units"])].copy()
        plots = plots[pd.to_datetime(plots["LP80_date"], errors="coerce").dt.year.eq(year)]
        plots = plots[pd.to_datetime(plots["LP80_date"], errors="coerce").dt.month.eq(month)]
        print(f"open {path.name} candidate_plots={len(plots)}", flush=True)
        collected.extend(_extract_zip(meta, plots))
    frame = pd.DataFrame(collected)
    if frame.empty:
        raise SystemExit("no ESU fell inside a mosaic")
    frame["span"] = frame["image_zip"].map(lambda name: len(_units(name.split("_")[1])))
    frame = frame.sort_values("span").drop_duplicates(["ESU_ID", "Campaign"], keep="first")
    frame = frame.drop(columns=["span"])
    DEST.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DEST, index=False)
    print(f"wrote {DEST} n={len(frame)}", flush=True)
    print(frame.groupby("Campaign")[["G", "R", "RE", "NIR", "lcc"]].agg(["count", "median"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
