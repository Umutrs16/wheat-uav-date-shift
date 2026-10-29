from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from wheat_ppqmtl.io_utils import read_table, write_table
from wheat_ppqmtl.observations import image_name
from wheat_ppqmtl.paths import configure_stdio, configs


def _resize(image: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    from skimage.transform import resize

    resized = resize(
        np.transpose(image, (1, 2, 0)),
        (size, size, image.shape[0]),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    valid_r = resize(valid.astype(float), (size, size), order=0, preserve_range=True, anti_aliasing=False) >= 0.5
    return np.transpose(resized, (2, 0, 1)).astype(np.float32), valid_r


def crop_dataset(frame: pd.DataFrame, raw_root: Path, out_dir: Path, inner_m: float, size: int, valid_min: float, valid_max: float, min_valid: float) -> pd.DataFrame:
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import box, mapping

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    half = inner_m / 2.0
    for day, part in frame.groupby("date_key"):
        day_obj = date.fromisoformat(part["date_iso"].iloc[0])
        matches = list(raw_root.rglob(image_name(day_obj)))
        if not matches:
            print(f"no image for {day}")
            continue
        with rasterio.open(matches[0]) as src:
            if src.crs is None:
                raise RuntimeError(f"{matches[0].name} has no CRS")
            to_raster = Transformer.from_crs("EPSG:32650", src.crs, always_xy=True)
            for rec in part.itertuples(index=False):
                rx, ry = to_raster.transform(float(rec.x), float(rec.y))
                geom = box(rx - half, ry - half, rx + half, ry + half)
                window, _ = mask(src, [mapping(geom)], crop=True, filled=False)
                image = np.asarray(window.filled(np.nan), dtype=float)
                if image.shape[0] < 5:
                    raise RuntimeError(f"{matches[0].name} has {image.shape[0]} bands, expected 5")
                image = image[:5]
                invalid = ~np.isfinite(image) | (image < valid_min) | (image > valid_max)
                image[invalid] = np.nan
                valid = np.isfinite(image).all(axis=0)
                fraction = float(valid.mean()) if valid.size else 0.0
                if fraction < min_valid or image.shape[1] < 2 or image.shape[2] < 2:
                    rows.append({"sample_id": rec.sample_id, "path": "", "valid_fraction": fraction, "kept": False})
                    continue
                # Bilinear resize cannot accept NaN. Invalid pixels are zeroed and tracked in the mask.
                filled = np.where(np.isfinite(image), image, 0.0)
                resized, valid_r = _resize(filled, valid, size)
                dest = out_dir / f"{rec.sample_id}.npz"
                np.savez_compressed(dest, reflectance=resized, valid=valid_r, date_key=rec.date_key, plot_id=rec.plot_id)
                rows.append({"sample_id": rec.sample_id, "path": str(dest), "valid_fraction": fraction, "kept": True})
        print(f"cropped {day}: {sum(r['kept'] for r in rows if True)} running total kept")
    return pd.DataFrame(rows)


def main() -> None:
    configure_stdio()
    argparse.ArgumentParser(description="Cut the pre-registered 2.4 m inner core and resize it to 48x48.").parse_args()
    cfg = configs()
    root = cfg["root"]
    frame = read_table(root / "data" / "interim" / "observations_plot.csv")
    patch_cfg = cfg["data"]["patch"]
    refl = cfg["data"]["reflectance"]
    index = crop_dataset(
        frame,
        root / "data" / "raw",
        root / "data" / "patches",
        float(patch_cfg["inner_m"]),
        int(patch_cfg["size_px"]),
        float(refl["valid_min"]),
        float(refl["valid_max"]),
        float(patch_cfg["min_valid_fraction"]),
    )
    write_table(index, root / "data" / "patches" / "patch_index.csv")
    print(f"kept={int(index['kept'].sum()) if len(index) else 0} / {len(index)}")


if __name__ == "__main__":
    main()
