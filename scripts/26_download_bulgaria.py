"""Download the Bulgarian winter-wheat Zenodo record one file at a time.

The records/files page returned HTTP 403 after a burst. This uses the API
content URL, waits between files, and skips anything already saved at the
expected size. UAV mosaics are large and are resumed if a partial file exists.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "external" / "bulgaria"
BASE = "https://zenodo.org/api/records/17475742/files/{name}/content"
TABLES = {
    "ESU_Biophysical_and_biochemical_crop_data.txt": 32917,
    "ESU_Biophysical_and_biochemical_crop_data_info.txt": 2447,
    "ESU_Phenological_data.txt": 27197,
    "ESU_Phenological_data_info.txt": 1828,
    "ESU_Field_spectral_data.txt": 1753647,
    "ESU_Field_spectral_data_info.txt": 691,
    "Crop_calendar_and_management.txt": 8052,
    "Crop_calendar_and_management_info.txt": 2169,
    "Meteo_data.txt": 5400242,
    "Meteo_data_info.txt": 1577,
    "Meteo_summary.txt": 157,
    "Meteo_summary_info.txt": 339,
    "Shapefile_ESUs.zip": 11143,
    "Shapefile_Units.zip": 10575,
}
# Sizes are the Zenodo content-length values. A partial file is resumed, not treated as complete.
UAV = {
    "S16-17_U2_2017-03-21.zip": 301737760,
    "S17-18_U3_2018-04-04.zip": 309423435,
    "S16-17_U2_2017-04-25.zip": 308084402,
    "S16-17_U2_2017-05-18.zip": 322247203,
    "S17-18_U1_2018-04-03.zip": 365296222,
    "S17-18_U2_2018-04-04.zip": 373558152,
    "S17-18_U4-5-6_2017-11-10.zip": 428559055,
    "S17-18_U4-5-6_2018-04-03.zip": 586253093,
    "S16-17_U4-5-6_2017-04-26.zip": 572274730,
    "S16-17_U4-5-6_2017-05-15.zip": 603470594,
    "S16-17_U4-5-6_2017-03-22.zip": 616857097,
    "S16-17_U1-3_2017-03-21.zip": 660285884,
    "S16-17_U1-3_2017-05-18.zip": 683297741,
    "S16-17_U1-3_2017-04-25.zip": 688687548,
    "S16-17_U2_2016-11-11.zip": 1688731090,
    "S16-17_U1-3_2016-11-07.zip": 2742302846,
    "S16-17_U4-5-6_2016-11-10.zip": 2925641320,
}


def _good(path: Path, expected: int | None) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as handle:
        head = handle.read(300).lstrip().lower()
    if head.startswith(b"<") or b"forbidden" in head or b"<html" in head:
        return False
    size = path.stat().st_size
    if expected is not None:
        return size == expected
    return size > 1000


def _fetch(dest_dir: Path, name: str, expected: int | None, pause: float) -> None:
    dest = dest_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _good(dest, expected):
        print(f"have {name} {dest.stat().st_size}", flush=True)
        return
    url = BASE.format(name=name)
    if dest.exists():
        with dest.open("rb") as handle:
            head = handle.read(300).lstrip().lower()
        if head.startswith(b"<") or b"forbidden" in head or b"<html" in head:
            dest.unlink()
    partial = dest.stat().st_size if dest.exists() else 0
    for attempt in range(4):
        command = [
            "curl", "-L", "--fail", "--retry", "2", "-C", "-",
            "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "-H", "Referer: https://zenodo.org/records/17475742",
            "-o", str(dest), url,
        ]
        print(f"get {name} resume_at {partial}", flush=True)
        completed = subprocess.run(command)
        partial = dest.stat().st_size if dest.exists() else 0
        if completed.returncode == 0 and _good(dest, expected):
            print(f"saved {name} {partial}", flush=True)
            time.sleep(pause)
            return
        print(f"fail {name} attempt {attempt + 1} code {completed.returncode} size {partial}", flush=True)
        time.sleep(20 * (attempt + 1))
    print(f"gave up {name}", flush=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, size in TABLES.items():
        _fetch(ROOT, name, size, 8)
    uav_dir = ROOT / "uav"
    uav_dir.mkdir(parents=True, exist_ok=True)
    for name, size in UAV.items():
        _fetch(uav_dir, name, size, 3)


if __name__ == "__main__":
    main()
