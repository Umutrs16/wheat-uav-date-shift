"""Replace mosaic zips whose compressed bands failed a CRC check.

Files are written to a temporary name and moved only after every band reads
without a CRC error. Downloads do not resume an existing partial file.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "external" / "bulgaria" / "uav"
BAD = [
    "S16-17_U1-3_2017-03-21.zip",
    "S16-17_U1-3_2017-05-18.zip",
    "S16-17_U2_2017-03-21.zip",
    "S16-17_U4-5-6_2017-03-22.zip",
    "S16-17_U4-5-6_2017-04-26.zip",
    "S16-17_U4-5-6_2017-05-15.zip",
    "S17-18_U1_2018-04-03.zip",
    "S17-18_U4-5-6_2017-11-10.zip",
    "S16-17_U1-3_2016-11-07.zip",
    "S16-17_U4-5-6_2016-11-10.zip",
]


def _crc_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.compress_type == 9:
                    continue
                with archive.open(info) as member:
                    while member.read(1 << 20):
                        pass
        return True
    except (zipfile.BadZipFile, zlib.error, OSError, EOFError):
        return False


def main() -> None:
    for name in BAD:
        dest = ROOT / name
        temp = dest.with_suffix(".zip.fresh")
        if dest.exists() and _crc_ok(dest):
            print(f"already valid {name}", flush=True)
            continue
        if temp.exists():
            temp.unlink()
        url = f"https://zenodo.org/api/records/17475742/files/{name}/content"
        print(f"download {name}", flush=True)
        completed = subprocess.run([
            "curl", "-L", "--fail",
            "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "-H", "Referer: https://zenodo.org/records/17475742",
            "-o", str(temp), url,
        ])
        if completed.returncode != 0 or not _crc_ok(temp):
            print(f"rejected {name}", flush=True)
            continue
        if dest.exists():
            dest.unlink()
        shutil.move(temp, dest)
        print(f"replaced {name} {dest.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
