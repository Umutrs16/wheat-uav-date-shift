"""Truncate mosaic zips that contain a valid archive followed by a bad resume tail.

The local headers are walked from the start. The file is truncated at the first
end-of-central-directory record that follows those headers. A member is then read
before the truncated file replaces the original.
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "external" / "bulgaria" / "uav"


def _archive_end(path: Path) -> int:
    with path.open("rb") as handle:
        pos = 0
        while True:
            handle.seek(pos)
            sig = handle.read(4)
            if sig != b"PK\x03\x04":
                break
            header = handle.read(26)
            if len(header) < 26:
                raise RuntimeError(f"{path.name}: truncated local header")
            _, flag, _, _, _, _, comp, _, nlen, elen = struct.unpack("<HHHHHIIIHH", header)
            if flag & 8:
                raise RuntimeError(f"{path.name}: data descriptor is not handled")
            handle.seek(nlen + elen, 1)
            pos = handle.tell() + comp
        if sig != b"PK\x01\x02":
            raise RuntimeError(f"{path.name}: expected central directory at {pos}, found {sig!r}")
        while True:
            handle.seek(pos)
            sig = handle.read(4)
            if sig == b"PK\x05\x06":
                rest = handle.read(18)
                comment_len = struct.unpack("<H", rest[16:18])[0]
                return pos + 22 + comment_len
            if sig != b"PK\x01\x02":
                raise RuntimeError(f"{path.name}: unexpected signature {sig!r} at {pos}")
            rest = handle.read(42)
            nlen, elen, clen = struct.unpack("<HHH", rest[24:30])
            pos += 46 + nlen + elen + clen
    raise RuntimeError(path.name)


class _Prefix:
    def __init__(self, raw, limit: int):
        self.raw = raw
        self.limit = limit
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.limit + offset
        self.pos = max(0, min(self.pos, self.limit))
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.limit:
            return b""
        self.raw.seek(self.pos)
        if size < 0 or self.pos + size > self.limit:
            size = self.limit - self.pos
        data = self.raw.read(size)
        self.pos += len(data)
        return data


def main() -> None:
    for path in sorted(ROOT.glob("*.zip")):
        end = _archive_end(path)
        size = path.stat().st_size
        with path.open("rb") as raw:
            with zipfile.ZipFile(_Prefix(raw, end)) as archive:
                names = archive.namelist()
                if len(names) != 4:
                    raise RuntimeError(f"{path.name}: expected 4 bands, found {names}")
        if end < size:
            with path.open("r+b") as handle:
                handle.truncate(end)
        print(f"repaired {path.name} {size} -> {end}", flush=True)


if __name__ == "__main__":
    main()
