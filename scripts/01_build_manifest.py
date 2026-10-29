import _path  # noqa: F401
from pathlib import Path

from wheat_ppqmtl.paths import configs


def main() -> None:
    root = configs()["root"]
    raw = root / "data" / "raw"
    rows = []
    for path in raw.rglob("*"):
        if path.is_file():
            rows.append(f"{path.relative_to(raw)}\t{path.stat().st_size}")
    dest = root / "data" / "interim" / "local_manifest.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(sorted(rows)), encoding="utf-8")
    print(f"{len(rows)} files -> {dest}")


if __name__ == "__main__":
    main()
