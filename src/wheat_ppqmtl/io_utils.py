from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv") if path.suffix == ".parquet" else path
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(path)
