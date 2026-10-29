from __future__ import annotations

import sys
from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_yaml(name: str) -> dict:
    path = project_root() / "configs" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def configs() -> dict:
    return {
        "data": load_yaml("data.yaml"),
        "prosail": load_yaml("prosail.yaml"),
        "model": load_yaml("model.yaml"),
        "experiment": load_yaml("experiment.yaml"),
        "root": project_root(),
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
