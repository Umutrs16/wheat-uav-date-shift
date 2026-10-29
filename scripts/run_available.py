"""Run every stage that does not need the orthomosaics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wheat_ppqmtl import audit, baselines, calibrate, download, evaluate, features, make_splits, mapping, plot_id, pretrain, prosail_sim, train_ppq
from wheat_ppqmtl.paths import configure_stdio


def _call(func, argv: list[str]) -> None:
    print("\n===", func.__module__, " ".join(argv), "===")
    sys.argv = [func.__module__, *argv]
    func()


def main() -> None:
    configure_stdio()
    _call(download.main, ["--skip-images"])
    _call(audit.main, [])
    _call(plot_id.main, [])
    _call(features.main, [])
    _call(make_splits.main, [])
    _call(prosail_sim.main, [])
    _call(pretrain.main, [])
    _call(baselines.main, [])
    _call(train_ppq.main, [])
    _call(evaluate.main, [])
    _call(calibrate.main, [])
    try:
        _call(mapping.main, [])
    except Exception as exc:
        print(f"mapping skipped: {exc}")
    print("\nOrthomosaic stages not run: 03_crop_patches.py and the CNN variant. Download images with scripts/00_download.py, then rerun crop, features, and 09_train_ppqmtl.py.")


if __name__ == "__main__":
    main()
