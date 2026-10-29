# UAV wheat chlorophyll and leaf-area analysis

Data preparation, radiative-transfer pretraining, model training, date-hold-out evaluation, uncertainty calibration, and the lagged Sentinel comparison for one Gaocheng winter-wheat field.

This repository contains the download, audit, split, training, evaluation, and Sentinel sampling scripts. It does not contain manuscript files, figure-composition code, Word export, orthomosaics, or trained weights.

Field observations and orthomosaics come from Science Data Bank version V4, DOI [10.57760/sciencedb.18749](https://doi.org/10.57760/sciencedb.18749). They are not stored here.

## Setup

```bash
pip install -r requirements.txt
```

Run the scripts from the repository root with `src` on `PYTHONPATH`.

## Table-only pipeline

This path does not download the orthomosaics.

```bash
python scripts/run_available.py
```

The same stages can be run separately:

```bash
python scripts/00_download.py --skip-images
python scripts/00_audit_data.py
python scripts/02_assign_plot_id.py
python scripts/04_extract_features.py
python scripts/07_make_splits.py
python scripts/05_simulate_prosail.py
python scripts/06_pretrain_physics.py
python scripts/08_train_baselines.py
python scripts/09_train_ppqmtl.py
python scripts/10_evaluate.py
python scripts/11_calibrate_uncertainty.py
```

`configs/prosail.yaml` sets `draft_samples` to 20000 for that run. `paper_samples` records the pre-specified library size of 100000 and is not selected from a test score.

## Orthomosaics

```bash
python scripts/00_download.py
python scripts/03_crop_patches.py
python scripts/04_extract_features.py
python scripts/12_map_unseen_date.py --date 2024.4.25
```

## Sentinel comparison

Scripts `16`–`19` need the Earth Engine Python API and a cloud project. Set `EE_PROJECT` before running them. `scripts/gee_wheat_s1_s2_audit.js` is the same inventory for the Code Editor. The project is selected there, not stored in this repository.

`scripts/20_recompute_reported_metrics.py` recomputes the headline scores from the tables in `outputs/reproducibility/`. Those tables are included. Orthomosaics and trained weights are not. `requirements-lock.txt` records the package versions used to produce the saved scores. The submission snapshot is git tag `v1.0.0`.

## Tests

```bash
python -m unittest tests.test_core
```
