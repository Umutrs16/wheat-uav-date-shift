# UAV wheat chlorophyll and leaf-area analysis

Data preparation, radiative-transfer pretraining, model training, date-hold-out evaluation, uncertainty calibration, the lagged Sentinel comparison, transfer to a second site (Knezha, Bulgaria), applicability and risk diagnostics, and ground-calibration budgets for UAV winter-wheat chlorophyll and leaf-area retrieval.

This repository contains the download, audit, split, training, evaluation, transfer and diagnostic scripts. It does not contain manuscript files, figure-composition code, Word export, orthomosaics, or trained weights.

Gaocheng field observations and orthomosaics come from Science Data Bank version V4, DOI [10.57760/sciencedb.18749](https://doi.org/10.57760/sciencedb.18749). Knezha field data and Sequoia mosaics come from Zenodo, DOI [10.5281/zenodo.17475742](https://doi.org/10.5281/zenodo.17475742). Neither is stored here.

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

## Date dependence and validation designs

These scripts read the features, frozen splits and saved predictions written above. They do not retrain the primary models or regenerate the splits.

```bash
python scripts/21_variance_and_doy.py
python scripts/22_spad_gdd_sizematch.py
python scripts/23_trend_residual.py
python scripts/37_generalization_matrix.py
```

`21` separates between-date and within-date variance and scores the day-of-year and training-date-mean references. `22` back-calculates SPAD from the published date-specific conversion equations (not the original meter readings), adds a growing-degree-day reference from the Open-Meteo archive, and runs a size-matched hold-out. `23` fits an additive day-of-year trend plus a within-date spectral residual. `37` scores every model under the grouped, spatial-block, forward and leave-one-date-out designs with plot-bootstrap intervals. Run it after `21` and `23`, because it reads their prediction tables.

## Knezha transfer

```bash
python scripts/26_download_bulgaria.py
python scripts/25_bulgaria_inventory.py
python scripts/34_redownload_bad_uav.py
python scripts/29b_repair_uav_zips.py
python scripts/29_extract_bulgaria_uav.py
python scripts/30_uav_transfer.py
```

`26` downloads the Zenodo record one file at a time with `curl`; the mosaics total about 14 GB. `34` replaces mosaic archives that fail a CRC check, and `29b` truncates archives that end in a damaged resume tail. `29` extracts 20 m mean Sequoia reflectance at the ESU centres. Some archives use Deflate64, which needs 7-Zip (`7za` or `7z` on `PATH`). `30` uses that reflectance for zero-shot transfer from Gaocheng, a few-shot intercept, and leave-campaign-out, leave-field-out and leave-year-out tests within Knezha.

## Applicability, risk and ground calibration

```bash
python scripts/35_rtm_diagnostics.py
python scripts/36_sampling_budget.py
python scripts/40_field_map_offset.py
```

`35` computes, for every held-out Gaocheng date and every Knezha campaign, the dissimilarity index and area of applicability, a PROSAIL reconstruction residual at the predicted traits, the ensemble spread, and their equal-weight risk score. `36` repeats a pool/evaluation split 200 times per date or campaign and compares random, representative and highest-risk selection of k labelled plots, for an offset-corrected model and for the label mean alone. `40` applies the dissimilarity threshold and the five-plot offset to the 25 April map written by `12_map_unseen_date.py --date 2024.4.25`.

Scripts `24` (five-band area of applicability and selective prediction), `27` (transfer to the Knezha field spectra), `31` (four PROSAIL environments and a Gaocheng–Sequoia band comparison) and `32` (lookup-table distance risk score) are additional checks. No later script reads their outputs.

Results are written to `outputs/metrics/` and `outputs/predictions/`.

## Recomputing the headline scores

`scripts/20_recompute_reported_metrics.py` recomputes the headline Gaocheng scores from the tables in `outputs/reproducibility/`. Those tables are included. Orthomosaics and trained weights are not. `requirements-lock.txt` records the package versions used to produce the saved scores. Git tag `v1.0.0` is the first snapshot; tag `v1.1.0` adds the date-dependence, Knezha transfer and diagnostic scripts.

## Tests

```bash
python -m unittest tests.test_core
```

## Precision Agriculture revision scripts (v1.2.0)

These scripts consume the released Gaocheng and Knezha table products and the
existing LODO / zero-shot predictions. They do not retrain the five-band ensemble.

```bash
python scripts/41_knezha_hierarchy.py
python scripts/42_cross_sensor_transfer.py
python scripts/43_within_date_metrics.py
python scripts/44_calibration_budget.py
python scripts/45_rtm_robustness.py
python scripts/46_field_maps.py
```

Derived tables are written under `outputs/metrics/` locally and mirrored in
`outputs/reproducibility/` in this release (nested variance, SRF-harmonised
transfer, within-date ranking metrics, ASRC budgets, PROSAIL residual robustness
and field-map summaries). Large GeoTIFF mosaics and map `.npz` rasters are not
stored here.

