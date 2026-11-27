These tables are the saved test predictions used to recompute the headline scores.
Sample identifiers come from the public field table. The tables do not contain names or contact details.
Orthomosaics and trained weights are not included.

From the repository root:

python scripts/20_recompute_reported_metrics.py

v1.2.0 additions
----------------
knezha_hierarchy.json          Nested campaign/field/ESU variance shares
cross_sensor_transfer.json     T1–T5 / reverse transfer scores and CIs
cross_sensor_predictions.csv   Per-ESU zero-shot and reverse predictions
knezha_harmonized_bands.csv    Sequoia bands mapped toward Gaocheng SRF
within_date_metrics.json       Within-date R2, Spearman, SD ratio, low/high recall
calibration_budget.json        ASRC / offset / label-mean budgets (200 repeats)
calibration_budget_groups.csv  Per-group budget rows
rtm_robustness.json            PROSAIL residual convergence and Cab profiles
field_maps.json                Crop-masked map summary statistics
figure_numbers.json            Numbers used by the paper figures

v1.3.0 additions
----------------
unified_results.json           Cluster-bootstrap intervals and paired differences,
                               Knezha between- and within-field agreement, both variance
                               estimators, the full transfer matrix, calibration
                               summaries, k* and the uncalibrated-map screening check
calibration_budget.json        Updated: six estimators, random and spatial halves, six
                               selection strategies and two release rules (200 repeats)
calibration_budget_groups.csv  Updated per-group budget rows (mean, 2.5-97.5% range)
cross_sensor_transfer.json     Updated scenario note (T2m, T4m, R1-R4m)
field_maps.json                Updated map summaries (Knezha 20 m ESU support)
figure_numbers.json            Numbers used by the figures
