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

