from __future__ import annotations

import sys
import unittest
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wheat_ppqmtl.conformal import conformal_quantile
from wheat_ppqmtl.indices import safe_div, vegetation_indices
from wheat_ppqmtl.losses import date_level_lai_loss, hetero_nll, multitask_loss
from wheat_ppqmtl.metrics import optimism_gap, regression_metrics
from wheat_ppqmtl.observations import date_key, parse_survey_date, standardize_observations
from wheat_ppqmtl.solar import solar_zenith_deg
from wheat_ppqmtl.splits import (
    assign_forward_folds,
    assign_grouped_folds,
    assign_lodo_folds,
    assign_spatial_block_folds,
    leakage_report,
)


class CoreTests(unittest.TestCase):
    def test_indices_match_definitions(self):
        frame = pd.DataFrame({"B": [0.04], "G": [0.06], "R": [0.05], "RE": [0.16], "NIR": [0.40]})
        out = vegetation_indices(frame)
        self.assertAlmostEqual(out.loc[0, "ndvi"], (0.40 - 0.05) / (0.40 + 0.05))
        self.assertAlmostEqual(out.loc[0, "cire"], 0.40 / 0.16 - 1)
        self.assertTrue(np.isnan(safe_div(np.array([1.0]), np.array([0.0]))[0]))

    def test_hetero_nll_at_zero_variance(self):
        y = torch.tensor([1.0])
        mu = torch.tensor([0.0])
        logvar = torch.tensor([0.0])
        self.assertAlmostEqual(float(hetero_nll(y, mu, logvar)), 0.5)

    def test_multitask_masks_missing_lai(self):
        pred = {
            "mu_lcc": torch.tensor([1.0]),
            "logvar_lcc": torch.tensor([0.0]),
            "mu_lai": torch.tensor([5.0]),
            "logvar_lai": torch.tensor([0.0]),
            "mu_ccc": torch.tensor([0.0]),
            "logvar_ccc": torch.tensor([0.0]),
        }
        batch = {
            "y_lcc": torch.tensor([0.0]),
            "y_lai": torch.tensor([0.0]),
            "y_ccc": torch.tensor([0.0]),
            "mask_lcc": torch.tensor([1.0]),
            "mask_lai": torch.tensor([0.0]),
            "qc_weight": torch.tensor([0.4]),
            "ccc_factor": torch.tensor(0.01),
            "lcc_center": torch.tensor(0.0),
            "lcc_scale": torch.tensor(1.0),
            "lai_center": torch.tensor(0.0),
            "lai_scale": torch.tensor(1.0),
            "ccc_center": torch.tensor(0.0),
            "ccc_scale": torch.tensor(1.0),
        }
        loss, parts = multitask_loss(pred, batch, {"lcc": 1, "lai": 1, "ccc": 0.5, "consistency": 0.2})
        self.assertAlmostEqual(parts["lai"], 0.0)
        self.assertGreater(float(loss), 0.0)

    def test_date_level_lai_loss_uses_the_mean(self):
        mu = torch.tensor([0.0, 2.0])
        loss = date_level_lai_loss(mu, torch.tensor(0.0), torch.tensor(1.0), 2.5)
        self.assertAlmostEqual(float(loss), (1.0 - 2.5) ** 2)
        scaled = date_level_lai_loss(mu, torch.tensor(0.0), torch.tensor(2.0), 2.5)
        self.assertAlmostEqual(float(scaled), ((2.0 - 2.5) / 2.0) ** 2)

    def test_date_key_matches_image_names(self):
        day = parse_survey_date("2024.3.31")
        self.assertEqual(day, date(2024, 3, 31))
        self.assertEqual(date_key(day), "2024.3.31")

    def test_standardize_real_column_names(self):
        raw = pd.DataFrame({
            "PLOT_ID": [1],
            "date": ["2024.4.12"],
            "Lat (°)": [37.88],
            "Lon (°)": [114.85],
            "B": [0.02], "G": [0.04], "R": [0.03], "RE": [0.1], "NIR": [0.4],
            "LCC (μg/cm²)": [50.0],
            "LAI": [np.nan],
            "LAI_QC": [np.nan],
        })
        out = standardize_observations(raw)
        self.assertEqual(out.loc[0, "sample_id"], "1_2024.4.12")
        self.assertTrue(np.isnan(out.loc[0, "lai"]))

    def test_splits_do_not_leak(self):
        rows = []
        for plot_id in range(1, 21):
            for day in ("2024.3.31", "2024.4.12", "2024.4.25"):
                rows.append({
                    "sample_id": f"{plot_id}_{day}",
                    "plot_id": plot_id,
                    "date_key": day,
                    "block_id": f"{plot_id // 5}",
                })
        frame = pd.DataFrame(rows)
        grouped = assign_grouped_folds(frame, "plot_id", "grouped_random", 5, 1)
        spatial = assign_grouped_folds(frame, "block_id", "spatial_block", 4, 2)
        lodo = assign_lodo_folds(frame, 3)
        problems = leakage_report(pd.concat([grouped, spatial, lodo], ignore_index=True))
        self.assertEqual(problems, [])

    def test_forward_uses_only_earlier_dates(self):
        rows = []
        for plot_id in range(1, 12):
            for day in ("2024.3.31", "2024.4.12", "2024.4.25"):
                rows.append({
                    "sample_id": f"{plot_id}_{day}",
                    "plot_id": plot_id,
                    "date_key": day,
                })
        splits = assign_forward_folds(pd.DataFrame(rows), seed=1)
        first = splits[splits["outer_fold"] == 0]
        self.assertEqual(set(first.loc[first["role"] == "test", "date_key"]), {"2024.4.12"})
        used = first["role"].isin(["train", "tune", "calibration"])
        self.assertEqual(set(first.loc[used, "date_key"]), {"2024.3.31"})
        self.assertEqual(set(first.loc[first["role"] == "excluded", "date_key"]), {"2024.4.25"})
        self.assertEqual(leakage_report(splits), [])

    def test_spatial_buffer_removes_adjacent_training_points(self):
        rows = [
            {"sample_id": "a", "plot_id": 1, "date_key": "2024.4.12", "x": 39.0, "y": 0.0},
            {"sample_id": "near", "plot_id": 2, "date_key": "2024.4.12", "x": 41.0, "y": 0.0},
        ]
        for index, easting in enumerate((150, 250, 350, 450, 550, 650), start=3):
            rows.append({"sample_id": f"p{index}", "plot_id": index, "date_key": "2024.4.12", "x": float(easting), "y": 0.0})
        frame = pd.DataFrame(rows)
        splits = assign_spatial_block_folds(frame, block_m=40, buffer_m=10, scheme="spatial_block", n_splits=4, seed=1)
        xy = frame.set_index("sample_id")[["x", "y"]]
        separated = False
        for _, fold in splits.groupby("outer_fold"):
            test_ids = fold.loc[fold["role"] == "test", "sample_id"]
            used_ids = fold.loc[fold["role"].isin(["train", "tune", "calibration"]), "sample_id"]
            for left in test_ids:
                for right in used_ids:
                    distance = np.linalg.norm(xy.loc[left].to_numpy(float) - xy.loc[right].to_numpy(float))
                    self.assertGreaterEqual(distance, 10.0)
            roles = dict(zip(fold["sample_id"], fold["role"]))
            if roles["a"] == "test" and roles["near"] != "test":
                self.assertEqual(roles["near"], "buffer")
                separated = True
        self.assertTrue(separated)

    def test_june_sun_is_higher_than_march(self):
        march = solar_zenith_deg(37.88, 114.85, datetime(2024, 3, 31, 3, 0))
        june = solar_zenith_deg(37.88, 114.85, datetime(2024, 6, 6, 3, 0))
        self.assertLess(june, march)
        self.assertTrue(0 < june < 90)

    def test_conformal_quantile_is_conservative(self):
        scores = np.arange(1, 11, dtype=float)
        self.assertEqual(conformal_quantile(scores, 0.9), 10.0)

    def test_optimism_gap_sign(self):
        gap = optimism_gap({"rmse": 8, "r2": 0.4}, {"rmse": 5, "r2": 0.8})
        self.assertEqual(gap["rmse"], 3)
        self.assertAlmostEqual(gap["r2"], 0.4)

    def test_regression_perfect(self):
        stats = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(stats["rmse"], 0.0)
        self.assertAlmostEqual(stats["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
