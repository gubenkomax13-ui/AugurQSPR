# -*- coding: utf-8 -*-

import unittest

import numpy as np

from modules.validation_extensions_core import (
    group_holdout_validation,
    learning_curve_validation,
    prediction_interval_holdout_coverage,
    repeated_kfold_validation,
)


class ValidationExtensionsCoreTests(unittest.TestCase):
    def setUp(self):
        x0 = np.linspace(0.0, 1.0, 12)
        x1 = np.linspace(1.0, 2.0, 12)
        self.X = np.column_stack([x0, x1, x0 * x1])
        self.y = 1.5 + 2.0 * x0 - 0.4 * x1 + 0.1 * np.arange(12)
        self.smiles = [
            "Cc1ccccc1",
            "Clc1ccccc1",
            "Fc1ccccc1",
            "Brc1ccccc1",
            "CCO",
            "CCCO",
            "CCCCO",
            "CCCCCO",
            "c1ccncc1",
            "Cc1ccncc1",
            "Clc1ccncc1",
            "Fc1ccncc1",
        ]
        self.indices = list(range(100, 112))
        self.model_name = "Ridge"

    def test_repeated_kfold_returns_split_and_aggregate_tables(self):
        result = repeated_kfold_validation(
            self.X,
            self.y,
            self.model_name,
            valid_indices=self.indices,
            smiles=self.smiles,
            k=3,
            n_repeats=2,
        )
        self.assertEqual(result["split_table"].shape[0], 6)
        self.assertEqual(len(result["aggregate_prediction_table"]), len(self.y))
        self.assertIn("RMSE", result["metrics"])

    def test_group_holdout_keeps_groups_out_of_train(self):
        groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
        result = group_holdout_validation(
            self.X,
            self.y,
            self.model_name,
            groups=groups,
            valid_indices=self.indices,
            smiles=self.smiles,
            test_size=0.34,
        )
        self.assertTrue(set(result["train_groups"]).isdisjoint(result["test_groups"]))
        self.assertGreater(len(result["test_table"]), 0)
        self.assertIn("R2", result["metrics_test"])

    def test_learning_curve_returns_expected_columns(self):
        result = learning_curve_validation(
            self.X,
            self.y,
            self.model_name,
            k=3,
        )
        table = result["table"]
        self.assertIn("train_size", table.columns)
        self.assertIn("cv_rmse_mean", table.columns)
        self.assertGreaterEqual(len(table), 2)

    def test_prediction_interval_coverage_returns_observed_coverage(self):
        result = prediction_interval_holdout_coverage(
            self.X,
            self.y,
            self.model_name,
            valid_indices=self.indices,
            smiles=self.smiles,
            test_size=0.25,
            confidence=0.8,
            calibration_cv=3,
        )
        self.assertGreaterEqual(result["coverage"], 0.0)
        self.assertLessEqual(result["coverage"], 1.0)
        self.assertIn("inside_interval", result["table"].columns)


if __name__ == "__main__":
    unittest.main()
