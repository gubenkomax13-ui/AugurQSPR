# -*- coding: utf-8 -*-

import unittest

import numpy as np

from modules.error_analysis_core import (
    error_analysis_classify_structure,
    error_analysis_prepare_table,
    error_analysis_structural_annotations,
    error_analysis_structural_series_summary,
    error_analysis_substitution_effects,
)


class StructuralClassificationTests(unittest.TestCase):
    def classify(self, smiles):
        return error_analysis_classify_structure(smiles)

    def test_alkane_branching_series_are_separate(self):
        expected = {
            "CCCCC": ("none", "n-alkanes"),
            "CC(C)CC": ("methyl@2", "2-methylalkanes"),
            "CCC(C)CC": ("methyl@3", "3-methylalkanes"),
            "CC(C)(C)CC": (
                "methyl@2;methyl@2", "2,2-dimethylalkanes"
            ),
            "CC(C)C(C)C": (
                "methyl@2;methyl@3", "2,3-dimethylalkanes"
            ),
        }
        for smiles, values in expected.items():
            with self.subTest(smiles=smiles):
                result = self.classify(smiles)
                self.assertEqual(result["family"], "alkane")
                self.assertEqual(result["substitution_scheme"], values[0])
                self.assertEqual(result["structural_series"], values[1])

    def test_symmetric_alkane_smiles_have_same_signature(self):
        left = self.classify("CC(C)CCC")
        right = self.classify("CCCC(C)C")
        self.assertEqual(left["series_id"], right["series_id"])
        self.assertEqual(left["substitution_scheme"], "methyl@2")

    def test_benzene_ortho_meta_para_are_distinct(self):
        ortho = self.classify("Cc1ccccc1C")
        meta = self.classify("Cc1cccc(C)c1")
        para = self.classify("Cc1ccc(C)cc1")
        self.assertEqual(ortho["substitution_scheme"], "methyl@1;methyl@2")
        self.assertEqual(meta["substitution_scheme"], "methyl@1;methyl@3")
        self.assertEqual(para["substitution_scheme"], "methyl@1;methyl@4")
        self.assertEqual(len({
            ortho["series_id"], meta["series_id"], para["series_id"]
        }), 3)

    def test_toluene_and_halogen_derivatives(self):
        toluene = self.classify("Cc1ccccc1")
        chlorobenzene = self.classify("Clc1ccccc1")
        fluorobenzene = self.classify("Fc1ccccc1")
        bromopyridine = self.classify("Brc1ccncc1")
        methylfuran = self.classify("Cc1ccoc1")
        self.assertEqual(toluene["scaffold"], "benzene")
        self.assertEqual(toluene["substitution_scheme"], "methyl@1")
        self.assertEqual(chlorobenzene["substitution_scheme"], "chloro@1")
        self.assertEqual(fluorobenzene["substitution_scheme"], "fluoro@1")
        self.assertEqual(bromopyridine["family"], "heterocyclic")
        self.assertEqual(bromopyridine["scaffold"], "pyridine")
        self.assertIn("bromo@", bromopyridine["substitution_scheme"])
        self.assertEqual(methylfuran["family"], "heterocyclic")
        self.assertEqual(methylfuran["scaffold"], "furan")

    def test_pyridine_positions_are_normalized_from_nitrogen(self):
        two = self.classify("Cc1ccccn1")
        three = self.classify("Cc1cccnc1")
        four = self.classify("Cc1ccncc1")
        self.assertEqual(two["substitution_scheme"], "methyl@2")
        self.assertEqual(three["substitution_scheme"], "methyl@3")
        self.assertEqual(four["substitution_scheme"], "methyl@4")

    def test_one_and_two_propanol_are_separate_series(self):
        one = self.classify("CCCO")
        two = self.classify("CC(O)C")
        branched = self.classify("CC(C)CO")
        self.assertEqual(one["structural_series"], "1-alkanols")
        self.assertEqual(two["structural_series"], "2-alkanols")
        self.assertEqual(
            branched["structural_series"], "2-methyl-1-alkanols"
        )

    def test_invalid_smiles_does_not_raise(self):
        result = self.classify("not-a-smiles")
        self.assertFalse(result["valid_structure"])
        self.assertEqual(result["family"], "invalid")


class StructuralMetricsTests(unittest.TestCase):
    def setUp(self):
        self.smiles = [
            "CCCCC", "CC(C)CC",
            "CCCCCC", "CC(C)CCC",
            "CCCCCCC", "CC(C)CCCC",
        ]
        self.experimental = np.array([
            36.0, 28.0,
            69.0, 60.0,
            98.0, 90.0,
        ])
        self.predicted = np.array([
            35.0, 30.0,
            68.0, 63.0,
            97.0, 94.0,
        ])
        errors = error_analysis_prepare_table(
            self.smiles, self.experimental, self.predicted
        )
        annotations = error_analysis_structural_annotations(self.smiles)
        self.summary, self.table = error_analysis_structural_series_summary(
            errors,
            annotations,
            min_series_size=3,
            n_bootstrap=100,
        )

    def test_each_series_has_independent_metrics(self):
        by_name = self.summary.set_index("structural_series")
        self.assertIn("n-alkanes", by_name.index)
        self.assertIn("2-methylalkanes", by_name.index)
        self.assertEqual(int(by_name.loc["n-alkanes", "n"]), 3)
        self.assertEqual(int(by_name.loc["2-methylalkanes", "n"]), 3)
        self.assertNotEqual(
            by_name.loc["n-alkanes", "mae"],
            by_name.loc["2-methylalkanes", "mae"],
        )
        self.assertTrue(np.isfinite(
            by_name.loc["2-methylalkanes", "mae_ci_low"]
        ))

    def test_substitution_effect_pairs_use_equal_total_carbon_count(self):
        pairs, summary = error_analysis_substitution_effects(
            self.table, min_series_size=3
        )
        methyl_pairs = pairs[
            pairs["structural_series"] == "2-methylalkanes"
        ]
        self.assertEqual(len(methyl_pairs), 3)
        self.assertEqual(
            methyl_pairs["comparison_size"].astype(int).tolist(),
            [5, 6, 7],
        )
        np.testing.assert_allclose(
            methyl_pairs["delta_error"].to_numpy(),
            [3.0, 4.0, 5.0],
        )
        effect_row = summary[
            summary["structural_series"] == "2-methylalkanes"
        ].iloc[0]
        self.assertEqual(effect_row["reliability"], "adequate")
        self.assertEqual(int(effect_row["n"]), 3)


if __name__ == "__main__":
    unittest.main()
