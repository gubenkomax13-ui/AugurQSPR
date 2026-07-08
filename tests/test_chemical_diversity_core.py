import unittest

import pandas as pd

from modules import chemical_diversity_core as diversity


@unittest.skipIf(diversity.Chem is None, "RDKit is not available")
class ChemicalDiversityCoreTests(unittest.TestCase):
    def test_analyze_chemical_diversity_basic_dataset(self):
        data = pd.DataFrame({
            "SMILES": ["CC", "CCC", "c1ccccc1", "CCO", "not_smiles"],
            "Name": ["ethane", "propane", "benzene", "ethanol", "bad"],
            "Y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = diversity.analyze_chemical_diversity(
            data,
            smiles_col="SMILES",
            label_col="Name",
            max_full_molecules=20,
        )

        summary = result["summary"]
        self.assertEqual(summary["valid_structures"], 4)
        self.assertEqual(summary["invalid_structures"], 1)
        self.assertEqual(summary["pairwise_mode"], "full")
        self.assertEqual(summary["total_pairs"], 6)
        self.assertIn("status", summary)
        self.assertFalse(result["top_similar_pairs"].empty)
        self.assertFalse(result["unique_molecules"].empty)
        self.assertFalse(result["cluster_summary"].empty)
        self.assertIn("final_chemical_space", result)
        final_space = result["final_chemical_space"]
        self.assertEqual(len(final_space["map"]), 4)
        self.assertIn("csa_class", final_space["map"].columns)
        self.assertIn("nearest_neighbor_tanimoto", final_space["map"].columns)
        self.assertFalse(final_space["nearest_neighbors"].empty)
        self.assertIn("exact_patterns", final_space)
        self.assertFalse(final_space["exact_patterns"]["groups"].empty)
        self.assertIn("exact_pattern", final_space["exact_patterns"]["groups"].columns)

        communities = diversity.analyze_structural_communities(
            final_space["map"],
            final_space["similarity_matrix"],
            method="Connected components",
            threshold=0.5,
            top_k=3,
        )
        self.assertFalse(communities["nodes"].empty)
        self.assertFalse(communities["groups"].empty)
        self.assertIn("n_groups", communities["summary"])
        self.assertIn("degree", communities["nodes"].columns)
        self.assertIn("method", communities["nodes"].columns)
        self.assertTrue((communities["nodes"]["method"] == "Connected components").all())

        singleton_view = diversity.analyze_structural_communities(
            final_space["map"],
            final_space["similarity_matrix"],
            method="Singletons only",
            threshold=0.95,
            top_k=1,
        )
        self.assertIn("singletons", singleton_view)


if __name__ == "__main__":
    unittest.main()
