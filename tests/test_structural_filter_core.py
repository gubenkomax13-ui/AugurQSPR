import unittest

try:
    from modules import structural_filter_core as structural_filter
except ModuleNotFoundError as exc:
    if exc.name != "rdkit":
        raise
    structural_filter = None


@unittest.skipIf(structural_filter is None, "RDKit is not available")
class StructuralFilterCoreTests(unittest.TestCase):
    def test_formula_includes_implicit_hydrogens(self):
        mol = structural_filter.structural_filter_mol_from_smiles("CCO")

        self.assertEqual(
            structural_filter.structural_filter_formula_from_mol(mol),
            "C2H6O",
        )


if __name__ == "__main__":
    unittest.main()
