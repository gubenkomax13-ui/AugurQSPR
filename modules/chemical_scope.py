# -*- coding: utf-8 -*-
"""
Chemical-scope helpers for prediction model selection.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def _has_substructure(mol, smarts: str) -> bool:
    patt = Chem.MolFromSmarts(smarts)
    return bool(patt is not None and mol.HasSubstructMatch(patt))


def _longest_carbon_chain(mol) -> int:
    carbon_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
    ]
    carbon_set = set(carbon_indices)
    adjacency = {
        idx: [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetIdx() in carbon_set
        ]
        for idx in carbon_indices
    }

    best = 0

    def dfs(idx: int, seen: set[int]) -> None:
        nonlocal best
        best = max(best, len(seen))
        for nbr in adjacency.get(idx, []):
            if nbr not in seen:
                dfs(nbr, seen | {nbr})

    for idx in carbon_indices:
        dfs(idx, {idx})

    return best


def _alkane_branch_tags(mol, carbon_count: int) -> List[str]:
    if carbon_count == 0:
        return []

    carbon_degrees = []
    methyl_branches = 0
    ethyl_like = False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        carbon_neighbors = [
            nbr
            for nbr in atom.GetNeighbors()
            if nbr.GetAtomicNum() == 6
        ]
        carbon_degrees.append(len(carbon_neighbors))
        if len(carbon_neighbors) >= 3:
            terminal_carbons = [
                nbr
                for nbr in carbon_neighbors
                if sum(
                    1
                    for nn in nbr.GetNeighbors()
                    if nn.GetAtomicNum() == 6
                ) == 1
            ]
            methyl_branches += len(terminal_carbons)

    longest_chain = _longest_carbon_chain(mol)
    side_chain_carbons = max(carbon_count - longest_chain, 0)
    if side_chain_carbons >= 2:
        ethyl_like = True

    tags = []
    if all(deg <= 2 for deg in carbon_degrees):
        tags.append("linear_alkane")
    else:
        tags.append("branched_alkane")
        if methyl_branches == 1:
            tags.append("monomethyl_alkane")
        elif methyl_branches == 2:
            tags.append("dimethyl_alkane")
        elif methyl_branches == 3:
            tags.append("trimethyl_alkane")
        elif methyl_branches > 3:
            tags.append("highly_branched_alkane")
        if ethyl_like:
            tags.append("ethyl_substituted_alkane")

    return tags


def classify_smiles_scope(smiles: str) -> Dict[str, Any]:
    """Classify one SMILES into broad chemical-scope tags."""
    smiles = str(smiles or "").strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    if mol is None:
        return {
            "valid": False,
            "input_smiles": smiles,
            "error": "Invalid SMILES",
        }

    canonical = Chem.MolToSmiles(mol, canonical=True)
    atom_symbols = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    hetero_atoms = sorted({
        atom.GetSymbol()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in (1, 6)
    })
    heavy_atom_count = mol.GetNumHeavyAtoms()
    formula = rdMolDescriptors.CalcMolFormula(mol)

    class_tags = set()
    substructure_tags = set()
    warnings = []

    if carbon_count > 0:
        class_tags.add("organic")

    aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    if aromatic:
        class_tags.add("aromatic")

    if _has_substructure(mol, "c1ccccc1"):
        class_tags.add("benzene_derivative")
        substructure_tags.add("benzene_ring")

    has_alkene = _has_substructure(mol, "[CX3]=[CX3]")
    has_alkyne = _has_substructure(mol, "[CX2]#[CX2]")
    if has_alkene:
        class_tags.add("alkene")
    if has_alkyne:
        class_tags.add("alkyne")

    only_c_h = all(atom.GetAtomicNum() in (1, 6) for atom in Chem.AddHs(mol).GetAtoms())
    has_multiple = any(
        bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
        for bond in mol.GetBonds()
    )
    if only_c_h and not aromatic and not has_multiple:
        class_tags.add("alkane")
        class_tags.update(_alkane_branch_tags(mol, carbon_count))

    functional_smarts = {
        "alcohol": "[CX4][OX2H]",
        "phenol": "c[OX2H]",
        "ether": "[OD2]([#6])[#6]",
        "aldehyde": "[CX3H1](=O)[#6]",
        "ketone": "[#6][CX3](=O)[#6]",
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "ester": "[CX3](=O)[OX2][#6]",
        "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
        "amide": "[NX3][CX3](=O)[#6]",
        "nitrile": "[CX2]#N",
        "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
        "thiol": "[SX2H]",
        "sulfide": "[#6][SX2][#6]",
        "sulfoxide": "[SX3](=O)([#6])[#6]",
        "sulfone": "[SX4](=O)(=O)([#6])[#6]",
    }

    for tag, smarts in functional_smarts.items():
        if _has_substructure(mol, smarts):
            class_tags.add(tag)
            substructure_tags.add(tag)

    if "sulfide" in class_tags:
        class_tags.add("thioether")
        substructure_tags.add("thioether")
    if any(atom.GetAtomicNum() == 16 for atom in mol.GetAtoms()):
        class_tags.add("organosulfur")

    halogens = {"F", "Cl", "Br", "I"}
    if any(atom.GetSymbol() in halogens for atom in mol.GetAtoms()):
        class_tags.add("halogenated")
        if aromatic:
            class_tags.add("halogenated_aromatic")
        if any(atom.GetSymbol() == "F" for atom in mol.GetAtoms()) and aromatic:
            class_tags.add("fluoro_aromatic")
            substructure_tags.add("aryl_fluoride")

    if any(
        atom.IsInRing()
        and atom.GetAtomicNum() not in (6, 1)
        for atom in mol.GetAtoms()
    ):
        class_tags.add("heterocycle")

    if aromatic and "sulfide" in class_tags:
        class_tags.add("aryl_sulfide")

    longest_chain = _longest_carbon_chain(mol)
    if longest_chain >= 4:
        class_tags.add("alkyl_chain")
        substructure_tags.add(f"alkyl_chain_C{longest_chain}")
        if longest_chain >= 8:
            substructure_tags.add(f"long_alkyl_chain_C{longest_chain}")

    return {
        "valid": True,
        "input_smiles": smiles,
        "canonical_smiles": canonical,
        "formula_tags": {
            "formula": formula,
            "atom_symbols": atom_symbols,
            "heavy_atom_count": heavy_atom_count,
            "carbon_count": carbon_count,
            "hetero_atoms": hetero_atoms,
        },
        "class_tags": sorted(class_tags),
        "substructure_tags": sorted(substructure_tags),
        "warnings": warnings,
    }


def _as_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def find_applicable_models(
    scope_result: Dict[str, Any],
    model_cards: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Match a classified molecule against model cards."""
    results = []

    for card in model_cards:
        card_result = {
            "model_id": card.get("model_id", ""),
            "property_label": card.get("property_label", card.get("property", "")),
            "model_name": card.get("model_name", ""),
            "applicability": "undefined",
            "reasons": [],
            "metrics": card.get("metrics", {}) or {},
            "n_train": card.get("n_train", ""),
            "target_units": card.get("target_units", ""),
            "descriptor_set": card.get("descriptor_set", ""),
            "package_path": card.get("package_path", ""),
            "card": card,
        }

        if not scope_result.get("valid"):
            card_result["applicability"] = "not_suitable"
            card_result["reasons"].append(scope_result.get("error", "Invalid SMILES"))
            results.append(card_result)
            continue

        chemical_scope = card.get("chemical_scope")
        if not isinstance(chemical_scope, dict):
            card_result["reasons"].append("chemical_scope модели не задан")
            results.append(card_result)
            continue

        tags = set(scope_result.get("class_tags", [])) | set(scope_result.get("substructure_tags", []))
        atoms = set(scope_result.get("formula_tags", {}).get("atom_symbols", []))
        carbon_count = int(scope_result.get("formula_tags", {}).get("carbon_count", 0))
        reasons = []
        hard_fail = False
        partial = False

        excluded_tags = _as_set(chemical_scope.get("excluded_tags"))
        matched_excluded = sorted(tags & excluded_tags)
        if matched_excluded:
            hard_fail = True
            reasons.append(
                "есть запрещённый структурный класс: " + ", ".join(matched_excluded)
            )

        required_tags = _as_set(chemical_scope.get("required_tags"))
        missing_required = sorted(required_tags - tags)
        if missing_required:
            hard_fail = True
            reasons.append(
                "не выполнены обязательные структурные теги: " + ", ".join(missing_required)
            )

        allowed_atoms = _as_set(chemical_scope.get("allowed_atoms"))
        if allowed_atoms:
            extra_atoms = sorted(atoms - allowed_atoms - {"H"})
            if extra_atoms:
                hard_fail = True
                for atom in extra_atoms:
                    reasons.append(f"атом {atom} отсутствует в области применимости модели")

        carbon_min = chemical_scope.get("carbon_min")
        carbon_max = chemical_scope.get("carbon_max")
        if carbon_min is not None and carbon_count < int(carbon_min):
            partial = True
            reasons.append("число атомов C ниже диапазона обучения")
        if carbon_max is not None and carbon_count > int(carbon_max):
            partial = True
            reasons.append("число атомов C выше диапазона обучения")

        allowed_tags = _as_set(chemical_scope.get("allowed_tags"))
        if allowed_tags and not (tags & allowed_tags):
            partial = True
            reasons.append("нет совпадения с разрешёнными структурными классами модели")

        if hard_fail:
            applicability = "not_suitable"
        elif partial:
            applicability = "partial"
        else:
            applicability = "suitable"
            if not reasons:
                reasons.append("структурные теги и атомный состав соответствуют паспорту модели")

        card_result["applicability"] = applicability
        card_result["reasons"] = reasons
        results.append(card_result)

    return results
