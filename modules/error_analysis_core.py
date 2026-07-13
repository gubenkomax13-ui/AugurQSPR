# -*- coding: utf-8 -*-

"""
Chemical error profiling for QSPR regression models.

The module contains no Streamlit code. It:
- annotates molecules with overlapping chemical groups;
- summarizes independent prediction errors by group;
- creates automatic structure-similarity clusters;
- calculates bootstrap confidence intervals for group MAE.

Bias convention: prediction - experiment.
Positive bias means systematic overprediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

try:
    from rdkit.Chem import rdFingerprintGenerator
except Exception:
    rdFingerprintGenerator = None


HALOGENS = {9, 17, 35, 53}

FUNCTIONAL_GROUPS = {
    "fg_hydroxyl": "[OX2H]",
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "fg_ester": "[CX3](=O)[OX2][#6]",
    "fg_amide": "[NX3][CX3](=[OX1])",
    "fg_amine": "[NX3;!$(N-C=O);!$(N-S=O)]",
    "fg_ether": "[OD2]([#6])[#6]",
    "fg_carbonyl": "[CX3]=[OX1]",
    "fg_nitrile": "[CX2]#N",
    "fg_nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
}

FUNCTIONAL_GROUP_PATTERNS = {
    key: Chem.MolFromSmarts(smarts)
    for key, smarts in FUNCTIONAL_GROUPS.items()
}


def _numeric_array(values, name):
    array = np.ravel(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _coerce_boolean_series(values, index=None):
    series = pd.Series(values, index=index)

    def coerce_one(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None or pd.isna(value):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "да", "истина"}:
            return True
        if text in {"false", "0", "no", "n", "нет", "ложь", ""}:
            return False
        return False

    return series.map(coerce_one).astype(bool)


def _bootstrap_mae_ci(errors, n_bootstrap=500, random_state=42):
    errors = np.ravel(np.asarray(errors, dtype=float))
    errors = errors[np.isfinite(errors)]
    if len(errors) < 2 or int(n_bootstrap) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(int(random_state))
    values = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sample = rng.choice(errors, size=len(errors), replace=True)
        values[index] = np.mean(np.abs(sample))
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def _metric_row(errors, group_id, group_label, overall_mae, large_error_cutoff,
                n_bootstrap=500, random_state=42):
    errors = np.ravel(np.asarray(errors, dtype=float))
    errors = errors[np.isfinite(errors)]
    n = len(errors)
    if n == 0:
        return None

    absolute = np.abs(errors)
    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    median_ae = float(np.median(absolute))
    max_ae = float(np.max(absolute))
    ci_low, ci_high = _bootstrap_mae_ci(
        errors,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    if np.isfinite(overall_mae) and overall_mae > 1e-12:
        mae_vs_overall_percent = float((mae / overall_mae - 1.0) * 100.0)
    else:
        mae_vs_overall_percent = np.nan

    return {
        "group_id": str(group_id),
        "group": str(group_label),
        "n": int(n),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "median_ae": median_ae,
        "max_ae": max_ae,
        "large_error_fraction": float(np.mean(absolute >= large_error_cutoff)),
        "mae_vs_overall_percent": mae_vs_overall_percent,
        "mae_ci_low": ci_low,
        "mae_ci_high": ci_high,
    }


def error_analysis_prepare_table(
    smiles,
    y_true,
    y_pred,
    original_indices=None,
):
    """Create the common molecule-level error table."""
    smiles = [str(value) for value in list(smiles)]
    y_true = _numeric_array(y_true, "y_true")
    y_pred = _numeric_array(y_pred, "y_pred")

    if not (len(smiles) == len(y_true) == len(y_pred)):
        raise ValueError("SMILES, y_true and y_pred lengths differ.")

    if original_indices is None:
        original_indices = list(range(len(smiles)))
    if len(original_indices) != len(smiles):
        raise ValueError("original_indices length differs from SMILES.")

    table = pd.DataFrame({
        "row_position": np.arange(len(smiles), dtype=int),
        "original_index": list(original_indices),
        "SMILES": smiles,
        "experimental": y_true,
        "predicted": y_pred,
    })
    table["error"] = table["predicted"] - table["experimental"]
    table["absolute_error"] = table["error"].abs()
    table["squared_error"] = table["error"] ** 2
    return table


def error_analysis_chemical_annotations(
    smiles,
    high_molwt=300.0,
    high_logp=3.0,
    high_tpsa=90.0,
):
    """Calculate structural flags and simple physicochemical properties."""
    rows = []

    for position, smiles_value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smiles_value))
        row = {
            "row_position": position,
            "valid_structure": mol is not None,
        }

        if mol is None:
            rows.append(row)
            continue

        atomic_numbers = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        heavy_atomic_numbers = {
            number for number in atomic_numbers if number != 1
        }
        aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
        ring_count = int(rdMolDescriptors.CalcNumRings(mol))
        heterocycle_count = int(rdMolDescriptors.CalcNumHeterocycles(mol))
        formal_charge = int(Chem.GetFormalCharge(mol))
        molwt = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))

        row.update({
            "molwt": molwt,
            "logp": logp,
            "tpsa": tpsa,
            "ring_count": ring_count,
            "heterocycle_count": heterocycle_count,
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "formal_charge": formal_charge,
            "group_hydrocarbon": heavy_atomic_numbers.issubset({6}),
            "group_contains_n": 7 in atomic_numbers,
            "group_contains_o": 8 in atomic_numbers,
            "group_contains_s": 16 in atomic_numbers,
            "group_contains_p": 15 in atomic_numbers,
            "group_halogenated": bool(atomic_numbers & HALOGENS),
            "group_aromatic": aromatic_atoms > 0,
            "group_non_aromatic": aromatic_atoms == 0,
            "group_cyclic": ring_count > 0,
            "group_acyclic": ring_count == 0,
            "group_heterocyclic": heterocycle_count > 0,
            "group_charged": formal_charge != 0,
            "group_high_molwt": molwt > float(high_molwt),
            "group_high_logp": logp > float(high_logp),
            "group_high_tpsa": tpsa > float(high_tpsa),
            "group_flexible": int(Lipinski.NumRotatableBonds(mol)) >= 5,
            "group_multiple_rings": ring_count >= 2,
        })

        for group_key, pattern in FUNCTIONAL_GROUP_PATTERNS.items():
            row[group_key] = bool(
                pattern is not None and mol.HasSubstructMatch(pattern)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def error_analysis_default_group_columns(annotation_table):
    preferred = [
        "group_hydrocarbon",
        "group_aromatic",
        "group_non_aromatic",
        "group_cyclic",
        "group_acyclic",
        "group_heterocyclic",
        "group_contains_n",
        "group_contains_o",
        "group_contains_s",
        "group_contains_p",
        "group_halogenated",
        "group_charged",
        "group_high_molwt",
        "group_high_logp",
        "group_high_tpsa",
        "group_flexible",
        "group_multiple_rings",
        *FUNCTIONAL_GROUPS.keys(),
    ]
    return [column for column in preferred if column in annotation_table.columns]


def error_analysis_group_summary(
    error_table,
    annotation_table,
    group_labels=None,
    min_group_size=5,
    n_bootstrap=500,
    large_error_multiplier=2.0,
    random_state=42,
):
    """Summarize errors for overlapping chemical groups."""
    merged = error_table.merge(
        annotation_table,
        on="row_position",
        how="left",
    )
    finite_errors = merged["error"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    overall_mae = float(np.mean(np.abs(finite_errors)))
    large_error_cutoff = max(
        1e-12,
        float(large_error_multiplier) * overall_mae,
    )
    labels = group_labels or {}
    rows = []

    overall_row = _metric_row(
        finite_errors,
        group_id="overall",
        group_label=labels.get("overall", "Overall"),
        overall_mae=overall_mae,
        large_error_cutoff=large_error_cutoff,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    rows.append(overall_row)

    for group_column in error_analysis_default_group_columns(annotation_table):
        mask = _coerce_boolean_series(
            merged[group_column],
            index=merged.index,
        )
        if int(mask.sum()) < int(min_group_size):
            continue

        row = _metric_row(
            merged.loc[mask, "error"],
            group_id=group_column,
            group_label=labels.get(group_column, group_column),
            overall_mae=overall_mae,
            large_error_cutoff=large_error_cutoff,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        if row is not None:
            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, merged

    summary["reliability"] = np.where(
        summary["n"] >= 10,
        "adequate",
        "small_group",
    )
    summary = summary.sort_values(
        ["mae_vs_overall_percent", "mae"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    return summary, merged


def _morgan_fingerprint(mol, radius=2, n_bits=2048):
    if rdFingerprintGenerator is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=int(radius),
            fpSize=int(n_bits),
        )
        return generator.GetFingerprint(mol)

    from rdkit.Chem import AllChem
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=int(radius),
        nBits=int(n_bits),
    )


def error_analysis_structure_clusters(
    smiles,
    similarity_threshold=0.60,
    min_cluster_size=3,
    radius=2,
    n_bits=2048,
):
    """Cluster valid molecules by Morgan/Tanimoto similarity using Butina."""
    valid_positions = []
    fingerprints = []

    for position, smiles_value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smiles_value))
        if mol is None:
            continue
        valid_positions.append(position)
        fingerprints.append(
            _morgan_fingerprint(mol, radius=radius, n_bits=n_bits)
        )

    assignments = pd.DataFrame({
        "row_position": np.arange(len(smiles), dtype=int),
        "cluster_id": pd.Series([pd.NA] * len(smiles), dtype="Int64"),
        "cluster_size": 0,
    })

    if not fingerprints:
        return assignments

    if len(fingerprints) == 1:
        clusters = ((0,),)
    else:
        distances = []
        for index in range(1, len(fingerprints)):
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[index],
                fingerprints[:index],
            )
            distances.extend([1.0 - value for value in similarities])

        distance_cutoff = 1.0 - float(similarity_threshold)
        clusters = Butina.ClusterData(
            distances,
            len(fingerprints),
            distance_cutoff,
            isDistData=True,
            reordering=True,
        )

    visible_cluster_id = 1
    for cluster in clusters:
        if len(cluster) < int(min_cluster_size):
            continue
        positions = [valid_positions[index] for index in cluster]
        assignments.loc[
            assignments["row_position"].isin(positions),
            "cluster_id",
        ] = visible_cluster_id
        assignments.loc[
            assignments["row_position"].isin(positions),
            "cluster_size",
        ] = len(positions)
        visible_cluster_id += 1

    return assignments


def error_analysis_cluster_summary(
    error_table,
    cluster_assignments,
    min_cluster_size=3,
    n_bootstrap=500,
    large_error_multiplier=2.0,
    random_state=42,
):
    """Summarize errors for automatic structure clusters."""
    merged = error_table.merge(
        cluster_assignments,
        on="row_position",
        how="left",
    )
    finite_errors = merged["error"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    overall_mae = float(np.mean(np.abs(finite_errors)))
    large_error_cutoff = max(
        1e-12,
        float(large_error_multiplier) * overall_mae,
    )
    rows = []

    for cluster_id, cluster_df in merged.dropna(
        subset=["cluster_id"]
    ).groupby("cluster_id"):
        if len(cluster_df) < int(min_cluster_size):
            continue
        row = _metric_row(
            cluster_df["error"],
            group_id=f"cluster_{int(cluster_id)}",
            group_label=f"Cluster {int(cluster_id)}",
            overall_mae=overall_mae,
            large_error_cutoff=large_error_cutoff,
            n_bootstrap=n_bootstrap,
            random_state=random_state + int(cluster_id),
        )
        if row is not None:
            row["cluster_id"] = int(cluster_id)
            rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["reliability"] = np.where(
            summary["n"] >= 10,
            "adequate",
            "small_group",
        )
        summary = summary.sort_values(
            ["mae_vs_overall_percent", "mae"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return summary, merged


def error_analysis_select_group_members(
    annotated_error_table,
    group_id,
):
    """Return molecule-level rows for one chemical group."""
    if group_id == "overall":
        return annotated_error_table.copy()
    if group_id not in annotated_error_table.columns:
        return pd.DataFrame()
    mask = _coerce_boolean_series(
        annotated_error_table[group_id],
        index=annotated_error_table.index,
    )
    return annotated_error_table.loc[mask].copy()


def error_analysis_select_cluster_members(
    clustered_error_table,
    cluster_id,
):
    """Return molecule-level rows for one automatic cluster."""
    if "cluster_id" not in clustered_error_table.columns:
        return pd.DataFrame()
    values = pd.to_numeric(
        clustered_error_table["cluster_id"], errors="coerce"
    )
    return clustered_error_table.loc[
        values == int(cluster_id)
    ].copy()


# ---------------------------------------------------------------------------
# Structural-series analysis

ALKYL_NAMES = {
    1: "methyl",
    2: "ethyl",
    3: "propyl",
    4: "butyl",
    5: "pentyl",
    6: "hexyl",
}

SCAFFOLD_NAMES = {
    "c1ccccc1": "benzene",
    "C1CCCCC1": "cyclohexane",
    "C1CCCC1": "cyclopentane",
    "c1ccncc1": "pyridine",
    "c1cc[nH]c1": "pyrrole",
    "c1cncnc1": "pyrimidine",
    "c1ncc[nH]1": "imidazole",
    "c1ccoc1": "furan",
    "c1ccsc1": "thiophene",
}

ACYCLIC_FAMILY_PATTERNS = [
    ("carboxylic_acid", Chem.MolFromSmarts("[CX3](=O)[OX2H1]")),
    ("ester", Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")),
    ("amide", Chem.MolFromSmarts("[CX3](=O)[NX3]")),
    ("nitrile", Chem.MolFromSmarts("[CX2]#N")),
    ("aldehyde", Chem.MolFromSmarts("[CX3H1](=O)[#6]")),
    ("ketone", Chem.MolFromSmarts("[#6][CX3](=O)[#6]")),
    ("alcohol", Chem.MolFromSmarts("[CX4][OX2H1]")),
    ("amine", Chem.MolFromSmarts("[NX3;!$(N-C=O);!$(N-S=O)]")),
    ("ether", Chem.MolFromSmarts("[#6][OX2][#6]")),
    ("alkene", Chem.MolFromSmarts("[CX3]=[CX3]")),
    ("alkyne", Chem.MolFromSmarts("[CX2]#[CX2]")),
]


def _canonical_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _component_atoms(mol, start, blocked):
    """Return one connected component while treating blocked atoms as removed."""
    blocked = set(blocked)
    seen = set()
    stack = [int(start)]
    while stack:
        atom_index = stack.pop()
        if atom_index in seen or atom_index in blocked:
            continue
        seen.add(atom_index)
        for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if neighbor_index not in seen and neighbor_index not in blocked:
                stack.append(neighbor_index)
    return seen


def _fragment_name(mol, atom_indices):
    atom_indices = sorted(set(int(value) for value in atom_indices))
    atoms = [mol.GetAtomWithIdx(index) for index in atom_indices]
    atomic_numbers = [atom.GetAtomicNum() for atom in atoms]

    if len(atoms) == 1:
        return {
            6: "methyl",
            9: "fluoro",
            17: "chloro",
            35: "bromo",
            53: "iodo",
            8: "hydroxy",
            7: "amino",
        }.get(atomic_numbers[0], atoms[0].GetSymbol().lower())

    if set(atomic_numbers).issubset({6}):
        carbon_count = atomic_numbers.count(6)
        return ALKYL_NAMES.get(carbon_count, f"alkyl-C{carbon_count}")

    fragment_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=atom_indices,
        canonical=True,
        isomericSmiles=True,
    )
    return f"fragment:{fragment_smiles}"


def _carbon_paths(mol, required_atoms=None):
    """Enumerate paths in an acyclic carbon graph."""
    carbon_indices = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
    ]
    carbon_set = set(carbon_indices)
    required_atoms = set(required_atoms or [])
    paths = []

    for left_pos, left in enumerate(carbon_indices):
        for right in carbon_indices[left_pos:]:
            path = (
                (left,)
                if left == right
                else tuple(Chem.GetShortestPath(mol, left, right))
            )
            if not path or not set(path).issubset(carbon_set):
                continue
            if required_atoms and not required_atoms.issubset(path):
                continue
            paths.append(path)

    if not paths:
        return []
    maximum = max(len(path) for path in paths)
    return [path for path in paths if len(path) == maximum]


def _path_substituents(mol, path, ignored_atoms=None):
    path_set = set(path)
    ignored_atoms = set(ignored_atoms or [])
    substituents = []
    for position, atom_index in enumerate(path, start=1):
        atom = mol.GetAtomWithIdx(atom_index)
        for neighbor in atom.GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if neighbor_index in path_set or neighbor_index in ignored_atoms:
                continue
            component = _component_atoms(mol, neighbor_index, path_set)
            component -= ignored_atoms
            if component:
                substituents.append(
                    (position, _fragment_name(mol, component), len(component))
                )
    return substituents


def _normalise_oriented_substituents(substituents, path_length):
    forward = tuple(sorted(
        (int(position), str(name))
        for position, name, _ in substituents
    ))
    reverse = tuple(sorted(
        (int(path_length + 1 - position), str(name))
        for position, name, _ in substituents
    ))
    return min(forward, reverse)


def _select_parent_path(mol, required_atoms=None, priority_locants=None):
    candidates = _carbon_paths(mol, required_atoms=required_atoms)
    if not candidates:
        return tuple(), tuple()

    canonical_ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
    best = None
    for path in candidates:
        substituents = _path_substituents(
            mol,
            path,
            ignored_atoms=set(priority_locants or {}),
        )
        forward_subs = tuple(sorted(
            (position, name) for position, name, _ in substituents
        ))
        reverse_subs = tuple(sorted(
            (len(path) + 1 - position, name)
            for position, name, _ in substituents
        ))

        priority_positions = [
            path.index(atom_index) + 1
            for atom_index in (priority_locants or {})
            if atom_index in path
        ]
        reverse_priority = [
            len(path) + 1 - value for value in priority_positions
        ]
        forward_key = (tuple(sorted(priority_positions)), forward_subs)
        reverse_key = (tuple(sorted(reverse_priority)), reverse_subs)
        if reverse_key < forward_key:
            oriented_path = tuple(reversed(path))
            oriented_subs = reverse_subs
            orientation_key = reverse_key
        else:
            oriented_path = tuple(path)
            oriented_subs = forward_subs
            orientation_key = forward_key

        candidate_key = (
            -len(substituents),
            orientation_key,
            tuple(canonical_ranks[index] for index in oriented_path),
        )
        if best is None or candidate_key < best[0]:
            best = (candidate_key, oriented_path, oriented_subs)

    return best[1], best[2]


def _format_substitution_scheme(substituents):
    if not substituents:
        return "none"
    return ";".join(
        f"{name}@{int(position)}"
        for position, name in sorted(substituents)
    )


def _condensed_substituent_label(substituents):
    if not substituents:
        return ""
    grouped = {}
    for position, name in substituents:
        grouped.setdefault(name, []).append(int(position))
    multiplicity = {2: "di", 3: "tri", 4: "tetra"}
    parts = []
    for name in sorted(grouped):
        positions = sorted(grouped[name])
        prefix = multiplicity.get(len(positions), "" if len(positions) == 1 else f"{len(positions)}-")
        parts.append(f"{','.join(map(str, positions))}-{prefix}{name}")
    return "-".join(parts)


def _is_acyclic_saturated_hydrocarbon(mol):
    if rdMolDescriptors.CalcNumRings(mol) != 0:
        return False
    if any(atom.GetAtomicNum() != 6 for atom in mol.GetAtoms()):
        return False
    return all(
        bond.GetBondType() == Chem.BondType.SINGLE
        for bond in mol.GetBonds()
    )


def _classify_alkane(mol):
    path, substituents = _select_parent_path(mol)
    scheme = _format_substitution_scheme(substituents)
    branch_label = _condensed_substituent_label(substituents)
    series = "n-alkanes" if not substituents else f"{branch_label}alkanes"
    carbon_count = sum(
        atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()
    )
    return {
        "family": "alkane",
        "scaffold": "alkane_chain",
        "scaffold_smiles": "",
        "substituents": scheme,
        "substitution_scheme": scheme,
        "structural_series": series,
        "parent_size": len(path),
        "carbon_count": carbon_count,
        "size": carbon_count,
        "reference_scheme": "none",
        "substituent_count": len(substituents),
        "main_chain_length": len(path),
        "branch_count": len(substituents),
        "branch_positions": ";".join(
            str(int(position)) for position, _ in substituents
        ),
        "branch_types": ";".join(str(name) for _, name in substituents),
        "substitution_pattern": (
            "n-alkane" if not substituents else branch_label
        ),
        "series_label": _saod_alkane_series_label(substituents),
    }


def _saod_alkane_series_label(substituents):
    if not substituents:
        return "n-alkanes"

    typed = [(int(position), str(name)) for position, name in substituents]
    methyl_positions = sorted(
        position for position, name in typed if name == "methyl"
    )
    ethyl_count = sum(1 for _, name in typed if name == "ethyl")
    methyl_count = len(methyl_positions)
    branch_count = len(typed)

    if ethyl_count >= 1 and methyl_count >= 1:
        return "ethyl-methyl alkanes"
    if ethyl_count >= 1:
        return "ethyl-substituted alkanes"

    if methyl_count == 1 and branch_count == 1:
        position = methyl_positions[0]
        if position in {2, 3, 4}:
            return f"{position}-methylalkanes"
        return "methylalkanes"

    if methyl_count == 2 and branch_count == 2:
        pattern = ",".join(str(position) for position in methyl_positions)
        if pattern in {"2,2", "2,3", "2,4", "3,3"}:
            return f"{pattern}-dimethylalkanes"
        return "dimethylalkanes"

    if methyl_count == 3 and branch_count == 3:
        return "trimethylalkanes"
    if methyl_count == 4 and branch_count == 4:
        return "tetramethylalkanes"
    if branch_count >= 4:
        return "highly-branched alkanes"
    return "branched alkanes"


def _ring_substitution_signature(mol, scaffold):
    matches = mol.GetSubstructMatches(
        scaffold,
        uniquify=False,
        useChirality=False,
        maxMatches=10000,
    )
    if not matches:
        return tuple()

    signatures = []
    for match in matches:
        scaffold_atoms = set(match)
        entries = []
        visited_components = set()
        for position, mol_atom_index in enumerate(match, start=1):
            atom = mol.GetAtomWithIdx(mol_atom_index)
            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index in scaffold_atoms:
                    continue
                component = frozenset(
                    _component_atoms(mol, neighbor_index, scaffold_atoms)
                )
                if not component or component in visited_components:
                    continue
                visited_components.add(component)
                entries.append((
                    position,
                    _fragment_name(mol, component),
                ))
        signatures.append(tuple(sorted(entries)))
    return min(signatures)


def _number_ring_scaffold(scaffold):
    """Put the highest-priority heteroatom first for simple monocyclic rings."""
    atom_rings = list(scaffold.GetRingInfo().AtomRings())
    if (
        len(atom_rings) != 1
        or len(atom_rings[0]) != scaffold.GetNumAtoms()
    ):
        return scaffold

    ring = list(atom_rings[0])
    candidates = []
    for direction in (ring, list(reversed(ring))):
        for offset in range(len(direction)):
            order = direction[offset:] + direction[:offset]
            atomic_numbers = tuple(
                -scaffold.GetAtomWithIdx(index).GetAtomicNum()
                for index in order
            )
            candidates.append((atomic_numbers, tuple(order)))
    order = min(candidates)[1]
    return Chem.RenumberAtoms(scaffold, list(order))


def _classify_ring_system(mol):
    raw_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(
        raw_scaffold, canonical=True, isomericSmiles=False
    )
    scaffold = _number_ring_scaffold(Chem.MolFromSmiles(scaffold_smiles))
    substituents = _ring_substitution_signature(mol, scaffold)
    scheme = _format_substitution_scheme(substituents)
    scaffold_name = SCAFFOLD_NAMES.get(
        scaffold_smiles, f"scaffold:{scaffold_smiles}"
    )
    has_aromatic = any(atom.GetIsAromatic() for atom in scaffold.GetAtoms())
    has_hetero = any(
        atom.GetAtomicNum() not in {1, 6} for atom in scaffold.GetAtoms()
    )
    if has_hetero:
        family = "heterocyclic"
    elif has_aromatic:
        family = "aromatic"
    else:
        family = "carbocyclic"

    label = _condensed_substituent_label(substituents)
    series = (
        f"{scaffold_name} derivatives"
        if not label else f"{label} {scaffold_name} derivatives"
    )
    carbon_count = sum(
        atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()
    )
    return {
        "family": family,
        "scaffold": scaffold_name,
        "scaffold_smiles": scaffold_smiles,
        "substituents": scheme,
        "substitution_scheme": scheme,
        "structural_series": series,
        "parent_size": scaffold.GetNumHeavyAtoms(),
        "carbon_count": carbon_count,
        "size": scaffold.GetNumHeavyAtoms(),
        "reference_scheme": "none",
        "substituent_count": len(substituents),
    }


def _functional_family(mol):
    for family, pattern in ACYCLIC_FAMILY_PATTERNS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return family, mol.GetSubstructMatch(pattern)
    return "acyclic_other", tuple()


def _classify_acyclic_functional(mol):
    family, match = _functional_family(mol)
    priority_carbon = None
    functional_atoms = set(match)

    if family == "alcohol" and len(match) >= 2:
        priority_carbon = next(
            (index for index in match
             if mol.GetAtomWithIdx(index).GetAtomicNum() == 6),
            None,
        )
    elif match:
        priority_carbon = next(
            (index for index in match
             if mol.GetAtomWithIdx(index).GetAtomicNum() == 6),
            None,
        )

    required = {priority_carbon} if priority_carbon is not None else set()
    path, substituents = _select_parent_path(
        mol,
        required_atoms=required,
        priority_locants=required,
    )
    functional_locant = (
        path.index(priority_carbon) + 1
        if priority_carbon is not None and priority_carbon in path
        else None
    )

    side_substituents = []
    path_set = set(path)
    for position, atom_index in enumerate(path, start=1):
        for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if neighbor_index in path_set or neighbor_index in functional_atoms:
                continue
            component = _component_atoms(mol, neighbor_index, path_set)
            component -= functional_atoms
            if component:
                side_substituents.append(
                    (position, _fragment_name(mol, component))
                )

    functional_name = {
        "alcohol": "hydroxyl",
        "amine": "amino",
        "carboxylic_acid": "carboxyl",
        "nitrile": "nitrile",
        "aldehyde": "formyl",
        "ketone": "oxo",
        "ester": "ester",
        "amide": "amide",
        "ether": "ether",
        "alkene": "double-bond",
        "alkyne": "triple-bond",
    }.get(family, family)
    signature_items = list(side_substituents)
    if functional_locant is not None:
        signature_items.append((functional_locant, functional_name))
    scheme = _format_substitution_scheme(signature_items)
    branch_label = _condensed_substituent_label(side_substituents)

    if family == "alcohol" and functional_locant is not None:
        series = (
            f"{branch_label}-{functional_locant}-alkanols"
            if branch_label else f"{functional_locant}-alkanols"
        )
    else:
        series = (
            f"{branch_label}-{family} series"
            if branch_label else f"{family} series"
        )

    carbon_count = sum(
        atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()
    )
    return {
        "family": family,
        "scaffold": f"{family}_chain",
        "scaffold_smiles": "",
        "substituents": _format_substitution_scheme(side_substituents),
        "substitution_scheme": scheme,
        "structural_series": series,
        "parent_size": len(path),
        "carbon_count": carbon_count,
        "size": carbon_count,
        "functional_locant": functional_locant,
        "reference_scheme": (
            _format_substitution_scheme(
                [(functional_locant, functional_name)]
            )
            if functional_locant is not None else "none"
        ),
        "substituent_count": len(side_substituents),
    }


def error_analysis_classify_structure(smiles):
    """Return a deterministic graph-based structural-series annotation."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            "valid_structure": False,
            "canonical_smiles": "",
            "family": "invalid",
            "scaffold": "invalid",
            "scaffold_smiles": "",
            "substituents": "",
            "substitution_scheme": "",
            "structural_series": "invalid structure",
            "series_id": "invalid|invalid|invalid",
            "parent_size": np.nan,
            "carbon_count": np.nan,
            "size": np.nan,
            "reference_scheme": "",
            "substituent_count": 0,
        }

    if _is_acyclic_saturated_hydrocarbon(mol):
        result = _classify_alkane(mol)
    elif rdMolDescriptors.CalcNumRings(mol) > 0:
        result = _classify_ring_system(mol)
    else:
        result = _classify_acyclic_functional(mol)

    result["valid_structure"] = True
    result["canonical_smiles"] = _canonical_smiles(mol)
    result["series_id"] = "|".join([
        str(result["family"]),
        str(result["scaffold"]),
        str(result["substitution_scheme"]),
    ])
    return result


def error_analysis_structural_annotations(smiles):
    rows = []
    for position, smiles_value in enumerate(smiles):
        row = error_analysis_classify_structure(smiles_value)
        row["row_position"] = position
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_slope(x, y, minimum=3):
    frame = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < int(minimum) or frame["x"].nunique() < 2:
        return np.nan
    x_values = frame["x"].to_numpy(dtype=float)
    y_values = frame["y"].to_numpy(dtype=float)
    x_centered = x_values - np.mean(x_values)
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 1e-15:
        return np.nan
    return float(np.dot(x_centered, y_values - np.mean(y_values)) / denominator)


def _safe_correlation(x, y, minimum=3):
    frame = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < int(minimum) or min(
        frame["x"].nunique(), frame["y"].nunique()
    ) < 2:
        return np.nan
    x_ranks = frame["x"].rank(method="average")
    y_ranks = frame["y"].rank(method="average")
    x_values = x_ranks.to_numpy(dtype=float)
    y_values = y_ranks.to_numpy(dtype=float)
    x_centered = x_values - np.mean(x_values)
    y_centered = y_values - np.mean(y_values)
    denominator = float(np.sqrt(
        np.dot(x_centered, x_centered)
        * np.dot(y_centered, y_centered)
    ))
    if denominator <= 1e-15:
        return np.nan
    return float(np.dot(x_centered, y_centered) / denominator)


def _monotonic_direction(size, values, minimum=3):
    frame = pd.DataFrame({"size": size, "value": values}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < int(minimum) or frame["size"].nunique() < 3:
        return "insufficient"
    grouped = frame.groupby("size", as_index=False)["value"].mean()
    differences = np.diff(grouped["value"].to_numpy(dtype=float))
    if np.all(differences >= 0):
        return "increasing"
    if np.all(differences <= 0):
        return "decreasing"
    return "non_monotonic"


def _order_preservation_fraction(experimental, predicted):
    experimental = np.asarray(experimental, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    comparable = 0
    preserved = 0
    for left in range(len(experimental)):
        for right in range(left + 1, len(experimental)):
            exp_delta = experimental[right] - experimental[left]
            pred_delta = predicted[right] - predicted[left]
            if not np.isfinite(exp_delta) or not np.isfinite(pred_delta):
                continue
            if abs(exp_delta) <= 1e-12:
                continue
            comparable += 1
            if exp_delta * pred_delta > 0:
                preserved += 1
    return (
        float(preserved / comparable) if comparable else np.nan
    )


def _trend_outlier_mask(size, values, minimum=4):
    frame = pd.DataFrame({"size": size, "value": values}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    result = pd.Series(False, index=getattr(size, "index", None), dtype=bool)
    if len(frame) < int(minimum) or frame["size"].nunique() < 3:
        return result

    slope = _safe_slope(frame["size"], frame["value"], minimum=minimum)
    if not np.isfinite(slope):
        return result
    intercept = float(
        frame["value"].mean() - slope * frame["size"].mean()
    )
    residuals = frame["value"] - (
        intercept + slope * frame["size"]
    )
    residual_median = float(residuals.median())
    mad = float(np.median(np.abs(residuals - residual_median)))
    if mad <= 1e-15:
        return result
    robust_z = np.abs(residuals - residual_median) / (1.4826 * mad)
    result.loc[frame.index] = robust_z > 3.5
    return result


def error_analysis_structural_series_summary(
    error_table,
    structural_annotations,
    min_series_size=5,
    n_bootstrap=500,
    large_error_multiplier=2.0,
    random_state=42,
):
    """Calculate independent error and trend diagnostics for every series."""
    merged = error_table.merge(
        structural_annotations, on="row_position", how="left"
    )
    valid_structure = _coerce_boolean_series(
        merged["valid_structure"],
        index=merged.index,
    )
    valid = merged[
        valid_structure
        & np.isfinite(merged["error"])
    ].copy()
    merged["series_trend_outlier"] = False
    overall_mae = float(valid["absolute_error"].mean()) if len(valid) else np.nan
    cutoff = max(
        1e-12,
        float(large_error_multiplier) * overall_mae,
    ) if np.isfinite(overall_mae) else np.nan
    rows = []

    for series_id, group in valid.groupby("series_id", sort=True):
        outlier_mask = _trend_outlier_mask(
            pd.to_numeric(group["size"], errors="coerce"),
            group["experimental"],
        )
        merged.loc[group.index, "series_trend_outlier"] = outlier_mask
        metric = _metric_row(
            group["error"],
            group_id=series_id,
            group_label=group["structural_series"].iloc[0],
            overall_mae=overall_mae,
            large_error_cutoff=cutoff,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        if metric is None:
            continue
        sizes = pd.to_numeric(group["size"], errors="coerce")
        experimental_direction = _monotonic_direction(
            sizes, group["experimental"]
        )
        predicted_direction = _monotonic_direction(
            sizes, group["predicted"]
        )
        metric.update({
            "series_id": series_id,
            "family": group["family"].iloc[0],
            "scaffold": group["scaffold"].iloc[0],
            "substitution_scheme": group["substitution_scheme"].iloc[0],
            "structural_series": group["structural_series"].iloc[0],
            "size_min": float(sizes.min()) if sizes.notna().any() else np.nan,
            "size_max": float(sizes.max()) if sizes.notna().any() else np.nan,
            "size_range": (
                f"{int(sizes.min())}–{int(sizes.max())}"
                if sizes.notna().any() else ""
            ),
            "error_size_slope": _safe_slope(sizes, group["error"]),
            "absolute_error_size_slope": _safe_slope(
                sizes, group["absolute_error"]
            ),
            "property_size_correlation": _safe_correlation(
                sizes, group["experimental"]
            ),
            "experimental_monotonicity": experimental_direction,
            "predicted_monotonicity": predicted_direction,
            "trend_direction_preserved": (
                experimental_direction == predicted_direction
                and experimental_direction not in {
                    "insufficient", "non_monotonic"
                }
            ),
            "order_preservation": _order_preservation_fraction(
                group["experimental"], group["predicted"]
            ),
            "trend_outlier_count": int(outlier_mask.sum()),
            "problem_compounds": ";".join(
                str(value) for value in group.nlargest(
                    min(3, len(group)), "absolute_error"
                )["original_index"].tolist()
            ),
            "reliability": (
                "adequate"
                if len(group) >= int(min_series_size)
                else "insufficient"
            ),
        })
        rows.append(metric)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["mae_vs_overall_percent", "mae", "n"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return summary, merged


def error_analysis_saod_alkane_series_summary(structural_error_table):
    """
    Summarize model residuals by SAOD-style alkane substitution series.

    Residual convention for this table is experimental - predicted.
    Positive residual means the model underestimated the property.
    """
    required = {
        "family", "series_label", "experimental", "predicted",
        "carbon_count", "SMILES",
    }
    if structural_error_table is None or not required.issubset(
        set(structural_error_table.columns)
    ):
        return pd.DataFrame(), pd.DataFrame()

    table = structural_error_table.copy()
    table = table[
        (table["family"] == "alkane")
        & table["series_label"].astype(str).str.strip().ne("")
    ].copy()
    if table.empty:
        return pd.DataFrame(), table

    table["residual"] = (
        pd.to_numeric(table["experimental"], errors="coerce")
        - pd.to_numeric(table["predicted"], errors="coerce")
    )
    table["absolute_residual"] = table["residual"].abs()
    table["carbon_count"] = pd.to_numeric(
        table["carbon_count"], errors="coerce"
    )
    table["main_chain_length"] = pd.to_numeric(
        table.get("main_chain_length", np.nan), errors="coerce"
    )
    table["branch_count"] = pd.to_numeric(
        table.get("branch_count", np.nan), errors="coerce"
    )
    table = table[np.isfinite(table["residual"])].copy()
    if table.empty:
        return pd.DataFrame(), table

    rows = []
    for series_label, group in table.groupby("series_label", sort=False):
        residual = group["residual"].to_numpy(dtype=float)
        examples = (
            group.sort_values("carbon_count")["SMILES"]
            .astype(str)
            .head(3)
            .tolist()
        )
        rows.append({
            "series_label": str(series_label),
            "n_compounds": int(len(group)),
            "mean_experimental": float(group["experimental"].mean()),
            "mean_predicted": float(group["predicted"].mean()),
            "mean_residual": float(np.mean(residual)),
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "bias": float(np.mean(residual)),
            "min_carbon_count": (
                int(group["carbon_count"].min())
                if group["carbon_count"].notna().any() else np.nan
            ),
            "max_carbon_count": (
                int(group["carbon_count"].max())
                if group["carbon_count"].notna().any() else np.nan
            ),
            "example_compounds": "; ".join(examples),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, table

    preferred = {
        "n-alkanes": 0,
        "2-methylalkanes": 1,
        "3-methylalkanes": 2,
        "4-methylalkanes": 3,
        "methylalkanes": 4,
        "2,2-dimethylalkanes": 5,
        "2,3-dimethylalkanes": 6,
        "2,4-dimethylalkanes": 7,
        "3,3-dimethylalkanes": 8,
        "dimethylalkanes": 9,
        "ethyl-substituted alkanes": 10,
        "ethyl-methyl alkanes": 11,
        "trimethylalkanes": 12,
        "tetramethylalkanes": 13,
        "highly-branched alkanes": 14,
        "branched alkanes": 15,
    }
    summary["_order"] = summary["series_label"].map(preferred).fillna(100)
    summary = summary.sort_values(
        ["_order", "series_label"]
    ).drop(columns="_order").reset_index(drop=True)
    return summary, table


def error_analysis_substitution_effects(
    structural_error_table,
    min_series_size=3,
):
    """Build reference/substituted pairs and summarize substitution effects."""
    table = structural_error_table.copy()
    valid_structure = _coerce_boolean_series(
        table["valid_structure"],
        index=table.index,
    )
    table = table[
        valid_structure
        & np.isfinite(table["experimental"])
        & np.isfinite(table["predicted"])
    ]
    pair_rows = []

    for _, substituted in table.iterrows():
        if int(substituted.get("substituent_count", 0) or 0) <= 0:
            continue

        if substituted["family"] == "alkane":
            candidates = table[
                (table["family"] == "alkane")
                & (table["substitution_scheme"] == "none")
                & (table["carbon_count"] == substituted["carbon_count"])
            ]
            comparison_size = substituted["carbon_count"]
        elif substituted["family"] in {
            "aromatic", "heterocyclic", "carbocyclic"
        }:
            candidates = table[
                (table["family"] == substituted["family"])
                & (table["scaffold"] == substituted["scaffold"])
                & (table["substitution_scheme"] == "none")
            ]
            comparison_size = substituted["parent_size"]
        else:
            candidates = table[
                (table["family"] == substituted["family"])
                & (table["reference_scheme"] == substituted["reference_scheme"])
                & (table["substituent_count"] == 0)
                & (table["carbon_count"] == substituted["carbon_count"])
            ]
            comparison_size = substituted["carbon_count"]

        if candidates.empty:
            continue
        reference = candidates.sort_values(
            ["original_index", "row_position"]
        ).iloc[0]
        delta_experimental = (
            substituted["experimental"] - reference["experimental"]
        )
        delta_predicted = substituted["predicted"] - reference["predicted"]
        delta_error = delta_predicted - delta_experimental
        pair_rows.append({
            "effect_series_id": substituted["series_id"],
            "family": substituted["family"],
            "scaffold": substituted["scaffold"],
            "substitution_scheme": substituted["substitution_scheme"],
            "structural_series": substituted["structural_series"],
            "comparison_size": comparison_size,
            "reference_index": reference["original_index"],
            "reference_smiles": reference["SMILES"],
            "substituted_index": substituted["original_index"],
            "substituted_smiles": substituted["SMILES"],
            "delta_experimental": float(delta_experimental),
            "delta_predicted": float(delta_predicted),
            "delta_error": float(delta_error),
            "absolute_delta_error": float(abs(delta_error)),
            "direction_correct": bool(
                abs(delta_experimental) <= 1e-12
                and abs(delta_predicted) <= 1e-12
                or delta_experimental * delta_predicted > 0
            ),
        })

    pairs = pd.DataFrame(pair_rows)
    summaries = []
    if not pairs.empty:
        for series_id, group in pairs.groupby("effect_series_id", sort=True):
            summaries.append({
                "effect_series_id": series_id,
                "family": group["family"].iloc[0],
                "scaffold": group["scaffold"].iloc[0],
                "substitution_scheme": group["substitution_scheme"].iloc[0],
                "structural_series": group["structural_series"].iloc[0],
                "n": len(group),
                "effect_mae": float(group["absolute_delta_error"].mean()),
                "effect_rmse": float(np.sqrt(
                    np.mean(group["delta_error"] ** 2)
                )),
                "effect_bias": float(group["delta_error"].mean()),
                "direction_accuracy": float(group["direction_correct"].mean()),
                "experimental_effect_slope": _safe_slope(
                    group["comparison_size"],
                    group["delta_experimental"],
                ),
                "predicted_effect_slope": _safe_slope(
                    group["comparison_size"],
                    group["delta_predicted"],
                ),
                "reliability": (
                    "adequate"
                    if len(group) >= int(min_series_size)
                    else "insufficient"
                ),
            })
    return pairs, pd.DataFrame(summaries)


def error_analysis_problem_molecules(
    structural_error_table,
    large_error_multiplier=2.0,
):
    table = structural_error_table.copy()
    finite = table[np.isfinite(table["absolute_error"])].copy()
    if finite.empty:
        return finite
    cutoff = max(
        1e-12,
        float(large_error_multiplier) * float(finite["absolute_error"].mean()),
    )
    finite["large_error"] = finite["absolute_error"] >= cutoff
    finite["error_rank"] = finite["absolute_error"].rank(
        method="first", ascending=False
    ).astype(int)
    if "series_trend_outlier" not in finite:
        finite["series_trend_outlier"] = False
    return finite.sort_values(
        ["large_error", "series_trend_outlier", "absolute_error"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
