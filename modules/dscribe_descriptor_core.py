# -*- coding: utf-8 -*-

"""
dscribe_descriptor_core.py

DScribe-дескрипторы для Augur QSPR.

Первая стабильная версия:
- RDKit SMILES -> 3D conformer;
- ASE Atoms;
- DScribe Coulomb Matrix eigenspectrum;
- фиксированная длина вектора: dscribe_coulomb_eig_001 ... dscribe_coulomb_eig_N.

Пока не добавляем SOAP/MBTR, чтобы не перегрузить модель сотнями/тысячами признаков.
"""

import traceback

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from modules.qspr_core import qspr_rename_duplicate_columns


# ---------------------------------------------------------------------
# Availability


def dscribe_is_available():
    """
    Проверяет, доступны ли dscribe и ase.
    """
    try:
        import dscribe  # noqa: F401
        import ase  # noqa: F401
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------
# RDKit 3D preparation


def dscribe_smiles_to_rdkit_3d(
    smiles,
    random_seed=42,
    optimize=True,
    max_uff_iters=500
):
    """
    SMILES -> RDKit Mol с 3D-конформером.

    Возвращает:
        mol, status, message
    """
    try:
        if smiles is None:
            return None, "empty_smiles", "SMILES is None"

        smiles = str(smiles).strip()

        if not smiles or smiles.lower() in ["nan", "none"]:
            return None, "empty_smiles", "SMILES is empty"

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None, "invalid_smiles", "RDKit не распознал SMILES"

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)

        embed_code = AllChem.EmbedMolecule(mol, params)

        if embed_code != 0:
            embed_code = AllChem.EmbedMolecule(
                mol,
                randomSeed=int(random_seed)
            )

        if embed_code != 0:
            return None, "embed_failed", f"RDKit EmbedMolecule code: {embed_code}"

        if optimize:
            try:
                AllChem.UFFOptimizeMolecule(
                    mol,
                    maxIters=int(max_uff_iters)
                )
            except Exception as e:
                return mol, "ok_with_uff_warning", f"UFF warning: {e}"

        return mol, "ok", ""

    except Exception as e:
        return None, "3d_prepare_error", str(e)


def dscribe_rdkit_mol_to_ase_atoms(mol):
    """
    RDKit Mol -> ase.Atoms.
    """
    from ase import Atoms

    conf = mol.GetConformer()

    atomic_numbers = []
    positions = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)

        atomic_numbers.append(int(atom.GetAtomicNum()))
        positions.append([float(pos.x), float(pos.y), float(pos.z)])

    atoms = Atoms(
        numbers=atomic_numbers,
        positions=np.asarray(positions, dtype=float),
    )

    return atoms


# ---------------------------------------------------------------------
# Coulomb Matrix descriptors

def dscribe_make_coulomb_matrix_descriptor(max_atoms=60):
    """
    Создаёт DScribe CoulombMatrix.

    В актуальных версиях DScribe используется n_atoms_max.
    Аргумент flatten в новых версиях не нужен и может вызывать ошибку.
    """
    from dscribe.descriptors import CoulombMatrix

    return CoulombMatrix(
        n_atoms_max=int(max_atoms),
        permutation="eigenspectrum",
        sparse=False,
    )


def calculate_dscribe_coulomb_descriptors_for_atoms(
    atoms,
    max_atoms=60
):
    """
    Считает Coulomb Matrix eigenspectrum через DScribe.
    """
    result = {}

    n_atoms = len(atoms)

    result["dscribe_atom_count_with_H"] = int(n_atoms)
    result["dscribe_max_atoms"] = int(max_atoms)

    if n_atoms > int(max_atoms):
        result["dscribe_coulomb_status"] = (
            f"too_many_atoms: {n_atoms} > max_atoms={max_atoms}"
        )

        for i in range(1, int(max_atoms) + 1):
            result[f"dscribe_coulomb_eig_{i:03d}"] = np.nan

        return result

    try:
        cm = dscribe_make_coulomb_matrix_descriptor(max_atoms=max_atoms)
        values = cm.create(atoms)

        values = np.asarray(values, dtype=float).reshape(-1)

        # На всякий случай приводим длину ровно к max_atoms.
        if len(values) < int(max_atoms):
            values = np.pad(
                values,
                (0, int(max_atoms) - len(values)),
                mode="constant",
                constant_values=0.0,
            )
        elif len(values) > int(max_atoms):
            values = values[:int(max_atoms)]

        for i, value in enumerate(values, start=1):
            result[f"dscribe_coulomb_eig_{i:03d}"] = float(value)

        result["dscribe_coulomb_status"] = "ok"

    except Exception as e:
        result["dscribe_coulomb_status"] = f"error: {e}"

        for i in range(1, int(max_atoms) + 1):
            result[f"dscribe_coulomb_eig_{i:03d}"] = np.nan

    return result


# ---------------------------------------------------------------------
# Main API


def calculate_dscribe_descriptors_for_smiles(
    smiles,
    row_index=None,
    compound_id=None,
    random_seed=42,
    optimize=True,
    max_atoms=60,
    calc_coulomb=True
):
    """
    Рассчитывает DScribe-дескрипторы для одного SMILES.
    """
    result = {
        "row_index": row_index,
        "compound_id": compound_id,
        "input_smiles": smiles,
        "dscribe_status": "not_started",
        "dscribe_error": "",
    }

    mol, status, message = dscribe_smiles_to_rdkit_3d(
        smiles=smiles,
        random_seed=random_seed,
        optimize=optimize,
    )

    result["dscribe_3d_status"] = status
    result["dscribe_3d_message"] = message

    if mol is None:
        result["dscribe_status"] = status
        result["dscribe_error"] = message
        return result

    try:
        atoms = dscribe_rdkit_mol_to_ase_atoms(mol)

        result["dscribe_atom_count_with_H"] = int(len(atoms))
        result["dscribe_heavy_atom_count"] = int(
            sum(1 for z in atoms.get_atomic_numbers() if int(z) > 1)
        )

        if calc_coulomb:
            result.update(
                calculate_dscribe_coulomb_descriptors_for_atoms(
                    atoms=atoms,
                    max_atoms=max_atoms,
                )
            )

        if (
            result.get("dscribe_coulomb_status", "ok") == "ok"
            or not calc_coulomb
        ):
            result["dscribe_status"] = "ok"
        else:
            result["dscribe_status"] = result.get(
                "dscribe_coulomb_status",
                "error"
            )

    except Exception as e:
        result["dscribe_status"] = "error"
        result["dscribe_error"] = str(e)
        result["dscribe_traceback"] = traceback.format_exc()

    return result


def calculate_dscribe_descriptors_for_dataframe(
    df,
    smiles_col="SMILES",
    id_col=None,
    random_seed=42,
    optimize=True,
    max_atoms=60,
    calc_coulomb=True,
    max_molecules=None,
    progress_callback=None
):
    """
    Рассчитывает DScribe-дескрипторы для DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if smiles_col not in df.columns:
        raise ValueError(f"Колонка SMILES не найдена: {smiles_col}")

    work = df.copy()
    work = qspr_rename_duplicate_columns(work)

    if max_molecules is not None:
        work = work.head(int(max_molecules)).copy()

    rows = []
    total = len(work)

    for done, (idx, row) in enumerate(work.iterrows(), start=1):
        smiles = row.get(smiles_col, "")

        if id_col is not None and id_col in work.columns:
            compound_id = row.get(id_col, idx)
        else:
            compound_id = idx

        if progress_callback is not None:
            progress_callback(
                done,
                total,
                f"DScribe: {done}/{total}"
            )

        desc = calculate_dscribe_descriptors_for_smiles(
            smiles=smiles,
            row_index=idx,
            compound_id=compound_id,
            random_seed=random_seed,
            optimize=optimize,
            max_atoms=max_atoms,
            calc_coulomb=calc_coulomb,
        )

        rows.append(desc)

    out = pd.DataFrame(rows)
    out = qspr_rename_duplicate_columns(out)

    return out


def dscribe_get_numeric_descriptor_columns(df):
    """
    Возвращает список числовых DScribe-дескрипторов.
    """
    if df is None or df.empty:
        return []

    service_cols = {
        "row_index",
        "compound_id",
        "input_smiles",
        "dscribe_status",
        "dscribe_error",
        "dscribe_traceback",
        "dscribe_3d_status",
        "dscribe_3d_message",
        "dscribe_coulomb_status",
    }

    numeric_cols = []

    for col in df.columns:
        if col in service_cols:
            continue

        if not str(col).startswith("dscribe_"):
            continue

        test = pd.to_numeric(df[col], errors="coerce")

        if test.notna().any():
            numeric_cols.append(col)

    return numeric_cols
