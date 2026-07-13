# -*- coding: utf-8 -*-

"""
morfeus_descriptor_core.py

Расчёт 3D / steric / dispersion дескрипторов через morfeus-ml.

Первый стабильный набор:
- RDKit 3D geometry preparation;
- morfeus SASA;
- morfeus Dispersion / P_int;
- простые геометрические 3D-дескрипторы;
- опционально morfeus XTB electronic descriptors.

Важно:
этот модуль пока не меняет qspr_app.py.
Сначала проверяем его отдельно.
"""

import math
import traceback

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from modules.qspr_core import qspr_rename_duplicate_columns


# ---------------------------------------------------------------------
# Availability

def morfeus_is_available():
    """
    Проверяет, доступен ли morfeus.
    """
    try:
        import morfeus  # noqa: F401
        return True, ""
    except Exception as e:
        return False, str(e)


def morfeus_xtb_is_available():
    """
    Проверяет, доступен ли XTB-класс внутри morfeus.
    """
    try:
        from morfeus import XTB  # noqa: F401
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------
# RDKit 3D preparation

def morfeus_smiles_to_rdkit_3d(
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

        # ВАЖНО:
        # Не используем params.maxAttempts — в твоей версии RDKit это уже давало ошибку.
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)

        embed_code = AllChem.EmbedMolecule(mol, params)

        if embed_code != 0:
            # fallback без params
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
                # Не считаем это фатальной ошибкой.
                return mol, "ok_with_uff_warning", f"UFF warning: {e}"

        return mol, "ok", ""

    except Exception as e:
        return None, "3d_prepare_error", str(e)


def morfeus_geometry_from_rdkit_mol(mol):
    """
    RDKit Mol -> elements, coordinates для morfeus.

    elements: список атомных номеров
    coordinates: numpy array shape = (n_atoms, 3), Å
    """
    conf = mol.GetConformer()

    elements = []
    coords = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)

        elements.append(int(atom.GetAtomicNum()))
        coords.append([float(pos.x), float(pos.y), float(pos.z)])

    return elements, np.asarray(coords, dtype=float)


# ---------------------------------------------------------------------
# Simple geometry descriptors

def morfeus_calculate_basic_geometry_descriptors(elements, coordinates):
    """
    Простые 3D-геометрические дескрипторы из координат.

    Не требуют morfeus, но логически относятся к 3D-блоку.
    """
    result = {}

    coords = np.asarray(coordinates, dtype=float)
    elements = list(elements)

    n_atoms = len(elements)
    heavy_mask = np.asarray([int(z) > 1 for z in elements], dtype=bool)

    result["morfeus_geom_atom_count"] = n_atoms
    result["morfeus_geom_heavy_atom_count"] = int(heavy_mask.sum())

    if n_atoms == 0:
        return result

    center = coords.mean(axis=0)
    centered = coords - center

    distances = np.sqrt((centered ** 2).sum(axis=1))

    result["morfeus_geom_radius_gyration"] = float(
        np.sqrt(np.mean(distances ** 2))
    )

    result["morfeus_geom_max_distance_from_centroid"] = float(np.max(distances))
    result["morfeus_geom_mean_distance_from_centroid"] = float(np.mean(distances))
    result["morfeus_geom_std_distance_from_centroid"] = float(np.std(distances))

    # pairwise distances
    pair_dists = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            pair_dists.append(d)

    if pair_dists:
        pair_dists = np.asarray(pair_dists, dtype=float)
        result["morfeus_geom_pair_distance_min"] = float(np.min(pair_dists))
        result["morfeus_geom_pair_distance_max"] = float(np.max(pair_dists))
        result["morfeus_geom_pair_distance_mean"] = float(np.mean(pair_dists))
        result["morfeus_geom_pair_distance_std"] = float(np.std(pair_dists))
    else:
        result["morfeus_geom_pair_distance_min"] = np.nan
        result["morfeus_geom_pair_distance_max"] = np.nan
        result["morfeus_geom_pair_distance_mean"] = np.nan
        result["morfeus_geom_pair_distance_std"] = np.nan

    # Principal moments from covariance-like matrix
    try:
        cov = np.cov(centered.T)

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.asarray(eigvals, dtype=float))

        result["morfeus_geom_principal_moment_1"] = float(eigvals[0])
        result["morfeus_geom_principal_moment_2"] = float(eigvals[1])
        result["morfeus_geom_principal_moment_3"] = float(eigvals[2])

        denom = eigvals[2] if abs(eigvals[2]) > 1e-12 else np.nan

        result["morfeus_geom_asphericity"] = float(eigvals[2] - 0.5 * (eigvals[0] + eigvals[1]))
        result["morfeus_geom_eccentricity"] = float(
            math.sqrt(max(0.0, 1.0 - eigvals[0] / denom))
        ) if np.isfinite(denom) else np.nan

    except Exception:
        result["morfeus_geom_principal_moment_1"] = np.nan
        result["morfeus_geom_principal_moment_2"] = np.nan
        result["morfeus_geom_principal_moment_3"] = np.nan
        result["morfeus_geom_asphericity"] = np.nan
        result["morfeus_geom_eccentricity"] = np.nan

    return result


# ---------------------------------------------------------------------
# morfeus SASA

def morfeus_calculate_sasa_descriptors(elements, coordinates):
    """
    morfeus SASA descriptors:
    - total solvent accessible surface area;
    - volume inside SASA;
    - atomic SASA statistics.
    """
    result = {}

    try:
        from morfeus import SASA

        sasa = SASA(elements, coordinates)

        result["morfeus_sasa_area"] = float(getattr(sasa, "area", np.nan))
        result["morfeus_sasa_volume"] = float(getattr(sasa, "volume", np.nan))

        atom_areas = getattr(sasa, "atom_areas", {})

        if isinstance(atom_areas, dict) and atom_areas:
            vals = np.asarray(list(atom_areas.values()), dtype=float)

            result["morfeus_sasa_atom_area_min"] = float(np.nanmin(vals))
            result["morfeus_sasa_atom_area_max"] = float(np.nanmax(vals))
            result["morfeus_sasa_atom_area_mean"] = float(np.nanmean(vals))
            result["morfeus_sasa_atom_area_std"] = float(np.nanstd(vals))
        else:
            result["morfeus_sasa_atom_area_min"] = np.nan
            result["morfeus_sasa_atom_area_max"] = np.nan
            result["morfeus_sasa_atom_area_mean"] = np.nan
            result["morfeus_sasa_atom_area_std"] = np.nan

        result["morfeus_sasa_status"] = "ok"

    except Exception as e:
        result["morfeus_sasa_area"] = np.nan
        result["morfeus_sasa_volume"] = np.nan
        result["morfeus_sasa_atom_area_min"] = np.nan
        result["morfeus_sasa_atom_area_max"] = np.nan
        result["morfeus_sasa_atom_area_mean"] = np.nan
        result["morfeus_sasa_atom_area_std"] = np.nan
        result["morfeus_sasa_status"] = f"error: {e}"

    return result


# ---------------------------------------------------------------------
# morfeus Dispersion

def morfeus_calculate_dispersion_descriptors(elements, coordinates):
    """
    morfeus Dispersion descriptors:
    - dispersion surface area;
    - dispersion surface volume;
    - P_int.
    """
    result = {}

    try:
        from morfeus import Dispersion

        disp = Dispersion(elements, coordinates)

        # По документации print_report показывает area, volume, P_int.
        result["morfeus_dispersion_area"] = float(getattr(disp, "area", np.nan))
        result["morfeus_dispersion_volume"] = float(getattr(disp, "volume", np.nan))
        result["morfeus_dispersion_p_int"] = float(getattr(disp, "p_int", np.nan))

        atom_p_int = getattr(disp, "atom_p_int", {})

        if isinstance(atom_p_int, dict) and atom_p_int:
            vals = np.asarray(list(atom_p_int.values()), dtype=float)

            result["morfeus_dispersion_atom_p_int_min"] = float(np.nanmin(vals))
            result["morfeus_dispersion_atom_p_int_max"] = float(np.nanmax(vals))
            result["morfeus_dispersion_atom_p_int_mean"] = float(np.nanmean(vals))
            result["morfeus_dispersion_atom_p_int_std"] = float(np.nanstd(vals))
        else:
            result["morfeus_dispersion_atom_p_int_min"] = np.nan
            result["morfeus_dispersion_atom_p_int_max"] = np.nan
            result["morfeus_dispersion_atom_p_int_mean"] = np.nan
            result["morfeus_dispersion_atom_p_int_std"] = np.nan

        result["morfeus_dispersion_status"] = "ok"

    except Exception as e:
        result["morfeus_dispersion_area"] = np.nan
        result["morfeus_dispersion_volume"] = np.nan
        result["morfeus_dispersion_p_int"] = np.nan
        result["morfeus_dispersion_atom_p_int_min"] = np.nan
        result["morfeus_dispersion_atom_p_int_max"] = np.nan
        result["morfeus_dispersion_atom_p_int_mean"] = np.nan
        result["morfeus_dispersion_atom_p_int_std"] = np.nan
        result["morfeus_dispersion_status"] = f"error: {e}"

    return result


# ---------------------------------------------------------------------
# Optional morfeus XTB descriptors

def morfeus_calculate_xtb_descriptors(elements, coordinates):
    """
    Опциональные electronic descriptors через morfeus.XTB.

    В morfeus XTB доступны electronic properties:
    charges, bond orders, energy, HOMO/LUMO, dipole, IP, EA,
    global/local conceptual DFT descriptors.

    Этот блок может быть медленнее и может требовать корректной установки xtb.
    """
    result = {}

    try:
        from morfeus import XTB

        xtb = XTB(elements, coordinates)

        # Energy / orbitals
        for key, getter in [
            ("morfeus_xtb_energy", "get_energy"),
            ("morfeus_xtb_homo", "get_homo"),
            ("morfeus_xtb_lumo", "get_lumo"),
            ("morfeus_xtb_ip", "get_ip"),
            ("morfeus_xtb_ea", "get_ea"),
        ]:
            try:
                value = getattr(xtb, getter)()
                result[key] = float(value)
            except Exception:
                result[key] = np.nan

        if (
            np.isfinite(result.get("morfeus_xtb_homo", np.nan))
            and np.isfinite(result.get("morfeus_xtb_lumo", np.nan))
        ):
            result["morfeus_xtb_gap"] = float(
                result["morfeus_xtb_lumo"] - result["morfeus_xtb_homo"]
            )
        else:
            result["morfeus_xtb_gap"] = np.nan

        # Dipole
        try:
            dip = np.asarray(xtb.get_dipole(), dtype=float)

            result["morfeus_xtb_dipole_x"] = float(dip[0])
            result["morfeus_xtb_dipole_y"] = float(dip[1])
            result["morfeus_xtb_dipole_z"] = float(dip[2])
            result["morfeus_xtb_dipole_norm"] = float(np.linalg.norm(dip))
        except Exception:
            result["morfeus_xtb_dipole_x"] = np.nan
            result["morfeus_xtb_dipole_y"] = np.nan
            result["morfeus_xtb_dipole_z"] = np.nan
            result["morfeus_xtb_dipole_norm"] = np.nan

        # Charges
        try:
            charges_dict = xtb.get_charges()

            if isinstance(charges_dict, dict) and charges_dict:
                charges = np.asarray(list(charges_dict.values()), dtype=float)

                result["morfeus_xtb_charge_min"] = float(np.nanmin(charges))
                result["morfeus_xtb_charge_max"] = float(np.nanmax(charges))
                result["morfeus_xtb_charge_mean"] = float(np.nanmean(charges))
                result["morfeus_xtb_charge_std"] = float(np.nanstd(charges))
                result["morfeus_xtb_charge_abs_mean"] = float(np.nanmean(np.abs(charges)))
                result["morfeus_xtb_charge_abs_max"] = float(np.nanmax(np.abs(charges)))
                result["morfeus_xtb_charge_range"] = float(np.nanmax(charges) - np.nanmin(charges))
            else:
                raise ValueError("empty charges")

        except Exception:
            result["morfeus_xtb_charge_min"] = np.nan
            result["morfeus_xtb_charge_max"] = np.nan
            result["morfeus_xtb_charge_mean"] = np.nan
            result["morfeus_xtb_charge_std"] = np.nan
            result["morfeus_xtb_charge_abs_mean"] = np.nan
            result["morfeus_xtb_charge_abs_max"] = np.nan
            result["morfeus_xtb_charge_range"] = np.nan

        # Conceptual DFT global descriptors
        for desc_name in [
            "electrophilicity",
            "nucleophilicity",
            "electrofugality",
            "nucleofugality",
        ]:
            key = f"morfeus_xtb_global_{desc_name}"

            try:
                result[key] = float(xtb.get_global_descriptor(desc_name))
            except Exception:
                result[key] = np.nan

        # Bond orders summary
        bond_orders = []

        try:
            n = len(elements)

            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    try:
                        bo = float(xtb.get_bond_order(i, j))

                        if np.isfinite(bo) and bo > 1e-8:
                            bond_orders.append(bo)
                    except Exception:
                        pass

            if bond_orders:
                bo_arr = np.asarray(bond_orders, dtype=float)

                result["morfeus_xtb_bond_order_min"] = float(np.nanmin(bo_arr))
                result["morfeus_xtb_bond_order_max"] = float(np.nanmax(bo_arr))
                result["morfeus_xtb_bond_order_mean"] = float(np.nanmean(bo_arr))
                result["morfeus_xtb_bond_order_std"] = float(np.nanstd(bo_arr))
                result["morfeus_xtb_bond_order_sum"] = float(np.nansum(bo_arr))
                result["morfeus_xtb_bond_order_count"] = int(len(bo_arr))
            else:
                result["morfeus_xtb_bond_order_min"] = np.nan
                result["morfeus_xtb_bond_order_max"] = np.nan
                result["morfeus_xtb_bond_order_mean"] = np.nan
                result["morfeus_xtb_bond_order_std"] = np.nan
                result["morfeus_xtb_bond_order_sum"] = np.nan
                result["morfeus_xtb_bond_order_count"] = 0

        except Exception:
            result["morfeus_xtb_bond_order_min"] = np.nan
            result["morfeus_xtb_bond_order_max"] = np.nan
            result["morfeus_xtb_bond_order_mean"] = np.nan
            result["morfeus_xtb_bond_order_std"] = np.nan
            result["morfeus_xtb_bond_order_sum"] = np.nan
            result["morfeus_xtb_bond_order_count"] = np.nan

        result["morfeus_xtb_status"] = "ok"

    except Exception as e:
        result["morfeus_xtb_status"] = f"error: {e}"

    return result


# ---------------------------------------------------------------------
# Main descriptor functions

def calculate_morfeus_descriptors_for_smiles(
    smiles,
    row_index=None,
    compound_id=None,
    random_seed=42,
    optimize=True,
    calc_sasa=True,
    calc_dispersion=True,
    calc_xtb=False
):
    """
    Рассчитывает morfeus-дескрипторы для одного SMILES.

    Возвращает dict.
    """
    result = {
        "row_index": row_index,
        "compound_id": compound_id,
        "input_smiles": smiles,
        "morfeus_status": "not_started",
        "morfeus_error": "",
    }

    mol, status, message = morfeus_smiles_to_rdkit_3d(
        smiles=smiles,
        random_seed=random_seed,
        optimize=optimize
    )

    result["morfeus_3d_status"] = status
    result["morfeus_3d_message"] = message

    if mol is None:
        result["morfeus_status"] = status
        result["morfeus_error"] = message
        return result

    try:
        elements, coordinates = morfeus_geometry_from_rdkit_mol(mol)

        result["morfeus_mol_weight"] = float(Descriptors.MolWt(Chem.RemoveHs(mol)))
        result["morfeus_atom_count_with_H"] = int(len(elements))
        result["morfeus_heavy_atom_count"] = int(sum(1 for z in elements if int(z) > 1))

        result.update(
            morfeus_calculate_basic_geometry_descriptors(
                elements=elements,
                coordinates=coordinates
            )
        )

        if calc_sasa:
            result.update(
                morfeus_calculate_sasa_descriptors(
                    elements=elements,
                    coordinates=coordinates
                )
            )

        if calc_dispersion:
            result.update(
                morfeus_calculate_dispersion_descriptors(
                    elements=elements,
                    coordinates=coordinates
                )
            )

        if calc_xtb:
            result.update(
                morfeus_calculate_xtb_descriptors(
                    elements=elements,
                    coordinates=coordinates
                )
            )

        result["morfeus_status"] = "ok"

    except Exception as e:
        result["morfeus_status"] = "error"
        result["morfeus_error"] = str(e)
        result["morfeus_traceback"] = traceback.format_exc()

    return result


def calculate_morfeus_descriptors_for_dataframe(
    df,
    smiles_col="SMILES",
    id_col=None,
    random_seed=42,
    optimize=True,
    calc_sasa=True,
    calc_dispersion=True,
    calc_xtb=False,
    max_molecules=None,
    progress_callback=None
):
    """
    Рассчитывает morfeus-дескрипторы для DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица веществ.
    smiles_col : str
        Колонка SMILES.
    id_col : str or None
        Необязательная колонка ID.
    progress_callback : callable or None
        Функция вида progress_callback(done, total, message).
        Потом удобно подключить к Streamlit progress bar.

    Returns
    -------
    pd.DataFrame
        Таблица дескрипторов.
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
                f"morfeus: {done}/{total}"
            )

        desc = calculate_morfeus_descriptors_for_smiles(
            smiles=smiles,
            row_index=idx,
            compound_id=compound_id,
            random_seed=random_seed,
            optimize=optimize,
            calc_sasa=calc_sasa,
            calc_dispersion=calc_dispersion,
            calc_xtb=calc_xtb
        )

        rows.append(desc)

    out = pd.DataFrame(rows)

    # Чистим дубли колонок на всякий случай.
    out = qspr_rename_duplicate_columns(out)

    return out


def morfeus_get_numeric_descriptor_columns(df):
    """
    Возвращает список числовых morfeus-дескрипторов,
    исключая служебные колонки и статусы.
    """
    if df is None or df.empty:
        return []

    service_cols = {
        "row_index",
        "compound_id",
        "input_smiles",
        "morfeus_status",
        "morfeus_error",
        "morfeus_traceback",
        "morfeus_3d_status",
        "morfeus_3d_message",
        "morfeus_sasa_status",
        "morfeus_dispersion_status",
        "morfeus_xtb_status",
    }

    numeric_cols = []

    for col in df.columns:
        if col in service_cols:
            continue

        if not str(col).startswith("morfeus_"):
            continue

        test = pd.to_numeric(df[col], errors="coerce")

        if test.notna().any():
            numeric_cols.append(col)

    return numeric_cols


def morfeus_make_descriptor_bundle(
    morfeus_df,
    target_series=None,
    target_col=None
):
    """
    Делает простой bundle, похожий на QSPR descriptor bundle.

    Пока это вспомогательная функция.
    В qspr_app.py подключим позже.
    """
    if morfeus_df is None or morfeus_df.empty:
        raise ValueError("morfeus_df пуст.")

    desc_cols = morfeus_get_numeric_descriptor_columns(morfeus_df)

    if not desc_cols:
        raise ValueError("Не найдено числовых morfeus-дескрипторов.")

    X_df = morfeus_df[desc_cols].copy()

    for col in desc_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # Удаляем полностью пустые колонки.
    X_df = X_df.dropna(axis=1, how="all")

    # Заполняем частичные NaN медианами.
    for col in X_df.columns:
        med = X_df[col].median()

        if pd.isna(med):
            med = 0.0

        X_df[col] = X_df[col].fillna(med)

    desc_cols = list(X_df.columns)

    bundle = {
        "X_all": X_df.values.astype(float),
        "desc_names": desc_cols,
        "df_desc": X_df.copy(),
        "source": "morfeus_descriptors",
    }

    if target_series is not None and target_col is not None:
        y = pd.to_numeric(target_series, errors="coerce")
        bundle["y_all"] = y.values.astype(float)
        bundle["target_col"] = target_col

    return bundle
