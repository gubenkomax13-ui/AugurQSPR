# -*- coding: utf-8 -*-

"""
qspr_core.py

Ядро QSPR для GenQSPR:
- расчёт RDKit/Mordred/PaDEL дескрипторов;
- использование готовых дескрипторов из файла;
- создание моделей регрессии;
- обучение аналитической модели;
- Hold-out;
- K-Fold;
- Leave-One-Out;
- расчёт метрик;
- автосохранение результатов.

Этот модуль не содержит Streamlit-интерфейс.
"""

import os
import json
from datetime import datetime
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import numpy as np
import pandas as pd
import streamlit as st

try:
    from .model_catalog import (
        MODEL_GROUP_BOOSTING,
        MODEL_GROUP_KERNEL_SIMILARITY,
        MODEL_GROUP_LINEAR,
        MODEL_GROUP_META_ENSEMBLES,
        MODEL_GROUP_NEURAL,
        MODEL_GROUP_SPLINE,
        MODEL_GROUP_SYMBOLIC,
        MODEL_GROUP_TREES,
        MODEL_GROUP_TREE_ENSEMBLES,
        get_model_display_name,
        get_model_encyclopedia_key,
        get_model_group,
        get_models_by_group,
        normalize_runtime_name,
    )
except ImportError:
    from modules.model_catalog import (
        MODEL_GROUP_BOOSTING,
        MODEL_GROUP_KERNEL_SIMILARITY,
        MODEL_GROUP_LINEAR,
        MODEL_GROUP_META_ENSEMBLES,
        MODEL_GROUP_NEURAL,
        MODEL_GROUP_SPLINE,
        MODEL_GROUP_SYMBOLIC,
        MODEL_GROUP_TREES,
        MODEL_GROUP_TREE_ENSEMBLES,
        get_model_display_name,
        get_model_encyclopedia_key,
        get_model_group,
        get_models_by_group,
        normalize_runtime_name,
    )

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    train_test_split,
    KFold,
    LeaveOneOut,
    cross_val_predict
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


# ------------------------------------------------------------------
# Опциональные библиотеки

mordred_available = False

try:
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors
    mordred_available = True
except Exception:
    mordred_available = False


padel_available = False

try:
    from padelpy import from_smiles
    padel_available = True
except Exception:
    padel_available = False


xgboost_available = False

try:
    import xgboost as xgb
    xgboost_available = True
except Exception:
    xgboost_available = False


lightgbm_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
except Exception:
    lightgbm_available = False


catboost_available = False

try:
    from catboost import CatBoostRegressor
    catboost_available = True
except Exception:
    catboost_available = False


pysr_available = False

try:
    from pysr import PySRRegressor
    pysr_available = True
except Exception:
    pysr_available = False

xtb_python_available = False

try:
    from xtb.interface import Calculator as XTBCalculator
    from xtb.utils import get_method

    try:
        from xtb.libxtb import VERBOSITY_MUTED
    except Exception:
        VERBOSITY_MUTED = 0

    xtb_python_available = True
except Exception:
    xtb_python_available = False

# ------------------------------------------------------------------
# Файлы и папки

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DESC_LISTS_FILE = "descriptor_lists.json"
PADEL_UNIQUE_FILE = "padel_unique_descriptors.txt"
DESC_MEANINGS_FILE = "descriptor_meanings.json"

MODEL_ENCYCLOPEDIA_FILE = "model_encyclopedia.json"
MODEL_ENCYCLOPEDIA_FALLBACK_FILE = os.path.join(
    "help",
    "model_encyclopedia.json"
)

# ------------------------------------------------------------------
# Служебные функции

def qspr_safe_target_name(target_col):
    """
    Безопасное имя свойства для имени файла.
    """
    safe = str(target_col)
    safe = safe.replace(" ", "_")
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")
    safe = safe.replace(":", "_")
    return safe


def qspr_save_results_auto(df, prefix, target_col, n_compounds, results_dir=RESULTS_DIR):
    """
    Автосохранение DataFrame в CSV.

    Имя:
    prefix_property_N_timestamp.csv
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_target = qspr_safe_target_name(target_col)

    filename = f"{prefix}_{safe_target}_{n_compounds}_{timestamp}.csv"
    path = os.path.join(results_dir, filename)

    df.to_csv(path, index=False)

    return path


def qspr_to_numeric(series):
    """
    Безопасное приведение серии к числу.
    Поддерживает десятичную запятую.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def qspr_clean_numeric_dataframe(df):
    """
    Приводит DataFrame к числовому виду, удаляет пустые/константные колонки,
    заполняет пропуски медианами.
    """
    work = df.copy()

    for col in work.columns:
        work[col] = pd.to_numeric(
            work[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

    work = work.replace([np.inf, -np.inf], np.nan)

    work = work.dropna(axis=1, how="all")

    for col in work.columns:
        median_value = work[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        work[col] = work[col].fillna(median_value)

    const_cols = [
        col for col in work.columns
        if work[col].nunique(dropna=True) <= 1
    ]

    if const_cols:
        work = work.drop(columns=const_cols)

    return work


def qspr_metrics(y_true, y_pred):
    """
    Возвращает основные метрики регрессии.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        mape_values = np.abs((y_true - y_pred) / y_true) * 100
        mape_values = mape_values[np.isfinite(mape_values)]

    if len(mape_values) > 0:
        mape = float(np.mean(mape_values))
    else:
        mape = np.nan

    return {
        "R2": r2,
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": mae,
        "ME": float(np.mean(errors)),
        "SD": float(np.std(errors)),
        "MedianError": float(np.median(errors)),
        "MAPE_percent": mape
    }


def qspr_prediction_table(
    y_true,
    y_pred,
    smiles=None,
    original_indices=None,
    dataset_label=None
):
    """
    Формирует таблицу эксперимент/расчёт/ошибка.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)

    table = pd.DataFrame({
        "№": range(1, n + 1),
        "Экспериментальное значение": y_true,
        "Расчётное значение": y_pred,
        "Ошибка": y_true - y_pred
    })

    if original_indices is not None:
        table.insert(1, "Индекс", list(original_indices))

    if smiles is not None:
        insert_pos = 2 if original_indices is not None else 1
        table.insert(insert_pos, "SMILES", list(smiles))

    if dataset_label is not None:
        table["Выборка"] = dataset_label

    return table


# ------------------------------------------------------------------
# Списки дескрипторов

def qspr_load_padel_unique_from_file(filepath=None):
    """
    Загружает список уникальных PaDEL-дескрипторов из txt-файла.

    Если filepath не задан, ищет файл:
    1) в текущей рабочей папке;
    2) в корне проекта рядом с папкой modules.
    """
    candidate_paths = []

    if filepath is not None:
        candidate_paths.append(filepath)
    else:
        candidate_paths.append("padel_unique_descriptors.txt")

        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(module_dir)

        candidate_paths.append(
            os.path.join(project_dir, "padel_unique_descriptors.txt")
        )

    for path in candidate_paths:
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                names = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]

            if names:
                return sorted(set(names))

        except Exception:
            continue

    return []

def qspr_compute_descriptor_lists():
    """
    Создаёт списки дескрипторов:
    - rdkit_all;
    - mordred_unique;
    - padel_all;
    - padel_unique.

    Важно:
    - padel_all нужен для режима "🎯 Максимальная точность";
    - padel_unique нужен для режима "⚡ Умный".
    """
    result = {}

    rdkit_all = [
        name for name, _ in Descriptors._descList
    ]

    result["rdkit_all"] = sorted(rdkit_all)

    rdkit_set = set(rdkit_all)

    mordred_set = set()

    if mordred_available:
        try:
            calc = MordredCalculator(mordred_descriptors)

            for desc in calc.descriptors:
                name = getattr(desc, "name", str(desc))

                if name:
                    mordred_set.add(name)

        except Exception:
            mordred_set = set()

    unique_mordred = sorted(list(mordred_set - rdkit_set))
    result["mordred_unique"] = unique_mordred

    # ------------------------------------------------------------
    # PaDEL: полный список
    # Собираем оба источника:
    # - fingerprints=True  -> PaDEL fingerprints;
    # - fingerprints=False -> PaDEL 1D/2D descriptors.

    padel_fp = []
    padel_desc = []

    if padel_available:
        try:
            test_fp = from_smiles(["CCO"], fingerprints=True)

            if test_fp and len(test_fp) > 0:
                padel_fp = sorted(list(test_fp[0].keys()))

        except Exception:
            padel_fp = []

        try:
            test_desc = from_smiles(["CCO"], fingerprints=False)

            if test_desc and len(test_desc) > 0:
                padel_desc = sorted(list(test_desc[0].keys()))

        except Exception:
            padel_desc = []

    padel_all = sorted(set(padel_fp) | set(padel_desc))

    result["padel_fingerprints"] = padel_fp
    result["padel_1d2d"] = padel_desc
    result["padel_all"] = padel_all

    # ------------------------------------------------------------
    # PaDEL: уникальный список

    padel_unique_from_file = qspr_load_padel_unique_from_file()

    if padel_unique_from_file:
        padel_unique = sorted(padel_unique_from_file)
    else:
        padel_unique = sorted(
            list(
                set(padel_all)
                - rdkit_set
                - set(unique_mordred)
            )
        )

    # Если полный список PaDEL не удалось получить через padelpy,
    # не оставляем padel_all пустым: минимальный полный список = известные уникальные.
    # Тогда интерфейс не ломается, но максимум не будет отличаться до починки padelpy.
    if not padel_all and padel_unique:
        padel_all = sorted(padel_unique)
        result["padel_all"] = padel_all

    result["padel_unique"] = padel_unique

    return result


def qspr_save_descriptor_lists(lists, filename=DESC_LISTS_FILE):
    """
    Сохраняет списки дескрипторов в JSON.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(lists, f, ensure_ascii=False, indent=2)


def qspr_load_descriptor_lists(filename=DESC_LISTS_FILE):
    """
    Загружает descriptor_lists.json.
    """
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def qspr_load_descriptor_meanings(filename=DESC_MEANINGS_FILE):
    """
    Загружает расшифровку дескрипторов.
    """
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}

def qspr_load_model_encyclopedia(
    filename=MODEL_ENCYCLOPEDIA_FILE
):
    """
    Загружает help/model_encyclopedia.json
    """

    if not os.path.exists(filename) and os.path.exists(MODEL_ENCYCLOPEDIA_FALLBACK_FILE):
        filename = MODEL_ENCYCLOPEDIA_FALLBACK_FILE

    if os.path.exists(filename):
        try:
            with open(
                filename,
                "r",
                encoding="utf-8"
            ) as f:

                return json.load(f)

        except Exception:
            return {}

    return {}


def qspr_get_model_help(model_name):
    """
    Возвращает описание модели из энциклопедии.
    """

    encyclopedia = qspr_load_model_encyclopedia()

    lookup_name = get_model_encyclopedia_key(model_name)

    return encyclopedia.get(
        lookup_name,
        {}
    )

# ------------------------------------------------------------------
# Расчёт дескрипторов

def qspr_calc_rdkit_descriptors_filtered(mol, allowed_names=None):
    """
    Расчёт RDKit-дескрипторов.
    """
    desc_dict = {}

    for name, func in Descriptors._descList:
        if allowed_names is not None and name not in allowed_names:
            continue

        try:
            desc_dict[name] = func(mol)
        except Exception:
            desc_dict[name] = np.nan

    return desc_dict


def qspr_calc_mordred_descriptors_filtered(mol, calculator, allowed_names=None):
    """
    Расчёт Mordred-дескрипторов.
    """
    if calculator is None:
        return {}

    try:
        results = calculator(mol)

        desc_dict = {}

        for desc, value in zip(calculator.descriptors, results):
            dname = getattr(desc, "name", str(desc))

            if allowed_names is not None and dname not in allowed_names:
                continue

            desc_dict[dname] = value

        return desc_dict

    except Exception:
        return {}


def qspr_calc_padel_descriptors_filtered(smiles, allowed_names=None):
    """
    Расчёт PaDEL-дескрипторов через padelpy.
    """
    if not padel_available:
        return {}

    try:
        smiles_str = str(smiles).strip()

        if not smiles_str:
            return {}

        padel_dict = {}

        try:
            result_fp = from_smiles([smiles_str], fingerprints=True)

            if result_fp and len(result_fp) > 0:
                padel_dict.update(result_fp[0])

        except Exception:
            pass

        try:
            result_desc = from_smiles([smiles_str], fingerprints=False)

            if result_desc and len(result_desc) > 0:
                padel_dict.update(result_desc[0])

        except Exception:
            pass

        if padel_dict:
            if allowed_names is not None:
                allowed_set = set(allowed_names)

                return {
                    k: v
                    for k, v in padel_dict.items()
                    if k in allowed_set
                }

            return padel_dict

    except Exception:
        return {}

    return {}

# ------------------------------------------------------------------
# xTB quantum descriptors

def qspr_xtb_prepare_3d_mol_from_smiles(
    smiles,
    random_seed=1,
    max_embed_attempts=20,
    optimize_with_rdkit=True
):
    """
    Готовит 3D-структуру для xTB:
    SMILES -> RDKit Mol + H -> ETKDG 3D -> UFF/MMFF оптимизация.
    """
    smiles = str(smiles).strip()

    if not smiles or smiles.lower() in ["nan", "none"]:
        return None, "empty_smiles"

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None, "invalid_smiles"

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)

        embed_status = AllChem.EmbedMolecule(
            mol,
            params
        )

        if embed_status != 0:
            embed_status = AllChem.EmbedMolecule(
                mol,
                randomSeed=int(random_seed)
            )

        if embed_status != 0:
            return None, "3d_embedding_failed"

        if optimize_with_rdkit:
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
                else:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=300)
            except Exception:
                pass

        return mol, "ok"

    except Exception as e:
        return None, f"3d_prepare_error: {e}"


def qspr_xtb_mol_to_numbers_positions(mol):
    """
    Преобразует RDKit Mol с 3D-конформером в atomic numbers и координаты.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None, None

    conf = mol.GetConformer()

    numbers = []
    positions = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)

        numbers.append(atom.GetAtomicNum())
        positions.append([float(pos.x), float(pos.y), float(pos.z)])

    return np.array(numbers, dtype=int), np.array(positions, dtype=float)


def qspr_xtb_homo_lumo_from_result(result):
    """
    Извлекает HOMO, LUMO и gap из результата xTB.
    """
    homo = np.nan
    lumo = np.nan
    gap = np.nan

    try:
        eigenvalues = np.asarray(result.get_orbital_eigenvalues(), dtype=float)
        occupations = np.asarray(result.get_orbital_occupations(), dtype=float)

        if len(eigenvalues) == len(occupations) and len(eigenvalues) > 0:
            occ_idx = np.where(occupations > 1e-6)[0]
            virt_idx = np.where(occupations <= 1e-6)[0]

            if len(occ_idx) > 0:
                homo_idx = int(occ_idx[-1])
                homo = float(eigenvalues[homo_idx])

            if len(virt_idx) > 0:
                lumo_idx = int(virt_idx[0])
                lumo = float(eigenvalues[lumo_idx])

            if np.isfinite(homo) and np.isfinite(lumo):
                gap = float(lumo - homo)

    except Exception:
        pass

    return homo, lumo, gap


def qspr_calc_xtb_descriptors_single(
    smiles,
    method="GFN2-xTB",
    charge=0,
    uhf=0,
    accuracy=1.0,
    electronic_temperature=300.0,
    max_iterations=250,
    random_seed=1,
    optimize_with_rdkit=True
):
    """
    Считает xTB-дескрипторы для одной молекулы.
    """
    base = {
        "xtb_energy_hartree": np.nan,
        "xtb_gradient_norm": np.nan,
        "xtb_charge_min": np.nan,
        "xtb_charge_max": np.nan,
        "xtb_charge_mean": np.nan,
        "xtb_charge_std": np.nan,
        "xtb_dipole_x": np.nan,
        "xtb_dipole_y": np.nan,
        "xtb_dipole_z": np.nan,
        "xtb_dipole_norm": np.nan,
        "xtb_homo_hartree": np.nan,
        "xtb_lumo_hartree": np.nan,
        "xtb_gap_hartree": np.nan,
        "xtb_n_orbitals": np.nan,
        "xtb_status": "",
    }

    if not xtb_python_available:
        base["xtb_status"] = "xtb_python_not_available"
        return base

    try:
        mol3d, prep_status = qspr_xtb_prepare_3d_mol_from_smiles(
            smiles=smiles,
            random_seed=random_seed,
            optimize_with_rdkit=optimize_with_rdkit
        )

        if mol3d is None:
            base["xtb_status"] = prep_status
            return base

        numbers, positions = qspr_xtb_mol_to_numbers_positions(mol3d)

        if numbers is None or positions is None:
            base["xtb_status"] = "no_3d_coordinates"
            return base

        calc = XTBCalculator(
            get_method(method),
            numbers,
            positions,
            charge=float(charge),
            uhf=int(uhf)
        )

        try:
            calc.set_verbosity(VERBOSITY_MUTED)
        except Exception:
            pass

        try:
            calc.set_accuracy(float(accuracy))
        except Exception:
            pass

        try:
            calc.set_electronic_temperature(float(electronic_temperature))
        except Exception:
            pass

        try:
            calc.set_max_iterations(int(max_iterations))
        except Exception:
            pass

        result = calc.singlepoint()

        try:
            base["xtb_energy_hartree"] = float(result.get_energy())
        except Exception:
            pass

        try:
            gradient = np.asarray(result.get_gradient(), dtype=float)
            base["xtb_gradient_norm"] = float(np.linalg.norm(gradient))
        except Exception:
            pass

        try:
            charges = np.asarray(result.get_charges(), dtype=float)

            if charges.size > 0:
                base["xtb_charge_min"] = float(np.nanmin(charges))
                base["xtb_charge_max"] = float(np.nanmax(charges))
                base["xtb_charge_mean"] = float(np.nanmean(charges))
                base["xtb_charge_std"] = float(np.nanstd(charges))
        except Exception:
            pass

        try:
            dipole = np.asarray(result.get_dipole(), dtype=float)

            if dipole.size >= 3:
                base["xtb_dipole_x"] = float(dipole[0])
                base["xtb_dipole_y"] = float(dipole[1])
                base["xtb_dipole_z"] = float(dipole[2])
                base["xtb_dipole_norm"] = float(np.linalg.norm(dipole[:3]))
        except Exception:
            pass

        try:
            homo, lumo, gap = qspr_xtb_homo_lumo_from_result(result)
            base["xtb_homo_hartree"] = homo
            base["xtb_lumo_hartree"] = lumo
            base["xtb_gap_hartree"] = gap
        except Exception:
            pass

        try:
            base["xtb_n_orbitals"] = int(result.get_number_of_orbitals())
        except Exception:
            pass

        base["xtb_status"] = "ok"
        return base

    except Exception as e:
        base["xtb_status"] = f"xtb_error: {e}"
        return base


def qspr_calc_xtb_descriptors_dataframe(
    data,
    smiles_col,
    target_col=None,
    method="GFN2-xTB",
    charge=0,
    uhf=0,
    accuracy=1.0,
    electronic_temperature=300.0,
    max_iterations=250,
    random_seed=1,
    optimize_with_rdkit=True,
    max_molecules=None
):
    """
    Считает xTB-дескрипторы для таблицы.
    """
    if data is None or data.empty:
        raise ValueError("Нет данных для расчёта xTB-дескрипторов.")

    if smiles_col not in data.columns:
        raise ValueError(f"Колонка SMILES не найдена: {smiles_col}")

    work = data.copy().reset_index(drop=False).rename(columns={"index": "_original_index"})

    if max_molecules is not None:
        work = work.head(int(max_molecules)).copy()

    rows = []

    for local_i, row in work.iterrows():
        smiles = str(row.get(smiles_col, "")).strip()

        desc = qspr_calc_xtb_descriptors_single(
            smiles=smiles,
            method=method,
            charge=charge,
            uhf=uhf,
            accuracy=accuracy,
            electronic_temperature=electronic_temperature,
            max_iterations=max_iterations,
            random_seed=random_seed + int(local_i),
            optimize_with_rdkit=optimize_with_rdkit
        )

        out = {
            "_original_index": int(row.get("_original_index", local_i)),
            "SMILES": smiles,
        }

        if target_col is not None and target_col in row.index:
            out[target_col] = row.get(target_col)

        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                out["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)

                try:
                    out["inchikey"] = Chem.MolToInchiKey(mol)
                except Exception:
                    out["inchikey"] = ""
            else:
                out["canonical_smiles"] = ""
                out["inchikey"] = ""

        except Exception:
            out["canonical_smiles"] = ""
            out["inchikey"] = ""

        out.update(desc)
        rows.append(out)

    df_desc = pd.DataFrame(rows)

    status_counts = (
        df_desc["xtb_status"]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
        if "xtb_status" in df_desc.columns
        else {}
    )

    descriptor_cols = [
        c for c in df_desc.columns
        if c.startswith("xtb_") and c != "xtb_status"
    ]

    report = {
        "total": len(df_desc),
        "ok": int((df_desc["xtb_status"] == "ok").sum()) if "xtb_status" in df_desc.columns else 0,
        "failed": int((df_desc["xtb_status"] != "ok").sum()) if "xtb_status" in df_desc.columns else len(df_desc),
        "status_counts": status_counts,
        "descriptor_cols": descriptor_cols,
        "method": method,
    }

    return df_desc, report

def qspr_descriptor_mode_settings(mode, desc_lists=None):
    """
    Возвращает настройки расчёта дескрипторов по режиму.

    Поддерживаемые режимы:
    - "rdkit_fast"
    - "mordred"
    - "mordred_padel_unique"
    - "max_accuracy"

    Также поддерживаются старые русские названия из интерфейса.
    """
    if desc_lists is None:
        desc_lists = qspr_load_descriptor_lists()

    if desc_lists:
        rdkit_all_set = set(desc_lists.get("rdkit_all", []))
        mordred_unique_set = set(desc_lists.get("mordred_unique", []))
        padel_unique_set = set(desc_lists.get("padel_unique", []))
        padel_all_set = set(desc_lists.get("padel_all", []))
    else:
        rdkit_all_set = set()
        mordred_unique_set = set()
        padel_unique_set = set()
        padel_all_set = set()

    mode_map = {
        # Русский
        "🚀 Максимальная скорость (RDKit)": "rdkit_fast",
        "👁️‍🗨️ Расширенный (Mordred)": "mordred",
        "⚡ Умный (Mordred + уникальные PaDEL)": "mordred_padel_unique",
        "🎯 Максимальная точность": "max_accuracy",
        
        # Английский (из вашего интерфейса)
        "🚀 Maximum speed (RDKit)": "rdkit_fast",
        "👁️‍🗨️ Extended (Mordred)": "mordred",
        "⚡ Smart (Mordred + unique PaDEL)": "mordred_padel_unique",
        "🎯 Maximum accuracy": "max_accuracy",
        
        # Казахский (добавьте ваши строки, если они отличаются)
        "🚀 Максималды жылдамдық (RDKit)": "rdkit_fast",
        "👁️‍🗨️ Кеңейтілген (Mordred)": "mordred",
        "⚡ Ақылды (Mordred + бірегей PaDEL)": "mordred_padel_unique",
        "🎯 Максималды дәлдік": "max_accuracy",
    }

    mode = mode_map.get(mode, mode)

    if mode == "rdkit_fast":
        return {
            "use_rdkit": True,
            "use_mordred": False,
            "use_padel": False,
            "rdkit_names": rdkit_all_set if rdkit_all_set else None,
            "mordred_names": set(),
            "padel_names": set()
        }

    if mode == "mordred":
        return {
            "use_rdkit": True,
            "use_mordred": True,
            "use_padel": False,
            "rdkit_names": rdkit_all_set if rdkit_all_set else None,
            "mordred_names": None,
            "padel_names": set()
        }

    if mode == "mordred_padel_unique":
        return {
            "use_rdkit": True,
            "use_mordred": True,
            "use_padel": True,
            "rdkit_names": rdkit_all_set if rdkit_all_set else None,
            "mordred_names": mordred_unique_set,
            "padel_names": padel_unique_set
        }

    if mode == "max_accuracy":
        return {
            "use_rdkit": True,
            "use_mordred": True,
            "use_padel": True,
            "rdkit_names": None,
            "mordred_names": None,
            "padel_names": padel_all_set if padel_all_set else None
        }

    raise ValueError(f"Неизвестный режим дескрипторов: {mode}")


def qspr_calculate_molecular_descriptors(
    data,
    smiles_col="SMILES",
    target_col=None,
    mode="mordred",
    desc_lists=None,
    allowed_rdkit_names=None,
    allowed_mordred_names=None,
    allowed_padel_names=None,
    preserve_constant_columns=False
):
    """
    Рассчитывает молекулярные дескрипторы для датасета.

    Возвращает словарь:
    {
        "df_desc": DataFrame,
        "X_all": ndarray,
        "y_all": ndarray или None,
        "valid_indices": list,
        "desc_names": list,
        "report": dict
    }
    """
    if smiles_col not in data.columns:
        raise ValueError(f"Не найдена колонка SMILES: {smiles_col}")

    settings = qspr_descriptor_mode_settings(
        mode=mode,
        desc_lists=desc_lists
    )

    def _apply_allowed_names(current_names, allowed_names):
        if allowed_names is None:
            return current_names

        allowed_set = set(allowed_names)

        if current_names is None:
            return allowed_set

        return set(current_names) & allowed_set

    settings["rdkit_names"] = _apply_allowed_names(
        settings.get("rdkit_names"),
        allowed_rdkit_names
    )

    settings["mordred_names"] = _apply_allowed_names(
        settings.get("mordred_names"),
        allowed_mordred_names
    )

    settings["padel_names"] = _apply_allowed_names(
        settings.get("padel_names"),
        allowed_padel_names
    )

    if allowed_rdkit_names is not None and len(settings["rdkit_names"]) == 0:
        settings["use_rdkit"] = False

    if allowed_mordred_names is not None and len(settings["mordred_names"]) == 0:
        settings["use_mordred"] = False

    if allowed_padel_names is not None and len(settings["padel_names"]) == 0:
        settings["use_padel"] = False

    if settings["use_mordred"] and not mordred_available:
        raise ValueError("Mordred недоступен. Установите пакет mordred.")

    if settings["use_padel"] and not padel_available:
        raise ValueError("PaDEL недоступен. Установите пакет padelpy.")

    mordred_calc = None

    if settings["use_mordred"]:
        mordred_calc = MordredCalculator(mordred_descriptors)

    all_desc = []
    valid_indices = []

    invalid_smiles_count = 0
    padel_error_count = 0

    smiles_list = data[smiles_col].astype(str).tolist()

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles).strip())

        if mol is None:
            invalid_smiles_count += 1
            continue

        desc_combined = {}

        if settings["use_rdkit"]:
            desc_combined.update(
                qspr_calc_rdkit_descriptors_filtered(
                    mol,
                    allowed_names=settings["rdkit_names"]
                )
            )

        if settings["use_mordred"]:
            desc_combined.update(
                qspr_calc_mordred_descriptors_filtered(
                    mol,
                    mordred_calc,
                    allowed_names=settings["mordred_names"]
                )
            )

        if settings["use_padel"]:
            padel_dict = qspr_calc_padel_descriptors_filtered(
                smiles,
                allowed_names=settings["padel_names"]
            )

            if not padel_dict:
                padel_error_count += 1

            desc_combined.update(padel_dict)

        all_desc.append(desc_combined)
        valid_indices.append(idx)

    if not all_desc:
        raise ValueError("Нет валидных молекул для расчёта дескрипторов.")

    df_desc_raw = pd.DataFrame(all_desc)

    if preserve_constant_columns:
        df_desc = df_desc_raw.copy()
        for col in df_desc.columns:
            df_desc[col] = pd.to_numeric(df_desc[col], errors="coerce")
        df_desc = df_desc.replace([np.inf, -np.inf], np.nan)
        df_desc = df_desc.dropna(axis=1, how="all")
        for col in df_desc.columns:
            median_value = df_desc[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            df_desc[col] = df_desc[col].fillna(median_value)
    else:
        df_desc = qspr_clean_numeric_dataframe(df_desc_raw)

    if df_desc.empty:
        raise ValueError("После очистки не осталось числовых дескрипторов.")

    X_all = df_desc.values.astype(float)
    desc_names = df_desc.columns.tolist()

    if target_col is not None:
        if target_col not in data.columns:
            raise ValueError(f"Целевое свойство не найдено: {target_col}")

        y_series = qspr_to_numeric(data[target_col])
        y_all = y_series.iloc[valid_indices].values.astype(float)

        valid_y = np.isfinite(y_all)

        if not valid_y.all():
            df_desc = df_desc.loc[valid_y].reset_index(drop=True)
            X_all = df_desc.values.astype(float)
            y_all = y_all[valid_y]
            valid_indices = [
                idx for idx, keep in zip(valid_indices, valid_y)
                if keep
            ]
            desc_names = df_desc.columns.tolist()

    else:
        y_all = None

    report = {
        "mode": mode,
        "n_input_rows": len(data),
        "n_valid_molecules": len(valid_indices),
        "n_descriptors": len(desc_names),
        "invalid_smiles_count": invalid_smiles_count,
        "padel_error_count": padel_error_count,
    }

    return {
        "df_desc": df_desc.reset_index(drop=True),
        "X_all": X_all,
        "y_all": y_all,
        "valid_indices": valid_indices,
        "desc_names": desc_names,
        "report": report
    }


def qspr_prepare_custom_descriptors_from_file(
    data,
    target_col,
    descriptor_cols,
    smiles_col="SMILES"
):
    """
    Использует уже рассчитанные пользователем дескрипторы из загруженного файла.

    Возвращает:
    {
        "df_desc": DataFrame,
        "X_all": ndarray,
        "y_all": ndarray,
        "valid_indices": list,
        "desc_names": list,
        "report": dict
    }
    """
    if not descriptor_cols:
        raise ValueError("Не выбраны колонки с дескрипторами.")

    if target_col not in data.columns:
        raise ValueError(f"Не найдена колонка свойства: {target_col}")

    work = data.copy()

    work[target_col] = qspr_to_numeric(work[target_col])

    for col in descriptor_cols:
        if col not in work.columns:
            raise ValueError(f"Колонка-дескриптор не найдена: {col}")

        work[col] = qspr_to_numeric(work[col])

    valid_mask = work[target_col].notna()
    valid_mask = valid_mask & work[descriptor_cols].notna().any(axis=1)

    work_valid = work.loc[valid_mask].copy()

    if work_valid.empty:
        raise ValueError(
            "После очистки не осталось строк с валидным свойством и дескрипторами."
        )

    df_desc = work_valid[descriptor_cols].copy()
    df_desc = qspr_clean_numeric_dataframe(df_desc)

    if df_desc.empty:
        raise ValueError(
            "После очистки выбранных дескрипторов не осталось числовых признаков."
        )

    y_all = work_valid[target_col].values.astype(float)
    X_all = df_desc.values.astype(float)

    desc_names = df_desc.columns.tolist()
    valid_indices = work_valid.index.tolist()

    report = {
        "source": "custom_descriptors",
        "n_input_rows": len(data),
        "n_valid_rows": len(valid_indices),
        "n_descriptors": len(desc_names),
    }

    return {
        "df_desc": df_desc.reset_index(drop=True),
        "X_all": X_all,
        "y_all": y_all,
        "valid_indices": valid_indices,
        "desc_names": desc_names,
        "report": report
    }




# ------------------------------------------------------------------
# Простая рабочая symbolic regression модель для пункта GEP

class QSPRSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Лёгкая sklearn-совместимая symbolic regression модель.

    Это практический fallback для пункта "GEP Symbolic Regression" без тяжёлых
    внешних зависимостей. Модель эволюционно ищет короткую математическую
    формулу из дескрипторов и затем калибрует её линейно:

        y = scale * expression(X) + offset

    Реализация намеренно компактная и безопасная для Streamlit:
    - ограничивает эффективную популяцию и глубину выражений;
    - использует защищённое деление и защищённые функции;
    - штрафует слишком длинные формулы.
    """

    def __init__(
        self,
        population_size=500,
        generations=20,
        max_depth=4,
        tournament_size=5,
        mutation_rate=0.35,
        crossover_rate=0.45,
        parsimony=0.001,
        random_state=42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.parsimony = parsimony
        self.random_state = random_state

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def _safe_div(self, a, b):
        return a / np.where(np.abs(b) < 1e-8, 1.0, b)

    def _safe_log(self, a):
        return np.log1p(np.abs(a))

    def _safe_sqrt(self, a):
        return np.sqrt(np.abs(a))

    def _random_expr(self, rng, depth):
        if depth <= 0 or rng.random() < 0.25:
            if rng.random() < 0.75:
                return ("var", int(rng.integers(0, self.n_features_in_)))
            return ("const", float(rng.normal(0, 1)))

        unary_ops = ["sin", "cos", "log", "sqrt"]
        binary_ops = ["add", "sub", "mul", "div"]

        if rng.random() < 0.35:
            op = str(rng.choice(unary_ops))
            return (op, self._random_expr(rng, depth - 1))

        op = str(rng.choice(binary_ops))
        return (
            op,
            self._random_expr(rng, depth - 1),
            self._random_expr(rng, depth - 1),
        )

    def _eval_expr(self, expr, X):
        op = expr[0]

        if op == "var":
            idx = int(expr[1])
            return X[:, idx]

        if op == "const":
            return np.full(X.shape[0], float(expr[1]))

        if op in ["add", "sub", "mul", "div"]:
            a = self._eval_expr(expr[1], X)
            b = self._eval_expr(expr[2], X)

            if op == "add":
                out = a + b
            elif op == "sub":
                out = a - b
            elif op == "mul":
                out = a * b
            else:
                out = self._safe_div(a, b)

        else:
            a = self._eval_expr(expr[1], X)
            if op == "sin":
                out = np.sin(a)
            elif op == "cos":
                out = np.cos(a)
            elif op == "log":
                out = self._safe_log(a)
            elif op == "sqrt":
                out = self._safe_sqrt(a)
            else:
                out = np.zeros(X.shape[0])

        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def _size(self, expr):
        op = expr[0]
        if op in ["var", "const"]:
            return 1
        if op in ["sin", "cos", "log", "sqrt"]:
            return 1 + self._size(expr[1])
        return 1 + self._size(expr[1]) + self._size(expr[2])

    def _calibrate(self, z, y):
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.nanstd(z) < 1e-12:
            return 0.0, float(np.nanmean(y)), np.full_like(y, float(np.nanmean(y)))

        A = np.column_stack([z, np.ones(len(z))])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        scale = float(coef[0])
        offset = float(coef[1])
        pred = scale * z + offset
        return scale, offset, pred

    def _fitness(self, expr, X, y):
        try:
            z = self._eval_expr(expr, X)
            scale, offset, pred = self._calibrate(z, y)
            mse = float(np.mean((y - pred) ** 2))
            penalty = float(self.parsimony) * self._size(expr)
            if not np.isfinite(mse):
                mse = 1e12
            return mse + penalty, scale, offset
        except Exception:
            return 1e12, 0.0, float(np.mean(y))

    def _all_paths(self, expr, prefix=()):
        paths = [prefix]
        op = expr[0]
        if op in ["sin", "cos", "log", "sqrt"]:
            paths.extend(self._all_paths(expr[1], prefix + (1,)))
        elif op in ["add", "sub", "mul", "div"]:
            paths.extend(self._all_paths(expr[1], prefix + (1,)))
            paths.extend(self._all_paths(expr[2], prefix + (2,)))
        return paths

    def _get_subtree(self, expr, path):
        node = expr
        for idx in path:
            node = node[idx]
        return node

    def _replace_subtree(self, expr, path, subtree):
        if not path:
            return subtree
        idx = path[0]
        expr_list = list(expr)
        expr_list[idx] = self._replace_subtree(expr_list[idx], path[1:], subtree)
        return tuple(expr_list)

    def _mutate(self, expr, rng):
        path = self._all_paths(expr)[int(rng.integers(0, len(self._all_paths(expr))))]
        subtree = self._random_expr(rng, max(1, int(self.max_depth) // 2))
        return self._replace_subtree(expr, path, subtree)

    def _crossover(self, a, b, rng):
        paths_a = self._all_paths(a)
        paths_b = self._all_paths(b)
        path_a = paths_a[int(rng.integers(0, len(paths_a)))]
        path_b = paths_b[int(rng.integers(0, len(paths_b)))]
        subtree_b = self._get_subtree(b, path_b)
        return self._replace_subtree(a, path_a, subtree_b)

    def _tournament(self, population, fitnesses, rng):
        k = min(int(self.tournament_size), len(population))
        ids = rng.choice(len(population), size=k, replace=False)
        best_id = min(ids, key=lambda i: fitnesses[int(i)][0])
        return population[int(best_id)]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X должен быть двумерной матрицей.")

        self.n_features_in_ = X.shape[1]
        self.y_mean_ = float(np.mean(y))

        rng = self._rng()
        population_size = int(max(20, min(int(self.population_size), 1000)))
        generations = int(max(1, min(int(self.generations), 100)))
        max_depth = int(max(1, min(int(self.max_depth), 6)))
        self.max_depth = max_depth

        population = [
            self._random_expr(rng, max_depth)
            for _ in range(population_size)
        ]

        best_expr = population[0]
        best_score, best_scale, best_offset = self._fitness(best_expr, X, y)

        for _ in range(generations):
            fitnesses = [self._fitness(expr, X, y) for expr in population]

            current_best_idx = int(np.argmin([f[0] for f in fitnesses]))
            current_score, current_scale, current_offset = fitnesses[current_best_idx]

            if current_score < best_score:
                best_score = current_score
                best_expr = population[current_best_idx]
                best_scale = current_scale
                best_offset = current_offset

            new_population = [best_expr]

            while len(new_population) < population_size:
                parent = self._tournament(population, fitnesses, rng)
                child = parent

                if rng.random() < float(self.crossover_rate):
                    other = self._tournament(population, fitnesses, rng)
                    child = self._crossover(child, other, rng)

                if rng.random() < float(self.mutation_rate):
                    child = self._mutate(child, rng)

                new_population.append(child)

            population = new_population

        self.expression_ = best_expr
        self.scale_ = float(best_scale)
        self.offset_ = float(best_offset)
        self.best_score_ = float(best_score)
        self.expression_string_ = self._expr_to_string(best_expr)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "expression_"):
            raise ValueError("Модель ещё не обучена.")
        z = self._eval_expr(self.expression_, X)
        return self.scale_ * z + self.offset_

    def _expr_to_string(self, expr, feature_names=None):
        op = expr[0]
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(getattr(self, "n_features_in_", 0))]

        if op == "var":
            idx = int(expr[1])
            if 0 <= idx < len(feature_names):
                return str(feature_names[idx])
            return f"x{idx}"

        if op == "const":
            return f"{float(expr[1]):.4g}"

        if op in ["add", "sub", "mul", "div"]:
            a = self._expr_to_string(expr[1], feature_names)
            b = self._expr_to_string(expr[2], feature_names)
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
            return f"({a} {symbol} {b})"

        a = self._expr_to_string(expr[1], feature_names)
        return f"{op}({a})"

    def get_formula(self, feature_names=None):
        if not hasattr(self, "expression_"):
            return ""
        expr = self._expr_to_string(self.expression_, feature_names)
        return f"y = {self.scale_:.6g} * ({expr}) + {self.offset_:.6g}"


# ------------------------------------------------------------------
# Модели

def qspr_default_model_params():
    """
    Параметры моделей по умолчанию.
    """
    return {
        "pls_components": 2,
        "ridge_alpha": 1.0,
        "lasso_alpha": 0.01,
        "elastic_alpha": 0.01,
        "elastic_l1_ratio": 0.5,

        "rf_n_estimators": 300,

        "xgb_n_estimators": 300,
        "xgb_learning_rate": 0.05,
        "xgb_max_depth": 4,

        "lightgbm_n_estimators": 300,
        "lightgbm_learning_rate": 0.05,
        "lightgbm_num_leaves": 31,

        "catboost_iterations": 300,
        "catboost_learning_rate": 0.05,
        "catboost_depth": 6,
        
        "cart_max_depth": 5,
        "cart_min_samples_leaf": 2,

        "mars_degree": 2,
        "mars_alpha": 1.0,

        "gep_population_size": 500,
        "gep_generations": 20,
        "gep_max_depth": 4,

        "spline_n_knots": 5,
        "spline_degree": 3,
        "spline_alpha": 1.0,

        "gam_n_splines": 6,
        "gam_degree": 3,
        "gam_alpha": 1.0,

        "gp_population_size": 500,
        "gp_generations": 20,
        "gp_max_depth": 4,

        "pysr_niterations": 40,
        "pysr_populations": 8,
        "pysr_maxsize": 20,

        "stacking_cv": 5,
        "stacking_passthrough": False,
        "voting_rf_weight": 1.0,
        "voting_extra_trees_weight": 1.0,
        "voting_ridge_weight": 1.0,
        
        "svr_c": 10.0,
        "svr_epsilon": 0.1,
        "svr_gamma": "scale",

        "gpr_alpha": 0.000001,
        "gpr_length_scale": 1.0,
        "gpr_noise_level": 0.1,

        "knn_n_neighbors": 5,
        "knn_weights": "distance",
        
        "et_n_estimators": 300,
        "et_max_depth": None,
        "et_min_samples_split": 2,
        "et_min_samples_leaf": 1,
        "et_max_features": "sqrt",
        
        "mlp_hidden_layer_sizes": "100,50",
        "mlp_activation": "relu",
        "mlp_alpha": 0.0001,
        "mlp_learning_rate_init": 0.001,
        "mlp_max_iter": 2000,
        
        "adaboost_n_estimators": 300,
        "adaboost_learning_rate": 1.0,
        
        "hgb_max_iter": 300,
        "hgb_learning_rate": 0.1,
        "hgb_max_depth": None,
        "hgb_min_samples_leaf": 20,
        "hgb_l2_regularization": 0.0,
    }


def qspr_create_regression_model(
    model_name,
    n_samples=None,
    n_features=None,
    params=None
):
    """
    Создаёт модель регрессии по названию.

    model_name:
    - Random Forest
    - Множественная линейная регрессия (MLR)
    - PLS Regression
    - Ridge
    - LASSO
    - Elastic Net
    - XGBoost
    - LightGBM
    - CatBoost
    - Stacking
    """
    model_name = normalize_runtime_name(model_name)
    p = qspr_default_model_params()

    if params:
        p.update(params)

    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=int(p["rf_n_estimators"]),
            random_state=42,
            n_jobs=-1
        )
    
    if model_name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=int(p["et_n_estimators"]),
            max_depth=p["et_max_depth"] if p["et_max_depth"] is not None else None,
            min_samples_split=int(p["et_min_samples_split"]),
            min_samples_leaf=int(p["et_min_samples_leaf"]),
            max_features=p["et_max_features"],
            random_state=42,
            n_jobs=-1
        )
    
    if model_name == "Множественная линейная регрессия (MLR)":
        return LinearRegression()

    if model_name == "PLS Regression":
        n_comp = int(p["pls_components"])

        if n_samples is not None and n_features is not None:
            n_comp = min(n_comp, n_samples - 1, n_features)
            n_comp = max(1, n_comp)

        return PLSRegression(
            n_components=n_comp
        )

    if model_name == "Ridge":
        return Ridge(
            alpha=float(p["ridge_alpha"])
        )

    if model_name == "LASSO":
        return Lasso(
            alpha=float(p["lasso_alpha"]),
            max_iter=10000,
            random_state=42
        )

    if model_name == "Elastic Net":
        return ElasticNet(
            alpha=float(p["elastic_alpha"]),
            l1_ratio=float(p["elastic_l1_ratio"]),
            max_iter=10000,
            random_state=42
        )

    if model_name == "XGBoost":
        if not xgboost_available:
            raise ValueError("XGBoost недоступен. Установите пакет xgboost.")

        return xgb.XGBRegressor(
            n_estimators=int(p["xgb_n_estimators"]),
            learning_rate=float(p["xgb_learning_rate"]),
            max_depth=int(p["xgb_max_depth"]),
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror"
        )

    if model_name == "LightGBM":
        if not lightgbm_available:
            raise ValueError("LightGBM недоступен. Установите пакет lightgbm.")

        return lgb.LGBMRegressor(
            n_estimators=int(p["lightgbm_n_estimators"]),
            learning_rate=float(p["lightgbm_learning_rate"]),
            num_leaves=int(p["lightgbm_num_leaves"]),
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

    if model_name == "CatBoost":
        if not catboost_available:
            raise ValueError("CatBoost недоступен. Установите пакет catboost.")

        return CatBoostRegressor(
            iterations=int(p["catboost_iterations"]),
            learning_rate=float(p["catboost_learning_rate"]),
            depth=int(p["catboost_depth"]),
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )

    if model_name == "SVR":
        return SVR(
            kernel="rbf",
            C=float(p["svr_c"]),
            epsilon=float(p["svr_epsilon"]),
            gamma=p["svr_gamma"]
        )

    if model_name == "Gaussian Process Regression (GPR)":
        length_scale = float(p["gpr_length_scale"])
        noise_level = float(p["gpr_noise_level"])

        kernel = (
            ConstantKernel(1.0, constant_value_bounds="fixed")
            * RBF(length_scale=length_scale)
            + WhiteKernel(noise_level=noise_level)
        )

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(p["gpr_alpha"]),
            normalize_y=True,
            random_state=42,
            n_restarts_optimizer=2
        )

    if model_name == "KNN Regression":
        n_neighbors = int(p["knn_n_neighbors"])

        if n_samples is not None:
            n_neighbors = min(n_neighbors, max(1, n_samples))

        return KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=p["knn_weights"],
            metric="minkowski"
        )

    if model_name == "MLP Regression":
        hidden_raw = str(p["mlp_hidden_layer_sizes"]).strip()

        try:
            hidden_layer_sizes = tuple(
                int(x.strip())
                for x in hidden_raw.split(",")
                if x.strip()
            )
        except Exception:
            hidden_layer_sizes = (100, 50)

        if not hidden_layer_sizes:
            hidden_layer_sizes = (100, 50)

        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=str(p["mlp_activation"]),
            solver="adam",
            alpha=float(p["mlp_alpha"]),
            learning_rate_init=float(p["mlp_learning_rate_init"]),
            max_iter=int(p["mlp_max_iter"]),
            random_state=42,
            early_stopping=True
        )

    if model_name == "CART Regression":
        return DecisionTreeRegressor(
            max_depth=int(p["cart_max_depth"]),
            min_samples_leaf=int(p["cart_min_samples_leaf"]),
            random_state=42
        )

    if model_name == "MARS-like Regression":
        return make_pipeline(
            PolynomialFeatures(
                degree=int(p["mars_degree"]),
                include_bias=False
            ),
            Ridge(
                alpha=float(p["mars_alpha"])
            )
        )

    if model_name == "Spline Regression":
        return make_pipeline(
            SplineTransformer(
                n_knots=int(p["spline_n_knots"]),
                degree=int(p["spline_degree"]),
                include_bias=False
            ),
            Ridge(alpha=float(p["spline_alpha"]))
        )

    if model_name == "GAM Regression":
        return make_pipeline(
            SplineTransformer(
                n_knots=int(p["gam_n_splines"]),
                degree=int(p["gam_degree"]),
                include_bias=False
            ),
            Ridge(alpha=float(p["gam_alpha"]))
        )

    if model_name == "GEP Symbolic Regression":
        return QSPRSymbolicRegressor(
            population_size=int(p["gep_population_size"]),
            generations=int(p["gep_generations"]),
            max_depth=int(p.get("gep_max_depth", 4)),
            random_state=42
        )

    if model_name == "Genetic Programming Regression":
        return QSPRSymbolicRegressor(
            population_size=int(p["gp_population_size"]),
            generations=int(p["gp_generations"]),
            max_depth=int(p["gp_max_depth"]),
            random_state=42
        )

    if model_name == "PySR":
        if not pysr_available:
            raise ValueError(
                "PySR недоступен. Установите пакет pysr и настройте Julia."
            )

        return PySRRegressor(
            niterations=int(p["pysr_niterations"]),
            populations=int(p["pysr_populations"]),
            maxsize=int(p["pysr_maxsize"]),
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp"],
            model_selection="best",
            random_state=42,
            parallelism="serial",
            progress=False,
            verbosity=0
        )

    if model_name == "AdaBoost Regressor":
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor(
            n_estimators=int(p["adaboost_n_estimators"]),
            learning_rate=float(p["adaboost_learning_rate"]),
            random_state=42
        )

    if model_name == "HistGradientBoosting Regressor":
        return HistGradientBoostingRegressor(
            max_iter=int(p.get("hgb_max_iter", 300)),
            learning_rate=float(p.get("hgb_learning_rate", 0.1)),
            max_depth=int(p.get("hgb_max_depth", 0)) if p.get("hgb_max_depth") is not None else None,
            min_samples_leaf=int(p.get("hgb_min_samples_leaf", 20)),
            l2_regularization=float(p.get("hgb_l2_regularization", 0.0)),
            random_state=42
        )
    
    if model_name == "Stacking":
        estimators = []

        estimators.append(
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=int(p["rf_n_estimators"]),
                    random_state=42,
                    n_jobs=-1
                )
            )
        )

        if xgboost_available:
            estimators.append(
                (
                    "xgb",
                    xgb.XGBRegressor(
                        n_estimators=int(p["xgb_n_estimators"]),
                        learning_rate=float(p["xgb_learning_rate"]),
                        max_depth=int(p["xgb_max_depth"]),
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        objective="reg:squarederror"
                    )
                )
            )

        if lightgbm_available:
            estimators.append(
                (
                    "lgbm",
                    lgb.LGBMRegressor(
                        n_estimators=int(p["lightgbm_n_estimators"]),
                        learning_rate=float(p["lightgbm_learning_rate"]),
                        num_leaves=int(p["lightgbm_num_leaves"]),
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1
                    )
                )
            )

        if catboost_available:
            estimators.append(
                (
                    "cat",
                    CatBoostRegressor(
                        iterations=int(p["catboost_iterations"]),
                        learning_rate=float(p["catboost_learning_rate"]),
                        depth=int(p["catboost_depth"]),
                        loss_function="RMSE",
                        random_seed=42,
                        verbose=False
                    )
                )
            )

        if len(estimators) < 2:
            raise ValueError(
                "Для Stacking нужно минимум две базовые модели. "
                "Установите xgboost, lightgbm или catboost."
            )

        cv_value = int(p["stacking_cv"])

        if n_samples is not None:
            cv_value = min(cv_value, max(2, n_samples - 1))

        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=cv_value,
            passthrough=bool(p["stacking_passthrough"]),
            n_jobs=-1
        )

    if model_name == "Voting Regressor":
        estimators = [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=int(p["rf_n_estimators"]),
                    random_state=42,
                    n_jobs=-1
                )
            ),
            (
                "extra",
                ExtraTreesRegressor(
                    n_estimators=int(p["et_n_estimators"]),
                    max_depth=p["et_max_depth"],
                    min_samples_split=int(p["et_min_samples_split"]),
                    min_samples_leaf=int(p["et_min_samples_leaf"]),
                    max_features=p["et_max_features"],
                    random_state=42,
                    n_jobs=-1
                )
            ),
            ("ridge", Ridge(alpha=float(p["ridge_alpha"]))),
        ]
        weights = [
            float(p["voting_rf_weight"]),
            float(p["voting_extra_trees_weight"]),
            float(p["voting_ridge_weight"]),
        ]
        if sum(weights) <= 0:
            raise ValueError(
                "Хотя бы один вес Voting Regressor должен быть больше нуля."
            )
        return VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=-1
        )

    raise ValueError(f"Неизвестная модель: {model_name}")

def qspr_get_param_grid(model_name):
    """
    Небольшие безопасные сетки гиперпараметров.
    Сделаны компактными, чтобы не зависать на маленьких QSPR-выборках.
    """
    model_name = normalize_runtime_name(model_name)

    if model_name == "Random Forest":
        return {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_leaf": [1, 2, 3],
        }

    if model_name == "Extra Trees":
        return {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 3, 5],
            "model__min_samples_leaf": [1, 2, 3],
        }

    if model_name == "Ridge":
        return {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        }

    if model_name == "LASSO":
        return {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        }

    if model_name == "Elastic Net":
        return {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    if model_name == "SVR":
        return {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            "model__gamma": ["scale", "auto"],
        }

    if model_name == "KNN Regression":
        return {
            "model__n_neighbors": [2, 3, 5, 7, 10],
            "model__weights": ["uniform", "distance"],
        }

    if model_name == "PLS Regression":
        return {
            "model__n_components": [1, 2, 3, 4, 5, 8, 10],
        }

    if model_name == "MLP Regression":
        return {
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.0005, 0.001, 0.005],
        }

    if model_name == "Spline Regression":
        return {
            "model__splinetransformer__n_knots": [3, 5, 7],
            "model__splinetransformer__degree": [2, 3],
            "model__ridge__alpha": [0.01, 0.1, 1.0, 10.0],
        }

    if model_name == "GAM Regression":
        return {
            "model__splinetransformer__n_knots": [4, 6, 8],
            "model__splinetransformer__degree": [2, 3],
            "model__ridge__alpha": [0.1, 1.0, 10.0, 100.0],
        }

    if model_name == "AdaBoost Regressor":
        return {
            "model__n_estimators": [50, 100, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        }
    
    if model_name == "HistGradientBoosting Regressor":
        return {
            "model__max_iter": [100, 300, 500, 1000],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.3],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_leaf": [10, 20, 30],
            "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
        }    
        
    return {}

# ------------------------------------------------------------------
# Полный отбор дескрипторов для QSPR

class QSPRDescriptorSelector(BaseEstimator, TransformerMixin):
    """
    Универсальный селектор дескрипторов для QSPR.

    Логика:
    1. числовая очистка;
    2. удаление константных признаков;
    3. удаление сильно коррелирующих признаков;
    4. финальный отбор одним из методов:
       - none;
       - fast;
       - f_regression;
       - mutual_info;
       - lasso;
       - random_forest;
       - rfe_ridge.

    Важно:
    - трансформер сохраняет индексы выбранных дескрипторов;
    - может быть частью sklearn Pipeline;
    - модель после обучения принимает исходную полную X-матрицу.
    """

    def __init__(
        self,
        desc_names=None,
        method="fast",
        max_features=50,
        remove_constant=True,
        remove_correlated=True,
        corr_threshold=0.95,
        lasso_alpha=0.01,
        rf_n_estimators=300,
        rfe_step=0.2,
        random_state=42
    ):
        self.desc_names = desc_names
        self.method = method
        self.max_features = max_features
        self.remove_constant = remove_constant
        self.remove_correlated = remove_correlated
        self.corr_threshold = corr_threshold
        self.lasso_alpha = lasso_alpha
        self.rf_n_estimators = rf_n_estimators
        self.rfe_step = rfe_step
        self.random_state = random_state

    def _normalize_method(self):
        method = str(self.method).strip().lower()

        aliases = {
            "без отбора": "none",
            "none": "none",
            "no_selection": "none",
            "быстрый отбор": "fast",
            "fast": "fast",
            "f_regression": "f_regression",
            "selectkbest_f": "f_regression",
            "mutual_info": "mutual_info",
            "mutual information": "mutual_info",
            "lasso": "lasso",
            "lasso selection": "lasso",
            "random_forest": "random_forest",
            "rf_importance": "random_forest",
            "random forest importance": "random_forest",
            "rfe": "rfe_ridge",
            "rfe_ridge": "rfe_ridge",
            "rfe + ridge": "rfe_ridge",
        }

        return aliases.get(method, method)

    def _safe_numeric_matrix(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        X = X.copy()
        X = np.where(np.isfinite(X), X, np.nan)

        if fit:
            medians = np.nanmedian(X, axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            self.medians_ = medians
        else:
            medians = getattr(self, "medians_", None)

            if medians is None:
                medians = np.nanmedian(X, axis=0)
                medians = np.where(np.isfinite(medians), medians, 0.0)

        inds = np.where(np.isnan(X))
        if len(inds[0]) > 0:
            X[inds] = np.take(medians, inds[1])

        return X

    def _descriptor_names(self, n_features):
        if self.desc_names is None:
            return [f"x{i}" for i in range(n_features)]

        names = list(self.desc_names)

        if len(names) != n_features:
            return [f"x{i}" for i in range(n_features)]

        return names

    def _target_scores(self, X, y, method="f_regression"):
        method = str(method).lower()

        try:
            if method == "mutual_info":
                scores = mutual_info_regression(
                    X,
                    y,
                    random_state=self.random_state
                )
            else:
                scores, _ = f_regression(X, y)

            scores = np.asarray(scores, dtype=float)
            scores = np.where(np.isfinite(scores), scores, 0.0)

        except Exception:
            # fallback: если статистический скоринг не сработал,
            # используем дисперсию признака.
            scores = np.nanvar(X, axis=0)
            scores = np.where(np.isfinite(scores), scores, 0.0)

        return scores

    def _constant_mask(self, X):
        if not self.remove_constant:
            return np.ones(X.shape[1], dtype=bool)

        std = np.nanstd(X, axis=0)

        return np.asarray(std > 1e-12, dtype=bool)

    def _correlation_filter(self, X, scores, candidate_indices):
        if not self.remove_correlated:
            return list(candidate_indices), {}

        threshold = float(self.corr_threshold)

        if threshold <= 0 or threshold >= 1:
            threshold = 0.95

        candidate_indices = list(candidate_indices)

        if len(candidate_indices) <= 1:
            return candidate_indices, {}

        # Сначала сохраняем признаки с большей связью со свойством.
        order = sorted(
            candidate_indices,
            key=lambda i: float(scores[i]) if i < len(scores) else 0.0,
            reverse=True
        )

        kept = []
        removed_by_corr = {}

        for idx in order:
            if not kept:
                kept.append(idx)
                continue

            x = X[:, idx]

            should_remove = False
            remove_reason = ""

            for kept_idx in kept:
                x_kept = X[:, kept_idx]

                try:
                    r = np.corrcoef(x, x_kept)[0, 1]
                except Exception:
                    r = 0.0

                if not np.isfinite(r):
                    r = 0.0

                if abs(r) >= threshold:
                    should_remove = True
                    remove_reason = f"коррелирует с {kept_idx}, r={r:.4f}"
                    break

            if should_remove:
                removed_by_corr[idx] = remove_reason
            else:
                kept.append(idx)

        # Возвращаем в порядке силы связи со свойством.
        return kept, removed_by_corr

    def _select_final_indices(self, X, y, candidate_indices, scores):
        method = self._normalize_method()

        candidate_indices = list(candidate_indices)

        if not candidate_indices:
            return []

        max_features = int(self.max_features)
        max_features = max(1, min(max_features, len(candidate_indices)))

        X_candidate = X[:, candidate_indices]

        if method == "none":
            return candidate_indices[:max_features]

        if method in ["fast", "f_regression", "mutual_info"]:
            scoring_method = "mutual_info" if method == "mutual_info" else "f_regression"
            local_scores = self._target_scores(X_candidate, y, method=scoring_method)

            order_local = np.argsort(local_scores)[::-1]
            selected_local = order_local[:max_features]

            return [candidate_indices[i] for i in selected_local]

        if method == "lasso":
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_candidate)

                lasso = Lasso(
                    alpha=float(self.lasso_alpha),
                    max_iter=20000,
                    random_state=self.random_state
                )
                lasso.fit(X_scaled, y)

                coefs = np.abs(np.asarray(lasso.coef_, dtype=float))
                nonzero = np.where(coefs > 1e-12)[0]

                if len(nonzero) > 0:
                    order_local = nonzero[np.argsort(coefs[nonzero])[::-1]]
                else:
                    local_scores = self._target_scores(X_candidate, y, method="f_regression")
                    order_local = np.argsort(local_scores)[::-1]

                selected_local = order_local[:max_features]

                return [candidate_indices[i] for i in selected_local]

            except Exception:
                local_scores = self._target_scores(X_candidate, y, method="f_regression")
                order_local = np.argsort(local_scores)[::-1]
                selected_local = order_local[:max_features]

                return [candidate_indices[i] for i in selected_local]

        if method == "random_forest":
            try:
                rf = RandomForestRegressor(
                    n_estimators=int(self.rf_n_estimators),
                    random_state=self.random_state,
                    n_jobs=-1
                )
                rf.fit(X_candidate, y)

                importances = np.asarray(rf.feature_importances_, dtype=float)
                importances = np.where(np.isfinite(importances), importances, 0.0)

                order_local = np.argsort(importances)[::-1]
                selected_local = order_local[:max_features]

                return [candidate_indices[i] for i in selected_local]

            except Exception:
                local_scores = self._target_scores(X_candidate, y, method="f_regression")
                order_local = np.argsort(local_scores)[::-1]
                selected_local = order_local[:max_features]

                return [candidate_indices[i] for i in selected_local]

        if method == "rfe_ridge":
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_candidate)

                estimator = Ridge(alpha=1.0)

                rfe = RFE(
                    estimator=estimator,
                    n_features_to_select=max_features,
                    step=float(self.rfe_step)
                )
                rfe.fit(X_scaled, y)

                selected_local = np.where(rfe.support_)[0]

                # Сортируем выбранные признаки по рангу RFE.
                selected_local = sorted(
                    selected_local,
                    key=lambda i: rfe.ranking_[i]
                )

                return [candidate_indices[i] for i in selected_local]

            except Exception:
                local_scores = self._target_scores(X_candidate, y, method="f_regression")
                order_local = np.argsort(local_scores)[::-1]
                selected_local = order_local[:max_features]

                return [candidate_indices[i] for i in selected_local]

        # fallback
        order = sorted(
            candidate_indices,
            key=lambda i: float(scores[i]) if i < len(scores) else 0.0,
            reverse=True
        )

        return order[:max_features]

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Для отбора дескрипторов нужен y.")

        X = self._safe_numeric_matrix(X, fit=True)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape
        desc_names = self._descriptor_names(n_features)

        self.n_features_in_ = n_features
        self.desc_names_ = desc_names

        constant_mask = self._constant_mask(X)

        nonconstant_indices = [
            i for i in range(n_features)
            if bool(constant_mask[i])
        ]

        if not nonconstant_indices:
            nonconstant_indices = list(range(n_features))

        target_scores = self._target_scores(X, y, method="f_regression")
        mi_scores = self._target_scores(X, y, method="mutual_info")

        corr_kept_indices, removed_by_corr = self._correlation_filter(
            X=X,
            scores=target_scores,
            candidate_indices=nonconstant_indices
        )

        if not corr_kept_indices:
            corr_kept_indices = nonconstant_indices

        final_indices = self._select_final_indices(
            X=X,
            y=y,
            candidate_indices=corr_kept_indices,
            scores=target_scores
        )

        if not final_indices:
            final_indices = corr_kept_indices[:max(1, min(int(self.max_features), len(corr_kept_indices)))]

        self.selected_indices_ = list(final_indices)
        self.selected_names_ = [desc_names[i] for i in self.selected_indices_]

        self.nonconstant_indices_ = list(nonconstant_indices)
        self.corr_kept_indices_ = list(corr_kept_indices)
        self.removed_by_corr_ = dict(removed_by_corr)

        selected_set = set(self.selected_indices_)
        nonconstant_set = set(self.nonconstant_indices_)
        corr_kept_set = set(self.corr_kept_indices_)
        removed_corr_set = set(self.removed_by_corr_.keys())

        rows = []

        for i, name in enumerate(desc_names):
            if i not in nonconstant_set:
                status = "удалён: константный"
                selected = False
            elif i in removed_corr_set:
                status = "удалён: высокая корреляция"
                selected = False
            elif i in selected_set:
                status = "выбран финально"
                selected = True
            elif i in corr_kept_set:
                status = "оставлен после корреляционного фильтра, но не выбран финально"
                selected = False
            else:
                status = "не выбран"
                selected = False

            rows.append({
                "Дескриптор": name,
                "Индекс": i,
                "Статус": status,
                "Выбран": selected,
                "F-score": float(target_scores[i]) if i < len(target_scores) else np.nan,
                "Mutual information": float(mi_scores[i]) if i < len(mi_scores) else np.nan,
                "Причина удаления": self.removed_by_corr_.get(i, ""),
            })

        self.selection_table_ = pd.DataFrame(rows)

        self.selection_summary_ = {
            "method": self._normalize_method(),
            "n_samples": int(n_samples),
            "n_features_initial": int(n_features),
            "n_after_constant_filter": int(len(nonconstant_indices)),
            "n_removed_constant": int(n_features - len(nonconstant_indices)),
            "n_after_correlation_filter": int(len(corr_kept_indices)),
            "n_removed_correlated": int(len(nonconstant_indices) - len(corr_kept_indices)),
            "n_selected_final": int(len(self.selected_indices_)),
            "corr_threshold": float(self.corr_threshold),
            "remove_constant": bool(self.remove_constant),
            "remove_correlated": bool(self.remove_correlated),
        }

        return self

    def transform(self, X):
        X = self._safe_numeric_matrix(X, fit=False)

        if not hasattr(self, "selected_indices_"):
            raise ValueError("QSPRDescriptorSelector ещё не обучен.")

        return X[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "selected_names_"):
            return np.asarray(self.selected_names_, dtype=object)

        return np.asarray([], dtype=object) 

def qspr_auto_select_and_tune(
    X,
    y,
    desc_names,
    model_name,
    params=None,
    scale=True,
    feature_selection_method="fast",
    max_features=50,
    optimize_hyperparams=True,
    cv=5,
    search_method="grid",
    n_iter=30,
    remove_constant=True,
    remove_correlated=True,
    corr_threshold=0.95,
    lasso_selection_alpha=0.01,
    rf_selection_estimators=300,
    rfe_step=0.2
):
    """
    Полный автоматический отбор дескрипторов + оптимизация гиперпараметров.

    Поддерживает:
    - удаление константных дескрипторов;
    - удаление сильно коррелирующих дескрипторов;
    - быстрый отбор;
    - SelectKBest F-regression;
    - SelectKBest Mutual Information;
    - LASSO selection;
    - Random Forest importance;
    - RFE + Ridge;
    - GridSearchCV / RandomizedSearchCV.

    Возвращает:
    - обученную Pipeline-модель, принимающую исходную полную X-матрицу;
    - выбранные дескрипторы;
    - CV-прогноз;
    - CV-метрики;
    - таблицу отбора дескрипторов;
    - сводку отбора.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape

    if n_samples < 5:
        raise ValueError("Для автоотбора и оптимизации желательно минимум 5 веществ.")

    desc_names = list(desc_names)

    if len(desc_names) != n_features:
        desc_names = [f"x{i}" for i in range(n_features)]

    method = str(feature_selection_method).strip().lower()

    if method in ["none", "no_selection", "без отбора"]:
        max_features_effective = n_features
        remove_constant_effective = False
        remove_correlated_effective = False
    else:
        max_features_effective = int(max_features)
        max_features_effective = max(1, min(max_features_effective, n_features))
        remove_constant_effective = bool(remove_constant)
        remove_correlated_effective = bool(remove_correlated)

    cv = int(cv)
    cv = max(2, min(cv, n_samples))

    selector = QSPRDescriptorSelector(
        desc_names=desc_names,
        method=method,
        max_features=max_features_effective,
        remove_constant=remove_constant_effective,
        remove_correlated=remove_correlated_effective,
        corr_threshold=float(corr_threshold),
        lasso_alpha=float(lasso_selection_alpha),
        rf_n_estimators=int(rf_selection_estimators),
        rfe_step=float(rfe_step),
        random_state=42
    )

    # Для создания базовой модели ориентируемся на верхнюю оценку числа признаков.
    n_model_features = max(1, min(max_features_effective, n_features))

    base_model = qspr_create_regression_model(
        model_name=model_name,
        n_samples=n_samples,
        n_features=n_model_features,
        params=params
    )

    steps = []
    steps.append(("preselect", selector))

    if scale:
        steps.append(("scale", StandardScaler()))

    steps.append(("model", base_model))

    pipe = Pipeline(steps)

    param_grid = {}

    if optimize_hyperparams:
        param_grid = qspr_get_param_grid(model_name)

        if model_name == "PLS Regression" and "model__n_components" in param_grid:
            max_comp = max(1, min(n_model_features, n_samples - 1))
            param_grid["model__n_components"] = [
                v for v in param_grid["model__n_components"]
                if v <= max_comp
            ]

            if not param_grid["model__n_components"]:
                param_grid["model__n_components"] = [1]

        if model_name == "KNN Regression" and "model__n_neighbors" in param_grid:
            max_neighbors = max(1, n_samples - 1)
            param_grid["model__n_neighbors"] = [
                v for v in param_grid["model__n_neighbors"]
                if v <= max_neighbors
            ]

            if not param_grid["model__n_neighbors"]:
                param_grid["model__n_neighbors"] = [1]

    scoring = "neg_root_mean_squared_error"

    if optimize_hyperparams and param_grid:
        if search_method == "random":
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                n_iter=int(n_iter),
                scoring=scoring,
                cv=cv,
                random_state=42,
                n_jobs=-1,
                error_score=np.nan
            )
        else:
            search = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                error_score=np.nan
            )

        search.fit(X, y)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_rmse = float(-search.best_score_)

    else:
        best_model = pipe
        best_model.fit(X, y)
        best_params = {}
        best_cv_rmse = np.nan

    # Кросс-валидационный прогноз уже для выбранной схемы.
    try:
        y_cv_pred = cross_val_predict(
            best_model,
            X,
            y,
            cv=cv,
            n_jobs=-1
        )
        cv_metrics = qspr_metrics(y, y_cv_pred)
    except Exception:
        y_cv_pred = np.full_like(y, np.nan, dtype=float)
        cv_metrics = {
            "R2": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "ME": np.nan,
            "SD": np.nan,
            "MedianError": np.nan,
            "MAPE_percent": np.nan
        }

    # Финальный fit на всей выборке.
    best_model.fit(X, y)

    preselector = best_model.named_steps.get("preselect")
    selected_desc_names = list(getattr(preselector, "selected_names_", []))
    selection_table = getattr(preselector, "selection_table_", pd.DataFrame())
    selection_summary = getattr(preselector, "selection_summary_", {})

    X_selected = preselector.transform(X) if preselector is not None else X

    if scale and "scale" in best_model.named_steps:
        X_model_space = best_model.named_steps["scale"].transform(X_selected)
    else:
        X_model_space = X_selected

    y_fit_pred = np.ravel(best_model.predict(X))
    fit_metrics = qspr_metrics(y, y_fit_pred)

    return {
        "model": best_model,
        "selected_desc_names": selected_desc_names,
        "selected_indices": list(getattr(preselector, "selected_indices_", [])),
        "selection_table": selection_table,
        "selection_summary": selection_summary,
        "X_selected": X_selected,
        "X_model_space": X_model_space,
        "y_cv_pred": y_cv_pred,
        "cv_metrics": cv_metrics,
        "fit_metrics": fit_metrics,
        "best_params": best_params,
        "best_cv_rmse": best_cv_rmse,
        "feature_selection_method": method,
        "max_features": int(max_features_effective),
        "optimize_hyperparams": bool(optimize_hyperparams),
        "cv": int(cv),
        "search_method": search_method,
    }

def qspr_available_model_options(active_filters=None, match_mode="all"):
    """
    Возвращает доступные модели по группам из единого каталога.
    """
    availability = {
        "xgboost": xgboost_available,
        "lightgbm": lightgbm_available,
        "catboost": catboost_available,
        "pysr": pysr_available,
    }
    return get_models_by_group(
        active_filters=active_filters,
        match_mode=match_mode,
        availability=availability,
    )


# ------------------------------------------------------------------
# Обучение и валидация

def qspr_train_analysis_model(
    X,
    y,
    model_name,
    params=None,
    scale=True
):
    """
    Обучает аналитическую модель на всех данных.

    Возвращает:
    {
        "model": model,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "y_pred": y_pred,
        "metrics": metrics
    }
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if scale:
        scaler = StandardScaler()
        X_model = scaler.fit_transform(X)
    else:
        scaler = None
        X_model = X

    model = qspr_create_regression_model(
        model_name,
        n_samples=X_model.shape[0],
        n_features=X_model.shape[1],
        params=params
    )

    model.fit(X_model, y)

    y_pred = np.ravel(model.predict(X_model))

    metrics = qspr_metrics(y, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "X_scaled": X_model,
        "y_pred": y_pred,
        "metrics": metrics
    }


def qspr_holdout_validation(
    X,
    y,
    model_name,
    valid_indices=None,
    smiles=None,
    test_size=0.2,
    random_state=42,
    use_random=True,
    manual_indices=None,
    params=None,
    scale=True
):
    """
    Hold-out валидация.

    Если use_random=False, manual_indices должны быть индексами исходного датасета.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(y)

    if valid_indices is None:
        valid_indices = list(range(n))

    if smiles is None:
        smiles = [""] * n

    if use_random:
        indices = np.arange(n)

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state
        )

    else:
        if manual_indices is None:
            raise ValueError("Не указаны manual_indices для ручного Hold-out.")

        manual_indices_set = set(manual_indices)

        test_idx = [
            i for i, orig_idx in enumerate(valid_indices)
            if orig_idx in manual_indices_set
        ]

        train_idx = [
            i for i in range(n)
            if i not in test_idx
        ]

        if len(test_idx) == 0:
            raise ValueError("Ручные индексы тестовой выборки не найдены.")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    if scale:
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train)
        X_test_model = scaler.transform(X_test)
    else:
        scaler = None
        X_train_model = X_train
        X_test_model = X_test

    model = qspr_create_regression_model(
        model_name,
        n_samples=X_train_model.shape[0],
        n_features=X_train_model.shape[1],
        params=params
    )

    model.fit(X_train_model, y_train)

    y_pred_train = np.ravel(model.predict(X_train_model))
    y_pred_test = np.ravel(model.predict(X_test_model))

    train_orig_indices = [
        valid_indices[i]
        for i in train_idx
    ]

    test_orig_indices = [
        valid_indices[i]
        for i in test_idx
    ]

    train_smiles = [
        smiles[i]
        for i in train_idx
    ]

    test_smiles = [
        smiles[i]
        for i in test_idx
    ]

    train_table = qspr_prediction_table(
        y_true=y_train,
        y_pred=y_pred_train,
        smiles=train_smiles,
        original_indices=train_orig_indices,
        dataset_label="train"
    )

    test_table = qspr_prediction_table(
        y_true=y_test,
        y_pred=y_pred_test,
        smiles=test_smiles,
        original_indices=test_orig_indices,
        dataset_label="test"
    )

    metrics_train = qspr_metrics(y_train, y_pred_train)
    metrics_test = qspr_metrics(y_test, y_pred_test)

    return {
        "model": model,
        "scaler": scaler,

        "train_idx": train_idx,
        "test_idx": test_idx,

        "train_orig_indices": train_orig_indices,
        "test_orig_indices": test_orig_indices,

        "train_smiles": train_smiles,
        "test_smiles": test_smiles,

        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,

        "train_table": train_table,
        "test_table": test_table,

        "metrics_train": metrics_train,
        "metrics_test": metrics_test,

        "model_name": model_name
    }


def qspr_kfold_validation(
    X,
    y,
    model_name,
    valid_indices=None,
    smiles=None,
    k=5,
    params=None,
    scale=True,
    shuffle=True,
    random_state=42
):
    """
    K-Fold cross-validation.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(y)

    if valid_indices is None:
        valid_indices = list(range(n))

    if smiles is None:
        smiles = [""] * n

    if k < 2:
        raise ValueError("k должно быть >= 2.")

    if k > n:
        k = n

    min_train_size = n - int(np.ceil(n / k))

    model_cv = qspr_create_regression_model(
        model_name,
        n_samples=min_train_size,
        n_features=X.shape[1],
        params=params
    )

    if scale:
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", model_cv)
        ])
    else:
        pipe = Pipeline([
            ("model", model_cv)
        ])

    cv = KFold(
        n_splits=k,
        shuffle=shuffle,
        random_state=random_state if shuffle else None
    )

    y_pred_cv = np.ravel(
        cross_val_predict(
            pipe,
            X,
            y,
            cv=cv
        )
    )

    metrics = qspr_metrics(y, y_pred_cv)

    result_table = qspr_prediction_table(
        y_true=y,
        y_pred=y_pred_cv,
        smiles=smiles,
        original_indices=valid_indices,
        dataset_label=f"{k}-fold"
    )

    return {
        "y": y,
        "y_pred_cv": y_pred_cv,
        "valid_indices": valid_indices,
        "smiles": smiles,
        "k": k,
        "metrics": metrics,
        "result_table": result_table,
        "model_name": model_name
    }


def qspr_loo_validation(
    X,
    y,
    model_name,
    valid_indices=None,
    smiles=None,
    params=None,
    scale=True
):
    """
    Leave-One-Out cross-validation.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(y)

    if n < 3:
        raise ValueError("Для LOO нужно минимум 3 вещества.")

    if valid_indices is None:
        valid_indices = list(range(n))

    if smiles is None:
        smiles = [""] * n

    model_loo = qspr_create_regression_model(
        model_name,
        n_samples=n - 1,
        n_features=X.shape[1],
        params=params
    )

    if scale:
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", model_loo)
        ])
    else:
        pipe = Pipeline([
            ("model", model_loo)
        ])

    y_pred_loo = np.ravel(
        cross_val_predict(
            pipe,
            X,
            y,
            cv=LeaveOneOut()
        )
    )

    metrics = qspr_metrics(y, y_pred_loo)

    result_table = qspr_prediction_table(
        y_true=y,
        y_pred=y_pred_loo,
        smiles=smiles,
        original_indices=valid_indices,
        dataset_label="LOO"
    )

    return {
        "y": y,
        "y_pred_loo": y_pred_loo,
        "valid_indices": valid_indices,
        "smiles": smiles,
        "metrics": metrics,
        "result_table": result_table,
        "model_name": model_name
    }

# =============================================================================
# External Validation Simulator (валидация на наиболее удалённых веществах)
# =============================================================================

def qspr_external_validation_simulator(
    X,
    y,
    model_name,
    valid_indices=None,
    smiles=None,
    fraction=0.2,
    n_repeats=10,
    distance_metric='euclidean',
    params=None,
    scale=True,
    random_state=42
):
    """
    External Validation Simulator: обучает модель на (1-fraction) веществ,
    наиболее близких друг к другу, и тестирует на fraction самых удалённых.

    Параметры:
    - fraction: доля веществ для теста (0.2 = 20% самых удалённых)
    - n_repeats: число повторений с разными начальными условиями
      (для устойчивости, т.к. выборка удалённых точек может зависеть от случайности при жадном отборе)
    - distance_metric: 'euclidean' или 'mahalanobis' (пока реализован только евклидов)
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if valid_indices is None:
        valid_indices = list(range(n))
    if smiles is None:
        smiles = [""] * n

    if fraction <= 0 or fraction >= 1:
        raise ValueError("fraction должна быть между 0 и 1.")

    test_size = int(round(n * fraction))
    if test_size < 1:
        test_size = 1
    if test_size >= n:
        test_size = n - 1

    # Стандартизация для расстояний (всегда, чтобы избежать влияния масштаба)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Жадный алгоритм Farthest Point Sampling для отбора наиболее удалённых точек
    def farthest_point_sampling(X_scaled, k, random_seed=42):
        rng = np.random.default_rng(random_seed)
        n_pts = X_scaled.shape[0]
        if k >= n_pts:
            return list(range(n_pts))
        # Начинаем со случайной точки
        indices = [rng.integers(0, n_pts)]
        while len(indices) < k:
            distances = np.full(n_pts, np.inf)
            for idx in indices:
                d = np.linalg.norm(X_scaled - X_scaled[idx], axis=1)
                distances = np.minimum(distances, d)
            next_idx = np.argmax(distances)
            indices.append(int(next_idx))
        return indices

    all_metrics = []
    all_tables = []

    for rep in range(n_repeats):
        seed = random_state + rep
        test_indices_local = farthest_point_sampling(X_scaled, test_size, random_seed=seed)
        train_indices = [i for i in range(n) if i not in test_indices_local]

        X_train = X[train_indices]
        X_test = X[test_indices_local]
        y_train = y[train_indices]
        y_test = y[test_indices_local]

        if scale:
            scaler_model = StandardScaler()
            X_train_scaled = scaler_model.fit_transform(X_train)
            X_test_scaled = scaler_model.transform(X_test)
        else:
            scaler_model = None
            X_train_scaled = X_train
            X_test_scaled = X_test

        model = qspr_create_regression_model(
            model_name,
            n_samples=X_train_scaled.shape[0],
            n_features=X_train_scaled.shape[1],
            params=params
        )
        model.fit(X_train_scaled, y_train)
        y_pred_train = np.ravel(model.predict(X_train_scaled))
        y_pred_test = np.ravel(model.predict(X_test_scaled))

        metrics_train = qspr_metrics(y_train, y_pred_train)
        metrics_test = qspr_metrics(y_test, y_pred_test)

        test_orig_indices = [valid_indices[i] for i in test_indices_local]
        test_smiles = [smiles[i] for i in test_indices_local]
        test_table = qspr_prediction_table(
            y_true=y_test,
            y_pred=y_pred_test,
            smiles=test_smiles,
            original_indices=test_orig_indices,
            dataset_label=f"External Test (rep {rep+1})"
        )
        all_tables.append(test_table)
        all_metrics.append({
            'repeat': rep+1,
            'test_size': len(test_indices_local),
            'train_size': len(train_indices),
            **{f'train_{k}': v for k, v in metrics_train.items()},
            **{f'test_{k}': v for k, v in metrics_test.items()}
        })

    metrics_df = pd.DataFrame(all_metrics)
    summary = {
        'n_repeats': n_repeats,
        'fraction': fraction,
        'test_size_mean': metrics_df['test_size'].mean(),
        'train_size_mean': metrics_df['train_size'].mean(),
        'train_R2_mean': metrics_df['train_R2'].mean(),
        'train_R2_std': metrics_df['train_R2'].std(),
        'test_R2_mean': metrics_df['test_R2'].mean(),
        'test_R2_std': metrics_df['test_R2'].std(),
        'test_RMSE_mean': metrics_df['test_RMSE'].mean(),
        'test_RMSE_std': metrics_df['test_RMSE'].std(),
        'test_MAE_mean': metrics_df['test_MAE'].mean(),
        'test_MAE_std': metrics_df['test_MAE'].std(),
    }

    combined_test_table = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()

    return {
        'metrics_df': metrics_df,
        'summary': summary,
        'combined_test_table': combined_test_table,
        'model_name': model_name,
        'fraction': fraction,
        'n_repeats': n_repeats
    }

def qspr_consensus_predictions(models_scalers_dict, X, model_names=None):
    """
    Вычисляет консенсусный прогноз для набора моделей.
    
    Параметры:
    - models_scalers_dict: dict {model_name: (model, scaler)}
      где scaler — объект StandardScaler (может быть None, если масштабирование не требуется)
    - X: исходная матрица признаков (n_samples, n_features)
    - model_names: список имён (если не указаны, берутся из ключей словаря)
    
    Возвращает:
    - predictions_df: DataFrame с колонками:
        прогнозы каждой модели,
        'Consensus_mean' (среднее),
        'Consensus_std' (стандартное отклонение)
    """
    import numpy as np
    import pandas as pd

    if not models_scalers_dict:
        raise ValueError("Нет моделей для консенсуса.")

    names = model_names or list(models_scalers_dict.keys())
    X = np.asarray(X, dtype=float)

    preds = []
    for name in names:
        model, scaler = models_scalers_dict[name]
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        try:
            pred = np.ravel(model.predict(X_scaled))
            preds.append(pred)
        except Exception as e:
            raise RuntimeError(f"Ошибка прогноза для модели {name}: {e}")

    preds = np.array(preds)          # форма: (n_models, n_samples)
    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0, ddof=1)   # выборочное СКО

    df = pd.DataFrame()
    for i, name in enumerate(names):
        df[f'pred_{name}'] = preds[i]
    df['Consensus_mean'] = mean_pred
    df['Consensus_std'] = std_pred

    return df

# ------------------------------------------------------------------
# Коэффициенты моделей

def qspr_extract_model_coefficients(model, desc_names, model_name):
    """
    Извлекает коэффициенты для линейных моделей и формулу для Symbolic Regression.
    """
    model_name = normalize_runtime_name(model_name)

    if (
        model_name in ["GEP Symbolic Regression", "Genetic Programming Regression"]
        and hasattr(model, "get_formula")
    ):
        try:
            return pd.DataFrame({
                "Параметр": ["Формула", "Best score"],
                "Значение": [
                    model.get_formula(feature_names=desc_names),
                    getattr(model, "best_score_", np.nan)
                ],
                "Модель": [model_name, model_name]
            })
        except Exception:
            return pd.DataFrame()

    if model_name == "PySR" and hasattr(model, "sympy"):
        try:
            return pd.DataFrame({
                "Параметр": ["Формула"],
                "Значение": [str(model.sympy())],
                "Модель": [model_name]
            })
        except Exception:
            return pd.DataFrame()

    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    try:
        coefs = np.ravel(model.coef_)
    except Exception:
        return pd.DataFrame()

    if len(coefs) != len(desc_names):
        return pd.DataFrame()

    if model_name == "PLS Regression":
        coef_col = "Коэффициент PLS"
    else:
        coef_col = "Коэффициент"

    coef_table = pd.DataFrame({
        "Дескриптор": desc_names,
        coef_col: coefs,
        "Абсолютный коэффициент": np.abs(coefs),
        "Модель": model_name
    })

    coef_table = coef_table.sort_values(
        "Абсолютный коэффициент",
        ascending=False
    ).reset_index(drop=True)

    return coef_table


# ------------------------------------------------------------------
# Объединение молекулярных и спектральных дескрипторов

def qspr_build_descriptor_matrix_from_sources(
    current_df,
    target_col,
    use_molecular=True,
    molecular_desc_df=None,
    molecular_valid_indices=None,
    use_spectral=False,
    spectral_desc_df=None,
    smiles_col="SMILES",
    restrict_to_spectral_subset=True
):
    """
    Собирает итоговую матрицу признаков из источников:

    - молекулярные дескрипторы;
    - спектральные дескрипторы;
    - молекулярные + спектральные.

    Важно:
    если use_spectral=True, итоговая выборка ограничивается веществами,
    для которых есть спектральные дескрипторы.
    """
    if not use_molecular and not use_spectral:
        raise ValueError("Нужно выбрать хотя бы один источник дескрипторов.")

    if target_col not in current_df.columns:
        raise ValueError(f"Целевое свойство не найдено: {target_col}")

    target_numeric = qspr_to_numeric(current_df[target_col])

    parts = []
    row_index_source = None
    meta = None

    # ------------------------------------------------------------
    # Спектральная часть

    if use_spectral:
        if spectral_desc_df is None or spectral_desc_df.empty:
            raise ValueError("Спектральные дескрипторы не переданы или пусты.")

        spectral_work = spectral_desc_df.copy()

        if "row_index" not in spectral_work.columns:
            raise ValueError(
                "В spectral_desc_df нет row_index. "
                "Нельзя корректно сопоставить спектральные дескрипторы с датасетом."
            )

        spectral_work["row_index"] = spectral_work["row_index"].astype(int)

        spectral_meta_cols = [
            "row_index",
            "compound_id",
            "name",
            "input_smiles",
            "canonical_smiles",
            "inchikey",
            "spectrum_type",
            "spectrum_id",
            "spectrum_source",
            "processed_file"
        ]

        spectral_meta_cols = [
            c for c in spectral_meta_cols
            if c in spectral_work.columns
        ]

        spectral_desc_cols = [
            c for c in spectral_work.columns
            if c not in spectral_meta_cols
        ]

        spectral_numeric = spectral_work[spectral_desc_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        spectral_numeric = qspr_clean_numeric_dataframe(spectral_numeric)

        spectral_numeric.columns = [
            f"SPEC_{c}"
            if not str(c).startswith("SPEC_")
            else c
            for c in spectral_numeric.columns
        ]

        spectral_part = pd.concat(
            [
                spectral_work[["row_index"]].reset_index(drop=True),
                spectral_numeric.reset_index(drop=True)
            ],
            axis=1
        )

        row_index_source = spectral_work[["row_index"]].copy()

        meta = spectral_work[spectral_meta_cols].copy()

        parts.append(("spectral", spectral_part))

    # ------------------------------------------------------------
    # Молекулярная часть

    if use_molecular:
        if molecular_desc_df is None or molecular_desc_df.empty:
            raise ValueError("Молекулярные дескрипторы не переданы или пусты.")

        if molecular_valid_indices is None:
            raise ValueError("Не переданы molecular_valid_indices.")

        mol_work = molecular_desc_df.copy().reset_index(drop=True)
        mol_work["row_index"] = list(molecular_valid_indices)

        mol_desc_cols = [
            c for c in mol_work.columns
            if c != "row_index"
        ]

        mol_numeric = mol_work[mol_desc_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        mol_numeric = qspr_clean_numeric_dataframe(mol_numeric)

        mol_numeric.columns = [
            f"MOL_{c}"
            if not str(c).startswith("MOL_")
            else c
            for c in mol_numeric.columns
        ]

        mol_part = pd.concat(
            [
                mol_work[["row_index"]].reset_index(drop=True),
                mol_numeric.reset_index(drop=True)
            ],
            axis=1
        )

        if use_spectral and restrict_to_spectral_subset:
            if row_index_source is None:
                raise ValueError("Внутренняя ошибка: нет row_index_source.")

            mol_part = row_index_source.merge(
                mol_part,
                on="row_index",
                how="inner"
            )

        else:
            if row_index_source is None:
                row_index_source = mol_work[["row_index"]].copy()

        parts.append(("molecular", mol_part))

    # ------------------------------------------------------------
    # Объединение

    if len(parts) == 1:
        final = parts[0][1].copy()

    else:
        final = parts[0][1].copy()

        for _, part_df in parts[1:]:
            final = final.merge(
                part_df,
                on="row_index",
                how="inner"
            )

    if final.empty:
        raise ValueError("После объединения источников дескрипторов не осталось веществ.")

    final["target_value"] = target_numeric.iloc[
        final["row_index"].astype(int)
    ].values

    final = final[final["target_value"].notna()].copy()

    if final.empty:
        raise ValueError("После удаления строк без целевого свойства не осталось веществ.")

    y_all = final["target_value"].values.astype(float)

    desc_df = final.drop(
        columns=["row_index", "target_value"],
        errors="ignore"
    )

    desc_df = qspr_clean_numeric_dataframe(desc_df)

    if desc_df.empty:
        raise ValueError("После подготовки итоговой матрицы не осталось дескрипторов.")

    X_all = desc_df.values.astype(float)
    valid_indices = final["row_index"].astype(int).tolist()
    desc_names = desc_df.columns.tolist()

    if meta is not None:
        match_info = meta[
            meta["row_index"].astype(int).isin(valid_indices)
        ].copy()
    else:
        match_info = pd.DataFrame({
            "row_index": valid_indices
        })

    if smiles_col in current_df.columns:
        match_info = match_info.merge(
            current_df[[smiles_col]].reset_index().rename(
                columns={"index": "row_index", smiles_col: "SMILES"}
            ),
            on="row_index",
            how="left"
        )

    match_info[target_col] = target_numeric.iloc[
        match_info["row_index"].astype(int)
    ].values

    if use_molecular and use_spectral:
        source_label = "molecular_plus_spectral"
    elif use_molecular:
        source_label = "molecular_only"
    else:
        source_label = "spectral_only"

    report = {
        "descriptor_source": source_label,
        "n_compounds": len(valid_indices),
        "n_descriptors": len(desc_names),
        "use_molecular": bool(use_molecular),
        "use_spectral": bool(use_spectral),
    }

    return {
        "df_desc": desc_df.reset_index(drop=True),
        "X_all": X_all,
        "y_all": y_all,
        "valid_indices": valid_indices,
        "desc_names": desc_names,
        "match_info": match_info.reset_index(drop=True),
        "report": report
    }
