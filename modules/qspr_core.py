# -*- coding: utf-8 -*-

"""
qspr_core.py

Ядро QSPR для Augur QSPR:
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
import gc
import json
import re
import uuid
import hashlib
import shutil
import importlib.util
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
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
        normalize_model_id,
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
        normalize_model_id,
        normalize_runtime_name,
    )

from rdkit import Chem

try:
    from .runtime_mode import qspr_is_online_mode as _runtime_is_online_mode
except Exception:
    try:
        from runtime_mode import qspr_is_online_mode as _runtime_is_online_mode  # type: ignore
    except Exception:
        def _runtime_is_online_mode():
            return False
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
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


# ------------------------------------------------------------------
# Опциональные библиотеки

def _optional_dependency_status(package_name, exc=None):
    status = {
        "package": package_name,
        "available": exc is None,
        "error_type": "",
        "error_message": "",
    }
    if exc is not None:
        status.update({
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
    return status


mordred_available = False
mordred_status = _optional_dependency_status("mordred", RuntimeError("not checked"))

try:
    from mordred import Calculator as MordredCalculator
    from mordred import descriptors as mordred_descriptors
    mordred_available = True
    mordred_status = _optional_dependency_status("mordred")
except Exception as exc:
    mordred_available = False
    mordred_status = _optional_dependency_status("mordred", exc)


padel_available = False
padel_status = _optional_dependency_status("padelpy", RuntimeError("not checked"))

try:
    from padelpy import from_smiles
    padel_available = True
    padel_status = _optional_dependency_status("padelpy")
except Exception as exc:
    padel_available = False
    padel_status = _optional_dependency_status("padelpy", exc)


xgboost_available = False
xgboost_status = _optional_dependency_status("xgboost", RuntimeError("not checked"))

try:
    import xgboost as xgb
    xgboost_available = True
    xgboost_status = _optional_dependency_status("xgboost")
except Exception as exc:
    xgboost_available = False
    xgboost_status = _optional_dependency_status("xgboost", exc)


lightgbm_available = False
lightgbm_status = _optional_dependency_status("lightgbm", RuntimeError("not checked"))

try:
    import lightgbm as lgb
    lightgbm_available = True
    lightgbm_status = _optional_dependency_status("lightgbm")
except Exception as exc:
    lightgbm_available = False
    lightgbm_status = _optional_dependency_status("lightgbm", exc)


catboost_available = False
catboost_status = _optional_dependency_status("catboost", RuntimeError("not checked"))

try:
    from catboost import CatBoostRegressor
    catboost_available = True
    catboost_status = _optional_dependency_status("catboost")
except Exception as exc:
    catboost_available = False
    catboost_status = _optional_dependency_status("catboost", exc)


pysr_available = False
pysr_status = _optional_dependency_status("pysr", RuntimeError("not checked"))

pysr_spec = importlib.util.find_spec("pysr")
if pysr_spec is not None:
    pysr_available = True
    pysr_status = _optional_dependency_status("pysr")
else:
    pysr_available = False
    pysr_status = _optional_dependency_status("pysr", ModuleNotFoundError("pysr"))

xtb_python_available = False
xtb_python_status = _optional_dependency_status("xtb", RuntimeError("not checked"))

try:
    from xtb.interface import Calculator as XTBCalculator
    from xtb.utils import get_method

    try:
        from xtb.libxtb import VERBOSITY_MUTED
    except Exception:
        VERBOSITY_MUTED = 0

    xtb_python_available = True
    xtb_python_status = _optional_dependency_status("xtb")
except Exception as exc:
    xtb_python_available = False
    xtb_python_status = _optional_dependency_status("xtb", exc)

HARTREE_TO_EV = 27.211386245988
XTB_REQUIRED_DESCRIPTOR_COLUMNS = [
    "xtb_energy_hartree",
    "xtb_singlepoint_gradient_norm_on_rdkit_geometry",
    "xtb_charge_min",
    "xtb_charge_max",
    "xtb_charge_mean",
    "xtb_charge_std",
    "xtb_dipole_x",
    "xtb_dipole_y",
    "xtb_dipole_z",
    "xtb_dipole_norm",
    "xtb_homo_hartree",
    "xtb_lumo_hartree",
    "xtb_gap_hartree",
    "xtb_homo_eV",
    "xtb_lumo_eV",
    "xtb_gap_eV",
    "xtb_n_orbitals",
]
XTB_MODEL_MIN_COMPLETENESS = 0.75
DESCRIPTOR_MODE_CODES = {
    "rdkit_fast",
    "mordred",
    "mordred_padel_unique",
    "max_coverage",
}
DESCRIPTOR_MODE_LEGACY_ALIASES = {
    "🚀 Максимальная скорость (RDKit)": "rdkit_fast",
    "👁️‍🗨️ Расширенный (Mordred)": "mordred",
    "⚙️ Расширенный (Mordred)": "mordred",
    "⚡ Умный (Mordred + уникальные PaDEL)": "mordred_padel_unique",
    "🎯 Максимальная точность": "max_coverage",
    "🚀 Maximum speed (RDKit)": "rdkit_fast",
    "👁️‍🗨️ Extended (Mordred)": "mordred",
    "⚙️ Extended (Mordred)": "mordred",
    "⚡ Smart (Mordred + unique PaDEL)": "mordred_padel_unique",
    "⚡ Smart (unique Mordred + unique PaDEL)": "mordred_padel_unique",
    "⚡ Умный (уникальные Mordred + уникальные PaDEL)": "mordred_padel_unique",
    "🎯 Maximum accuracy": "max_coverage",
    "max_accuracy": "max_coverage",
}


def qspr_optional_dependency_statuses():
    return {
        "mordred": dict(mordred_status),
        "padelpy": dict(padel_status),
        "xgboost": dict(xgboost_status),
        "lightgbm": dict(lightgbm_status),
        "catboost": dict(catboost_status),
        "pysr": dict(pysr_status),
        "xtb": dict(xtb_python_status),
    }


def qspr_normalize_descriptor_mode(mode):
    mode_text = str(mode or "").strip()
    if mode_text in DESCRIPTOR_MODE_CODES:
        return mode_text
    return DESCRIPTOR_MODE_LEGACY_ALIASES.get(mode_text, mode_text)
XTB_OCCUPATION_EPS = 1e-6
XTB_PARTIAL_OCCUPATION_LOW = 1e-6
XTB_PARTIAL_OCCUPATION_HIGH = 1.999999


def qspr_is_streamlit_cloud_runtime():
    """Return True when the app appears to run in Streamlit Cloud."""
    return bool(_runtime_is_online_mode())


def qspr_n_jobs():
    """Conservative parallelism with AUGUR_MAX_JOBS override."""
    if qspr_is_streamlit_cloud_runtime():
        return 1
    env_value = os.getenv("AUGUR_MAX_JOBS", "").strip()
    if env_value:
        try:
            return max(1, int(env_value))
        except Exception as exc:
            pass
    cpu_count = os.cpu_count() or 2
    return max(1, min(cpu_count - 1, 8))


def qspr_search_n_jobs():
    """Avoid nested parallelism: models may already use qspr_n_jobs()."""
    return 1

# ------------------------------------------------------------------
# Файлы и папки

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

DESC_LISTS_FILE = PROJECT_ROOT / "descriptor_lists.json"
PADEL_UNIQUE_FILE = PROJECT_ROOT / "padel_unique_descriptors.txt"
DESC_MEANINGS_FILE = PROJECT_ROOT / "descriptor_meanings.json"

MODEL_ENCYCLOPEDIA_FILE = PROJECT_ROOT / "model_encyclopedia.json"
MODEL_ENCYCLOPEDIA_FALLBACK_FILE = PROJECT_ROOT / "help" / "model_encyclopedia.json"

# ------------------------------------------------------------------
# Служебные функции

def qspr_safe_target_name(target_col):
    """
    Безопасное имя свойства для имени файла.
    """
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    safe = str(target_col).strip()
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", safe)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("._ ")
    if not safe or safe.upper() in reserved or safe in {".", ".."}:
        safe = "target"
    return safe[:120]


def qspr_csv_download_bytes(df):
    """CSV bytes for user downloads: UTF-8 with BOM for Excel on Windows."""
    return df.to_csv(index=False).encode("utf-8-sig")


def qspr_rename_duplicate_columns(df):
    """
    Preserve duplicate columns by renaming later occurrences instead of
    silently dropping data.
    """
    if df is None or not hasattr(df, "columns"):
        return df
    counts = {}
    new_columns = []
    used_columns = set()
    duplicate_rows = []
    for position, column in enumerate(df.columns):
        name = str(column)
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            new_columns.append(column)
            used_columns.add(name)
        else:
            new_name = f"{name}__duplicate_{counts[name]}"
            while new_name in used_columns:
                counts[name] += 1
                new_name = f"{name}__duplicate_{counts[name]}"
            new_columns.append(new_name)
            used_columns.add(new_name)
            duplicate_rows.append({
                "original_column": name,
                "renamed_column": new_name,
                "position": int(position),
            })
    out = df.copy()
    out.columns = new_columns
    out.attrs["duplicate_column_report"] = pd.DataFrame(duplicate_rows)
    return out


def qspr_core_is_online_mode():
    return bool(_runtime_is_online_mode())


def qspr_save_results_auto(df, prefix, target_col, n_compounds, results_dir=RESULTS_DIR):
    """
    Автосохранение DataFrame в CSV.

    Имя:
    prefix_property_N_timestamp.csv
    """
    if qspr_core_is_online_mode():
        return ""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    safe_target = qspr_safe_target_name(target_col)
    suffix = uuid.uuid4().hex[:8]

    filename = f"{prefix}_{safe_target}_{n_compounds}_{timestamp}_{suffix}.csv"
    path = results_dir / filename

    df.to_csv(path, index=False, encoding="utf-8")

    return str(path)


def qspr_to_numeric(series):
    """
    Безопасное приведение серии к числу.
    Поддерживает десятичную запятую.
    """
    parsed = qspr_parse_numeric_series(series)
    numeric = parsed["value"].copy()
    numeric.attrs["numeric_parse_report"] = parsed
    return numeric


def _qspr_missing_text(text):
    return text.strip().lower() in {
        "",
        "-",
        "--",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "missing",
    }


def _qspr_failure_text(text):
    lower = text.strip().lower()
    return any(token in lower for token in ("failed", "error", "invalid", "exception"))


def _qspr_normalize_numeric_token(text):
    token = str(text).strip()
    token = token.replace("\u00a0", " ").replace("\u202f", " ")
    token = token.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    token = re.sub(r"\s+", "", token)
    token = re.sub(
        r"(?i)([+-]?\d+(?:[.,]\d+)?)\s*(?:\u00d7|x|\*)\s*10\s*\^?\s*([+-]?\d+)",
        r"\1e\2",
        token,
    )

    comma_pos = token.rfind(",")
    dot_pos = token.rfind(".")
    if comma_pos >= 0 and dot_pos >= 0:
        if comma_pos > dot_pos:
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+", token):
            token = token.replace(",", "")
        else:
            token = token.replace(",", ".")
    elif "." in token and re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+", token):
        token = token.replace(".", "")
    return token


def _qspr_parse_float_token(text):
    token = _qspr_normalize_numeric_token(text)
    if re.fullmatch(r"[+-]?(?:inf|infinity)", token, flags=re.IGNORECASE):
        return np.inf if not token.startswith("-") else -np.inf
    try:
        return float(token)
    except (TypeError, ValueError):
        return np.nan


def qspr_parse_numeric_value(value):
    """
    Parse one scientific numeric value while preserving censoring/status.
    """
    result = {
        "original_value": value,
        "value": np.nan,
        "relation": "",
        "censoring": "",
        "uncertainty": np.nan,
        "status": "parse_failed",
    }

    if isinstance(value, (bool, np.bool_)):
        result.update(value=float(value), status="boolean")
        return result

    try:
        if pd.isna(value):
            result["status"] = "missing"
            return result
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float, np.number)):
        numeric_value = float(value)
        if np.isposinf(numeric_value):
            result["status"] = "positive_infinity"
            return result
        if np.isneginf(numeric_value):
            result["status"] = "negative_infinity"
            return result
        result.update(value=numeric_value, status="ok")
        return result

    text = str(value).strip()
    if _qspr_missing_text(text):
        result["status"] = "missing"
        return result
    if _qspr_failure_text(text):
        result["status"] = "calculation_failed"
        return result

    normalized_text = text.replace("\u2264", "<=").replace("\u2265", ">=")
    relation_match = re.match(r"^\s*(<=|>=|<|>|=|~|\u2248)", normalized_text)
    if relation_match:
        relation = relation_match.group(1)
        result["relation"] = "~" if relation == "\u2248" else relation
        normalized_text = normalized_text[relation_match.end():].strip()

    uncertainty_parts = re.split(r"\s*(?:\u00b1|\+/-)\s*", normalized_text, maxsplit=1)
    numeric_text = uncertainty_parts[0]
    if len(uncertainty_parts) == 2:
        result["uncertainty"] = _qspr_parse_float_token(uncertainty_parts[1])

    numeric_value = _qspr_parse_float_token(numeric_text)
    if np.isposinf(numeric_value):
        result["status"] = "positive_infinity"
        return result
    if np.isneginf(numeric_value):
        result["status"] = "negative_infinity"
        return result
    if not np.isfinite(numeric_value):
        result["status"] = "parse_failed"
        return result

    result["value"] = numeric_value
    relation = result["relation"]
    if relation in {"<", "<="}:
        result.update(censoring="left", status="censored")
    elif relation in {">", ">="}:
        result.update(censoring="right", status="censored")
    elif relation in {"~", "\u2248"}:
        result["status"] = "approximate"
    elif np.isfinite(result["uncertainty"]):
        result["status"] = "value_with_uncertainty"
    else:
        result["status"] = "ok"
    return result


def qspr_parse_numeric_series(series):
    """Return a row-level parse report for a target or descriptor series."""
    if isinstance(series, pd.Series):
        input_series = series
    else:
        input_series = pd.Series(series)
    parsed_rows = [qspr_parse_numeric_value(value) for value in input_series]
    report = pd.DataFrame(parsed_rows, index=input_series.index)
    report["value"] = pd.to_numeric(report["value"], errors="coerce")
    report["uncertainty"] = pd.to_numeric(report["uncertainty"], errors="coerce")
    return report


def qspr_clean_numeric_dataframe(df, return_report=False):
    """
    Приводит DataFrame к числовому виду, удаляет пустые/константные колонки,
    заполняет пропуски медианами.
    """
    initial_columns = list(df.columns)
    work = df.copy()
    parse_reports = []
    quality_flag_columns = {}

    for col in work.columns:
        parsed = qspr_parse_numeric_series(work[col])
        parsed = parsed.copy()
        parsed.insert(0, "column", col)
        parse_reports.append(parsed)
        work[col] = parsed["value"]
        col_name = str(col)
        status = parsed["status"]
        quality_flag_columns[f"{col_name}__missing_original"] = status.eq("missing")
        quality_flag_columns[f"{col_name}__positive_infinity"] = status.eq("positive_infinity")
        quality_flag_columns[f"{col_name}__negative_infinity"] = status.eq("negative_infinity")
        quality_flag_columns[f"{col_name}__calculation_failed"] = status.eq("calculation_failed")
        quality_flag_columns[f"{col_name}__censored"] = status.eq("censored")
        quality_flag_columns[f"{col_name}__parse_failed"] = status.eq("parse_failed")

    all_nan_cols = [col for col in work.columns if work[col].isna().all()]
    non_numeric_cols = []
    empty_cols = []
    for col in all_nan_cols:
        report = parse_reports[initial_columns.index(col)]
        statuses = set(report["status"].dropna().astype(str).unique())
        if statuses and statuses.issubset({"missing"}):
            empty_cols.append(col)
        elif statuses.intersection({"parse_failed", "calculation_failed"}):
            non_numeric_cols.append(col)
        else:
            empty_cols.append(col)

    work = work.drop(columns=all_nan_cols)

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

    cleaning_report = {
        "initial_descriptor_count": int(len(initial_columns)),
        "non_numeric_descriptor_count": int(len(non_numeric_cols)),
        "empty_descriptor_count": int(len(empty_cols)),
        "constant_descriptor_count": int(len(const_cols)),
        "kept_descriptor_count": int(len(work.columns)),
        "non_numeric_descriptors": [str(col) for col in non_numeric_cols],
        "empty_descriptors": [str(col) for col in empty_cols],
        "constant_descriptors": [str(col) for col in const_cols],
    }

    if parse_reports:
        work.attrs["numeric_quality_report"] = pd.concat(parse_reports, axis=0)
    else:
        work.attrs["numeric_quality_report"] = pd.DataFrame()
    if quality_flag_columns:
        work.attrs["numeric_quality_flags"] = pd.DataFrame(
            quality_flag_columns,
            index=work.index,
        )
    else:
        work.attrs["numeric_quality_flags"] = pd.DataFrame(index=work.index)
    work.attrs["cleaning_report"] = cleaning_report

    if return_report:
        return work, cleaning_report
    return work


def qspr_metrics(y_true, y_pred):
    """
    Возвращает основные метрики регрессии.
    """
    y_true_raw = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_raw = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true_raw) != len(y_pred_raw):
        raise ValueError(
            f"Cannot calculate regression metrics: y_true length ({len(y_true_raw)}) "
            f"differs from y_pred length ({len(y_pred_raw)})."
        )
    n_raw = int(len(y_true_raw))
    if n_raw == 0:
        raise ValueError("Cannot calculate regression metrics: empty y_true/y_pred arrays.")

    mask = np.isfinite(y_true_raw) & np.isfinite(y_pred_raw)
    n_excluded_nonfinite = int(n_raw - int(mask.sum()))
    y_true = y_true_raw[mask]
    y_pred = y_pred_raw[mask]

    if len(y_true) == 0:
        return {
            "R2": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "ME": np.nan,
            "SD": np.nan,
            "SD_ddof": 1,
            "Residual_SD_population": np.nan,
            "MedianError": np.nan,
            "MAPE_percent": np.nan,
            "MAPE_applicable": False,
            "MAPE_warning": "No finite y_true/y_pred pairs.",
            "MAPE_n_used": 0,
            "MAPE_n_excluded": n_raw,
            "N": 0,
            "N_raw": n_raw,
            "N_used": 0,
            "N_excluded_nonfinite": n_excluded_nonfinite,
        }

    errors = y_true - y_pred

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))

    n = int(len(y_true))

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan

    abs_y = np.abs(y_true)
    median_abs_y = float(np.nanmedian(abs_y)) if len(abs_y) else np.nan
    y_range = float(np.nanmax(y_true) - np.nanmin(y_true)) if len(y_true) else np.nan
    y_sd = float(np.nanstd(y_true, ddof=1)) if len(y_true) > 1 else np.nan
    q75, q25 = np.nanpercentile(y_true, [75, 25]) if len(y_true) else (np.nan, np.nan)
    y_iqr = float(q75 - q25) if np.isfinite(q75) and np.isfinite(q25) else np.nan

    mape_warnings = []
    if np.any(y_true == 0):
        mape_warnings.append("Target contains zero values.")
    if np.any(y_true < 0):
        mape_warnings.append("Target contains negative values.")
    if not np.isfinite(median_abs_y) or median_abs_y < 1e-8:
        mape_warnings.append("Median absolute target value is too close to zero.")

    if mape_warnings:
        mape = np.nan
        mape_applicable = False
        mape_n_used = 0
        mape_n_excluded = n
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            mape_values = np.abs((y_true - y_pred) / y_true) * 100
            mape_values = mape_values[np.isfinite(mape_values)]
        mape = float(np.mean(mape_values)) if len(mape_values) > 0 else np.nan
        mape_n_used = int(len(mape_values))
        mape_n_excluded = int(n - mape_n_used)
        mape_applicable = bool(np.isfinite(mape))

    if n < 5:
        r2_reliability = "not_interpretable_n_lt_5"
    elif n < 10:
        r2_reliability = "high_uncertainty_n_lt_10"
    else:
        r2_reliability = "standard"

    pearson_r = np.nan
    spearman_rho = np.nan
    if n > 1:
        try:
            pearson_r = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))
        except Exception as exc:
            pearson_r = np.nan
        try:
            spearman_rho = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
        except Exception as exc:
            spearman_rho = np.nan

    ccc = np.nan
    if n > 1:
        mean_true = float(np.nanmean(y_true))
        mean_pred = float(np.nanmean(y_pred))
        var_true = float(np.nanvar(y_true, ddof=1))
        var_pred = float(np.nanvar(y_pred, ddof=1))
        covariance = float(np.cov(y_true, y_pred, ddof=1)[0, 1])
        denom = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = float((2 * covariance) / denom) if denom > 1e-12 else np.nan

    return {
        "R2": r2,
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": mae,
        "ME": float(np.mean(errors)),
        "SD": float(np.std(errors, ddof=1)) if n > 1 else np.nan,
        "SD_ddof": 1,
        "Residual_SD_population": float(np.std(errors, ddof=0)),
        "MedianError": float(np.median(errors)),
        "MAPE_percent": mape,
        "MAPE_applicable": mape_applicable,
        "MAPE_warning": " ".join(mape_warnings),
        "MAPE_n_used": mape_n_used,
        "MAPE_n_excluded": mape_n_excluded,
        "N": n,
        "N_raw": n_raw,
        "N_used": n,
        "N_excluded_nonfinite": n_excluded_nonfinite,
        "R2_reliability": r2_reliability,
        "NRMSE_range": float(rmse / y_range) if np.isfinite(y_range) and y_range > 1e-12 else np.nan,
        "NRMSE_sd": float(rmse / y_sd) if np.isfinite(y_sd) and y_sd > 1e-12 else np.nan,
        "MAE_IQR": float(mae / y_iqr) if np.isfinite(y_iqr) and y_iqr > 1e-12 else np.nan,
        "CCC": ccc,
        "Pearson_r": pearson_r,
        "Spearman_rho": spearman_rho,
        "target_range": y_range,
        "target_sd": y_sd,
        "target_iqr": y_iqr,
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
    if dataset_label is not None and not isinstance(dataset_label, str):
        raise TypeError("dataset_label must be a string or None.")

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError(
            f"Prediction table length mismatch: y_true has {n} rows, "
            f"y_pred has {len(y_pred)} rows."
        )

    if original_indices is not None:
        original_indices = list(original_indices)
        if len(original_indices) != n:
            raise ValueError(
                f"Prediction table length mismatch: original_indices has "
                f"{len(original_indices)} rows, expected {n}."
            )

    if smiles is not None:
        smiles = list(smiles)
        if len(smiles) != n:
            raise ValueError(
                f"Prediction table length mismatch: smiles has {len(smiles)} rows, "
                f"expected {n}."
            )

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

def qspr_compute_descriptor_lists(probe_padel=False):
    """
    Создаёт списки дескрипторов:
    - rdkit_all;
    - mordred_unique;
    - padel_all;
    - padel_unique.

    Важно:
    - padel_all нужен для режима расширенного охвата;
    - padel_unique нужен для режима "⚡ Умный".
    """
    result = {
        "schema_version": "1.0",
        "padel_catalog_status": "not_checked",
        "padel_catalog_message": "PaDEL runtime probe was not executed.",
    }

    rdkit_all = [
        name for name, _ in Descriptors._descList
    ]

    result["rdkit_all"] = sorted(rdkit_all)

    rdkit_set = set(rdkit_all)

    mordred_set = set()

    if mordred_available:
        try:
            calc = MordredCalculator(mordred_descriptors, ignore_3D=True)

            for desc in calc.descriptors:
                name = qspr_mordred_descriptor_name(desc)

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

    if padel_available and probe_padel:
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
    elif not padel_available:
        result["padel_catalog_status"] = "unavailable"
        result["padel_catalog_message"] = "padelpy is not importable."

    padel_all = sorted(set(padel_fp) | set(padel_desc))
    if padel_all:
        result["padel_catalog_status"] = "complete"
        result["padel_catalog_message"] = "Full PaDEL catalog was collected from padelpy."

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
        result["padel_catalog_status"] = "fallback_partial"
        result["padel_catalog_message"] = (
            "Full PaDEL catalog is unavailable. Fallback limited unique PaDEL list is used."
        )

    result["padel_unique"] = padel_unique

    return result


def qspr_validate_descriptor_lists_schema(data):
    if not isinstance(data, dict):
        raise ValueError("descriptor_lists.json must contain a JSON object.")
    required_list_keys = {
        "rdkit_all",
        "mordred_unique",
        "padel_all",
        "padel_unique",
        "padel_fingerprints",
        "padel_1d2d",
    }
    optional_list_keys = {
        "rdkit_unique",
        "mordred_all",
        "xtb",
    }
    list_like_keys = required_list_keys | optional_list_keys
    schema_version = str(data.get("schema_version", "") or "").strip()
    if schema_version and schema_version != "1.0":
        raise ValueError(f"Unsupported descriptor list schema_version `{schema_version}`.")
    missing = sorted(key for key in required_list_keys if key not in data)
    if missing:
        raise ValueError(
            "descriptor_lists.json is missing required list keys: "
            + ", ".join(missing)
        )
    normalized = dict(data)
    normalized["schema_version"] = schema_version or "1.0"
    for key, value in normalized.items():
        if key == "schema_version":
            continue
        if key.endswith("_status") or key.endswith("_message"):
            continue
        if key in list_like_keys or key.endswith("_all") or key.endswith("_unique"):
            if not isinstance(value, list):
                raise ValueError(f"Descriptor list `{key}` must be a list.")
            if any(not isinstance(item, str) for item in value):
                raise ValueError(
                    f"Descriptor list `{key}` must contain only strings."
                )
    return normalized


def _descriptor_lists_backup_path(filename):
    path = Path(filename)
    return path.with_name(f"{path.stem}.backup{path.suffix}")


def qspr_save_descriptor_lists(lists, filename=DESC_LISTS_FILE):
    """
    Сохраняет списки дескрипторов в JSON.
    """
    qspr_validate_descriptor_lists_schema(lists)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    backup_path = _descriptor_lists_backup_path(path)

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(lists, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                qspr_validate_descriptor_lists_schema(json.load(f))
            shutil.copy2(path, backup_path)
        except Exception:
            pass
    os.replace(tmp_path, path)
    try:
        shutil.copy2(path, backup_path)
    except Exception:
        pass


def qspr_load_descriptor_lists(filename=DESC_LISTS_FILE):
    """
    Загружает descriptor_lists.json.
    """
    path = Path(filename)
    qspr_load_descriptor_lists.last_error = None
    first_error = None
    for candidate in [path, _descriptor_lists_backup_path(path)]:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                loaded = qspr_validate_descriptor_lists_schema(json.load(f))
            if first_error is not None:
                qspr_load_descriptor_lists.last_error = {
                    **first_error,
                    "recovered_from": str(candidate),
                }
            return loaded
        except Exception as exc:
            error = {
                "status": (
                    "invalid_json"
                    if isinstance(exc, json.JSONDecodeError)
                    else "invalid_schema"
                ),
                "file": str(candidate),
                "error": str(exc),
            }
            if first_error is None:
                first_error = error
    qspr_load_descriptor_lists.last_error = first_error
    return None


qspr_load_descriptor_lists.last_error = None


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
    allowed_set = None if allowed_names is None else set(allowed_names)

    for name, func in Descriptors._descList:
        if allowed_set is not None and name not in allowed_set:
            continue

        try:
            desc_dict[name] = func(mol)
        except Exception:
            desc_dict[name] = np.nan

    return desc_dict


def qspr_mordred_descriptor_name(descriptor):
    return str(descriptor)


@lru_cache(maxsize=32)
def qspr_cached_mordred_calculator(allowed_names_key=None):
    if not mordred_available:
        return None

    base_calc = MordredCalculator(mordred_descriptors, ignore_3D=True)
    if allowed_names_key is None:
        return base_calc

    allowed_set = set(allowed_names_key)
    selected_descriptors = [
        descriptor
        for descriptor in base_calc.descriptors
        if qspr_mordred_descriptor_name(descriptor) in allowed_set
    ]
    if not selected_descriptors:
        return None
    return MordredCalculator(selected_descriptors, ignore_3D=True)


def qspr_make_mordred_calculator(allowed_names=None):
    if allowed_names is None:
        return qspr_cached_mordred_calculator(None)
    names_key = tuple(dict.fromkeys(allowed_names))
    return qspr_cached_mordred_calculator(names_key)


def qspr_mordred_value_error(value):
    value_type = type(value)
    module_name = getattr(value_type, "__module__", "")
    class_name = getattr(value_type, "__name__", "")
    if module_name.startswith("mordred.error") or class_name in {
        "Missing",
        "Error",
        "MissingValueException",
    }:
        return {
            "status": "failed",
            "error_type": class_name,
            "error_message": str(value),
        }
    return None


def qspr_calc_mordred_descriptors_filtered(
    mol,
    calculator,
    allowed_names=None,
    return_diagnostics=False,
):
    """
    Расчёт Mordred-дескрипторов.
    """
    if calculator is None:
        return ({}, []) if return_diagnostics else {}

    try:
        results = calculator(mol)

        desc_dict = {}
        diagnostics = []
        allowed_set = None if allowed_names is None else set(allowed_names)

        for desc, value in zip(calculator.descriptors, results):
            dname = qspr_mordred_descriptor_name(desc)

            if allowed_set is not None and dname not in allowed_set:
                continue

            value_error = qspr_mordred_value_error(value)
            if value_error is not None:
                diagnostics.append({
                    "descriptor_name": dname,
                    **value_error,
                })
                continue

            desc_dict[dname] = value

        return (desc_dict, diagnostics) if return_diagnostics else desc_dict

    except Exception as exc:
        diagnostics = [{
            "descriptor_name": "",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }]
        return ({}, diagnostics) if return_diagnostics else {}


def qspr_calc_padel_descriptors_filtered(
    smiles,
    allowed_names=None,
    prefix_internal_names=False,
    return_diagnostics=False,
):
    """
    Расчёт PaDEL-дескрипторов через padelpy.
    """
    if not padel_available:
        return ({}, []) if return_diagnostics else {}

    diagnostics = []

    try:
        smiles_str = str(smiles).strip()

        if not smiles_str:
            return ({}, diagnostics) if return_diagnostics else {}

        padel_dict = {}
        allowed_set = set(allowed_names or []) if allowed_names is not None else None

        if allowed_set is None:
            try:
                result_all = from_smiles([smiles_str], fingerprints=True)

                if result_all and len(result_all) > 0:
                    raw_all = dict(result_all[0] or {})
                    padel_dict = {
                        f"PaDEL::{name}" if prefix_internal_names else name: value
                        for name, value in raw_all.items()
                    }

            except Exception as exc:
                diagnostics.append({
                    "source": "PaDEL",
                    "descriptor_name": "",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })
        else:
            try:
                result_desc = from_smiles([smiles_str], fingerprints=False)
                desc_dict = dict(result_desc[0] or {}) if result_desc else {}
                for name, value in desc_dict.items():
                    if name in allowed_set or f"PaDEL_2D::{name}" in allowed_set:
                        out_name = f"PaDEL_2D::{name}" if prefix_internal_names else name
                        padel_dict[out_name] = value
            except Exception as exc:
                diagnostics.append({
                    "source": "PaDEL_2D",
                    "descriptor_name": "",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

            found_unprefixed = {
                str(name).removeprefix("PaDEL_2D::").removeprefix("PaDEL_FP::")
                for name in padel_dict
            }
            requested_unprefixed = {
                str(name).removeprefix("PaDEL_2D::").removeprefix("PaDEL_FP::")
                for name in allowed_set
            }
            missing_after_2d = requested_unprefixed - found_unprefixed
            explicitly_fp = any(str(name).startswith("PaDEL_FP::") for name in allowed_set)

            if missing_after_2d or explicitly_fp:
                try:
                    result_fp = from_smiles([smiles_str], fingerprints=True)
                    fp_dict = dict(result_fp[0] or {}) if result_fp else {}
                    for name, value in fp_dict.items():
                        if (
                            name in missing_after_2d
                            or name in allowed_set
                            or f"PaDEL_FP::{name}" in allowed_set
                        ):
                            out_name = f"PaDEL_FP::{name}" if prefix_internal_names else name
                            padel_dict[out_name] = value
                except Exception as exc:
                    diagnostics.append({
                        "source": "PaDEL_FP",
                        "descriptor_name": "",
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    })

        if padel_dict:
            if allowed_names is not None:
                filtered = {
                    k: v
                    for k, v in padel_dict.items()
                    if k in allowed_set
                    or k.removeprefix("PaDEL_FP::") in allowed_set
                    or k.removeprefix("PaDEL_2D::") in allowed_set
                }
                return (filtered, diagnostics) if return_diagnostics else filtered

            return (padel_dict, diagnostics) if return_diagnostics else padel_dict

    except Exception as exc:
        diagnostics.append({
            "source": "PaDEL",
            "descriptor_name": "",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        return ({}, diagnostics) if return_diagnostics else {}

    return ({}, diagnostics) if return_diagnostics else {}

# ------------------------------------------------------------------
# xTB quantum descriptors


def qspr_prefix_descriptor_names(source, descriptors):
    return {
        f"{source}::{name}": value
        for name, value in (descriptors or {}).items()
    }


def qspr_descriptor_provenance_table(desc_names):
    rows = []
    for internal_name in desc_names:
        text = str(internal_name)
        parts = text.split("::", 1)
        if len(parts) == 2:
            source, display_name = parts
        else:
            source, display_name = "", text
        rows.append({
            "internal_name": text,
            "display_name": display_name,
            "source": source,
        })
    return pd.DataFrame(rows)

def qspr_xtb_formal_charge_and_electrons(mol):
    if mol is None:
        return 0, 0
    try:
        mol = Chem.AddHs(mol)
    except Exception:
        pass
    formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    electrons = int(sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - formal_charge)
    return formal_charge, electrons


def qspr_xtb_validate_charge_uhf(mol, charge=0, uhf=0):
    formal_charge, electrons = qspr_xtb_formal_charge_and_electrons(mol)
    charge = int(charge)
    uhf = int(uhf)

    if formal_charge != charge:
        return False, (
            f"formal_charge_mismatch: rdkit={formal_charge}, requested={charge}"
        )

    if uhf < 0:
        return False, "invalid_uhf_negative"

    unpaired_parity = uhf % 2
    electron_parity = electrons % 2
    if unpaired_parity != electron_parity:
        return False, (
            f"electron_uhf_parity_mismatch: electrons={electrons}, uhf={uhf}"
        )

    return True, "ok"


def qspr_xtb_prepare_3d_mol_from_smiles(
    smiles,
    random_seed=1,
    max_embed_attempts=20,
    conformer_mode="fast",
    conformer_count=1,
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

        mode_defaults = {
            "fast": 1,
            "standard": 20,
            "ensemble": 20,
        }
        conformer_mode = str(conformer_mode or "fast").strip().lower()
        requested_conformers = int(
            conformer_count or mode_defaults.get(conformer_mode, 1)
        )
        requested_conformers = max(1, requested_conformers)
        max_embed_attempts = max(1, int(max_embed_attempts or 1))

        if conformer_mode == "fast" or requested_conformers == 1:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(random_seed)
            embed_status = -1
            for attempt in range(max_embed_attempts):
                params.randomSeed = int(random_seed) + attempt
                embed_status = AllChem.EmbedMolecule(mol, params)
                if embed_status == 0:
                    break
            conformer_ids = [0] if embed_status == 0 else []
        else:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(random_seed)
            params.numThreads = 0
            conformer_ids = list(
                AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=requested_conformers,
                    params=params,
                )
            )
            embed_status = 0 if conformer_ids else -1

        if not conformer_ids:
            return None, "3d_embedding_failed"

        selected_conf_id = int(conformer_ids[0])
        selected_energy = np.nan
        if optimize_with_rdkit:
            energies = []
            for conf_id in conformer_ids:
                conf_id = int(conf_id)
                try:
                    if AllChem.MMFFHasAllMoleculeParams(mol):
                        props = AllChem.MMFFGetMoleculeProperties(mol)
                        ff = AllChem.MMFFGetMoleculeForceField(
                            mol,
                            props,
                            confId=conf_id,
                        )
                    else:
                        ff = AllChem.UFFGetMoleculeForceField(
                            mol,
                            confId=conf_id,
                        )
                    if ff is None:
                        continue
                    ff.Minimize(maxIts=300)
                    energies.append((float(ff.CalcEnergy()), conf_id))
                except Exception:
                    continue
        if energies:
            selected_energy, selected_conf_id = min(energies, key=lambda item: item[0])
            mol.SetProp(
                "_qspr_conformer_energy_table",
                ";".join(
                    f"{conf_id}:{energy}"
                    for energy, conf_id in sorted(energies, key=lambda item: item[0])
                ),
            )
        else:
            mol.SetProp(
                "_qspr_conformer_energy_table",
                ";".join(f"{int(conf_id)}:nan" for conf_id in conformer_ids),
            )

        mol.SetProp("_qspr_conformer_mode", conformer_mode)
        mol.SetIntProp("_qspr_conformer_count_generated", len(conformer_ids))
        if np.isfinite(selected_energy):
            mol.SetDoubleProp("_qspr_selected_conformer_energy", float(selected_energy))
        mol.SetIntProp("_qspr_selected_conformer_id", int(selected_conf_id))

        return mol, "ok"

    except Exception as e:
        return None, f"3d_prepare_error: {e}"


def qspr_xtb_mol_to_numbers_positions(mol, conf_id=None):
    """
    Преобразует RDKit Mol с 3D-конформером в atomic numbers и координаты.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None, None

    if conf_id is None:
        try:
            conf_id = int(mol.GetIntProp("_qspr_selected_conformer_id"))
        except Exception:
            conf_id = -1

    conf = mol.GetConformer(int(conf_id)) if int(conf_id) >= 0 else mol.GetConformer()

    numbers = []
    positions = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)

        numbers.append(atom.GetAtomicNum())
        positions.append([float(pos.x), float(pos.y), float(pos.z)])

    return np.array(numbers, dtype=int), np.array(positions, dtype=float)


def qspr_xtb_orbital_diagnostics(result):
    diagnostics = {
        "xtb_homo_hartree": np.nan,
        "xtb_lumo_hartree": np.nan,
        "xtb_gap_hartree": np.nan,
        "xtb_somo_hartree": np.nan,
        "xtb_somo_occupation": np.nan,
        "xtb_n_orbitals_with_fractional_occupation": 0,
        "xtb_min_fractional_occupation": np.nan,
        "xtb_max_fractional_occupation": np.nan,
        "xtb_frontier_orbital_assignment_status": "unavailable",
    }
    try:
        eigenvalues = np.asarray(result.get_orbital_eigenvalues(), dtype=float)
        occupations = np.asarray(result.get_orbital_occupations(), dtype=float)
        if len(eigenvalues) != len(occupations) or len(eigenvalues) == 0:
            return diagnostics

        occupied = occupations > XTB_OCCUPATION_EPS
        virtual = occupations <= XTB_OCCUPATION_EPS
        fractional = (
            (occupations > XTB_PARTIAL_OCCUPATION_LOW)
            & (occupations < XTB_PARTIAL_OCCUPATION_HIGH)
        )
        diagnostics["xtb_n_orbitals_with_fractional_occupation"] = int(
            np.count_nonzero(fractional)
        )
        if np.any(fractional):
            frac_occ = occupations[fractional]
            diagnostics["xtb_min_fractional_occupation"] = float(np.nanmin(frac_occ))
            diagnostics["xtb_max_fractional_occupation"] = float(np.nanmax(frac_occ))
            somo_candidates = np.where(fractional)[0]
            somo_idx = int(somo_candidates[-1])
            diagnostics["xtb_somo_hartree"] = float(eigenvalues[somo_idx])
            diagnostics["xtb_somo_occupation"] = float(occupations[somo_idx])

        occ_idx = np.where(occupied)[0]
        virt_idx = np.where(virtual)[0]
        if len(occ_idx) > 0:
            homo_idx = int(occ_idx[-1])
            diagnostics["xtb_homo_hartree"] = float(eigenvalues[homo_idx])
        if len(virt_idx) > 0:
            lumo_idx = int(virt_idx[0])
            diagnostics["xtb_lumo_hartree"] = float(eigenvalues[lumo_idx])
        homo = diagnostics["xtb_homo_hartree"]
        lumo = diagnostics["xtb_lumo_hartree"]
        if np.isfinite(homo) and np.isfinite(lumo):
            diagnostics["xtb_gap_hartree"] = float(lumo - homo)
        diagnostics["xtb_frontier_orbital_assignment_status"] = (
            "ambiguous_fractional_occupations" if np.any(fractional) else "closed_shell_like"
        )
    except Exception:
        pass
    return diagnostics


def qspr_xtb_homo_lumo_from_result(result):
    """
    Извлекает HOMO, LUMO и gap из результата xTB.
    """
    homo = np.nan
    lumo = np.nan
    gap = np.nan

    diagnostics = qspr_xtb_orbital_diagnostics(result)
    homo = diagnostics["xtb_homo_hartree"]
    lumo = diagnostics["xtb_lumo_hartree"]
    gap = diagnostics["xtb_gap_hartree"]

    return homo, lumo, gap


def qspr_xtb_update_descriptors_from_result(base, result):
    try:
        base["xtb_energy_hartree"] = float(result.get_energy())
    except Exception:
        pass

    try:
        gradient = np.asarray(result.get_gradient(), dtype=float)
        gradient_norm = float(np.linalg.norm(gradient))
        base["xtb_singlepoint_gradient_norm_on_rdkit_geometry"] = gradient_norm
        base["xtb_gradient_norm"] = gradient_norm
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
        orbital_diagnostics = qspr_xtb_orbital_diagnostics(result)
        base["xtb_homo_hartree"] = homo
        base["xtb_lumo_hartree"] = lumo
        base["xtb_gap_hartree"] = gap
        for key, value in orbital_diagnostics.items():
            base[key] = value
        if np.isfinite(homo):
            base["xtb_homo_eV"] = float(homo * HARTREE_TO_EV)
        if np.isfinite(lumo):
            base["xtb_lumo_eV"] = float(lumo * HARTREE_TO_EV)
        if np.isfinite(gap):
            base["xtb_gap_eV"] = float(gap * HARTREE_TO_EV)
        somo = base.get("xtb_somo_hartree", np.nan)
        if np.isfinite(somo):
            base["xtb_somo_eV"] = float(somo * HARTREE_TO_EV)
    except Exception:
        pass

    try:
        base["xtb_n_orbitals"] = int(result.get_number_of_orbitals())
    except Exception:
        pass

    return base


def qspr_calc_xtb_descriptors_single(
    smiles,
    method="GFN2-xTB",
    charge=0,
    uhf=0,
    accuracy=1.0,
    electronic_temperature=300.0,
    max_iterations=250,
    random_seed=1,
    max_embed_attempts=20,
    conformer_mode="fast",
    conformer_count=1,
    optimize_with_rdkit=True
):
    """
    Считает xTB-дескрипторы для одной молекулы.
    """
    base = {
        "xtb_energy_hartree": np.nan,
        "xtb_gradient_norm": np.nan,
        "xtb_singlepoint_gradient_norm_on_rdkit_geometry": np.nan,
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
        "xtb_homo_eV": np.nan,
        "xtb_lumo_eV": np.nan,
        "xtb_gap_eV": np.nan,
        "xtb_somo_hartree": np.nan,
        "xtb_somo_eV": np.nan,
        "xtb_somo_occupation": np.nan,
        "xtb_n_orbitals_with_fractional_occupation": 0,
        "xtb_min_fractional_occupation": np.nan,
        "xtb_max_fractional_occupation": np.nan,
        "xtb_frontier_orbital_assignment_status": "",
        "xtb_n_orbitals": np.nan,
        "xtb_formal_charge_rdkit": np.nan,
        "xtb_requested_charge": int(charge),
        "xtb_uhf": int(uhf),
        "xtb_multiplicity": int(uhf) + 1,
        "xtb_electron_count": np.nan,
        "xtb_electronic_structure_type": "",
        "conformer_count_generated": 0,
        "conformer_count_xtb_evaluated": 0,
        "selected_conformer_energy": np.nan,
        "conformer_method": str(conformer_mode),
        "xtb_status": "",
        "descriptor_completeness": 0.0,
        "xtb_descriptor_n_expected": len(XTB_REQUIRED_DESCRIPTOR_COLUMNS),
        "xtb_descriptor_n_observed": 0,
    }

    if not xtb_python_available:
        base["xtb_status"] = "xtb_python_not_available"
        return base

    try:
        mol_for_charge = Chem.MolFromSmiles(str(smiles).strip())
        if mol_for_charge is not None:
            formal_charge, electron_count = qspr_xtb_formal_charge_and_electrons(mol_for_charge)
            base["xtb_formal_charge_rdkit"] = int(formal_charge)
            base["xtb_electron_count"] = int(electron_count)
            base["xtb_requested_charge"] = int(charge)
            base["xtb_uhf"] = int(uhf)
            base["xtb_multiplicity"] = int(uhf) + 1
            base["xtb_electronic_structure_type"] = (
                "closed_shell" if int(uhf) == 0 else "open_shell"
            )
        charge_ok, charge_status = qspr_xtb_validate_charge_uhf(
            mol_for_charge,
            charge=charge,
            uhf=uhf,
        )
        if not charge_ok:
            base["xtb_status"] = charge_status
            return base

        mol3d, prep_status = qspr_xtb_prepare_3d_mol_from_smiles(
            smiles=smiles,
            random_seed=random_seed,
            max_embed_attempts=max_embed_attempts,
            conformer_mode=conformer_mode,
            conformer_count=conformer_count,
            optimize_with_rdkit=optimize_with_rdkit
        )

        if mol3d is None:
            base["xtb_status"] = prep_status
            return base

        try:
            base["conformer_count_generated"] = int(
                mol3d.GetIntProp("_qspr_conformer_count_generated")
            )
        except Exception:
            pass
        try:
            base["selected_conformer_energy"] = float(
                mol3d.GetDoubleProp("_qspr_selected_conformer_energy")
            )
        except Exception:
            pass
        try:
            base["conformer_method"] = str(mol3d.GetProp("_qspr_conformer_mode"))
        except Exception:
            pass

        conformer_ids_to_run = []
        if str(conformer_mode).strip().lower() == "ensemble":
            try:
                energy_table = mol3d.GetProp("_qspr_conformer_energy_table")
            except Exception:
                energy_table = ""
            for item in str(energy_table).split(";"):
                if not item or ":" not in item:
                    continue
                conf_id_text, energy_text = item.split(":", 1)
                try:
                    conf_id = int(conf_id_text)
                    energy = float(energy_text)
                except Exception:
                    conf_id = int(conf_id_text) if conf_id_text.strip().isdigit() else -1
                    energy = np.inf
                if conf_id >= 0:
                    conformer_ids_to_run.append((energy, conf_id))
            conformer_ids_to_run = [
                conf_id
                for _, conf_id in sorted(conformer_ids_to_run, key=lambda item: item[0])
            ][:max(1, min(5, int(conformer_count or 5)))]

        if not conformer_ids_to_run:
            try:
                conformer_ids_to_run = [int(mol3d.GetIntProp("_qspr_selected_conformer_id"))]
            except Exception:
                conformer_ids_to_run = [-1]

        best_result = None
        best_energy = np.inf
        evaluated = 0

        for conf_id in conformer_ids_to_run:
            numbers, positions = qspr_xtb_mol_to_numbers_positions(mol3d, conf_id=conf_id)

            if numbers is None or positions is None:
                continue

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

            result_candidate = calc.singlepoint()
            evaluated += 1
            try:
                energy_candidate = float(result_candidate.get_energy())
            except Exception:
                energy_candidate = np.inf
            if energy_candidate < best_energy:
                best_energy = energy_candidate
                best_result = result_candidate

        if best_result is None:
            base["xtb_status"] = "no_3d_coordinates"
            return base

        base["conformer_count_xtb_evaluated"] = int(evaluated)
        if np.isfinite(best_energy):
            base["selected_conformer_energy"] = float(best_energy)
        result = best_result

        try:
            base["xtb_energy_hartree"] = float(result.get_energy())
        except Exception:
            pass

        try:
            gradient = np.asarray(result.get_gradient(), dtype=float)
            gradient_norm = float(np.linalg.norm(gradient))
            base["xtb_singlepoint_gradient_norm_on_rdkit_geometry"] = gradient_norm
            base["xtb_gradient_norm"] = gradient_norm
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
            orbital_diagnostics = qspr_xtb_orbital_diagnostics(result)
            base["xtb_homo_hartree"] = homo
            base["xtb_lumo_hartree"] = lumo
            base["xtb_gap_hartree"] = gap
            for key, value in orbital_diagnostics.items():
                base[key] = value
            if np.isfinite(homo):
                base["xtb_homo_eV"] = float(homo * HARTREE_TO_EV)
            if np.isfinite(lumo):
                base["xtb_lumo_eV"] = float(lumo * HARTREE_TO_EV)
            if np.isfinite(gap):
                base["xtb_gap_eV"] = float(gap * HARTREE_TO_EV)
            somo = base.get("xtb_somo_hartree", np.nan)
            if np.isfinite(somo):
                base["xtb_somo_eV"] = float(somo * HARTREE_TO_EV)
        except Exception:
            pass

        try:
            base["xtb_n_orbitals"] = int(result.get_number_of_orbitals())
        except Exception:
            pass

        observed_count = 0
        for col in XTB_REQUIRED_DESCRIPTOR_COLUMNS:
            try:
                if np.isfinite(float(base.get(col, np.nan))):
                    observed_count += 1
            except Exception:
                pass
        completeness = (
            observed_count / len(XTB_REQUIRED_DESCRIPTOR_COLUMNS)
            if XTB_REQUIRED_DESCRIPTOR_COLUMNS
            else 0.0
        )
        base["xtb_descriptor_n_observed"] = int(observed_count)
        base["descriptor_completeness"] = float(completeness)
        if observed_count == len(XTB_REQUIRED_DESCRIPTOR_COLUMNS):
            base["xtb_status"] = "complete"
        elif observed_count > 0:
            base["xtb_status"] = "partial"
        else:
            base["xtb_status"] = "singlepoint_only"
        return base

    except Exception as e:
        base["xtb_status"] = "failed"
        base["xtb_error"] = str(e)
        return base


def qspr_select_xtb_work_rows(
    data,
    target_col=None,
    max_molecules=None,
    sampling_strategy="first",
    random_seed=1,
):
    if data is None or not isinstance(data, pd.DataFrame):
        return pd.DataFrame()
    work = data.copy()
    if max_molecules is None:
        return work
    try:
        limit = int(max_molecules)
    except Exception:
        return work
    if limit <= 0 or len(work) <= limit:
        return work
    strategy = str(sampling_strategy or "first").strip().lower()
    if strategy in {"random", "reproducible_random"}:
        return work.sample(n=limit, random_state=int(random_seed)).sort_index()
    if strategy in {"property_range", "target_range", "uniform_target"} and target_col in work.columns:
        y = qspr_to_numeric(work[target_col])
        valid = work.loc[y.notna()].copy()
        if valid.empty:
            return work.head(limit).copy()
        valid_y = y.loc[valid.index]
        order = valid_y.sort_values().index.to_numpy()
        if len(order) <= limit:
            selected_idx = list(order)
        else:
            positions = np.linspace(0, len(order) - 1, num=limit)
            selected_idx = list(order[np.unique(np.rint(positions).astype(int))])
        if len(selected_idx) < limit:
            remaining = [idx for idx in work.index if idx not in set(selected_idx)]
            selected_idx.extend(remaining[: limit - len(selected_idx)])
        return work.loc[selected_idx[:limit]].sort_index().copy()
    return work.head(limit).copy()


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
    max_embed_attempts=20,
    conformer_mode="fast",
    conformer_count=1,
    optimize_with_rdkit=True,
    max_molecules=None,
    sampling_strategy="first",
    charge_col=None,
    uhf_col=None,
    multiplicity_col=None,
    auto_charge_from_smiles=True
):
    """
    Считает xTB-дескрипторы для таблицы.
    """
    if data is None or data.empty:
        raise ValueError("Нет данных для расчёта xTB-дескрипторов.")

    if smiles_col not in data.columns:
        raise ValueError(f"Колонка SMILES не найдена: {smiles_col}")

    data_for_xtb = data.copy()
    if "row_position" not in data_for_xtb.columns:
        data_for_xtb["row_position"] = list(data_for_xtb.index)
    if "source_index" not in data_for_xtb.columns:
        data_for_xtb["source_index"] = list(data_for_xtb.index)

    work = qspr_select_xtb_work_rows(
        data=data_for_xtb,
        target_col=target_col,
        max_molecules=max_molecules,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
    ).reset_index(drop=False).rename(columns={"index": "_original_index"})


    rows = []

    for local_i, row in work.iterrows():
        smiles = str(row.get(smiles_col, "")).strip()
        row_charge = charge
        row_uhf = uhf
        if charge_col and charge_col in row.index:
            try:
                row_charge = int(row.get(charge_col))
            except Exception:
                row_charge = charge
        elif auto_charge_from_smiles:
            try:
                mol_charge = Chem.MolFromSmiles(smiles)
                if mol_charge is not None:
                    row_charge = int(Chem.GetFormalCharge(mol_charge))
            except Exception:
                row_charge = charge
        if uhf_col and uhf_col in row.index:
            try:
                row_uhf = int(row.get(uhf_col))
            except Exception:
                row_uhf = uhf
        elif multiplicity_col and multiplicity_col in row.index:
            try:
                row_uhf = max(0, int(row.get(multiplicity_col)) - 1)
            except Exception:
                row_uhf = uhf

        desc = qspr_calc_xtb_descriptors_single(
            smiles=smiles,
            method=method,
            charge=row_charge,
            uhf=row_uhf,
            accuracy=accuracy,
            electronic_temperature=electronic_temperature,
            max_iterations=max_iterations,
            random_seed=random_seed + int(local_i),
            max_embed_attempts=max_embed_attempts,
            conformer_mode=conformer_mode,
            conformer_count=conformer_count,
            optimize_with_rdkit=optimize_with_rdkit
        )

        out = {
            "_original_index": int(row.get("row_position", local_i)),
            "row_position": int(row.get("row_position", local_i)),
            "source_index": row.get("source_index", row.get("_original_index", local_i)),
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

    xtb_service_cols = {
        "xtb_status",
        "xtb_error",
        "xtb_message",
        "xtb_gradient_norm",
        "xtb_descriptor_n_expected",
        "xtb_descriptor_n_observed",
        "xtb_frontier_orbital_assignment_status",
        "xtb_formal_charge_rdkit",
        "xtb_requested_charge",
        "xtb_uhf",
        "xtb_multiplicity",
        "xtb_electron_count",
        "xtb_electronic_structure_type",
    }
    descriptor_cols = [
        c for c in df_desc.columns
        if c.startswith("xtb_") and c not in xtb_service_cols
    ]

    complete_mask = (
        df_desc["xtb_status"].astype(str).str.lower().str.strip().isin(["complete", "ok"])
        if "xtb_status" in df_desc.columns
        else pd.Series(False, index=df_desc.index)
    )
    report = {
        "total": len(df_desc),
        "ok": int(complete_mask.sum()),
        "complete": int(complete_mask.sum()),
        "partial": int((df_desc["xtb_status"].astype(str) == "partial").sum()) if "xtb_status" in df_desc.columns else 0,
        "singlepoint_only": int((df_desc["xtb_status"].astype(str) == "singlepoint_only").sum()) if "xtb_status" in df_desc.columns else 0,
        "failed": int((~complete_mask).sum()) if "xtb_status" in df_desc.columns else len(df_desc),
        "status_counts": status_counts,
        "descriptor_cols": descriptor_cols,
        "method": method,
        "sampling_strategy": sampling_strategy,
        "max_molecules": max_molecules,
    }

    return df_desc, report

def qspr_descriptor_mode_settings(mode, desc_lists=None):
    """
    Возвращает настройки расчёта дескрипторов по режиму.

    Поддерживаемые режимы:
    - "rdkit_fast"
    - "mordred"
    - "mordred_padel_unique"
    - "max_coverage"

    Также поддерживаются старые русские названия из интерфейса.
    """
    if desc_lists is None:
        desc_lists = qspr_load_descriptor_lists()

    if desc_lists:
        rdkit_all_list = list(dict.fromkeys(desc_lists.get("rdkit_all", [])))
        mordred_unique_list = list(dict.fromkeys(desc_lists.get("mordred_unique", [])))
        padel_unique_list = list(dict.fromkeys(desc_lists.get("padel_unique", [])))
        padel_all_list = list(dict.fromkeys(desc_lists.get("padel_all", [])))
        padel_catalog_status = str(desc_lists.get("padel_catalog_status", "unknown"))
    else:
        rdkit_all_list = []
        mordred_unique_list = []
        padel_unique_list = []
        padel_all_list = []
        padel_catalog_status = "unknown"
    padel_catalog_names = padel_all_list or padel_unique_list
    padel_catalog_hash = hashlib.sha1(
        "\n".join(map(str, padel_catalog_names)).encode("utf-8", errors="replace")
    ).hexdigest()[:16] if padel_catalog_names else "none"

    mode_map = {
        # Русский
        "🚀 Максимальная скорость (RDKit)": "rdkit_fast",
        "👁️‍🗨️ Расширенный (Mordred)": "mordred",
        "⚡ Умный (Mordred + уникальные PaDEL)": "mordred_padel_unique",
        "⚡ Умный (уникальные Mordred + уникальные PaDEL)": "mordred_padel_unique",
        "🎯 Максимальная точность": "max_coverage",
        
        # Английский (из вашего интерфейса)
        "🚀 Maximum speed (RDKit)": "rdkit_fast",
        "👁️‍🗨️ Extended (Mordred)": "mordred",
        "⚡ Smart (Mordred + unique PaDEL)": "mordred_padel_unique",
        "⚡ Smart (unique Mordred + unique PaDEL)": "mordred_padel_unique",
        "🎯 Maximum accuracy": "max_coverage",
        
        # Казахский (добавьте ваши строки, если они отличаются)
        "🚀 Максималды жылдамдық (RDKit)": "rdkit_fast",
        "👁️‍🗨️ Кеңейтілген (Mordred)": "mordred",
        "⚡ Ақылды (Mordred + бірегей PaDEL)": "mordred_padel_unique",
        "🎯 Максималды дәлдік": "max_coverage",
        "max_accuracy": "max_coverage",
    }

    mode = qspr_normalize_descriptor_mode(mode_map.get(mode, mode))

    if mode == "rdkit_fast":
        return {
            "use_rdkit": True,
            "use_mordred": False,
            "use_padel": False,
            "rdkit_names": rdkit_all_list if rdkit_all_list else None,
            "mordred_names": [],
            "padel_names": [],
            "descriptor_catalog_status": padel_catalog_status,
            "descriptor_catalog_hash": padel_catalog_hash,
            "padel_available_count": len(padel_catalog_names),
        }

    if mode == "mordred":
        return {
            "use_rdkit": True,
            "use_mordred": True,
            "use_padel": False,
            "rdkit_names": rdkit_all_list if rdkit_all_list else None,
            "mordred_names": mordred_unique_list,
            "padel_names": [],
            "descriptor_catalog_status": padel_catalog_status,
            "descriptor_catalog_hash": padel_catalog_hash,
            "padel_available_count": len(padel_catalog_names),
        }

    if mode == "mordred_padel_unique":
        return {
            "use_rdkit": True,
            "use_mordred": True,
            "use_padel": True,
            "rdkit_names": rdkit_all_list if rdkit_all_list else None,
            "mordred_names": mordred_unique_list,
            "padel_names": padel_unique_list,
            "descriptor_catalog_status": padel_catalog_status,
            "descriptor_catalog_hash": padel_catalog_hash,
            "padel_available_count": len(padel_catalog_names),
        }

    if mode == "max_coverage":
        return {
            "use_rdkit": True,
            "use_mordred": True,
            "use_padel": True,
            "rdkit_names": None,
            "mordred_names": None,
            "padel_names": padel_all_list if padel_all_list else None,
            "descriptor_catalog_status": padel_catalog_status,
            "descriptor_catalog_hash": padel_catalog_hash,
            "padel_available_count": len(padel_catalog_names),
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
    preserve_constant_columns=False,
    max_descriptor_missing_fraction=0.30
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

    started_at = time.perf_counter()
    settings = qspr_descriptor_mode_settings(
        mode=mode,
        desc_lists=desc_lists
    )

    def _apply_allowed_names(current_names, allowed_names):
        if allowed_names is None:
            return current_names

        allowed_set = set(allowed_names)

        if current_names is None:
            return list(dict.fromkeys(allowed_names))

        return [
            name
            for name in current_names
            if name in allowed_set
        ]

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

    if not any([
        settings.get("use_rdkit", False),
        settings.get("use_mordred", False),
        settings.get("use_padel", False),
    ]):
        raise ValueError(
            "After descriptor filters were applied, no descriptor sources remain."
        )

    if settings["use_mordred"] and not mordred_available:
        raise ValueError("Mordred недоступен. Установите пакет mordred.")

    if settings["use_padel"] and not padel_available:
        raise ValueError("PaDEL недоступен. Установите пакет padelpy.")

    mordred_calc = None

    if settings["use_mordred"]:
        mordred_calc = qspr_make_mordred_calculator(settings["mordred_names"])
        if mordred_calc is None:
            settings["use_mordred"] = False

    all_desc = []
    valid_indices = []
    descriptor_quality_rows = []
    mordred_descriptor_error_rows = []
    mordred_descriptor_error_count = 0
    mordred_error_sample_limit = 500
    padel_descriptor_warning_rows = []

    invalid_smiles_count = 0
    padel_error_count = 0
    mordred_error_count = 0
    timing_totals = {"rdkit": 0.0, "mordred": 0.0, "padel": 0.0}

    smiles_list = data[smiles_col].astype(str).tolist()

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles).strip())

        if mol is None:
            invalid_smiles_count += 1
            continue

        desc_combined = {}
        module_counts = {"RDKit": 0, "Mordred": 0, "PaDEL": 0}
        module_status = {"RDKit": "not_requested", "Mordred": "not_requested", "PaDEL": "not_requested"}
        failure_reasons = []

        if settings["use_rdkit"]:
            step_started = time.perf_counter()
            rdkit_dict = qspr_calc_rdkit_descriptors_filtered(
                mol,
                allowed_names=settings["rdkit_names"]
            )
            timing_totals["rdkit"] += time.perf_counter() - step_started
            module_counts["RDKit"] = len(rdkit_dict)
            module_status["RDKit"] = "ok" if rdkit_dict else "failed"
            if not rdkit_dict:
                failure_reasons.append("RDKit returned no descriptors")
            desc_combined.update(qspr_prefix_descriptor_names("RDKit", rdkit_dict))

        if settings["use_mordred"]:
            step_started = time.perf_counter()
            mordred_dict, mordred_diagnostics = qspr_calc_mordred_descriptors_filtered(
                mol,
                mordred_calc,
                allowed_names=settings["mordred_names"],
                return_diagnostics=True,
            )
            timing_totals["mordred"] += time.perf_counter() - step_started
            for diagnostic in mordred_diagnostics:
                if diagnostic.get("status") != "ok":
                    mordred_descriptor_error_count += 1
                    if len(mordred_descriptor_error_rows) >= mordred_error_sample_limit:
                        continue
                    mordred_descriptor_error_rows.append({
                        "row_position": int(idx),
                        "source_index": data.index[idx],
                        "record_id": str(data.iloc[idx].get("record_id", f"record_{idx + 1:06d}")),
                        **diagnostic,
                    })
            module_counts["Mordred"] = len(mordred_dict)
            module_status["Mordred"] = "ok" if mordred_dict else "failed"
            if not mordred_dict:
                mordred_error_count += 1
                failure_reasons.append("Mordred returned no descriptors")
            desc_combined.update(qspr_prefix_descriptor_names("Mordred", mordred_dict))

            # Mordred error objects can retain calculation stacks. Reclaim them in
            # bounded batches so a long dataset does not pressure the page file.
            if (idx + 1) % 32 == 0:
                del mordred_diagnostics
                gc.collect()

        if settings["use_padel"]:
            step_started = time.perf_counter()
            padel_dict, padel_diagnostics = qspr_calc_padel_descriptors_filtered(
                smiles,
                allowed_names=settings["padel_names"],
                prefix_internal_names=True,
                return_diagnostics=True,
            )
            timing_totals["padel"] += time.perf_counter() - step_started
            for diagnostic in padel_diagnostics:
                padel_descriptor_warning_rows.append({
                    "row_position": int(idx),
                    "source_index": data.index[idx],
                    "record_id": str(data.iloc[idx].get("record_id", f"record_{idx + 1:06d}")),
                    **diagnostic,
                })

            if not padel_dict:
                padel_error_count += 1
                failure_reasons.append("PaDEL returned no descriptors")

            module_counts["PaDEL"] = len(padel_dict)
            module_status["PaDEL"] = "ok" if padel_dict else "failed"
            desc_combined.update(padel_dict)

        descriptor_calculation_success = bool(desc_combined)
        all_desc.append(desc_combined)
        valid_indices.append(idx)
        descriptor_quality_rows.append({
            "row_position": int(idx),
            "source_index": data.index[idx],
            "record_id": str(data.iloc[idx].get("record_id", f"record_{idx + 1:06d}")),
            "structure_valid": True,
            "descriptor_calculation_success": descriptor_calculation_success,
            "RDKit": module_status["RDKit"],
            "Mordred": module_status["Mordred"],
            "PaDEL": module_status["PaDEL"],
            "rdkit_descriptor_count": module_counts["RDKit"],
            "mordred_descriptor_count": module_counts["Mordred"],
            "padel_descriptor_count": module_counts["PaDEL"],
            "raw_descriptor_count": len(desc_combined),
            "descriptor_failure_reason": "; ".join(failure_reasons),
        })

    if not all_desc:
        raise ValueError("Нет валидных молекул для расчёта дескрипторов.")

    df_desc_raw = pd.DataFrame(all_desc)
    descriptor_quality_df = pd.DataFrame(descriptor_quality_rows)
    del all_desc
    del descriptor_quality_rows
    numeric_raw = df_desc_raw.apply(pd.to_numeric, errors="coerce")
    requested_descriptor_count = int(numeric_raw.shape[1])
    if requested_descriptor_count <= 0:
        raise ValueError("РќРё РѕРґРёРЅ РґРµСЃРєСЂРёРїС‚РѕСЂ РЅРµ Р±С‹Р» СЂР°СЃСЃС‡РёС‚Р°РЅ.")
    successful_counts = numeric_raw.notna().sum(axis=1).astype(int)
    success_fraction = successful_counts / float(requested_descriptor_count)
    missing_fraction = 1.0 - success_fraction
    del numeric_raw
    descriptor_quality_df["n_successful_descriptors"] = successful_counts.values
    descriptor_quality_df["n_requested_descriptors"] = requested_descriptor_count
    descriptor_quality_df["descriptor_success_fraction"] = success_fraction.values
    descriptor_quality_df["descriptor_missing_fraction"] = missing_fraction.values

    keep_descriptor_rows = (successful_counts > 0) & (
        missing_fraction <= float(max_descriptor_missing_fraction)
    )
    exclusion_reasons = []
    for success_count, miss_frac in zip(successful_counts, missing_fraction):
        if int(success_count) <= 0:
            exclusion_reasons.append("no_descriptors_calculated")
        elif float(miss_frac) > float(max_descriptor_missing_fraction):
            exclusion_reasons.append("too_many_missing_descriptors")
        else:
            exclusion_reasons.append("")
    descriptor_quality_df["descriptor_exclusion_reason"] = exclusion_reasons
    descriptor_quality_df["descriptor_row_included"] = keep_descriptor_rows.values

    if not keep_descriptor_rows.any():
        raise ValueError(
            "РџРѕСЃР»Рµ С„РёР»СЊС‚СЂР° СѓСЃРїРµС€РЅРѕСЃС‚Рё СЂР°СЃС‡С‘С‚Р° РґРµСЃРєСЂРёРїС‚РѕСЂРѕРІ РЅРµ РѕСЃС‚Р°Р»РѕСЃСЊ СЃС‚СЂРѕРє."
        )

    df_desc_raw = df_desc_raw.loc[keep_descriptor_rows.values].reset_index(drop=True)
    valid_indices = [
        idx for idx, keep in zip(valid_indices, keep_descriptor_rows.values)
        if bool(keep)
    ]

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
        "descriptor_catalog_status": settings.get("descriptor_catalog_status", "unknown"),
        "descriptor_catalog_hash": settings.get("descriptor_catalog_hash", "none"),
        "padel_available_count": int(settings.get("padel_available_count", 0) or 0),
        "mordred_error_count": mordred_error_count,
        "mordred_descriptor_error_count": int(mordred_descriptor_error_count),
        "mordred_descriptor_error_sample_count": int(len(mordred_descriptor_error_rows)),
        "mordred_descriptor_errors_truncated": bool(
            mordred_descriptor_error_count > len(mordred_descriptor_error_rows)
        ),
        "padel_descriptor_warning_count": int(len(padel_descriptor_warning_rows)),
        "descriptor_missing_fraction_limit": float(max_descriptor_missing_fraction),
        "descriptor_rows_excluded": int((~keep_descriptor_rows).sum()),
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "rdkit_elapsed_seconds": float(timing_totals["rdkit"]),
        "mordred_elapsed_seconds": float(timing_totals["mordred"]),
        "padel_elapsed_seconds": float(timing_totals["padel"]),
        "rdkit_requested_count": (
            len(settings["rdkit_names"])
            if settings.get("rdkit_names") is not None
            else None
        ),
        "mordred_requested_count": (
            len(settings["mordred_names"])
            if settings.get("mordred_names") is not None
            else (
                len(mordred_calc.descriptors)
                if mordred_calc is not None
                else 0
            )
        ),
        "padel_requested_count": (
            len(settings["padel_names"])
            if settings.get("padel_names") is not None
            else None
        ),
    }

    result = {
        "df_desc": df_desc.reset_index(drop=True),
        "X_all": X_all,
        "y_all": y_all,
        "valid_indices": valid_indices,
        "row_positions": valid_indices,
        "source_indices": [
            data.index[idx]
            for idx in valid_indices
        ],
        "record_ids": [
            str(data.iloc[idx].get("record_id", f"record_{idx + 1:06d}"))
            for idx in valid_indices
        ],
        "desc_names": desc_names,
        "descriptor_provenance": qspr_descriptor_provenance_table(desc_names),
        "descriptor_quality": descriptor_quality_df.reset_index(drop=True),
        "mordred_descriptor_errors": pd.DataFrame(mordred_descriptor_error_rows),
        "padel_descriptor_warnings": pd.DataFrame(padel_descriptor_warning_rows),
        "report": report
    }
    if settings.get("use_mordred", False):
        gc.collect()
    return result


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
    work["row_position"] = np.arange(len(work), dtype=int)
    if "source_index" not in work.columns:
        work["source_index"] = list(data.index)

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
    valid_indices = work_valid["row_position"].astype(int).tolist()
    source_indices = work_valid["source_index"].tolist()
    record_ids = (
        work_valid["record_id"].astype(str).tolist()
        if "record_id" in work_valid.columns
        else [f"record_{int(i) + 1:06d}" for i in valid_indices]
    )

    report = {
        "source": "custom_descriptors",
        "n_input_rows": len(data),
        "n_valid_rows": len(valid_indices),
        "n_descriptors": len(desc_names),
        "index_standard": "valid_indices_are_row_positions",
    }

    return {
        "df_desc": df_desc.reset_index(drop=True),
        "X_all": X_all,
        "y_all": y_all,
        "valid_indices": valid_indices,
        "row_positions": valid_indices,
        "source_indices": source_indices,
        "record_ids": record_ids,
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

    def _depth(self, expr):
        op = expr[0]
        if op in ["var", "const"]:
            return 1
        if op in ["sin", "cos", "log", "sqrt"]:
            return 1 + self._depth(expr[1])
        return 1 + max(self._depth(expr[1]), self._depth(expr[2]))

    def _used_variables(self, expr):
        op = expr[0]
        if op == "var":
            return {int(expr[1])}
        if op == "const":
            return set()
        if op in ["sin", "cos", "log", "sqrt"]:
            return self._used_variables(expr[1])
        return self._used_variables(expr[1]) | self._used_variables(expr[2])

    def _used_operations(self, expr):
        op = expr[0]
        if op in ["var", "const"]:
            return []
        if op in ["sin", "cos", "log", "sqrt"]:
            return [op] + self._used_operations(expr[1])
        return [op] + self._used_operations(expr[1]) + self._used_operations(expr[2])

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
        used_ops = self._used_operations(best_expr)
        used_vars = sorted(self._used_variables(best_expr))
        protected_ops = [op for op in used_ops if op in {"div", "log", "sqrt"}]
        self.formula_complexity_ = {
            "operation_count": int(self._size(best_expr)),
            "tree_depth": int(self._depth(best_expr)),
            "descriptor_count": int(len(used_vars)),
            "descriptor_indices": used_vars,
            "protected_operations": sorted(set(protected_ops)),
            "protected_math_note": (
                "This Augur symbolic fallback uses protected division, protected log1p(abs(x)), "
                "and protected sqrt(abs(x)) when those operators appear."
            ),
            "algorithm_label": "Augur evolutionary symbolic regression fallback",
            "is_classic_gep": False,
        }
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

    def get_formula_complexity(self):
        return dict(getattr(self, "formula_complexity_", {}))


# ------------------------------------------------------------------
# Модели

def qspr_default_model_params():
    """
    Параметры моделей по умолчанию.
    """
    return {
        "random_seed": 42,
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


def qspr_estimate_mlp_parameter_count(n_features, hidden_layer_sizes, n_outputs=1):
    try:
        layers = [
            int(x.strip())
            for x in str(hidden_layer_sizes).split(",")
            if x.strip()
        ]
    except Exception:
        layers = [100, 50]
    if not layers:
        layers = [100, 50]
    sizes = [int(n_features)] + layers + [int(n_outputs)]
    return int(sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1)))


def qspr_estimate_gpr_fit_time_seconds(n_samples, n_restarts=2):
    n = max(1, int(n_samples))
    restarts = max(1, int(n_restarts) + 1)
    return float(restarts * (n / 300.0) ** 3 * 2.0)


def qspr_model_applicability_guidance(model_name, n_samples, n_features, params=None, online_mode=None):
    params = dict(params or {})
    model_id = normalize_model_id(model_name)
    n_samples = int(n_samples)
    n_features = int(n_features)
    online_mode = qspr_core_is_online_mode() if online_mode is None else bool(online_mode)
    rows = []

    if model_id == "gpr":
        estimated_seconds = qspr_estimate_gpr_fit_time_seconds(n_samples, n_restarts=2)
        level = "info"
        if n_samples > 500:
            level = "warning"
        if online_mode and n_samples > 250:
            level = "error"
        rows.append({
            "level": level,
            "topic": "GPR applicability",
            "message": (
                "Gaussian Process Regression scales roughly as O(N^3). "
                "Recommended QSPR range: about 30-300 compounds; use caution above 500."
            ),
            "recommended_size": "30-300 compounds",
            "online_limit": 250,
            "estimated_fit_time_seconds": estimated_seconds,
        })

    if model_id == "mlp_regression":
        param_count = qspr_estimate_mlp_parameter_count(
            n_features,
            params.get("mlp_hidden_layer_sizes", "100,50"),
        )
        ratio = float(n_samples) / max(float(param_count), 1.0)
        level = "info"
        if ratio < 1.0 or n_samples < 150:
            level = "warning"
        rows.append({
            "level": level,
            "topic": "MLP data-to-parameter ratio",
            "message": (
                "MLP can be unstable on small QSPR datasets. "
                "Report repeated CV or seed stability when N is small relative to network parameters."
            ),
            "n_samples": n_samples,
            "n_parameters_estimated": int(param_count),
            "samples_per_parameter": ratio,
        })

    if model_id in {"gep_symbolic", "genetic_programming", "pysr"}:
        rows.append({
            "level": "warning",
            "topic": "Symbolic regression interpretation",
            "message": (
                "Symbolic formulas require complexity and domain checks. "
                "The internal Augur fallback is evolutionary symbolic regression, not a full GEP implementation."
            ),
        })

    return rows


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
    model_id = normalize_model_id(model_name)
    p = qspr_default_model_params()

    if params:
        p.update(params)

    if qspr_is_streamlit_cloud_runtime():
        p["rf_n_estimators"] = min(int(p.get("rf_n_estimators", 300)), 100)
        p["et_n_estimators"] = min(int(p.get("et_n_estimators", 300)), 100)
        p["xgb_n_estimators"] = min(int(p.get("xgb_n_estimators", 300)), 100)
        p["lightgbm_n_estimators"] = min(int(p.get("lightgbm_n_estimators", 300)), 100)
        p["catboost_iterations"] = min(int(p.get("catboost_iterations", 300)), 100)
        p["mlp_max_iter"] = min(int(p.get("mlp_max_iter", 1000)), 500)

    random_seed = int(p.get("random_seed", p.get("random_state", 42)))

    if model_id == "gpr" and n_samples is not None:
        gpr_online_limit = 250
        gpr_local_hard_limit = 3000
        if qspr_is_streamlit_cloud_runtime() and int(n_samples) > gpr_online_limit:
            raise ValueError(
                f"GPR is disabled above {gpr_online_limit} compounds in online mode "
                "because Gaussian Process fitting scales roughly as O(N^3)."
            )
        if int(n_samples) > gpr_local_hard_limit:
            raise ValueError(
                f"GPR is not allowed above {gpr_local_hard_limit} compounds. "
                "Use SVR, Random Forest, boosting, or reduce the dataset."
            )

    if model_id == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(p["rf_n_estimators"]),
            random_state=random_seed,
            n_jobs=qspr_n_jobs()
        )
    
    if model_id == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(p["et_n_estimators"]),
            max_depth=p["et_max_depth"] if p["et_max_depth"] is not None else None,
            min_samples_split=int(p["et_min_samples_split"]),
            min_samples_leaf=int(p["et_min_samples_leaf"]),
            max_features=p["et_max_features"],
            random_state=random_seed,
            n_jobs=qspr_n_jobs()
        )
    
    if model_id == "linear_regression":
        return LinearRegression()

    if model_id == "pls_regression":
        n_comp = int(p["pls_components"])

        if n_samples is not None and n_features is not None:
            n_comp = min(n_comp, n_samples - 1, n_features)
            n_comp = max(1, n_comp)

        return PLSRegression(
            n_components=n_comp
        )

    if model_id == "ridge_regression":
        return Ridge(
            alpha=float(p["ridge_alpha"])
        )

    if model_id == "lasso_regression":
        return Lasso(
            alpha=float(p["lasso_alpha"]),
            max_iter=50000,
            tol=1e-3,
            random_state=random_seed
        )

    if model_id == "elastic_net":
        return ElasticNet(
            alpha=float(p["elastic_alpha"]),
            l1_ratio=float(p["elastic_l1_ratio"]),
            max_iter=50000,
            tol=1e-3,
            random_state=random_seed
        )

    if model_id == "xgboost":
        if not xgboost_available:
            raise ValueError("XGBoost недоступен. Установите пакет xgboost.")

        return xgb.XGBRegressor(
            n_estimators=int(p["xgb_n_estimators"]),
            learning_rate=float(p["xgb_learning_rate"]),
            max_depth=int(p["xgb_max_depth"]),
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_seed,
            objective="reg:squarederror",
            n_jobs=qspr_n_jobs()
        )

    if model_id == "lightgbm":
        if not lightgbm_available:
            raise ValueError("LightGBM недоступен. Установите пакет lightgbm.")

        return lgb.LGBMRegressor(
            n_estimators=int(p["lightgbm_n_estimators"]),
            learning_rate=float(p["lightgbm_learning_rate"]),
            num_leaves=int(p["lightgbm_num_leaves"]),
            random_state=random_seed,
            n_jobs=qspr_n_jobs(),
            verbosity=-1
        )

    if model_id == "catboost":
        if not catboost_available:
            raise ValueError("CatBoost недоступен. Установите пакет catboost.")

        return CatBoostRegressor(
            iterations=int(p["catboost_iterations"]),
            learning_rate=float(p["catboost_learning_rate"]),
            depth=int(p["catboost_depth"]),
            loss_function="RMSE",
            random_seed=random_seed,
            thread_count=qspr_n_jobs(),
            verbose=False
        )

    if model_id == "svr":
        return SVR(
            kernel="rbf",
            C=float(p["svr_c"]),
            epsilon=float(p["svr_epsilon"]),
            gamma=p["svr_gamma"]
        )

    if model_id == "gpr":
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
            random_state=random_seed,
            n_restarts_optimizer=2
        )

    if model_id == "knn_regression":
        n_neighbors = int(p["knn_n_neighbors"])

        if n_samples is not None:
            n_neighbors = min(n_neighbors, max(1, n_samples))

        return KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=p["knn_weights"],
            metric="minkowski"
        )

    if model_id == "mlp_regression":
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
            random_state=random_seed,
            early_stopping=True
        )

    if model_id == "cart_regression":
        return DecisionTreeRegressor(
            max_depth=int(p["cart_max_depth"]),
            min_samples_leaf=int(p["cart_min_samples_leaf"]),
            random_state=random_seed
        )

    if model_id == "mars_like":
        return make_pipeline(
            PolynomialFeatures(
                degree=int(p["mars_degree"]),
                include_bias=False
            ),
            Ridge(
                alpha=float(p["mars_alpha"])
            )
        )

    if model_id == "spline_regression":
        return make_pipeline(
            SplineTransformer(
                n_knots=int(p["spline_n_knots"]),
                degree=int(p["spline_degree"]),
                include_bias=False
            ),
            Ridge(alpha=float(p["spline_alpha"]))
        )

    if model_id == "gam_regression":
        return make_pipeline(
            SplineTransformer(
                n_knots=int(p["gam_n_splines"]),
                degree=int(p["gam_degree"]),
                include_bias=False
            ),
            Ridge(alpha=float(p["gam_alpha"]))
        )

    if model_id == "gep_symbolic":
        return QSPRSymbolicRegressor(
            population_size=int(p["gep_population_size"]),
            generations=int(p["gep_generations"]),
            max_depth=int(p.get("gep_max_depth", 4)),
            random_state=random_seed
        )

    if model_id == "genetic_programming":
        return QSPRSymbolicRegressor(
            population_size=int(p["gp_population_size"]),
            generations=int(p["gp_generations"]),
            max_depth=int(p["gp_max_depth"]),
            random_state=random_seed
        )

    if model_id == "pysr":
        if not pysr_available:
            raise ValueError(
                "PySR недоступен. Установите пакет pysr и настройте Julia."
            )

        from pysr import PySRRegressor

        return PySRRegressor(
            niterations=int(p["pysr_niterations"]),
            populations=int(p["pysr_populations"]),
            maxsize=int(p["pysr_maxsize"]),
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp"],
            model_selection="best",
            random_state=random_seed,
            parallelism="serial",
            progress=False,
            verbosity=0
        )

    if model_id == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor(
            n_estimators=int(p["adaboost_n_estimators"]),
            learning_rate=float(p["adaboost_learning_rate"]),
            random_state=random_seed
        )

    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=int(p.get("hgb_max_iter", 300)),
            learning_rate=float(p.get("hgb_learning_rate", 0.1)),
            max_depth=int(p.get("hgb_max_depth", 0)) if p.get("hgb_max_depth") is not None else None,
            min_samples_leaf=int(p.get("hgb_min_samples_leaf", 20)),
            l2_regularization=float(p.get("hgb_l2_regularization", 0.0)),
            random_state=random_seed
        )
    
    if model_id == "stacking_regressor":
        estimators = []
        stacking_jobs = qspr_search_n_jobs()

        estimators.append(
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=int(p["rf_n_estimators"]),
                    random_state=random_seed,
                    n_jobs=stacking_jobs
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
                        random_state=random_seed,
                        objective="reg:squarederror",
                        n_jobs=stacking_jobs
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
                        random_state=random_seed,
                        n_jobs=stacking_jobs,
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
                        random_seed=random_seed,
                        thread_count=stacking_jobs,
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
            min_inner_train_size = int(n_samples) - int(np.ceil(int(n_samples) / cv_value))
            if int(n_samples) < 8 or min_inner_train_size < 6:
                raise ValueError(
                    "Stacking requires more compounds for a meaningful internal CV. "
                    f"n={int(n_samples)}, cv={cv_value}, minimum inner train fold={min_inner_train_size}."
                )
        stacking_cv = KFold(
            n_splits=cv_value,
            shuffle=True,
            random_state=random_seed,
        )

        model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=stacking_cv,
            passthrough=bool(p["stacking_passthrough"]),
            n_jobs=qspr_search_n_jobs()
        )
        model.augur_evaluation_note_ = (
            "Stacking is intended for final training. Honest performance "
            "assessment requires nested CV with an outer validation loop; "
            "ordinary reuse of the same CV can be optimistic."
        )
        return model

    if model_id == "voting_regressor":
        estimators = [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=int(p["rf_n_estimators"]),
                    random_state=random_seed,
                    n_jobs=qspr_n_jobs()
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
                    random_state=random_seed,
                    n_jobs=qspr_n_jobs()
                )
            ),
            ("ridge", Ridge(alpha=float(p["ridge_alpha"]))),
        ]
        weights = [
            float(p["voting_rf_weight"]),
            float(p["voting_extra_trees_weight"]),
            float(p["voting_ridge_weight"]),
        ]
        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Voting Regressor weights must be non-negative. "
                "Negative weights are not enabled in this interface."
            )
        if sum(weights) <= 0:
            raise ValueError(
                "Хотя бы один вес Voting Regressor должен быть больше нуля."
            )
        return VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=qspr_n_jobs()
        )

    raise ValueError(f"Неизвестная модель: {model_name}")


def qspr_make_descriptor_selector(selector_config=None):
    if not selector_config:
        return None
    method = str(selector_config.get("method", "fast")).strip().lower()
    no_selection = method in {"none", "no_selection", "без отбора"}
    return QSPRDescriptorSelector(
        desc_names=selector_config.get("desc_names"),
        method=selector_config.get("method", "fast"),
        max_features=int(selector_config.get("max_features", 50)),
        remove_constant=False if no_selection else bool(selector_config.get("remove_constant", True)),
        remove_correlated=False if no_selection else bool(selector_config.get("remove_correlated", True)),
        corr_threshold=float(selector_config.get("corr_threshold", 0.95)),
        lasso_alpha=float(selector_config.get("lasso_alpha", 0.01)),
        rf_n_estimators=int(selector_config.get("rf_n_estimators", 300)),
        rfe_step=float(selector_config.get("rfe_step", 0.2)),
        random_state=int(selector_config.get("random_state", 42)),
    )


def qspr_make_model_pipeline(model, scale=True, selector_config=None):
    """Build a leakage-safe descriptor preprocessing and model pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    selector = qspr_make_descriptor_selector(selector_config)
    if selector is not None:
        steps.append(("preselect", selector))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def qspr_min_inner_train_size(n_samples, cv):
    n_samples = int(n_samples)
    cv = max(2, min(int(cv), max(2, n_samples)))
    return max(1, n_samples - int(np.ceil(n_samples / cv)))


def qspr_filter_param_grid_for_cv(model_name, param_grid, n_samples, n_features, cv):
    filtered = {key: list(value) for key, value in (param_grid or {}).items()}
    min_inner_train = qspr_min_inner_train_size(n_samples, cv)
    model_id = normalize_model_id(model_name)

    if model_id == "pls_regression" and "model__n_components" in filtered:
        max_comp = max(1, min(int(n_features), int(min_inner_train) - 1))
        filtered["model__n_components"] = [
            int(v) for v in filtered["model__n_components"]
            if int(v) <= max_comp
        ] or [1]

    if model_id == "knn_regression" and "model__n_neighbors" in filtered:
        max_neighbors = max(1, int(min_inner_train))
        filtered["model__n_neighbors"] = [
            int(v) for v in filtered["model__n_neighbors"]
            if int(v) <= max_neighbors
        ] or [1]

    return filtered


def qspr_get_param_grid(model_name):
    """
    Небольшие безопасные сетки гиперпараметров.
    Сделаны компактными, чтобы не зависать на маленьких QSPR-выборках.
    """
    model_id = normalize_model_id(model_name)

    if model_id == "random_forest":
        return {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_leaf": [1, 2, 3],
        }

    if model_id == "extra_trees":
        return {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 3, 5],
            "model__min_samples_leaf": [1, 2, 3],
        }

    if model_id == "ridge_regression":
        return {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        }

    if model_id == "lasso_regression":
        return {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        }

    if model_id == "elastic_net":
        return {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    if model_id == "svr":
        return {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            "model__gamma": ["scale", "auto"],
        }

    if model_id == "knn_regression":
        return {
            "model__n_neighbors": [2, 3, 5, 7, 10],
            "model__weights": ["uniform", "distance"],
        }

    if model_id == "pls_regression":
        return {
            "model__n_components": [1, 2, 3, 4, 5, 8, 10],
        }

    if model_id == "mlp_regression":
        return {
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.0005, 0.001, 0.005],
        }

    if model_id == "spline_regression":
        return {
            "model__splinetransformer__n_knots": [3, 5, 7],
            "model__splinetransformer__degree": [2, 3],
            "model__ridge__alpha": [0.01, 0.1, 1.0, 10.0],
        }

    if model_id == "gam_regression":
        return {
            "model__splinetransformer__n_knots": [4, 6, 8],
            "model__splinetransformer__degree": [2, 3],
            "model__ridge__alpha": [0.1, 1.0, 10.0, 100.0],
        }

    if model_id == "adaboost":
        return {
            "model__n_estimators": [50, 100, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        }
    
    if model_id == "hist_gradient_boosting":
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
        random_state=42,
        max_correlation_pool=2500,
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
        self.max_correlation_pool = max_correlation_pool

    def _normalize_method(self):
        method = str(self.method).strip().lower()

        aliases = {
            "без отбора": "none",
            "none": "none",
            "no_selection": "none",
            "basic_cleaning": "basic_cleaning",
            "basic cleaning": "basic_cleaning",
            "cleaning": "basic_cleaning",
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

        except Exception as exc:
            # fallback: если статистический скоринг не сработал,
            # используем дисперсию признака.
            if not hasattr(self, "score_fallbacks_"):
                self.score_fallbacks_ = []
            self.score_fallbacks_.append({
                "requested_method": method,
                "actual_method": "feature_variance",
                "reason": f"{type(exc).__name__}: {exc}",
            })
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
        max_pool = int(getattr(self, "max_correlation_pool", 2500) or 2500)
        max_pool = max(1, max_pool)
        self.correlation_filter_warning_ = ""
        self.correlation_filter_initial_feature_count_ = int(len(candidate_indices))
        self.correlation_filter_pool_feature_count_ = int(len(candidate_indices))
        if len(candidate_indices) > max_pool:
            candidate_indices = sorted(
                candidate_indices,
                key=lambda i: float(scores[i]) if i < len(scores) else 0.0,
                reverse=True,
            )[:max_pool]
            self.correlation_filter_pool_feature_count_ = int(len(candidate_indices))
            self.correlation_filter_warning_ = "correlation_filter_prelimited_feature_pool"

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
        self.requested_method_ = method
        self.actual_method_ = method
        self.fallback_reason_ = ""

        candidate_indices = list(candidate_indices)

        if not candidate_indices:
            return []

        max_features = int(self.max_features)
        max_features = max(1, min(max_features, len(candidate_indices)))

        X_candidate = X[:, candidate_indices]

        if method == "none":
            return candidate_indices

        if method == "basic_cleaning":
            return candidate_indices

        if method in ["fast", "f_regression", "mutual_info"]:
            scoring_method = "mutual_info" if method == "mutual_info" else "f_regression"
            fallback_count_before = len(getattr(self, "score_fallbacks_", []))
            local_scores = self._target_scores(X_candidate, y, method=scoring_method)
            if len(getattr(self, "score_fallbacks_", [])) > fallback_count_before:
                last_fallback = self.score_fallbacks_[-1]
                self.actual_method_ = f"{scoring_method}_variance_fallback"
                self.fallback_reason_ = str(last_fallback.get("reason", ""))

            order_local = np.argsort(local_scores)[::-1]
            selected_local = order_local[:max_features]

            return [candidate_indices[i] for i in selected_local]

        if method == "lasso":
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_candidate)

                lasso = Lasso(
                    alpha=float(self.lasso_alpha),
                    max_iter=50000,
                    tol=1e-3,
                    random_state=self.random_state
                )
                lasso.fit(X_scaled, y)

                coefs = np.abs(np.asarray(lasso.coef_, dtype=float))
                nonzero = np.where(coefs > 1e-12)[0]

                if len(nonzero) > 0:
                    order_local = nonzero[np.argsort(coefs[nonzero])[::-1]]
                else:
                    self.actual_method_ = "f_regression_fallback"
                    self.fallback_reason_ = "lasso selected no nonzero coefficients"
                    local_scores = self._target_scores(X_candidate, y, method="f_regression")
                    order_local = np.argsort(local_scores)[::-1]

                selected_local = order_local[:max_features]

                return [candidate_indices[i] for i in selected_local]

            except Exception as exc:
                self.actual_method_ = "f_regression_fallback"
                self.fallback_reason_ = f"lasso failed: {type(exc).__name__}"
                local_scores = self._target_scores(X_candidate, y, method="f_regression")
                order_local = np.argsort(local_scores)[::-1]
                selected_local = order_local[:max_features]

                return [candidate_indices[i] for i in selected_local]

        if method == "random_forest":
            try:
                rf = RandomForestRegressor(
                    n_estimators=int(self.rf_n_estimators),
                    random_state=self.random_state,
                    n_jobs=qspr_n_jobs()
                )
                rf.fit(X_candidate, y)

                importances = np.asarray(rf.feature_importances_, dtype=float)
                importances = np.where(np.isfinite(importances), importances, 0.0)

                order_local = np.argsort(importances)[::-1]
                selected_local = order_local[:max_features]

                return [candidate_indices[i] for i in selected_local]

            except Exception as exc:
                self.actual_method_ = "f_regression_fallback"
                self.fallback_reason_ = f"random_forest failed: {type(exc).__name__}"
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

            except Exception as exc:
                self.actual_method_ = "f_regression_fallback"
                self.fallback_reason_ = f"rfe_ridge failed: {type(exc).__name__}"
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
        method = self._normalize_method()
        self.score_fallbacks_ = []

        if method == "none":
            constant_mask = np.ones(n_features, dtype=bool)
        else:
            constant_mask = self._constant_mask(X)

        nonconstant_indices = [
            i for i in range(n_features)
            if bool(constant_mask[i])
        ]

        if not nonconstant_indices:
            raise ValueError(
                "No descriptors remain after removing constant descriptors. "
                "Use feature-selection method 'none' only if you intentionally "
                "want to keep constant descriptors."
            )

        target_scores = self._target_scores(X, y, method="f_regression")
        mi_scores = self._target_scores(X, y, method="mutual_info")

        if method == "none":
            corr_kept_indices, removed_by_corr = nonconstant_indices, {}
        else:
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
            raise ValueError(
                f"Descriptor selection method '{method}' selected no descriptors."
            )

        self.selected_indices_ = list(final_indices)
        self.selected_names_ = [desc_names[i] for i in self.selected_indices_]
        self.selected_feature_order_policy_ = (
            "selection_order_by_target_score_or_selector_rank; "
            "prediction must use selected_names_ in this exact order"
        )

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
        self.selection_table_["requested_method"] = getattr(
            self, "requested_method_", method
        )
        self.selection_table_["actual_method"] = getattr(
            self, "actual_method_", method
        )
        self.selection_table_["fallback_reason"] = getattr(
            self, "fallback_reason_", ""
        )

        self.selection_summary_ = {
            "method": method,
            "requested_method": getattr(self, "requested_method_", method),
            "actual_method": getattr(self, "actual_method_", method),
            "fallback_reason": getattr(self, "fallback_reason_", ""),
            "scoring_fallbacks": list(getattr(self, "score_fallbacks_", [])),
            "n_samples": int(n_samples),
            "n_features_initial": int(n_features),
            "n_after_constant_filter": int(len(nonconstant_indices)),
            "n_removed_constant": int(n_features - len(nonconstant_indices)),
            "n_after_correlation_filter": int(len(corr_kept_indices)),
            "n_removed_correlated": int(len(nonconstant_indices) - len(corr_kept_indices)),
            "n_selected_final": int(len(self.selected_indices_)),
            "correlation_filter_initial_feature_count": int(getattr(self, "correlation_filter_initial_feature_count_", len(nonconstant_indices))),
            "correlation_filter_pool_feature_count": int(getattr(self, "correlation_filter_pool_feature_count_", len(corr_kept_indices))),
            "correlation_filter_warning": str(getattr(self, "correlation_filter_warning_", "")),
            "selected_feature_order_policy": self.selected_feature_order_policy_,
            "corr_threshold": float(self.corr_threshold),
            "remove_constant": bool(self.remove_constant) and method != "none",
            "remove_correlated": bool(self.remove_correlated) and method != "none",
            "selection_contract": (
                "NONE: no descriptor removal, no correlation filter, no max_features limit"
                if method == "none"
                else "BASIC_CLEANING/SUPERVISED_SELECTION: preprocessing filters may apply"
            ),
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
    rfe_step=0.2,
    random_state=None,
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
    params = dict(params or {})
    random_seed = int(
        random_state
        if random_state is not None
        else params.get("random_seed", params.get("random_state", 42))
    )

    n_samples, n_features = X.shape

    if n_samples < 5:
        raise ValueError("Для автоотбора и оптимизации желательно минимум 5 веществ.")

    finite_y = y[np.isfinite(y)]
    if len(finite_y) == 0 or float(np.nanstd(finite_y)) <= 1e-12:
        raise ValueError("Target property has no variation; regression QSPR is not defined.")
    unique_y, counts_y = np.unique(finite_y, return_counts=True)
    dominant_y_fraction = float(np.max(counts_y) / len(finite_y)) if len(finite_y) else np.nan
    y_quality = {
        "n_objects": int(n_samples),
        "n_finite_y": int(len(finite_y)),
        "n_unique_y": int(len(unique_y)),
        "dominant_y_fraction": dominant_y_fraction,
        "warnings": [],
    }
    if len(unique_y) < 5:
        y_quality["warnings"].append("fewer_than_5_unique_target_values")
    if np.isfinite(dominant_y_fraction) and dominant_y_fraction > 0.80:
        y_quality["warnings"].append("one_target_value_exceeds_80_percent")

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
    inner_cv = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_seed,
    )
    outer_cv = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_seed + 1,
    )

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
        random_state=random_seed
    )

    # Для создания базовой модели ориентируемся на верхнюю оценку числа признаков.
    n_model_features = max(1, min(max_features_effective, n_features))

    base_model = qspr_create_regression_model(
        model_name=model_name,
        n_samples=n_samples,
        n_features=n_model_features,
        params=params
    )

    steps = [("imputer", SimpleImputer(strategy="median"))]
    steps.append(("preselect", selector))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", base_model))

    pipe = Pipeline(steps)

    param_grid = {}

    if optimize_hyperparams:
        param_grid = qspr_get_param_grid(model_name)
        param_grid = qspr_filter_param_grid_for_cv(
            model_name,
            param_grid,
            n_samples=n_samples,
            n_features=n_model_features,
            cv=cv,
        )

    scoring = "neg_root_mean_squared_error"

    if optimize_hyperparams and param_grid:
        if search_method == "random":
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                n_iter=int(n_iter),
                scoring=scoring,
                cv=inner_cv,
                random_state=random_seed,
                n_jobs=qspr_search_n_jobs(),
                error_score=np.nan
            )
        else:
            search = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=qspr_search_n_jobs(),
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

    # Кросс-валидационный прогноз. При оптимизации гиперпараметров используем
    # nested-подход: search заново подбирается внутри каждой внешней складки.
    try:
        cv_estimator = best_model
        if optimize_hyperparams and param_grid:
            if search_method == "random":
                cv_estimator = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_grid,
                    n_iter=int(n_iter),
                    scoring=scoring,
                    cv=inner_cv,
                    random_state=random_seed,
                    n_jobs=qspr_search_n_jobs(),
                    error_score=np.nan
                )
            else:
                cv_estimator = GridSearchCV(
                    pipe,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=inner_cv,
                    n_jobs=qspr_search_n_jobs(),
                    error_score=np.nan
                )

        y_cv_pred = cross_val_predict(
            cv_estimator,
            X,
            y,
            cv=outer_cv,
            n_jobs=qspr_n_jobs()
        )
        cv_metrics = qspr_metrics(y, y_cv_pred)
        cv_status = "ok"
        cv_error_type = ""
        cv_error_message = ""
        failed_stage = ""
    except Exception as exc:
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
        cv_status = "failed"
        cv_error_type = type(exc).__name__
        cv_error_message = str(exc)
        failed_stage = "cross_val_predict"

    # Финальный fit на всей выборке.
    best_model.fit(X, y)
    model_validation_status = (
        "MODEL_FITTED_AND_CV_VALIDATED"
        if cv_status == "ok"
        else "MODEL_FITTED_BUT_NOT_VALIDATED"
    )

    preselector = best_model.named_steps.get("preselect")
    selected_desc_names = list(getattr(preselector, "selected_names_", []))
    selection_table = getattr(preselector, "selection_table_", pd.DataFrame())
    selection_summary = getattr(preselector, "selection_summary_", {})
    p_selected = int(len(selected_desc_names))
    residual_degrees_of_freedom = int(n_samples - p_selected - 1)
    feature_ratio_diagnostics = {
        "n": int(n_samples),
        "p": p_selected,
        "n_per_p": float(n_samples / p_selected) if p_selected > 0 else np.nan,
        "p_per_n": float(p_selected / n_samples) if n_samples > 0 else np.nan,
        "residual_degrees_of_freedom": residual_degrees_of_freedom,
        "warnings": [],
    }
    model_id_for_ratio = normalize_model_id(model_name)
    linear_like_models = {
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "elastic_net",
        "pls_regression",
    }
    if p_selected >= n_samples:
        feature_ratio_diagnostics["warnings"].append("p_ge_n_model_underdetermined")
    if model_id_for_ratio in linear_like_models and p_selected > n_samples / 5:
        feature_ratio_diagnostics["warnings"].append("linear_model_high_p_to_n_instability_risk")

    X_work = best_model.named_steps["imputer"].transform(X)
    X_selected = preselector.transform(X_work) if preselector is not None else X_work
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
        "target_quality": y_quality,
        "feature_ratio_diagnostics": feature_ratio_diagnostics,
        "X_selected": X_selected,
        "X_model_space": X_model_space,
        "y_cv_pred": y_cv_pred,
        "cv_metrics": cv_metrics,
        "cv_status": cv_status,
        "cv_error_type": cv_error_type,
        "cv_error_message": cv_error_message,
        "failed_stage": failed_stage,
        "model_validation_status": model_validation_status,
        "validation_warning": (
            ""
            if cv_status == "ok"
            else "Final model was fitted on all data, but cross-validation failed."
        ),
        "fit_metrics": fit_metrics,
        "best_params": best_params,
        "best_cv_rmse": best_cv_rmse,
        "best_cv_rmse_label": "Internal hyperparameter-search CV RMSE",
        "cv_metrics_label": (
            "Repeated CV prediction of selected configuration; not an independent external validation"
        ),
        "cv_splitter": f"KFold(shuffle=True, random_state={random_seed + 1})",
        "inner_cv_splitter": f"KFold(shuffle=True, random_state={random_seed})",
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

    model = qspr_create_regression_model(
        model_name,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        params=params
    )
    pipeline = qspr_make_model_pipeline(model, scale=scale)

    pipeline.fit(X, y)

    y_pred = np.ravel(pipeline.predict(X))

    metrics = qspr_metrics(y, y_pred)
    X_model = pipeline.named_steps["imputer"].transform(X)
    if scale and "scale" in pipeline.named_steps:
        X_model = pipeline.named_steps["scale"].transform(X_model)

    return {
        "model": pipeline,
        "scaler": None,
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
    scale=True,
    selector_config=None,
    stratify_y_quantiles=False
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
        stratify_labels = None
        stratification_note = "not_used"
        if stratify_y_quantiles:
            try:
                n_test = int(np.ceil(float(test_size) * n)) if float(test_size) < 1 else int(test_size)
                n_test = max(1, min(n - 1, n_test))
                max_bins = max(2, min(10, n_test, n - n_test))
                labels = pd.qcut(y, q=max_bins, labels=False, duplicates="drop")
                labels = np.asarray(labels)
                counts = pd.Series(labels).value_counts(dropna=False)
                if len(counts) >= 2 and int(counts.min()) >= 2:
                    stratify_labels = labels
                    stratification_note = f"quantile_bins={len(counts)}"
                else:
                    stratification_note = "fallback_random_small_or_sparse_bins"
            except Exception as exc:
                stratification_note = f"fallback_random_error: {exc}"

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )

    else:
        stratification_note = "manual_split"
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

    model = qspr_create_regression_model(
        model_name,
        n_samples=X_train.shape[0],
        n_features=X_train.shape[1],
        params=params
    )
    pipeline = qspr_make_model_pipeline(model, scale=scale, selector_config=selector_config)

    pipeline.fit(X_train, y_train)

    y_pred_train = np.ravel(pipeline.predict(X_train))
    y_pred_test = np.ravel(pipeline.predict(X_test))

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
        "model": pipeline,
        "scaler": None,

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

        "model_name": model_name,
        "preprocessing_pipeline": (
            "imputer -> preselect -> scale -> model"
            if selector_config else "imputer -> scale -> model"
        ),
        "selector_config": selector_config or {},
        "split_strategy": "stratified_y_quantiles" if stratify_y_quantiles and use_random else (
            "random" if use_random else "manual"
        ),
        "stratification_note": stratification_note,
    }


def qspr_seed_stability_holdout(
    X,
    y,
    model_name,
    seeds=None,
    valid_indices=None,
    smiles=None,
    test_size=0.2,
    params=None,
    scale=True,
    selector_config=None,
):
    seeds = [1, 7, 42, 101, 2026] if seeds is None else [int(seed) for seed in seeds]
    rows = []
    for seed in seeds:
        seed_params = dict(params or {})
        seed_params["random_seed"] = int(seed)
        try:
            result = qspr_holdout_validation(
                X=X,
                y=y,
                model_name=model_name,
                valid_indices=valid_indices,
                smiles=smiles,
                test_size=test_size,
                random_state=int(seed),
                use_random=True,
                manual_indices=None,
                params=seed_params,
                scale=scale,
                selector_config=selector_config,
            )
            metrics = result.get("metrics_test", {})
            rows.append({
                "seed": int(seed),
                "status": "ok",
                "test_R2": metrics.get("R2", np.nan),
                "test_RMSE": metrics.get("RMSE", np.nan),
                "test_MAE": metrics.get("MAE", np.nan),
                "test_MAPE_percent": metrics.get("MAPE_percent", np.nan),
            })
        except Exception as exc:
            rows.append({
                "seed": int(seed),
                "status": f"error: {exc}",
                "test_R2": np.nan,
                "test_RMSE": np.nan,
                "test_MAE": np.nan,
                "test_MAPE_percent": np.nan,
            })

    results_df = pd.DataFrame(rows)
    ok = results_df[results_df["status"] == "ok"].copy()
    summary = {
        "model_name": model_name,
        "seeds": seeds,
        "n_runs": int(len(seeds)),
        "n_ok": int(len(ok)),
        "test_R2_mean": float(ok["test_R2"].mean()) if not ok.empty else np.nan,
        "test_R2_std": float(ok["test_R2"].std(ddof=1)) if len(ok) > 1 else np.nan,
        "test_RMSE_mean": float(ok["test_RMSE"].mean()) if not ok.empty else np.nan,
        "test_RMSE_std": float(ok["test_RMSE"].std(ddof=1)) if len(ok) > 1 else np.nan,
        "test_MAE_mean": float(ok["test_MAE"].mean()) if not ok.empty else np.nan,
        "test_MAE_std": float(ok["test_MAE"].std(ddof=1)) if len(ok) > 1 else np.nan,
        "note": "Seed stability is a robustness diagnostic, not a replacement for cross-validation.",
    }
    return {"summary": summary, "results_df": results_df}


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
    random_state=42,
    selector_config=None
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
    if n < 2 or k < 2:
        raise ValueError("K-fold validation requires at least 2 objects after filtering and k >= 2.")

    min_train_size = n - int(np.ceil(n / k))

    model_cv = qspr_create_regression_model(
        model_name,
        n_samples=min_train_size,
        n_features=X.shape[1],
        params=params
    )

    pipe = qspr_make_model_pipeline(model_cv, scale=scale, selector_config=selector_config)

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
        "model_name": model_name,
        "preprocessing_pipeline": (
            "imputer -> preselect -> scale -> model"
            if selector_config else "imputer -> scale -> model"
        ),
        "selector_config": selector_config or {},
    }


def qspr_loo_validation(
    X,
    y,
    model_name,
    valid_indices=None,
    smiles=None,
    params=None,
    scale=True,
    selector_config=None
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

    pipe = qspr_make_model_pipeline(model_loo, scale=scale, selector_config=selector_config)

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
        "model_name": model_name,
        "preprocessing_pipeline": (
            "imputer -> preselect -> scale -> model"
            if selector_config else "imputer -> scale -> model"
        ),
        "selector_config": selector_config or {},
    }

# =============================================================================
# Distance-based hold-out stress test (not strict external validation)
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
    random_state=42,
    selector_config=None
):
    """
    Distance-based hold-out stress test: обучает модель на (1-fraction) веществ,
    наиболее близких друг к другу, и тестирует на fraction самых удалённых.

    Параметры:
    - fraction: доля веществ для теста (0.2 = 20% самых удалённых)
    - n_repeats: число повторений с разными начальными условиями
      (для устойчивости, т.к. выборка удалённых точек может зависеть от случайности при жадном отборе)
    - distance_metric: 'euclidean' или 'mahalanobis' (пока реализован только евклидов)
    """
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
    distance_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X_scaled = distance_preprocessor.fit_transform(X)
    distance_metric_norm = str(distance_metric or "euclidean").strip().lower()
    if distance_metric_norm not in {"euclidean", "mahalanobis", "cosine"}:
        raise ValueError("distance_metric must be 'euclidean', 'mahalanobis', or 'cosine'.")

    covariance_inv = None
    if distance_metric_norm == "mahalanobis":
        cov = np.cov(X_scaled, rowvar=False)
        cov = np.atleast_2d(cov)
        cov = cov + np.eye(cov.shape[0]) * 1e-8
        covariance_inv = np.linalg.pinv(cov)

    def distance_to_point(matrix, point):
        if distance_metric_norm == "euclidean":
            return np.linalg.norm(matrix - point, axis=1)
        if distance_metric_norm == "cosine":
            point_norm = np.linalg.norm(point)
            matrix_norm = np.linalg.norm(matrix, axis=1)
            denom = np.maximum(matrix_norm * max(point_norm, 1e-12), 1e-12)
            similarity = np.dot(matrix, point) / denom
            return 1.0 - np.clip(similarity, -1.0, 1.0)
        diff = matrix - point
        return np.sqrt(np.maximum(np.sum((diff @ covariance_inv) * diff, axis=1), 0.0))

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
                d = distance_to_point(X_scaled, X_scaled[idx])
                distances = np.minimum(distances, d)
            next_idx = np.argmax(distances)
            indices.append(int(next_idx))
        return indices

    all_metrics = []
    all_tables = []
    all_test_sets = []

    for rep in range(n_repeats):
        seed = random_state + rep
        test_indices_local = farthest_point_sampling(X_scaled, test_size, random_seed=seed)
        train_indices = [i for i in range(n) if i not in test_indices_local]
        all_test_sets.append(tuple(sorted(int(i) for i in test_indices_local)))

        X_train = X[train_indices]
        X_test = X[test_indices_local]
        y_train = y[train_indices]
        y_test = y[test_indices_local]

        model = qspr_create_regression_model(
            model_name,
            n_samples=X_train.shape[0],
            n_features=X_train.shape[1],
            params=params
        )
        model = qspr_make_model_pipeline(model, scale=scale, selector_config=selector_config)
        model.fit(X_train, y_train)
        y_pred_train = np.ravel(model.predict(X_train))
        y_pred_test = np.ravel(model.predict(X_test))

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
    unique_test_splits = len(set(all_test_sets))
    jaccard_values = []
    for i in range(len(all_test_sets)):
        left = set(all_test_sets[i])
        for j in range(i + 1, len(all_test_sets)):
            right = set(all_test_sets[j])
            union = left | right
            jaccard_values.append(len(left & right) / len(union) if union else np.nan)
    selection_counts = {int(i): 0 for i in range(n)}
    for split in all_test_sets:
        for idx in split:
            selection_counts[int(idx)] = selection_counts.get(int(idx), 0) + 1
    test_selection_frequency = pd.DataFrame([
        {
            "local_index": int(idx),
            "original_index": int(valid_indices[idx]) if idx < len(valid_indices) else int(idx),
            "smiles": smiles[idx] if idx < len(smiles) else "",
            "test_count": int(count),
            "test_frequency": float(count) / max(int(n_repeats), 1),
        }
        for idx, count in selection_counts.items()
    ]).sort_values(["test_count", "local_index"], ascending=[False, True]).reset_index(drop=True)
    test_split_jaccard = pd.DataFrame([
        {
            "repeat_a": i + 1,
            "repeat_b": j + 1,
            "jaccard_similarity": (
                len(set(all_test_sets[i]) & set(all_test_sets[j]))
                / len(set(all_test_sets[i]) | set(all_test_sets[j]))
            ) if (set(all_test_sets[i]) | set(all_test_sets[j])) else np.nan,
        }
        for i in range(len(all_test_sets))
        for j in range(i + 1, len(all_test_sets))
    ])
    summary = {
        'validation_mode': 'distance_based_holdout_stress_test',
        'validation_label': 'Distance-based hold-out stress test',
        'is_external_validation': False,
        'methodology_note': (
            'Test objects are selected from the original dataset by global descriptor-space distance. '
            'This is a stress test, not strict external validation on an independent dataset.'
        ),
        'test_selection_geometry': 'global_descriptor_distribution',
        'distance_metric': distance_metric_norm,
        'distance_preprocessing': 'median_imputer_plus_standard_scaler_fit_on_all_X_for_test_selection',
        'unique_test_splits': int(unique_test_splits),
        'mean_test_split_jaccard': float(np.nanmean(jaccard_values)) if jaccard_values else np.nan,
        'max_test_split_jaccard': float(np.nanmax(jaccard_values)) if jaccard_values else np.nan,
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
        'validation_mode': 'distance_based_holdout_stress_test',
        'is_external_validation': False,
        'distance_metric': distance_metric_norm,
        'test_split_jaccard': test_split_jaccard,
        'test_selection_frequency': test_selection_frequency,
        'unique_test_splits': int(unique_test_splits),
        'preprocessing_pipeline': (
            "imputer -> preselect -> scale -> model"
            if selector_config else "imputer -> scale -> model"
        ),
        'selector_config': selector_config or {},
        'fraction': fraction,
        'n_repeats': n_repeats
    }

def qspr_consensus_predictions(
    models_scalers_dict,
    X,
    model_names=None,
    method="equal_mean",
    weights=None,
    trim_fraction=0.1,
):
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
    n_models = int(preds.shape[0])
    method = str(method or "equal_mean").strip().lower()

    if method == "median":
        mean_pred = np.nanmedian(preds, axis=0)
        consensus_method = "median"
    elif method == "trimmed_mean":
        trim = min(max(float(trim_fraction), 0.0), 0.45)
        sorted_preds = np.sort(preds, axis=0)
        cut = int(np.floor(n_models * trim))
        if cut > 0 and (n_models - 2 * cut) >= 1:
            mean_pred = np.nanmean(sorted_preds[cut:n_models - cut, :], axis=0)
        else:
            mean_pred = np.nanmean(sorted_preds, axis=0)
        consensus_method = "trimmed_mean"
    elif method in {"weighted_cv_rmse", "weighted_holdout_rmse"} and weights:
        weight_values = np.asarray(
            [float(weights.get(name, 0.0)) for name in names],
            dtype=float,
        )
        weight_values[~np.isfinite(weight_values)] = 0.0
        if np.sum(weight_values) <= 0:
            weight_values = np.ones(n_models, dtype=float)
        weight_values = weight_values / np.sum(weight_values)
        mean_pred = np.average(preds, axis=0, weights=weight_values)
        consensus_method = method
    else:
        mean_pred = np.mean(preds, axis=0)
        consensus_method = "equal_mean"

    if n_models >= 2:
        std_pred = np.std(preds, axis=0, ddof=1)
    else:
        std_pred = np.zeros(preds.shape[1], dtype=float)

    df = pd.DataFrame()
    for i, name in enumerate(names):
        df[f'pred_{name}'] = preds[i]
    df['Consensus_mean'] = mean_pred
    df['Consensus_std'] = std_pred
    df['Intermodel_disagreement'] = std_pred
    df['Consensus_method'] = consensus_method
    df['Consensus_note'] = (
        "Intermodel_disagreement is spread between selected models, not a "
        "confidence interval. Low disagreement does not guarantee correctness."
    )

    return df

# ------------------------------------------------------------------
# Коэффициенты моделей

def qspr_extract_model_coefficients(model, desc_names, model_name):
    """
    Извлекает коэффициенты для линейных моделей и формулу для Symbolic Regression.
    """
    model_id = normalize_model_id(model_name)

    if (
        model_id in ["gep_symbolic", "genetic_programming"]
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

    if model_id == "pysr" and hasattr(model, "sympy"):
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

    if model_id == "pls_regression":
        coef_col = "Коэффициент PLS"
    else:
        coef_col = "Коэффициент"

    coef_table = pd.DataFrame({
        "Дескриптор": desc_names,
        coef_col: coefs,
        "Абсолютный коэффициент": np.abs(coefs),
        "Модель": model_name
    })

    coef_table["coefficient_scale"] = "standardized_features"
    coef_table["coefficient_note"] = (
        "Coefficients belong to the model feature scale used during training. "
        "If descriptors were standardized, they are not raw-unit equation coefficients."
    )

    if model_id == "pls_regression":
        coef_table["pls_n_components"] = getattr(model, "n_components", np.nan)
        try:
            x_weights = np.asarray(model.x_weights_, dtype=float)
            if x_weights.shape[0] == len(desc_names):
                coef_table["pls_x_weight_first_component"] = x_weights[:, 0]
        except Exception:
            pass
        try:
            x_loadings = np.asarray(model.x_loadings_, dtype=float)
            if x_loadings.shape[0] == len(desc_names):
                coef_table["pls_x_loading_first_component"] = x_loadings[:, 0]
        except Exception:
            pass
        try:
            t_scores = np.asarray(model.x_scores_, dtype=float)
            w = np.asarray(model.x_weights_, dtype=float)
            q = np.asarray(model.y_loadings_, dtype=float)
            if w.shape[0] == len(desc_names):
                s = np.sum(t_scores ** 2, axis=0) * np.sum(q ** 2, axis=0)
                total_s = float(np.sum(s))
                if total_s > 1e-12:
                    coef_table["PLS_VIP"] = np.sqrt(
                        len(desc_names)
                        * np.sum((w ** 2) * s.reshape(1, -1), axis=1)
                        / total_s
                    )
        except Exception:
            pass
        coef_table["PLS_note"] = (
            "PLS coefficients depend on scaling and number of latent components; "
            "use VIP/loadings/weights together with validation."
        )

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

    current_work = current_df.copy()
    if "row_position" not in current_work.columns:
        current_work["row_position"] = np.arange(len(current_work), dtype=int)
    if "source_index" not in current_work.columns:
        current_work["source_index"] = list(current_work.index)
    if "source_row" not in current_work.columns:
        current_work["source_row"] = current_work["source_index"]
    if "record_id" not in current_work.columns:
        current_work["record_id"] = [
            f"record_{int(i) + 1:06d}"
            for i in range(len(current_work))
        ]
    if "compound_id" not in current_work.columns:
        current_work["compound_id"] = current_work["record_id"].astype(str)

    target_numeric = qspr_to_numeric(current_work[target_col]).reset_index(drop=True)
    source_meta_cols_base = [
        c for c in [
            "row_position",
            "source_index",
            "source_row",
            "record_id",
            "compound_id",
            "canonical_smiles",
            "inchikey",
            smiles_col,
        ]
        if c in current_work.columns
    ]
    source_meta = current_work.reset_index(drop=True)[source_meta_cols_base].copy()

    parts = []
    row_position_source = None
    meta = None

    # ------------------------------------------------------------
    # Спектральная часть

    if use_spectral:
        if spectral_desc_df is None or spectral_desc_df.empty:
            raise ValueError("Спектральные дескрипторы не переданы или пусты.")

        spectral_work = spectral_desc_df.copy()

        if "row_position" not in spectral_work.columns and "row_index" not in spectral_work.columns:
            raise ValueError(
                "В spectral_desc_df нет row_index. "
                "Нельзя корректно сопоставить спектральные дескрипторы с датасетом."
            )

        if "row_position" not in spectral_work.columns:
            if "source_index" in spectral_work.columns:
                source_to_position = {
                    source_index: position
                    for position, source_index in enumerate(current_work.index.tolist())
                }
                mapped_positions = spectral_work["source_index"].map(source_to_position)
                if mapped_positions.notna().all():
                    spectral_work["row_position"] = mapped_positions.astype(int)
                else:
                    missing_sources = spectral_work.loc[
                        mapped_positions.isna(),
                        "source_index",
                    ].astype(str).head(10).tolist()
                    raise ValueError(
                        "Spectral descriptors contain source_index values that "
                        "cannot be matched to the current dataset index: "
                        + ", ".join(missing_sources)
                    )
            else:
                row_index_values = pd.to_numeric(
                    spectral_work["row_index"],
                    errors="coerce",
                )
                if (
                    row_index_values.notna().all()
                    and (row_index_values >= 0).all()
                    and (row_index_values < len(current_work)).all()
                ):
                    spectral_work["row_position"] = row_index_values.astype(int)
                else:
                    raise ValueError(
                        "Spectral descriptors use legacy row_index, but it is "
                        "not a valid 0-based row position for the current dataset. "
                        "Provide row_position or source_index to avoid row mismatch."
                    )
        spectral_work["row_position"] = spectral_work["row_position"].astype(int)
        spectral_work["row_index"] = spectral_work["row_position"]
        if "record_id" not in spectral_work.columns:
            spectral_work["record_id"] = spectral_work["row_position"].map(
                lambda value: f"record_{int(value) + 1:06d}"
            )
        if "source_row" not in spectral_work.columns:
            spectral_work["source_row"] = spectral_work["row_position"]

        spectral_meta_cols = [
            "record_id",
            "row_position",
            "row_index",
            "source_index",
            "source_row",
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
                spectral_work[["row_position"]].reset_index(drop=True),
                spectral_numeric.reset_index(drop=True)
            ],
            axis=1
        )

        row_position_source = spectral_work[["row_position"]].copy()

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
        mol_work["row_position"] = list(molecular_valid_indices)
        mol_work["row_index"] = mol_work["row_position"]

        mol_desc_cols = [
            c for c in mol_work.columns
            if c not in {"row_position", "row_index"}
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
                mol_work[["row_position"]].reset_index(drop=True),
                mol_numeric.reset_index(drop=True)
            ],
            axis=1
        )

        if use_spectral and restrict_to_spectral_subset:
            if row_position_source is None:
                raise ValueError("Внутренняя ошибка: нет row_position_source.")

            mol_part = row_position_source.merge(
                mol_part,
                on="row_position",
                how="inner"
            )

        else:
            if row_position_source is None:
                row_position_source = mol_work[["row_position"]].copy()

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
                on="row_position",
                how="inner"
            )

    if final.empty:
        raise ValueError("После объединения источников дескрипторов не осталось веществ.")

    final["target_value"] = target_numeric.iloc[
        final["row_position"].astype(int)
    ].values

    final = final[final["target_value"].notna()].copy()

    if final.empty:
        raise ValueError("После удаления строк без целевого свойства не осталось веществ.")

    y_all = final["target_value"].values.astype(float)

    desc_df = final.drop(
        columns=["row_position", "row_index", "target_value"],
        errors="ignore"
    )

    desc_df = qspr_clean_numeric_dataframe(desc_df)

    if desc_df.empty:
        raise ValueError("После подготовки итоговой матрицы не осталось дескрипторов.")

    X_all = desc_df.values.astype(float)
    valid_indices = final["row_position"].astype(int).tolist()
    desc_names = desc_df.columns.tolist()

    if meta is not None:
        match_info = meta[
            meta["row_position"].astype(int).isin(valid_indices)
        ].copy()
    else:
        match_info = pd.DataFrame({
            "record_id": [
                f"record_{idx + 1:06d}"
                for idx in valid_indices
            ],
            "row_position": valid_indices
        })

    if "row_position" not in match_info.columns and "row_index" in match_info.columns:
        row_index_values = pd.to_numeric(match_info["row_index"], errors="coerce")
        if (
            row_index_values.notna().all()
            and (row_index_values >= 0).all()
            and (row_index_values < len(current_work)).all()
        ):
            match_info["row_position"] = row_index_values.astype(int)
        else:
            raise ValueError(
                "match_info contains legacy row_index values that are not valid "
                "0-based row positions for the current dataset."
            )
    match_info["row_index"] = match_info["row_position"].astype(int)
    if "source_row" not in match_info.columns:
        match_info["source_row"] = match_info["row_position"]

    if smiles_col in current_work.columns:
        match_info = match_info.merge(
            source_meta[["row_position", smiles_col]].rename(columns={smiles_col: "SMILES"}),
            on="row_position",
            how="left"
        )

    source_meta_cols = [
        col for col in [
            "record_id",
            "compound_id",
            "source_index",
            "source_row",
            "canonical_smiles",
            "inchikey",
        ]
        if col in current_work.columns and col not in match_info.columns
    ]
    if source_meta_cols:
        match_info = match_info.merge(
            source_meta[["row_position"] + source_meta_cols],
            on="row_position",
            how="left",
        )

    match_info[target_col] = target_numeric.iloc[
        match_info["row_position"].astype(int)
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
        "row_positions": valid_indices,
        "source_indices": [
            current_work.iloc[idx].get("source_index", idx)
            for idx in valid_indices
        ],
        "record_ids": [
            str(current_work.iloc[idx].get("record_id", f"record_{idx + 1:06d}"))
            for idx in valid_indices
        ],
        "desc_names": desc_names,
        "match_info": match_info.reset_index(drop=True),
        "report": report
    }
