# -*- coding: utf-8 -*-
"""
prognostic_model_core.py

Минимальный отдельный модуль для прогностической модели QSPR Forge:
- выбор вручную проверенных веществ;
- обучение финальной прогностической модели;
- сохранение полного пакета модели;
- прогноз для новых веществ по SMILES или готовым дескрипторам;
- Applicability Domain по leverage.

Модуль намеренно содержит Streamlit-интерфейс этого блока, чтобы qspr_app.py
оставался тонким и только вызывал две функции:
- qspr_show_prognostic_training_section(...)
- qspr_show_new_compound_prediction_section(...)
"""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score
from datetime import datetime
import os

from rdkit import Chem

try:
    from .i18n import t
except Exception:
    try:
        from i18n import t  # type: ignore
    except Exception:
        def t(key, **kwargs):
            return key

try:
    from streamlit_ketcher import st_ketcher
    ketcher_available = True
except Exception:
    ketcher_available = False


def safe_len(value):
    """Return len(value) without forcing numpy/pandas objects into bool context."""
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def safe_list(value):
    """Return list(value) without `value or []` for numpy/pandas objects."""
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []

try:
    from .qspr_core import (
        qspr_train_analysis_model,
        qspr_save_results_auto,
        qspr_calculate_molecular_descriptors,
        qspr_calc_xtb_descriptors_dataframe,
    )
except Exception:  # fallback для прямого запуска файла вне пакета modules
    from qspr_core import (  # type: ignore
        qspr_train_analysis_model,
        qspr_save_results_auto,
        qspr_calculate_molecular_descriptors,
        qspr_calc_xtb_descriptors_dataframe,
    )

try:
    from .chemical_scope import (
        classify_smiles_scope,
        find_applicable_models,
    )
except Exception:
    try:
        from chemical_scope import (  # type: ignore
            classify_smiles_scope,
            find_applicable_models,
        )
    except Exception:
        classify_smiles_scope = None
        find_applicable_models = None

try:
    from .prediction_uncertainty import (
        uncertainty_full_analysis,
        uncertainty_make_estimator,
    )
    prog_uncertainty_available = True
except Exception:
    try:
        from prediction_uncertainty import (  # type: ignore
            uncertainty_full_analysis,
            uncertainty_make_estimator,
        )
        prog_uncertainty_available = True
    except Exception:
        uncertainty_full_analysis = None
        uncertainty_make_estimator = None
        prog_uncertainty_available = False


try:
    from .morfeus_descriptor_core import calculate_morfeus_descriptors_for_dataframe
    prog_morfeus_available = True
except Exception:
    try:
        from morfeus_descriptor_core import calculate_morfeus_descriptors_for_dataframe
        prog_morfeus_available = True
    except Exception:
        calculate_morfeus_descriptors_for_dataframe = None
        prog_morfeus_available = False


try:
    from .dscribe_descriptor_core import calculate_dscribe_descriptors_for_dataframe
    prog_dscribe_available = True
except Exception:
    try:
        from dscribe_descriptor_core import calculate_dscribe_descriptors_for_dataframe
        prog_dscribe_available = True
    except Exception:
        calculate_dscribe_descriptors_for_dataframe = None
        prog_dscribe_available = False

try:
    from .spectra_core import (
        spectra_prepare_compounds_from_df,
        spectra_find_in_bank,
        spectra_search_one_compound,
        spectra_get_source_columns_for_type,
        spectra_normalize_spectrum_type,
        spectral_build_descriptors_for_dataset,
    )
except Exception:
    try:
        from spectra_core import (
            spectra_prepare_compounds_from_df,
            spectra_find_in_bank,
            spectra_search_one_compound,
            spectra_get_source_columns_for_type,
            spectra_normalize_spectrum_type,
            spectral_build_descriptors_for_dataset,
        )
    except Exception:
        spectra_prepare_compounds_from_df = None
        spectra_find_in_bank = None
        spectra_search_one_compound = None
        spectra_get_source_columns_for_type = None
        spectra_normalize_spectrum_type = None
        spectral_build_descriptors_for_dataset = None

# ------------------------------------------------------------------
# Applicability Domain


def qspr_prog_calculate_leverage_ad(X_train, X_query=None, desc_names=None):
    """
    Leverage Applicability Domain.

    h = x (X'X)^-1 x'
    h* = 3(p + 1) / n

    X_train и X_query должны быть в том же пространстве признаков, в котором
    работает модель. Обычно это уже scale-transform матрицы.
    """
    X_train = np.asarray(X_train, dtype=float)

    if X_train.ndim != 2:
        raise ValueError("X_train должен быть двумерной матрицей.")

    if X_train.shape[0] < 2:
        raise ValueError("Для AD нужно минимум 2 обучающих вещества.")

    if X_query is None:
        X_query = X_train
    else:
        X_query = np.asarray(X_query, dtype=float)

    if X_query.ndim == 1:
        X_query = X_query.reshape(1, -1)

    if X_query.shape[1] != X_train.shape[1]:
        raise ValueError(
            f"Размерность X_query ({X_query.shape[1]}) не совпадает "
            f"с X_train ({X_train.shape[1]})."
        )

    n, p = X_train.shape
    threshold = 3.0 * (p + 1.0) / float(n)

    # Добавляем intercept, как в классической leverage-формуле.
    X_aug = np.column_stack([np.ones(n), X_train])
    Xq_aug = np.column_stack([np.ones(X_query.shape[0]), X_query])

    xtx_inv = np.linalg.pinv(X_aug.T @ X_aug)
    leverage = np.einsum("ij,jk,ik->i", Xq_aug, xtx_inv, Xq_aug)
    status = np.where(leverage <= threshold, "в AD", "вне AD")

    return {
        "leverage": leverage,
        "threshold": float(threshold),
        "status": status,
        "n": int(n),
        "p": int(p),
        "desc_names": safe_list(desc_names),
    }


def qspr_prog_descriptor_group_key(name):
    """Classifies a descriptor name into broad AD groups."""
    raw = str(name or "").strip()
    upper = raw.upper()

    spectral_ir_prefixes = (
        "IR_",
        "FTIR_",
        "SPEC_IR_",
        "SPEC_FTIR_",
        "WN_",
        "WAVENUMBER_",
    )
    spectral_mass_prefixes = (
        "MS_",
        "MASS_",
        "MZ_",
        "FRAG_",
        "SPEC_MS_",
        "SPEC_MASS_",
    )

    if upper.startswith(spectral_ir_prefixes):
        return "spectral_ir"
    if upper.startswith(spectral_mass_prefixes):
        return "spectral_mass"
    if upper.startswith("SPEC_") or "SPECTR" in upper:
        return "spectral"
    return "molecular"


def qspr_prog_build_descriptor_groups(desc_names):
    """Builds a stable descriptor_groups passport for a model package."""
    groups = {
        "molecular": [],
        "spectral_ir": [],
        "spectral_mass": [],
        "spectral": [],
    }

    for name in safe_list(desc_names):
        key = qspr_prog_descriptor_group_key(name)
        groups.setdefault(key, []).append(str(name))

    spectral_all = []
    for key in ("spectral_ir", "spectral_mass", "spectral"):
        spectral_all.extend(groups.get(key, []))
    groups["spectral_all"] = spectral_all
    groups["combined"] = [str(name) for name in safe_list(desc_names)]
    return groups


def qspr_prog_get_descriptor_groups(desc_names=None):
    groups = st.session_state.get("prog_descriptor_groups")
    if isinstance(groups, dict) and groups.get("combined"):
        return groups
    return qspr_prog_build_descriptor_groups(
        desc_names if desc_names is not None else st.session_state.get("prog_desc_names", [])
    )


def qspr_prog_descriptor_indices(desc_names, group_names):
    positions = {str(name): idx for idx, name in enumerate(safe_list(desc_names))}
    return [
        positions[str(name)]
        for name in safe_list(group_names)
        if str(name) in positions
    ]


def qspr_prog_safe_numeric_matrix(value):
    try:
        matrix = np.asarray(value, dtype=float)
    except Exception:
        return None
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return None
    return matrix


def qspr_prog_nearest_distance_ad(X_train, X_query):
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    if X_query.ndim == 1:
        X_query = X_query.reshape(1, -1)

    train_diff = X_train[:, None, :] - X_train[None, :, :]
    train_dist = np.sqrt(np.sum(train_diff * train_diff, axis=2))
    np.fill_diagonal(train_dist, np.nan)
    train_nn = np.nanmin(train_dist, axis=1)
    threshold = float(np.nanpercentile(train_nn, 95))

    query_diff = X_query[:, None, :] - X_train[None, :, :]
    query_dist = np.sqrt(np.sum(query_diff * query_diff, axis=2))
    nearest = np.min(query_dist, axis=1)
    return nearest, threshold, nearest <= threshold


def qspr_prog_range_ad(X_train_raw, X_query_raw):
    X_train_raw = np.asarray(X_train_raw, dtype=float)
    X_query_raw = np.asarray(X_query_raw, dtype=float)
    if X_query_raw.ndim == 1:
        X_query_raw = X_query_raw.reshape(1, -1)

    mins = np.nanmin(X_train_raw, axis=0)
    maxs = np.nanmax(X_train_raw, axis=0)
    inside_mask = (X_query_raw >= mins) & (X_query_raw <= maxs)
    outside_fraction = 1.0 - np.mean(inside_mask, axis=1)
    return outside_fraction, outside_fraction <= 0.0


def qspr_prog_cosine_similarity_ad(X_train, X_query):
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    if X_query.ndim == 1:
        X_query = X_query.reshape(1, -1)

    eps = 1e-12
    train_norm = np.linalg.norm(X_train, axis=1, keepdims=True)
    query_norm = np.linalg.norm(X_query, axis=1, keepdims=True)
    train_unit = X_train / np.maximum(train_norm, eps)
    query_unit = X_query / np.maximum(query_norm, eps)

    train_sim = train_unit @ train_unit.T
    np.fill_diagonal(train_sim, np.nan)
    train_nn = np.nanmax(train_sim, axis=1)
    threshold = float(np.nanpercentile(train_nn, 5))

    query_sim = query_unit @ train_unit.T
    max_similarity = np.max(query_sim, axis=1)
    return max_similarity, threshold, max_similarity >= threshold


def qspr_prog_calculate_subspace_ad(
    X_train_scaled,
    X_query_scaled,
    X_train_raw,
    X_query_raw,
    indices,
    include_similarity=False,
):
    if not indices:
        n_query = np.asarray(X_query_scaled).shape[0]
        return {
            "status": np.array(["not_calculated"] * n_query),
            "distance": np.full(n_query, np.nan),
            "distance_threshold": np.nan,
            "leverage": np.full(n_query, np.nan),
            "leverage_threshold": np.nan,
            "range_outside_fraction": np.full(n_query, np.nan),
            "similarity": np.full(n_query, np.nan),
            "similarity_threshold": np.nan,
        }

    train_scaled = qspr_prog_safe_numeric_matrix(np.asarray(X_train_scaled)[:, indices])
    query_scaled = np.asarray(X_query_scaled, dtype=float)[:, indices]
    train_raw = qspr_prog_safe_numeric_matrix(np.asarray(X_train_raw)[:, indices])
    query_raw = np.asarray(X_query_raw, dtype=float)[:, indices]

    if train_scaled is None or train_raw is None:
        return {
            "status": np.array(["not_calculated"] * query_scaled.shape[0]),
            "distance": np.full(query_scaled.shape[0], np.nan),
            "distance_threshold": np.nan,
            "leverage": np.full(query_scaled.shape[0], np.nan),
            "leverage_threshold": np.nan,
            "range_outside_fraction": np.full(query_scaled.shape[0], np.nan),
            "similarity": np.full(query_scaled.shape[0], np.nan),
            "similarity_threshold": np.nan,
        }

    leverage_ad = qspr_prog_calculate_leverage_ad(train_scaled, query_scaled)
    distance, distance_threshold, distance_inside = qspr_prog_nearest_distance_ad(
        train_scaled,
        query_scaled,
    )
    range_outside, range_inside = qspr_prog_range_ad(train_raw, query_raw)

    similarity = np.full(query_scaled.shape[0], np.nan)
    similarity_threshold = np.nan
    similarity_inside = np.ones(query_scaled.shape[0], dtype=bool)
    if include_similarity:
        similarity, similarity_threshold, similarity_inside = (
            qspr_prog_cosine_similarity_ad(train_scaled, query_scaled)
        )

    leverage_inside = np.asarray(leverage_ad["leverage"]) <= float(leverage_ad["threshold"])
    inside = leverage_inside & distance_inside & range_inside & similarity_inside
    return {
        "status": np.where(inside, "inside", "outside"),
        "distance": distance,
        "distance_threshold": distance_threshold,
        "leverage": np.asarray(leverage_ad["leverage"], dtype=float),
        "leverage_threshold": float(leverage_ad["threshold"]),
        "range_outside_fraction": range_outside,
        "similarity": similarity,
        "similarity_threshold": similarity_threshold,
    }


def qspr_prog_spectral_ad_interpretation(row):
    mol = str(row.get("molecular_ad_status", "not_calculated"))
    spec = str(row.get("spectral_ad_status", "not_calculated"))
    comb = str(row.get("combined_ad_status", "not_calculated"))

    if mol == "inside" and spec == "inside" and comb == "inside":
        return t("prediction_page.spectral_ad_interp_reliable")
    if mol == "inside" and spec == "outside":
        return t("prediction_page.spectral_ad_interp_spectral_out")
    if mol == "outside" and spec == "inside":
        return t("prediction_page.spectral_ad_interp_molecular_out")
    if comb == "outside":
        return t("prediction_page.spectral_ad_interp_combined_out")
    return t("prediction_page.spectral_ad_interp_not_calculated")


def qspr_prog_add_spectral_ad_columns(pred_df, X_new_raw, X_new_scaled):
    desc_names = list(st.session_state.get("prog_desc_names", []))
    X_train_raw = qspr_prog_safe_numeric_matrix(st.session_state.get("prog_X_train_raw"))
    X_train_scaled = qspr_prog_safe_numeric_matrix(
        qspr_prog_get_train_matrix_for_ad(st.session_state.get("prog_X_train_scaled"))
    )

    X_new_raw = np.asarray(X_new_raw, dtype=float)
    X_new_scaled = np.asarray(X_new_scaled, dtype=float)

    if X_train_raw is None or X_train_scaled is None:
        return pred_df
    if X_train_raw.shape[1] != len(desc_names) or X_train_scaled.shape[1] != len(desc_names):
        return pred_df

    groups = qspr_prog_get_descriptor_groups(desc_names)
    molecular_indices = qspr_prog_descriptor_indices(
        desc_names,
        groups.get("molecular", []),
    )
    spectral_indices = qspr_prog_descriptor_indices(
        desc_names,
        groups.get("spectral_all", [])
        or groups.get("spectral", [])
        or groups.get("spectral_ir", [])
        or groups.get("spectral_mass", []),
    )
    if not spectral_indices:
        return pred_df

    combined_indices = list(range(len(desc_names)))

    molecular_ad = qspr_prog_calculate_subspace_ad(
        X_train_scaled,
        X_new_scaled,
        X_train_raw,
        X_new_raw,
        molecular_indices,
        include_similarity=False,
    )
    spectral_ad = qspr_prog_calculate_subspace_ad(
        X_train_scaled,
        X_new_scaled,
        X_train_raw,
        X_new_raw,
        spectral_indices,
        include_similarity=True,
    )
    combined_ad = qspr_prog_calculate_subspace_ad(
        X_train_scaled,
        X_new_scaled,
        X_train_raw,
        X_new_raw,
        combined_indices,
        include_similarity=False,
    )

    result = pred_df.copy()
    result["molecular_ad_status"] = molecular_ad["status"]
    result["molecular_ad_distance"] = molecular_ad["distance"]
    result["molecular_ad_leverage"] = molecular_ad["leverage"]
    result["spectral_ad_status"] = spectral_ad["status"]
    result["spectral_ad_distance"] = spectral_ad["distance"]
    result["spectral_ad_similarity"] = spectral_ad["similarity"]
    result["combined_ad_status"] = combined_ad["status"]
    result["combined_ad_distance"] = combined_ad["distance"]
    result["combined_ad_leverage"] = combined_ad["leverage"]
    result["descriptor_range_outside_fraction"] = combined_ad["range_outside_fraction"]
    result["spectral_ad_interpretation"] = result.apply(
        qspr_prog_spectral_ad_interpretation,
        axis=1,
    )
    return result


def qspr_prog_get_train_matrix_for_ad(fallback_X_train_scaled=None):
    """
    Возвращает scaled X_train для AD из session_state.
    Совместимо со старыми сохранениями, где X_train_scaled мог отсутствовать.
    """
    X_train_ad = st.session_state.get("prog_X_train_scaled", None)

    if X_train_ad is not None:
        return np.asarray(X_train_ad, dtype=float)

    X_train_raw = st.session_state.get("prog_X_train_raw", None)
    scaler = st.session_state.get("prog_scaler", None)

    if X_train_raw is not None:
        X_train_raw = np.asarray(X_train_raw, dtype=float)
        if scaler is not None:
            return np.asarray(scaler.transform(X_train_raw), dtype=float)
        return X_train_raw

    if fallback_X_train_scaled is not None:
        return np.asarray(fallback_X_train_scaled, dtype=float)

    return None


# ------------------------------------------------------------------
# Training helpers


def qspr_prog_make_training_table(data, smiles_col, valid_indices, y_true, y_pred):
    """
    Таблица ручного выбора веществ для финальной прогностической модели.
    """
    return pd.DataFrame({
        "№": range(1, len(y_true) + 1),
        "Индекс": list(valid_indices),
        "SMILES": data[smiles_col].iloc[list(valid_indices)].astype(str).values,
        "Экспериментальное значение": np.asarray(y_true, dtype=float),
        "Расчётное значение": np.asarray(y_pred, dtype=float),
        "Ошибка": np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
        "Выбрать": True,
    })


def qspr_prog_train_selected_model(
    data,
    smiles_col,
    target_col,
    X_all,
    y_all,
    valid_indices,
    selected_positions,
    desc_names,
    algorithm,
    params=None,
):
    """
    Обучает финальную прогностическую модель на выбранных позициях текущей QSPR-матрицы.
    Возвращает словарь с моделью, таблицей обучения и AD-матрицами.
    """
    selected_positions = list(selected_positions)

    if len(selected_positions) < 2:
        raise ValueError("Выберите хотя бы 2 вещества.")

    X_all = np.asarray(X_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)
    valid_indices = np.asarray(valid_indices, dtype=int)

    X_prog = X_all[selected_positions]
    y_prog = y_all[selected_positions]

    result = qspr_train_analysis_model(
        X_prog,
        y_prog,
        algorithm,
        params=params,
        scale=True,
    )

    y_pred = np.ravel(result["y_pred"])
    train_smiles = data[smiles_col].iloc[valid_indices[selected_positions]].astype(str).values

    train_df = pd.DataFrame({
        "SMILES": train_smiles,
        "Эксперимент": y_prog,
        "Расчёт": y_pred,
        "Ошибка": y_prog - y_pred,
    })

    scaler = result.get("scaler")
    if scaler is not None:
        X_train_scaled = np.asarray(scaler.transform(X_prog), dtype=float)
    else:
        X_train_scaled = np.asarray(X_prog, dtype=float)

    return {
        "model": result["model"],
        "scaler": scaler,
        "y_pred": y_pred,
        "r2_train": float(r2_score(y_prog, y_pred)),
        "X_train_raw": np.asarray(X_prog, dtype=float),
        "X_train_scaled": X_train_scaled,
        "y_train": np.asarray(y_prog, dtype=float),
        "train_smiles": train_smiles,
        "train_df": train_df,
        "desc_names": list(desc_names),
        "descriptor_groups": qspr_prog_build_descriptor_groups(desc_names),
        "target_col": target_col,
        "smiles_col": smiles_col,
        "algorithm": algorithm,
    }


def qspr_prog_store_model_in_session(package):
    """
    Кладёт модельный пакет в st.session_state в прежнем формате приложения.
    """
    st.session_state.prog_model = package["model"]
    st.session_state.prog_scaler = package["scaler"]
    st.session_state.prog_desc_names = list(package["desc_names"])
    st.session_state.prog_model_name = package["algorithm"]
    st.session_state.prog_target_col = package["target_col"]
    st.session_state.prog_smiles_col = package["smiles_col"]
    st.session_state.prog_X_train_raw = package["X_train_raw"]
    st.session_state.prog_X_train_scaled = package["X_train_scaled"]
    st.session_state.prog_y_train = package["y_train"]
    st.session_state.prog_train_smiles = package["train_smiles"]
    st.session_state.prog_descriptor_groups = package.get(
        "descriptor_groups",
        qspr_prog_build_descriptor_groups(package.get("desc_names", [])),
    )
    st.session_state.custom_descriptor_source = package.get(
        "descriptor_source",
        st.session_state.get("custom_descriptor_source", "molecular_calculated"),
    )

def qspr_prog_load_saved_package_to_session(package):
    """
    Загружает сохранённый prognostic package в st.session_state.
    Позволяет использовать прогноз без повторного обучения модели.
    """
    if not isinstance(package, dict):
        raise ValueError("Файл модели должен содержать словарь package.")

    required_keys = ["model", "scaler", "desc_names"]

    for key in required_keys:
        if key not in package:
            raise ValueError(f"В пакете модели нет обязательного поля: {key}")

    st.session_state.prog_model = package["model"]
    st.session_state.prog_scaler = package.get("scaler", None)
    st.session_state.prog_desc_names = list(package.get("desc_names", []))

    st.session_state.prog_target_col = package.get("target_col", "property")
    st.session_state.prog_smiles_col = package.get("smiles_col", "SMILES")
    st.session_state.prog_model_name = package.get("model_name", "loaded_model")

    st.session_state.prog_X_train_raw = package.get(
        "X_train_raw",
        package.get("X_train", None),
    )
    st.session_state.prog_X_train_scaled = package.get("X_train_scaled", None)
    st.session_state.prog_y_train = package.get("y_train", None)
    st.session_state.prog_train_smiles = package.get("train_smiles", None)
    st.session_state.prog_descriptor_groups = package.get(
        "descriptor_groups",
        qspr_prog_build_descriptor_groups(st.session_state.prog_desc_names),
    )

    descriptor_source = package.get("descriptor_source", None)
    if not descriptor_source:
        descriptor_source = qspr_prog_infer_descriptor_source_from_desc_names(
            st.session_state.prog_desc_names
        )

    st.session_state.custom_descriptor_source = descriptor_source
  
def qspr_prog_try_autoload_default_model():
    """
    Автоматически загружает model_prognostic_package.pkl,
    если файл лежит в текущей папке проекта и модель ещё не загружена.
    """
    if "prog_model" in st.session_state:
        return True

    default_paths = [
        os.path.join("prognostic_models", "model_prognostic_latest.pkl"),
        os.path.join(os.getcwd(), "prognostic_models", "model_prognostic_latest.pkl"),

        # Старое имя оставляем только для совместимости со старыми сохранениями.
        "model_prognostic_package.pkl",
        os.path.join(os.getcwd(), "model_prognostic_package.pkl"),
    ]

    for path in default_paths:
        if not os.path.exists(path):
            continue

        try:
            package = joblib.load(path)
            qspr_prog_load_saved_package_to_session(package)
            st.success(f"Автоматически подключена модель: `{path}`")
            return True
        except Exception as e:
            st.warning(f"Файл модели найден, но не удалось загрузить `{path}`: {e}")
            return False

    return False

def qspr_prog_safe_filename(text):
    """
    Делает безопасный фрагмент имени файла.
    """
    text = str(text).strip()

    if not text:
        return "model"

    bad_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']

    for ch in bad_chars:
        text = text.replace(ch, "_")

    text = text.replace(" ", "_")
    text = text.replace(",", "_")
    text = text.replace(";", "_")
    text = text.replace("(", "_")
    text = text.replace(")", "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_") or "model"


def qspr_prog_save_package_to_folder(package):
    """
    Сохраняет прогностическую модель в папку prognostic_models/.
    """
    models_dir = "prognostic_models"
    os.makedirs(models_dir, exist_ok=True)

    target_name = qspr_prog_safe_filename(
        package.get("target_col", "property")
    )

    model_name = qspr_prog_safe_filename(
        package.get("model_name", package.get("algorithm", "model"))
    )

    created_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"model_prognostic_{target_name}_{model_name}_{created_tag}.pkl"
    model_path = os.path.join(models_dir, filename)

    joblib.dump(package, model_path)

    # Последняя сохранённая модель в этой же папке.
    latest_path = os.path.join(models_dir, "model_prognostic_latest.pkl")
    joblib.dump(package, latest_path)

    return model_path, latest_path
  
def qspr_prog_make_save_package(target_col, smiles_col, desc_names, model_name):
    """
    Полный пакет для joblib.dump.
    """
    return {
        "model": st.session_state.prog_model,
        "scaler": st.session_state.prog_scaler,
        "target_col": st.session_state.get("prog_target_col", target_col),
        "smiles_col": st.session_state.get("prog_smiles_col", smiles_col),
        "desc_names": list(st.session_state.get("prog_desc_names", desc_names)),
        "descriptor_groups": qspr_prog_get_descriptor_groups(
            st.session_state.get("prog_desc_names", desc_names)
        ),
        "descriptor_source": st.session_state.get("custom_descriptor_source", ""),
        "model_name": st.session_state.get("prog_model_name", model_name),
        "X_train_raw": st.session_state.get("prog_X_train_raw", None),
        "X_train_scaled": st.session_state.get("prog_X_train_scaled", None),
        "y_train": st.session_state.get("prog_y_train", None),
        "train_smiles": st.session_state.get("prog_train_smiles", None),
        "n_train_compounds": len(st.session_state.get("prog_y_train", []))
        if st.session_state.get("prog_y_train", None) is not None
        else None,
        "ad_method": "leverage",
        "ad_threshold_formula": "h* = 3(p + 1) / n",
        "uncertainty_supported": True,
        "uncertainty_schema_version": "1.0",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ------------------------------------------------------------------
# Prediction helpers


def qspr_prog_prepare_ready_descriptor_matrix(new_df, required_desc):
    """
    Делает X для режима прогноза по готовым дескрипторам.
    Отсутствующие дескрипторы добавляет нулями, NaN заполняет медианой.
    """
    work = new_df.copy()

    for col in required_desc:
        if col not in work.columns:
            work[col] = 0.0

        work[col] = pd.to_numeric(
            work[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    X_df = work[required_desc].replace([np.inf, -np.inf], np.nan)

    for col in X_df.columns:
        median_value = X_df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        X_df[col] = X_df[col].fillna(median_value)

    return X_df.values.astype(float)


def qspr_prog_uncertainty_model_entries(X_train, y_train):
    """
    Формирует сопоставимый набор моделей для consensus.

    Дополнительные алгоритмы заново обучаются на той же финальной обучающей
    выборке, что и прогностическая модель. Это исключает искусственный разброс,
    вызванный разными наборами веществ.
    """
    primary_name = str(
        st.session_state.get("prog_model_name", "Прогностическая модель")
    )
    entries = {
        primary_name: {
            "model": st.session_state.prog_model,
            "scaler": st.session_state.get("prog_scaler"),
        }
    }

    if not st.session_state.get("prediction_uncertainty_use_consensus", True):
        return entries

    trained = st.session_state.get("trained_models", {}) or {}
    selected = st.session_state.get(
        "prediction_uncertainty_consensus_models",
        list(trained.keys())[:4],
    )

    for name in list(selected)[:6]:
        if name == primary_name or name not in trained:
            continue
        entry = trained[name]
        model = entry.get("model")
        if model is None:
            continue
        try:
            estimator = uncertainty_make_estimator(
                model,
                entry.get("scaler"),
            )
            estimator.fit(X_train, y_train)
            entries[str(name)] = {
                "model": estimator,
                "scaler": None,
            }
        except Exception:
            continue

    return entries


def qspr_prog_query_smiles(new_df):
    """Возвращает SMILES новых объектов, если колонка доступна."""
    preferred = st.session_state.get("prog_smiles_col", "SMILES")
    candidates = [
        preferred,
        "SMILES",
        "smiles",
        "canonical_smiles",
        "Canonical_SMILES",
        "input_smiles",
    ]
    for column in candidates:
        if column in new_df.columns:
            return new_df[column].astype(str).tolist()
    return None


def qspr_prog_uncertainty_neighbour_table(result, query_smiles=None):
    """Разворачивает ближайших обучающих аналогов в отдельную таблицу."""
    train_smiles_raw = st.session_state.get("prog_train_smiles")
    train_smiles = (
        list(train_smiles_raw)
        if train_smiles_raw is not None
        else []
    )
    y_train = np.ravel(
        np.asarray(st.session_state.get("prog_y_train", []), dtype=float)
    )
    rows = []
    for query_index, local in enumerate(result.get("local", [])):
        indices = local.get("neighbour_indices", [])
        distances = local.get("neighbour_distances", [])
        for rank, train_index in enumerate(indices, start=1):
            row = {
                "Объект": query_index + 1,
                "Ранг аналога": rank,
                "Индекс в обучении": int(train_index),
                "Расстояние": float(distances[rank - 1]),
            }
            if query_smiles and query_index < len(query_smiles):
                row["SMILES нового вещества"] = query_smiles[query_index]
            if int(train_index) < len(train_smiles):
                row["SMILES аналога"] = train_smiles[int(train_index)]
            if int(train_index) < len(y_train):
                row["Эксперимент аналога"] = float(y_train[int(train_index)])
            rows.append(row)
    return pd.DataFrame(rows)


def qspr_prog_add_complete_uncertainty(X_new, pred_df):
    """Добавляет полную оценку неопределённости к таблице прогноза."""
    if (
        not prog_uncertainty_available
        or not st.session_state.get("prediction_uncertainty_enabled", True)
    ):
        return pred_df

    X_train = st.session_state.get("prog_X_train_raw")
    y_train = st.session_state.get("prog_y_train")
    if X_train is None or y_train is None:
        st.session_state.prediction_uncertainty_result = {
            "error": (
                "В пакете модели нет X_train/y_train. Полная оценка "
                "неопределённости доступна после переобучения или сохранения "
                "модели новым форматом."
            )
        }
        return pred_df

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.ravel(np.asarray(y_train, dtype=float))
    X_new = np.asarray(X_new, dtype=float)
    if len(y_train) < 4 or X_train.shape[0] != len(y_train):
        st.session_state.prediction_uncertainty_result = {
            "error": "Для полной оценки неопределённости нужно минимум 4 обучающих объекта."
        }
        return pred_df

    query_smiles = qspr_prog_query_smiles(pred_df)
    train_smiles = st.session_state.get("prog_train_smiles")
    leverage = pred_df.get("Leverage h")
    leverage_threshold = (
        float(pred_df["Порог h*"].iloc[0])
        if "Порог h*" in pred_df.columns and len(pred_df)
        else None
    )

    confidence = float(
        st.session_state.get("prediction_uncertainty_confidence", 0.90)
    )
    result = uncertainty_full_analysis(
        primary_model=st.session_state.prog_model,
        primary_scaler=st.session_state.get("prog_scaler"),
        X_train=X_train,
        y_train=y_train,
        X_query=X_new,
        model_entries=qspr_prog_uncertainty_model_entries(X_train, y_train),
        train_smiles=train_smiles,
        query_smiles=query_smiles,
        leverage=np.asarray(leverage, dtype=float) if leverage is not None else None,
        leverage_threshold=leverage_threshold,
        alpha=1.0 - confidence,
        cv=int(st.session_state.get("prediction_uncertainty_cv", 5)),
        n_bootstrap=int(
            st.session_state.get("prediction_uncertainty_bootstrap", 100)
        ),
        k_neighbors=int(
            st.session_state.get("prediction_uncertainty_neighbors", 5)
        ),
        random_state=int(
            st.session_state.get("prediction_uncertainty_random_state", 42)
        ),
    )

    uncertainty_table = result["table"].copy()
    status_map = {
        "high": "высокая",
        "medium": "средняя",
        "low": "низкая",
    }
    reason_map = {
        "wide_conformal_interval": "широкий калиброванный интервал",
        "moderate_conformal_interval": "умеренно широкий интервал",
        "outside_distance_ad": "вне локальной области по расстоянию",
        "outside_descriptor_range": "выход за диапазоны дескрипторов",
        "low_structural_similarity": "низкое структурное сходство",
        "outside_leverage_ad": "вне leverage AD",
        "high_consensus_uncertainty": "сильное расхождение моделей",
        "high_bootstrap_uncertainty": "высокий query-bootstrap разброс",
        "high_local_uncertainty": "высокая ошибка ближайших аналогов",
        "high_gpr_uncertainty": "высокая неопределённость GPR",
        "no_major_warning": "существенных предупреждений нет",
    }

    def translate_reasons(value):
        return "; ".join(
            reason_map.get(item.strip(), item.strip())
            for item in str(value).split(";")
        )

    column_map = {
        "conformal_lower": "PI нижняя граница",
        "conformal_upper": "PI верхняя граница",
        "conformal_confidence": "Доверительный уровень PI",
        "consensus_mean": "Consensus прогноз",
        "consensus_std": "Consensus std",
        "bootstrap_mean": "Query-bootstrap среднее",
        "bootstrap_std": "Query-bootstrap std",
        "bootstrap_p05": "Query-bootstrap P05",
        "bootstrap_p95": "Query-bootstrap P95",
        "local_mae": "Локальная MAE аналогов",
        "local_rmse": "Локальная RMSE аналогов",
        "local_bias": "Локальный bias аналогов",
        "nearest_distance": "Расстояние до ближайшего аналога",
        "distance_ad_threshold": "Порог distance AD",
        "distance_ad_inside": "Внутри distance AD",
        "descriptor_range_outside_fraction": "Доля дескрипторов вне диапазона",
        "descriptor_range_inside": "Внутри диапазонов дескрипторов",
        "gpr_std": "GPR std",
        "maximum_tanimoto_similarity": "Максимальное сходство Tanimoto",
        "similarity_ad_inside": "Внутри similarity AD",
        "reliability_score": "Надёжность, балл",
        "reliability_status": "Надёжность",
        "reliability_reasons": "Причины статуса",
    }
    uncertainty_table["reliability_status"] = (
        uncertainty_table["reliability_status"].map(status_map)
    )
    uncertainty_table["reliability_reasons"] = (
        uncertainty_table["reliability_reasons"].map(translate_reasons)
    )
    uncertainty_table = uncertainty_table.drop(
        columns=["prediction", "leverage"],
        errors="ignore",
    ).rename(columns=column_map)

    merged = pd.concat(
        [pred_df.reset_index(drop=True), uncertainty_table.reset_index(drop=True)],
        axis=1,
    )
    neighbours = qspr_prog_uncertainty_neighbour_table(result, query_smiles)
    st.session_state.prediction_uncertainty_result = {
        "table": merged.copy(),
        "neighbours": neighbours,
        "diagnostics": {
            "OOF RMSE": result["reliability"]["oof_rmse"],
            "Радиус conformal PI": result["conformal"]["radius"],
            "Успешных bootstrap-моделей": result["bootstrap"]["successful"],
            "Моделей в consensus": len(result["consensus"]["names"]),
        },
        "consensus_models": result["consensus"]["names"],
        "consensus_weights": result.get("consensus_weights", {}),
        "consensus_cv_rmse": result.get("consensus_cv_rmse", {}),
    }
    return merged


def qspr_prog_predict_from_matrix(X_new, new_df, target_col, fallback_X_train_scaled=None):
    """
    Прогноз по уже подготовленной матрице дескрипторов + AD.
    """
    model = st.session_state.prog_model
    scaler = st.session_state.prog_scaler
    required_desc = list(st.session_state.get("prog_desc_names", []))
    st.session_state.prediction_uncertainty_result = {}

    X_new = np.asarray(X_new, dtype=float)

    if scaler is not None:
        X_model = scaler.transform(X_new)
    else:
        X_model = X_new

    y_pred = np.ravel(model.predict(X_model))

    pred_df = new_df.copy()
    pred_col_name = f"Прогноз_{st.session_state.get('prog_target_col', target_col)}"
    pred_df[pred_col_name] = y_pred

    X_train_ad = qspr_prog_get_train_matrix_for_ad(fallback_X_train_scaled)
    ad_done = False

    if X_train_ad is not None:
        ad = qspr_prog_calculate_leverage_ad(
            X_train=X_train_ad,
            X_query=X_model,
            desc_names=required_desc,
        )
        pred_df["Leverage h"] = ad["leverage"]
        pred_df["Порог h*"] = ad["threshold"]
        pred_df["AD-статус"] = ad["status"]
        pred_df["Надёжность по AD"] = pred_df["AD-статус"].map({
            "в AD": "надёжнее: внутри области модели",
            "вне AD": "осторожно: экстраполяция, прогноз менее надёжен",
        }).fillna("не оценено")
        pred_df = qspr_prog_add_spectral_ad_columns(
            pred_df=pred_df,
            X_new_raw=X_new,
            X_new_scaled=X_model,
        )
        ad_done = True

    try:
        pred_df = qspr_prog_add_complete_uncertainty(X_new, pred_df)
    except Exception as error:
        st.session_state.prediction_uncertainty_result = {
            "error": f"Не удалось рассчитать полную неопределённость: {error}"
        }

    return pred_df, ad_done


def qspr_prog_predict_from_smiles(
    new_df,
    smiles_col,
    required_desc,
    descriptor_mode,
    desc_lists,
    target_col,
    fallback_X_train_scaled=None,
):
    """
    Рассчитывает молекулярные дескрипторы для новых SMILES и делает прогноз.

    Важно:
    qspr_calculate_molecular_descriptors удаляет константные колонки.
    Для одной молекулы все дескрипторы выглядят константными, поэтому
    добавляем временные dummy-молекулы только для расчёта/очистки.
    В прогноз они не попадают.
    """
    work_df = new_df.copy().reset_index(drop=True)
    original_n = len(work_df)

    if original_n == 0:
        raise ValueError("Нет новых веществ для прогноза.")

    # Для одной молекулы добавляем временные валидные структуры,
    # чтобы qspr_clean_numeric_dataframe не удалил все дескрипторы как константные.
    if original_n == 1:
        dummy_df = pd.DataFrame([
            {smiles_col: "CC"},
            {smiles_col: "c1ccccc1"},
        ])

        for col in work_df.columns:
            if col not in dummy_df.columns:
                dummy_df[col] = ""

        for col in dummy_df.columns:
            if col not in work_df.columns:
                work_df[col] = ""

        work_for_desc = pd.concat(
            [work_df, dummy_df[work_df.columns]],
            ignore_index=True,
        )
    else:
        work_for_desc = work_df

    bundle = qspr_calculate_molecular_descriptors(
        data=work_for_desc,
        smiles_col=smiles_col,
        target_col=None,
        mode=descriptor_mode,
        desc_lists=desc_lists,
        preserve_constant_columns=True,
    )

    desc_df_all = bundle["df_desc"].copy()
    valid_indices_all = list(bundle["valid_indices"])

    # Оставляем только реальные вещества, dummy-строки отбрасываем.
    keep_positions = [
        pos for pos, original_idx in enumerate(valid_indices_all)
        if int(original_idx) < original_n
    ]

    if not keep_positions:
        raise ValueError("Не удалось рассчитать дескрипторы для новых веществ.")

    desc_df = desc_df_all.iloc[keep_positions].copy()
    valid_indices = [
        int(valid_indices_all[pos])
        for pos in keep_positions
    ]

    for col in required_desc:
        if col not in desc_df.columns:
            desc_df[col] = 0.0

    desc_df = desc_df[required_desc]
    X_new = desc_df.values.astype(float)

    base_df = work_df.iloc[valid_indices].copy()

    return qspr_prog_predict_from_matrix(
        X_new=X_new,
        new_df=base_df,
        target_col=target_col,
        fallback_X_train_scaled=fallback_X_train_scaled,
    )

def qspr_prog_detect_spectrum_type_from_required_desc(required_desc):
    """
    Определяет, какой тип спектра нужен модели.
    Сейчас поддерживаем IR и Mass.
    """
    names = [str(c).upper() for c in required_desc]

    if any("MASS" in c or "MS_" in c or "SPEC_MASS" in c for c in names):
        return "Mass"

    return "IR"


def qspr_prog_default_spectral_descriptor_settings(required_desc):
    """
    Настройки расчёта спектральных дескрипторов для прогноза.

    Важно:
    они должны соответствовать настройкам, с которыми обучалась модель.
    Пока берём стандартный профиль, как в основном спектральном блоке.
    """
    spectrum_type = qspr_prog_detect_spectrum_type_from_required_desc(required_desc)

    return {
        "spectrum_type": spectrum_type,
        "wn_min": 550,
        "wn_max": 3798,
        "step": 4,
        "normalization": "min-max",
        "invert_signal": False,
        "use_grid": True,
        "use_binary_fp": True,
        "use_binned_numeric": True,
        "binary_window": 20,
        "binary_threshold": 0.10,
        "numeric_window": 100,
        "use_svd": False,
        "svd_components": 10,
        "spectrum_phase_mode": "prefer_gas",
        "allowed_phases": None,
        "allowed_sources": None,
        "allowed_intensity_types": None,
        "prefer_quantitative": False,
        "experimental_only": True,
    }


def qspr_prog_ensure_spectrum_for_compound(
    compound,
    spectrum_type="IR",
    allow_online_search=True,
    delay_seconds=0.5,
):
    """
    Проверяет наличие спектра:
    1. локальный spectra_bank;
    2. если нет — онлайн-поиск по доступным источникам;
    3. если найден онлайн — spectrum сохраняется в spectra_bank.
    """
    if spectra_find_in_bank is None or spectra_search_one_compound is None:
        return {
            "ok": False,
            "status": "spectra_module_unavailable",
            "message": "Спектральный модуль недоступен.",
        }

    inchikey = str(compound.get("inchikey", "")).strip()
    canonical_smiles = str(compound.get("canonical_smiles", "")).strip()

    local_record = spectra_find_in_bank(
        inchikey=inchikey,
        canonical_smiles=canonical_smiles,
        spectrum_type=spectrum_type,
    )

    if local_record is not None:
        return {
            "ok": True,
            "status": "already_in_local_spectra_bank",
            "message": "Спектр найден в локальном spectra_bank.",
            "record": local_record,
        }

    if not allow_online_search:
        return {
            "ok": False,
            "status": "not_found_in_local_spectra_bank",
            "message": "Спектр не найден в локальном spectra_bank.",
        }

    if spectra_get_source_columns_for_type is not None:
        selected_sources = spectra_get_source_columns_for_type(spectrum_type)
    else:
        if str(spectrum_type).strip().lower() == "mass":
            selected_sources = ["mona_mass"]
        else:
            selected_sources = ["nist_webbook"]

    result = spectra_search_one_compound(
        compound=compound,
        spectrum_type=spectrum_type,
        selected_sources=selected_sources,
        delay_seconds=delay_seconds,
    )

    status = str(
        result.get("spectrum_status", result.get("status", ""))
    ).strip()

    if status in ["found_downloaded", "already_in_bank"]:
        return {
            "ok": True,
            "status": status,
            "message": result.get("message", "Спектр найден."),
            "record": result.get("record", {}),
        }

    return {
        "ok": False,
        "status": status or "not_found_in_all_sources",
        "message": (
            "Модель требует спектральные признаки. "
            "Для этой структуры спектр не найден. "
            "Загрузите спектр или используйте модель без спектральных дескрипторов."
        ),
        "raw_result": result,
    }


def qspr_prog_calculate_spectral_descriptors_for_prediction(
    base_df,
    smiles_col,
    required_desc,
    allow_online_search=True,
):
    """
    Для новых SMILES:
    1. ищет спектр в локальном spectra_bank;
    2. если не найден — ищет онлайн;
    3. если найден — строит спектральные дескрипторы;
    4. приводит имена к виду, совместимому с QSPR-матрицей: SPEC_*.
    """
    if spectral_build_descriptors_for_dataset is None:
        raise ValueError("Спектральный модуль недоступен.")

    settings = qspr_prog_default_spectral_descriptor_settings(required_desc)
    spectrum_type = settings["spectrum_type"]

    compounds = spectra_prepare_compounds_from_df(
        base_df,
        smiles_col=smiles_col,
    )

    if compounds is None or compounds.empty:
        raise ValueError(
            "Не удалось подготовить структуру для поиска спектра."
        )

    failed_messages = []

    for _, compound_row in compounds.iterrows():
        compound = compound_row.to_dict()

        spectrum_check = qspr_prog_ensure_spectrum_for_compound(
            compound=compound,
            spectrum_type=spectrum_type,
            allow_online_search=allow_online_search,
            delay_seconds=0.5,
        )

        if not spectrum_check.get("ok", False):
            failed_messages.append(spectrum_check.get("message", ""))

    if failed_messages:
        raise ValueError(
            "Модель требует спектральные признаки. "
            "Для этой структуры спектр не найден. "
            "Загрузите спектр или используйте модель без спектральных дескрипторов."
        )

    descriptors_df, spectral_report = spectral_build_descriptors_for_dataset(
        input_df=base_df,
        smiles_col=smiles_col,
        spectrum_type=spectrum_type,
        wn_min=settings["wn_min"],
        wn_max=settings["wn_max"],
        step=settings["step"],
        normalization=settings["normalization"],
        invert_signal=settings["invert_signal"],
        use_grid=settings["use_grid"],
        use_binary_fp=settings["use_binary_fp"],
        use_binned_numeric=settings["use_binned_numeric"],
        binary_window=settings["binary_window"],
        binary_threshold=settings["binary_threshold"],
        numeric_window=settings["numeric_window"],
        use_svd=settings["use_svd"],
        svd_components=settings["svd_components"],
        spectrum_phase_mode=settings["spectrum_phase_mode"],
        allowed_phases=settings["allowed_phases"],
        allowed_sources=settings["allowed_sources"],
        allowed_intensity_types=settings["allowed_intensity_types"],
        prefer_quantitative=settings["prefer_quantitative"],
        experimental_only=settings["experimental_only"],
    )

    if descriptors_df is None or descriptors_df.empty:
        raise ValueError(
            "Модель требует спектральные признаки. "
            "Для этой структуры спектр не найден. "
            "Загрузите спектр или используйте модель без спектральных дескрипторов."
        )

    spectral_meta_cols = {
        "row_index",
        "compound_id",
        "name",
        "input_smiles",
        "canonical_smiles",
        "inchikey",
        "spectrum_type",
        "spectrum_id",
        "spectrum_source",
        "spectrum_source_database",
        "spectrum_phase",
        "spectrum_phase_norm",
        "spectrum_intensity_type",
        "spectrum_is_experimental",
        "spectrum_is_quantitative",
        "spectrum_selection_reason",
        "processed_file",
    }

    spectral_cols = [
        c for c in descriptors_df.columns
        if c not in spectral_meta_cols
    ]

    spectral_numeric = descriptors_df[spectral_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    spectral_numeric = spectral_numeric.replace([np.inf, -np.inf], np.nan)
    spectral_numeric = spectral_numeric.fillna(0.0)

    spectral_numeric.columns = [
        f"SPEC_{c}" if not str(c).startswith("SPEC_") else str(c)
        for c in spectral_numeric.columns
    ]

    return spectral_numeric.reset_index(drop=True), spectral_report

def qspr_prog_get_model_descriptor_source():
    """
    Источник дескрипторов текущей прогностической модели.
    """
    return st.session_state.get(
        "custom_descriptor_source",
        st.session_state.get("molecular_descriptor_source", "molecular_calculated"),
    )


def qspr_prog_descriptor_source_flags(descriptor_source):
    """
    По строке descriptor_source определяет, какие дескрипторы надо считать.
    """
    source = str(descriptor_source or "").lower().strip()

    return {
        "source": source,
        "has_spectral": "spectral" in source,
        "need_molecular": (
            source in ["", "molecular_calculated", "molecular_only"]
            or source.startswith("molecular")
            or "molecular" in source
        ),
        "need_xtb": "xtb" in source,
        "need_morfeus": "morfeus" in source,
        "need_dscribe": "dscribe" in source,
    }


def qspr_prog_descriptor_category_counts(desc_names):
    counts = {
        "spectral": 0,
        "xtb": 0,
        "morfeus": 0,
        "dscribe": 0,
        "molecular_or_other": 0,
    }

    for name in safe_list(desc_names):
        lower = str(name).lower()
        if lower.startswith("spec_") or lower.startswith("spectral_"):
            counts["spectral"] += 1
        elif lower.startswith("xtb_"):
            counts["xtb"] += 1
        elif lower.startswith("morfeus_"):
            counts["morfeus"] += 1
        elif lower.startswith("dscribe_"):
            counts["dscribe"] += 1
        else:
            counts["molecular_or_other"] += 1

    return counts


def qspr_prog_infer_descriptor_source_from_desc_names(desc_names):
    counts = qspr_prog_descriptor_category_counts(desc_names)
    parts = []

    if counts["spectral"] > 0:
        parts.append("spectral")
    if counts["xtb"] > 0:
        parts.append("xtb")
    if counts["morfeus"] > 0:
        parts.append("morfeus")
    if counts["dscribe"] > 0:
        parts.append("dscribe")
    if counts["molecular_or_other"] > 0:
        parts.insert(0, "molecular")

    if not parts:
        return "molecular_calculated"
    if parts == ["spectral"]:
        return "spectral_only"
    return "_plus_".join(parts)


def qspr_prog_descriptor_parts_from_flags(flags):
    parts = []
    if flags.get("need_molecular"):
        parts.append(t("prediction_page.part_molecular"))
    if flags.get("need_xtb"):
        parts.append("xTB")
    if flags.get("need_morfeus"):
        parts.append("morfeus")
    if flags.get("need_dscribe"):
        parts.append("DScribe")
    if flags.get("has_spectral"):
        parts.append(t("prediction_page.part_spectral"))
    return parts


def qspr_show_loaded_model_descriptor_summary():
    if "prog_model" not in st.session_state:
        return

    descriptor_source = qspr_prog_get_model_descriptor_source()
    desc_names = list(st.session_state.get("prog_desc_names", []))
    flags = qspr_prog_descriptor_source_flags(descriptor_source)
    counts = qspr_prog_descriptor_category_counts(desc_names)
    parts = qspr_prog_descriptor_parts_from_flags(flags)

    st.markdown(f"#### {t('prediction_page.model_descriptor_summary')}")

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric(t("prediction_page.model_descriptor_source"), descriptor_source or "—")
    col_2.metric(t("prediction_page.model_descriptor_count"), len(desc_names))
    col_3.metric(
        t("prediction_page.model_descriptor_sets"),
        ", ".join(parts) or t("prediction_page.part_unknown"),
    )

    category_rows = [
        {
            t("prediction_page.descriptor_type_col"): t("prediction_page.part_spectral"),
            t("prediction_page.descriptor_count_col"): counts["spectral"],
        },
        {
            t("prediction_page.descriptor_type_col"): "xTB",
            t("prediction_page.descriptor_count_col"): counts["xtb"],
        },
        {
            t("prediction_page.descriptor_type_col"): "morfeus",
            t("prediction_page.descriptor_count_col"): counts["morfeus"],
        },
        {
            t("prediction_page.descriptor_type_col"): "DScribe",
            t("prediction_page.descriptor_count_col"): counts["dscribe"],
        },
        {
            t("prediction_page.descriptor_type_col"): t("prediction_page.part_molecular_or_other"),
            t("prediction_page.descriptor_count_col"): counts["molecular_or_other"],
        },
    ]
    st.dataframe(pd.DataFrame(category_rows), width="stretch", hide_index=True)

    if flags["has_spectral"]:
        st.warning(t("prediction_page.model_has_spectral_warning"))
    elif flags["need_xtb"] or flags["need_morfeus"] or flags["need_dscribe"]:
        st.info(t("prediction_page.model_needs_3d_warning"))
    else:
        st.success(t("prediction_page.model_smiles_ready"))

    with st.expander(t("prediction_page.show_descriptor_names"), expanded=False):
        st.write(desc_names[:500])


def qspr_prog_prepare_descriptor_frame_for_model(desc_all, required_desc):
    """
    Приводит рассчитанные дескрипторы к точному набору признаков модели.
    """
    desc_all = desc_all.copy()
    desc_all = desc_all.loc[:, ~desc_all.columns.duplicated()].copy()

    missing_desc = []

    for col in required_desc:
        if col not in desc_all.columns:
            desc_all[col] = 0.0
            missing_desc.append(col)

    X_df = desc_all[required_desc].copy()

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(
            X_df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    for col in X_df.columns:
        if X_df[col].isna().any():
            median_value = X_df[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            X_df[col] = X_df[col].fillna(median_value)

    return X_df, missing_desc


def qspr_prog_calculate_descriptors_for_smiles_by_model_source(
    new_df,
    smiles_col,
    required_desc,
    descriptor_source,
    descriptor_mode,
    desc_lists,
):
    """
    Рассчитывает для новых SMILES тот набор дескрипторов,
    который соответствует источнику текущей прогностической модели.
    """
    if new_df is None or new_df.empty:
        raise ValueError("Нет новых веществ для прогноза.")

    if smiles_col not in new_df.columns:
        raise ValueError(f"Колонка SMILES не найдена: {smiles_col}")

    flags = qspr_prog_descriptor_source_flags(descriptor_source)

    if flags["has_spectral"]:
        raise ValueError(
            "Текущая модель содержит спектральные дескрипторы. "
            "По одной структуре/SMILES нельзя восстановить спектральные признаки. "
            "Используйте прогноз по готовым дескрипторам из файла."
        )

    work = new_df.copy().reset_index(drop=True)

    canonical_smiles = []
    valid_rows = []

    for idx, row in work.iterrows():
        smiles = str(row.get(smiles_col, "")).strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol is None:
            continue

        canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        valid_rows.append(idx)

    if not valid_rows:
        raise ValueError("Не удалось распознать ни одного SMILES для прогноза.")

    base_df = work.iloc[valid_rows].copy().reset_index(drop=True)
    base_df[smiles_col] = canonical_smiles
    base_df["canonical_smiles"] = canonical_smiles

    desc_parts = []

    # ------------------------------------------------------------
    # Molecular RDKit/Mordred/PaDEL

    if flags["need_molecular"]:
        molecular_desc_df = qspr_prog_calculate_molecular_descriptors_for_prediction(
            new_df=base_df,
            smiles_col=smiles_col,
            descriptor_mode=descriptor_mode,
            desc_lists=desc_lists,
        )

        desc_parts.append(molecular_desc_df.reset_index(drop=True))

    # ------------------------------------------------------------
    # xTB default

    if flags["need_xtb"]:
        xtb_work = base_df.copy()
        xtb_work["_tmp_target"] = 0.0

        xtb_df, _ = qspr_calc_xtb_descriptors_dataframe(
            data=xtb_work,
            smiles_col=smiles_col,
            target_col="_tmp_target",
            method="GFN2-xTB",
            charge=0,
            uhf=0,
            accuracy=1.0,
            electronic_temperature=300.0,
            max_iterations=250,
            random_seed=1,
            optimize_with_rdkit=True,
            max_molecules=None,
        )

        if xtb_df is None or xtb_df.empty:
            raise ValueError("xTB не вернул таблицу дескрипторов.")

        if "xtb_status" in xtb_df.columns:
            ok_mask = xtb_df["xtb_status"].astype(str).str.lower().str.strip() == "ok"

            if not ok_mask.any():
                err = ""
                if "xtb_error" in xtb_df.columns:
                    err = str(xtb_df["xtb_error"].iloc[0])
                raise ValueError(f"xTB не смог рассчитать дескрипторы: {err}")

            xtb_df = xtb_df.loc[ok_mask].copy()

        xtb_service_cols = {
            "xtb_status",
            "xtb_error",
            "xtb_message",
        }

        xtb_cols = [
            c for c in xtb_df.columns
            if str(c).startswith("xtb_") and c not in xtb_service_cols
        ]

        desc_parts.append(xtb_df[xtb_cols].reset_index(drop=True))

    # ------------------------------------------------------------
    # morfeus default

    if flags["need_morfeus"]:
        if calculate_morfeus_descriptors_for_dataframe is None:
            raise ValueError("Модуль morfeus недоступен в текущем окружении.")

        morfeus_df = calculate_morfeus_descriptors_for_dataframe(
            df=base_df,
            smiles_col=smiles_col,
            id_col=None,
            random_seed=42,
            optimize=True,
            calc_sasa=True,
            calc_dispersion=True,
            calc_xtb=True,
            max_molecules=None,
            progress_callback=None,
        )

        if morfeus_df is None or morfeus_df.empty:
            raise ValueError("morfeus не вернул таблицу дескрипторов.")

        if "morfeus_status" in morfeus_df.columns:
            ok_mask = morfeus_df["morfeus_status"].astype(str).str.lower().str.strip() == "ok"

            if not ok_mask.any():
                err = ""
                if "morfeus_error" in morfeus_df.columns:
                    err = str(morfeus_df["morfeus_error"].iloc[0])
                raise ValueError(f"morfeus не смог рассчитать дескрипторы: {err}")

            morfeus_df = morfeus_df.loc[ok_mask].copy()

        morfeus_service_cols = {
            "morfeus_status",
            "morfeus_error",
            "morfeus_3d_status",
            "morfeus_3d_message",
            "morfeus_sasa_status",
            "morfeus_dispersion_status",
            "morfeus_xtb_status",
        }

        morfeus_cols = [
            c for c in morfeus_df.columns
            if str(c).startswith("morfeus_") and c not in morfeus_service_cols
        ]

        desc_parts.append(morfeus_df[morfeus_cols].reset_index(drop=True))

    # ------------------------------------------------------------
    # DScribe default

    if flags["need_dscribe"]:
        if calculate_dscribe_descriptors_for_dataframe is None:
            raise ValueError("Модуль DScribe недоступен в текущем окружении.")

        dscribe_df = calculate_dscribe_descriptors_for_dataframe(
            df=base_df,
            smiles_col=smiles_col,
            id_col=None,
            random_seed=42,
            optimize=True,
            max_atoms=60,
            calc_coulomb=True,
            max_molecules=None,
            progress_callback=None,
        )

        if dscribe_df is None or dscribe_df.empty:
            raise ValueError("DScribe не вернул таблицу дескрипторов.")

        if "dscribe_status" in dscribe_df.columns:
            ok_mask = dscribe_df["dscribe_status"].astype(str).str.lower().str.strip() == "ok"

            if not ok_mask.any():
                err = ""
                if "dscribe_error" in dscribe_df.columns:
                    err = str(dscribe_df["dscribe_error"].iloc[0])
                raise ValueError(f"DScribe не смог рассчитать дескрипторы: {err}")

            dscribe_df = dscribe_df.loc[ok_mask].copy()

        dscribe_service_cols = {
            "dscribe_status",
            "dscribe_error",
            "dscribe_3d_status",
            "dscribe_3d_message",
            "dscribe_coulomb_status",
        }

        dscribe_cols = [
            c for c in dscribe_df.columns
            if str(c).startswith("dscribe_") and c not in dscribe_service_cols
        ]

        desc_parts.append(dscribe_df[dscribe_cols].reset_index(drop=True))

    # ------------------------------------------------------------
    # Spectral descriptors

    if flags["has_spectral"]:
        spectral_desc_df, spectral_report = qspr_prog_calculate_spectral_descriptors_for_prediction(
            base_df=base_df,
            smiles_col=smiles_col,
            required_desc=required_desc,
            allow_online_search=True,
        )

        desc_parts.append(
            spectral_desc_df.reset_index(drop=True)
        )
        
    if not desc_parts:
        raise ValueError(
            f"Для источника дескрипторов модели `{descriptor_source}` "
            "не найден автоматический расчёт по SMILES."
        )

    min_len = min(len(part) for part in desc_parts)
    desc_parts = [part.iloc[:min_len].reset_index(drop=True) for part in desc_parts]
    result_df = base_df.iloc[:min_len].reset_index(drop=True)

    desc_all = pd.concat(desc_parts, axis=1)
    desc_all = desc_all.loc[:, ~desc_all.columns.duplicated()].copy()

    X_df, missing_desc = qspr_prog_prepare_descriptor_frame_for_model(
        desc_all=desc_all,
        required_desc=required_desc,
    )

    result_df["descriptor_source_for_prediction"] = str(descriptor_source)
    result_df["n_required_descriptors"] = len(required_desc)
    result_df["n_calculated_descriptor_columns"] = desc_all.shape[1]
    result_df["n_missing_descriptors_filled_by_zero"] = len(missing_desc)

    return result_df, X_df, missing_desc


def qspr_prog_calculate_molecular_descriptors_for_prediction(
    new_df,
    smiles_col,
    descriptor_mode,
    desc_lists,
):
    """
    Молекулярные дескрипторы для прогноза.
    Для одной молекулы добавляет dummy-структуры, чтобы не потерять все колонки.
    """
    work_df = new_df.copy().reset_index(drop=True)
    original_n = len(work_df)

    if original_n == 0:
        raise ValueError("Нет новых веществ для расчёта молекулярных дескрипторов.")

    if original_n == 1:
        dummy_df = pd.DataFrame([
            {smiles_col: "CC"},
            {smiles_col: "c1ccccc1"},
        ])

        for col in work_df.columns:
            if col not in dummy_df.columns:
                dummy_df[col] = ""

        for col in dummy_df.columns:
            if col not in work_df.columns:
                work_df[col] = ""

        work_for_desc = pd.concat(
            [work_df, dummy_df[work_df.columns]],
            ignore_index=True,
        )
    else:
        work_for_desc = work_df

    allowed_modes = [
        "🚀 Максимальная скорость (RDKit)",
        "👁️‍🗨️ Расширенный (Mordred)",
        "⚙️ Расширенный (Mordred)",
        "⚡ Умный (Mordred + уникальные PaDEL)",
        "🎯 Максимальная точность",
        "rdkit_fast",
        "mordred",
        "mordred_padel_unique",
        "max_accuracy",
    ]

    if descriptor_mode not in allowed_modes:
        descriptor_mode = "👁️‍🗨️ Расширенный (Mordred)"

    bundle = qspr_calculate_molecular_descriptors(
        data=work_for_desc,
        smiles_col=smiles_col,
        target_col=None,
        mode=descriptor_mode,
        desc_lists=desc_lists,
        preserve_constant_columns=True,
    )

    desc_df_all = bundle["df_desc"].copy()
    valid_indices_all = list(bundle["valid_indices"])

    keep_positions = [
        pos for pos, original_idx in enumerate(valid_indices_all)
        if int(original_idx) < original_n
    ]

    if not keep_positions:
        raise ValueError("Не удалось рассчитать молекулярные дескрипторы.")

    return desc_df_all.iloc[keep_positions].reset_index(drop=True)


def qspr_prog_predict_from_smiles_by_model_source(
    new_df,
    smiles_col,
    required_desc,
    target_col,
    fallback_X_train_scaled=None,
):
    """
    Прогноз по SMILES с расчётом дескрипторов, соответствующих модели.
    """
    descriptor_source = qspr_prog_get_model_descriptor_source()

    descriptor_mode = st.session_state.get(
        "molecular_descriptor_calculation_mode",
        st.session_state.get(
            "descriptor_calculation_mode",
            "👁️‍🗨️ Расширенный (Mordred)",
        ),
    )

    result_df, X_df, missing_desc = qspr_prog_calculate_descriptors_for_smiles_by_model_source(
        new_df=new_df,
        smiles_col=smiles_col,
        required_desc=required_desc,
        descriptor_source=descriptor_source,
        descriptor_mode=descriptor_mode,
        desc_lists=st.session_state.get("desc_lists"),
    )

    pred_df, ad_done = qspr_prog_predict_from_matrix(
        X_new=X_df.values.astype(float),
        new_df=result_df,
        target_col=target_col,
        fallback_X_train_scaled=fallback_X_train_scaled,
    )

    if missing_desc:
        pred_df["Отсутствующих дескрипторов заполнено нулями"] = len(missing_desc)

    return pred_df, ad_done

# ------------------------------------------------------------------
# Streamlit UI blocks


def qspr_prog_show_extended_ad_summary(pred_df):
    required_columns = {
        "molecular_ad_status",
        "spectral_ad_status",
        "combined_ad_status",
    }
    if not required_columns.issubset(set(pred_df.columns)):
        return

    st.markdown(t("prediction_page.spectral_ad_title"))

    status_labels = {
        "inside": t("prediction_page.ad_status_inside"),
        "outside": t("prediction_page.ad_status_outside"),
        "not_calculated": t("prediction_page.ad_status_not_calculated"),
    }

    def status_summary(column):
        values = pred_df[column].astype(str)
        return (
            f"{status_labels['inside']}: {int((values == 'inside').sum())}; "
            f"{status_labels['outside']}: {int((values == 'outside').sum())}; "
            f"{status_labels['not_calculated']}: "
            f"{int((values == 'not_calculated').sum())}"
        )

    ad_cols = st.columns(3)
    ad_cols[0].metric("Molecular AD", status_summary("molecular_ad_status"))
    ad_cols[1].metric("Spectral AD", status_summary("spectral_ad_status"))
    ad_cols[2].metric("Combined AD", status_summary("combined_ad_status"))

    interpretations = [
        str(value)
        for value in pred_df.get("spectral_ad_interpretation", [])
        if str(value).strip()
    ]
    for message in list(dict.fromkeys(interpretations))[:3]:
        if message == t("prediction_page.spectral_ad_interp_reliable"):
            st.success(message)
        elif message == t("prediction_page.spectral_ad_interp_not_calculated"):
            st.info(message)
        else:
            st.warning(message)


def qspr_prog_show_ad_summary(pred_df):
    """
    Единый блок отображения AD-сводки для новых прогнозов.
    """
    n_total = len(pred_df)
    n_out = int((pred_df["AD-статус"] == "вне AD").sum())

    qspr_prog_show_extended_ad_summary(pred_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Новых веществ", n_total)
    with col2:
        st.metric("Вне AD", n_out)
    with col3:
        st.metric("Вне AD, %", f"{n_out / n_total * 100:.1f}%" if n_total else "0.0%")
    with col4:
        if "Порог h*" in pred_df.columns and len(pred_df) > 0:
            st.metric("Порог h*", f"{float(pred_df['Порог h*'].iloc[0]):.4f}")
        else:
            st.metric("Порог h*", "—")

    if n_out > 0:
        st.warning(
            "Часть новых веществ находится вне применимой области прогностической модели. "
            "Для них прогноз следует считать экстраполяционным и менее надёжным."
        )
    else:
        st.success("Все новые вещества находятся внутри применимой области прогностической модели по leverage.")

    with st.expander("🧭 Пояснение Applicability Domain для новых прогнозов", expanded=False):
        st.markdown(
            """
            **Leverage h** показывает, насколько новое вещество далеко от обучающей
            выборки в пространстве дескрипторов.

            Классический порог:

            `h* = 3(p + 1) / n`

            где `p` — число дескрипторов, `n` — число веществ в обучающей выборке
            прогностической модели.

            Если `h > h*`, прогноз считается находящимся **вне применимой области модели**
            и должен интерпретироваться осторожно.
            """
        )

    if n_out > 0:
        st.markdown("### Новые вещества вне применимой области")
        st.dataframe(
            pred_df[pred_df["AD-статус"] == "вне AD"].copy(),
            width="stretch",
            hide_index=True,
        )


def qspr_show_prognostic_training_section(
    data,
    model_name,
    target_col,
    smiles_col_current,
    desc_names_current,
    X_all_current,
    y_all_current,
    valid_indices_current,
    get_model_params_from_session,
    add_log=None,
):
    """
    Streamlit-блок обучения и сохранения прогностической модели.
    """
    st.header("🔮 Прогностическая модель")
    st.markdown(
        "Все вещества выбраны по умолчанию. "
        "Снимите галочки у веществ, которые нужно исключить "
        "из финальной прогностической модели."
    )

    trained_models_available = list(
        st.session_state.get("trained_models", {}).keys()
    )

    if not trained_models_available:
        st.info("Сначала обучите хотя бы одну аналитическую модель.")
        return

    best_model_from_comparison = st.session_state.get(
        "best_model_from_comparison",
        None
    )

    if (
        best_model_from_comparison is not None
        and best_model_from_comparison in trained_models_available
    ):
        default_model_for_prog = best_model_from_comparison
    elif model_name in trained_models_available:
        default_model_for_prog = model_name
    else:
        default_model_for_prog = trained_models_available[0]

    default_model_index = trained_models_available.index(default_model_for_prog)

    model_name_for_prog = st.selectbox(
        "Какую модель использовать как основу прогностической модели",
        trained_models_available,
        index=default_model_index,
        key="prog_model_source_select",
        help=(
            "Если выполнено сравнение моделей, по умолчанию выбирается модель, "
            "занявшая 1 место в таблице сравнения."
        )
    )

    if best_model_from_comparison is not None:
        if best_model_from_comparison == model_name_for_prog:
            st.success(
                f"Выбрана лучшая модель по сравнению моделей: "
                f"`{best_model_from_comparison}`"
            )
        else:
            st.info(
                f"Лучшая модель по сравнению: `{best_model_from_comparison}`. "
                f"Сейчас вручную выбрана: `{model_name_for_prog}`."
            )

    model_data = st.session_state.trained_models[model_name_for_prog]
    model = model_data["model"]

    # Для обычных моделей можно использовать X_scaled.
    # Для auto-режима модель является Pipeline и сама выполняет
    # отбор/масштабирование, поэтому ей нужна исходная X_original.
    if "X_original" in model_data:
        X_for_predict = model_data["X_original"]
    else:
        X_for_predict = model_data.get("X_scaled", X_all_current)

    try:
        y_pred_all = np.ravel(model.predict(X_for_predict))
    except Exception as e:
        st.error(
            "Не удалось получить расчётные значения аналитической модели для "
            "таблицы прогностического отбора. Скорее всего, модель была обучена "
            "на другом наборе дескрипторов или после автоотбора изменилась "
            "размерность матрицы признаков."
        )
        st.exception(e)
        st.info("Переобучите аналитическую модель на текущем наборе дескрипторов.")
        return

    prog_table = qspr_prog_make_training_table(
        data=data,
        smiles_col=smiles_col_current,
        valid_indices=valid_indices_current,
        y_true=y_all_current,
        y_pred=y_pred_all,
    )

    edited = st.data_editor(
        prog_table,
        column_config={
            "Выбрать": st.column_config.CheckboxColumn(
                "Использовать",
                help="Снимите галочку, если вещество нужно исключить из прогностической модели",
                default=True,
            ),
        },
        disabled=["№", "Индекс", "SMILES", "Экспериментальное значение", "Расчётное значение", "Ошибка"],
        hide_index=True,
        width="stretch",
        key=f"prog_editor_{model_name_for_prog}",
    )

    selected_positions = edited.index[edited["Выбрать"] == True].tolist()

    if st.button("🧭 Обучить прогностическую модель на отобранных веществах", type="primary", key=f"train_prog_{model_name_for_prog}"):
        try:
            package = qspr_prog_train_selected_model(
                data=data,
                smiles_col=smiles_col_current,
                target_col=target_col,
                X_all=X_all_current,
                y_all=y_all_current,
                valid_indices=valid_indices_current,
                selected_positions=selected_positions,
                desc_names=desc_names_current,
                algorithm=model_name_for_prog,
                params=get_model_params_from_session(),
            )

            qspr_save_results_auto(package["train_df"], "prognostic", target_col, len(package["y_train"]))
            qspr_prog_store_model_in_session(package)

            st.success(f"Прогностическая модель обучена на {len(package['y_train'])} веществах.")
            st.metric("R² на обучающих данных", f"{package['r2_train']:.3f}")

        except Exception as e:
            st.error(f"Ошибка обучения прогностической модели: {e}")

    if "prog_model" in st.session_state and st.button(
        "💾 Сохранить прогностическую модель полным пакетом",
        key="save_prog"
    ):
        package = qspr_prog_make_save_package(
            target_col=target_col,
            smiles_col=smiles_col_current,
            desc_names=desc_names_current,
            model_name=st.session_state.get(
                "prog_model_name",
                model_name_for_prog
            ),
        )

        model_path, latest_path = qspr_prog_save_package_to_folder(package)

        st.success(f"Прогностическая модель сохранена: `{model_path}`")
        st.caption(f"Копия последней модели: `{latest_path}`")

        if add_log is not None:
            add_log(f"Прогностическая модель сохранена: {model_path}")


def qspr_show_prediction_uncertainty_controls():
    with st.expander(t("prediction_page.uncertainty_expander"), expanded=False):
        st.checkbox(
            t("prediction_page.full_uncertainty"),
            value=st.session_state.get(
                "prediction_uncertainty_enabled",
                True,
            ),
            key="prediction_uncertainty_enabled",
            help=t("prediction_page.full_uncertainty_help"),
        )

        col_unc_1, col_unc_2 = st.columns(2)
        with col_unc_1:
            st.slider(
                t("prediction_page.confidence_main"),
                min_value=0.80,
                max_value=0.99,
                value=float(
                    st.session_state.get(
                        "prediction_uncertainty_confidence",
                        0.90,
                    )
                ),
                step=0.01,
                key="prediction_uncertainty_confidence",
            )
            st.number_input(
                t("prediction_page.bootstrap_main"),
                min_value=20,
                max_value=1000,
                value=int(
                    st.session_state.get(
                        "prediction_uncertainty_bootstrap",
                        100,
                    )
                ),
                step=20,
                key="prediction_uncertainty_bootstrap",
            )
        with col_unc_2:
            st.number_input(
                t("prediction_page.cv_main"),
                min_value=2,
                max_value=20,
                value=int(
                    st.session_state.get(
                        "prediction_uncertainty_cv",
                        5,
                    )
                ),
                step=1,
                key="prediction_uncertainty_cv",
            )
            st.number_input(
                t("prediction_page.neighbors_main"),
                min_value=1,
                max_value=50,
                value=int(
                    st.session_state.get(
                        "prediction_uncertainty_neighbors",
                        5,
                    )
                ),
                step=1,
                key="prediction_uncertainty_neighbors",
            )

        trained_names = list(
            (st.session_state.get("trained_models", {}) or {}).keys()
        )
        st.checkbox(
            t("prediction_page.use_consensus_main"),
            value=st.session_state.get(
                "prediction_uncertainty_use_consensus",
                True,
            ),
            key="prediction_uncertainty_use_consensus",
        )
        if trained_names:
            saved_names = st.session_state.get(
                "prediction_uncertainty_consensus_models",
                trained_names[:4],
            )
            defaults = [name for name in saved_names if name in trained_names]
            st.multiselect(
                t("prediction_page.consensus_models_main"),
                options=trained_names,
                default=defaults,
                max_selections=6,
                key="prediction_uncertainty_consensus_models",
            )

        st.caption(
            t("prediction_page.settings_note")
        )


def qspr_show_new_compound_prediction_section(
    target_col,
    desc_names_current,
    fallback_X_train_scaled=None,
    show_uncertainty_controls=True,
):
    """
    Streamlit-блок прогноза свойства для новых веществ.
    """
    st.header(t("prediction_page.new_compound_header"))
    st.markdown(t("prediction_page.new_compound_description"))

    if "prog_model" not in st.session_state:
        st.info(t("prediction_page.train_prog_first"))
        return

    if show_uncertainty_controls:
        qspr_show_prediction_uncertainty_controls()

    legacy_prediction_mode = st.session_state.get("new_prediction_mode")
    if legacy_prediction_mode not in {"smiles", "descriptors"}:
        if legacy_prediction_mode and "SMILES" in str(legacy_prediction_mode):
            st.session_state["new_prediction_mode"] = "smiles"
        else:
            st.session_state["new_prediction_mode"] = "descriptors"

    prediction_mode = st.radio(
        t("prediction_page.prediction_mode"),
        ["smiles", "descriptors"],
        format_func=lambda mode: {
            "smiles": t("prediction_page.mode_smiles"),
            "descriptors": t("prediction_page.mode_descriptors"),
        }[mode],
        horizontal=True,
        key="new_prediction_mode",
    )

    required_desc = list(st.session_state.get("prog_desc_names", desc_names_current))

    if prediction_mode == "smiles":
        with st.expander(t("prediction_page.draw_molecule_expander"), expanded=True):
            qspr_prog_show_molecular_editor_prediction_ui(
                required_desc=required_desc,
                target_col=target_col,
                fallback_X_train_scaled=fallback_X_train_scaled,
            )

    st.markdown(t("prediction_page.file_prediction_header"))

    new_file = st.file_uploader(
        t("prediction_page.new_file"),
        type=["csv", "xlsx"],
        key="new_prediction_file",
    )

    if new_file is None:
        return

    try:
        if new_file.name.lower().endswith(".csv"):
            new_df = pd.read_csv(new_file)
        else:
            new_df = pd.read_excel(new_file)

        new_df.columns = new_df.columns.str.strip()
        st.dataframe(new_df.head(50), width="stretch")

        if prediction_mode == "smiles":
            qspr_prog_show_smiles_prediction_ui(
                new_df,
                required_desc,
                target_col,
                fallback_X_train_scaled,
            )
        else:
            qspr_prog_show_ready_descriptor_prediction_ui(
                new_df,
                required_desc,
                target_col,
                fallback_X_train_scaled,
            )

    except Exception as e:
        st.error(t("prediction_page.new_prediction_error", error=e))

def qspr_prog_show_molecular_editor_prediction_ui(
    required_desc,
    target_col,
    fallback_X_train_scaled=None,
):
    """
    Молекулярный редактор для прогноза одного нового вещества.
    Работает в режиме SMILES: структура -> SMILES -> дескрипторы -> прогноз.
    """
    st.markdown(t("prediction_page.molecular_editor_header"))

    st.caption(t("prediction_page.molecular_editor_caption"))

    smiles_from_editor = ""

    if ketcher_available:
        smiles_from_editor = st_ketcher(
            "",
            height=450,
            key="prognostic_ketcher_editor",
        )

        if smiles_from_editor is None:
            smiles_from_editor = ""

        smiles_from_editor = str(smiles_from_editor).strip()

        if smiles_from_editor:
            if st.session_state.get("prognostic_editor_last_smiles") != smiles_from_editor:
                st.session_state["prognostic_editor_smiles_input"] = smiles_from_editor
                st.session_state["prognostic_editor_last_smiles"] = smiles_from_editor
    else:
        st.warning(
            t("prediction_page.ketcher_missing")
        )

    manual_smiles = st.text_input(
        t("prediction_page.smiles_input_label"),
        value=st.session_state.get("prognostic_editor_smiles_input", smiles_from_editor),
        key="prognostic_editor_smiles_input",
        help=t("prediction_page.smiles_input_help"),
    )

    compound_name = st.text_input(
        t("prediction_page.compound_name_label"),
        value="drawn_compound",
        key="prognostic_editor_compound_name",
    )

    descriptor_source = qspr_prog_get_model_descriptor_source()
    flags = qspr_prog_descriptor_source_flags(descriptor_source)

    if flags["has_spectral"]:
        st.error(t("prediction_page.spectral_model_draw_block"))
        st.info(t("prediction_page.spectral_model_file_advice"))
        return

    if flags["need_xtb"] or flags["need_morfeus"] or flags["need_dscribe"]:
        st.warning(t("prediction_page.model_needs_3d_warning"))

    if not st.button(
        t("prediction_page.run_drawn_prediction"),
        type="primary",
        key="run_prediction_from_molecular_editor",
    ):
        return

    smiles = str(manual_smiles).strip()

    if not smiles:
        st.error(t("prediction_page.empty_smiles_error"))
        return

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error(t("prediction_page.invalid_smiles_error"))
        return

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    needed_parts = []

    if flags["need_molecular"]:
        needed_parts.append(t("prediction_page.part_molecular"))
    if flags["need_xtb"]:
        needed_parts.append("xTB")
    if flags["need_morfeus"]:
        needed_parts.append("morfeus")
    if flags["need_dscribe"]:
        needed_parts.append("DScribe")

    st.info(
        t(
            "prediction_page.descriptor_calc_info",
            source=descriptor_source,
            parts=", ".join(needed_parts) or t("prediction_page.part_unknown"),
            count=len(required_desc),
        )
    )

    new_df = pd.DataFrame([
        {
            "name": compound_name,
            "SMILES": canonical_smiles,
        }
    ])

    with st.spinner(t("prediction_page.drawn_spinner")):
        try:
            pred_df, ad_done = qspr_prog_predict_from_smiles_by_model_source(
                new_df=new_df,
                smiles_col="SMILES",
                required_desc=required_desc,
                target_col=target_col,
                fallback_X_train_scaled=fallback_X_train_scaled,
            )
        except Exception as e:
            st.error(t("prediction_page.drawn_prediction_error", error=e))
            return

    qspr_prog_show_prediction_result(
        pred_df=pred_df,
        ad_done=ad_done,
        download_key="download_prediction_from_molecular_editor",
    )

def qspr_prog_show_smiles_prediction_ui(new_df, required_desc, target_col, fallback_X_train_scaled=None):
    """
    UI-подблок прогноза по SMILES.
    Рассчитывает тот набор дескрипторов, который соответствует текущей модели.
    """
    descriptor_source = qspr_prog_get_model_descriptor_source()
    flags = qspr_prog_descriptor_source_flags(descriptor_source)

    if flags["has_spectral"]:
        st.error(t("prediction_page.spectral_smiles_warning"))
        st.info(t("prediction_page.spectral_model_file_advice"))
        return

    possible_smiles_cols = [
        c for c in new_df.columns
        if c.lower() in ["smiles", "canonical_smiles", "input_smiles"]
    ]

    if possible_smiles_cols:
        smiles_col = st.selectbox(
            t("prediction_page.smiles_column"),
            possible_smiles_cols,
            key="new_prediction_smiles_col",
        )
    else:
        smiles_col = st.selectbox(
            t("prediction_page.smiles_column"),
            list(new_df.columns),
            key="new_prediction_smiles_col_manual",
        )

    needed_parts = []

    if flags["need_molecular"]:
        needed_parts.append(t("prediction_page.part_molecular"))
    if flags["need_xtb"]:
        needed_parts.append("xTB")
    if flags["need_morfeus"]:
        needed_parts.append("morfeus")
    if flags["need_dscribe"]:
        needed_parts.append("DScribe")

    st.info(
        t(
            "prediction_page.smiles_descriptor_info",
            source=descriptor_source,
            parts=", ".join(needed_parts) or t("prediction_page.part_unknown"),
            count=len(required_desc),
        )
    )

    if not st.button(t("prediction_page.run_prediction"), type="primary", key="run_new_prediction_smiles"):
        return

    with st.spinner(t("prediction_page.prediction_spinner")):
        try:
            pred_df, ad_done = qspr_prog_predict_from_smiles_by_model_source(
                new_df=new_df,
                smiles_col=smiles_col,
                required_desc=required_desc,
                target_col=target_col,
                fallback_X_train_scaled=fallback_X_train_scaled,
            )
        except Exception as e:
            st.error(t("prediction_page.smiles_prediction_error", error=e))
            return

    qspr_prog_show_prediction_result(
        pred_df,
        ad_done,
        "download_new_predictions_smiles",
    )


def qspr_prog_show_ready_descriptor_prediction_ui(new_df, required_desc, target_col, fallback_X_train_scaled=None):
    """
    UI-подблок прогноза по готовым дескрипторам.
    """
    st.caption(t("prediction_page.ready_descriptor_caption"))

    available_desc = [c for c in required_desc if c in new_df.columns]
    missing_desc = [c for c in required_desc if c not in new_df.columns]

    st.write(t(
        "prediction_page.descriptor_found_count",
        found=len(available_desc),
        total=len(required_desc),
    ))

    if missing_desc:
        with st.expander(t("prediction_page.show_missing_descriptors"), expanded=False):
            st.write(missing_desc[:300])

    if not st.button(t("prediction_page.run_ready_descriptor_prediction"), type="primary", key="run_new_prediction_descriptors"):
        return

    if len(available_desc) == 0:
        st.error(t("prediction_page.no_required_descriptors"))
        return

    X_new = qspr_prog_prepare_ready_descriptor_matrix(new_df, required_desc)
    pred_df, ad_done = qspr_prog_predict_from_matrix(
        X_new=X_new,
        new_df=new_df,
        target_col=target_col,
        fallback_X_train_scaled=fallback_X_train_scaled,
    )

    qspr_prog_show_prediction_result(pred_df, ad_done, "download_new_predictions_descriptors")


def qspr_prog_show_prediction_result(pred_df, ad_done, download_key):
    """
    Единый вывод результата прогноза.
    """
    st.success(t("prediction_page.prediction_done"))

    if ad_done:
        qspr_prog_show_ad_summary(pred_df)

    uncertainty_payload = st.session_state.get(
        "prediction_uncertainty_result",
        {},
    )
    if uncertainty_payload.get("error"):
        st.warning(uncertainty_payload["error"])

    if "Надёжность" in pred_df.columns:
        statuses = pred_df["Надёжность"].astype(str)
        high = int((statuses == "высокая").sum())
        medium = int((statuses == "средняя").sum())
        low = int((statuses == "низкая").sum())
        col_rel_1, col_rel_2, col_rel_3, col_rel_4 = st.columns(4)
        col_rel_1.metric("Объектов", len(pred_df))
        col_rel_2.metric("Высокая надёжность", high)
        col_rel_3.metric("Средняя надёжность", medium)
        col_rel_4.metric("Низкая надёжность", low)

    st.dataframe(pred_df, width="stretch", hide_index=True)

    diagnostics = uncertainty_payload.get("diagnostics", {})
    if diagnostics:
        with st.expander("📊 Диагностика неопределённости", expanded=False):
            diagnostic_df = pd.DataFrame({
                "Показатель": list(diagnostics.keys()),
                "Значение": list(diagnostics.values()),
            })
            st.dataframe(
                diagnostic_df,
                width="stretch",
                hide_index=True,
            )

            model_names = uncertainty_payload.get("consensus_models", [])
            weights = uncertainty_payload.get("consensus_weights", {})
            cv_rmse = uncertainty_payload.get("consensus_cv_rmse", {})
            if model_names:
                model_df = pd.DataFrame([
                    {
                        "Модель": name,
                        "Вес consensus": weights.get(name, np.nan),
                        "CV RMSE": cv_rmse.get(name, np.nan),
                    }
                    for name in model_names
                ])
                st.dataframe(
                    model_df,
                    width="stretch",
                    hide_index=True,
                )

            st.caption(
                "Prediction interval калибруется по out-of-fold остаткам. "
                "Bootstrap, consensus и GPR std не заменяют prediction interval."
            )

    neighbours = uncertainty_payload.get("neighbours")
    if isinstance(neighbours, pd.DataFrame) and not neighbours.empty:
        with st.expander("🧭 Ближайшие обучающие аналоги", expanded=False):
            st.dataframe(
                neighbours,
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "📥 Скачать таблицу аналогов CSV",
                neighbours.to_csv(index=False).encode("utf-8"),
                "prediction_nearest_analogues.csv",
                "text/csv",
                key=f"{download_key}_neighbours",
            )

    st.download_button(
        "📥 Скачать прогноз CSV",
        pred_df.to_csv(index=False).encode("utf-8"),
        "new_compounds_predictions.csv",
        "text/csv",
        key=download_key,
    )


def qspr_prog_find_available_model_rows():
    available_model_rows = []

    search_folders = [
        os.path.join(os.getcwd(), "prognostic_models"),
    ]

    seen_model_paths = set()

    for folder in search_folders:
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            filename_lower = filename.lower()

            if not filename_lower.startswith("model_prognostic_"):
                continue

            if not (
                filename_lower.endswith(".pkl")
                or filename_lower.endswith(".joblib")
            ):
                continue

            model_path = os.path.abspath(os.path.join(folder, filename))

            if model_path in seen_model_paths:
                continue

            seen_model_paths.add(model_path)

            try:
                package = joblib.load(model_path)

                if not isinstance(package, dict):
                    continue

                if "model" not in package or "desc_names" not in package:
                    continue

                target_name = package.get("target_col", "property")
                model_name_from_file = package.get("model_name", "loaded_model")
                descriptor_source = package.get("descriptor_source", "")
                created_at = package.get("created_at", "")
                n_desc = len(package.get("desc_names", []))
                n_train_compounds = package.get("n_train_compounds", None)

                if n_train_compounds is not None:
                    try:
                        n_train_compounds = int(n_train_compounds)
                    except Exception:
                        n_train_compounds = ""
                else:
                    y_train_saved = package.get("y_train", None)
                    train_smiles_saved = package.get("train_smiles", None)
                    x_train_saved = package.get("X_train_raw", None)

                    if y_train_saved is not None:
                        try:
                            n_train_compounds = len(y_train_saved)
                        except Exception:
                            n_train_compounds = ""
                    elif train_smiles_saved is not None:
                        try:
                            n_train_compounds = len(train_smiles_saved)
                        except Exception:
                            n_train_compounds = ""
                    elif x_train_saved is not None:
                        try:
                            n_train_compounds = len(x_train_saved)
                        except Exception:
                            n_train_compounds = ""
                    else:
                        n_train_compounds = ""

                available_model_rows.append({
                    "Свойство": target_name,
                    "Модель": model_name_from_file,
                    "Веществ": n_train_compounds,
                    "Дескрипторов": n_desc,
                    "Источник дескрипторов": descriptor_source,
                    "Создана": created_at,
                    "Файл": filename,
                    "Путь": model_path,
                })

            except Exception:
                continue

    return available_model_rows


def qspr_prog_build_available_model_cards():
    cards = []

    for row in qspr_prog_find_available_model_rows():
        package_path = row.get("Путь", "")
        package = None
        try:
            package = joblib.load(package_path)
        except Exception:
            package = None

        desc_names = []
        if isinstance(package, dict):
            desc_names = safe_list(package.get("desc_names", []))
        descriptor_source = row.get("Источник дескрипторов", "")
        if not descriptor_source:
            descriptor_source = qspr_prog_infer_descriptor_source_from_desc_names(desc_names)

        flags = qspr_prog_descriptor_source_flags(descriptor_source)
        descriptor_counts = qspr_prog_descriptor_category_counts(desc_names)

        chemical_scope = None
        metrics = {}
        target_units = ""

        if isinstance(package, dict):
            chemical_scope = package.get("chemical_scope", None)
            metrics = package.get("metrics", {}) or package.get("model_metrics", {}) or {}
            target_units = package.get("target_units", package.get("units", ""))

        model_id = os.path.splitext(str(row.get("Файл", "")))[0]
        cards.append({
            "model_id": model_id,
            "property": row.get("Свойство", ""),
            "property_label": row.get("Свойство", ""),
            "target_units": target_units,
            "model_name": row.get("Модель", ""),
            "n_train": row.get("Веществ", ""),
            "metrics": metrics,
            "chemical_scope": chemical_scope,
            "descriptor_set": descriptor_source,
            "descriptor_source": descriptor_source,
            "descriptor_counts": descriptor_counts,
            "descriptor_flags": flags,
            "n_descriptors": row.get("Дескрипторов", ""),
            "package_path": package_path,
            "filename": row.get("Файл", ""),
        })

    if "prog_model" in st.session_state:
        desc_names = list(st.session_state.get("prog_desc_names", []))
        descriptor_source = qspr_prog_get_model_descriptor_source()
        y_train = st.session_state.get("prog_y_train", None)
        cards.append({
            "model_id": "current_session_model",
            "property": st.session_state.get("prog_target_col", "property"),
            "property_label": st.session_state.get("prog_target_col", "property"),
            "target_units": "",
            "model_name": st.session_state.get("prog_model_name", "loaded_model"),
            "n_train": safe_len(y_train),
            "metrics": {},
            "chemical_scope": "unknown",
            "descriptor_set": descriptor_source,
            "descriptor_source": descriptor_source,
            "descriptor_counts": qspr_prog_descriptor_category_counts(desc_names),
            "descriptor_flags": qspr_prog_descriptor_source_flags(descriptor_source),
            "n_descriptors": len(desc_names),
            "package_path": "",
            "filename": "current session",
        })

    return cards


def qspr_prog_smiles_runnable_for_card(card):
    flags = card.get("descriptor_flags") or qspr_prog_descriptor_source_flags(
        card.get("descriptor_source", card.get("descriptor_set", ""))
    )

    if flags.get("has_spectral"):
        return False, t("prediction_page.capability_reason_spectral_descriptors")

    if flags.get("need_xtb") or flags.get("need_morfeus") or flags.get("need_dscribe"):
        return True, t("prediction_page.capability_reason_requires_3d")

    return True, t("prediction_page.capability_reason_smiles_runnable")


def qspr_show_prediction_sidebar():
    """
    Sidebar автономного режима прогноза.
    """
    st.header(t("prediction_page.sidebar_title"))

    st.checkbox(
        t("prediction_page.full_uncertainty"),
        value=st.session_state.get("prediction_uncertainty_enabled", True),
        key="prediction_uncertainty_enabled",
        help=t("prediction_page.full_uncertainty_help"),
    )
    st.slider(
        t("prediction_page.confidence"),
        min_value=0.80,
        max_value=0.99,
        value=float(st.session_state.get("prediction_uncertainty_confidence", 0.90)),
        step=0.01,
        key="prediction_uncertainty_confidence",
    )
    st.number_input(
        t("prediction_page.bootstrap"),
        min_value=20,
        max_value=1000,
        value=int(st.session_state.get("prediction_uncertainty_bootstrap", 100)),
        step=20,
        key="prediction_uncertainty_bootstrap",
    )
    st.number_input(
        t("prediction_page.cv_main"),
        min_value=2,
        max_value=20,
        value=int(st.session_state.get("prediction_uncertainty_cv", 5)),
        step=1,
        key="prediction_uncertainty_cv",
    )
    st.number_input(
        t("prediction_page.neighbors"),
        min_value=1,
        max_value=50,
        value=int(st.session_state.get("prediction_uncertainty_neighbors", 5)),
        step=1,
        key="prediction_uncertainty_neighbors",
    )

    trained_names = list((st.session_state.get("trained_models", {}) or {}).keys())
    st.checkbox(
        t("prediction_page.use_consensus_short"),
        value=st.session_state.get("prediction_uncertainty_use_consensus", True),
        key="prediction_uncertainty_use_consensus",
    )
    if trained_names:
        saved_names = st.session_state.get(
            "prediction_uncertainty_consensus_models",
            trained_names[:4],
        )
        defaults = [name for name in saved_names if name in trained_names]
        st.multiselect(
            t("prediction_page.consensus_models_short"),
            options=trained_names,
            default=defaults,
            max_selections=6,
            key="prediction_uncertainty_consensus_models",
        )

    st.divider()
    st.subheader(t("prediction_page.model_source"))

    available_model_rows = qspr_prog_find_available_model_rows()

    if available_model_rows:
        available_model_df = pd.DataFrame(available_model_rows)
        available_model_df = available_model_df.sort_values(
            by=["Свойство", "Модель", "Файл"],
            ascending=True
        ).reset_index(drop=True)

        selected_model_index = st.selectbox(
            t("prediction_page.available_model"),
            options=available_model_df.index.tolist(),
            format_func=lambda i: (
                f"{available_model_df.loc[i, 'Свойство']} | "
                f"{available_model_df.loc[i, 'Модель']} | "
                f"{available_model_df.loc[i, 'Источник дескрипторов']} | "
                f"{available_model_df.loc[i, 'Дескрипторов']} desc | "
                f"{available_model_df.loc[i, 'Файл']}"
            ),
            key="available_prognostic_model_select",
        )

        selected_model_row = available_model_df.loc[selected_model_index]

        if st.button(
            t("prediction_page.connect_model"),
            type="primary",
            key="load_selected_available_prognostic_model",
        ):
            try:
                package = joblib.load(selected_model_row["Путь"])
                qspr_prog_load_saved_package_to_session(package)
                st.success(t("prediction_page.model_connected"))
                st.rerun()
            except Exception as e:
                st.error(t("prediction_page.model_connect_error", error=e))
    else:
        qspr_prog_try_autoload_default_model()

    model_file = st.file_uploader(
        t("prediction_page.upload_model"),
        type=["pkl", "joblib"],
        key="standalone_prog_model_file",
    )

    if model_file is not None:
        st.warning(
            "Upload only models from trusted sources. Pickle/joblib files can execute Python code when loaded."
        )
        try:
            package = joblib.load(model_file)
            qspr_prog_load_saved_package_to_session(package)
            st.success(t("prediction_page.model_loaded_short"))
        except Exception as e:
            st.error(t("prediction_page.model_load_error", error=e))

    if "prog_model" in st.session_state:
        st.caption(
            t(
                "prediction_page.active_model_caption",
                model=st.session_state.get('prog_model_name', 'loaded_model'),
                target=st.session_state.get('prog_target_col', 'property'),
                count=len(st.session_state.get('prog_desc_names', [])),
            )
        )
        qspr_show_loaded_model_descriptor_summary()


def qspr_show_prediction_model_source_controls():
    st.markdown(f"### {t('prediction_page.model_source')}")

    available_model_rows = qspr_prog_find_available_model_rows()

    if available_model_rows:
        available_model_df = pd.DataFrame(available_model_rows)
        available_model_df = available_model_df.sort_values(
            by=["Свойство", "Модель", "Файл"],
            ascending=True
        ).reset_index(drop=True)

        selected_model_index = st.selectbox(
            t("prediction_page.available_model"),
            options=available_model_df.index.tolist(),
            format_func=lambda i: (
                f"{available_model_df.loc[i, 'Свойство']} | "
                f"{available_model_df.loc[i, 'Модель']} | "
                f"{available_model_df.loc[i, 'Источник дескрипторов']} | "
                f"{available_model_df.loc[i, 'Дескрипторов']} desc | "
                f"{available_model_df.loc[i, 'Файл']}"
            ),
            key="available_prognostic_model_select",
        )

        selected_model_row = available_model_df.loc[selected_model_index]

        if st.button(
            t("prediction_page.connect_model"),
            type="primary",
            key="load_selected_available_prognostic_model",
        ):
            try:
                package = joblib.load(selected_model_row["Путь"])
                qspr_prog_load_saved_package_to_session(package)
                st.success(t("prediction_page.model_connected"))
                st.rerun()
            except Exception as e:
                st.error(t("prediction_page.model_connect_error", error=e))
    else:
        qspr_prog_try_autoload_default_model()

    model_file = st.file_uploader(
        t("prediction_page.upload_model"),
        type=["pkl", "joblib"],
        key="standalone_prog_model_file",
    )

    if model_file is not None:
        st.warning(
            "Upload only models from trusted sources. Pickle/joblib files can execute Python code when loaded."
        )
        try:
            package = joblib.load(model_file)
            qspr_prog_load_saved_package_to_session(package)
            st.success(t("prediction_page.model_loaded_short"))
        except Exception as e:
            st.error(t("prediction_page.model_load_error", error=e))

    if "prog_model" in st.session_state:
        st.caption(
            t(
                "prediction_page.active_model_caption",
                model=st.session_state.get('prog_model_name', 'loaded_model'),
                target=st.session_state.get('prog_target_col', 'property'),
                count=len(st.session_state.get('prog_desc_names', [])),
            )
        )
        qspr_show_loaded_model_descriptor_summary()


def qspr_format_capability_status(status):
    return {
        "suitable": t("prediction_page.capability_status_suitable"),
        "partial": t("prediction_page.capability_status_partial"),
        "not_suitable": t("prediction_page.capability_status_not_suitable"),
        "undefined": t("prediction_page.capability_status_undefined"),
    }.get(status, status)


def qspr_show_prediction_capability_tool():
    if classify_smiles_scope is None or find_applicable_models is None:
        st.error(t("prediction_page.capability_tool_unavailable"))
        return

    st.markdown(t("prediction_page.capability_intro"))

    smiles_text = st.text_area(
        t("prediction_page.capability_smiles_input"),
        value=st.session_state.get("prediction_capability_smiles", ""),
        key="prediction_capability_smiles",
        height=120,
    )

    if not st.button(
        t("prediction_page.capability_run"),
        type="primary",
        key="prediction_capability_run",
    ):
        return

    smiles_list = [
        line.strip()
        for line in str(smiles_text).replace(";", "\n").splitlines()
        if line.strip()
    ]

    if not smiles_list:
        st.warning(t("prediction_page.capability_no_smiles"))
        return

    model_cards = qspr_prog_build_available_model_cards()
    if not model_cards:
        st.warning(t("prediction_page.capability_no_models"))
        return

    all_rows = []
    runnable_options = {}

    for query_index, smiles in enumerate(smiles_list, start=1):
        scope = classify_smiles_scope(smiles)

        st.markdown(
            t(
                "prediction_page.capability_detected_header",
                index=query_index,
                smiles=smiles,
            )
        )

        if not scope.get("valid"):
            st.error(scope.get("error", "Invalid SMILES"))
            continue

        tag_items = sorted(set(scope.get("class_tags", [])) | set(scope.get("substructure_tags", [])))
        st.write(", ".join(tag_items) or t("prediction_page.part_unknown"))

        matches = find_applicable_models(scope, model_cards)

        for match in matches:
            card = match.get("card", {})
            runnable, runnable_reason = qspr_prog_smiles_runnable_for_card(card)
            action_available = (
                runnable
                and match.get("applicability") in {"suitable", "partial", "undefined"}
            )

            metrics = match.get("metrics", {}) or {}
            row = {
                t("prediction_page.capability_col_query"): query_index,
                t("prediction_page.capability_col_property"): match.get("property_label", ""),
                t("prediction_page.capability_col_model"): match.get("model_name", ""),
                t("prediction_page.capability_col_applicability"): qspr_format_capability_status(match.get("applicability")),
                t("prediction_page.capability_col_reason"): "; ".join(match.get("reasons", []) + [runnable_reason]),
                "R² CV": metrics.get("r2_cv", metrics.get("R2_CV", "")),
                "RMSE CV": metrics.get("rmse_cv", metrics.get("RMSE_CV", "")),
                t("prediction_page.capability_col_n_train"): match.get("n_train", ""),
                t("prediction_page.capability_col_units"): match.get("target_units", ""),
                t("prediction_page.capability_col_descriptors"): card.get("descriptor_set", ""),
                t("prediction_page.capability_col_action"): (
                    t("prediction_page.capability_action_predict")
                    if action_available
                    else "—"
                ),
            }
            all_rows.append(row)

            if action_available and card.get("package_path"):
                option_id = f"{query_index}::{card.get('package_path')}"
                runnable_options[option_id] = {
                    "label": (
                        f"{query_index}. {match.get('property_label', '')} | "
                        f"{match.get('model_name', '')} | {card.get('descriptor_set', '')}"
                    ),
                    "smiles": scope.get("canonical_smiles", smiles),
                    "package_path": card.get("package_path"),
                }

    if not all_rows:
        return

    st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)

    if not runnable_options:
        st.info(t("prediction_page.capability_no_runnable_models"))
        return

    selected_option = st.selectbox(
        t("prediction_page.capability_select_prediction"),
        options=list(runnable_options.keys()),
        format_func=lambda key: runnable_options[key]["label"],
        key="prediction_capability_selected_model",
    )

    if not st.button(
        t("prediction_page.capability_launch_prediction"),
        type="primary",
        key="prediction_capability_launch_prediction",
    ):
        return

    option = runnable_options[selected_option]
    try:
        package = joblib.load(option["package_path"])
        qspr_prog_load_saved_package_to_session(package)
    except Exception as e:
        st.error(t("prediction_page.model_load_error", error=e))
        return

    target_col = st.session_state.get("prog_target_col", "property")
    required_desc = list(st.session_state.get("prog_desc_names", []))
    new_df = pd.DataFrame([{"SMILES": option["smiles"]}])

    with st.spinner(t("prediction_page.prediction_spinner")):
        try:
            pred_df, ad_done = qspr_prog_predict_from_smiles_by_model_source(
                new_df=new_df,
                smiles_col="SMILES",
                required_desc=required_desc,
                target_col=target_col,
                fallback_X_train_scaled=st.session_state.get("prog_X_train_scaled", None),
            )
        except Exception as e:
            st.error(t("prediction_page.smiles_prediction_error", error=e))
            return

    qspr_prog_show_prediction_result(
        pred_df,
        ad_done,
        "download_capability_prediction",
    )


def qspr_show_standalone_prediction_page():
    """
    Отдельная страница прогноза свойства для новых веществ.

    Работает в двух режимах:
    1. если модель уже обучена в текущей сессии;
    2. если пользователь загрузил сохранённый model_prognostic_package.pkl.
    """
    st.header(t("prediction_page.new_compound_header"))

    st.markdown(t("prediction_page.standalone_description"))

    scenario = st.radio(
        t("prediction_page.scenario_label"),
        ["known_model", "capability_search"],
        format_func=lambda value: {
            "known_model": t("prediction_page.scenario_known_model"),
            "capability_search": t("prediction_page.scenario_capability_search"),
        }[value],
        horizontal=True,
        key="prediction_page_scenario",
    )

    if scenario == "capability_search":
        qspr_show_prediction_capability_tool()
        return

    qspr_show_prediction_model_source_controls()

    if "prog_model" not in st.session_state:
        qspr_prog_try_autoload_default_model()

    if "prog_model" not in st.session_state:
        st.info(t("prediction_page.load_or_train_first"))
        return

    target_col = st.session_state.get("prog_target_col", "property")
    desc_names_current = list(st.session_state.get("prog_desc_names", []))

    fallback_X_train_scaled = st.session_state.get("prog_X_train_scaled", None)

    qspr_show_new_compound_prediction_section(
        target_col=target_col,
        desc_names_current=desc_names_current,
        fallback_X_train_scaled=fallback_X_train_scaled,
        show_uncertainty_controls=True,
    )
