# -*- coding: utf-8 -*-
"""Mechanistic descriptor interpretation helpers for QSPR diagnostics."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


BLACK_BOX_KEYWORDS = (
    "random forest",
    "extra trees",
    "gradient boosting",
    "histogram gradient",
    "adaboost",
    "mlp",
    "support vector",
    "svr",
    "knn",
    "gaussian process",
    "voting",
    "stacking",
    "genetic",
    "augur evolutionary",
)

LINEAR_KEYWORDS = (
    "linear regression",
    "множественная линейная",
    "mlr",
    "ridge",
    "lasso",
    "elastic net",
)


def mechanism_model_interpretation_mode(model_name):
    name = str(model_name or "").lower()
    if any(keyword in name for keyword in BLACK_BOX_KEYWORDS):
        return (
            "black_box",
            "Интерпретация ограничена: модель нелинейная/black-box, поэтому используется локальное или surrogate-объяснение по importance/SHAP.",
        )
    if any(keyword in name for keyword in LINEAR_KEYWORDS):
        return (
            "linear",
            "Для линейной модели направление эффекта можно интерпретировать по знаку коэффициента при условии стабильной валидации и отсутствия сильной коллинеарности.",
        )
    if "pls" in name:
        return (
            "latent_linear",
            "PLS использует латентные компоненты: знак коэффициента полезен, но его нужно читать вместе с VIP/loadings и масштабированием.",
        )
    return (
        "generic",
        "Интерпретация является статистической гипотезой и должна подтверждаться химической экспертизой и внешней проверкой.",
    )


def mechanism_descriptor_source(descriptor):
    name = str(descriptor or "")
    upper = name.upper()
    if upper.startswith(("IR_", "FTIR_", "SPEC_IR_", "WN_")):
        return "spectral_ir"
    if upper.startswith(("MS_", "MASS_", "MZ_", "FRAG_", "SPEC_MS_")):
        return "spectral_mass"
    if upper.startswith("BCUT2D") or name in {"MolWt", "ExactMolWt", "TPSA", "MolLogP"}:
        return "RDKit"
    if re.match(r"^[A-Za-z]+[0-9]+[a-zA-Z]*$", name) or name.startswith(("n", "ATS", "AATS", "GATS", "MATS")):
        return "Mordred/PaDEL family"
    if "VSA" in upper or upper.startswith(("PEOE_", "SMR_", "SLOGP_", "E_STATE")):
        return "RDKit VSA/E-state"
    return "descriptor_set"


def mechanism_descriptor_formula(descriptor):
    name = str(descriptor or "")
    formula_map = {
        "MolWt": "Σ atomic weights",
        "ExactMolWt": "Σ exact isotope masses",
        "TPSA": "topological polar surface area",
        "MolLogP": "Crippen logP estimate",
        "HeavyAtomCount": "count(non-H atoms)",
        "NumHAcceptors": "count(H-bond acceptors)",
        "NumHDonors": "count(H-bond donors)",
        "NumRotatableBonds": "count(rotatable bonds)",
        "FractionCSP3": "sp3 carbons / total carbons",
        "RingCount": "count(rings)",
    }
    if name in formula_map:
        return formula_map[name]
    if "VSA" in name.upper():
        return "Σ van der Waals surface area in descriptor bin"
    if name.upper().startswith("BCUT2D"):
        return "eigenvalue of weighted molecular graph matrix"
    if name.startswith("Chi"):
        return "Kier-Hall molecular connectivity index"
    if name.startswith("Kappa"):
        return "Kier molecular shape index"
    if name.upper().startswith(("IR_", "FTIR_", "WN_")):
        return "spectral intensity/bin feature"
    if name.upper().startswith(("MS_", "MZ_", "MASS_")):
        return "mass-spectral peak/bin feature"
    return "см. источник дескриптора"


def mechanism_property_axis(descriptor, meaning=""):
    text = f"{descriptor} {meaning}".lower()
    if any(token in text for token in ("logp", "hydrophob", "липоф", "slogp")):
        return "липофильность / гидрофобность"
    if any(token in text for token in ("charge", "заряд", "peoe", "electro")):
        return "электронное распределение / заряд"
    if any(token in text for token in ("surface", "vsa", "asa", "tpsa", "площад")):
        return "площадь поверхности / полярная доступность"
    if any(token in text for token in ("mass", "molwt", "weight", "масса")):
        return "размер и масса молекулы"
    if any(token in text for token in ("ring", "aromatic", "цик", "kappa", "shape", "форма")):
        return "форма, цикличность и топология"
    if any(token in text for token in ("hbond", "acceptor", "donor", "донор", "акцептор")):
        return "водородное связывание / полярные центры"
    if any(token in text for token in ("spect", "ir_", "ftir", "wn_", "mass-spectral", "ms_")):
        return "спектральная характеристика структуры"
    if any(token in text for token in ("connectivity", "chi", "path", "тополог")):
        return "топологическая связность"
    return "обобщённое структурное свойство"


def mechanism_endpoint_link(property_axis, target_col, effect_direction):
    direction_text = "увеличением" if effect_direction == "positive" else "уменьшением"
    if effect_direction == "mixed":
        direction_text = "изменением"
    return (
        f"Дескриптор отражает {property_axis}; в наблюдаемой области данных его рост связан с {direction_text} "
        f"модельного прогноза `{target_col}`."
    )


def mechanism_reference_note(source):
    return {
        "RDKit": "RDKit descriptor; проверьте определение в документации RDKit Descriptors.",
        "RDKit VSA/E-state": "RDKit VSA/E-state descriptor; интерпретируйте как площадь поверхности в заданном физико-химическом интервале.",
        "Mordred/PaDEL family": "Mordred/PaDEL descriptor family; требуется сверка с оригинальным определением конкретного дескриптора.",
        "spectral_ir": "Спектральный IR-дескриптор; механизм связан с полосой/бином спектра и требует спектральной атрибуции.",
        "spectral_mass": "Mass-spectral descriptor; механизм связан с фрагментацией/пиком и требует атрибуции фрагмента.",
    }.get(source, "Источник не детализирован; добавьте авторский комментарий для публикационного отчёта.")


def mechanism_effect_direction(row):
    for column in ("coefficient_signed", "shap_signed", "permutation_signed"):
        if column in row and pd.notna(row[column]):
            value = float(row[column])
            if value > 0:
                return "positive"
            if value < 0:
                return "negative"
    return "mixed"


def mechanism_effect_label(direction):
    return {
        "positive": "рост дескриптора повышает прогноз в наблюдаемой области",
        "negative": "рост дескриптора снижает прогноз в наблюдаемой области",
        "mixed": "направление эффекта неоднозначно или локально",
    }.get(direction, "направление не оценено")


def mechanism_observed_range(X, desc_names, descriptor):
    try:
        names = [str(name) for name in list(desc_names or [])]
        index = names.index(str(descriptor))
        values = np.asarray(X, dtype=float)[:, index]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return ""
        return (
            f"{np.nanmin(values):.4g} .. {np.nanmax(values):.4g}; "
            f"median {np.nanmedian(values):.4g}"
        )
    except Exception:
        return ""


def build_mechanistic_interpretation_table(
    unified_importance,
    descriptor_meanings,
    X,
    desc_names,
    target_col,
    model_name,
    top_n=20,
):
    if not isinstance(unified_importance, pd.DataFrame) or unified_importance.empty:
        return pd.DataFrame(), mechanism_model_interpretation_mode(model_name)[1]

    mode, mode_note = mechanism_model_interpretation_mode(model_name)
    table = unified_importance.copy()
    if "combined_rank" in table.columns:
        table = table.sort_values("combined_rank", ascending=True)
    table = table.head(int(top_n)).copy()

    rows = []
    meanings = descriptor_meanings if isinstance(descriptor_meanings, dict) else {}
    for _, item in table.iterrows():
        descriptor = str(item.get("descriptor", ""))
        meaning = meanings.get(descriptor, "")
        direction = mechanism_effect_direction(item)
        source = mechanism_descriptor_source(descriptor)
        property_axis = mechanism_property_axis(descriptor, meaning)
        rows.append({
            "descriptor": descriptor,
            "rank": item.get("combined_rank", np.nan),
            "combined_score": item.get("combined_score", np.nan),
            "formula_or_definition": mechanism_descriptor_formula(descriptor),
            "source": source,
            "physicochemical_meaning": meaning or "Нет расшифровки в descriptor_meanings.json",
            "effect_direction": mechanism_effect_label(direction),
            "observed_domain": mechanism_observed_range(X, desc_names, descriptor),
            "descriptor_to_property": property_axis,
            "property_to_endpoint": mechanism_endpoint_link(property_axis, target_col, direction),
            "reference_or_author_note": mechanism_reference_note(source),
            "author_comment": (
                "Проверьте химическую правдоподобность связи и устойчивость дескриптора на внешней валидации."
            ),
            "interpretation_limit": mode_note,
            "model_interpretation_mode": mode,
        })
    return pd.DataFrame(rows), mode_note


def raw_scale_linear_equation(model, scaler, feature_names, target_name="y", max_terms=80):
    """Convert linear coefficients from StandardScaler space to raw descriptor units."""
    estimator = model
    pipeline_scaler = None
    pipeline_feature_names = list(feature_names or [])
    if isinstance(model, Pipeline):
        pipeline_scaler = model.named_steps.get("scale")
        preselect = model.named_steps.get("preselect")
        selected_names = list(getattr(preselect, "selected_names_", []) or [])
        if selected_names:
            pipeline_feature_names = selected_names
        estimator = model.steps[-1][1]

    if not hasattr(estimator, "coef_"):
        return None
    try:
        coef_scaled = np.ravel(np.asarray(estimator.coef_, dtype=float))
        intercept_scaled = float(np.ravel(np.asarray(getattr(estimator, "intercept_", 0.0), dtype=float))[0])
    except Exception:
        return None

    active_scaler = pipeline_scaler if pipeline_scaler is not None else scaler
    names = [str(name) for name in pipeline_feature_names]
    if len(names) != len(coef_scaled):
        return None

    if active_scaler is not None and hasattr(active_scaler, "scale_") and hasattr(active_scaler, "mean_"):
        scale = np.asarray(active_scaler.scale_, dtype=float)
        mean = np.asarray(active_scaler.mean_, dtype=float)
        if len(scale) != len(coef_scaled):
            return None
        safe_scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
        coef_raw = coef_scaled / safe_scale
        intercept_raw = intercept_scaled - float(np.sum(coef_scaled * mean / safe_scale))
        scale_note = "raw_descriptor_units_from_standard_scaler"
    else:
        coef_raw = coef_scaled
        intercept_raw = intercept_scaled
        scale_note = "model_features_no_scaler_detected"

    parts = [f"{target_name} = {intercept_raw:.6g}"]
    order = np.argsort(np.abs(coef_raw))[::-1]
    for index in order[: int(max_terms)]:
        value = float(coef_raw[index])
        sign = "+" if value >= 0 else "-"
        parts.append(f" {sign} {abs(value):.6g}·{names[index]}")
    if len(order) > int(max_terms):
        parts.append(f" ... ({len(order) - int(max_terms)} terms omitted)")

    table = pd.DataFrame({
        "descriptor": names,
        "raw_coefficient": coef_raw,
        "scaled_coefficient": coef_scaled,
    }).sort_values("raw_coefficient", key=lambda s: s.abs(), ascending=False)

    return {
        "equation": "".join(parts),
        "coefficient_table": table.reset_index(drop=True),
        "scale_note": scale_note,
    }
