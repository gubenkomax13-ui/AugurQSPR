# -*- coding: utf-8 -*-
"""Интерфейс сохранения проверенной QSPR-модели."""

from datetime import datetime
import os

import joblib
import numpy as np
import streamlit as st

from modules.i18n import t
from modules.module_explain_ui import render_module_explanation
from modules.runtime_mode import qspr_is_online_mode


ONLINE_LOCK_MESSAGE = (
    "Эта функция показана как возможность полной локальной версии Augur QSPR, "
    "но в публичном онлайн-режиме она отключена для безопасности и стабильности."
)


def qspr_save_is_online_mode():
    return qspr_is_online_mode()

try:
    from modules.prognostic_model_core import (
        qspr_prog_build_ad_profile,
        qspr_prog_build_descriptor_groups,
        qspr_prog_descriptor_schema,
    )
except Exception:
    def qspr_prog_build_ad_profile(
        X_train_raw=None,
        X_train_scaled=None,
        train_smiles=None,
        desc_names=None,
    ):
        return {}

    def qspr_prog_build_descriptor_groups(desc_names):
        spectral_prefixes = (
            "IR_", "FTIR_", "SPEC_IR_", "WN_", "MS_", "MASS_",
            "MZ_", "FRAG_", "SPEC_MS_", "SPEC_MASS_", "SPEC_",
        )
        groups = {"molecular": [], "spectral_ir": [], "spectral_mass": [], "spectral": []}
        for name in list(desc_names or []):
            upper = str(name).upper()
            if upper.startswith(("IR_", "FTIR_", "SPEC_IR_", "WN_")):
                groups["spectral_ir"].append(str(name))
            elif upper.startswith(("MS_", "MASS_", "MZ_", "FRAG_", "SPEC_MS_", "SPEC_MASS_")):
                groups["spectral_mass"].append(str(name))
            elif upper.startswith(spectral_prefixes) or "SPECTR" in upper:
                groups["spectral"].append(str(name))
            else:
                groups["molecular"].append(str(name))
        groups["spectral_all"] = groups["spectral_ir"] + groups["spectral_mass"] + groups["spectral"]
        groups["combined"] = [str(name) for name in list(desc_names or [])]
        return groups

    def qspr_prog_descriptor_schema(desc_names, descriptor_source="", descriptor_mode="", desc_lists=None):
        return {
            "schema_version": "1.0",
            "descriptor_names": list(desc_names or []),
            "descriptor_count": len(list(desc_names or [])),
            "descriptor_source": str(descriptor_source or ""),
            "descriptor_mode": str(descriptor_mode or ""),
        }


def render_verified_model_save(
    *,
    model_name,
    model_data,
    target_col,
    smiles_col,
    descriptor_names,
    X_train,
    y_train,
    train_smiles,
    validation_completed,
    add_log,
    output_path="model_analysis_package.pkl",
):
    """Показывает сохранение модели только после независимой валидации."""
    st.header(t("save_model.verified_header"))
    render_module_explanation("save_model")

    if not validation_completed:
        st.info(t("save_model.validation_required"))
        return

    st.caption(t("save_model.verified_caption"))
    if qspr_save_is_online_mode():
        st.info(ONLINE_LOCK_MESSAGE)

    if not st.button(
        t("save_model.button"),
        key="save_analysis",
        disabled=qspr_save_is_online_mode(),
    ):
        return

    package_desc_names = list(model_data.get("selected_desc_names", descriptor_names))
    descriptor_source = st.session_state.get("custom_descriptor_source", "")
    descriptor_mode = st.session_state.get(
        "molecular_descriptor_calculation_mode",
        st.session_state.get("descriptor_calculation_mode", "mordred"),
    )
    desc_lists = st.session_state.get("desc_lists")

    model_package = {
        "model": model_data["model"],
        "scaler": model_data.get("scaler"),
        "target_col": target_col,
        "smiles_col": smiles_col,
        "desc_names": package_desc_names,
        "all_desc_names": list(descriptor_names),
        "descriptor_source": descriptor_source,
        "descriptor_groups": qspr_prog_build_descriptor_groups(
            package_desc_names
        ),
        "descriptor_schema": qspr_prog_descriptor_schema(
            package_desc_names,
            descriptor_source=descriptor_source,
            descriptor_mode=descriptor_mode,
            desc_lists=desc_lists,
        ),
        "model_name": model_name,
        "metrics": model_data.get("metrics", {}),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "X_train": np.asarray(X_train, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "train_smiles": list(train_smiles),
        "ad_profile": qspr_prog_build_ad_profile(
            X_train_raw=np.asarray(X_train, dtype=float),
            X_train_scaled=np.asarray(X_train, dtype=float),
            train_smiles=list(train_smiles),
            desc_names=package_desc_names,
        ),
        "descriptor_mode": descriptor_mode,
        "desc_lists": desc_lists,
        "validation_completed": True,
    }

    joblib.dump(model_package, output_path)
    st.success(t("save_model.success"))
    add_log(t("save_model.log"))
