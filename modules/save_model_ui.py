# -*- coding: utf-8 -*-
"""Интерфейс сохранения проверенной QSPR-модели."""

from datetime import datetime
import os

import joblib
import numpy as np
import streamlit as st

from modules.i18n import t


ONLINE_LOCK_MESSAGE = (
    "Эта функция показана как возможность полной локальной версии Augur QSPR, "
    "но в публичном онлайн-режиме она отключена для безопасности и стабильности."
)


def qspr_save_is_online_mode():
    for source in (os.environ.get("AUGUR_MODE"), os.environ.get("AUGUR_RUNTIME_MODE")):
        value = str(source or "").strip().lower()
        if value in {"online", "demo", "cloud", "public"}:
            return True
        if value in {"local", "full", "desktop"}:
            return False

    try:
        value = str(st.secrets.get("AUGUR_MODE", "") or "").strip().lower()
        if value in {"online", "demo", "cloud", "public"}:
            return True
        if value in {"local", "full", "desktop"}:
            return False
    except Exception:
        pass

    try:
        context = getattr(st, "context", None)
        headers = getattr(context, "headers", {}) if context is not None else {}
        host = str(headers.get("host") or headers.get("Host") or "").lower()
        url = str(getattr(context, "url", "") or "").lower()
    except Exception:
        host = ""
        url = ""

    return any(marker in host or marker in url for marker in ("streamlit.app", "share.streamlit.io"))

try:
    from modules.prognostic_model_core import qspr_prog_build_descriptor_groups
except Exception:
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

    model_package = {
        "model": model_data["model"],
        "scaler": model_data.get("scaler"),
        "target_col": target_col,
        "smiles_col": smiles_col,
        "desc_names": list(
            model_data.get("selected_desc_names", descriptor_names)
        ),
        "all_desc_names": list(descriptor_names),
        "descriptor_source": st.session_state.get(
            "custom_descriptor_source", ""
        ),
        "descriptor_groups": qspr_prog_build_descriptor_groups(
            model_data.get("selected_desc_names", descriptor_names)
        ),
        "model_name": model_name,
        "metrics": model_data.get("metrics", {}),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "X_train": np.asarray(X_train, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "train_smiles": list(train_smiles),
        "descriptor_mode": st.session_state.get(
            "molecular_descriptor_calculation_mode",
            st.session_state.get("descriptor_calculation_mode", "mordred"),
        ),
        "desc_lists": st.session_state.get("desc_lists"),
        "validation_completed": True,
    }

    joblib.dump(model_package, output_path)
    st.success(t("save_model.success"))
    add_log(t("save_model.log"))
