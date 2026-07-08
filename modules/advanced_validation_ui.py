# -*- coding: utf-8 -*-

"""Advanced validation UI for transferability and uncertainty diagnostics."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.validation_extensions_core import (
    group_holdout_validation,
    learning_curve_validation,
    prediction_interval_holdout_coverage,
    repeated_kfold_validation,
    scaffold_holdout_validation,
)
from modules.module_explain_ui import render_module_explanation


TOOL_EXPLANATIONS = {
    "repeated_kfold": (
        "**Repeated K-Fold** многократно делит данные на фолды. "
        "Это показывает, насколько метрики устойчивы к случайному разбиению."
    ),
    "group_scaffold": (
        "**Group / Scaffold split** держит целые группы или химические каркасы "
        "только в test. Это проверка переноса между семействами, а не запоминания "
        "близких аналогов."
    ),
    "learning_curves": (
        "**Learning curves** сравнивают train и CV ошибку при разном размере "
        "обучающей выборки. Они помогают увидеть переобучение и дефицит данных."
    ),
    "interval_coverage": (
        "**Prediction interval coverage** проверяет, честны ли интервалы "
        "неопределённости: если заявлено 90%, близко ли фактическое покрытие "
        "на hold-out к 90%."
    ),
}


def _advanced_validation_model_name():
    return st.session_state.get("last_model_algorithm")


def _advanced_validation_smiles():
    return data[smiles_col_current].iloc[valid_indices_current].values.tolist()


def _advanced_validation_store(key, model_name, value):
    if key not in st.session_state:
        st.session_state[key] = {}
    st.session_state[key][model_name] = value


def _advanced_validation_get(key, model_name):
    return st.session_state.get(key, {}).get(model_name)


def _format_metric(value, digits=3):
    try:
        if pd.notna(value):
            return f"{float(value):.{digits}f}"
    except Exception:
        pass
    return "-"


def _render_metric_triplet(metrics, prefix=""):
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{prefix}R2", _format_metric(metrics.get("R2")))
    col2.metric(f"{prefix}RMSE", _format_metric(metrics.get("RMSE")))
    col3.metric(f"{prefix}MAE", _format_metric(metrics.get("MAE")))


def render_advanced_validation_section(context):
    """Render advanced validation tools with plain-language explanations."""
    globals().update(context)

    st.header("Расширенная валидация")
    render_module_explanation("advanced_validation")

    model_name = _advanced_validation_model_name()
    if not model_name or model_name not in st.session_state.get("trained_models", {}):
        st.info("Train an analytical model first.")
        return

    params = get_model_params_from_session()
    smiles_values = _advanced_validation_smiles()

    with st.expander("Repeated K-Fold", expanded=False):
        st.markdown(TOOL_EXPLANATIONS["repeated_kfold"])
        col1, col2, col3 = st.columns(3)
        with col1:
            rkf_k = st.slider(
                "Folds",
                min_value=2,
                max_value=max(2, min(10, len(y_all_current))),
                value=min(5, max(2, min(10, len(y_all_current)))),
                key="advanced_rkf_k",
            )
        with col2:
            rkf_repeats = st.number_input(
                "Repeats",
                min_value=2,
                max_value=100,
                value=5,
                step=1,
                key="advanced_rkf_repeats",
            )
        with col3:
            rkf_seed = st.number_input(
                "random_state",
                value=42,
                step=1,
                key="advanced_rkf_seed",
            )

        if st.button("Run Repeated K-Fold", type="primary", key="run_advanced_rkf"):
            try:
                progress = st.progress(0, text="Preparing Repeated K-Fold...")

                def _progress(done, total):
                    progress.progress(
                        int(done / total * 100),
                        text=f"Repeated K-Fold: {done}/{total}",
                    )

                result = repeated_kfold_validation(
                    X=X_all_current,
                    y=y_all_current,
                    model_name=model_name,
                    valid_indices=valid_indices_current,
                    smiles=smiles_values,
                    k=int(rkf_k),
                    n_repeats=int(rkf_repeats),
                    params=params,
                    scale=True,
                    random_state=int(rkf_seed),
                    progress_callback=_progress,
                )
                _advanced_validation_store(
                    "repeated_kfold_results_dict",
                    model_name,
                    result,
                )
                progress.progress(100, text="Repeated K-Fold complete.")
                st.success("Repeated K-Fold complete.")
            except Exception as exc:
                st.error(f"Repeated K-Fold error: {exc}")

        rkf_result = _advanced_validation_get("repeated_kfold_results_dict", model_name)
        if isinstance(rkf_result, dict):
            _render_metric_triplet(rkf_result.get("metrics", {}), prefix="Mean CV ")
            st.dataframe(
                rkf_result.get("summary_table", pd.DataFrame()),
                width="stretch",
                hide_index=True,
            )
            split_table = rkf_result.get("split_table", pd.DataFrame())
            if isinstance(split_table, pd.DataFrame) and not split_table.empty:
                fig_rkf, ax_rkf = plt.subplots(figsize=(7, 4))
                ax_rkf.hist(
                    pd.to_numeric(split_table["R2"], errors="coerce").dropna(),
                    bins=20,
                    alpha=0.75,
                )
                ax_rkf.set_xlabel("R2 per fold")
                ax_rkf.set_ylabel("Count")
                ax_rkf.set_title("Repeated K-Fold R2 distribution")
                ax_rkf.grid(True, alpha=0.3)
                fig_rkf.tight_layout()
                st.pyplot(fig_rkf)
                plt.close(fig_rkf)

    with st.expander("Group / Scaffold split", expanded=False):
        st.markdown(TOOL_EXPLANATIONS["group_scaffold"])
        group_mode = st.radio(
            "Group source",
            ["Bemis-Murcko scaffold", "Dataset column"],
            horizontal=True,
            key="advanced_group_mode",
        )
        group_column = None
        if group_mode == "Dataset column":
            candidate_columns = list(getattr(data, "columns", []))
            group_column = st.selectbox(
                "Group column",
                candidate_columns,
                key="advanced_group_column",
            )

        col1, col2 = st.columns(2)
        with col1:
            group_test_percent = st.slider(
                "Test groups, %",
                min_value=5,
                max_value=80,
                value=20,
                step=5,
                key="advanced_group_test_percent",
            )
        with col2:
            group_seed = st.number_input(
                "random_state",
                value=42,
                step=1,
                key="advanced_group_seed",
            )

        if st.button("Run Group / Scaffold split", type="primary", key="run_advanced_group"):
            try:
                if group_mode == "Dataset column":
                    groups = (
                        data[group_column]
                        .iloc[valid_indices_current]
                        .fillna("missing")
                        .astype(str)
                        .values
                    )
                    result = group_holdout_validation(
                        X=X_all_current,
                        y=y_all_current,
                        model_name=model_name,
                        groups=groups,
                        valid_indices=valid_indices_current,
                        smiles=smiles_values,
                        test_size=float(group_test_percent) / 100.0,
                        random_state=int(group_seed),
                        params=params,
                        scale=True,
                        group_label=str(group_column),
                    )
                else:
                    result = scaffold_holdout_validation(
                        X=X_all_current,
                        y=y_all_current,
                        model_name=model_name,
                        smiles=smiles_values,
                        valid_indices=valid_indices_current,
                        test_size=float(group_test_percent) / 100.0,
                        random_state=int(group_seed),
                        params=params,
                        scale=True,
                    )
                _advanced_validation_store(
                    "group_split_results_dict",
                    model_name,
                    result,
                )
                st.success("Group / Scaffold split complete.")
            except Exception as exc:
                st.error(f"Group / Scaffold split error: {exc}")

        group_result = _advanced_validation_get("group_split_results_dict", model_name)
        if isinstance(group_result, dict):
            _render_metric_triplet(group_result.get("metrics_test", {}), prefix="Test ")
            st.caption(
                f"Train groups: {len(group_result.get('train_groups', []))}; "
                f"test groups: {len(group_result.get('test_groups', []))}"
            )
            st.dataframe(
                group_result.get("test_table", pd.DataFrame()),
                width="stretch",
                hide_index=True,
            )

    with st.expander("Learning curves", expanded=False):
        st.markdown(TOOL_EXPLANATIONS["learning_curves"])
        lc_k = st.slider(
            "CV folds",
            min_value=2,
            max_value=max(2, min(10, len(y_all_current))),
            value=min(5, max(2, min(10, len(y_all_current)))),
            key="advanced_lc_k",
        )
        if st.button("Run learning curves", type="primary", key="run_advanced_lc"):
            try:
                result = learning_curve_validation(
                    X=X_all_current,
                    y=y_all_current,
                    model_name=model_name,
                    params=params,
                    scale=True,
                    k=int(lc_k),
                    random_state=42,
                )
                _advanced_validation_store(
                    "learning_curve_results_dict",
                    model_name,
                    result,
                )
                st.success("Learning curves complete.")
            except Exception as exc:
                st.error(f"Learning curve error: {exc}")

        lc_result = _advanced_validation_get("learning_curve_results_dict", model_name)
        if isinstance(lc_result, dict):
            lc_table = lc_result.get("table", pd.DataFrame())
            if isinstance(lc_table, pd.DataFrame) and not lc_table.empty:
                fig_lc, ax_lc = plt.subplots(figsize=(7, 4))
                ax_lc.plot(
                    lc_table["train_size"],
                    lc_table["train_rmse_mean"],
                    marker="o",
                    label="Train RMSE",
                )
                ax_lc.plot(
                    lc_table["train_size"],
                    lc_table["cv_rmse_mean"],
                    marker="o",
                    label="CV RMSE",
                )
                ax_lc.set_xlabel("Training set size")
                ax_lc.set_ylabel("RMSE")
                ax_lc.set_title(f"Learning curve: {lc_result.get('diagnosis')}")
                ax_lc.legend()
                ax_lc.grid(True, alpha=0.3)
                fig_lc.tight_layout()
                st.pyplot(fig_lc)
                plt.close(fig_lc)
                st.dataframe(lc_table, width="stretch", hide_index=True)

    with st.expander("Prediction interval hold-out coverage", expanded=False):
        st.markdown(TOOL_EXPLANATIONS["interval_coverage"])
        col1, col2, col3 = st.columns(3)
        with col1:
            pi_confidence = st.slider(
                "Nominal coverage",
                min_value=0.50,
                max_value=0.99,
                value=0.90,
                step=0.01,
                key="advanced_pi_confidence",
            )
        with col2:
            pi_test_percent = st.slider(
                "Hold-out test, %",
                min_value=5,
                max_value=80,
                value=20,
                step=5,
                key="advanced_pi_test_percent",
            )
        with col3:
            pi_cv = st.slider(
                "Calibration CV folds",
                min_value=2,
                max_value=max(2, min(10, len(y_all_current))),
                value=min(5, max(2, min(10, len(y_all_current)))),
                key="advanced_pi_cv",
            )

        if st.button("Audit interval coverage", type="primary", key="run_advanced_pi"):
            try:
                result = prediction_interval_holdout_coverage(
                    X=X_all_current,
                    y=y_all_current,
                    model_name=model_name,
                    valid_indices=valid_indices_current,
                    smiles=smiles_values,
                    test_size=float(pi_test_percent) / 100.0,
                    confidence=float(pi_confidence),
                    calibration_cv=int(pi_cv),
                    random_state=42,
                    params=params,
                    scale=True,
                )
                _advanced_validation_store(
                    "interval_coverage_results_dict",
                    model_name,
                    result,
                )
                st.success("Prediction interval coverage audit complete.")
            except Exception as exc:
                st.error(f"Prediction interval coverage error: {exc}")

        pi_result = _advanced_validation_get("interval_coverage_results_dict", model_name)
        if isinstance(pi_result, dict):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Nominal coverage", _format_metric(pi_result.get("confidence"), 2))
            col_b.metric("Observed coverage", _format_metric(pi_result.get("coverage"), 2))
            col_c.metric("Interval radius", _format_metric(pi_result.get("radius")))
            st.dataframe(
                pi_result.get("table", pd.DataFrame()),
                width="stretch",
                hide_index=True,
            )
