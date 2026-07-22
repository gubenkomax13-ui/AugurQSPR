# -*- coding: utf-8 -*-

"""Advanced validation UI for transferability and uncertainty diagnostics."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.i18n import t
from modules.validation_extensions_core import (
    group_holdout_validation,
    interpret_learning_curve_table,
    learning_curve_validation,
    prediction_interval_holdout_coverage,
    repeated_kfold_validation,
    scaffold_holdout_validation,
)
from modules.module_explain_ui import render_module_explanation


TOOL_EXPLANATION_KEYS = {
    "repeated_kfold": "advanced_validation.repeated_kfold_explanation",
    "group_scaffold": "advanced_validation.group_scaffold_explanation",
    "learning_curves": "advanced_validation.learning_curves_explanation",
    "interval_coverage": "advanced_validation.interval_coverage_explanation",
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


def _render_tool_badge():
    """Use the same visual marker as validation tools in the main module."""
    st.markdown(
        f'<div class="tool-badge">{t("repeated_holdout.tool_badge")}</div>',
        unsafe_allow_html=True,
    )


def _render_metric_triplet(metrics, prefix=""):
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{prefix}R2", _format_metric(metrics.get("R2")))
    col2.metric(f"{prefix}RMSE", _format_metric(metrics.get("RMSE")))
    col3.metric(f"{prefix}MAE", _format_metric(metrics.get("MAE")))


def _render_learning_curve_interpretation(result):
    interpretation = result.get("interpretation", {}) or {}
    if not interpretation:
        table = result.get("table")
        try:
            interpretation = interpret_learning_curve_table(table, y=y_all_current)
            result["interpretation"] = interpretation
            result["diagnosis"] = interpretation["primary_diagnosis"]
            result["rmse_gap"] = interpretation["rmse_gap"]
        except (TypeError, ValueError):
            interpretation = {}
    signals = interpretation.get("signals", {}) or {}
    if not interpretation:
        return

    values = {
        "train": _format_metric(interpretation.get("train_rmse")),
        "cv": _format_metric(interpretation.get("cv_rmse")),
        "gap": _format_metric(interpretation.get("rmse_gap")),
        "gap_percent": _format_metric(100.0 * interpretation.get("gap_fraction", 0.0), digits=1),
    }
    diagnosis = interpretation.get("primary_diagnosis")
    diagnosis_key = {
        "possible_overfitting": "overfitting",
        "possible_underfitting": "underfitting",
        "more_data_may_help": "more_data",
        "curve_plateau": "plateau",
    }.get(diagnosis, "plateau")
    message = t(f"advanced_validation.lc_diagnosis_{diagnosis_key}", **values)
    if diagnosis in {"possible_overfitting", "possible_underfitting"}:
        st.warning(message)
    elif diagnosis == "more_data_may_help":
        st.info(message)
    else:
        st.success(message)

    improvement = interpretation.get("improvement_fraction")
    if signals.get("more_data_may_help") and pd.notna(improvement):
        st.markdown(t(
            "advanced_validation.lc_more_data_signal",
            improvement=_format_metric(100.0 * improvement, digits=1),
        ))
    elif signals.get("curve_plateau"):
        st.markdown(t("advanced_validation.lc_plateau_signal"))

    cv_variation = interpretation.get("cv_variation")
    if signals.get("unstable_cv") and pd.notna(cv_variation):
        st.markdown(t(
            "advanced_validation.lc_unstable_signal",
            variation=_format_metric(100.0 * cv_variation, digits=1),
        ))
    else:
        st.markdown(t("advanced_validation.lc_stable_signal"))

    st.caption(t(f"advanced_validation.lc_recommendation_{diagnosis_key}"))


def render_advanced_validation_section(context):
    """Render advanced validation tools with plain-language explanations."""
    globals().update(context)

    st.header(t("advanced_validation.header"))
    render_module_explanation("advanced_validation")

    model_name = _advanced_validation_model_name()
    if not model_name or model_name not in st.session_state.get("trained_models", {}):
        st.info(t("advanced_validation.train_model_first"))
        return

    params = get_model_params_from_session()
    smiles_values = _advanced_validation_smiles()

    _render_tool_badge()
    with st.expander(t("advanced_validation.repeated_kfold_title"), expanded=False):
        st.markdown(t(TOOL_EXPLANATION_KEYS["repeated_kfold"]))
        col1, col2, col3 = st.columns(3)
        with col1:
            rkf_k = st.slider(
                t("advanced_validation.folds"),
                min_value=2,
                max_value=max(2, min(10, len(y_all_current))),
                value=min(5, max(2, min(10, len(y_all_current)))),
                key="advanced_rkf_k",
            )
        with col2:
            rkf_repeats = st.number_input(
                t("advanced_validation.repeats"),
                min_value=2,
                max_value=100,
                value=5,
                step=1,
                key="advanced_rkf_repeats",
            )
        with col3:
            rkf_seed = st.number_input(
                t("advanced_validation.random_state"),
                value=42,
                step=1,
                key="advanced_rkf_seed",
            )

        if st.button(t("advanced_validation.run_repeated_kfold"), type="primary", key="run_advanced_rkf"):
            try:
                progress = st.progress(0, text=t("advanced_validation.preparing_repeated_kfold"))

                def _progress(done, total):
                    progress.progress(
                        int(done / total * 100),
                        text=t("advanced_validation.repeated_kfold_progress", done=done, total=total),
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
                progress.progress(100, text=t("advanced_validation.repeated_kfold_complete"))
                st.success(t("advanced_validation.repeated_kfold_complete"))
            except Exception as exc:
                st.error(t("advanced_validation.repeated_kfold_error", error=exc))

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
                ax_rkf.set_xlabel(t("advanced_validation.r2_per_fold"))
                ax_rkf.set_ylabel(t("advanced_validation.count"))
                ax_rkf.set_title(t("advanced_validation.repeated_kfold_distribution"))
                ax_rkf.grid(True, alpha=0.3)
                fig_rkf.tight_layout()
                st.pyplot(fig_rkf)
                plt.close(fig_rkf)

    _render_tool_badge()
    with st.expander(t("advanced_validation.group_scaffold_title"), expanded=False):
        st.markdown(t(TOOL_EXPLANATION_KEYS["group_scaffold"]))
        group_mode = st.radio(
            t("advanced_validation.group_source"),
            ["scaffold", "dataset_column"],
            horizontal=True,
            key="advanced_group_mode",
            format_func=lambda value: {
                "scaffold": t("advanced_validation.group_mode_scaffold"),
                "dataset_column": t("advanced_validation.group_mode_dataset_column"),
            }.get(value, value),
        )
        group_column = None
        if group_mode == "dataset_column":
            candidate_columns = list(getattr(data, "columns", []))
            group_column = st.selectbox(
                t("advanced_validation.group_column"),
                candidate_columns,
                key="advanced_group_column",
            )

        col1, col2 = st.columns(2)
        with col1:
            group_test_percent = st.slider(
                t("advanced_validation.test_groups_percent"),
                min_value=5,
                max_value=80,
                value=20,
                step=5,
                key="advanced_group_test_percent",
            )
        with col2:
            group_seed = st.number_input(
                t("advanced_validation.random_state"),
                value=42,
                step=1,
                key="advanced_group_seed",
            )

        if st.button(t("advanced_validation.run_group_scaffold"), type="primary", key="run_advanced_group"):
            try:
                if group_mode == "dataset_column":
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
                st.success(t("advanced_validation.group_scaffold_complete"))
            except Exception as exc:
                st.error(t("advanced_validation.group_scaffold_error", error=exc))

        group_result = _advanced_validation_get("group_split_results_dict", model_name)
        if isinstance(group_result, dict):
            _render_metric_triplet(group_result.get("metrics_test", {}), prefix="Test ")
            st.caption(
                t(
                    "advanced_validation.train_test_groups_caption",
                    train=len(group_result.get("train_groups", [])),
                    test=len(group_result.get("test_groups", [])),
                )
            )
            st.dataframe(
                group_result.get("test_table", pd.DataFrame()),
                width="stretch",
                hide_index=True,
            )

    _render_tool_badge()
    with st.expander(t("advanced_validation.learning_curves_title"), expanded=False):
        st.markdown(t(TOOL_EXPLANATION_KEYS["learning_curves"]))
        lc_k = st.slider(
            t("advanced_validation.cv_folds"),
            min_value=2,
            max_value=max(2, min(10, len(y_all_current))),
            value=min(5, max(2, min(10, len(y_all_current)))),
            key="advanced_lc_k",
        )
        if st.button(t("advanced_validation.run_learning_curves"), type="primary", key="run_advanced_lc"):
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
                st.success(t("advanced_validation.learning_curves_complete"))
            except Exception as exc:
                st.error(t("advanced_validation.learning_curve_error", error=exc))

        lc_result = _advanced_validation_get("learning_curve_results_dict", model_name)
        if isinstance(lc_result, dict):
            lc_table = lc_result.get("table", pd.DataFrame())
            if isinstance(lc_table, pd.DataFrame) and not lc_table.empty:
                fig_lc, ax_lc = plt.subplots(figsize=(7, 4))
                ax_lc.plot(
                    lc_table["train_size"],
                    lc_table["train_rmse_mean"],
                    marker="o",
                    label=t("advanced_validation.train_rmse"),
                )
                ax_lc.plot(
                    lc_table["train_size"],
                    lc_table["cv_rmse_mean"],
                    marker="o",
                    label=t("advanced_validation.cv_rmse"),
                )
                ax_lc.set_xlabel(t("advanced_validation.training_set_size"))
                ax_lc.set_ylabel("RMSE")
                diagnosis_key = {
                    "possible_overfitting": "overfitting",
                    "possible_underfitting": "underfitting",
                    "more_data_may_help": "more_data",
                    "curve_plateau": "plateau",
                }.get(lc_result.get("diagnosis"), "plateau")
                ax_lc.set_title(t(
                    "advanced_validation.learning_curve_title",
                    diagnosis=t(f"advanced_validation.lc_label_{diagnosis_key}"),
                ))
                ax_lc.legend()
                ax_lc.grid(True, alpha=0.3)
                fig_lc.tight_layout()
                st.pyplot(fig_lc)
                plt.close(fig_lc)
                _render_learning_curve_interpretation(lc_result)
                st.dataframe(lc_table, width="stretch", hide_index=True)

    _render_tool_badge()
    with st.expander(t("advanced_validation.interval_coverage_title"), expanded=False):
        st.markdown(t(TOOL_EXPLANATION_KEYS["interval_coverage"]))
        col1, col2, col3 = st.columns(3)
        with col1:
            pi_confidence = st.slider(
                t("advanced_validation.nominal_coverage"),
                min_value=0.50,
                max_value=0.99,
                value=0.90,
                step=0.01,
                key="advanced_pi_confidence",
            )
        with col2:
            pi_test_percent = st.slider(
                t("advanced_validation.holdout_test_percent"),
                min_value=5,
                max_value=80,
                value=20,
                step=5,
                key="advanced_pi_test_percent",
            )
        with col3:
            pi_cv = st.slider(
                t("advanced_validation.calibration_cv_folds"),
                min_value=2,
                max_value=max(2, min(10, len(y_all_current))),
                value=min(5, max(2, min(10, len(y_all_current)))),
                key="advanced_pi_cv",
            )

        if st.button(t("advanced_validation.audit_interval_coverage"), type="primary", key="run_advanced_pi"):
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
                st.success(t("advanced_validation.interval_coverage_complete"))
            except Exception as exc:
                st.error(t("advanced_validation.interval_coverage_error", error=exc))

        pi_result = _advanced_validation_get("interval_coverage_results_dict", model_name)
        if isinstance(pi_result, dict):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(t("advanced_validation.nominal_coverage"), _format_metric(pi_result.get("confidence"), 2))
            col_b.metric(t("advanced_validation.observed_coverage"), _format_metric(pi_result.get("coverage"), 2))
            col_c.metric(t("advanced_validation.interval_radius"), _format_metric(pi_result.get("radius")))
            st.dataframe(
                pi_result.get("table", pd.DataFrame()),
                width="stretch",
                hide_index=True,
            )
