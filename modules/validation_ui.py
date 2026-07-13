# -*- coding: utf-8 -*-
"""Интерфейс валидации QSPR-моделей."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from modules.advanced_validation_ui import render_advanced_validation_section
from modules.analysis_state import analysis_result_hash, attach_result_cache_metadata, cached_result_is_current
from modules.i18n import t
from modules.module_explain_ui import render_module_explanation
from modules.qspr_core import qspr_csv_download_bytes


def _fmt_validation_metric(value, digits=3):
    try:
        value = float(value)
    except Exception:
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _fmt_mape_metric(metrics):
    if not metrics.get("MAPE_applicable", True):
        return "N/A"
    return _fmt_validation_metric(metrics.get("MAPE_percent"), 2)


def _render_metric_diagnostics(metrics, label):
    if not isinstance(metrics, dict):
        return
    r2_reliability = metrics.get("R2_reliability", "")
    if r2_reliability == "not_interpretable_n_lt_5":
        st.warning(f"{label}: R2 is not interpretable because n < 5.")
    elif r2_reliability == "high_uncertainty_n_lt_10":
        st.warning(f"{label}: R2 has high uncertainty because n < 10.")
    if not metrics.get("MAPE_applicable", True):
        warning = metrics.get("MAPE_warning") or "MAPE is not applicable for this target scale."
        st.warning(f"{label}: {warning}")


def _render_advanced_metric_table(metrics, label):
    if not isinstance(metrics, dict):
        return
    rows = [
        ("N", metrics.get("N")),
        ("NRMSE_range", metrics.get("NRMSE_range")),
        ("RMSE/SD", metrics.get("NRMSE_sd")),
        ("MAE/IQR", metrics.get("MAE_IQR")),
        ("CCC", metrics.get("CCC")),
        ("Pearson r", metrics.get("Pearson_r")),
        ("Spearman rho", metrics.get("Spearman_rho")),
        ("R2 reliability", metrics.get("R2_reliability")),
        ("MAPE applicable", metrics.get("MAPE_applicable")),
        ("MAPE warning", metrics.get("MAPE_warning")),
    ]
    with st.expander(f"Advanced statistics: {label}", expanded=False):
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric", "Value"]),
            width="stretch",
            hide_index=True,
        )


def _validation_selector_config(desc_names_current):
    if not st.session_state.get("auto_feature_selection", False):
        return None
    desc_names_current = list(desc_names_current or [])
    return {
        "desc_names": desc_names_current,
        "method": st.session_state.get("auto_feature_selection_method", "fast"),
        "max_features": min(
            int(st.session_state.get("auto_max_features", 50) or 50),
            max(1, len(desc_names_current)),
        ),
        "remove_constant": bool(st.session_state.get("auto_remove_constant_descriptors", True)),
        "remove_correlated": bool(st.session_state.get("auto_remove_correlated_descriptors", True)),
        "corr_threshold": float(st.session_state.get("auto_corr_threshold", 0.95) or 0.95),
        "lasso_alpha": float(st.session_state.get("auto_lasso_selection_alpha", 0.01) or 0.01),
        "rf_n_estimators": int(st.session_state.get("auto_rf_selection_estimators", 300) or 300),
        "rfe_step": float(st.session_state.get("auto_rfe_step", 0.2) or 0.2),
        "random_state": int(st.session_state.get("random_seed", 42)),
    }


def render_validation_section(context):
    """Рендерит полный этап валидации в переданном контексте проекта."""
    globals().update(context)
    desc_names_for_validation = (
        context.get("desc_names_current")
        or context.get("desc_names")
        or st.session_state.get("desc_names", [])
        or []
    )
    selector_config_current = _validation_selector_config(desc_names_for_validation)

    def _validation_cache_hash(model_name, settings):
        return analysis_result_hash(
            st.session_state,
            model_name,
            params=get_model_params_from_session(),
            validation_settings=settings,
            X=X_all_current,
            y=y_all_current,
            desc_names=desc_names_for_validation,
            valid_indices=valid_indices_current,
        )

    def _current_cached_validation_result(store_name, model_name, settings):
        result = st.session_state.get(store_name, {}).get(model_name)
        if not isinstance(result, dict):
            return None
        expected_hash = _validation_cache_hash(model_name, settings)
        if not cached_result_is_current(result, expected_hash):
            return None
        return result
    # ------------------------------------------------------------------
    # Validation
    
    st.header(t('validation.header'))
    if selector_config_current:
        st.caption(
            "Automatic feature selection is refit inside validation folds: "
            "imputation -> constant/correlation/feature selection -> scaling -> model."
        )
    render_module_explanation("validation")
    with st.expander("Heuristic quality indicators", expanded=False):
        st.caption(
            "These thresholds are heuristic diagnostic indicators, not universal "
            "statistical rules. Adjust them for dataset size and endpoint noise."
        )
        col_hq_1, col_hq_2 = st.columns(2)
        with col_hq_1:
            st.number_input(
                "Bootstrap unstable if P95 RMSE / median RMSE is above",
                min_value=1.0,
                max_value=10.0,
                value=float(st.session_state.get("bootstrap_rmse_p95_ratio_threshold", 2.0)),
                step=0.1,
                key="bootstrap_rmse_p95_ratio_threshold",
            )
        with col_hq_2:
            st.number_input(
                "Y-randomization risky if Q2 gap is below",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("y_randomization_q2_gap_threshold", 0.10)),
                step=0.01,
                key="y_randomization_q2_gap_threshold",
            )
    
    tab_holdout, tab_kfold, tab_loo, tab_ext = st.tabs([
        t('validation.holdout_tab'),
        t('validation.kfold_tab'),
        t('validation.loo_tab'),
        t('validation.external_tab')
    ])
    
    with tab_holdout:
        st.subheader(t('validation.holdout_settings'))
                
        test_size_percent = st.slider(
            t('validation.holdout_test_size_percent'),
            1, 99, 20,
            key="holdout_test_size"
        )
        test_size = test_size_percent / 100.0
        use_random = st.checkbox(
            t('validation.holdout_random_split'),
            value=True,
            key="holdout_random"
        )
    
        if use_random:
            random_state = st.number_input(
                t('validation.holdout_random_state'),
                value=42,
                step=1,
                key="holdout_rs"
            )
            stratify_y_quantiles = st.checkbox(
                "Stratify random split by target quantile bins",
                value=bool(st.session_state.get("holdout_stratify_y_quantiles", False)),
                key="holdout_stratify_y_quantiles",
            )
            manual_indices = None
        else:
            stratify_y_quantiles = False
            manual_str = st.text_input(
                t('validation.holdout_manual_indices'),
                key="holdout_manual"
            )
            if manual_str:
                try:
                    manual_indices = [int(x.strip()) for x in manual_str.split(",")]
                except Exception:
                    manual_indices = None
            else:
                manual_indices = None
    
        if st.button(t('validation.holdout_run_button'), type="primary", key="run_holdout"):
            try:
                smiles_current = data[smiles_col_current].iloc[valid_indices_current].values
    
                res_hold = qspr_holdout_validation(
                    X=X_all_current,
                    y=y_all_current,
                    model_name=st.session_state.last_model_algorithm,
                    valid_indices=valid_indices_current,
                    smiles=smiles_current,
                    test_size=test_size,
                    random_state=int(random_state) if use_random else 42,
                    use_random=use_random,
                    manual_indices=manual_indices,
                    params=get_model_params_from_session(),
                    scale=True,
                    selector_config=selector_config_current,
                    stratify_y_quantiles=bool(stratify_y_quantiles),
                )
    
                holdout_settings = {
                    "kind": "holdout",
                    "test_size": test_size,
                    "random_state": int(random_state) if use_random else 42,
                    "use_random": use_random,
                    "stratify_y_quantiles": bool(stratify_y_quantiles),
                    "manual_indices": manual_indices,
                }
                holdout_hash = analysis_result_hash(
                    st.session_state,
                    st.session_state.last_model_algorithm,
                    params=get_model_params_from_session(),
                    validation_settings=holdout_settings,
                    X=X_all_current,
                    y=y_all_current,
                    desc_names=desc_names_for_validation,
                    valid_indices=valid_indices_current,
                )
                res_hold["validation_settings"] = holdout_settings
                st.session_state.holdout_results_dict[
                    st.session_state.last_model_algorithm
                ] = attach_result_cache_metadata(res_hold, holdout_hash)
                st.session_state.pop(
                    f"descriptor_importance_result_{st.session_state.last_model_algorithm}",
                    None
                )
                st.session_state.pop(
                    f"error_analysis_result_{st.session_state.last_model_algorithm}",
                    None
                )
    
                combined_df = pd.concat([res_hold["train_table"], res_hold["test_table"]], ignore_index=True)
                qspr_save_results_auto(combined_df, "holdout", target_col, len(y_all_current))
                log_validation_result(
                    "holdout",
                    st.session_state.last_model_algorithm,
                    res_hold,
                    y_values=y_all_current,
                    details={
                        "split": "random" if use_random else "manual",
                        "test_size": test_size,
                        "stratify_y_quantiles": bool(stratify_y_quantiles),
                        "stratification_note": res_hold.get("stratification_note"),
                    },
                )
                st.rerun()
    
            except Exception as e:
                message = t('validation.holdout_error', error=e)
                st.error(message)
                log_streamlit_message(
                    "VALIDATION",
                    message,
                    level="error",
                    details={"error": str(e), "model": st.session_state.last_model_algorithm},
                    event="holdout_failed",
                )
    
        res = st.session_state.holdout_results_dict.get(st.session_state.last_model_algorithm)
        if isinstance(res, dict):
            expected_hash = analysis_result_hash(
                st.session_state,
                st.session_state.last_model_algorithm,
                params=get_model_params_from_session(),
                validation_settings={
                    "kind": "holdout",
                    "test_size": test_size,
                    "random_state": int(random_state) if use_random else 42,
                    "use_random": use_random,
                    "stratify_y_quantiles": bool(stratify_y_quantiles),
                    "manual_indices": manual_indices,
                },
                X=X_all_current,
                y=y_all_current,
                desc_names=desc_names_for_validation,
                valid_indices=valid_indices_current,
            )
            if not cached_result_is_current(res, expected_hash):
                st.warning(
                    "Stored hold-out result belongs to an older data/descriptor/validation configuration. Run hold-out again."
                )
                res = None
    
        if res is not None:
            y_train = np.asarray(res["y_train"], dtype=float)
            y_pred_train = np.asarray(res["y_pred_train"], dtype=float)
            y_test = np.asarray(res["y_test"], dtype=float)
            y_pred_test = np.asarray(res["y_pred_test"], dtype=float)
    
            # ------------------------------------------------------------
            # Тренировочная выборка: таблица
    
            st.subheader(t('validation.holdout_train_table_title'))
    
            st.dataframe(
                res["train_table"],
                width="stretch",
                hide_index=True
            )
    
            # ------------------------------------------------------------
            # Метрики тренировочной выборки
    
            st.subheader(t('validation.holdout_train_metrics_title'))
    
            col_train_m1, col_train_m2, col_train_m3 = st.columns(3)
    
            with col_train_m1:
                st.metric(t('validation.metric_r2_train'), _fmt_validation_metric(res['metrics_train'].get('R2')))
    
            with col_train_m2:
                st.metric(t('validation.metric_rmse_train'), _fmt_validation_metric(res['metrics_train'].get('RMSE')))
    
            with col_train_m3:
                st.metric(t('validation.metric_mae_train'), _fmt_validation_metric(res['metrics_train'].get('MAE')))

            _render_metric_diagnostics(res.get("metrics_train", {}), "Train")
            _render_advanced_metric_table(res.get("metrics_train", {}), "Train")
    
            # ------------------------------------------------------------
            # Тренировочная выборка: графики
    
            st.markdown(t('validation.holdout_train_plots_title'))
    
            col_train_plot, col_train_err = st.columns(2)
    
            with col_train_plot:
                fig_train, ax_train = plt.subplots(figsize=(5, 4))
    
                min_train = min(np.nanmin(y_train), np.nanmin(y_pred_train))
                max_train = max(np.nanmax(y_train), np.nanmax(y_pred_train))
                pad_train = (max_train - min_train) * 0.05 if max_train > min_train else 1.0
    
                ax_train.scatter(
                    y_train,
                    y_pred_train,
                    alpha=0.7,
                    s=45
                )
    
                ax_train.plot(
                    [min_train - pad_train, max_train + pad_train],
                    [min_train - pad_train, max_train + pad_train],
                    "r--",
                    lw=1.5,
                    label=t('validation.plot_ideal_label')
                )
    
                ax_train.set_xlabel(t('validation.plot_exp_label'))
                ax_train.set_ylabel(t('validation.plot_pred_label'))
                ax_train.set_title(t('validation.plot_train_title', model=st.session_state.last_model_algorithm))
                ax_train.legend(fontsize=8)
                ax_train.grid(True, alpha=0.3)
    
                fig_train.tight_layout()
                st.pyplot(fig_train)
    
            with col_train_err:
                errors_train = y_train - y_pred_train
    
                fig_train_err, ax_train_err = plt.subplots(figsize=(5, 4))
    
                safe_histplot(ax_train_err, errors_train, bins=30, kde=True, color='steelblue', edgecolor='black', alpha=0.7)
    
                if len(errors_train) > 1:
                    mu_train = np.mean(errors_train)
                    std_train = np.std(errors_train)
    
                    if std_train > 1e-12:
                        x_train_err = np.linspace(
                            np.nanmin(errors_train),
                            np.nanmax(errors_train),
                            100
                        )
    
                        ax_train_err.plot(
                            x_train_err,
                            norm.pdf(x_train_err, mu_train, std_train) * len(errors_train),
                            "r--",
                            label=t('validation.plot_normal_label', mu=mu_train, sigma=std_train)
                        )
    
                ax_train_err.set_xlabel(t('validation.plot_error_label'))
                ax_train_err.set_ylabel(t('validation.plot_count_label'))
                ax_train_err.set_title(t('validation.plot_train_error_title'))
                ax_train_err.legend(fontsize=8)
                ax_train_err.grid(True, alpha=0.3)
    
                fig_train_err.tight_layout()
                st.pyplot(fig_train_err)
    
            # ------------------------------------------------------------
            # Контрольная выборка: таблица
    
            st.subheader(t('validation.holdout_test_table_title'))
    
            st.dataframe(
                res["test_table"],
                width="stretch",
                hide_index=True
            )
    
            # ------------------------------------------------------------
            # Контрольная выборка: графики
    
            st.markdown(t('validation.holdout_test_plots_title'))
    
            col_test_plot, col_test_err = st.columns(2)
    
            with col_test_plot:
                fig_test, ax_test = plt.subplots(figsize=(5, 4))
    
                min_test = min(np.nanmin(y_test), np.nanmin(y_pred_test))
                max_test = max(np.nanmax(y_test), np.nanmax(y_pred_test))
                pad_test = (max_test - min_test) * 0.05 if max_test > min_test else 1.0
    
                ax_test.scatter(
                    y_test,
                    y_pred_test,
                    alpha=0.8,
                    s=55,
                    marker="^"
                )
    
                ax_test.plot(
                    [min_test - pad_test, max_test + pad_test],
                    [min_test - pad_test, max_test + pad_test],
                    "r--",
                    lw=1.5,
                    label=t('validation.plot_ideal_label')
                )
    
                ax_test.set_xlabel(t('validation.plot_exp_label'))
                ax_test.set_ylabel(t('validation.plot_pred_label'))
                ax_test.set_title(t('validation.plot_test_title', model=st.session_state.last_model_algorithm))
                ax_test.legend(fontsize=8)
                ax_test.grid(True, alpha=0.3)
    
                fig_test.tight_layout()
                st.pyplot(fig_test)
    
            with col_test_err:
                errors_test = y_test - y_pred_test
    
                fig_test_err, ax_test_err = plt.subplots(figsize=(5, 4))
    
                safe_histplot(ax_test_err, errors_test, bins=30, kde=True, color='coral', edgecolor='black', alpha=0.8)
    
                if len(errors_test) > 1:
                    mu_test = np.mean(errors_test)
                    std_test = np.std(errors_test)
    
                    if std_test > 1e-12:
                        x_test_err = np.linspace(
                            np.nanmin(errors_test),
                            np.nanmax(errors_test),
                            100
                        )
    
                        ax_test_err.plot(
                            x_test_err,
                            norm.pdf(x_test_err, mu_test, std_test) * len(errors_test),
                            "r--",
                            label=t('validation.plot_normal_label', mu=mu_test, sigma=std_test)
                        )
    
                ax_test_err.set_xlabel(t('validation.plot_error_label'))
                ax_test_err.set_ylabel(t('validation.plot_count_label'))
                ax_test_err.set_title(t('validation.plot_test_error_title'))
                ax_test_err.legend(fontsize=8)
                ax_test_err.grid(True, alpha=0.3)
    
                fig_test_err.tight_layout()
                st.pyplot(fig_test_err)
    
            # ------------------------------------------------------------
            # Метрики
    
            st.subheader(t('validation.holdout_test_metrics_title'))
    
            col_m1, col_m2, col_m3 = st.columns(3)
    
            with col_m1:
                st.metric(t('validation.metric_r2_test'), _fmt_validation_metric(res['metrics_test'].get('R2')))
    
            with col_m2:
                st.metric(t('validation.metric_rmse_test'), _fmt_validation_metric(res['metrics_test'].get('RMSE')))
    
            with col_m3:
                st.metric(t('validation.metric_mae_test'), _fmt_validation_metric(res['metrics_test'].get('MAE')))

            _render_metric_diagnostics(res.get("metrics_test", {}), "Test")
            _render_advanced_metric_table(res.get("metrics_test", {}), "Test")
    
            # ------------------------------------------------------------
            # Сводный график
    
            st.subheader(t('validation.holdout_combined_title'))
    
            show_train = st.checkbox(
                t('validation.holdout_show_train'),
                value=True,
                key="show_train"
            )
    
            show_test = st.checkbox(
                t('validation.holdout_show_test'),
                value=True,
                key="show_test"
            )
    
            fig_comb, ax_comb = plt.subplots(figsize=(7, 5))
    
            all_y = np.concatenate([y_train, y_test])
            min_y, max_y = all_y.min(), all_y.max()
            padding = (max_y - min_y) * 0.05 if max_y > min_y else 1.0
            plot_min = min_y - padding
            plot_max = max_y + padding
    
            ax_comb.plot(
                [plot_min, plot_max],
                [plot_min, plot_max],
                "k--",
                lw=1.5,
                label=t('validation.plot_ideal_label')
            )
    
            if show_train:
                ax_comb.scatter(
                    y_train,
                    y_pred_train,
                    alpha=0.65,
                    s=55,
                    color="#1f77b4",
                    label=t('validation.plot_train_label'),
                    edgecolors="w",
                    linewidth=0.6
                )
    
            if show_test:
                ax_comb.scatter(
                    y_test,
                    y_pred_test,
                    alpha=0.85,
                    s=75,
                    color="#ff7f0e",
                    marker="^",
                    label=t('validation.plot_test_label'),
                    edgecolors="w",
                    linewidth=0.6
                )
    
            ax_comb.set_xlim(plot_min, plot_max)
            ax_comb.set_ylim(plot_min, plot_max)
            ax_comb.set_xlabel(t('validation.plot_exp_label'), fontsize=14)
            ax_comb.set_ylabel(t('validation.plot_pred_label'), fontsize=14)
            ax_comb.set_title(
                t('validation.plot_combined_title', model=st.session_state.last_model_algorithm),
                fontsize=16
            )
            ax_comb.tick_params(axis="both", labelsize=12)
            ax_comb.legend(fontsize=12, loc="upper left", frameon=True)
    
            fig_comb.tight_layout()
    
            show_compact_matplotlib_plot(
                fig_comb,
                width=1200,
                dpi=120
            )
    
        else:
            st.info(t('validation.holdout_not_run'))
    
    with tab_kfold:
        k = st.slider(t('validation.kfold_k_label'), 3, 10, 5, key="kfold_k")
    
        if st.button(t('validation.kfold_run_button'), type="primary", key="run_kfold"):
            try:
                smiles_current = data[smiles_col_current].iloc[valid_indices_current].values.tolist()
    
                res_kfold = qspr_kfold_validation(
                    X=X_all_current,
                    y=y_all_current,
                    model_name=st.session_state.last_model_algorithm,
                    valid_indices=valid_indices_current,
                    smiles=smiles_current,
                    k=k,
                    params=get_model_params_from_session(),
                    scale=True,
                    shuffle=True,
                    random_state=42,
                    selector_config=selector_config_current,
                )
    
                kfold_settings = {
                    "kind": "kfold",
                    "k": k,
                    "shuffle": True,
                    "random_state": 42,
                }
                kfold_hash = analysis_result_hash(
                    st.session_state,
                    st.session_state.last_model_algorithm,
                    params=get_model_params_from_session(),
                    validation_settings=kfold_settings,
                    X=X_all_current,
                    y=y_all_current,
                    desc_names=desc_names_for_validation,
                    valid_indices=valid_indices_current,
                )
                res_kfold["validation_settings"] = kfold_settings
                st.session_state.kfold_results_dict[
                    st.session_state.last_model_algorithm
                ] = attach_result_cache_metadata(res_kfold, kfold_hash)
                st.session_state.pop(
                    f"error_analysis_result_{st.session_state.last_model_algorithm}",
                    None
                )
                qspr_save_results_auto(res_kfold["result_table"], "kfold", target_col, len(y_all_current))
                log_validation_result(
                    "kfold",
                    st.session_state.last_model_algorithm,
                    res_kfold,
                    y_values=y_all_current,
                    details={"folds": k},
                )
                st.rerun()
    
            except Exception as e:
                message = t('validation.kfold_error', error=e)
                st.error(message)
                log_streamlit_message(
                    "VALIDATION",
                    message,
                    level="error",
                    details={"error": str(e), "model": st.session_state.last_model_algorithm, "folds": k},
                    event="kfold_failed",
                )
    
        res = st.session_state.kfold_results_dict.get(st.session_state.last_model_algorithm)
        if isinstance(res, dict):
            expected_hash = analysis_result_hash(
                st.session_state,
                st.session_state.last_model_algorithm,
                params=get_model_params_from_session(),
                validation_settings={
                    "kind": "kfold",
                    "k": k,
                    "shuffle": True,
                    "random_state": 42,
                },
                X=X_all_current,
                y=y_all_current,
                desc_names=desc_names_for_validation,
                valid_indices=valid_indices_current,
            )
            if not cached_result_is_current(res, expected_hash):
                st.warning(
                    "Stored K-fold result belongs to an older data/descriptor/validation configuration. Run K-fold again."
                )
                res = None
    
        if res is not None:
            st.subheader(t('validation.kfold_results_title', k=res['k']))
    
            y_kfold = np.asarray(res["y"], dtype=float)
            y_pred_kfold = np.asarray(res["y_pred_cv"], dtype=float)
            errors_kfold = y_kfold - y_pred_kfold
    
            st.dataframe(
                res["result_table"],
                width="stretch",
                hide_index=True
            )
    
            col_k_m1, col_k_m2, col_k_m3, col_k_m4 = st.columns(4)
    
            with col_k_m1:
                st.metric(t('validation.kfold_metric_r2q2'), _fmt_validation_metric(res['metrics'].get('R2')))
    
            with col_k_m2:
                st.metric(t('validation.metric_rmse'), _fmt_validation_metric(res['metrics'].get('RMSE')))
    
            with col_k_m3:
                st.metric(t('validation.metric_mae'), _fmt_validation_metric(res['metrics'].get('MAE')))
    
            with col_k_m4:
                st.metric(t('validation.metric_mape'), _fmt_mape_metric(res['metrics']))

            _render_metric_diagnostics(res.get("metrics", {}), "K-Fold")
            _render_advanced_metric_table(res.get("metrics", {}), "K-Fold")
    
            st.markdown(t('validation.kfold_plots_title'))
    
            col_k_plot, col_k_err = st.columns(2)
    
            with col_k_plot:
                fig_k, ax_k = plt.subplots(figsize=(5, 4))
    
                min_k = min(np.nanmin(y_kfold), np.nanmin(y_pred_kfold))
                max_k = max(np.nanmax(y_kfold), np.nanmax(y_pred_kfold))
                pad_k = (max_k - min_k) * 0.05 if max_k > min_k else 1.0
    
                ax_k.scatter(
                    y_kfold,
                    y_pred_kfold,
                    alpha=0.75,
                    s=45
                )
    
                ax_k.plot(
                    [min_k - pad_k, max_k + pad_k],
                    [min_k - pad_k, max_k + pad_k],
                    "r--",
                    lw=1.5,
                    label=t('validation.plot_ideal_label')
                )
    
                ax_k.set_xlim(min_k - pad_k, max_k + pad_k)
                ax_k.set_ylim(min_k - pad_k, max_k + pad_k)
                ax_k.set_xlabel(t('validation.plot_exp_label'))
                ax_k.set_ylabel(t('validation.kfold_plot_ylabel', k=res['k']))
                ax_k.set_title(t('validation.kfold_plot_title', k=res['k'], model=st.session_state.last_model_algorithm))
                ax_k.legend(fontsize=8)
                ax_k.grid(True, alpha=0.3)
    
                fig_k.tight_layout()
                st.pyplot(fig_k)
    
            with col_k_err:
                fig_k_err, ax_k_err = plt.subplots(figsize=(5, 4))
    
                safe_histplot(ax_k_err, errors_kfold, bins=30, kde=True, color='steelblue', edgecolor='black', alpha=0.75)
    
                if len(errors_kfold) > 1:
                    mu_k = np.mean(errors_kfold)
                    std_k = np.std(errors_kfold)
    
                    if std_k > 1e-12:
                        x_k_err = np.linspace(
                            np.nanmin(errors_kfold),
                            np.nanmax(errors_kfold),
                            100
                        )
    
                        ax_k_err.plot(
                            x_k_err,
                            norm.pdf(x_k_err, mu_k, std_k) * len(errors_kfold),
                            "r--",
                            label=t('validation.plot_normal_label', mu=mu_k, sigma=std_k)
                        )
    
                ax_k_err.set_xlabel(t('validation.plot_error_label'))
                ax_k_err.set_ylabel(t('validation.plot_count_label'))
                ax_k_err.set_title(t('validation.kfold_error_hist_title', k=res['k']))
                ax_k_err.legend(fontsize=8)
                ax_k_err.grid(True, alpha=0.3)
    
                fig_k_err.tight_layout()
                st.pyplot(fig_k_err)
    
            st.markdown(t('validation.kfold_residuals_title'))
    
            fig_k_res, ax_k_res = plt.subplots(figsize=(7, 4))
    
            ax_k_res.scatter(
                y_pred_kfold,
                errors_kfold,
                alpha=0.75,
                s=45
            )
    
            ax_k_res.axhline(
                0,
                color="red",
                linestyle="--",
                linewidth=1.5
            )
    
            ax_k_res.set_xlabel(t('validation.kfold_residuals_xlabel', k=res['k']))
            ax_k_res.set_ylabel(t('validation.kfold_residuals_ylabel'))
            ax_k_res.set_title(t('validation.kfold_residuals_plot_title', k=res['k']))
            ax_k_res.grid(True, alpha=0.3)
    
            fig_k_res.tight_layout()
            st.pyplot(fig_k_res)
    
        else:
            st.info(t('validation.kfold_not_run'))
    
    with tab_loo:
        st.info(t('validation.loo_info'))
    
        if st.button(t('validation.loo_run_button'), type="primary", key="run_loo"):
            try:
                smiles_current = data[smiles_col_current].iloc[valid_indices_current].values.tolist()
                loo_skip_checker = globals().get("qspr_loo_skip_reason")
                if callable(loo_skip_checker):
                    loo_skip_reason = loo_skip_checker(
                        st.session_state.last_model_algorithm,
                        len(y_all_current),
                    )
                    if loo_skip_reason:
                        st.warning(loo_skip_reason)
                        log_streamlit_message(
                            "VALIDATION",
                            loo_skip_reason,
                            level="warning",
                            details={"model": st.session_state.last_model_algorithm},
                            event="loo_skipped_cloud_guard",
                        )
                        st.stop()
    
                res_loo = qspr_loo_validation(
                    X=X_all_current,
                    y=y_all_current,
                    model_name=st.session_state.last_model_algorithm,
                    valid_indices=valid_indices_current,
                    smiles=smiles_current,
                    params=get_model_params_from_session(),
                    scale=True,
                    selector_config=selector_config_current,
                )
    
                loo_settings = {"kind": "loo"}
                loo_hash = analysis_result_hash(
                    st.session_state,
                    st.session_state.last_model_algorithm,
                    params=get_model_params_from_session(),
                    validation_settings=loo_settings,
                    X=X_all_current,
                    y=y_all_current,
                    desc_names=desc_names_for_validation,
                    valid_indices=valid_indices_current,
                )
                res_loo["validation_settings"] = loo_settings
                st.session_state.loo_results_dict[
                    st.session_state.last_model_algorithm
                ] = attach_result_cache_metadata(res_loo, loo_hash)
                st.session_state.pop(
                    f"error_analysis_result_{st.session_state.last_model_algorithm}",
                    None
                )
                qspr_save_results_auto(res_loo["result_table"], "loo", target_col, len(y_all_current))
                log_validation_result(
                    "loo",
                    st.session_state.last_model_algorithm,
                    res_loo,
                    y_values=y_all_current,
                )
                st.rerun()
    
            except Exception as e:
                message = t('validation.loo_error', error=e)
                st.error(message)
                log_streamlit_message(
                    "VALIDATION",
                    message,
                    level="error",
                    details={"error": str(e), "model": st.session_state.last_model_algorithm},
                    event="loo_failed",
                )
    
        res = st.session_state.loo_results_dict.get(st.session_state.last_model_algorithm)
        if isinstance(res, dict):
            expected_hash = analysis_result_hash(
                st.session_state,
                st.session_state.last_model_algorithm,
                params=get_model_params_from_session(),
                validation_settings={"kind": "loo"},
                X=X_all_current,
                y=y_all_current,
                desc_names=desc_names_for_validation,
                valid_indices=valid_indices_current,
            )
            if not cached_result_is_current(res, expected_hash):
                st.warning(
                    "Stored LOO result belongs to an older data/descriptor/validation configuration. Run LOO again."
                )
                res = None
    
        if res is not None:
            st.subheader(t('validation.loo_results_title'))
    
            y_loo = np.asarray(res["y"], dtype=float)
            y_pred_loo = np.asarray(res["y_pred_loo"], dtype=float)
            errors_loo = y_loo - y_pred_loo
    
            st.dataframe(
                res["result_table"],
                width="stretch",
                hide_index=True
            )
    
            col_loo_m1, col_loo_m2, col_loo_m3, col_loo_m4 = st.columns(4)
    
            with col_loo_m1:
                st.metric(t('validation.loo_metric_r2q2'), _fmt_validation_metric(res['metrics'].get('R2')))
    
            with col_loo_m2:
                st.metric(t('validation.metric_rmse'), _fmt_validation_metric(res['metrics'].get('RMSE')))
    
            with col_loo_m3:
                st.metric(t('validation.metric_mae'), _fmt_validation_metric(res['metrics'].get('MAE')))
    
            with col_loo_m4:
                st.metric(t('validation.metric_mape'), _fmt_mape_metric(res['metrics']))

            _render_metric_diagnostics(res.get("metrics", {}), "LOO")
            _render_advanced_metric_table(res.get("metrics", {}), "LOO")
    
            st.markdown(t('validation.loo_plots_title'))
    
            col_loo_plot, col_loo_err = st.columns(2)
    
            with col_loo_plot:
                fig_loo, ax_loo = plt.subplots(figsize=(5, 4))
    
                min_loo = min(np.nanmin(y_loo), np.nanmin(y_pred_loo))
                max_loo = max(np.nanmax(y_loo), np.nanmax(y_pred_loo))
                pad_loo = (max_loo - min_loo) * 0.05 if max_loo > min_loo else 1.0
    
                ax_loo.scatter(
                    y_loo,
                    y_pred_loo,
                    alpha=0.75,
                    s=45
                )
    
                ax_loo.plot(
                    [min_loo - pad_loo, max_loo + pad_loo],
                    [min_loo - pad_loo, max_loo + pad_loo],
                    "r--",
                    lw=1.5,
                    label=t('validation.plot_ideal_label')
                )
    
                ax_loo.set_xlim(min_loo - pad_loo, max_loo + pad_loo)
                ax_loo.set_ylim(min_loo - pad_loo, max_loo + pad_loo)
                ax_loo.set_xlabel(t('validation.plot_exp_label'))
                ax_loo.set_ylabel(t('validation.loo_plot_ylabel'))
                ax_loo.set_title(t('validation.loo_plot_title', model=st.session_state.last_model_algorithm))
                ax_loo.legend(fontsize=8)
                ax_loo.grid(True, alpha=0.3)
    
                fig_loo.tight_layout()
                st.pyplot(fig_loo)
    
            with col_loo_err:
                fig_loo_err, ax_loo_err = plt.subplots(figsize=(5, 4))
    
                safe_histplot(ax_loo_err, errors_loo, bins=30, kde=True, color='coral', edgecolor='black', alpha=0.8)
    
                if len(errors_loo) > 1:
                    mu_loo = np.mean(errors_loo)
                    std_loo = np.std(errors_loo)
    
                    if std_loo > 1e-12:
                        x_loo_err = np.linspace(
                            np.nanmin(errors_loo),
                            np.nanmax(errors_loo),
                            100
                        )
    
                        ax_loo_err.plot(
                            x_loo_err,
                            norm.pdf(x_loo_err, mu_loo, std_loo) * len(errors_loo),
                            "r--",
                            label=t('validation.plot_normal_label', mu=mu_loo, sigma=std_loo)
                        )
    
                ax_loo_err.set_xlabel(t('validation.plot_error_label'))
                ax_loo_err.set_ylabel(t('validation.plot_count_label'))
                ax_loo_err.set_title(t('validation.loo_error_hist_title'))
                ax_loo_err.legend(fontsize=8)
                ax_loo_err.grid(True, alpha=0.3)
    
                fig_loo_err.tight_layout()
                st.pyplot(fig_loo_err)
    
            st.markdown(t('validation.loo_residuals_title'))
    
            fig_loo_res, ax_loo_res = plt.subplots(figsize=(7, 4))
    
            ax_loo_res.scatter(
                y_pred_loo,
                errors_loo,
                alpha=0.75,
                s=45
            )
    
            ax_loo_res.axhline(
                0,
                color="red",
                linestyle="--",
                linewidth=1.5
            )
    
            ax_loo_res.set_xlabel(t('validation.loo_residuals_xlabel'))
            ax_loo_res.set_ylabel(t('validation.loo_residuals_ylabel'))
            ax_loo_res.set_title(t('validation.loo_residuals_plot_title'))
            ax_loo_res.grid(True, alpha=0.3)
    
            fig_loo_res.tight_layout()
            st.pyplot(fig_loo_res)
    
        else:
            st.info(t('validation.loo_not_run'))
    
    with tab_ext:
        st.subheader(t('validation.external.subheader'))
        st.markdown(t('validation.external.idea'))
    
        col1, col2, col3 = st.columns(3)
        with col1:
            ext_fraction = st.slider(
                t('validation.external.fraction_label'),
                min_value=5,
                max_value=50,
                value=20,
                step=1,
                key="ext_fraction"
            )
        with col2:
            ext_repeats = st.number_input(
                t('validation.external.repeats_label'),
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="ext_repeats"
            )
        with col3:
            ext_metric = st.selectbox(
                t('validation.external.metric_label'),
                ["euclidean", "mahalanobis", "cosine"],
                index=0,
                key="ext_metric",
                help=t('validation.external.metric_help')
            )
    
        if st.button(t('validation.external.run_button'), type="primary", key="run_ext_validation"):
            try:
                # Проверяем, что дескрипторы рассчитаны и модель обучена
                if not st.session_state.get("desc_calculated", False):
                    message = t('validation.external.error_no_descriptors')
                    st.error(message)
                    log_streamlit_message(
                        "VALIDATION",
                        message,
                        level="error",
                        details={"model": st.session_state.last_model_algorithm},
                        event="external_validation_no_descriptors",
                    )
                    st.stop()
    
                # Берём актуальные данные из session_state
                X_current = st.session_state.get("X_all")
                y_current = st.session_state.get("y_all")
                valid_idx_current = st.session_state.get("valid_indices", list(range(len(y_current))))
                smiles_current = data[smiles_col_current].iloc[valid_idx_current].values.tolist()
    
                with st.spinner(t('validation.external.spinner')):
                    ext_settings = {
                        "kind": "distance_holdout",
                        "fraction": ext_fraction / 100.0,
                        "n_repeats": int(ext_repeats),
                        "distance_metric": ext_metric,
                        "random_state": int(st.session_state.get("random_seed", 42)),
                    }
                    ext_result = qspr_external_validation_simulator(
                        X=X_current,
                        y=y_current,
                        model_name=st.session_state.last_model_algorithm,
                        valid_indices=valid_idx_current,
                        smiles=smiles_current,
                        fraction=ext_fraction/100.0,
                        n_repeats=int(ext_repeats),
                        distance_metric=ext_metric,
                        params=get_model_params_from_session(),
                        scale=True,
                        random_state=int(st.session_state.get("random_seed", 42)),
                        selector_config=selector_config_current,
                    )
    
                ext_result["model_name"] = st.session_state.last_model_algorithm
                ext_result["validation_result_key"] = st.session_state.last_model_algorithm
                ext_result["validation_settings"] = ext_settings
                ext_hash = analysis_result_hash(
                    st.session_state,
                    st.session_state.last_model_algorithm,
                    params=get_model_params_from_session(),
                    validation_settings=ext_settings,
                    X=X_current,
                    y=y_current,
                    desc_names=desc_names_for_validation,
                    valid_indices=valid_idx_current,
                )
                ext_result = attach_result_cache_metadata(ext_result, ext_hash)
                st.session_state.ext_validation_result = ext_result
                st.session_state.setdefault("ext_validation_results_dict", {})
                st.session_state.ext_validation_results_dict[
                    st.session_state.last_model_algorithm
                ] = ext_result
                ext_summary = ext_result.get("summary", {})
                ext_quality = interpret_validation_quality(
                    r2=ext_summary.get("test_R2_mean"),
                    rmse=ext_summary.get("test_RMSE_mean"),
                    y_std=float(np.nanstd(y_current, ddof=1)) if len(y_current) > 1 else None,
                    metric_std=ext_summary.get("test_R2_std"),
                    method="Distance-based hold-out stress test",
                )
                add_event_log(
                    "VALIDATION",
                    (
                        f"Distance-based hold-out stress test {st.session_state.last_model_algorithm}: "
                        f"{ext_summary.get('n_repeats')} повторов, test={float(ext_summary.get('fraction', 0)):.0%}, "
                        f"R²={_fmt_mean_std(ext_summary.get('test_R2_mean'), ext_summary.get('test_R2_std'))}, "
                        f"RMSE={_fmt_mean_std(ext_summary.get('test_RMSE_mean'), ext_summary.get('test_RMSE_std'))}; "
                        f"качество {validation_quality_text(ext_quality)}."
                    ),
                    level=ext_quality.get("level", "info"),
                    details={
                        "model": st.session_state.last_model_algorithm,
                        "mae": _fmt_mean_std(ext_summary.get("test_MAE_mean"), ext_summary.get("test_MAE_std")),
                        "validation_mode": ext_summary.get("validation_mode"),
                        "distance_metric": ext_summary.get("distance_metric"),
                        "methodology_note": ext_summary.get("methodology_note"),
                    },
                    event="distance_holdout_stress_test_completed",
                )
                st.success(t('validation.external.success'))
                st.rerun()
    
            except Exception as e:
                message = t('validation.external.error', error=e)
                st.error(message)
                log_streamlit_message(
                    "VALIDATION",
                    message,
                    level="error",
                    details={"error": str(e), "model": st.session_state.last_model_algorithm},
                    event="external_validation_failed",
                )
    
        ext_res = st.session_state.get("ext_validation_result")
        if isinstance(ext_res, dict):
            ext_settings_current = ext_res.get("validation_settings")
            if isinstance(ext_settings_current, dict):
                expected_ext_hash = analysis_result_hash(
                    st.session_state,
                    st.session_state.last_model_algorithm,
                    params=get_model_params_from_session(),
                    validation_settings=ext_settings_current,
                    X=X_all_current,
                    y=y_all_current,
                    desc_names=desc_names_for_validation,
                    valid_indices=valid_indices_current,
                )
                if not cached_result_is_current(ext_res, expected_ext_hash):
                    st.warning(
                        "Stored distance-based hold-out result belongs to an older "
                        "data/descriptor/validation configuration. Run the stress test again."
                    )
                    ext_res = None
            else:
                ext_res = None
        if ext_res is not None:
            st.subheader(t('validation.external.results_subheader'))
            summary = ext_res['summary']
            st.info(
                summary.get(
                    "methodology_note",
                    "This is a distance-based hold-out stress test, not strict external validation.",
                )
            )
            st.caption(
                f"Mode: {summary.get('validation_label', 'Distance-based hold-out stress test')}; "
                f"distance metric: {summary.get('distance_metric', 'euclidean')}; "
                f"test selection geometry: {summary.get('test_selection_geometry', '')}."
            )
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric(
                    t('validation.external.metric_test_r2'),
                    f"{summary['test_R2_mean']:.3f} ± {summary['test_R2_std']:.3f}"
                )
            with col_s2:
                st.metric(
                    t('validation.external.metric_test_rmse'),
                    f"{summary['test_RMSE_mean']:.3f} ± {summary['test_RMSE_std']:.3f}"
                )
            with col_s3:
                st.metric(t('validation.external.metric_test_size'), int(summary['test_size_mean']))
            with col_s4:
                st.metric(t('validation.external.metric_repeats'), ext_res['n_repeats'])
    
            # График устойчивости
            col_split_1, col_split_2, col_split_3 = st.columns(3)
            with col_split_1:
                st.metric(
                    "Unique test splits",
                    int(summary.get("unique_test_splits", 0) or 0)
                )
            with col_split_2:
                mean_jaccard = summary.get("mean_test_split_jaccard", np.nan)
                st.metric(
                    "Mean test-set Jaccard",
                    f"{mean_jaccard:.3f}" if np.isfinite(mean_jaccard) else "n/a"
                )
            with col_split_3:
                max_jaccard = summary.get("max_test_split_jaccard", np.nan)
                st.metric(
                    "Max test-set Jaccard",
                    f"{max_jaccard:.3f}" if np.isfinite(max_jaccard) else "n/a"
                )

            with st.expander("Test-set selection frequency", expanded=False):
                st.dataframe(
                    ext_res.get("test_selection_frequency", pd.DataFrame()),
                    width="stretch",
                    hide_index=True
                )

            with st.expander("Pairwise test-set Jaccard similarity", expanded=False):
                st.dataframe(
                    ext_res.get("test_split_jaccard", pd.DataFrame()),
                    width="stretch",
                    hide_index=True
                )

            metrics_df = ext_res['metrics_df']
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(metrics_df['repeat'], metrics_df['test_R2'], marker='o', linestyle='-', label=t('validation.external.plot_label'))
            mean_r2 = metrics_df['test_R2'].mean()
            ax.axhline(y=mean_r2, color='r', linestyle='--', label=t('validation.external.plot_mean', mean=mean_r2))
            ax.set_xlabel(t('validation.external.plot_xlabel'))
            ax.set_ylabel(t('validation.external.plot_ylabel'))
            ax.set_title(t('validation.external.plot_title'))
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
    
            # Таблица всех прогнозов
            with st.expander(t('validation.external.expander_predictions')):
                st.dataframe(ext_res['combined_test_table'], width="stretch", hide_index=True)
    
            csv_ext = qspr_csv_download_bytes(ext_res['combined_test_table'])
            st.download_button(
                t('validation.external.download_button'),
                csv_ext,
                "external_validation_results.csv",
                "text/csv",
                key="download_ext_validation"
            )
    
    # ------------------------------------------------------------
    # Repeated Hold-out / Monte-Carlo CV
    st.markdown(
        f'<div class="tool-badge">{t("repeated_holdout.tool_badge")}</div>',
        unsafe_allow_html=True
    )
    with st.expander(t('repeated_holdout.expander_title'), expanded=False):
        st.markdown(t('repeated_holdout.description'))
    
        current_model_for_repeated_holdout = st.session_state.last_model_algorithm
    
        if current_model_for_repeated_holdout not in st.session_state.get("trained_models", {}):
            st.info(t('repeated_holdout.info_train_first'))
        else:
            col_rh_1, col_rh_2, col_rh_3, col_rh_4 = st.columns(4)
    
            with col_rh_1:
                repeated_holdout_n = st.number_input(
                    t('repeated_holdout.n_repeats_label'),
                    min_value=5,
                    max_value=1000,
                    value=100,
                    step=5,
                    key="repeated_holdout_n"
                )
    
            with col_rh_2:
                repeated_holdout_test_percent = st.slider(
                    t('repeated_holdout.test_percent_label'),
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    key="repeated_holdout_test_percent"
                )
    
            with col_rh_3:
                repeated_holdout_seed = st.number_input(
                    t('repeated_holdout.random_state_label'),
                    value=42,
                    step=1,
                    key="repeated_holdout_seed"
                )
    
            with col_rh_4:
                repeated_holdout_save_details = st.checkbox(
                    t('repeated_holdout.save_details_label'),
                    value=True,
                    key="repeated_holdout_save_details"
                )
    
            st.caption(t('repeated_holdout.recommendation_caption'))
    
            if st.button(
                t('repeated_holdout.run_button'),
                type="primary",
                key="run_repeated_holdout"
            ):
                try:
                    smiles_for_repeated_holdout = data[smiles_col_current].iloc[
                        valid_indices_current
                    ].values.tolist()
    
                    progress_rh = st.progress(
                        0,
                        text=t('repeated_holdout.progress_preparing')
                    )
    
                    def _rh_progress(done, total):
                        progress_rh.progress(
                            int(done / total * 100),
                            text=t('repeated_holdout.progress_text', done=done, total=total)
                        )
    
                    with st.spinner(t('repeated_holdout.spinner')):
                        repeated_holdout_result = qspr_repeated_holdout_validation(
                            X=X_all_current,
                            y=y_all_current,
                            model_name=current_model_for_repeated_holdout,
                            valid_indices=valid_indices_current,
                            smiles=smiles_for_repeated_holdout,
                            target_col=target_col,
                            params=get_model_params_from_session(),
                            n_repeats=int(repeated_holdout_n),
                            test_size=float(repeated_holdout_test_percent) / 100.0,
                            random_state=int(repeated_holdout_seed),
                            scale=True,
                            progress_callback=_rh_progress,
                            selector_config=selector_config_current
                        )
    
                    if not repeated_holdout_save_details:
                        repeated_holdout_result["combined_train_table"] = pd.DataFrame()
                        repeated_holdout_result["combined_test_table"] = pd.DataFrame()
    
                    if "repeated_holdout_results_dict" not in st.session_state:
                        st.session_state.repeated_holdout_results_dict = {}
    
                    st.session_state.repeated_holdout_results_dict[
                        current_model_for_repeated_holdout
                    ] = repeated_holdout_result
    
                    log_repeated_holdout_result(
                        current_model_for_repeated_holdout,
                        repeated_holdout_result,
                    )
    
                    progress_rh.progress(100, text=t('repeated_holdout.progress_done'))
                    st.success(t('repeated_holdout.success'))
                    st.rerun()
    
                except Exception as e:
                    message = t('repeated_holdout.error', error=e)
                    st.error(message)
                    log_streamlit_message(
                        "VALIDATION",
                        message,
                        level="error",
                        details={"error": str(e), "model": current_model_for_repeated_holdout},
                        event="repeated_holdout_failed",
                    )
    
            repeated_holdout_res = st.session_state.get(
                "repeated_holdout_results_dict",
                {}
            ).get(current_model_for_repeated_holdout)
    
            if repeated_holdout_res is not None:
                st.markdown(t('repeated_holdout.results_title'))
    
                summary_table_rh = repeated_holdout_res.get(
                    "summary_table",
                    pd.DataFrame()
                )
    
                repeats_table_rh = repeated_holdout_res.get(
                    "repeats_table",
                    pd.DataFrame()
                )
    
                col_rh_m1, col_rh_m2, col_rh_m3, col_rh_m4 = st.columns(4)
    
                with col_rh_m1:
                    r2_mean = repeated_holdout_res.get("test_r2_mean", np.nan)
                    r2_std = repeated_holdout_res.get("test_r2_std", np.nan)
    
                    if pd.notna(r2_mean):
                        st.metric(t('repeated_holdout.metric_r2'), f"{r2_mean:.3f} ± {r2_std:.3f}")
                    else:
                        st.metric(t('repeated_holdout.metric_r2'), "—")
    
                with col_rh_m2:
                    rmse_mean = repeated_holdout_res.get("test_rmse_mean", np.nan)
                    rmse_std = repeated_holdout_res.get("test_rmse_std", np.nan)
    
                    if pd.notna(rmse_mean):
                        st.metric(t('repeated_holdout.metric_rmse'), f"{rmse_mean:.3f} ± {rmse_std:.3f}")
                    else:
                        st.metric(t('repeated_holdout.metric_rmse'), "—")
    
                with col_rh_m3:
                    st.metric(
                        t('repeated_holdout.metric_ok'),
                        repeated_holdout_res.get("n_ok", 0)
                    )
    
                with col_rh_m4:
                    st.metric(
                        t('repeated_holdout.metric_failed'),
                        repeated_holdout_res.get("n_failed", 0)
                    )
    
                conclusion_rh = str(
                    repeated_holdout_res.get("conclusion", "")
                )
    
                # Проверяем, содержит ли заключение ключевые слова для выбора типа сообщения
                if "устойчива" in conclusion_rh and "низкая" not in conclusion_rh:
                    st.success(t('repeated_holdout.conclusion_label', conclusion=conclusion_rh))
                elif "умеренная" in conclusion_rh:
                    st.info(t('repeated_holdout.conclusion_label', conclusion=conclusion_rh))
                else:
                    st.warning(t('repeated_holdout.conclusion_label', conclusion=conclusion_rh))
    
                if isinstance(summary_table_rh, pd.DataFrame) and not summary_table_rh.empty:
                    st.markdown(t('repeated_holdout.summary_title'))
    
                    st.dataframe(
                        summary_table_rh,
                        width="stretch",
                        hide_index=True
                    )
    
                if isinstance(repeats_table_rh, pd.DataFrame) and not repeats_table_rh.empty:
                    st.markdown(t('repeated_holdout.distribution_title'))
    
                    col_rh_plot_1, col_rh_plot_2 = st.columns(2)
    
                    with col_rh_plot_1:
                        test_r2_series = pd.to_numeric(
                            repeats_table_rh[t('repeated_holdout.test_r2')],
                            errors="coerce"
                        ).replace([np.inf, -np.inf], np.nan).dropna()
    
                        if len(test_r2_series) > 0:
                            fig_rh_r2, ax_rh_r2 = plt.subplots(figsize=(6, 4))
    
                            safe_histplot(ax_rh_r2, test_r2_series, bins=30, kde=True, color='coral', edgecolor='black', alpha=0.7)
    
                            ax_rh_r2.set_xlabel(t('repeated_holdout.hist_r2_xlabel'))
                            ax_rh_r2.set_ylabel(t('repeated_holdout.hist_ylabel'))
                            ax_rh_r2.set_title(t('repeated_holdout.hist_r2_title'))
                            ax_rh_r2.grid(True, alpha=0.3)
    
                            fig_rh_r2.tight_layout()
                            st.pyplot(fig_rh_r2)
                            plt.close(fig_rh_r2)
    
                    with col_rh_plot_2:
                        test_rmse_series = pd.to_numeric(
                            repeats_table_rh[t('repeated_holdout.test_rmse')],
                            errors="coerce"
                        ).replace([np.inf, -np.inf], np.nan).dropna()
    
                        if len(test_rmse_series) > 0:
                            fig_rh_rmse, ax_rh_rmse = plt.subplots(figsize=(6, 4))
    
                            safe_histplot(ax_rh_rmse, test_rmse_series, bins=30, kde=True, color='coral', edgecolor='black', alpha=0.7)
    
                            ax_rh_rmse.set_xlabel(t('repeated_holdout.hist_rmse_xlabel'))
                            ax_rh_rmse.set_ylabel(t('repeated_holdout.hist_ylabel'))
                            ax_rh_rmse.set_title(t('repeated_holdout.hist_rmse_title'))
                            ax_rh_rmse.grid(True, alpha=0.3)
    
                            fig_rh_rmse.tight_layout()
                            st.pyplot(fig_rh_rmse)
                            plt.close(fig_rh_rmse)
    
                    with st.expander(t('repeated_holdout.expander_repeats_table'), expanded=False):
                        st.dataframe(
                            repeats_table_rh,
                            width="stretch",
                            hide_index=True
                        )
    
                    csv_rh = qspr_csv_download_bytes(repeats_table_rh)
    
                    st.download_button(
                        t('repeated_holdout.download_repeats_csv'),
                        csv_rh,
                        f"repeated_holdout_{current_model_for_repeated_holdout}.csv",
                        "text/csv",
                        key=f"download_repeated_holdout_{current_model_for_repeated_holdout}"
                    )
    
                combined_test_table_rh = repeated_holdout_res.get(
                    "combined_test_table",
                    pd.DataFrame()
                )
    
                if isinstance(combined_test_table_rh, pd.DataFrame) and not combined_test_table_rh.empty:
                    with st.expander(t('repeated_holdout.expander_test_predictions'), expanded=False):
                        st.dataframe(
                            combined_test_table_rh.head(2000),
                            width="stretch",
                            hide_index=True
                        )
    
                        st.download_button(
                            t('repeated_holdout.download_test_predictions_csv'),
                            qspr_csv_download_bytes(combined_test_table_rh),
                            f"repeated_holdout_test_predictions_{current_model_for_repeated_holdout}.csv",
                            "text/csv",
                            key=f"download_repeated_holdout_test_predictions_{current_model_for_repeated_holdout}"
                        )
    
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Bootstrap validation
    st.markdown(
        f'<div class="tool-badge">{t("bootstrap.ui_tool_badge")}</div>',
        unsafe_allow_html=True
    )
    with st.expander(t('bootstrap.ui_expander_title'), expanded=False):
        st.markdown(t('bootstrap.ui_description'))
    
        current_model_for_bootstrap = st.session_state.last_model_algorithm
    
        if current_model_for_bootstrap not in st.session_state.get("trained_models", {}):
            st.info(t('bootstrap.ui_train_first'))
        else:
            col_bs_1, col_bs_2, col_bs_3, col_bs_4 = st.columns(4)
    
            with col_bs_1:
                bootstrap_n_iterations = st.number_input(
                    t('bootstrap.ui_iterations_label'),
                    min_value=10,
                    max_value=2000,
                    value=200,
                    step=10,
                    key="bootstrap_n_iterations"
                )
    
            with col_bs_2:
                bootstrap_sample_percent = st.slider(
                    t('bootstrap.ui_sample_percent_label'),
                    min_value=50,
                    max_value=150,
                    value=100,
                    step=5,
                    key="bootstrap_sample_percent"
                )
    
            with col_bs_3:
                bootstrap_seed = st.number_input(
                    t('bootstrap.ui_random_state_label'),
                    value=42,
                    step=1,
                    key="bootstrap_seed"
                )
    
            with col_bs_4:
                bootstrap_save_oob_predictions = st.checkbox(
                    t('bootstrap.ui_save_oob_label'),
                    value=True,
                    key="bootstrap_save_oob_predictions"
                )
    
            st.caption(t('bootstrap.ui_recommendation'))
    
            if st.button(
                t('bootstrap.ui_run_button'),
                type="primary",
                key="run_bootstrap_validation"
            ):
                try:
                    smiles_for_bootstrap = data[smiles_col_current].iloc[
                        valid_indices_current
                    ].values.tolist()
    
                    progress_bs = st.progress(
                        0,
                        text=t('bootstrap.ui_progress_preparing')
                    )
    
                    def _bs_progress(done, total):
                        progress_bs.progress(
                            int(done / total * 100),
                            text=t('bootstrap.ui_progress_text', done=done, total=total)
                        )
    
                    bootstrap_settings = {
                        "kind": "bootstrap",
                        "n_iterations": int(bootstrap_n_iterations),
                        "sample_fraction": float(bootstrap_sample_percent) / 100.0,
                        "random_state": int(bootstrap_seed),
                        "save_oob_predictions": bool(bootstrap_save_oob_predictions),
                    }

                    with st.spinner(t('bootstrap.ui_spinner')):
                        bootstrap_result = qspr_bootstrap_validation(
                            X=X_all_current,
                            y=y_all_current,
                            model_name=current_model_for_bootstrap,
                            valid_indices=valid_indices_current,
                            smiles=smiles_for_bootstrap,
                            target_col=target_col,
                            params=get_model_params_from_session(),
                            n_iterations=int(bootstrap_n_iterations),
                            sample_fraction=float(bootstrap_sample_percent) / 100.0,
                            random_state=int(bootstrap_seed),
                            scale=True,
                            progress_callback=_bs_progress,
                            selector_config=selector_config_current
                        )
    
                    if not bootstrap_save_oob_predictions:
                        bootstrap_result["oob_predictions_table"] = pd.DataFrame()
                    bootstrap_result["validation_settings"] = bootstrap_settings
    
                    if "bootstrap_results_dict" not in st.session_state:
                        st.session_state.bootstrap_results_dict = {}
    
                    st.session_state.bootstrap_results_dict[
                        current_model_for_bootstrap
                    ] = attach_result_cache_metadata(
                        bootstrap_result,
                        _validation_cache_hash(
                            current_model_for_bootstrap,
                            bootstrap_settings,
                        ),
                    )
    
                    log_bootstrap_result(
                        current_model_for_bootstrap,
                        bootstrap_result,
                    )
    
                    st.success(t('bootstrap.ui_success'))
                    st.rerun()
    
                except Exception as e:
                    message = t('bootstrap.ui_error', error=e)
                    st.error(message)
                    log_streamlit_message(
                        "BOOTSTRAP",
                        message,
                        level="error",
                        details={"error": str(e), "model": current_model_for_bootstrap},
                        event="bootstrap_failed",
                    )
    
            bootstrap_settings_current = {
                "kind": "bootstrap",
                "n_iterations": int(bootstrap_n_iterations),
                "sample_fraction": float(bootstrap_sample_percent) / 100.0,
                "random_state": int(bootstrap_seed),
                "save_oob_predictions": bool(bootstrap_save_oob_predictions),
            }
            bootstrap_res = _current_cached_validation_result(
                "bootstrap_results_dict",
                current_model_for_bootstrap,
                bootstrap_settings_current,
            )
    
            if isinstance(bootstrap_res, dict):
                bootstrap_summary = bootstrap_res.get("summary", {})
                bootstrap_summary_table = bootstrap_res.get("summary_table", pd.DataFrame())
                bootstrap_iterations_table = bootstrap_res.get("iterations_table", pd.DataFrame())
    
                st.markdown(t('bootstrap.ui_summary_title'))
    
                col_bs_m1, col_bs_m2, col_bs_m3 = st.columns(3)
    
                with col_bs_m1:
                    r2_mean = bootstrap_summary.get('r2_oob_mean', np.nan)
                    r2_std = bootstrap_summary.get('r2_oob_std', np.nan)
                    st.metric(
                        t('bootstrap.ui_metric_r2'),
                        f"{r2_mean:.3f} ± {r2_std:.3f}" if pd.notna(r2_mean) else "—"
                    )
    
                with col_bs_m2:
                    rmse_mean = bootstrap_summary.get('rmse_oob_mean', np.nan)
                    rmse_std = bootstrap_summary.get('rmse_oob_std', np.nan)
                    st.metric(
                        t('bootstrap.ui_metric_rmse'),
                        f"{rmse_mean:.3f} ± {rmse_std:.3f}" if pd.notna(rmse_mean) else "—"
                    )
    
                with col_bs_m3:
                    mae_mean = bootstrap_summary.get('mae_oob_mean', np.nan)
                    mae_std = bootstrap_summary.get('mae_oob_std', np.nan)
                    st.metric(
                        t('bootstrap.ui_metric_mae'),
                        f"{mae_mean:.3f} ± {mae_std:.3f}" if pd.notna(mae_mean) else "—"
                    )
    
                if isinstance(bootstrap_summary_table, pd.DataFrame) and not bootstrap_summary_table.empty:
                    st.dataframe(
                        bootstrap_summary_table,
                        width="stretch",
                        hide_index=True
                    )
    
                if isinstance(bootstrap_iterations_table, pd.DataFrame) and not bootstrap_iterations_table.empty:
                    ok_bs = bootstrap_iterations_table[
                        bootstrap_iterations_table[t('bootstrap.status')] == "ok"
                    ].copy()
    
                    if not ok_bs.empty:
                        st.markdown(t('bootstrap.ui_distribution_title'))
    
                        fig_bs_r2, ax_bs_r2 = plt.subplots(figsize=(7, 4))
                        ax_bs_r2.hist(
                            pd.to_numeric(ok_bs[t('bootstrap.r2_oob')], errors="coerce").dropna(),
                            bins=25,
                            alpha=0.75
                        )
                        ax_bs_r2.set_xlabel(t('bootstrap.ui_hist_r2_xlabel'))
                        ax_bs_r2.set_ylabel(t('bootstrap.ui_hist_ylabel'))
                        ax_bs_r2.set_title(t('bootstrap.ui_hist_r2_title'))
                        ax_bs_r2.grid(True, alpha=0.3)
                        fig_bs_r2.tight_layout()
                        st.pyplot(fig_bs_r2)
                        plt.close(fig_bs_r2)
    
                        fig_bs_rmse, ax_bs_rmse = plt.subplots(figsize=(7, 4))
                        ax_bs_rmse.hist(
                            pd.to_numeric(ok_bs[t('bootstrap.rmse_oob')], errors="coerce").dropna(),
                            bins=25,
                            alpha=0.75
                        )
                        ax_bs_rmse.set_xlabel(t('bootstrap.ui_hist_rmse_xlabel'))
                        ax_bs_rmse.set_ylabel(t('bootstrap.ui_hist_ylabel'))
                        ax_bs_rmse.set_title(t('bootstrap.ui_hist_rmse_title'))
                        ax_bs_rmse.grid(True, alpha=0.3)
                        fig_bs_rmse.tight_layout()
                        st.pyplot(fig_bs_rmse)
                        plt.close(fig_bs_rmse)
    
                    with st.expander(t('bootstrap.ui_expander_iterations'), expanded=False):
                        st.dataframe(
                            bootstrap_iterations_table,
                            width="stretch",
                            hide_index=True
                        )
    
                    st.download_button(
                        t('bootstrap.ui_download_iterations'),
                        qspr_csv_download_bytes(bootstrap_iterations_table),
                        f"bootstrap_iterations_{current_model_for_bootstrap}.csv",
                        "text/csv",
                        key=f"download_bootstrap_iterations_{current_model_for_bootstrap}"
                    )
    
                bootstrap_failed_iterations = bootstrap_res.get(
                    "failed_iterations_table",
                    pd.DataFrame()
                )
                if isinstance(bootstrap_failed_iterations, pd.DataFrame) and not bootstrap_failed_iterations.empty:
                    with st.expander("Failed / skipped bootstrap iterations", expanded=False):
                        st.dataframe(
                            bootstrap_failed_iterations,
                            width="stretch",
                            hide_index=True
                        )
    
                    st.download_button(
                        "Download failed / skipped bootstrap iterations",
                        qspr_csv_download_bytes(bootstrap_failed_iterations),
                        f"bootstrap_failed_iterations_{current_model_for_bootstrap}.csv",
                        "text/csv",
                        key=f"download_bootstrap_failed_iterations_{current_model_for_bootstrap}"
                    )
    
                bootstrap_oob_predictions = bootstrap_res.get(
                    "oob_predictions_table",
                    pd.DataFrame()
                )
    
                if isinstance(bootstrap_oob_predictions, pd.DataFrame) and not bootstrap_oob_predictions.empty:
                    with st.expander(t('bootstrap.ui_expander_oob'), expanded=False):
                        st.dataframe(
                            bootstrap_oob_predictions.head(2000),
                            width="stretch",
                            hide_index=True
                        )
    
                        st.download_button(
                            t('bootstrap.ui_download_oob'),
                            qspr_csv_download_bytes(bootstrap_oob_predictions),
                            f"bootstrap_oob_predictions_{current_model_for_bootstrap}.csv",
                            "text/csv",
                            key=f"download_bootstrap_oob_predictions_{current_model_for_bootstrap}"
                        )
    
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Y-randomization / permutation test
    st.markdown(
        f'<div class="tool-badge">{t("y_rand.ui_tool_badge")}</div>',
        unsafe_allow_html=True
    )
    with st.expander(t('y_rand.ui_expander_title'), expanded=False):
        st.markdown(t('y_rand.ui_description'))
    
        current_model_for_y_rand = st.session_state.last_model_algorithm
    
        if current_model_for_y_rand not in st.session_state.get("trained_models", {}):
            st.info(t('y_rand.ui_train_first'))
        else:
            col_yr_1, col_yr_2, col_yr_3, col_yr_4 = st.columns(4)
    
            with col_yr_1:
                y_rand_method_options = {
                    "kfold": t('y_rand.ui_method_kfold'),
                    "loo": t('y_rand.ui_method_loo'),
                }
                old_y_rand_method = st.session_state.get("y_randomization_method")
                if old_y_rand_method in set(y_rand_method_options.values()):
                    st.session_state.y_randomization_method = (
                        "loo"
                        if old_y_rand_method == t('y_rand.ui_method_loo')
                        else "kfold"
                    )
                y_rand_method = st.selectbox(
                    t('y_rand.ui_method_label'),
                    list(y_rand_method_options.keys()),
                    index=0,
                    key="y_randomization_method",
                    format_func=lambda key: y_rand_method_options.get(key, str(key)),
                )
    
            with col_yr_2:
                y_rand_n_perm = st.number_input(
                    t('y_rand.ui_n_perm_label'),
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10,
                    key="y_randomization_n_perm"
                )
    
            with col_yr_3:
                y_rand_k = st.slider(
                    t('y_rand.ui_k_label'),
                    min_value=3,
                    max_value=max(3, min(10, len(y_all_current))),
                    value=min(5, max(3, min(10, len(y_all_current)))),
                    step=1,
                    key="y_randomization_k"
                )
    
            with col_yr_4:
                y_rand_seed = st.number_input(
                    t('y_rand.ui_random_state_label'),
                    value=42,
                    step=1,
                    key="y_randomization_seed"
                )
    
            if y_rand_method == "loo":
                message = t('y_rand.ui_loo_warning')
                st.warning(message)
                warning_key = f"y_rand_loo_warning_logged_{current_model_for_y_rand}_{len(y_all_current)}"
                if not st.session_state.get(warning_key):
                    log_streamlit_message(
                        "Y_RANDOMIZATION",
                        message,
                        level="warning",
                        details={"model": current_model_for_y_rand, "n": len(y_all_current)},
                        event="y_randomization_loo_warning",
                    )
                    st.session_state[warning_key] = True
    
            y_rand_run_col_1, y_rand_run_col_2 = st.columns([1, 2])
    
            with y_rand_run_col_1:
                run_y_randomization = st.button(
                    t('y_rand.ui_run_button'),
                    type="primary",
                    key="run_y_randomization"
                )
    
            with y_rand_run_col_2:
                st.caption(t('y_rand.ui_recommendation'))
    
            if run_y_randomization:
                try:
                    smiles_for_y_rand = data[smiles_col_current].iloc[
                        valid_indices_current
                    ].values.tolist()
    
                    progress_y_rand = st.progress(
                        0,
                        text=t('y_rand.ui_progress_preparing')
                    )
    
                    def _yr_progress(done, total):
                        progress_y_rand.progress(
                            int(done / total * 100),
                            text=t('y_rand.ui_progress_text', done=done, total=total)
                        )
    
                    y_rand_settings = {
                        "kind": "y_randomization",
                        "method": str(y_rand_method),
                        "n_permutations": int(y_rand_n_perm),
                        "k": int(y_rand_k),
                        "random_state": int(y_rand_seed),
                    }

                    with st.spinner(t('y_rand.ui_spinner')):
                        y_rand_result = qspr_y_randomization_test(
                            X=X_all_current,
                            y=y_all_current,
                            model_name=current_model_for_y_rand,
                            valid_indices=valid_indices_current,
                            smiles=smiles_for_y_rand,
                            params=get_model_params_from_session(),
                            method=y_rand_method,
                            n_permutations=int(y_rand_n_perm),
                            k=int(y_rand_k),
                            random_state=int(y_rand_seed),
                            scale=True,
                            progress_callback=_yr_progress,
                            selector_config=selector_config_current
                        )
                    y_rand_result["validation_settings"] = y_rand_settings
    
                    if "y_randomization_results_dict" not in st.session_state:
                        st.session_state.y_randomization_results_dict = {}
    
                    st.session_state.y_randomization_results_dict[
                        current_model_for_y_rand
                    ] = attach_result_cache_metadata(
                        y_rand_result,
                        _validation_cache_hash(
                            current_model_for_y_rand,
                            y_rand_settings,
                        ),
                    )
    
                    log_y_randomization_result(
                        current_model_for_y_rand,
                        y_rand_result,
                    )
    
                    progress_y_rand.progress(100, text=t('y_rand.ui_progress_done'))
                    st.success(t('y_rand.ui_success'))
                    st.rerun()
    
                except Exception as e:
                    message = t('y_rand.ui_error', error=e)
                    st.error(message)
                    log_streamlit_message(
                        "Y_RANDOMIZATION",
                        message,
                        level="error",
                        details={"error": str(e), "model": current_model_for_y_rand},
                        event="y_randomization_failed",
                    )
    
            y_rand_settings_current = {
                "kind": "y_randomization",
                "method": str(y_rand_method),
                "n_permutations": int(y_rand_n_perm),
                "k": int(y_rand_k),
                "random_state": int(y_rand_seed),
            }
            y_rand_res = _current_cached_validation_result(
                "y_randomization_results_dict",
                current_model_for_y_rand,
                y_rand_settings_current,
            )
    
            if y_rand_res is not None:
                y_rand_summary = y_rand_res.get("summary", {})
                y_rand_summary_table = y_rand_res.get("summary_table", pd.DataFrame())
                y_rand_perm_table = y_rand_res.get("permutation_table", pd.DataFrame())
    
                st.markdown(t('y_rand.ui_results_title'))
    
                col_yr_m1, col_yr_m2, col_yr_m3, col_yr_m4 = st.columns(4)
    
                with col_yr_m1:
                    original_q2 = y_rand_summary.get("original_q2", np.nan)
                    if pd.notna(original_q2):
                        st.metric(t('y_rand.ui_metric_original_q2'), f"{float(original_q2):.3f}")
                    else:
                        st.metric(t('y_rand.ui_metric_original_q2'), "—")
    
                with col_yr_m2:
                    mean_q2_perm = y_rand_summary.get("mean_q2_permuted", np.nan)
                    if pd.notna(mean_q2_perm):
                        st.metric(t('y_rand.ui_metric_mean_q2'), f"{float(mean_q2_perm):.3f}")
                    else:
                        st.metric(t('y_rand.ui_metric_mean_q2'), "—")
    
                with col_yr_m3:
                    max_q2_perm = y_rand_summary.get("max_q2_permuted", np.nan)
                    if pd.notna(max_q2_perm):
                        st.metric(t('y_rand.ui_metric_max_q2'), f"{float(max_q2_perm):.3f}")
                    else:
                        st.metric(t('y_rand.ui_metric_max_q2'), "—")
    
                with col_yr_m4:
                    p_value = y_rand_summary.get("p_value", np.nan)
                    if pd.notna(p_value):
                        st.metric(t('y_rand.ui_metric_p_value'), f"{float(p_value):.4f}")
                    else:
                        st.metric(t('y_rand.ui_metric_p_value'), "—")
    
                conclusion_y_rand = str(y_rand_summary.get("conclusion", ""))
    
                if "неслучайна" in conclusion_y_rand and "не подтверждена" not in conclusion_y_rand:
                    st.success(t('y_rand.ui_conclusion', conclusion=conclusion_y_rand))
                else:
                    st.warning(t('y_rand.ui_conclusion', conclusion=conclusion_y_rand))
    
                st.dataframe(
                    y_rand_summary_table,
                    width="stretch",
                    hide_index=True
                )
    
                if isinstance(y_rand_perm_table, pd.DataFrame) and not y_rand_perm_table.empty:
                    st.markdown(t('y_rand.ui_distribution_title'))
    
                    q2_perm_plot = (
                        y_rand_perm_table[t('y_randomization.q2_perm')]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )
    
                    if len(q2_perm_plot) > 0:
                        fig_yr, ax_yr = plt.subplots(figsize=(7, 4))
    
                        safe_histplot(ax_yr, q2_perm_plot, bins=30, kde=True, color='steelblue', edgecolor='black', alpha=0.7)
    
                        if pd.notna(y_rand_summary.get("original_q2", np.nan)):
                            ax_yr.axvline(
                                float(y_rand_summary["original_q2"]),
                                linestyle="--",
                                linewidth=2,
                                label=t('y_rand.ui_plot_original_label')
                            )
    
                        ax_yr.set_xlabel(t('y_rand.ui_plot_xlabel'))
                        ax_yr.set_ylabel(t('y_rand.ui_plot_ylabel'))
                        ax_yr.set_title(t('y_rand.ui_plot_title'))
                        ax_yr.legend(fontsize=8)
                        ax_yr.grid(True, alpha=0.3)
    
                        fig_yr.tight_layout()
                        st.pyplot(fig_yr)
                        plt.close(fig_yr)
    
                    with st.expander(t('y_rand.ui_expander_table'), expanded=False):
                        st.dataframe(
                            y_rand_perm_table,
                            width="stretch",
                            hide_index=True
                        )
    
                    csv_yr = qspr_csv_download_bytes(y_rand_perm_table)
    
                    st.download_button(
                        t('y_rand.ui_download_csv'),
                        csv_yr,
                        f"y_randomization_{current_model_for_y_rand}.csv",
                        "text/csv",
                        key=f"download_y_randomization_{current_model_for_y_rand}"
                    )
    
    render_advanced_validation_section({**globals(), **locals()})

    # ------------------------------------------------------------------
