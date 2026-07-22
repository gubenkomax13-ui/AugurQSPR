# -*- coding: utf-8 -*-
"""Интерфейс автоматического сравнения QSPR-моделей."""

import numpy as np
import pandas as pd
import streamlit as st

from modules.i18n import t
from modules.consensus_ui import (
    comparison_df_with_consensus_row,
    render_comparison_consensus_section,
)
from modules.model_catalog import get_model_catalog, get_model_display_name, get_models_by_group
from modules.module_explain_ui import render_module_explanation
from modules.ui_timer import finish_elapsed_timer, show_last_elapsed_time, start_elapsed_timer


def _yrandom_p_value(result):
    """Return a finite empirical p-value, rebuilding it from permutations if needed."""
    if not isinstance(result, dict):
        return np.nan

    summary = result.get("summary", {}) or {}
    p_value = pd.to_numeric(summary.get("p_value", np.nan), errors="coerce")
    if np.isfinite(p_value):
        return float(p_value)

    original_q2 = pd.to_numeric(summary.get("original_q2", np.nan), errors="coerce")
    permutation_table = result.get("permutation_table")
    q2_column = t('y_randomization.q2_perm')
    if (
        not np.isfinite(original_q2)
        or not isinstance(permutation_table, pd.DataFrame)
        or q2_column not in permutation_table.columns
    ):
        return np.nan

    q2_values = pd.to_numeric(permutation_table[q2_column], errors="coerce")
    q2_values = q2_values.replace([np.inf, -np.inf], np.nan).dropna()
    if q2_values.empty:
        return np.nan
    return float(((q2_values >= float(original_q2)).sum() + 1) / (len(q2_values) + 1))


def _append_check_label(table, mask, label):
    """Add a completed validation name to the comparison checks column."""
    checks_column = t('comparison.checks')
    if checks_column not in table.columns:
        return
    table.loc[mask, checks_column] = (
        table.loc[mask, checks_column]
        .fillna("")
        .astype(str)
        .map(
            lambda value: (
                value
                if label in value
                else f"{value}, {label}".strip(", ")
            )
        )
    )


def _cache_comparison_validation_result(
    result,
    model_name,
    settings,
    X,
    y,
    valid_indices,
):
    """Store comparison output in the same cache format as single-model validation."""
    cached_result = dict(result or {})
    cached_result["validation_settings"] = dict(settings)
    config_hash = analysis_result_hash(
        st.session_state,
        model_name,
        params=get_model_params_from_session(),
        validation_settings=settings,
        X=X,
        y=y,
        desc_names=st.session_state.get("desc_names"),
        valid_indices=valid_indices,
    )
    return attach_result_cache_metadata(cached_result, config_hash)


def render_model_comparison_section(context):
    """Рендерит сравнение моделей в переданном контексте проекта."""
    globals().update(context)
    if (
        "post_descriptor_training_mode" in st.session_state
        and st.session_state.post_descriptor_training_mode != "compare"
    ):
        return
    # ------------------------------------------------------------------
    # ================================================================
    # True automatic model comparison
    
    st.header(t('model_comparison.header'))
    render_module_explanation("model_comparison")
    
    st.markdown(t('model_comparison.description'))
    
    try:
        all_model_groups_for_compare = qspr_core.qspr_available_model_options()
    except Exception:
        all_model_groups_for_compare = get_models_by_group(include_unavailable=False)
    
    all_models_for_compare = []
    for _group_name, _models in all_model_groups_for_compare.items():
        for _m in _models:
            if _m not in all_models_for_compare:
                all_models_for_compare.append(_m)

    try:
        online_light_mode = bool(qspr_is_online_mode())
    except Exception:
        try:
            online_light_mode = bool(qspr_is_streamlit_cloud_runtime())
        except Exception:
            online_light_mode = False

    requested_default_models = [
        "linear_regression",
        "pls_regression",
        "random_forest",
        "svr",
        "hist_gradient_boosting",
        "ridge_regression",
        "lasso_regression",
        "elastic_net",
    ]

    online_default_models = [
        "linear_regression",
        "pls_regression",
        "ridge_regression",
        "lasso_regression",
        "elastic_net",
    ]

    safe_default_models = requested_default_models
    if online_light_mode:
        safe_default_models = online_default_models
    
    safe_default_models = [m for m in safe_default_models if m in all_models_for_compare]
    compare_models_widget_key = "auto_compare_selected_models_v3"
    if compare_models_widget_key not in st.session_state:
        legacy_selection = st.session_state.get("auto_compare_selected_models")
        if isinstance(legacy_selection, list) and legacy_selection:
            initial_selection = [
                model_name
                for model_name in legacy_selection
                if model_name in all_models_for_compare
            ]
        else:
            initial_selection = []
        st.session_state[compare_models_widget_key] = (
            initial_selection or list(safe_default_models)
        )
    
    if not all_models_for_compare:
        trained_models = st.session_state.get("trained_models", {}) or {}
        if trained_models:
            st.info(t('model_comparison.no_models_warning'))
        else:
            st.warning(t('model_comparison.no_models_warning'))

    if all_models_for_compare:
        if online_light_mode:
            st.info(
                t("model_comparison.online_light_info")
            )
            cloud_widget_limits = {
                "cmp_loo_top_n": 5,
                "cmp_mc_top_n": 5,
                "cmp_bootstrap_top_n": 5,
                "cmp_yrandom_top_n": 5,
                "cmp_mc_repeats": 50,
                "cmp_bootstrap_repeats": 50,
                "cmp_yrandom_repeats": 50,
            }

            for state_key, max_value in cloud_widget_limits.items():
                try:
                    current_value = int(st.session_state.get(state_key, max_value))
                except (TypeError, ValueError):
                    current_value = max_value

                if current_value > max_value:
                    st.session_state[state_key] = max_value

        with st.expander(t('model_comparison.settings_expander'), expanded=True):
            st.caption(t(
                'model_comparison.available_models_count',
                available=len(all_models_for_compare),
                total=len(get_model_catalog()),
            ))

            selected_compare_models = st.multiselect(
                t('model_comparison.candidates_label'),
                options=all_models_for_compare,
                default=safe_default_models,
                key=compare_models_widget_key,
                help=t('model_comparison.candidates_help'),
                format_func=get_model_display_name,
            )

            def restore_recommended_models():
                st.session_state[compare_models_widget_key] = list(safe_default_models)

            st.button(
                t('model_comparison.restore_recommended_models'),
                key="restore_recommended_compare_models",
                on_click=restore_recommended_models,
            )

            st.caption(t(
                'model_comparison.selected_models_count',
                selected=len(selected_compare_models),
                total=len(all_models_for_compare),
            ))
    
            col_cmp_opts_1, col_cmp_opts_2, col_cmp_opts_3, col_cmp_opts_4 = st.columns(4)
    
            with col_cmp_opts_1:
                cmp_run_holdout = st.checkbox(
                    t('model_comparison.run_holdout'),
                    value=True,
                    key="cmp_run_holdout"
                )
    
            with col_cmp_opts_2:
                cmp_run_kfold = st.checkbox(
                    t('model_comparison.run_kfold'),
                    value=True,
                    key="cmp_run_kfold"
                )
    
            with col_cmp_opts_3:
                cmp_run_loo = st.checkbox(
                    t('model_comparison.run_loo'),
                    value=False,
                    key="cmp_run_loo",
                    help=t('model_comparison.run_loo_help')
                )
    
            with col_cmp_opts_4:
                cmp_force_retrain = st.checkbox(
                    t('model_comparison.force_retrain'),
                    value=True,
                    key="cmp_force_retrain",
                    help=t('model_comparison.force_retrain_help')
                )
                
            col_cmp_cfg_1, col_cmp_cfg_2 = st.columns(2)
    
            with col_cmp_cfg_1:
                default_kfold_k = 3 if online_light_mode else min(5, max(3, min(10, len(y_all_current))))
                cmp_kfold_k = st.slider(
                    t('model_comparison.kfold_k_label'),
                    min_value=3,
                    max_value=max(3, min(10, len(y_all_current))),
                    value=default_kfold_k,
                    step=1,
                    key="cmp_kfold_k"
                )
    
            with col_cmp_cfg_2:
                cmp_holdout_test_size_percent = st.slider(
                    t('model_comparison.holdout_test_size_label'),
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    key="cmp_holdout_test_size_percent"
                )
    
            st.caption(t('model_comparison.recommendation_caption'))
            st.markdown(t('model_comparison.extra_checks_title'))

            extra_top_max = max(
                1,
                min(5 if online_light_mode else 10, len(selected_compare_models)),
            )
            for top_n_key in (
                "cmp_loo_top_n",
                "cmp_mc_top_n",
                "cmp_bootstrap_top_n",
                "cmp_yrandom_top_n",
            ):
                if int(st.session_state.get(top_n_key, 1)) > extra_top_max:
                    st.session_state[top_n_key] = extra_top_max
            # One check per row: this keeps its repeats and top-N setting
            # visually connected instead of splitting them across columns.
            head_check, head_repeats, head_top_n = st.columns([2, 1, 1])
            with head_check:
                st.caption("Настройки")
            with head_repeats:
                st.caption("Настройка повторов")
            with head_top_n:
                st.caption("Проверять лучших моделей")

            row_loo, row_loo_repeats, row_loo_top_n = st.columns([2, 1, 1])
            with row_loo:
                cmp_run_loo_top = st.checkbox(
                    t('model_comparison.loo_top_checkbox'),
                    value=not online_light_mode,
                    key="cmp_run_loo_top",
                )
            with row_loo_repeats:
                st.caption("Один запуск")
            with row_loo_top_n:
                cmp_loo_top_n = st.number_input(
                    t('model_comparison.top_n_per_check_label'),
                    min_value=1,
                    max_value=extra_top_max,
                    value=1,
                    step=1,
                    key="cmp_loo_top_n",
                    disabled=not cmp_run_loo_top,
                )

            row_mc, row_mc_repeats, row_mc_top_n = st.columns([2, 1, 1])
            with row_mc:
                cmp_run_montecarlo_top = st.checkbox(
                    t('model_comparison.montecarlo_top_checkbox'),
                    value=False,
                    key="cmp_run_montecarlo_top",
                )
            with row_mc_repeats:
                cmp_mc_repeats = st.number_input(
                    t('model_comparison.mc_repeats_label'),
                    min_value=10,
                    max_value=50 if online_light_mode else 500,
                    value=10,
                    step=10,
                    key="cmp_mc_repeats",
                    disabled=not cmp_run_montecarlo_top,
                )
            with row_mc_top_n:
                cmp_mc_top_n = st.number_input(
                    t('model_comparison.top_n_per_check_label'),
                    min_value=1,
                    max_value=extra_top_max,
                    value=1,
                    step=1,
                    key="cmp_mc_top_n",
                    disabled=not cmp_run_montecarlo_top,
                )

            row_bs, row_bs_repeats, row_bs_top_n = st.columns([2, 1, 1])
            with row_bs:
                cmp_run_bootstrap_top = st.checkbox(
                    t('model_comparison.bootstrap_top_checkbox'),
                    value=False,
                    key="cmp_run_bootstrap_top",
                )
            with row_bs_repeats:
                cmp_bootstrap_repeats = st.number_input(
                    t('model_comparison.bootstrap_repeats_label'),
                    min_value=10,
                    max_value=50 if online_light_mode else 500,
                    value=10,
                    step=10,
                    key="cmp_bootstrap_repeats",
                    disabled=not cmp_run_bootstrap_top,
                )
            with row_bs_top_n:
                cmp_bootstrap_top_n = st.number_input(
                    t('model_comparison.top_n_per_check_label'),
                    min_value=1,
                    max_value=extra_top_max,
                    value=1,
                    step=1,
                    key="cmp_bootstrap_top_n",
                    disabled=not cmp_run_bootstrap_top,
                )

            row_yr, row_yr_repeats, row_yr_top_n = st.columns([2, 1, 1])
            with row_yr:
                cmp_run_yrandom_top = st.checkbox(
                    t('model_comparison.yrandom_top_checkbox'),
                    value=False,
                    key="cmp_run_yrandom_top",
                )
            with row_yr_repeats:
                cmp_yrandom_repeats = st.number_input(
                    t('model_comparison.yrandom_repeats_label'),
                    min_value=10,
                    max_value=50 if online_light_mode else 1000,
                    value=10,
                    step=10,
                    key="cmp_yrandom_repeats",
                    disabled=not cmp_run_yrandom_top,
                )
            with row_yr_top_n:
                cmp_yrandom_top_n = st.number_input(
                    t('model_comparison.top_n_per_check_label'),
                    min_value=1,
                    max_value=extra_top_max,
                    value=1,
                    step=1,
                    key="cmp_yrandom_top_n",
                    disabled=not cmp_run_yrandom_top,
                )

            if online_light_mode:
                heavy_selected = [
                    get_model_display_name(model_name)
                    for model_name in selected_compare_models
                    if model_name not in online_default_models
                ]

                if heavy_selected:
                    st.warning(
                        t("model_comparison.heavy_models_warning", models=", ".join(heavy_selected))
                    )

                if len(selected_compare_models) > 5:
                    st.warning(
                        t("model_comparison.online_limit_warning")
                    )
                    selected_compare_models = selected_compare_models[:5]
            
    
        show_last_elapsed_time(
            "model_comparison",
            t('timing.last_model_comparison'),
        )

        if st.button(t('model_comparison.run_button'), type="primary", key="train_compare_and_choose_models"):
            if not selected_compare_models:
                st.warning(t('model_comparison.warning_select_models'))
            else:
                # Use the same canonical validation state that the single-model
                # screen will receive later, so cached results remain compatible.
                st.session_state.holdout_test_size = int(cmp_holdout_test_size_percent)
                st.session_state.holdout_random = True
                st.session_state.holdout_rs = 42
                st.session_state.holdout_stratify_y_quantiles = False
                st.session_state.kfold_k = int(cmp_kfold_k)
                timer_started_at, timer_placeholder = start_elapsed_timer(
                    "model_comparison",
                    t('timing.model_comparison_in_progress'),
                )
                timer_finished = False
                try:
                    st.session_state.yrandom_results_dict = {}
                    st.session_state.yrandom_best_model_result = None
                    st.session_state.yrandom_best_model_name = None

                    smiles_for_compare = data[smiles_col_current].iloc[valid_indices_current].values.tolist()
    
                    progress = st.progress(0, text=t('model_comparison.progress_preparing'))
    
                    with st.spinner(t('model_comparison.spinner_training')):
    
                        if (
                            cmp_run_loo_top
                            or cmp_run_montecarlo_top
                            or cmp_run_bootstrap_top
                            or cmp_run_yrandom_top
                        ):
                            # ------------------------------------------------------------
                            # Этап 1: все модели, но без LOO
    
                            progress.progress(10, text=t('model_comparison.progress_stage1'))

                            def stage1_model_progress(model_name, model_index, model_total):
                                progress.progress(
                                    10 + int(40 * (model_index - 1) / max(1, model_total)),
                                    text=t(
                                        'model_comparison.progress_training_model',
                                        model=get_model_display_name(model_name),
                                        idx=model_index,
                                        total=model_total,
                                    ),
                                )
    
                            messages, error_df = qspr_auto_train_validate_models_for_comparison(
                                model_names=selected_compare_models,
                                X=X_all_current,
                                y=y_all_current,
                                valid_indices=valid_indices_current,
                                smiles=smiles_for_compare,
                                target_col=target_col,
                                run_holdout=cmp_run_holdout,
                                run_kfold=cmp_run_kfold,
                                run_loo=False,
                                holdout_test_size=cmp_holdout_test_size_percent / 100.0,
                                holdout_random_state=42,
                                kfold_k=cmp_kfold_k,
                                force_retrain=cmp_force_retrain,
                                progress_callback=stage1_model_progress,
                            )
    
                            progress.progress(50, text=t('model_comparison.progress_ranking'))
    
                            preliminary_df = qspr_build_model_comparison_table()
    
                            preliminary_df = preliminary_df[
                                preliminary_df[t('comparison.model')].isin(selected_compare_models)
                            ].copy()
    
                            preliminary_df = preliminary_df.sort_values(
                                by=[t('comparison.rating'), "K-Fold Q²", "Hold-out R²"],
                                ascending=[True, False, False],
                                na_position="last",
                            ).reset_index(drop=True)
    
                            ranked_models = (
                                preliminary_df[t('comparison.model')]
                                .astype(str)
                                .tolist()
                            )
                            loo_models = (
                                ranked_models
                                if cmp_run_loo
                                else ranked_models[:int(cmp_loo_top_n)] if cmp_run_loo_top else []
                            )
                            mc_models = ranked_models[:int(cmp_mc_top_n)] if cmp_run_montecarlo_top else []
                            bootstrap_models = (
                                ranked_models[:int(cmp_bootstrap_top_n)]
                                if cmp_run_bootstrap_top
                                else []
                            )
                            yrandom_models = (
                                ranked_models[:int(cmp_yrandom_top_n)]
                                if cmp_run_yrandom_top
                                else []
                            )

                            check_selections = []
                            for check_label, check_models in [
                                ("LOO", loo_models),
                                ("Monte-Carlo", mc_models),
                                ("Bootstrap", bootstrap_models),
                                ("Y-randomization", yrandom_models),
                            ]:
                                if check_models:
                                    check_selections.append(
                                        f"{check_label}: "
                                        + ", ".join(
                                            get_model_display_name(name)
                                            for name in check_models
                                        )
                                    )
                            if check_selections:
                                st.info(t(
                                    'model_comparison.top_models_by_check_info',
                                    checks="; ".join(check_selections),
                                ))
    
                            # ------------------------------------------------------------
                            # Этап 2: LOO только для top-N
    
                            messages_loo, error_df_loo = [], None
                            if loo_models:
                                progress.progress(65, text=t('model_comparison.progress_stage2'))

                                def stage2_model_progress(model_name, model_index, model_total):
                                    progress.progress(
                                        65,
                                        text=t(
                                            'model_comparison.progress_loo_model',
                                            model=get_model_display_name(model_name),
                                            idx=model_index,
                                            total=model_total,
                                        ),
                                    )

                                messages_loo, error_df_loo = qspr_auto_train_validate_models_for_comparison(
                                    model_names=loo_models,
                                    X=X_all_current,
                                    y=y_all_current,
                                    valid_indices=valid_indices_current,
                                    smiles=smiles_for_compare,
                                    target_col=target_col,
                                    run_holdout=False,
                                    run_kfold=False,
                                    run_loo=True,
                                    holdout_test_size=cmp_holdout_test_size_percent / 100.0,
                                    holdout_random_state=42,
                                    kfold_k=cmp_kfold_k,
                                    force_retrain=False,
                                    progress_callback=stage2_model_progress,
                                )
    
                            messages = list(messages or []) + list(messages_loo or [])
                            
                            # ------------------------------------------------------------
                            # Monte-Carlo CV для лучших моделей (с обновлением прогресса)
                            if cmp_run_montecarlo_top:
                                montecarlo_results = {}
                                total_mc = len(mc_models)
                                for idx, model_name_mc in enumerate(mc_models, 1):
                                    def mc_repeat_progress(done, repeat_total, model_index=idx, model_name=model_name_mc):
                                        overall_done = (model_index - 1) * repeat_total + done
                                        overall_total = max(1, total_mc * repeat_total)
                                        progress.progress(
                                            65 + int(15 * overall_done / overall_total),
                                            text=t(
                                                'model_comparison.progress_mc_repeat',
                                                model=get_model_display_name(model_name),
                                                model_idx=model_index,
                                                model_total=total_mc,
                                                repeat=done,
                                                repeat_total=repeat_total,
                                            ),
                                        )
                                    mc_settings = {
                                        "kind": "repeated_holdout",
                                        "n_repeats": int(cmp_mc_repeats),
                                        "test_size": cmp_holdout_test_size_percent / 100.0,
                                        "random_state": 42,
                                        "save_details": True,
                                    }
                                    mc_result = qspr_repeated_holdout_validation(
                                        X=X_all_current,
                                        y=y_all_current,
                                        model_name=model_name_mc,
                                        valid_indices=valid_indices_current,
                                        smiles=smiles_for_compare,
                                        target_col=target_col,
                                        params=get_model_params_from_session(),
                                        n_repeats=int(cmp_mc_repeats),
                                        test_size=cmp_holdout_test_size_percent / 100.0,
                                        random_state=42,
                                        scale=True,
                                        progress_callback=mc_repeat_progress
                                    )
                                    montecarlo_results[model_name_mc] = (
                                        _cache_comparison_validation_result(
                                            mc_result,
                                            model_name_mc,
                                            mc_settings,
                                            X_all_current,
                                            y_all_current,
                                            valid_indices_current,
                                        )
                                    )
                                st.session_state.montecarlo_results_dict = montecarlo_results
                                st.session_state.repeated_holdout_results_dict = dict(montecarlo_results)
                                
                            # ------------------------------------------------------------
                            # Bootstrap CV для лучших моделей (с обновлением прогресса)
                            if cmp_run_bootstrap_top:
                                bootstrap_results = {}
                                total_bs = len(bootstrap_models)
                                for idx, model_name_bs in enumerate(bootstrap_models, 1):
                                    def bootstrap_repeat_progress(done, repeat_total, model_index=idx, model_name=model_name_bs):
                                        overall_done = (model_index - 1) * repeat_total + done
                                        overall_total = max(1, total_bs * repeat_total)
                                        progress.progress(
                                            80 + int(10 * overall_done / overall_total),
                                            text=t(
                                                'model_comparison.progress_bs_repeat',
                                                model=get_model_display_name(model_name),
                                                model_idx=model_index,
                                                model_total=total_bs,
                                                repeat=done,
                                                repeat_total=repeat_total,
                                            ),
                                        )
                                    bootstrap_settings = {
                                        "kind": "bootstrap",
                                        "n_iterations": int(cmp_bootstrap_repeats),
                                        "sample_fraction": 1.0,
                                        "random_state": 42,
                                        "save_oob_predictions": True,
                                    }
                                    bootstrap_result = qspr_bootstrap_validation(
                                        X=X_all_current,
                                        y=y_all_current,
                                        model_name=model_name_bs,
                                        valid_indices=valid_indices_current,
                                        smiles=smiles_for_compare,
                                        target_col=target_col,
                                        params=get_model_params_from_session(),
                                        n_iterations=int(cmp_bootstrap_repeats),
                                        random_state=42,
                                        scale=True,
                                        progress_callback=bootstrap_repeat_progress
                                    )
                                    bootstrap_results[model_name_bs] = (
                                        _cache_comparison_validation_result(
                                            bootstrap_result,
                                            model_name_bs,
                                            bootstrap_settings,
                                            X_all_current,
                                            y_all_current,
                                            valid_indices_current,
                                        )
                                    )
                                st.session_state.bootstrap_results_dict = bootstrap_results  
    
                            # ------------------------------------------------------------
                            # Y-randomization for every selected top-N model.
                            if cmp_run_yrandom_top and yrandom_models:
                                yrandom_results = {}
                                total_yr = len(yrandom_models)
                                for idx, model_name_yr in enumerate(yrandom_models, 1):
                                    def yrandom_permutation_progress(done, permutation_total, model_index=idx, model_name=model_name_yr):
                                        overall_done = (model_index - 1) * permutation_total + done
                                        overall_total = max(1, total_yr * permutation_total)
                                        progress.progress(
                                            90 + int(4 * overall_done / overall_total),
                                            text=t(
                                                'model_comparison.progress_yr_permutation',
                                                model=get_model_display_name(model_name),
                                                model_idx=model_index,
                                                model_total=total_yr,
                                                permutation=done,
                                                permutation_total=permutation_total,
                                            ),
                                        )
                                    yrandom_settings = {
                                        "kind": "y_randomization",
                                        "method": "kfold",
                                        "n_permutations": int(cmp_yrandom_repeats),
                                        "k": int(cmp_kfold_k),
                                        "random_state": 42,
                                    }
                                    yrandom_result = qspr_y_randomization_test(
                                        X=X_all_current,
                                        y=y_all_current,
                                        model_name=model_name_yr,
                                        valid_indices=valid_indices_current,
                                        smiles=smiles_for_compare,
                                        params=get_model_params_from_session(),
                                        method="kfold",
                                        n_permutations=int(cmp_yrandom_repeats),
                                        k=cmp_kfold_k,
                                        random_state=42,
                                        scale=True,
                                        progress_callback=yrandom_permutation_progress
                                    )
                                    yrandom_result = _cache_comparison_validation_result(
                                        yrandom_result,
                                        model_name_yr,
                                        yrandom_settings,
                                        X_all_current,
                                        y_all_current,
                                        valid_indices_current,
                                    )
                                    yrandom_results[model_name_yr] = yrandom_result

                                    if idx == 1:
                                        st.session_state.yrandom_best_model_result = yrandom_result
                                        st.session_state.yrandom_best_model_name = model_name_yr

                                    if not np.isfinite(_yrandom_p_value(yrandom_result)):
                                        yr_summary = yrandom_result.get("summary", {}) or {}
                                        st.warning(t(
                                            'model_comparison.yrandom_no_result_warning',
                                            model=get_model_display_name(model_name_yr),
                                            successful=yr_summary.get("n_permutations_successful", 0),
                                            failed=yr_summary.get("n_permutations_failed", int(cmp_yrandom_repeats)),
                                        ))
                                st.session_state.yrandom_results_dict = yrandom_results
                                st.session_state.y_randomization_results_dict = dict(yrandom_results)
    
                            if error_df is not None and error_df_loo is not None:
                                error_df = pd.concat([error_df, error_df_loo], ignore_index=True)
                            elif error_df is None:
                                error_df = error_df_loo
    
                        else:
                            # Старый полный режим
                            def full_model_progress(model_name, model_index, model_total):
                                progress.progress(
                                    10 + int(80 * (model_index - 1) / max(1, model_total)),
                                    text=t(
                                        'model_comparison.progress_training_model',
                                        model=get_model_display_name(model_name),
                                        idx=model_index,
                                        total=model_total,
                                    ),
                                )

                            messages, error_df = qspr_auto_train_validate_models_for_comparison(
                                model_names=selected_compare_models,
                                X=X_all_current,
                                y=y_all_current,
                                valid_indices=valid_indices_current,
                                smiles=smiles_for_compare,
                                target_col=target_col,
                                run_holdout=cmp_run_holdout,
                                run_kfold=cmp_run_kfold,
                                run_loo=cmp_run_loo,
                                holdout_test_size=cmp_holdout_test_size_percent / 100.0,
                                holdout_random_state=42,
                                kfold_k=cmp_kfold_k,
                                force_retrain=cmp_force_retrain,
                                progress_callback=full_model_progress,
                            )
    
                        # ------------------------------------------------------------
                        # Формирование финальной таблицы сравнения
                        progress.progress(95, text=t('model_comparison.progress_final_table'))
                        comparison_df = qspr_build_model_comparison_table()
    
                        # Добавление столбцов для дополнительных проверок
                        for col in [t('comparison.mc_test_r2'), t('comparison.bootstrap_r2_oob'), t('comparison.yrand_pvalue')]:
                            if col not in comparison_df.columns:
                                comparison_df[col] = np.nan
    
                        # Monte-Carlo CV
                        if st.session_state.get("cmp_run_montecarlo_top", False) and "montecarlo_results_dict" in st.session_state:
                            mc_dict = st.session_state.montecarlo_results_dict
                            for model_name_mc, res_mc in mc_dict.items():
                                test_r2_mean = res_mc.get("test_r2_mean", np.nan)
                                mask = comparison_df[t('comparison.model')] == model_name_mc
                                comparison_df.loc[mask, t('comparison.mc_test_r2')] = test_r2_mean
                                if np.isfinite(pd.to_numeric(test_r2_mean, errors="coerce")):
                                    _append_check_label(comparison_df, mask, "Monte-Carlo CV")
    
                        # Bootstrap CV
                        if st.session_state.get("cmp_run_bootstrap_top", False) and "bootstrap_results_dict" in st.session_state:
                            boot_dict = st.session_state.bootstrap_results_dict
                            for model_name_bs, res_bs in boot_dict.items():
                                r2_oob_mean = res_bs.get("summary", {}).get("r2_oob_mean", np.nan)
                                mask = comparison_df[t('comparison.model')] == model_name_bs
                                comparison_df.loc[mask, t('comparison.bootstrap_r2_oob')] = r2_oob_mean
                                if np.isfinite(pd.to_numeric(r2_oob_mean, errors="coerce")):
                                    _append_check_label(comparison_df, mask, "Bootstrap CV")

                        # Y-randomization p-value for every model with a stored result.
                        yrandom_results = st.session_state.get("yrandom_results_dict", {}) or {}
                        if not yrandom_results:
                            legacy_result = st.session_state.get("yrandom_best_model_result")
                            legacy_name = st.session_state.get("yrandom_best_model_name")
                            if isinstance(legacy_result, dict) and legacy_name:
                                yrandom_results = {legacy_name: legacy_result}

                        for yrand_model_name, yrand_res in yrandom_results.items():
                            p_val = _yrandom_p_value(yrand_res)
                            mask = comparison_df[t('comparison.model')].astype(str).str.strip() == str(yrand_model_name).strip()
                            comparison_df.loc[mask, t('comparison.yrand_pvalue')] = p_val
                            if np.isfinite(p_val):
                                _append_check_label(comparison_df, mask, "Y-randomization")
    
                        # Переупорядочивание столбцов (новые после LOO Q²)
                        cols = comparison_df.columns.tolist()
                        if "LOO Q²" in cols:
                            loo_pos = cols.index("LOO Q²")
                            new_cols = cols[:loo_pos+1] + [t('comparison.mc_test_r2'), t('comparison.bootstrap_r2_oob'), t('comparison.yrand_pvalue')] + [c for c in cols[loo_pos+1:] if c not in [t('comparison.mc_test_r2'), t('comparison.bootstrap_r2_oob'), t('comparison.yrand_pvalue')]]
                            comparison_df = comparison_df[new_cols]
    
                        progress.progress(100, text=t('model_comparison.progress_done'))
    
                        if error_df is not None and not error_df.empty:
                            st.session_state.model_comparison_errors_df = error_df
    
                        if comparison_df.empty:
                            st.warning(t('model_comparison.warning_no_comparison_table'))
                        else:
                            # Оставляем в таблице только выбранные кандидаты
                            comparison_df = comparison_df[comparison_df[t('comparison.model')].isin(selected_compare_models)].copy()
                            comparison_df = comparison_df.sort_values(
                                by=[t('comparison.rating'), "K-Fold Q²", "LOO Q²", "Hold-out R²"],
                                ascending=[True, False, False, False],
                                na_position="last"
                            ).reset_index(drop=True)
                            if t('comparison.place') in comparison_df.columns:
                                comparison_df = comparison_df.drop(columns=[t('comparison.place')])
                            comparison_df.insert(0, t('comparison.place'), range(1, len(comparison_df) + 1))
    
                            st.session_state.model_comparison_df = comparison_df
                            best_model = str(comparison_df.iloc[0][t('comparison.model')])
                            st.session_state.best_model_from_comparison = best_model
                            st.session_state.pending_selected_model = best_model
                            st.session_state.pending_comparison_validation_handoff = {
                                "model": best_model,
                                "holdout_enabled": bool(cmp_run_holdout),
                                "holdout_test_percent": int(cmp_holdout_test_size_percent),
                                "holdout_random_state": 42,
                                "kfold_enabled": bool(cmp_run_kfold),
                                "kfold_k": int(cmp_kfold_k),
                                "loo_enabled": bool(cmp_run_loo or cmp_run_loo_top),
                                "montecarlo_enabled": bool(cmp_run_montecarlo_top),
                                "montecarlo_repeats": int(cmp_mc_repeats),
                                "bootstrap_enabled": bool(cmp_run_bootstrap_top),
                                "bootstrap_repeats": int(cmp_bootstrap_repeats),
                                "yrandom_enabled": bool(cmp_run_yrandom_top),
                                "yrandom_repeats": int(cmp_yrandom_repeats),
                            }
                            add_log(t('model_comparison.log_best_model', model=best_model))
    
                            if messages:
                                st.success(t('model_comparison.success_models_trained', count=len(set(selected_compare_models))))
    
                            st.success(t('model_comparison.success_best_model', model=best_model))
                            finish_elapsed_timer(
                                "model_comparison",
                                timer_started_at,
                                timer_placeholder,
                                t('timing.last_model_comparison'),
                            )
                            timer_finished = True
                            st.rerun()
    
                except Exception as e:
                    st.error(t('model_comparison.error_comparison', error=e))
                finally:
                    if not timer_finished:
                        finish_elapsed_timer(
                            "model_comparison",
                            timer_started_at,
                            timer_placeholder,
                            t('timing.last_model_comparison'),
                        )
    
        comparison_df = st.session_state.get("model_comparison_df")
    
        if comparison_df is None or not isinstance(comparison_df, pd.DataFrame) or comparison_df.empty:
            comparison_df = qspr_build_model_comparison_table()
    
        if comparison_df is not None and isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
            comparison_table_title = t('comparison.table_title')
            if comparison_table_title == '!comparison.table_title!':
                comparison_table_title = {
                    "ru": "Сравнение моделей",
                    "en": "Model comparison",
                    "kk": "Модельдерді салыстыру",
                }.get(st.session_state.get("lang", "ru"), "Сравнение моделей")
    
            st.subheader(comparison_table_title)
    
            display_cmp = comparison_df_with_consensus_row(comparison_df).copy()
            st.session_state.model_comparison_with_consensus_df = display_cmp.copy()
            model_column = t('comparison.model')
            if model_column in display_cmp.columns:
                display_cmp[model_column] = display_cmp[model_column].map(
                    get_model_display_name
                )

            # Keep completed extra checks next to the core CV metrics and omit
            # disabled checks whose columns contain no results.
            extra_check_columns = [
                t('comparison.yrand_pvalue'),
                t('comparison.mc_test_r2'),
                t('comparison.bootstrap_r2_oob'),
            ]
            empty_extra_columns = [
                col
                for col in extra_check_columns
                if col in display_cmp.columns
                and pd.to_numeric(display_cmp[col], errors="coerce").notna().sum() == 0
            ]
            if empty_extra_columns:
                display_cmp = display_cmp.drop(columns=empty_extra_columns)

            visible_extra_columns = [
                col for col in extra_check_columns if col in display_cmp.columns
            ]
            if "LOO Q²" in display_cmp.columns and visible_extra_columns:
                base_columns = [
                    col for col in display_cmp.columns if col not in visible_extra_columns
                ]
                loo_position = base_columns.index("LOO Q²") + 1
                ordered_columns = (
                    base_columns[:loo_position]
                    + visible_extra_columns
                    + base_columns[loo_position:]
                )
                display_cmp = display_cmp[ordered_columns]
    
            numeric_cols = [
                "Train R²",
                "Hold-out R²",
                "K-Fold Q²",
                "LOO Q²",
                "RMSE",
                "MAE",
                "Вне AD, %",
                "Рейтинг",
                t('comparison.yrand_pvalue'),
            ]
    
            for col in numeric_cols:
                if col in display_cmp.columns:
                    display_cmp[col] = pd.to_numeric(display_cmp[col], errors="coerce").round(4)
    
            st.dataframe(
                display_cmp,
                width="stretch",
                hide_index=True
            )
    
            best_model = str(comparison_df.iloc[0][t('comparison.model')])
            best_comment = str(comparison_df.iloc[0].get(t('comparison.comment'), ""))
    
            col_best_1, col_best_2, col_best_3, col_best_4 = st.columns(4)
    
            with col_best_1:
                st.metric(
                    t('comparison.best_model_metric'),
                    get_model_display_name(best_model)
                )
    
            with col_best_2:
                st.metric("K-Fold Q²", f"{comparison_df.iloc[0].get('K-Fold Q²', np.nan):.3f}" if pd.notna(comparison_df.iloc[0].get('K-Fold Q²', np.nan)) else t('comparison.not_available'))
    
            with col_best_3:
                st.metric("LOO Q²", f"{comparison_df.iloc[0].get('LOO Q²', np.nan):.3f}" if pd.notna(comparison_df.iloc[0].get('LOO Q²', np.nan)) else t('comparison.not_available'))
    
            with col_best_4:
                st.metric("RMSE", f"{comparison_df.iloc[0].get('RMSE', np.nan):.3f}" if pd.notna(comparison_df.iloc[0].get('RMSE', np.nan)) else t('comparison.not_available'))
    
            st.info(t('comparison.best_comment', comment=best_comment))
    
            csv_cmp = display_cmp.to_csv(index=False).encode("utf-8")
            st.download_button(
                t('comparison.download_csv'),
                csv_cmp,
                "model_comparison.csv",
                "text/csv",
                key="download_model_comparison_csv"
            )
    
            error_df = st.session_state.get("model_comparison_errors_df")
            if isinstance(error_df, pd.DataFrame) and not error_df.empty:
                with st.expander(t('comparison.errors_expander'), expanded=False):
                    st.dataframe(error_df, width="stretch", hide_index=True)

            # Consensus belongs to comparison: it aggregates the same OOF
            # predictions that were produced for model selection.
            render_comparison_consensus_section({**globals(), **locals()})

            st.subheader("Продолжение анализа")
            has_consensus = isinstance(st.session_state.get("consensus_result"), dict)
            final_choice_options = ["Лучшая отдельная модель"]
            if has_consensus:
                final_choice_options.append("Консенсус моделей (OOF)")
            final_choice = st.radio(
                "Итоговое решение сравнения",
                options=final_choice_options,
                horizontal=True,
                key="comparison_final_choice",
            )
            st.session_state.final_analysis_choice = {
                "kind": "consensus" if final_choice.startswith("Консенсус") else "single_model",
                "model": best_model,
                "consensus": st.session_state.get("consensus_result") if final_choice.startswith("Консенсус") else None,
            }
            st.caption(
                "Диагностика консенсуса — его OOF-прогнозы, остатки и разброс — "
                "показана выше. Для диагностики дескрипторов, SHAP и сохранения "
                "выбирается одна модель-участник."
            )
            continuation_options = comparison_df[t('comparison.model')].astype(str).tolist()
            continuation_model = st.selectbox(
                "Модель для последующей диагностики",
                options=continuation_options,
                index=continuation_options.index(best_model) if best_model in continuation_options else 0,
                format_func=get_model_display_name,
                key="comparison_continue_model",
            )
    
            def switch_to_best_model():
                selected_model = str(st.session_state.get("comparison_continue_model", best_model))
                # Keep the single-model controls aligned with the settings
                # already used by the comparison run.
                st.session_state.holdout_test_size = int(cmp_holdout_test_size_percent)
                st.session_state.holdout_random = True
                st.session_state.holdout_rs = 42
                st.session_state.holdout_stratify_y_quantiles = False
                st.session_state.kfold_k = int(cmp_kfold_k)

                if cmp_run_montecarlo_top:
                    st.session_state.repeated_holdout_n = int(cmp_mc_repeats)
                    st.session_state.repeated_holdout_test_percent = int(cmp_holdout_test_size_percent)
                    st.session_state.repeated_holdout_seed = 42
                    st.session_state.repeated_holdout_save_details = True
                    mc_result = st.session_state.get("montecarlo_results_dict", {}).get(selected_model)
                    if isinstance(mc_result, dict):
                        st.session_state.setdefault("repeated_holdout_results_dict", {})[
                            selected_model
                        ] = mc_result

                if cmp_run_bootstrap_top:
                    st.session_state.bootstrap_n_iterations = int(cmp_bootstrap_repeats)
                    st.session_state.bootstrap_sample_percent = 100
                    st.session_state.bootstrap_seed = 42
                    st.session_state.bootstrap_save_oob_predictions = True
                    bootstrap_result = st.session_state.get("bootstrap_results_dict", {}).get(selected_model)
                    if isinstance(bootstrap_result, dict) and "config_hash" not in bootstrap_result:
                        bootstrap_settings = {
                            "kind": "bootstrap",
                            "n_iterations": int(cmp_bootstrap_repeats),
                            "sample_fraction": 1.0,
                            "random_state": 42,
                            "save_oob_predictions": True,
                        }
                        st.session_state.bootstrap_results_dict[selected_model] = (
                            _cache_comparison_validation_result(
                                bootstrap_result,
                                selected_model,
                                bootstrap_settings,
                                X_all_current,
                                y_all_current,
                                valid_indices_current,
                            )
                        )

                if cmp_run_yrandom_top:
                    st.session_state.y_randomization_method = "kfold"
                    st.session_state.y_randomization_n_perm = int(cmp_yrandom_repeats)
                    st.session_state.y_randomization_k = int(cmp_kfold_k)
                    st.session_state.y_randomization_seed = 42
                    yr_result = st.session_state.get("yrandom_results_dict", {}).get(selected_model)
                    if isinstance(yr_result, dict):
                        if "config_hash" not in yr_result:
                            yr_settings = {
                                "kind": "y_randomization",
                                "method": "kfold",
                                "n_permutations": int(cmp_yrandom_repeats),
                                "k": int(cmp_kfold_k),
                                "random_state": 42,
                            }
                            yr_result = _cache_comparison_validation_result(
                                yr_result,
                                selected_model,
                                yr_settings,
                                X_all_current,
                                y_all_current,
                                valid_indices_current,
                            )
                        st.session_state.setdefault("y_randomization_results_dict", {})[
                            selected_model
                        ] = yr_result

                st.session_state.pending_selected_model = selected_model
                st.session_state.best_model_from_comparison = selected_model
                st.session_state.post_descriptor_training_mode = "single"

            st.button(
                "🔬 Перейти к диагностике выбранной модели",
                key="switch_to_best_model",
                on_click=switch_to_best_model,
            )
        else:
            st.info(t('comparison.no_comparison_info'))
    # ================================================================
    # 🤝 Консенсусный прогноз
    # ================================================================
