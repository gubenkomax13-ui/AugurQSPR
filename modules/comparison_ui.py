# -*- coding: utf-8 -*-
"""Интерфейс автоматического сравнения QSPR-моделей."""

import numpy as np
import pandas as pd
import streamlit as st

from modules.i18n import t
from modules.module_explain_ui import render_module_explanation


def render_model_comparison_section(context):
    """Рендерит сравнение моделей в переданном контексте проекта."""
    globals().update(context)
    # ------------------------------------------------------------------
    # ================================================================
    # True automatic model comparison
    
    st.header(t('model_comparison.header'))
    render_module_explanation("model_comparison")
    
    st.markdown(t('model_comparison.description'))
    
    try:
        all_model_groups_for_compare = qspr_available_model_options()
    except Exception:
        all_model_groups_for_compare = {}
    
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

    online_default_models = [
        "Множественная линейная регрессия (MLR)",
        "PLS Regression",
        "Ridge",
        "LASSO",
        "Elastic Net",
        "CART Regression",
    ]
    
    safe_default_models = [
        "Множественная линейная регрессия (MLR)",
        "PLS Regression",
        "Ridge",
        "LASSO",
        "Elastic Net",
        "SVR",
        "KNN Regression",
        "Random Forest",
        "CART Regression",
        "MARS-like Regression",
        "Spline Regression",
        "GAM Regression",
        "Voting Regressor",
    ]

    if online_light_mode:
        safe_default_models = online_default_models
    
    safe_default_models = [m for m in safe_default_models if m in all_models_for_compare]
    
    if not all_models_for_compare:
        st.warning(t('model_comparison.no_models_warning'))
    else:
        if online_light_mode:
            st.info(
                t("model_comparison.online_light_info")
            )
            cloud_widget_limits = {
                "cmp_top_n": 5,
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
            selected_compare_models = st.multiselect(
                t('model_comparison.candidates_label'),
                options=all_models_for_compare,
                default=safe_default_models,
                key="auto_compare_selected_models",
                help=t('model_comparison.candidates_help'),
                format_func=get_model_display_name,
            )
    
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
    
            cmp_run_loo_top = st.checkbox(
                t('model_comparison.loo_top_checkbox'),
                value=not online_light_mode,
                key="cmp_run_loo_top"
            )
    
            cmp_run_montecarlo_top = st.checkbox(
                t('model_comparison.montecarlo_top_checkbox'),
                value=False,
                key="cmp_run_montecarlo_top"
            )
    
            cmp_run_bootstrap_top = st.checkbox(
                t('model_comparison.bootstrap_top_checkbox'),
                value=False,
                key="cmp_run_bootstrap_top"
            )
    
            cmp_run_yrandom_top = st.checkbox(
                t('model_comparison.yrandom_top_checkbox'),
                value=False,
                key="cmp_run_yrandom_top"
            )
    
            cmp_top_n = st.number_input(
                t('model_comparison.top_n_label'),
                min_value=1,
                max_value=5 if online_light_mode else 10,
                value=2 if online_light_mode else 3,
                step=1,
                key="cmp_top_n"
            )
    
            cmp_mc_repeats = st.number_input(
                t('model_comparison.mc_repeats_label'),
                min_value=10,
                max_value=50 if online_light_mode else 500,
                value=10 if online_light_mode else 50,
                step=10,
                key="cmp_mc_repeats",
                disabled=not cmp_run_montecarlo_top
            )
    
            cmp_bootstrap_repeats = st.number_input(
                t('model_comparison.bootstrap_repeats_label'),
                min_value=10,
                max_value=50 if online_light_mode else 500,
                value=10 if online_light_mode else 100,
                step=10,
                key="cmp_bootstrap_repeats",
                disabled=not cmp_run_bootstrap_top
            )
    
            cmp_yrandom_repeats = st.number_input(
                t('model_comparison.yrandom_repeats_label'),
                min_value=10,
                max_value=50 if online_light_mode else 1000,
                value=10 if online_light_mode else 100,
                step=10,
                key="cmp_yrandom_repeats",
                disabled=not cmp_run_yrandom_top
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
            
    
        if st.button(t('model_comparison.run_button'), type="primary", key="train_compare_and_choose_models"):
            if not selected_compare_models:
                st.warning(t('model_comparison.warning_select_models'))
            else:
                try:
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
    
                            top_models_for_extra_checks = (
                                preliminary_df[t('comparison.model')]
                                .head(int(cmp_top_n))
                                .astype(str)
                                .tolist()
                            )
    
                            st.info(t(
                                'model_comparison.top_models_info',
                                n=len(top_models_for_extra_checks),
                                count=len(top_models_for_extra_checks),
                                models=', '.join(
                                    get_model_display_name(name)
                                    for name in top_models_for_extra_checks
                                )
                            ))
    
                            # ------------------------------------------------------------
                            # Этап 2: LOO только для top-N
    
                            progress.progress(65, text=t('model_comparison.progress_stage2'))
    
                            messages_loo, error_df_loo = qspr_auto_train_validate_models_for_comparison(
                                model_names=top_models_for_extra_checks,
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
                            )
    
                            messages = list(messages or []) + list(messages_loo or [])
                            
                            # ------------------------------------------------------------
                            # Monte-Carlo CV для лучших моделей (с обновлением прогресса)
                            if cmp_run_montecarlo_top:
                                montecarlo_results = {}
                                total_mc = len(top_models_for_extra_checks)
                                for idx, model_name_mc in enumerate(top_models_for_extra_checks, 1):
                                    progress.progress(
                                        65 + int(15 * idx / total_mc),
                                        text=t('model_comparison.progress_mc', model=model_name_mc, idx=idx, total=total_mc)
                                    )
                                    montecarlo_results[model_name_mc] = (
                                        qspr_repeated_holdout_validation(
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
                                            progress_callback=None
                                        )
                                    )
                                st.session_state.montecarlo_results_dict = montecarlo_results
                                
                            # ------------------------------------------------------------
                            # Bootstrap CV для лучших моделей (с обновлением прогресса)
                            if cmp_run_bootstrap_top:
                                bootstrap_results = {}
                                total_bs = len(top_models_for_extra_checks)
                                for idx, model_name_bs in enumerate(top_models_for_extra_checks, 1):
                                    progress.progress(
                                        80 + int(10 * idx / total_bs),
                                        text=t('model_comparison.progress_bs', model=model_name_bs, idx=idx, total=total_bs)
                                    )
                                    bootstrap_results[model_name_bs] = (
                                        qspr_bootstrap_validation(
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
                                            progress_callback=None
                                        )
                                    )
                                st.session_state.bootstrap_results_dict = bootstrap_results  
    
                            # ------------------------------------------------------------
                            # Y-randomization для ЛУЧШЕЙ модели (до финальной таблицы)
                            if cmp_run_yrandom_top and top_models_for_extra_checks:
                                best_candidate = top_models_for_extra_checks[0]
                                progress.progress(90, text=t('model_comparison.progress_yr', model=best_candidate))
                                st.session_state.yrandom_best_model_result = qspr_y_randomization_test(
                                    X=X_all_current,
                                    y=y_all_current,
                                    model_name=best_candidate,
                                    valid_indices=valid_indices_current,
                                    smiles=smiles_for_compare,
                                    params=get_model_params_from_session(),
                                    method="K-Fold",
                                    n_permutations=int(cmp_yrandom_repeats),
                                    k=cmp_kfold_k,
                                    random_state=42,
                                    scale=True,
                                    progress_callback=None
                                )
    
                            if error_df is not None and error_df_loo is not None:
                                error_df = pd.concat([error_df, error_df_loo], ignore_index=True)
                            elif error_df is None:
                                error_df = error_df_loo
    
                        else:
                            # Старый полный режим
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
    
                        # Bootstrap CV
                        if st.session_state.get("cmp_run_bootstrap_top", False) and "bootstrap_results_dict" in st.session_state:
                            boot_dict = st.session_state.bootstrap_results_dict
                            for model_name_bs, res_bs in boot_dict.items():
                                r2_oob_mean = res_bs.get("summary", {}).get("r2_oob_mean", np.nan)
                                mask = comparison_df[t('comparison.model')] == model_name_bs
                                comparison_df.loc[mask, t('comparison.bootstrap_r2_oob')] = r2_oob_mean
    
                        # Y-randomization p-value для лучшей модели
                        if st.session_state.get("cmp_run_yrandom_top", False) and "yrandom_best_model_result" in st.session_state:
                            yrand_res = st.session_state.yrandom_best_model_result
                            p_val = np.nan
                            if isinstance(yrand_res, dict):
                                p_val = yrand_res.get("summary", {}).get("p_value", np.nan)
                            if not comparison_df.empty:
                                best_model_name = comparison_df.iloc[0][t('comparison.model')]
                                mask = comparison_df[t('comparison.model')] == best_model_name
                                comparison_df.loc[mask, t('comparison.yrand_pvalue')] = p_val
    
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
                            add_log(t('model_comparison.log_best_model', model=best_model))
    
                            if messages:
                                st.success(t('model_comparison.success_models_trained', count=len(set(selected_compare_models))))
    
                            st.success(t('model_comparison.success_best_model', model=best_model))
                            st.rerun()
    
                except Exception as e:
                    st.error(t('model_comparison.error_comparison', error=e))
    
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
    
            display_cmp = comparison_df.copy()
            model_column = t('comparison.model')
            if model_column in display_cmp.columns:
                display_cmp[model_column] = display_cmp[model_column].map(
                    get_model_display_name
                )
    
            numeric_cols = [
                "Train R²",
                "Hold-out R²",
                "K-Fold Q²",
                "LOO Q²",
                "RMSE",
                "MAE",
                "Вне AD, %",
                "Рейтинг",
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
    
            if st.button(t('comparison.switch_button'), key="switch_to_best_model"):
                st.session_state.pending_selected_model = best_model
                st.session_state.best_model_from_comparison = best_model
                st.success(t('comparison.switch_success', model=best_model))
                st.rerun()
        else:
            st.info(t('comparison.no_comparison_info'))
    # ================================================================
    # 🤝 Консенсусный прогноз
    # ================================================================
