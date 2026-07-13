# -*- coding: utf-8 -*-
"""Интерфейс consensus-прогноза QSPR-моделей."""

import numpy as np
import pandas as pd
import streamlit as st

from modules.i18n import t
from modules.module_explain_ui import render_module_explanation
from modules.qspr_core import qspr_csv_download_bytes


def _consensus_metric_value(comparison_df, model_name, candidates):
    if comparison_df is None or comparison_df.empty:
        return np.nan
    model_col = t('comparison.model')
    if model_col not in comparison_df.columns:
        return np.nan
    row = comparison_df[comparison_df[model_col].astype(str) == str(model_name)]
    if row.empty:
        return np.nan
    for col in candidates:
        if col in row.columns:
            try:
                return float(row.iloc[0][col])
            except Exception:
                continue
    return np.nan


def _consensus_weights_from_metrics(comparison_df, model_names, mode):
    if mode not in {"weighted_cv_rmse", "weighted_holdout_rmse"}:
        return None
    columns = (
        ["K-Fold RMSE", "CV RMSE", "RMSE"]
        if mode == "weighted_cv_rmse"
        else ["Hold-out RMSE", "Test RMSE", "RMSE"]
    )
    weights = {}
    for name in model_names:
        rmse = _consensus_metric_value(comparison_df, name, columns)
        weights[name] = 1.0 / max(float(rmse) ** 2, 1e-12) if np.isfinite(rmse) and rmse > 0 else 0.0
    if sum(weights.values()) <= 0:
        return None
    return weights


def render_consensus_section(context):
    """Рендерит consensus-прогноз в переданном контексте проекта."""
    globals().update(context)
    st.header(t('consensus.header'))
    render_module_explanation("consensus")
    st.markdown(t('consensus.description'))
    
    # Проверяем, есть ли обученные модели
    if "trained_models" in st.session_state and st.session_state.trained_models:
        model_names_all = list(st.session_state.trained_models.keys())
        st.write(t('consensus.models_available', count=len(model_names_all)))
    
        # Если есть таблица сравнения, используем её для ранжирования
        comparison_df = st.session_state.get("model_comparison_df")
        if comparison_df is not None and not comparison_df.empty:
            # Показываем метрики лучших моделей
            st.dataframe(
                comparison_df[[
                    t('comparison.model'),
                    "K-Fold Q²",
                    "LOO Q²",
                    "RMSE"
                ]].head(10),
                width="stretch",
                hide_index=True
            )
    
        # Выбор количества моделей для консенсуса
        max_models = len(model_names_all)
        default_n = min(3, max_models)
        top_n = st.number_input(
            t('consensus.top_n_label'),
            min_value=1,
            max_value=max_models,
            value=default_n,
            step=1,
            key="consensus_top_n"
        )
    
        # Дополнительно: отсев по минимальному R²
        min_r2_threshold = st.slider(
            t('consensus.min_r2_label'),
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="consensus_min_r2"
        )
        consensus_method = st.selectbox(
            t("consensus.aggregation_label"),
            [
                "equal_mean",
                "weighted_cv_rmse",
                "weighted_holdout_rmse",
                "median",
                "trimmed_mean",
            ],
            index=0,
            format_func=lambda value: {
                "equal_mean": t("consensus.aggregation_equal_mean"),
                "weighted_cv_rmse": t("consensus.aggregation_weighted_cv_rmse"),
                "weighted_holdout_rmse": t("consensus.aggregation_weighted_holdout_rmse"),
                "median": t("consensus.aggregation_median"),
                "trimmed_mean": t("consensus.aggregation_trimmed_mean"),
            }.get(value, value),
            key="consensus_aggregation_method",
        )
        st.caption(t("consensus.spread_caption"))
    
        if st.button(t('consensus.calc_button'), key="calc_consensus"):
            try:
                # --- Определяем список топ-моделей ---
                if comparison_df is not None and not comparison_df.empty:
                    # Сортируем по рейтингу (или по K-Fold Q²)
                    sorted_models = comparison_df.sort_values(
                        by=[t('comparison.rating'), "K-Fold Q²", "LOO Q²"],
                        ascending=[True, False, False]
                    )[t('comparison.model')].tolist()
                else:
                    # Если таблицы нет, берём все модели в произвольном порядке
                    sorted_models = model_names_all
    
                # Отфильтровываем модели с низким R² (если есть такая метрика)
                filtered_models = []
                for name in sorted_models:
                    model_data = st.session_state.trained_models.get(name, {})
                    r2 = model_data.get("metrics", {}).get("R2", np.nan)
                    if pd.notna(r2) and r2 >= min_r2_threshold:
                        filtered_models.append(name)
                    elif pd.isna(r2):
                        # Если R² нет, пропускаем модель (чтобы не рисковать)
                        continue
    
                if not filtered_models:
                    st.warning(t('consensus.warning_no_models_after_filter'))
                    st.stop()
    
                # Берём топ-N из отфильтрованного списка
                top_models = filtered_models[:top_n]
                if len(top_models) < 2:
                    st.warning(t('consensus.warning_need_two_models'))
                    st.stop()
    
                st.info(t('consensus.used_models_info', models=', '.join(top_models)))
    
                # --- Собираем модели и их скалеры ---
                models_scalers = {}
                for name in top_models:
                    model_entry = st.session_state.trained_models.get(name)
                    if model_entry is None:
                        continue
                    model = model_entry.get("model")
                    scaler = model_entry.get("scaler")   # может быть None
                    if model is not None:
                        models_scalers[name] = (model, scaler)
    
                if len(models_scalers) < 2:
                    st.error(t('consensus.error_loading_models'))
                    st.stop()
    
                # --- Данные для прогноза ---
                X_cons = X_all_current
                y_cons = y_all_current
                smiles_cons = data[smiles_col_current].iloc[valid_indices_current].values
    
                # --- Вычисляем консенсус ---
                with st.spinner(t('consensus.spinner')):
                    consensus_weights = _consensus_weights_from_metrics(
                        comparison_df,
                        list(models_scalers.keys()),
                        consensus_method,
                    )
                    if consensus_method.startswith("weighted") and not consensus_weights:
                        st.warning(t("consensus.weighted_fallback_warning"))
                    consensus_df = qspr_consensus_predictions(
                        models_scalers,
                        X_cons,
                        model_names=list(models_scalers.keys()),
                        method=consensus_method,
                        weights=consensus_weights,
                    )
                    # Добавляем SMILES и экспериментальные значения
                    consensus_df.insert(0, "SMILES", smiles_cons)
                    consensus_df.insert(1, t('consensus.col_experiment'), y_cons)
    
                # --- Отображаем таблицу ---
                st.subheader(t('consensus.subheader_predictions'))
                st.info(t("consensus.intermodel_disagreement_info"))
                # Округление числовых колонок до 1 знака после запятой
                numeric_cols = consensus_df.select_dtypes(include=[np.number]).columns
                consensus_display = consensus_df.copy()
                consensus_display[numeric_cols] = consensus_display[numeric_cols].round(1)
                st.dataframe(consensus_display, width="stretch", hide_index=True)
    
                # --- Два графика в одной строке ---
                col1, col2 = st.columns(2)
    
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_cons, consensus_df[t('consensus.col_mean')],
                               alpha=0.7, s=30, label=t('consensus.plot_label_consensus'))
                    ax.errorbar(
                        y_cons, consensus_df[t('consensus.col_mean')],
                        xerr=0, yerr=consensus_df[t('consensus.col_std')],
                        fmt='none', ecolor='gray', alpha=0.3, capsize=2
                    )
                    min_val = min(y_cons.min(), consensus_df[t('consensus.col_mean')].min())
                    max_val = max(y_cons.max(), consensus_df[t('consensus.col_mean')].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
                    ax.set_xlabel(t('consensus.plot_xlabel_exp'))
                    ax.set_ylabel(t('consensus.plot_ylabel_pred'))
                    ax.set_title(t('consensus.plot_title_consensus', n_models=len(models_scalers)))
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
    
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    ax2.hist(consensus_df[t('consensus.col_std')], bins=20,
                             alpha=0.7, color='orange', edgecolor='black')
                    ax2.set_xlabel(t('consensus.hist_xlabel_std'))
                    ax2.set_ylabel(t('consensus.hist_ylabel_count'))
                    ax2.set_title(t('consensus.hist_title_uncertainty'))
                    fig2.tight_layout()
                    st.pyplot(fig2)
    
                # --- Кнопка скачать ---
                csv_cons = qspr_csv_download_bytes(consensus_df)
                st.download_button(
                    t('consensus.download_csv'),
                    csv_cons,
                    "consensus_predictions.csv",
                    "text/csv",
                    key="download_consensus"
                )
    
                # Сохраняем в session_state для дальнейшего использования
                st.session_state.consensus_df = consensus_df
                st.session_state.consensus_models = top_models
    
            except Exception as e:
                st.error(t('consensus.error_calculation', error=e))
    
    else:
        st.info(t('consensus.info_train_first'))
        
    # ------------------------------------------------------------------
    # Current QSPR analysis report
    
