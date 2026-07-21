# -*- coding: utf-8 -*-
"""Интерфейс consensus-прогноза QSPR-моделей."""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modules.i18n import t
from modules.module_explain_ui import render_module_explanation
from modules.model_catalog import get_model_display_name
from modules.qspr_core import qspr_csv_download_bytes


def _consensus_metrics(y_true, y_pred):
    """Metrics for genuinely out-of-fold consensus predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _comparison_oof_table(comparison_df):
    """Return OOF predictions shared by at least two compared models.

    A consensus must never be evaluated on fitted predictions.  Every column
    returned here is a K-Fold out-of-fold prediction stored by comparison UI.
    """
    model_col = t("comparison.model")
    if comparison_df is None or comparison_df.empty or model_col not in comparison_df:
        return pd.DataFrame(), [], ""

    rows = []
    unavailable = []
    for model_name in comparison_df[model_col].astype(str).tolist():
        result = st.session_state.get("kfold_results_dict", {}).get(model_name)
        if not isinstance(result, dict):
            unavailable.append(model_name)
            continue
        indices = result.get("valid_indices")
        y = result.get("y")
        pred = result.get("y_pred_cv")
        if indices is None or y is None or pred is None:
            unavailable.append(model_name)
            continue
        if not (len(indices) == len(y) == len(pred)):
            unavailable.append(model_name)
            continue
        frame = pd.DataFrame({
            "source_index": list(indices),
            "Experimental": np.asarray(y, dtype=float),
            model_name: np.asarray(pred, dtype=float),
        }).set_index("source_index")
        rows.append((model_name, frame))

    if len(rows) < 2:
        return pd.DataFrame(), [], ", ".join(unavailable)

    base_name, base = rows[0]
    table = base[["Experimental", base_name]].copy()
    available = [base_name]
    for model_name, frame in rows[1:]:
        # The same K-Fold data must be available for all members.  An inner
        # join prevents mixing predictions from different datasets.
        table = table.join(frame[[model_name]], how="inner")
        available.append(model_name)
    table = table.dropna().sort_index()
    return table, available, ", ".join(unavailable)


def _comparison_holdout_table(model_names):
    """Return frozen hold-out predictions shared by all selected models."""
    rows = []
    unavailable = []
    settings_signature = None
    for model_name in model_names:
        result = st.session_state.get("holdout_results_dict", {}).get(model_name)
        if not isinstance(result, dict):
            unavailable.append(model_name)
            continue
        indices = result.get("test_orig_indices")
        y = result.get("y_test")
        pred = result.get("y_pred_test")
        settings = result.get("validation_settings", {})
        if indices is None or y is None or pred is None:
            unavailable.append(model_name)
            continue
        if not (len(indices) == len(y) == len(pred)):
            unavailable.append(model_name)
            continue
        test_size = settings.get("test_size")
        random_state = settings.get("random_state")
        signature = (
            tuple(list(indices)),
            float(test_size) if test_size is not None else None,
            int(random_state) if random_state is not None else None,
        )
        if settings_signature is None:
            settings_signature = signature
        elif signature != settings_signature:
            unavailable.append(f"{model_name} (другой hold-out split)")
            continue
        frame = pd.DataFrame({
            "source_index": list(indices),
            "Experimental": np.asarray(y, dtype=float),
            model_name: np.asarray(pred, dtype=float),
        }).set_index("source_index")
        rows.append((model_name, frame))

    if len(rows) < 2:
        return pd.DataFrame(), unavailable

    base_name, base = rows[0]
    table = base[["Experimental", base_name]].copy()
    for model_name, frame in rows[1:]:
        table = table.join(frame[[model_name]], how="inner")
    table = table.dropna().sort_index()
    return table, unavailable


def _consensus_predictions_from_table(table, model_names, method, weights=None):
    pred_matrix = table[model_names].to_numpy(dtype=float).T
    if method == "median":
        return np.median(pred_matrix, axis=0), None
    if method == "weighted_cv_rmse" and weights:
        weight_values = np.asarray([float(weights.get(name, 0.0)) for name in model_names])
        if np.isfinite(weight_values).all() and weight_values.sum() > 0:
            weight_values = weight_values / weight_values.sum()
            return np.average(pred_matrix, axis=0, weights=weight_values), weight_values
    return np.mean(pred_matrix, axis=0), None


def _consensus_row_from_result(comparison_df, consensus_result):
    """Build display row for the model-comparison table."""
    if (
        comparison_df is None
        or comparison_df.empty
        or not isinstance(consensus_result, dict)
        or not consensus_result.get("metrics")
    ):
        return None
    row = {column: np.nan for column in comparison_df.columns}
    model_col = t("comparison.model")
    row[model_col] = "Consensus"
    if t("comparison.group") in row:
        row[t("comparison.group")] = "Консенсус моделей"
    metrics = consensus_result.get("metrics", {})
    holdout_metrics = consensus_result.get("holdout_metrics", {})
    row["Train R²"] = np.nan
    row["K-Fold Q²"] = metrics.get("R2", np.nan)
    row["Hold-out R²"] = holdout_metrics.get("R2", np.nan)
    row["LOO Q²"] = np.nan
    rmse_values = [
        value
        for value in (holdout_metrics.get("RMSE", np.nan), metrics.get("RMSE", np.nan))
        if pd.notna(value)
    ]
    mae_values = [
        value
        for value in (holdout_metrics.get("MAE", np.nan), metrics.get("MAE", np.nan))
        if pd.notna(value)
    ]
    row["RMSE"] = float(rmse_values[0]) if rmse_values else np.nan
    row["MAE"] = float(mae_values[0]) if mae_values else np.nan
    if t("comparison.checks") in row:
        row[t("comparison.checks")] = consensus_result.get(
            "validation_label",
            "K-Fold OOF consensus",
        )
    if t("comparison.comment") in row:
        if consensus_result.get("independent_improvement") is True:
            row[t("comparison.comment")] = "Consensus лучше лучшей одиночной модели на замороженном hold-out."
        elif consensus_result.get("holdout_metrics"):
            row[t("comparison.comment")] = "Consensus оценён на OOF и замороженном hold-out; превосходство не подтверждено."
        else:
            row[t("comparison.comment")] = "Consensus оценён только по K-Fold OOF; независимое превосходство не заявляется."
    if t("comparison.rating") in row:
        row[t("comparison.rating")] = np.nan
    return row


def comparison_df_with_consensus_row(comparison_df):
    """Append current Consensus row to comparison table for display/export."""
    consensus_result = st.session_state.get("consensus_result")
    row = _consensus_row_from_result(comparison_df, consensus_result)
    if row is None:
        return comparison_df
    model_col = t("comparison.model")
    table = comparison_df.copy()
    table = table[table[model_col].astype(str) != "Consensus"].copy()
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    if t("comparison.place") in table.columns:
        table = table.drop(columns=[t("comparison.place")])
        table.insert(0, t("comparison.place"), range(1, len(table) + 1))
    return table


def render_comparison_consensus_section(context):
    """Consensus builder placed directly after automatic model comparison.

    It works only with cached K-Fold OOF predictions.  Therefore displayed
    metrics are honest internal-CV metrics, never fitted-training metrics.
    """
    globals().update(context)
    comparison_df = st.session_state.get("model_comparison_df")
    oof_table, available_models, unavailable = _comparison_oof_table(comparison_df)

    st.divider()
    st.header("🤝 Консенсус моделей")
    st.caption(
        "Консенсус рассчитывается только по out-of-fold прогнозам K-Fold. "
        "Это внутренняя CV-оценка, а не независимая внешняя валидация."
    )
    if len(available_models) < 2 or oof_table.empty:
        st.info(
            "Для консенсуса нужны K-Fold out-of-fold прогнозы минимум двух моделей. "
            "Включите K-Fold и повторите сравнение."
        )
        return

    default_models = available_models[:min(3, len(available_models))]
    selected_models = st.multiselect(
        "Модели в консенсусе",
        options=available_models,
        default=default_models,
        key="comparison_consensus_models",
        format_func=get_model_display_name,
    )
    col_method, col_note = st.columns([1, 2])
    with col_method:
        method = st.selectbox(
            "Агрегация",
            options=["equal_mean", "median", "weighted_cv_rmse"],
            format_func=lambda value: {
                "equal_mean": "Простое среднее",
                "median": "Медиана",
                "weighted_cv_rmse": "Взвешенное по CV RMSE",
            }[value],
            key="comparison_consensus_method",
        )
    with col_note:
        st.info("Весовой вариант использует только ошибки этих же OOF-прогнозов: 1 / RMSE².")

    if len(selected_models) < 2:
        st.warning("Выберите не менее двух моделей.")
        return

    selected_oof = oof_table[["Experimental", *selected_models]].copy()
    if method == "median":
        weights = None
    elif method == "weighted_cv_rmse":
        rmse = np.asarray([
            _consensus_metrics(selected_oof["Experimental"], selected_oof[name])["RMSE"]
            for name in selected_models
        ])
        weights = 1.0 / np.maximum(rmse ** 2, 1e-12)
        weights = weights / weights.sum()
    else:
        weights = None
    weight_dict = dict(zip(selected_models, weights.tolist())) if weights is not None else {}
    consensus_pred, normalized_weights = _consensus_predictions_from_table(
        selected_oof,
        selected_models,
        method,
        weights=weight_dict,
    )
    pred_matrix = selected_oof[selected_models].to_numpy(dtype=float).T

    selected_oof["Consensus"] = consensus_pred
    selected_oof["Consensus std"] = np.std(pred_matrix, axis=0, ddof=1)
    selected_oof["Consensus residual"] = selected_oof["Experimental"] - consensus_pred
    metrics = _consensus_metrics(selected_oof["Experimental"], consensus_pred)
    member_metrics = {
        name: _consensus_metrics(selected_oof["Experimental"], selected_oof[name])
        for name in selected_models
    }
    best_member = min(member_metrics, key=lambda name: member_metrics[name]["RMSE"])
    holdout_table, holdout_unavailable = _comparison_holdout_table(selected_models)
    holdout_metrics = {}
    holdout_member_metrics = {}
    holdout_best_member = None
    independent_improvement = None
    holdout_df = pd.DataFrame()
    if not holdout_table.empty:
        holdout_pred, _ = _consensus_predictions_from_table(
            holdout_table,
            selected_models,
            method,
            weights=weight_dict,
        )
        holdout_df = holdout_table.copy()
        holdout_df["Consensus"] = holdout_pred
        holdout_df["Consensus std"] = np.std(
            holdout_df[selected_models].to_numpy(dtype=float).T,
            axis=0,
            ddof=1,
        )
        holdout_df["Consensus residual"] = holdout_df["Experimental"] - holdout_pred
        holdout_metrics = _consensus_metrics(holdout_df["Experimental"], holdout_pred)
        holdout_member_metrics = {
            name: _consensus_metrics(holdout_df["Experimental"], holdout_df[name])
            for name in selected_models
        }
        holdout_best_member = min(
            holdout_member_metrics,
            key=lambda name: holdout_member_metrics[name]["RMSE"],
        )
        independent_improvement = (
            holdout_metrics["RMSE"] < holdout_member_metrics[holdout_best_member]["RMSE"]
        )

    st.session_state.consensus_df = selected_oof.reset_index()
    st.session_state.consensus_holdout_df = (
        holdout_df.reset_index() if not holdout_df.empty else pd.DataFrame()
    )
    st.session_state.consensus_models = list(selected_models)
    st.session_state.consensus_result = {
        "kind": "kfold_oof_consensus",
        "model_label": "Consensus",
        "models": list(selected_models),
        "method": method,
        "weights": (
            dict(zip(selected_models, normalized_weights.tolist()))
            if normalized_weights is not None
            else weight_dict
        ),
        "selection_rule": "top ranked comparison models with K-Fold OOF predictions; user-editable composition",
        "weight_rule": "1 / RMSE_CV^2 from K-Fold OOF" if method == "weighted_cv_rmse" else "no learned weights",
        "aggregation_rule": method,
        "metrics": metrics,
        "best_member": best_member,
        "best_member_metrics": member_metrics[best_member],
        "holdout_metrics": holdout_metrics,
        "holdout_best_member": holdout_best_member,
        "holdout_best_member_metrics": (
            holdout_member_metrics.get(holdout_best_member, {})
            if holdout_best_member is not None
            else {}
        ),
        "holdout_unavailable": holdout_unavailable,
        "independent_improvement": independent_improvement,
        "validation_label": (
            "K-Fold OOF + frozen hold-out"
            if holdout_metrics
            else "K-Fold OOF only"
        ),
        "n_oof": int(len(selected_oof)),
        "n_holdout": int(len(holdout_df)) if not holdout_df.empty else 0,
    }

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Consensus K-Fold R²", f"{metrics['R2']:.3f}")
    m2.metric("Consensus K-Fold RMSE", f"{metrics['RMSE']:.3f}")
    m3.metric("Consensus K-Fold MAE", f"{metrics['MAE']:.3f}")
    m4.metric("Лучшая отдельная модель", get_model_display_name(best_member))

    comparison_note = (
        f"OOF: Consensus RMSE {metrics['RMSE']:.3f}; "
        f"{get_model_display_name(best_member)}: {member_metrics[best_member]['RMSE']:.3f}."
    )
    if independent_improvement is True:
        st.success(
            comparison_note
            + f" Hold-out: Consensus {holdout_metrics['RMSE']:.3f}; "
            + f"{get_model_display_name(holdout_best_member)} "
            + f"{holdout_member_metrics[holdout_best_member]['RMSE']:.3f}. "
            + "Консенсус лучше на независимой замороженной проверке."
        )
    elif independent_improvement is False:
        st.warning(
            comparison_note
            + f" Hold-out: Consensus {holdout_metrics['RMSE']:.3f}; "
            + f"{get_model_display_name(holdout_best_member)} "
            + f"{holdout_member_metrics[holdout_best_member]['RMSE']:.3f}. "
            + "Независимое превосходство консенсуса не подтверждено."
        )
    else:
        st.info(
            comparison_note
            + " Это внутренняя OOF-оценка; без общего hold-out нельзя заявлять, что консенсус лучше."
        )
    if unavailable:
        st.caption("Без K-Fold OOF прогнозов: " + unavailable)
    if holdout_unavailable:
        st.caption("Без общего hold-out для независимой оценки: " + ", ".join(holdout_unavailable))

    st.subheader("OOF-прогнозы и разброс моделей")
    st.dataframe(st.session_state.consensus_df.round(4), width="stretch", hide_index=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    y = selected_oof["Experimental"].to_numpy(dtype=float)
    lo, hi = min(y.min(), consensus_pred.min()), max(y.max(), consensus_pred.max())
    ax1.scatter(y, consensus_pred, alpha=0.75)
    ax1.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax1.set_xlabel("Экспериментальное значение")
    ax1.set_ylabel("Consensus OOF прогноз")
    ax1.grid(alpha=0.3)
    ax2.scatter(consensus_pred, selected_oof["Consensus residual"], alpha=0.75)
    ax2.axhline(0, color="r", linestyle="--", lw=1)
    ax2.set_xlabel("Consensus OOF прогноз")
    ax2.set_ylabel("Остаток")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.download_button(
        "Скачать OOF-результаты консенсуса CSV",
        qspr_csv_download_bytes(st.session_state.consensus_df),
        "consensus_oof_predictions.csv",
        "text/csv",
        key="download_comparison_consensus",
    )
    if not holdout_df.empty:
        st.subheader("Замороженный hold-out для Consensus")
        st.dataframe(st.session_state.consensus_holdout_df.round(4), width="stretch", hide_index=True)
        st.download_button(
            "Скачать hold-out результаты консенсуса CSV",
            qspr_csv_download_bytes(st.session_state.consensus_holdout_df),
            "consensus_holdout_predictions.csv",
            "text/csv",
            key="download_comparison_consensus_holdout",
        )


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
    
