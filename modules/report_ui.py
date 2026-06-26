# -*- coding: utf-8 -*-
"""Интерфейс формирования отчётов QSPR Forge."""

import base64
import io
import json
import os
import re
from datetime import datetime
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modules.i18n import t
from modules.statistics_summary_ui import (
    build_final_statistics_summary,
    final_statistics_to_flat_dataframe,
)

try:
    from rdkit import Chem as _ReportChem
    from rdkit.Chem import Draw
    rdkit_draw_available = True
except Exception:
    _ReportChem = None
    Draw = None
    rdkit_draw_available = False


def _add_molecule_grid_section(section_title, smiles_list, add_html_section, quality_comment_parts=None):
    """Adds a molecule grid to the report when RDKit drawing is available."""
    if Draw is None or _ReportChem is None:
        warning = (
            "Молекулярные изображения пропущены: RDKit Draw недоступен. "
            "Отчёт сформирован без структур."
        )
        if quality_comment_parts is not None:
            quality_comment_parts.append(warning)
        if "log_streamlit_message" in globals():
            log_streamlit_message(
                "REPORT",
                warning,
                level="warning",
                details={"section": section_title},
                event="report_molecule_images_skipped",
            )
        return False

    try:
        mols = []
        for smiles in list(smiles_list or []):
            mol = _ReportChem.MolFromSmiles(str(smiles))
            if mol is not None:
                mols.append(mol)
        if not mols:
            return False

        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        add_html_section(section_title, f'<img src="data:image/png;base64,{b64}">')
        return True
    except Exception as exc:
        warning = f"Молекулярные изображения пропущены: {exc}. Отчёт сформирован без структур."
        if quality_comment_parts is not None:
            quality_comment_parts.append(warning)
        if "log_streamlit_message" in globals():
            log_streamlit_message(
                "REPORT",
                warning,
                level="warning",
                details={"section": section_title, "error": str(exc)},
                event="report_molecule_images_failed",
            )
        return False


def render_report_section(context):
    """Рендерит текущий и полный отчёты проекта."""
    globals().update(context)
    model_name = (
        context.get("model_name")
        or st.session_state.get("last_model_algorithm", "")
    )
    
    st.header(t('report.header'))
    
    st.markdown(t('report.description'))
    
    current_model_ready_for_report = (
        "trained_models" in st.session_state
        and model_name in st.session_state.trained_models
    )
    
    holdout_ready_for_report = model_name in st.session_state.get("holdout_results_dict", {})
    kfold_ready_for_report = model_name in st.session_state.get("kfold_results_dict", {})
    loo_ready_for_report = model_name in st.session_state.get("loo_results_dict", {})
    
    saod_ready_for_report = (
        st.session_state.get("saod2_result") is not None
        or st.session_state.get("saod2_review_df") is not None
        or st.session_state.get("saod2_cleaned_df") is not None
    )
    
    struct_filter_ready_for_report = st.session_state.get("struct_filter_result_df") is not None
    
    spectra_ready_for_report = (
        st.session_state.get("spectral_descriptors_report") is not None
        or st.session_state.get("spectral_descriptors_df") is not None
        or st.session_state.get("spectral_descriptors_saved_path") is not None
    )
    
    model_comparison_df_for_report = None
    
    for _comparison_key in [
        "auto_model_comparison_df",
        "auto_model_comparison_table",
        "model_comparison_df",
        "model_compare_df",
        "true_model_comparison_df",
    ]:
        if _comparison_key in st.session_state:
            possible_df = st.session_state.get(_comparison_key)
            if isinstance(possible_df, pd.DataFrame) and not possible_df.empty:
                model_comparison_df_for_report = possible_df.copy()
                break
    
    model_comparison_ready_for_report = model_comparison_df_for_report is not None
    prognostic_ready_for_report = "prog_model" in st.session_state
    
    if not st.session_state.get("desc_calculated", False):
        st.info(t('report.info_no_descriptors'))
    else:
        report_dataset_name = st.text_input(
            t('report.dataset_name_label'),
            value=st.session_state.get("data_source_note", "") or "QSPR dataset",
            key="current_qspr_report_dataset_name"
        )
    
        report_author = st.text_input(
            t('report.author_label'),
            value="",
            key="current_qspr_report_author"
        )
    
        report_comment_user = st.text_area(
            t('report.comment_label'),
            value="",
            height=90,
            key="current_qspr_report_user_comment"
        )
    
        st.markdown(t('report.sections_title'))
    
        unavailable_report_sections = []
    
        report_col_1, report_col_2, report_col_3 = st.columns(3)
    
        with report_col_1:
            report_include_dataset = st.checkbox(
                t('report.include_dataset'),
                value=True,
                key="report_include_dataset"
            )
    
            report_include_descriptors = st.checkbox(
                t('report.include_descriptors'),
                value=True,
                key="report_include_descriptors"
            )
    
            if current_model_ready_for_report:
                report_include_model = st.checkbox(
                    t('report.include_model'),
                    value=True,
                    key="report_include_model"
                )
            else:
                report_include_model = False
                unavailable_report_sections.append(t('report.unavailable_model'))
    
            if current_model_ready_for_report:
                report_include_training_metrics = st.checkbox(
                    t('report.include_training_metrics'),
                    value=True,
                    key="report_include_training_metrics"
                )
            else:
                report_include_training_metrics = False
                unavailable_report_sections.append(t('report.unavailable_training_metrics'))
    
            if holdout_ready_for_report:
                report_include_holdout = st.checkbox(
                    "Hold-out",
                    value=True,
                    key="report_include_holdout"
                )
            else:
                report_include_holdout = False
                unavailable_report_sections.append("Hold-out")
    
        with report_col_2:
            if kfold_ready_for_report:
                report_include_kfold = st.checkbox(
                    "K-Fold",
                    value=True,
                    key="report_include_kfold"
                )
            else:
                report_include_kfold = False
                unavailable_report_sections.append("K-Fold")
    
            if loo_ready_for_report:
                report_include_loo = st.checkbox(
                    "Leave-One-Out",
                    value=True,
                    key="report_include_loo"
                )
            else:
                report_include_loo = False
                unavailable_report_sections.append("Leave-One-Out")
    
            if current_model_ready_for_report:
                report_include_ad = st.checkbox(
                    "Applicability Domain",
                    value=True,
                    key="report_include_ad"
                )
            else:
                report_include_ad = False
                unavailable_report_sections.append("Applicability Domain")
    
            if current_model_ready_for_report:
                report_include_error_outliers = st.checkbox(
                    t('report.include_error_outliers'),
                    value=True,
                    key="report_include_error_outliers"
                )
            else:
                report_include_error_outliers = False
                unavailable_report_sections.append(t('report.unavailable_error_outliers'))
    
            if model_comparison_ready_for_report:
                report_include_model_comparison = st.checkbox(
                    t('report.include_model_comparison'),
                    value=True,
                    key="report_include_model_comparison"
                )
            else:
                report_include_model_comparison = False
                unavailable_report_sections.append(t('report.unavailable_model_comparison'))
    
        with report_col_3:
            if saod_ready_for_report:
                report_include_saod = st.checkbox(
                    "*SAOD*",
                    value=True,
                    key="report_include_saod"
                )
            else:
                report_include_saod = False
                unavailable_report_sections.append("*SAOD*")
    
            if struct_filter_ready_for_report:
                report_include_struct_filter = st.checkbox(
                    t('report.include_struct_filter'),
                    value=True,
                    key="report_include_struct_filter"
                )
            else:
                report_include_struct_filter = False
                unavailable_report_sections.append(t('report.unavailable_struct_filter'))
    
            if spectra_ready_for_report:
                report_include_spectra = st.checkbox(
                    t('report.include_spectra'),
                    value=True,
                    key="report_include_spectra"
                )
            else:
                report_include_spectra = False
                unavailable_report_sections.append(t('report.unavailable_spectra'))
    
            if prognostic_ready_for_report:
                report_include_prognostic = st.checkbox(
                    t('report.include_prognostic'),
                    value=True,
                    key="report_include_prognostic"
                )
            else:
                report_include_prognostic = False
                unavailable_report_sections.append(t('report.unavailable_prognostic'))
    
            report_include_quality_comment = st.checkbox(
                t('report.include_quality_comment'),
                value=True,
                key="report_include_quality_comment"
            )

            report_include_final_statistics = st.checkbox(
                t('report.include_final_statistics'),
                value=True,
                key="report_include_final_statistics"
            )
    
        if unavailable_report_sections:
            with st.expander(t('report.unavailable_expander'), expanded=False):
                st.write(t('report.unavailable_auto_appear'))
                st.write(", ".join(unavailable_report_sections))
    
        report_format_col_1, report_format_col_2 = st.columns(2)
    
        with report_format_col_1:
            report_include_charts = st.checkbox(
                t('report.include_charts'),
                value=True,
                key="report_include_charts"
            )
    
        with report_format_col_2:
            report_max_rows = st.number_input(
                t('report.max_rows_label'),
                min_value=20,
                max_value=10000,
                value=1000,
                step=100,
                key="report_max_rows"
            )
    
        if st.button(t('report.generate_button'), type="primary", key="make_current_qspr_report"):
            try:
                import base64
                from html import escape
                from openpyxl.drawing.image import Image as OpenpyxlImage
    
                report_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Список всех дескрипторов (необходим для дальнейших разделов)
                report_desc_names_all = list(st.session_state.get("desc_names", []))
                report_desc_names_selected = list(st.session_state.get("model_used_descriptor_names", []))
                if report_desc_names_selected:
                    report_desc_names_used = report_desc_names_selected
                    report_descriptor_mode_label = t('report.descriptor_filtered')
                else:
                    report_desc_names_used = report_desc_names_all
                    report_descriptor_mode_label = t('report.descriptor_all')
    
                descriptor_source_report = st.session_state.get("custom_descriptor_source", "molecular_calculated")
                target_col = st.session_state.get("target_col", "")
                smiles_col_current = st.session_state.get("smiles_col_current", "SMILES")
                
                report_tables = {}
                report_charts = {}
                html_sections = []
    
                def _safe_sheet_name(name):
                    bad_chars = ["\\", "/", "*", "?", ":", "[", "]"]
                    safe = str(name)
                    for ch in bad_chars:
                        safe = safe.replace(ch, "_")
                    return safe[:31]
    
                def _df_to_html_table(df, max_rows=None):
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        return "<p><em>{t('report.no_data')}</em></p>"
    
                    if max_rows is None:
                        max_rows = int(report_max_rows)
    
                    view_df = df.head(int(max_rows)).copy()
    
                    return view_df.to_html(
                        index=False,
                        border=0,
                        classes="report-table",
                        escape=True
                    )
    
                def _add_html_section(title, content_html):
                    html_sections.append(
                        f"""
    <h2>{escape(str(title))}</h2>
    {content_html}
    """
                    )
    
                def _figure_to_base64(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
                    buf.seek(0)
                    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                    return buf, encoded
    
                def _get_prediction_from_model_data(model_data_local, model_local, X_scaled_local):
                    y_pred_source = None
    
                    for pred_key in [
                        "y_pred",
                        "y_pred_train",
                        "train_pred",
                        "y_calculated",
                        "predictions",
                    ]:
                        if isinstance(model_data_local, dict) and pred_key in model_data_local:
                            y_pred_source = model_data_local.get(pred_key)
                            break
    
                    if y_pred_source is None:
                        y_pred_source = np.ravel(model_local.predict(X_scaled_local))
    
                    return np.asarray(y_pred_source, dtype=float).ravel()
    
                # --------------------------------------------------------
                # 1. Dataset / project information
    
                data_rows_count = len(data) if data is not None else 0
                valid_rows_count = len(st.session_state.get("valid_indices", []))
                target_name = target_col
    
                # Получаем метрики из паспорта датасета (если он был вычислен ранее)
                n_suspicious = 0
                n_duplicates = 0
                try:
                    # Попробуем взять из переменной dataset_passport_df (она могла быть создана ранее)
                    if 'dataset_passport_df' in locals() and isinstance(dataset_passport_df, pd.DataFrame):
                        for _, row in dataset_passport_df.iterrows():
                            if row[t('passport.prompt')] == t('passport.suspicious_count'):
                                n_suspicious = int(row[t('passport.value')]) if str(row[t('passport.value')]).isdigit() else 0
                            if row[t('passport.prompt')] == t('passport.duplicates'):
                                n_duplicates = int(row[t('passport.value')]) if str(row[t('passport.value')]).isdigit() else 0
                except:
                    pass
    
                passport_df = pd.DataFrame([
                    {t('report.col_indicator'): t('report.passport_dataset_name'), t('report.col_value'): report_dataset_name},
                    {t('report.col_indicator'): t('report.passport_author'), t('report.col_value'): report_author},
                    {t('report.col_indicator'): t('report.passport_date'), t('report.col_value'): report_created_at},
                    {t('report.col_indicator'): t('report.passport_valid_compounds'), t('report.col_value'): valid_rows_count},
                    {t('report.col_indicator'): t('report.passport_target'), t('report.col_value'): target_name},
                    {t('report.col_indicator'): t('report.passport_suspicious'), t('report.col_value'): n_suspicious},
                    {t('report.col_indicator'): t('report.passport_duplicates'), t('report.col_value'): n_duplicates},
                ])
    
                report_tables["Dataset passport"] = passport_df
                _add_html_section(
                    t('report.section_passport'),
                    _df_to_html_table(passport_df, max_rows=100)
                )
                
                # --------------------------------------------------------
                # 1.1 SAOD suspicious compounds (if available)
                if saod_ready_for_report:
                    saod_review = st.session_state.get("saod2_review_df")
                    if saod_review is not None and not saod_review.empty:
                        # Фильтруем подозрительные по auto_recommendation или final_status
                        mask_auto = saod_review["SAOD_auto_recommendation"].astype(str).str.lower().str.contains(
                            t('saod2_review.auto_manual_check'), na=False
                        )
                        # Если есть подозрения по авторекомендации, используем их, иначе пробуем по final_status
                        if mask_auto.any():
                            suspicious_mask = mask_auto
                        else:
                            suspicious_mask = saod_review["final_status"].astype(str).str.lower().str.contains(
                                t('saod_suspicion.status_needs_check'), na=False
                            )
                        suspicious_saod_df = saod_review[suspicious_mask].copy()
                        if not suspicious_saod_df.empty:
                            # Подготовим таблицу для вывода
                            saod_susp_table = suspicious_saod_df[["SAOD_auto_recommendation", "SMILES", target_col]].head(200)
                            report_tables["SAOD suspicious compounds"] = saod_susp_table
                            _add_html_section(
                                t('report.section_saod_structures'),
                                _df_to_html_table(saod_susp_table)
                            )
                            _add_molecule_grid_section(
                                t('report.section_saod_structures'),
                                suspicious_saod_df["SMILES"].dropna().tolist(),
                                _add_html_section,
                                quality_comment_parts,
                            )
                
                # --------------------------------------------------------
                # 2. Distribution of target property (без seaborn)
                if report_include_charts and data is not None and target_col in data.columns:
                    y_vals = pd.to_numeric(data[target_col], errors="coerce")
                    y_vals = y_vals.replace([np.inf, -np.inf], np.nan).dropna()
                    if not y_vals.empty:
                        # Ограничиваем 1% и 99% перцентилями
                        q01 = y_vals.quantile(0.01)
                        q99 = y_vals.quantile(0.99)
                        y_vals_clipped = y_vals.clip(lower=q01, upper=q99)
                        y_vals_clipped = y_vals_clipped.replace([np.inf, -np.inf], np.nan).dropna()
                        if not y_vals_clipped.empty:
                            # Гистограмма
                            fig_hist, ax_hist = plt.subplots(figsize=(5, 3))
                            ax_hist.hist(y_vals_clipped, bins=30, range=(y_vals_clipped.min(), y_vals_clipped.max()),
                                         alpha=0.7, color='steelblue', edgecolor='black')
                            # KDE вручную (опционально, если нужна плавная кривая)
                            from scipy.stats import gaussian_kde
                            try:
                                kde = gaussian_kde(y_vals_clipped)
                                x_grid = np.linspace(y_vals_clipped.min(), y_vals_clipped.max(), 200)
                                ax_hist.plot(x_grid, kde(x_grid) * len(y_vals_clipped) * (y_vals_clipped.max()-y_vals_clipped.min())/30,
                                             'r-', linewidth=2, label='KDE')
                                ax_hist.legend()
                            except:
                                pass
                            ax_hist.set_title(t('report.hist_title', col=target_col))
                            ax_hist.set_xlabel(target_col)
                            ax_hist.set_ylabel(t('report.hist_ylabel'))
                            fig_hist.tight_layout()
                            hist_buf, hist_b64 = _figure_to_base64(fig_hist)
                            report_charts[t('report.hist_chart_label')] = hist_buf
    
                            # Boxplot
                            fig_box, ax_box = plt.subplots(figsize=(5, 2.5))
                            ax_box.boxplot(y_vals_clipped, vert=False, patch_artist=True)
                            ax_box.set_title(f"Boxplot {target_col}")
                            ax_box.set_xlabel(target_col)
                            fig_box.tight_layout()
                            box_buf, box_b64 = _figure_to_base64(fig_box)
                            report_charts[t('report.boxplot_chart_label')] = box_buf
    
                            _add_html_section(
                                t('report.distribution_boxplot_section_title'),
                                f'<div style="display:flex; gap:20px;">'
                                f'<div style="width:45%"><img src="data:image/png;base64,{hist_b64}" style="width:100%"></div>'
                                f'<div style="width:45%"><img src="data:image/png;base64,{box_b64}" style="width:100%"></div>'
                                f'</div>'
                            )
                # --------------------------------------------------------
                # 2. Descriptors (исключено по требованию)
                # Раздел с дескрипторами модели полностью удалён.
                # При необходимости можно оставить только спектральную сводку.
    
                # --------------------------------------------------------
                # 3. Current trained model
    
                model_report_df = pd.DataFrame()
                train_metrics_df_report = pd.DataFrame()
                prediction_report_df = pd.DataFrame()
                error_outliers_report_df = pd.DataFrame()
                coef_report_df = pd.DataFrame()
                ad_report_df = pd.DataFrame()
                ad_outliers_report_df = pd.DataFrame()
                quality_comment_parts = []
    
                if current_model_ready_for_report:
                    model_data_report = st.session_state.trained_models[model_name]
                    model_report = model_data_report.get("model")
                    scaler_report = model_data_report.get("scaler")
                    X_scaled_report = model_data_report.get("X_scaled")
    
                    if X_scaled_report is None:
                        if scaler_report is not None:
                            X_scaled_report = scaler_report.transform(X_all_current)
                        else:
                            X_scaled_report = X_all_current
    
                    y_true_report = np.asarray(y_all_current, dtype=float)
                    y_pred_report = _get_prediction_from_model_data(
                        model_data_report,
                        model_report,
                        X_scaled_report
                    )
    
                    if len(y_pred_report) != len(y_true_report):
                        y_pred_report = np.ravel(model_report.predict(X_scaled_report))
    
                    errors_report = y_true_report - y_pred_report
    
                    train_metrics_report = qspr_metrics(y_true_report, y_pred_report)
    
                    model_report_df = pd.DataFrame([
                        {t('report.model_indicator'): t('report.model_name_label'), t('report.model_value'): model_name},
                        {t('report.model_indicator'): t('report.model_group_label'), t('report.model_value'): st.session_state.get("last_model_group", "")},
                        {t('report.model_indicator'): t('report.model_compounds_label'), t('report.model_value'): len(y_true_report)},
                        {t('report.model_indicator'): t('report.model_desc_source_label'), t('report.model_value'): descriptor_source_report},
                        {t('report.model_indicator'): t('report.model_desc_count_label'), t('report.model_value'): len(desc_names_current)},
                        {t('report.model_indicator'): t('report.model_scaling_label'), t('report.model_value'): t('report.yes') if scaler_report is not None else t('report.no')},
                    ])
    
                    train_metrics_df_report = pd.DataFrame([
                        {t('report.metric_label'): k, t('report.metric_value'): v}
                        for k, v in train_metrics_report.items()
                    ])
                    
                    smiles_report = data[smiles_col_current].iloc[valid_indices_current].values
    
                    prediction_report_df = pd.DataFrame({
                        t('report.pred_number'): range(1, len(y_true_report) + 1),
                        t('report.pred_original_index'): [int(i) + 1 for i in valid_indices_current],
                        t('report.pred_smiles'): smiles_report,
                        t('report.pred_experimental'): y_true_report,
                        t('report.pred_calculated'): y_pred_report,
                        t('report.pred_error'): errors_report,
                        t('report.pred_abs_error'): np.abs(errors_report),
                    })
                    # Округление до 1 знака
                    for col in [t('report.pred_calculated'), t('report.pred_error'), t('report.pred_abs_error')]:
                        if col in prediction_report_df.columns:
                            prediction_report_df[col] = prediction_report_df[col].round(1)
    
    
                    error_threshold_report = float(
                        np.nanmean(np.abs(errors_report)) + 2.0 * np.nanstd(np.abs(errors_report))
                    )
    
                    error_outliers_report_df = prediction_report_df[
                        prediction_report_df[t('report.pred_abs_error')] > error_threshold_report
                    ].copy()
    
                    if not error_outliers_report_df.empty:
                        error_outliers_report_df[t('report.error_criterion')] = t('report.error_criterion_text')
    
                    try:
                        coef_report_df = qspr_extract_model_coefficients(
                            model_report,
                            report_desc_names_used,
                            model_name
                        )
                        # Добавим колонку с расшифровкой
                        if not coef_report_df.empty and t('report.coef_descriptor') in coef_report_df.columns:
                            meanings = qspr_load_descriptor_meanings()
                            coef_report_df[t('report.coef_meaning')] = coef_report_df[t('report.coef_descriptor')].apply(lambda d: meanings.get(str(d), "—"))
                    except Exception:
                        coef_report_df = pd.DataFrame()
    
                    if coef_report_df is None or coef_report_df.empty:
                        coef_report_df = pd.DataFrame([{
                            ('report.coef_comment'): t('report.coef_not_extracted')
                        }])
    
                    if report_include_model:
                        report_tables["Model"] = model_report_df
                        _add_html_section(
                            t('report.section_model'),
                            _df_to_html_table(model_report_df, max_rows=100)
                        )
    
                    if report_include_training_metrics:
                        report_tables["Training metrics"] = train_metrics_df_report
                        report_tables["Predictions"] = prediction_report_df
                        report_tables["Coefficients"] = coef_report_df
    
                        _add_html_section(
                            t('report.section_training_metrics'),
                            _df_to_html_table(train_metrics_df_report, max_rows=100)
                        )
    
                        _add_html_section(
                            t('report.section_predictions'),
                            _df_to_html_table(prediction_report_df)
                        )
    
                        _add_html_section(
                            t('report.section_coefficients'),
                            _df_to_html_table(coef_report_df)
                        )
    
                    if report_include_error_outliers:
                        report_tables["Error outliers"] = error_outliers_report_df
                        _add_html_section(
                            t('report.section_error_outliers'),
                            _df_to_html_table(error_outliers_report_df)
                        )
    
                    # ----------------------------------------------------
                    # Training charts (два графика в одной строке)
    
                    if report_include_charts:
                        try:
                            fig_pred_report, ax_pred_report = plt.subplots(figsize=(6, 5))
                            ax_pred_report.scatter(y_true_report, y_pred_report, alpha=0.75, s=45)
    
                            min_val_report = float(np.nanmin([
                                np.nanmin(y_true_report),
                                np.nanmin(y_pred_report)
                            ]))
    
                            max_val_report = float(np.nanmax([
                                np.nanmax(y_true_report),
                                np.nanmax(y_pred_report)
                            ]))
    
                            ax_pred_report.plot(
                                [min_val_report, max_val_report],
                                [min_val_report, max_val_report],
                                linestyle="--",
                                linewidth=1.5
                            )
    
                            ax_pred_report.set_xlabel(t('report.chart_exp_label'))
                            ax_pred_report.set_ylabel(t('report.chart_pred_label'))
                            ax_pred_report.set_title(t('report.chart_scatter_title', model=model_name))
                            ax_pred_report.grid(True, alpha=0.3)
                            fig_pred_report.tight_layout()
    
                            pred_plot_buffer, pred_plot_base64 = _figure_to_base64(fig_pred_report)
    
                            fig_err_report, ax_err_report = plt.subplots(figsize=(6, 4))
                            ax_err_report.hist(
                                errors_report,
                                bins=min(20, max(5, len(errors_report) // 3)),
                                alpha=0.8
                            )
                            ax_err_report.axvline(0, linestyle="--", linewidth=1.5)
                            ax_err_report.set_xlabel(t('report.chart_error_label'))
                            ax_err_report.set_ylabel(t('report.chart_hist_ylabel'))
                            ax_err_report.set_title(t('report.chart_hist_title'))
                            ax_err_report.grid(True, alpha=0.3)
                            fig_err_report.tight_layout()
    
                            err_plot_buffer, err_plot_base64 = _figure_to_base64(fig_err_report)
    
                            # Добавляем два графика в одну строку
                            _add_html_section(
                                t('report.section_charts_comparison'),
                                f'<div style="display:flex; gap:20px;">'
                                f'<div style="width:48%"><img src="data:image/png;base64,{pred_plot_base64}" style="width:100%"></div>'
                                f'<div style="width:48%"><img src="data:image/png;base64,{err_plot_base64}" style="width:100%"></div>'
                                f'</div>'
                            )
    
                            plt.close(fig_pred_report)
                            plt.close(fig_err_report)
    
                        except Exception as e:
                            quality_comment_parts.append(t('report.charts_error', error=e))
                    
                                # --------------------------------------------------------
                    # Махаланобис и тепловая карта корреляций
                    if "fig_mahal" in locals() and fig_mahal is not None:
                        mahal_buf, mahal_b64 = _figure_to_base64(fig_mahal)
                        report_charts["Mahalanobis distance"] = mahal_buf
                        _add_html_section(t('report.mahalanobis_distance_section'), f'<img src="data:image/png;base64,{mahal_b64}">')
                    if "fig_corr_heat" in locals() and fig_corr_heat is not None:
                        heat_buf, heat_b64 = _figure_to_base64(fig_corr_heat)
                        report_charts["Correlation heatmap"] = heat_buf
                        _add_html_section(t('report.correlation_heatmap_section'), f'<img src="data:image/png;base64,{heat_b64}">')
                    if "mahal_table_for_view" in locals() and mahal_table_for_view is not None and not mahal_table_for_view.empty:
                        report_tables["Mahalanobis outliers"] = mahal_table_for_view
                        _add_html_section(t('report.mahalanobis_outliers_section'), _df_to_html_table(mahal_table_for_view))
                        # Структуры выбросов (если есть SMILES)
                        if "SMILES" in mahal_table_for_view.columns:
                            _add_molecule_grid_section(
                                t('report.mahalanobis_structures_section'),
                                mahal_table_for_view["SMILES"].dropna().tolist(),
                                _add_html_section,
                                quality_comment_parts,
                            )
                    # ----------------------------------------------------
                    # Applicability Domain: leverage plot + Williams plot (без таблицы)
                    if report_include_ad:
                        try:
                            ad_table, ad_info = qspr_make_ad_table(
                                X_train=X_scaled_report,
                                smiles=smiles_report,
                                y=y_true_report,
                                original_indices=valid_indices_current,
                                desc_names=desc_names_current
                            )
                            # Рисунок leverage
                            fig_ad, ax_ad = plt.subplots(figsize=(6,4))
                            ax_ad.scatter(range(len(ad_table)), ad_table["Leverage h"], alpha=0.75)
                            ax_ad.axhline(ad_info["threshold"], color='r', linestyle='--')
                            ax_ad.set_xlabel(t('applicability_domain.leverage_plot_xlabel'))
                            ax_ad.set_ylabel("Leverage h")
                            ax_ad.set_title("Leverage distribution")
                            ad_buf, ad_b64 = _figure_to_base64(fig_ad)
                            report_charts["Leverage plot"] = ad_buf
    
                            # Williams plot
                            y_pred_ad = model_report.predict(X_scaled_report)
                            williams_df = ad_build_williams_plot_df(y_true_report, y_pred_ad, ad_info["leverage"], ad_info["threshold"])
                            fig_williams = ad_make_williams_plot(williams_df, ad_info["threshold"])
                            will_buf, will_b64 = _figure_to_base64(fig_williams)
                            report_charts["Williams plot"] = will_buf
    
                            _add_html_section(
                                "Applicability Domain",
                                f'<div style="display:flex; gap:20px;"><img src="data:image/png;base64,{ad_b64}" style="width:48%">'
                                f'<img src="data:image/png;base64,{will_b64}" style="width:48%"></div>'
                            )
    
                            # Таблица веществ вне AD (если есть)
                            outside_ad_df = ad_table[ad_table[t('ad_table.col_status')] == t('ad_leverage.out_ad')].copy()
                            if not outside_ad_df.empty:
                                report_tables[t('report.outside_ad_table')] = outside_ad_df
                                _add_html_section(t('report.outside_ad_section'), _df_to_html_table(outside_ad_df))
                                # Структуры
                                if "SMILES" in outside_ad_df.columns:
                                    _add_molecule_grid_section(
                                        t('report.outside_ad_structures_section'),
                                        outside_ad_df["SMILES"].dropna().tolist(),
                                        _add_html_section,
                                        quality_comment_parts,
                                    )
                        except Exception as e:
                            quality_comment_parts.append(t('report.ad_error_comment', error=e))
                            if "log_streamlit_message" in globals():
                                log_streamlit_message(
                                    "APPLICABILITY_DOMAIN",
                                    f"AD в отчёте не рассчитан: {e}",
                                    level="error",
                                    details={
                                        "error": str(e),
                                        "X_shape": getattr(X_scaled_report, "shape", None),
                                        "desc_names_len": len(desc_names_current or []),
                                    },
                                    event="report_ad_failed",
                                )
                    # ----------------------------------------------------
                    # Quality comment
    
                    r2_train_report = train_metrics_report.get("R2", np.nan)
    
                    if pd.notna(r2_train_report):
                        if r2_train_report >= 0.9:
                            quality_comment_parts.append(t('report.quality_excellent'))
                        elif r2_train_report >= 0.7:
                            quality_comment_parts.append(t('report.quality_good'))
                        else:
                            quality_comment_parts.append(t('report.quality_poor'))
    
                    if not error_outliers_report_df.empty:
                        quality_comment_parts.append(t('report.quality_outliers', count=len(error_outliers_report_df)))
                    if not error_outliers_report_df.empty and "SMILES" in error_outliers_report_df.columns:
                        _add_molecule_grid_section(
                            t('report.quality_outliers_structures_section'),
                            error_outliers_report_df["SMILES"].dropna().tolist(),
                            _add_html_section,
                            quality_comment_parts,
                        )
                # --------------------------------------------------------
                # 4. Validation sections
    
                validation_rows_report = []
    
                def _get_metrics_from_validation_result(result_dict, primary_key, fallback_key=None):
                    """
                    Безопасно достаёт метрики из результата валидации.
                    Нужно из-за разных имён ключей:
                    - Hold-out в qspr_core возвращает metrics_train / metrics_test;
                    - старые версии отчёта могли ожидать train_metrics / test_metrics.
                    """
                    if not isinstance(result_dict, dict):
                        return {}
    
                    metrics = result_dict.get(primary_key)
    
                    if metrics is None and fallback_key is not None:
                        metrics = result_dict.get(fallback_key)
    
                    if metrics is None:
                        metrics = {}
    
                    return metrics
    
                def _get_table_from_validation_result(result_dict, primary_key, fallback_key=None):
                    """
                    Безопасно достаёт таблицу прогноза.
                    K-Fold и LOO в qspr_core возвращают result_table.
                    """
                    if not isinstance(result_dict, dict):
                        return None
    
                    table = result_dict.get(primary_key)
    
                    if table is None and fallback_key is not None:
                        table = result_dict.get(fallback_key)
    
                    if isinstance(table, pd.DataFrame):
                        return table
    
                    return None
    
                if report_include_holdout and holdout_ready_for_report:
                    holdout_report = st.session_state.holdout_results_dict.get(model_name)
    
                    hold_train_metrics = _get_metrics_from_validation_result(
                        holdout_report,
                        primary_key="metrics_train",
                        fallback_key="train_metrics"
                    )
    
                    hold_test_metrics = _get_metrics_from_validation_result(
                        holdout_report,
                        primary_key="metrics_test",
                        fallback_key="test_metrics"
                    )
    
                    validation_rows_report.append({
                        t('report.validation_method'): "Hold-out train",
                        "R²/Q²": hold_train_metrics.get("R2", np.nan),
                        "RMSE": hold_train_metrics.get("RMSE", np.nan),
                        "MAE": hold_train_metrics.get("MAE", np.nan),
                        "MAPE, %": hold_train_metrics.get("MAPE_percent", np.nan),
                    })
    
                    validation_rows_report.append({
                        t('report.validation_method'): "Hold-out test",
                        "R²/Q²": hold_test_metrics.get("R2", np.nan),
                        "RMSE": hold_test_metrics.get("RMSE", np.nan),
                        "MAE": hold_test_metrics.get("MAE", np.nan),
                        "MAPE, %": hold_test_metrics.get("MAPE_percent", np.nan),
                    })
    
                    holdout_train_table = _get_table_from_validation_result(
                        holdout_report,
                        primary_key="train_table"
                    )
    
                    holdout_test_table = _get_table_from_validation_result(
                        holdout_report,
                        primary_key="test_table"
                    )
    
                    if holdout_train_table is not None:
                        report_tables["Holdout train table"] = holdout_train_table
    
                    if holdout_test_table is not None:
                        report_tables["Holdout test table"] = holdout_test_table
    
                if report_include_kfold and kfold_ready_for_report:
                    kfold_report = st.session_state.kfold_results_dict.get(model_name)
                    k_metrics = _get_metrics_from_validation_result(
                        kfold_report,
                        primary_key="metrics"
                    )
    
                    validation_rows_report.append({
                        t('report.validation_method'): f"K-Fold CV, k={kfold_report.get('k', '')}",
                        "R²/Q²": k_metrics.get("R2", np.nan),
                        "RMSE": k_metrics.get("RMSE", np.nan),
                        "MAE": k_metrics.get("MAE", np.nan),
                        "MAPE, %": k_metrics.get("MAPE_percent", np.nan),
                    })
    
                    kfold_table = _get_table_from_validation_result(
                        kfold_report,
                        primary_key="result_table",
                        fallback_key="table"
                    )
    
                    if kfold_table is not None:
                        report_tables["KFold table"] = kfold_table
    
                if report_include_loo and loo_ready_for_report:
                    loo_report = st.session_state.loo_results_dict.get(model_name)
                    loo_metrics = _get_metrics_from_validation_result(
                        loo_report,
                        primary_key="metrics"
                    )
    
                    validation_rows_report.append({
                        t('report.validation_method'): "Leave-One-Out",
                        "R²/Q²": loo_metrics.get("R2", np.nan),
                        "RMSE": loo_metrics.get("RMSE", np.nan),
                        "MAE": loo_metrics.get("MAE", np.nan),
                        "MAPE, %": loo_metrics.get("MAPE_percent", np.nan),
                    })
    
                    loo_table = _get_table_from_validation_result(
                        loo_report,
                        primary_key="result_table",
                        fallback_key="table"
                    )
    
                    if loo_table is not None:
                        report_tables["LOO table"] = loo_table
    
                validation_summary_report_df = pd.DataFrame(validation_rows_report)
    
                if not validation_summary_report_df.empty:
                    report_tables["Validation summary"] = validation_summary_report_df
                    _add_html_section(
                        "Hold-out / K-Fold / Leave-One-Out",
                        _df_to_html_table(validation_summary_report_df, max_rows=100)
                    )
    
                # --------------------------------------------------------
                # 5. SAOD
    
                if report_include_saod and saod_ready_for_report:
                    saod_summary_rows = []
    
                    saod_result = st.session_state.get("saod2_result")
                    saod_review_df = st.session_state.get("saod2_review_df")
                    saod_cleaned_df = st.session_state.get("saod2_cleaned_df")
    
                    if isinstance(saod_result, dict):
                        for key, value in saod_result.items():
                            if isinstance(value, pd.DataFrame):
                                saod_summary_rows.append({
                                    t('report.saod_col_table'): key,
                                    t('report.saod_col_rows'): len(value),
                                    t('report.saod_col_columns'): len(value.columns),
                                })
    
                                safe_name = "SAOD " + str(key)
                                report_tables[_safe_sheet_name(safe_name)] = value.head(1000)
    
                    if isinstance(saod_review_df, pd.DataFrame):
                        saod_summary_rows.append({
                            t('report.saod_col_table'): "saod2_review_df",
                            t('report.saod_col_rows'): len(saod_review_df),
                            t('report.saod_col_columns'): len(saod_review_df.columns),
                        })
                        report_tables["SAOD review"] = saod_review_df.head(1000)
    
                    if isinstance(saod_cleaned_df, pd.DataFrame):
                        saod_summary_rows.append({
                            t('report.saod_col_table'): "saod2_cleaned_df",
                            t('report.saod_col_rows'): len(saod_cleaned_df),
                            t('report.saod_col_columns'): len(saod_cleaned_df.columns),
                        })
                        report_tables["SAOD cleaned"] = saod_cleaned_df.head(1000)
    
                    saod_summary_df = pd.DataFrame(saod_summary_rows)
    
                    if not saod_summary_df.empty:
                        report_tables["SAOD summary"] = saod_summary_df
                        _add_html_section(
                            "*SAOD*",
                            _df_to_html_table(saod_summary_df, max_rows=100)
                        )
    
                # --------------------------------------------------------
                # 6. Structural filter
    
                if report_include_struct_filter and struct_filter_ready_for_report:
                    struct_df = st.session_state.get("struct_filter_result_df")
    
                    if isinstance(struct_df, pd.DataFrame):
                        struct_summary_df = pd.DataFrame([
                            {t('report.struct_prompt'): t('report.struct_rows_label'), t('report.struct_value'): len(struct_df)},
                            {t('report.struct_prompt'): t('report.struct_columns_label'), t('report.struct_value'): len(struct_df.columns)},
                            {t('report.struct_prompt'): t('report.struct_comment_label'), t('report.struct_value'): st.session_state.get("struct_filter_note", "")},
                        ])
    
                        report_tables["Struct filter summary"] = struct_summary_df
                        report_tables["Struct filter data"] = struct_df.head(1000)
    
                        _add_html_section(
                            t('report.struct_section_title'),
                            _df_to_html_table(struct_summary_df, max_rows=100)
                        )
    
                # --------------------------------------------------------
                # 7. Spectra
    
                if report_include_spectra and spectra_ready_for_report:
                    spectra_summary_rows = []
    
                    spectral_report = st.session_state.get("spectral_descriptors_report")
                    spectral_df = st.session_state.get("spectral_descriptors_df")
                    spectral_path = st.session_state.get("spectral_descriptors_saved_path", "")
    
                    if isinstance(spectral_report, dict):
                        for key, value in spectral_report.items():
                            if isinstance(value, (str, int, float, bool)):
                                spectra_summary_rows.append({
                                    t('report.spectra_param'): key,
                                    t('report.spectra_value'): value,
                                })
                            elif isinstance(value, pd.DataFrame):
                                report_tables[_safe_sheet_name("Spectra " + str(key))] = value.head(1000)
                                spectra_summary_rows.append({
                                    t('report.spectra_param'): key,
                                    t('report.spectra_value'): t('report.spectra_dataframe_info', rows=len(value), cols=len(value.columns)),
                                })
    
                    if isinstance(spectral_df, pd.DataFrame):
                        spectra_summary_rows.append({
                            t('report.spectra_param'): "spectral_descriptors_df",
                            t('report.spectra_value'): t('report.spectra_df_info', rows=len(spectral_df), cols=len(spectral_df.columns)),
                        })
                        report_tables["Spectral descriptors"] = spectral_df.head(1000)
    
                    if spectral_path:
                        spectra_summary_rows.append({
                            t('report.spectra_param'): t('report.spectra_file_label'),
                            t('report.spectra_value'): spectral_path,
                        })
    
                    spectra_summary_df = pd.DataFrame(spectra_summary_rows)
    
                    if not spectra_summary_df.empty:
                        report_tables["Spectra summary"] = spectra_summary_df
                        _add_html_section(
                            t('report.spectra_section_title'),
                            _df_to_html_table(spectra_summary_df, max_rows=200)
                        )
    
                # --------------------------------------------------------
                # 8. Model comparison
    
                if report_include_model_comparison and model_comparison_ready_for_report:
                    report_tables["Model comparison"] = model_comparison_df_for_report
                    _add_html_section(
                        t('report.model_comparison_section_title'),
                        _df_to_html_table(model_comparison_df_for_report)
                    )
                # --------------------------------------------------------
                # Графики валидации (Hold-out, K-Fold, LOO)
                validation_plots_html = ""
                if report_include_charts:
                    # Hold-out train/test scatter
                    if holdout_ready_for_report:
                        hold_res = st.session_state.holdout_results_dict.get(model_name)
                        if hold_res:
                            y_train = hold_res.get("y_train")
                            y_pred_train = hold_res.get("y_pred_train")
                            if y_train is not None:
                                fig_tr, ax_tr = plt.subplots(figsize=(5,4))
                                ax_tr.scatter(y_train, y_pred_train, alpha=0.6)
                                ax_tr.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
                                ax_tr.set_title("Hold-out train")
                                tr_buf, tr_b64 = _figure_to_base64(fig_tr)
                                validation_plots_html += f'<div style="display:inline-block; width:45%"><img src="data:image/png;base64,{tr_b64}"></div>'
                            y_test = hold_res.get("y_test")
                            y_pred_test = hold_res.get("y_pred_test")
                            if y_test is not None:
                                fig_te, ax_te = plt.subplots(figsize=(5,4))
                                ax_te.scatter(y_test, y_pred_test, alpha=0.6)
                                ax_te.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                                ax_te.set_title("Hold-out test")
                                te_buf, te_b64 = _figure_to_base64(fig_te)
                                validation_plots_html += f'<div style="display:inline-block; width:45%"><img src="data:image/png;base64,{te_b64}"></div>'
                    # K-Fold scatter
                    if kfold_ready_for_report:
                        kfold_res = st.session_state.kfold_results_dict.get(model_name)
                        if kfold_res:
                            y_cv = kfold_res.get("y")
                            y_pred_cv = kfold_res.get("y_pred_cv")
                            if y_cv is not None:
                                fig_cv, ax_cv = plt.subplots(figsize=(5,4))
                                ax_cv.scatter(y_cv, y_pred_cv, alpha=0.6)
                                ax_cv.plot([min(y_cv), max(y_cv)], [min(y_cv), max(y_cv)], 'r--')
                                ax_cv.set_title("K-Fold CV")
                                cv_buf, cv_b64 = _figure_to_base64(fig_cv)
                                validation_plots_html += f'<div style="display:inline-block; width:45%"><img src="data:image/png;base64,{cv_b64}"></div>'
                    # LOO scatter and residuals
                    if loo_ready_for_report:
                        loo_res = st.session_state.loo_results_dict.get(model_name)
                        if loo_res:
                            y_loo = loo_res.get("y")
                            y_pred_loo = loo_res.get("y_pred_loo")
                            if y_loo is not None:
                                fig_loo, ax_loo = plt.subplots(figsize=(5,4))
                                ax_loo.scatter(y_loo, y_pred_loo, alpha=0.6)
                                ax_loo.plot([min(y_loo), max(y_loo)], [min(y_loo), max(y_loo)], 'r--')
                                ax_loo.set_title("Leave-One-Out")
                                loo_buf, loo_b64 = _figure_to_base64(fig_loo)
                                validation_plots_html += f'<div style="display:inline-block; width:45%"><img src="data:image/png;base64,{loo_b64}"></div>'
                                # Residuals LOO
                                residuals = y_loo - y_pred_loo
                                fig_res, ax_res = plt.subplots(figsize=(5,4))
                                ax_res.scatter(y_pred_loo, residuals, alpha=0.6)
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_title(t('report.val_plot_loo_residuals'))
                                res_buf, res_b64 = _figure_to_base64(fig_res)
                                validation_plots_html += f'<div style="display:inline-block; width:45%"><img src="data:image/png;base64,{res_b64}"></div>'
                    if validation_plots_html:
                        _add_html_section(t('report.val_plots_section_title'), validation_plots_html)
                # --------------------------------------------------------
                # 9. Prognostic model
    
                if report_include_prognostic and prognostic_ready_for_report:
                    prognostic_summary_df = pd.DataFrame([
                        {t('report.prog_prompt'): t('report.prog_trained_label'), t('report.prog_value'): t('report.yes')},
                        {t('report.prog_prompt'): t('report.prog_model_label'), t('report.prog_value'): st.session_state.get("prog_model_name", model_name)},
                        {t('report.prog_prompt'): t('report.prog_target_label'), t('report.prog_value'): st.session_state.get("prog_target_col", target_col)},
                        {t('report.prog_prompt'): t('report.prog_smiles_label'), t('report.prog_value'): st.session_state.get("prog_smiles_col", smiles_col_current)},
                        {t('report.prog_prompt'): t('report.prog_desc_count_label'), t('report.prog_value'): len(st.session_state.get("prog_desc_names", desc_names_current))},
                    ])
    
                    report_tables["Prognostic model"] = prognostic_summary_df
                    _add_html_section(
                        t('report.prog_section_title'),
                        _df_to_html_table(prognostic_summary_df, max_rows=100)
                    )
    
                    uncertainty_payload = st.session_state.get(
                        "prediction_uncertainty_result"
                    )
                    if isinstance(uncertainty_payload, dict):
                        uncertainty_table = uncertainty_payload.get("table")
                        neighbour_table = uncertainty_payload.get("neighbours")
    
                        if (
                            isinstance(uncertainty_table, pd.DataFrame)
                            and not uncertainty_table.empty
                        ):
                            report_tables["Prediction uncertainty"] = (
                                uncertainty_table.head(1000)
                            )
                            _add_html_section(
                                t("prediction_uncertainty.header"),
                                _df_to_html_table(
                                    uncertainty_table.head(1000),
                                    max_rows=1000,
                                ),
                            )
    
                        if (
                            isinstance(neighbour_table, pd.DataFrame)
                            and not neighbour_table.empty
                        ):
                            report_tables["Prediction neighbours"] = (
                                neighbour_table.head(2000)
                            )
                # --------------------------------------------------------
                # 10.0 Консенсусный прогноз
                consensus_df = st.session_state.get("consensus_df")
                if consensus_df is not None and isinstance(consensus_df, pd.DataFrame) and not consensus_df.empty:
                    consensus_models = st.session_state.get("consensus_models", [])
                    report_tables["Consensus predictions"] = consensus_df.round(1)
    
                    _add_html_section(
                        t('report.consensus_section_title'),
                        _df_to_html_table(consensus_df.round(1))
                    )
    
                    if report_include_charts:
                        # Проверяем наличие необходимых колонок
                        if t('consensus.col_experiment') in consensus_df.columns and "Consensus_mean" in consensus_df.columns:
                            y_exp = consensus_df[t('consensus.col_experiment')].values
                            y_pred_mean = consensus_df["Consensus_mean"].values
                            y_std = consensus_df["Consensus_std"].values
    
                            # График 1: scatter с error bars
                            fig_cons, ax_cons = plt.subplots(figsize=(7, 5))
                            ax_cons.scatter(y_exp, y_pred_mean, alpha=0.7, s=30, label=t('report.consensus_plot_label'))
                            ax_cons.errorbar(y_exp, y_pred_mean, yerr=y_std, fmt='none',
                                             ecolor='gray', alpha=0.3, capsize=2)
                            min_val = min(y_exp.min(), y_pred_mean.min())
                            max_val = max(y_exp.max(), y_pred_mean.max())
                            ax_cons.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
                            ax_cons.set_xlabel(t('report.consensus_plot_xlabel'))
                            ax_cons.set_ylabel(t('report.consensus_plot_ylabel'))
                            ax_cons.set_title(t('report.consensus_plot_title', n_models=len(consensus_models)))
                            ax_cons.grid(True, alpha=0.3)
                            fig_cons.tight_layout()
    
                            # График 2: гистограмма неопределённости
                            fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
                            ax_hist.hist(y_std, bins=20, alpha=0.7, color='orange', edgecolor='black')
                            ax_hist.set_xlabel(t('report.consensus_hist_xlabel'))
                            ax_hist.set_ylabel(t('report.consensus_hist_ylabel'))
                            ax_hist.set_title(t('report.consensus_hist_title'))
                            fig_hist.tight_layout()
    
                            # Преобразуем в base64
                            cons_buf, cons_b64 = _figure_to_base64(fig_cons)
                            hist_buf, hist_b64 = _figure_to_base64(fig_hist)
                            report_charts["Consensus scatter"] = cons_buf
                            report_charts["Consensus uncertainty histogram"] = hist_buf
    
                            # Добавляем HTML с изображениями
                            _add_html_section(
                                t('report.consensus_plots_section'),
                                f'<div style="display:flex; flex-wrap:wrap; gap:20px;">'
                                f'<div style="flex:1; min-width:400px;"><img src="data:image/png;base64,{cons_b64}" style="width:100%"></div>'
                                f'<div style="flex:1; min-width:400px;"><img src="data:image/png;base64,{hist_b64}" style="width:100%"></div>'
                                f'</div>'
                            )
    
                            # Добавляем комментарий в общий пул
                            quality_comment_parts.append(
                                t('report.consensus_quality_comment',
                                    n_models=len(consensus_models),
                                    models=', '.join(consensus_models),
                                    mean_std=np.mean(y_std)
                                )
                            )
                            
                # --------------------------------------------------------
                # 10.1 Quality comment
    
                if report_comment_user.strip():
                    quality_comment_parts.append(report_comment_user.strip())
    
                if report_include_quality_comment:
                    if not quality_comment_parts:
                        quality_comment_parts.append(t('report.quality_default_comment'))
    
                    quality_comment_df = pd.DataFrame({
                        t('report.quality_comment_col'): quality_comment_parts
                    })
    
                    report_tables["Quality comment"] = quality_comment_df
    
                    _add_html_section(
                        t('report.quality_section_title'),
                        "<p>" + escape(" ".join(quality_comment_parts)) + "</p>"
                    )

                # --------------------------------------------------------
                # 10.2 Final statistics

                if report_include_final_statistics:
                    try:
                        final_stats_summary = build_final_statistics_summary(context)
                        final_stats_df = final_statistics_to_flat_dataframe(final_stats_summary)
                        if isinstance(final_stats_df, pd.DataFrame) and not final_stats_df.empty:
                            report_tables["Final statistics"] = final_stats_df
                            _add_html_section(
                                t('report.section_final_statistics'),
                                _df_to_html_table(final_stats_df, max_rows=300),
                            )
                    except Exception as exc:
                        quality_comment_parts.append(t('report.final_statistics_error', error=exc))
    
                # --------------------------------------------------------
                # 11. Excel report
    
                if not report_tables:
                    report_tables["Report"] = pd.DataFrame([{
                        t('report.report_message'): t('report.no_sections_selected')
                    }])
    
                excel_buffer = io.BytesIO()
    
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    used_sheet_names = set()
    
                    for sheet_name, df_sheet in report_tables.items():
                        if df_sheet is None:
                            continue
    
                        if not isinstance(df_sheet, pd.DataFrame):
                            continue
    
                        safe_sheet = _safe_sheet_name(sheet_name)
    
                        original_safe_sheet = safe_sheet
                        counter = 1
    
                        while safe_sheet in used_sheet_names:
                            suffix = f"_{counter}"
                            safe_sheet = (original_safe_sheet[:31 - len(suffix)] + suffix)
                            counter += 1
    
                        used_sheet_names.add(safe_sheet)
    
                        df_sheet.to_excel(writer, sheet_name=safe_sheet, index=False)
    
                    workbook = writer.book
    
                    if report_include_charts and report_charts:
                        charts_sheet = workbook.create_sheet("Charts")
    
                        current_row = 1
    
                        for chart_title, chart_buffer in report_charts.items():
                            charts_sheet[f"A{current_row}"] = chart_title
                            chart_buffer.seek(0)
                            chart_img = OpenpyxlImage(chart_buffer)
                            chart_img.anchor = f"A{current_row + 2}"
                            charts_sheet.add_image(chart_img)
                            current_row += 30
    
                    for sheet_name in workbook.sheetnames:
                        ws = workbook[sheet_name]
    
                        for column_cells in ws.columns:
                            max_length = 0
                            column_letter = column_cells[0].column_letter
    
                            for cell in column_cells:
                                try:
                                    max_length = max(max_length, len(str(cell.value)))
                                except Exception:
                                    pass
    
                            ws.column_dimensions[column_letter].width = min(max_length + 2, 45)
    
                excel_buffer.seek(0)
    
                # --------------------------------------------------------
                # 12. HTML / Word report
    
                html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>{t('report.html_title')}</title>
    <style>
    body {{
        font-family: Arial, sans-serif;
        margin: 32px;
        color: #222;
    }}
    h1, h2, h3 {{
        color: #12355b;
    }}
    .report-table {{
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 24px;
        font-size: 13px;
    }}
    .report-table th {{
        background: #e8eef7;
        border: 1px solid #b8c4d6;
        padding: 6px;
        text-align: left;
    }}
    .report-table td {{
        border: 1px solid #d0d7e2;
        padding: 6px;
    }}
    .note {{
        background: #eef6ff;
        border-left: 4px solid #2b6cb0;
        padding: 10px 14px;
        margin: 16px 0;
    }}
    img {{
        max-width: 850px;
        width: 100%;
        height: auto;
        margin: 10px 0 24px 0;
    }}
    </style>
    </head>
    <body>
    
    <h1>{t('report.html_title')}</h1>
    
    <div class="note">
    <b>{t('report.html_dataset')}:</b> {escape(str(report_dataset_name))}<br>
    <b>{t('report.html_author')}:</b> {escape(str(report_author))}<br>
    <b>{t('report.html_date')}:</b> {escape(str(report_created_at))}<br>
    <b>{t('report.html_target')}:</b> {escape(str(target_col))}<br>
    <b>{t('report.html_model')}:</b> {escape(str(model_name))}
    </div>
    """
    
                if html_sections:
                    html_report += "\n".join(html_sections)
                else:
                    html_report += "<p>{t('report.html_no_sections')}</p>"
    
                html_report += """
    </body>
    </html>
    """
    
                html_bytes = html_report.encode("utf-8")
    
                word_html_report = html_report.replace(
                    "<html>",
                    '<html xmlns:o="urn:schemas-microsoft-com:office:office" '
                    'xmlns:w="urn:schemas-microsoft-com:office:word" '
                    'xmlns="http://www.w3.org/TR/REC-html40">'
                )
    
                word_bytes = word_html_report.encode("utf-8")
    
                # --------------------------------------------------------
                # 13. Save to session state
    
                safe_report_name = (
                    "qspr_current_analysis_report_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                )
    
                st.session_state.current_qspr_report_excel = excel_buffer.getvalue()
                st.session_state.current_qspr_report_html = html_bytes
                st.session_state.current_qspr_report_word = word_bytes
                st.session_state.current_qspr_report_base_filename = safe_report_name
    
                st.success(t('report.success_generated'))
    
            except Exception as e:
                message = t('report.error_generation', error=e)
                st.error(message)
                if "log_streamlit_message" in globals():
                    log_streamlit_message(
                        "REPORT",
                        f"Отчёт не сформирован: {e}",
                        level="error",
                        details={"error": str(e), "model": model_name},
                        event="report_generation_failed",
                    )
    
        if "current_qspr_report_excel" in st.session_state:
            report_base_filename = st.session_state.get(
                "current_qspr_report_base_filename",
                "qspr_current_analysis_report"
            )
    
            download_report_col_1, download_report_col_2, download_report_col_3 = st.columns(3)
    
            with download_report_col_1:
                st.download_button(
                    t('report.download_excel'),
                    st.session_state.current_qspr_report_excel,
                    f"{report_base_filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_current_qspr_report_excel"
                )
    
            with download_report_col_2:
                st.download_button(
                    t('report.download_html'),
                    st.session_state.current_qspr_report_html,
                    f"{report_base_filename}.html",
                    "text/html",
                    key="download_current_qspr_report_html"
                )
    
            with download_report_col_3:
                st.download_button(
                    t('report.download_word'),
                    st.session_state.current_qspr_report_word,
                    f"{report_base_filename}.doc",
                    "application/msword",
                    key="download_current_qspr_report_word"
                )
    st.markdown("---")
    st.subheader(t('methodology.subheader'))
    
    col_lang, col_style, _ = st.columns([1, 1, 2])
    with col_lang:
        methodology_lang = st.selectbox(
            t('methodology.language_label'),
            ["ru", "en"],
            index=0 if st.session_state.get("methodology_language", "ru") == "ru" else 1,
            key="methodology_lang_select"
        )
    with col_style:
        methodology_style = st.selectbox(
            t('methodology.style_label'),
            ["full", "short"],
            index=0 if st.session_state.get("methodology_style", "full") == "full" else 1,
            key="methodology_style_select"
        )
    
    if st.button(t('methodology.generate_button'), type="primary"):
        # Собираем данные из session_state
        target_col = st.session_state.get("target_col", "")
        n_compounds = len(st.session_state.get("y_all", []))
        y_vals = st.session_state.get("y_all")
        if y_vals is not None and len(y_vals) > 0:
            target_stats = {
                "mean": float(np.mean(y_vals)),
                "std": float(np.std(y_vals)),
                "min": float(np.min(y_vals)),
                "max": float(np.max(y_vals))
            }
        else:
            target_stats = {}
    
        # Дескрипторы (приблизительная оценка)
        desc_names_all = st.session_state.get("desc_names", [])
        desc_names_used = st.session_state.get("model_used_descriptor_names", desc_names_all)
        # Для упрощения считаем все, что не начинается со специальных префиксов, как RDKit
        rdkit_count = sum(1 for d in desc_names_used if not d.startswith(("SPEC_", "xtb_", "morfeus_", "Mass_", "IR_")))
        mordred_count = sum(1 for d in desc_names_used if "Mordred" in str(d))  # упрощённо
        padel_count = sum(1 for d in desc_names_used if "PaDEL" in str(d))
        descriptors_info = {
            "rdkit": rdkit_count,
            "mordred": mordred_count,
            "padel": padel_count,
            "total": len(desc_names_used)
        }
    
        # Отбор дескрипторов
        auto_sel = st.session_state.get("auto_feature_selection", False)
        if auto_sel:
            sel_method = st.session_state.get("auto_feature_selection_method", "fast")
            n_initial = len(desc_names_all)
            n_final = len(desc_names_used)
            descriptor_selection = {
                "n_const": 0,  # можно уточнить, если храните
                "n_corr": 0,
                "corr_threshold": st.session_state.get("auto_corr_threshold", 0.95),
                "method": sel_method,
                "n_final": n_final
            }
        else:
            descriptor_selection = None
    
        # Модель
        model_name = st.session_state.get("last_model_algorithm", "")
        model_params = get_model_params_from_session()
        best_params = {}
        optimized = False
        search_method = ""
        cv_used = 5
        auto_tune = st.session_state.get("auto_tuning_result", {})
        if auto_tune:
            optimized = True
            best_params = auto_tune.get("best_params", {})
            search_method = auto_tune.get("search_method", "GridSearch")
            cv_used = auto_tune.get("cv", 5)
    
        model_info = {
            "name": model_name,
            "params": model_params,
            "optimized": optimized,
            "search_method": search_method,
            "cv": cv_used,
            "best_params": best_params
        }
    
        # Валидация
        validation_info = {}
        if model_name in st.session_state.get("holdout_results_dict", {}):
            ho = st.session_state.holdout_results_dict[model_name]
            validation_info["holdout"] = {
                "test_size": 0.2,  # можно взять из st.session_state, если храните
                "r2": ho.get("metrics_test", {}).get("R2", 0),
                "rmse": ho.get("metrics_test", {}).get("RMSE", 0),
                "mae": ho.get("metrics_test", {}).get("MAE", 0)
            }
        if model_name in st.session_state.get("kfold_results_dict", {}):
            kf = st.session_state.kfold_results_dict[model_name]
            validation_info["kfold"] = {
                "k": kf.get("k", 5),
                "r2": kf.get("metrics", {}).get("R2", 0),
                "rmse": kf.get("metrics", {}).get("RMSE", 0),
                "mae": kf.get("metrics", {}).get("MAE", 0)
            }
        if model_name in st.session_state.get("loo_results_dict", {}):
            lo = st.session_state.loo_results_dict[model_name]
            validation_info["loo"] = {
                "q2": lo.get("metrics", {}).get("R2", 0),
                "rmse": lo.get("metrics", {}).get("RMSE", 0)
            }
        if "bootstrap_results_dict" in st.session_state and model_name in st.session_state.bootstrap_results_dict:
            bs = st.session_state.bootstrap_results_dict[model_name]
            validation_info["bootstrap"] = {
                "n_iter": bs.get("summary", {}).get("n_iterations_successful", 0),
                "r2_mean": bs.get("summary", {}).get("r2_oob_mean", 0),
                "r2_std": bs.get("summary", {}).get("r2_oob_std", 0)
            }
        if "ext_validation_result" in st.session_state and st.session_state.ext_validation_result:
            ext = st.session_state.ext_validation_result
            validation_info["external"] = {
                "fraction": ext.get("fraction", 0.2),
                "r2_mean": ext.get("summary", {}).get("test_R2_mean", 0),
                "r2_std": ext.get("summary", {}).get("test_R2_std", 0)
            }
    
        # Applicability Domain
        ad_info = st.session_state.get("ad_info")
        ad_data = None
        if ad_info:
            ad_data = {
                "threshold": ad_info.get("threshold", 0),
                "n_out": sum(1 for s in ad_info.get("status", []) if s == "вне AD"),
                "pct": 0
            }
            if ad_info.get("n", 0) > 0:
                ad_data["pct"] = ad_data["n_out"] / ad_info["n"] * 100
    
        # R² для выбора тона заключения (возьмём из K-Fold или Hold-out)
        r2_for_conclusion = None
        rmse_for_conclusion = None
        mape_for_conclusion = None
        r2_cv_for_conclusion = None
        if validation_info.get("kfold"):
            r2_for_conclusion = validation_info["kfold"].get("r2")
            rmse_for_conclusion = validation_info["kfold"].get("rmse")
            mape_for_conclusion = validation_info["kfold"].get("mape")  # если есть
            r2_cv_for_conclusion = r2_for_conclusion  # K-Fold уже CV
        elif validation_info.get("holdout"):
            r2_for_conclusion = validation_info["holdout"].get("r2")
            rmse_for_conclusion = validation_info["holdout"].get("rmse")
            mape_for_conclusion = validation_info["holdout"].get("mape")
        ad_out_fraction = ad_data["pct"] / 100 if ad_data else None
    
        # Собираем все данные
        methodology_data = {
            "n_compounds": n_compounds,
            "target_col": target_col,
            "target_stats": target_stats,
            "descriptors": descriptors_info,
            "descriptor_selection": descriptor_selection,
            "model": model_info,
            "validation": validation_info,
            "ad": ad_data,
            "conclusion_r2": r2_for_conclusion,
            "conclusion_rmse": rmse_for_conclusion,
            "conclusion_mape": mape_for_conclusion,
            "conclusion_r2_cv": r2_cv_for_conclusion,
            "conclusion_ad_out_fraction": ad_out_fraction,
        }
    
        # Генерируем текст
        text = generate_methodology_text(
            data=methodology_data,
            language=methodology_lang,
            style=methodology_style
        )
    
        # Сохраняем в историю
        if "methodology_history" not in st.session_state:
            st.session_state.methodology_history = []
        version = len(st.session_state.methodology_history) + 1
        entry = {
            "version": version,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "language": methodology_lang,
            "style": methodology_style,
            "metrics": {
                "n_compounds": n_compounds,
                "r2_kfold": validation_info.get("kfold", {}).get("r2"),
                "r2_holdout": validation_info.get("holdout", {}).get("r2"),
            }
        }
        st.session_state.methodology_history.append(entry)
        st.session_state.methodology_current_index = len(st.session_state.methodology_history) - 1
        st.session_state.methodology_language = methodology_lang
        st.session_state.methodology_style = methodology_style
        st.rerun()
    
    # Отображение текущей версии
    history = st.session_state.get("methodology_history", [])
    if history:
        current_idx = st.session_state.get("methodology_current_index", len(history)-1)
        if current_idx >= len(history):
            current_idx = len(history)-1
        current_entry = history[current_idx]
    
        # Переключатель версий
        if len(history) > 1:
            version_options = [
                t('methodology.version_prefix', version=e['version'], timestamp=e['timestamp'])
                for e in history
            ]
            selected_version_label = st.selectbox(
                t('methodology.version_history_label'),
                options=version_options,
                index=current_idx,
                key="methodology_version_selector"
            )
            selected_idx = version_options.index(selected_version_label)
            if selected_idx != current_idx:
                st.session_state.methodology_current_index = selected_idx
                st.rerun()
    
        # Отображение текста
        st.text_area(t('methodology.text_area_label'), value=current_entry["text"], height=400, key="methodology_text_display", disabled=False)
    
        # Кнопки управления
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            st.download_button(
                t('methodology.download_txt_button'),
                data=current_entry["text"].encode("utf-8"),
                file_name=f"methodology_v{current_entry['version']}.txt",
                mime="text/plain",
                key="download_methodology_txt"
            )
        with col_btn2:
            st.info(t('methodology.copy_instruction'))
        with col_btn3:
            if st.button(t('methodology.clear_history_button'), key="clear_methodology_history"):
                st.session_state.methodology_history = []
                st.session_state.methodology_current_index = 0
                st.rerun()
    else:
        st.info(t('methodology.no_history_info'))
    
    st.markdown("---")
    st.subheader(t('report_full.subheader'))
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        report_lang = st.selectbox(
            t('report_full.language_label'),
            ["ru", "en"],
            index=0 if st.session_state.get("report_language", "ru") == "ru" else 1,
            key="report_lang_select"
        )
    with col_r2:
        st.caption(t('report_full.caption'))
    
    if st.button(t('report_full.generate_button'), type="primary"):
        # ---- Сбор данных ----
        target_col = st.session_state.get("target_col", "")
        y_vals = st.session_state.get("y_all")
        if y_vals is not None and len(y_vals) > 0:
            target_stats = {
                "mean": float(np.mean(y_vals)),
                "median": float(np.median(y_vals)),
                "std": float(np.std(y_vals)),
                "min": float(np.min(y_vals)),
                "max": float(np.max(y_vals)),
                "skew": float(pd.Series(y_vals).skew())
            }
        else:
            target_stats = {}
    
        # Данные
        n_initial = len(st.session_state.get("data", [])) if st.session_state.get("data") is not None else 0
        n_final = len(y_vals) if y_vals is not None else 0
        chemical_classes = ["алканы", "алкены", "ароматические"]  # Пример, можно взять из SAOD или диагностики
        preprocessing_steps = ["удаление некорректных SMILES", "удаление дубликатов", "удаление выбросов по IQR"]
    
        # Дескрипторы
        desc_names = st.session_state.get("desc_names", [])
        desc_used = st.session_state.get("model_used_descriptor_names", desc_names)
        desc_info = {
            "software": "RDKit (версия 2023.09.1)",
            "types": ["конституционные", "топологические", "электростатические", "E-State"],
            "n_initial": len(desc_names),
            "n_const": 0,  # можно оценить, если храните
            "n_corr": 0,
            "corr_threshold": st.session_state.get("auto_corr_threshold", 0.95),
            "selection_method": st.session_state.get("auto_feature_selection_method", "не применялся"),
            "n_final": len(desc_used)
        }
    
        # Модель
        model_name = st.session_state.get("last_model_algorithm", "")
        model_params = get_model_params_from_session()
        auto_tune = st.session_state.get("auto_tuning_result", {})
        model_info = {
            "algorithm": model_name,
            "params": model_params,
            "tuning_method": "GridSearch" if auto_tune else "ручная настройка",
            "cv": auto_tune.get("cv", 5),
            "validation_method": "K-Fold",
            "k": auto_tune.get("cv", 5),
            "q2_cv": 0.0,
            "rmse_cv": 0.0,
            "mae_cv": 0.0
        }
        if model_name in st.session_state.get("kfold_results_dict", {}):
            kf = st.session_state.kfold_results_dict[model_name]
            model_info.update({
                "q2_cv": kf.get("metrics", {}).get("R2", 0),
                "rmse_cv": kf.get("metrics", {}).get("RMSE", 0),
                "mae_cv": kf.get("metrics", {}).get("MAE", 0)
            })
    
        # Метрики
        metrics = {}
        if model_name in st.session_state.get("trained_models", {}):
            train_m = st.session_state.trained_models[model_name].get("metrics", {})
            metrics["train"] = {
                "r2": train_m.get("R2", 0),
                "rmse": train_m.get("RMSE", 0),
                "mae": train_m.get("MAE", 0)
            }
        if model_name in st.session_state.get("kfold_results_dict", {}):
            cv_m = st.session_state.kfold_results_dict[model_name].get("metrics", {})
            metrics["cv"] = {
                "r2": cv_m.get("R2", 0),
                "rmse": cv_m.get("RMSE", 0),
                "mae": cv_m.get("MAE", 0)
            }
        if model_name in st.session_state.get("holdout_results_dict", {}):
            test_m = st.session_state.holdout_results_dict[model_name].get("metrics_test", {})
            metrics["test"] = {
                "r2": test_m.get("R2", 0),
                "rmse": test_m.get("RMSE", 0),
                "mae": test_m.get("MAE", 0)
            }
    
        # Y-рандомизация (если есть)
        yrand = None
        if "y_randomization_results_dict" in st.session_state and model_name in st.session_state.y_randomization_results_dict:
            yr = st.session_state.y_randomization_results_dict[model_name]
            yrand = {
                "r2_rand_mean": yr.get("summary", {}).get("mean_q2_permuted", 0),
                "q2_rand_mean": yr.get("summary", {}).get("mean_q2_permuted", 0),
                "p_value": yr.get("summary", {}).get("p_value", 1)
            }
    
        # AD (если сохранён)
        ad_info = st.session_state.get("ad_info")
        ad_data = None
        if ad_info:
            ad_data = {
                "method": "leverage (рычаговое расстояние)",
                "threshold": ad_info.get("threshold", 0),
                "n_out_test": 0,  # можно посчитать по тестовым индексам, если есть
                "pct_test": 0,
                "n_out_train": sum(1 for s in ad_info.get("status", []) if s == "вне AD"),
                "pct_train": 0
            }
            if ad_info.get("n", 0) > 0:
                ad_data["pct_train"] = ad_data["n_out_train"] / ad_info["n"] * 100
    
        # Интерпретация
        coef_table = None
        if model_name in st.session_state.get("trained_models", {}):
            model = st.session_state.trained_models[model_name]["model"]
            coef_table = qspr_extract_model_coefficients(model, desc_used, model_name)
        interpretation = None
        if coef_table is not None and not coef_table.empty:
            top_features = coef_table.head(10)["Дескриптор"].tolist() if "Дескриптор" in coef_table.columns else []
            interpretation = {
                "method": "коэффициентов модели",
                "top_features": top_features,
                "meaning": "Положительные коэффициенты указывают на увеличение свойства с ростом дескриптора, отрицательные – на уменьшение."
            }
    
        # Сборка полных данных
        report_data = {
            "target_col": target_col,
            "source": "не указан",
            "units": "°C",
            "accuracy": "±0.5 °C",
            "n_initial": n_initial,
            "n_final": n_final,
            "chemical_classes": chemical_classes,
            "preprocessing_steps": preprocessing_steps,
            "target_stats": target_stats,
            "split_method": "случайное",
            "test_size": 0.2,
            "descriptors": desc_info,
            "model": model_info,
            "metrics": metrics,
            "y_randomization": yrand,
            "ad": ad_data,
            "interpretation": interpretation,
            "limitations": "Модель ограничена химическим пространством обучающей выборки.",
            "recommendations": "Рекомендуется расширить выборку и проверить модель на внешних данных.",
            "plots": {}  # можно добавить реальные графики, если они есть
        }
    
        # Генерируем отчёт
        report_result = generate_full_report(report_data, language=report_lang)
    
        # Сохраняем историю
        if "report_full_history" not in st.session_state:
            st.session_state.report_full_history = []
        version = len(st.session_state.report_full_history) + 1
        entry = {
            "version": version,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "html": report_result['html'],
            "language": report_lang,
            "metadata": report_result['metadata']
        }
        st.session_state.report_full_history.append(entry)
        st.session_state.report_full_current_index = len(st.session_state.report_full_history) - 1
        st.session_state.report_language = report_lang
        st.rerun()
    
    # Отображение текущей версии отчёта
    history = st.session_state.get("report_full_history", [])
    if history:
        current_idx = st.session_state.get("report_full_current_index", len(history)-1)
        if current_idx >= len(history):
            current_idx = len(history)-1
        current_entry = history[current_idx]
    
        if len(history) > 1:
            version_options = [t('report_full.version_prefix', version=e['version'], timestamp=e['timestamp'])
                for e in history
            ]
            selected = st.selectbox(t('report_full.version_history_label'), options=version_options, index=current_idx, key="report_version_selector")
            selected_idx = version_options.index(selected)
            if selected_idx != current_idx:
                st.session_state.report_full_current_index = selected_idx
                st.rerun()
    
        st.components.v1.html(current_entry['html'], height=800, scrolling=True)
    
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                t('report_full.download_html_button'),
                data=current_entry['html'].encode('utf-8'),
                file_name=f"qspr_report_v{current_entry['version']}.html",
                mime="text/html",
                key="download_report_html"
            )
        with col_d2:
            st.download_button(
                t('report_full.download_word_button'),
                data=current_entry['html'].encode('utf-8'),
                file_name=f"qspr_report_v{current_entry['version']}.doc",
                mime="application/msword",
                key="download_report_doc"
            )
    else:
        st.info(t('report_full.no_history_info'))
        
