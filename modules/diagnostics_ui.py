# -*- coding: utf-8 -*-
"""Applicability Domain, важность дескрипторов и SHAP."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import streamlit as st

from modules.error_analysis_core import error_analysis_structural_annotations
from modules.i18n import t
from modules.module_explain_ui import render_module_explanation

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None


def _diagnostics_strength(abs_corr):
    if not np.isfinite(abs_corr):
        return t("diagnostics.residual_strength_unavailable")
    if abs_corr < 0.2:
        return t("diagnostics.residual_strength_weak")
    if abs_corr <= 0.5:
        return t("diagnostics.residual_strength_moderate")
    return t("diagnostics.residual_strength_strong")


def _diagnostics_direction(corr):
    if not np.isfinite(corr) or abs(corr) < 1e-12:
        return t("diagnostics.residual_direction_none")
    if corr > 0:
        return t("diagnostics.residual_direction_under")
    return t("diagnostics.residual_direction_over")


def _diagnostics_correlation(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, np.nan, np.nan

    if pearsonr is not None:
        try:
            corr, p_value = pearsonr(pair["x"], pair["y"])
            return float(corr), float(p_value), len(pair)
        except Exception:
            pass

    return float(pair["x"].corr(pair["y"], method="pearson")), np.nan, len(pair)


def _diagnostics_spearman(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, np.nan

    if spearmanr is not None:
        try:
            corr, p_value = spearmanr(pair["x"], pair["y"])
            return float(corr), float(p_value)
        except Exception:
            pass

    return float(pair["x"].corr(pair["y"], method="spearman")), np.nan


def _diagnostics_descriptor_matrix(context):
    desc_names = list(context.get("desc_names_current") or [])
    if not desc_names:
        return None, []

    for key in ("X_all_current", "X_current", "X_scaled"):
        value = context.get(key)
        if value is None:
            continue
        try:
            matrix = np.asarray(value, dtype=float)
        except Exception:
            continue
        if matrix.ndim == 2 and matrix.shape[1] == len(desc_names):
            return matrix, desc_names
    return None, desc_names


def _diagnostics_dimension_context(context):
    model_data = context.get("model_data") or {}
    if not model_data and "st" in globals():
        model_name = context.get("model_name")
        model_data = st.session_state.get("trained_models", {}).get(model_name, {}) or {}

    auto_result = model_data.get("auto_result") or st.session_state.get("auto_tuning_result", {}) or {}
    selection_summary = auto_result.get("selection_summary", {}) if isinstance(auto_result, dict) else {}
    selected_names = list(model_data.get("selected_desc_names") or auto_result.get("selected_desc_names") or [])
    desc_names = list(context.get("desc_names_current") or [])

    selected_indices = []
    for source in (model_data, auto_result, selection_summary):
        if not isinstance(source, dict):
            continue
        for key in ("selected_indices", "selected_idx", "feature_indices", "selected_feature_indices"):
            value = source.get(key)
            if value is None:
                continue
            try:
                selected_indices = [int(v) for v in list(value)]
            except Exception:
                selected_indices = []
            if selected_indices:
                break
        if selected_indices:
            break

    def _shape(value):
        try:
            arr = np.asarray(value)
            return tuple(arr.shape)
        except Exception:
            return None

    x_shape = _shape(context.get("X_scaled"))
    x_original_value = model_data.get("X_original")
    if x_original_value is None:
        x_original_value = context.get("X_all_current")
    x_original_shape = _shape(x_original_value)
    expected_features = getattr(context.get("model"), "n_features_in_", None)

    return {
        "original_feature_count": selection_summary.get("n_features_initial")
        or (x_original_shape[1] if x_original_shape and len(x_original_shape) > 1 else None),
        "selected_feature_count": len(selected_names) or selection_summary.get("n_selected_final"),
        "X_shape": x_shape,
        "X_original_shape": x_original_shape,
        "desc_names_len": len(desc_names),
        "selected_desc_names_len": len(selected_names),
        "max_selected_index": max(selected_indices) if selected_indices else None,
        "model_n_features_in": int(expected_features) if expected_features is not None else None,
    }


def _log_dimension_failure(stage, exc, context, event):
    details = _diagnostics_dimension_context(context)
    x_cols = details.get("X_shape", ("", ""))[1] if details.get("X_shape") and len(details["X_shape"]) > 1 else None
    requested = details.get("max_selected_index")
    if requested is None:
        match = re.search(r"index\s+(\d+)\s+is out of bounds", str(exc))
        requested = int(match.group(1)) if match else None

    user_message = (
        "Диагностика не рассчитана из-за несовместимости матрицы признаков после отбора дескрипторов. "
        "Вероятная причина: индексы выбранных дескрипторов применены повторно."
    )
    if "log_streamlit_message" in globals():
        if x_cols is not None and requested is not None:
            log_message = (
                f"AD не рассчитан: X имеет {x_cols} колонок, запрошен индекс {requested}; "
                "вероятна повторная фильтрация selected_indices."
            )
        else:
            log_message = f"Диагностика не рассчитана: {exc}"
        log_streamlit_message(
            stage,
            log_message,
            level="error",
            details={**details, "error": str(exc)},
            event=event,
        )
    return user_message


def _diagnostics_make_residual_descriptor_table(desc_df, residuals):
    rows = []
    for descriptor in desc_df.columns:
        x = desc_df[descriptor]
        pearson_corr, pearson_p, n_used = _diagnostics_correlation(x, residuals)
        spearman_corr, spearman_p = _diagnostics_spearman(x, residuals)
        if not np.isfinite(pearson_corr) and not np.isfinite(spearman_corr):
            continue
        rank_corr = spearman_corr if np.isfinite(spearman_corr) else pearson_corr
        rows.append({
            "descriptor": descriptor,
            "pearson_r": pearson_corr,
            "spearman_rho": spearman_corr,
            "pearson_p": pearson_p,
            "spearman_p": spearman_p,
            "n": n_used,
            "interpretation": _diagnostics_strength(abs(rank_corr)),
            "direction": _diagnostics_direction(rank_corr),
            "abs_spearman": abs(rank_corr) if np.isfinite(rank_corr) else 0.0,
        })

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("abs_spearman", ascending=False)
        .drop(columns=["abs_spearman"])
    )


def _diagnostics_mol_descriptor_options(desc_names):
    by_lower = {str(name).lower(): name for name in desc_names}
    quick = []
    for candidate in ("MolWt", "MolLogP"):
        found = by_lower.get(candidate.lower())
        if found is not None and found not in quick:
            quick.append(found)
    return quick + [name for name in desc_names if name not in quick]


def _diagnostics_structural_families(smiles_values, n_rows):
    if smiles_values is None:
        return np.array([t("diagnostics.residual_class_unknown")] * n_rows)
    try:
        annotations = error_analysis_structural_annotations(smiles_values)
        if "family" in annotations.columns and len(annotations) == n_rows:
            return annotations["family"].fillna(
                t("diagnostics.residual_class_unknown")
            ).astype(str).values
    except Exception:
        pass
    return np.array([t("diagnostics.residual_class_unknown")] * n_rows)


def _diagnostics_render_residual_vs_descriptor(context):
    matrix, desc_names = _diagnostics_descriptor_matrix(context)
    y_true = context.get("y_all_current")
    model = context.get("model")

    if matrix is None or y_true is None or model is None:
        st.info(t("diagnostics.residual_descriptor_unavailable"))
        return

    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(model.predict(context.get("X_scaled")), dtype=float)
    except Exception as exc:
        user_message = _log_dimension_failure(
            "RESIDUAL_DIAGNOSTICS",
            exc,
            context,
            "residual_diagnostics_failed",
        )
        st.warning(user_message)
        return

    n_rows = min(len(y_true), len(y_pred), matrix.shape[0])
    if n_rows < 3:
        st.info(t("diagnostics.residual_descriptor_unavailable"))
        return

    y_true = y_true[:n_rows]
    y_pred = y_pred[:n_rows]
    residuals = y_true - y_pred
    desc_df = pd.DataFrame(matrix[:n_rows], columns=desc_names)

    options = _diagnostics_mol_descriptor_options(desc_names)
    if not options:
        st.info(t("diagnostics.residual_descriptor_unavailable"))
        return

    with st.expander(t("diagnostics.residual_descriptor_title"), expanded=False):
        st.caption(t("diagnostics.residual_descriptor_caption"))

        col_select, col_opts = st.columns([2, 2])
        with col_select:
            selected_descriptor = st.selectbox(
                t("diagnostics.residual_descriptor_select"),
                options,
                key="diagnostics_residual_descriptor_select",
            )
        with col_opts:
            show_trend = st.checkbox(
                t("diagnostics.residual_show_trend"),
                value=True,
                key="diagnostics_residual_show_trend",
            )
            color_by_class = st.checkbox(
                t("diagnostics.residual_color_by_class"),
                value=False,
                key="diagnostics_residual_color_by_class",
            )
            label_outliers = st.checkbox(
                t("diagnostics.residual_label_outliers"),
                value=False,
                key="diagnostics_residual_label_outliers",
            )

        x_values = desc_df[selected_descriptor].replace(
            [np.inf, -np.inf],
            np.nan
        )
        plot_df = pd.DataFrame({
            "x": x_values,
            "residual": residuals,
            "row": np.arange(n_rows) + 1,
        }).dropna()

        if plot_df.empty or plot_df["x"].nunique() < 2:
            st.info(t("diagnostics.residual_descriptor_not_numeric"))
            return

        smiles_values = None
        data = context.get("data")
        smiles_col = context.get("smiles_col_current")
        valid_indices = context.get("valid_indices_current")
        if data is not None and smiles_col in getattr(data, "columns", []):
            try:
                smiles_values = data[smiles_col].iloc[valid_indices].values[:n_rows]
            except Exception:
                smiles_values = None

        if color_by_class:
            families = _diagnostics_structural_families(smiles_values, n_rows)
            plot_df["class"] = families[plot_df["row"].values - 1]
        else:
            plot_df["class"] = t("diagnostics.residual_class_all")

        fig_res, ax_res = plt.subplots(figsize=(8, 4.8))
        for class_name, group in plot_df.groupby("class", dropna=False):
            ax_res.scatter(
                group["x"],
                group["residual"],
                alpha=0.75,
                s=38,
                label=str(class_name),
            )

        ax_res.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=t("diagnostics.residual_zero_line"),
        )

        if show_trend and len(plot_df) >= 3 and plot_df["x"].nunique() >= 2:
            slope, intercept = np.polyfit(plot_df["x"], plot_df["residual"], 1)
            x_line = np.linspace(plot_df["x"].min(), plot_df["x"].max(), 100)
            ax_res.plot(
                x_line,
                slope * x_line + intercept,
                color="#d62728",
                linewidth=2,
                label=t("diagnostics.residual_linear_trend"),
            )

        if label_outliers:
            abs_res = plot_df["residual"].abs()
            threshold = abs_res.mean() + 2 * abs_res.std(ddof=0)
            outliers = plot_df[abs_res >= threshold]
            if outliers.empty:
                outliers = plot_df.loc[
                    abs_res.nlargest(min(5, len(plot_df))).index
                ]
            for _, row in outliers.head(12).iterrows():
                ax_res.annotate(
                    str(int(row["row"])),
                    (row["x"], row["residual"]),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

        ax_res.set_xlabel(selected_descriptor)
        ax_res.set_ylabel(t("diagnostics.residual_axis"))
        ax_res.set_title(
            t("diagnostics.residual_plot_title", descriptor=selected_descriptor)
        )
        ax_res.grid(True, alpha=0.25)
        if color_by_class or show_trend:
            ax_res.legend(fontsize=8)
        fig_res.tight_layout()
        st.pyplot(fig_res)

        pearson_corr, pearson_p, n_used = _diagnostics_correlation(
            plot_df["x"],
            plot_df["residual"],
        )
        spearman_corr, spearman_p = _diagnostics_spearman(
            plot_df["x"],
            plot_df["residual"],
        )
        rank_corr = spearman_corr if np.isfinite(spearman_corr) else pearson_corr
        strength = _diagnostics_strength(abs(rank_corr))
        direction = _diagnostics_direction(rank_corr)

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            t("diagnostics.residual_metric_pearson"),
            "n/a" if not np.isfinite(pearson_corr) else f"{pearson_corr:.3f}",
        )
        metric_cols[1].metric(
            t("diagnostics.residual_metric_spearman"),
            "n/a" if not np.isfinite(spearman_corr) else f"{spearman_corr:.3f}",
        )
        metric_cols[2].metric(
            t("diagnostics.residual_metric_p"),
            "n/a" if not np.isfinite(pearson_p) else f"{pearson_p:.3g}",
        )
        metric_cols[3].metric(
            t("diagnostics.residual_metric_n"),
            "n/a" if not np.isfinite(n_used) else int(n_used),
        )

        corr_for_message = rank_corr if np.isfinite(rank_corr) else 0.0
        st.info(t(
            "diagnostics.residual_interpretation",
            strength=strength,
            descriptor=selected_descriptor,
            r=corr_for_message,
            direction=direction,
        ))

        top_table = _diagnostics_make_residual_descriptor_table(desc_df, residuals)
        if top_table.empty:
            st.info(t("diagnostics.residual_top_empty"))
            return

        display = top_table.head(10).rename(columns={
            "descriptor": t("diagnostics.residual_col_descriptor"),
            "pearson_r": t("diagnostics.residual_col_pearson"),
            "spearman_rho": t("diagnostics.residual_col_spearman"),
            "pearson_p": t("diagnostics.residual_col_p_value"),
            "spearman_p": t("diagnostics.residual_col_spearman_p"),
            "n": t("diagnostics.residual_col_n"),
            "interpretation": t("diagnostics.residual_col_interpretation"),
            "direction": t("diagnostics.residual_col_direction"),
        })

        st.markdown(t("diagnostics.residual_top_title"))
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )


def render_model_diagnostics_section(context):
    """Рендерит диагностику уже обученной модели после валидации."""
    globals().update(context)
    st.header(t("diagnostics.header"))
    render_module_explanation("diagnostics")
    # ------------------------------------------------------------
    # Applicability Domain
    
    st.subheader(t('applicability_domain.header'))
    
    with st.expander(
        t('applicability_domain.expander_help_title'),
        expanded=False
    ):
        st.markdown(
            load_help_markdown(
                "applicability_domain_help.md"
            )
        )
    
    try:
        smiles_for_ad = data[smiles_col_current].iloc[valid_indices_current].values
    
        ad_table, ad_info = qspr_make_ad_table(
            X_train=X_scaled,
            smiles=smiles_for_ad,
            y=y_all_current,
            original_indices=valid_indices_current,
            desc_names=desc_names_current
        )
        
        st.session_state.ad_info = ad_info
        n_out_ad = int((ad_table[t('ad_table.col_status')] == t('ad_leverage.out_ad')).sum())
    
        # --------------------------------------------------
        # Williams Plot данные
    
        y_pred_ad = model.predict(X_scaled)
    
        williams_df = ad_build_williams_plot_df(
            y_true=y_all_current,
            y_pred=y_pred_ad,
            leverage=ad_info["leverage"],
            h_star=ad_info["threshold"]
        )
    
        summary = ad_williams_summary(
            williams_df
        )
    
        # --------------------------------------------------
        # Метрики AD + Williams
    
        col1, col2, col3, col4, col5, col6 = st.columns(6)
    
        with col1:
            st.metric(t('applicability_domain.metric_compounds'), ad_info["n"])
    
        with col2:
            st.metric(t('applicability_domain.metric_descriptors'), ad_info["p"])
    
        with col3:
            st.metric(
                t('applicability_domain.metric_threshold'),
                f"{ad_info['threshold']:.4f}"
            )
    
        with col4:
            st.metric(
                t('applicability_domain.metric_outside_ad'),
                n_out_ad
            )
    
        with col5:
            st.metric(
                t('applicability_domain.metric_high_residual'),
                summary["high_residual"]
            )
    
        with col6:
            st.metric(
                t('applicability_domain.metric_critical_points'),
                summary["critical"]
            )
    
        # --------------------------------------------------
        # Leverage plot
    
        fig_ad, ax_ad = plt.subplots(figsize=(7, 4))
    
        ax_ad.scatter(
            range(len(ad_table)),
            ad_table[t('ad_table.col_leverage')],
            alpha=0.75,
            s=35
        )
    
        ax_ad.axhline(
            ad_info["threshold"],
            color="r",
            linestyle="--",
            label=t('applicability_domain.leverage_plot_threshold_label')
        )
    
        ax_ad.set_xlabel(t('applicability_domain.leverage_plot_xlabel'))
        ax_ad.set_ylabel(t('applicability_domain.leverage_plot_ylabel'))
        ax_ad.set_title(t('applicability_domain.leverage_plot_title'))
        ax_ad.legend(fontsize=8)
        ax_ad.grid(True, alpha=0.25)
        fig_ad.tight_layout()
    
        # --------------------------------------------------
        # Williams plot
    
        fig_williams = ad_make_williams_plot(
            williams_df,
            ad_info["threshold"]
        )
    
        st.subheader(t('applicability_domain.subheader_ad_williams'))
    
        col_ad_left, col_ad_right = st.columns(2)
    
        with col_ad_left:
            st.pyplot(fig_ad)
    
        with col_ad_right:
            st.pyplot(fig_williams)
    
        # --------------------------------------------------
        # Критические точки Williams Plot
    
        critical_df = williams_df[
            williams_df["Critical Point"]
        ].copy()
        
        high_residual_df = williams_df[
            williams_df["High Residual"]
        ].copy()
    
        if len(high_residual_df) > 0:
            high_residual_df[t('applicability_domain.col_train_index')] = (
                high_residual_df.index + 1
            )
    
            if smiles_for_ad is not None:
                high_residual_df["SMILES"] = np.array(
                    smiles_for_ad
                )[high_residual_df.index]
    
            message = t('applicability_domain.warning_high_residual', count=len(high_residual_df))
            st.warning(message)
            if "log_streamlit_message" in globals():
                log_streamlit_message(
                    "APPLICABILITY_DOMAIN",
                    message,
                    level="warning",
                    details={"count": len(high_residual_df), **_diagnostics_dimension_context(context)},
                    event="ad_high_residual_warning",
                )
    
            with st.expander(t('applicability_domain.expander_high_residual'), expanded=False):
                st.dataframe(
                    high_residual_df[
                        [
                            c for c in [
                                t('applicability_domain.col_train_index'),
                                "SMILES",
                                "Leverage h",
                                "Standardized Residual"
                            ]
                            if c in high_residual_df.columns
                        ]
                    ],
                    width="stretch",
                    hide_index=True
                )
    
            if "SMILES" in high_residual_df.columns:
                show_molecule_grid_from_table(
                    table_df=high_residual_df,
                    title=t('applicability_domain.mol_grid_high_residual'),
                    smiles_col="SMILES",
                    target_col=None,
                    max_molecules=100,
                    key_prefix="williams_high_residual"
                )
        
        if len(critical_df) > 0:
            critical_df[t('applicability_domain.col_train_index')] = (
                np.arange(len(critical_df)) + 1
            )
    
            if smiles_for_ad is not None:
                critical_df["SMILES"] = np.array(smiles_for_ad)[
                    critical_df.index
                ]
    
            message = t('applicability_domain.error_critical', count=len(critical_df))
            st.error(message)
            if "log_streamlit_message" in globals():
                log_streamlit_message(
                    "APPLICABILITY_DOMAIN",
                    message,
                    level="error",
                    details={"count": len(critical_df), **_diagnostics_dimension_context(context)},
                    event="ad_critical_points",
                )
    
            with st.expander(t('applicability_domain.expander_critical'), expanded=False):
                show_cols = [
                    col for col in [
                        t('applicability_domain.col_train_index'),
                        "SMILES",
                        "Leverage h",
                        "Standardized Residual",
                        "Outside AD",
                        "High Residual"
                    ]
                    if col in critical_df.columns
                ]
    
                st.dataframe(
                    critical_df[show_cols],
                    width="stretch",
                    hide_index=True
                )
    
                critical_csv = (
                    critical_df[show_cols]
                    .to_csv(index=False)
                    .encode("utf-8")
                )
    
                st.download_button(
                    t('applicability_domain.download_critical'),
                    critical_csv,
                    "williams_critical_points.csv",
                    "text/csv"
                )
    
            if "SMILES" in critical_df.columns:
                show_molecule_grid_from_table(
                    table_df=critical_df,
                    title=t('applicability_domain.mol_grid_critical'),
                    target_col=None,
                    smiles_col="SMILES",
                    max_molecules=100,
                    key_prefix="williams_critical_molecules"
                )
    
        else:
            st.success(t('applicability_domain.success_no_critical'))
    
        if n_out_ad > 0:
            message = t('applicability_domain.warning_outside_ad', count=n_out_ad)
            st.warning(message)
            if "log_streamlit_message" in globals():
                log_streamlit_message(
                    "APPLICABILITY_DOMAIN",
                    message,
                    level="warning",
                    details={"count": n_out_ad, **_diagnostics_dimension_context(context)},
                    event="ad_outside_warning",
                )
        else:
            st.success(t('applicability_domain.success_all_in_ad'))
    
        with st.expander(t('applicability_domain.expander_show_table'), expanded=False):
            st.dataframe(
                ad_table,
                width="stretch",
                hide_index=True
            )
    
        ad_csv = ad_table.to_csv(index=False).encode("utf-8")
    
        st.download_button(
            t('applicability_domain.download_ad_csv'),
            ad_csv,
            "applicability_domain_leverage.csv",
            "text/csv"
        )
    
        if n_out_ad > 0:
            ad_outside = ad_table[
                ad_table[t('ad_table.col_status')] == t('ad_leverage.out_ad')
            ].copy()
    
            show_molecule_grid_from_table(
                table_df=ad_outside,
                title=t('applicability_domain.mol_grid_outside_ad'),
                target_col=t('applicability_domain.property_value'),
                smiles_col="SMILES",
                max_molecules=100,
                key_prefix="ad_outside_molecules"
            )
    
    except Exception as e:
        user_message = _log_dimension_failure(
            "APPLICABILITY_DOMAIN",
            e,
            context,
            "ad_calculation_failed",
        )
        st.warning(user_message)

    # ------------------------------------------------------------
    # Residual vs descriptor diagnostics

    _diagnostics_render_residual_vs_descriptor(context)
    
    # ------------------------------------------------------------
    # Unified descriptor importance
    
    st.subheader(t('feature_importance.title'))
    
    with st.expander(t('feature_importance.expander'), expanded=False):
        st.caption(t('feature_importance.causality_warning'))
    
        native_feature_names = list(
            model_data.get("selected_desc_names", desc_names_current)
        )
        coefficient_table = descriptor_coefficient_importance(
            model=model,
            feature_names=native_feature_names,
        )
        native_importance_table = descriptor_native_importance(
            model=model,
            feature_names=native_feature_names,
        )
        importance_validation_completed = any([
            model_name in st.session_state.get("holdout_results_dict", {}),
            model_name in st.session_state.get("kfold_results_dict", {}),
            model_name in st.session_state.get("loo_results_dict", {}),
            bool(st.session_state.get("ext_validation_result")),
        ])
    
        importance_tabs = st.tabs([
            t('feature_importance.tab_native'),
            t('feature_importance.tab_permutation'),
            t('feature_importance.tab_shap'),
            t('feature_importance.tab_summary'),
        ])
    
        with importance_tabs[0]:
            if not coefficient_table.empty:
                st.markdown(t('feature_importance.coefficients_title'))
                coefficient_display = coefficient_table.rename(columns={
                    "descriptor": t('feature_importance.col_descriptor'),
                    "coefficient": t('feature_importance.col_coefficient'),
                    "absolute_importance": t('feature_importance.col_absolute'),
                    "method": t('feature_importance.col_method'),
                })
                st.dataframe(
                    coefficient_display,
                    width="stretch",
                    hide_index=True,
                )
    
                top_coefficients = coefficient_table.head(20).iloc[::-1]
                fig_coef, ax_coef = plt.subplots(
                    figsize=(8, max(4, 0.28 * len(top_coefficients)))
                )
                colors_coef = [
                    "#d95f5f" if value < 0 else "#3f8f6b"
                    for value in top_coefficients["coefficient"]
                ]
                ax_coef.barh(
                    top_coefficients["descriptor"],
                    top_coefficients["coefficient"],
                    color=colors_coef,
                )
                ax_coef.axvline(0.0, color="black", linewidth=0.8)
                ax_coef.set_xlabel(t('feature_importance.col_coefficient'))
                ax_coef.set_title(t('feature_importance.coefficients_plot'))
                fig_coef.tight_layout()
                st.pyplot(fig_coef)
                plt.close(fig_coef)
    
            if not native_importance_table.empty:
                st.markdown(t('feature_importance.native_title'))
                native_display = native_importance_table.rename(columns={
                    "descriptor": t('feature_importance.col_descriptor'),
                    "importance": t('feature_importance.col_importance'),
                    "absolute_importance": t('feature_importance.col_absolute'),
                    "method": t('feature_importance.col_method'),
                })
                st.dataframe(
                    native_display,
                    width="stretch",
                    hide_index=True,
                )
                st.warning(t('feature_importance.native_warning'))
    
            if coefficient_table.empty and native_importance_table.empty:
                st.info(t('feature_importance.no_native'))
    
        with importance_tabs[1]:
            holdout_result = st.session_state.get(
                "holdout_results_dict", {}
            ).get(model_name)
    
            if holdout_result is not None:
                permutation_source = "holdout"
                st.success(t('feature_importance.holdout_source'))
            elif importance_validation_completed:
                permutation_source = "validated_training"
                st.warning(
                    t('feature_importance.validated_training_source')
                )
            else:
                permutation_source = None
                st.info(t('feature_importance.validation_required'))
    
            col_perm_1, col_perm_2, col_perm_3 = st.columns(3)
    
            with col_perm_1:
                permutation_scoring_label = st.selectbox(
                    t('feature_importance.scoring'),
                    [
                        t('feature_importance.scoring_rmse'),
                        t('feature_importance.scoring_r2'),
                    ],
                    disabled=not importance_validation_completed,
                    key=f"descriptor_importance_scoring_{model_name}",
                )
    
            with col_perm_2:
                permutation_repeats = st.number_input(
                    t('feature_importance.repeats'),
                    min_value=5,
                    max_value=200,
                    value=30,
                    step=5,
                    disabled=not importance_validation_completed,
                    key=f"descriptor_importance_repeats_{model_name}",
                )
    
            with col_perm_3:
                permutation_top_n = st.number_input(
                    t('feature_importance.top_n'),
                    min_value=5,
                    max_value=max(5, min(100, len(desc_names_current))),
                    value=min(20, max(5, len(desc_names_current))),
                    step=5,
                    disabled=not importance_validation_completed,
                    key=f"descriptor_importance_top_n_{model_name}",
                )
    
            grouped_importance_enabled = st.checkbox(
                t('feature_importance.group_correlated'),
                value=True,
                disabled=not importance_validation_completed,
                key=f"descriptor_grouped_importance_{model_name}",
            )
            correlation_threshold = st.slider(
                t('feature_importance.correlation_threshold'),
                min_value=0.70,
                max_value=0.99,
                value=0.90,
                step=0.01,
                disabled=(
                    not grouped_importance_enabled
                    or not importance_validation_completed
                ),
                key=f"descriptor_group_threshold_{model_name}",
            )
    
            scoring_name = (
                "r2"
                if permutation_scoring_label == t('feature_importance.scoring_r2')
                else "neg_root_mean_squared_error"
            )
    
            if st.button(
                t('feature_importance.calculate_button'),
                key=f"calculate_descriptor_importance_{model_name}",
                type="primary",
                disabled=not importance_validation_completed,
            ):
                try:
                    if holdout_result is not None:
                        train_idx = np.asarray(
                            holdout_result["train_idx"], dtype=int
                        )
                        test_idx = np.asarray(
                            holdout_result["test_idx"], dtype=int
                        )
                        X_train_importance = np.asarray(
                            X_all_current, dtype=float
                        )[train_idx]
                        X_eval_importance = np.asarray(
                            X_all_current, dtype=float
                        )[test_idx]
                        y_train_importance = np.asarray(
                            y_all_current, dtype=float
                        )[train_idx]
                        y_eval_importance = np.asarray(
                            y_all_current, dtype=float
                        )[test_idx]
                        importance_model = descriptor_refit_for_holdout(
                            model=model,
                            scaler=scaler,
                            X_train=X_train_importance,
                            y_train=y_train_importance,
                        )
                        permutation_names = list(desc_names_current)
                    elif isinstance(model, Pipeline):
                        importance_model = model
                        X_eval_importance = np.asarray(
                            X_all_current, dtype=float
                        )
                        y_eval_importance = np.asarray(
                            y_all_current, dtype=float
                        )
                        permutation_names = list(desc_names_current)
                    else:
                        importance_model = model
                        X_eval_importance = np.asarray(X_scaled, dtype=float)
                        y_eval_importance = np.asarray(
                            y_all_current, dtype=float
                        )
                        permutation_names = native_feature_names
    
                    permutation_table = descriptor_permutation_importance(
                        model=importance_model,
                        X=X_eval_importance,
                        y=y_eval_importance,
                        feature_names=permutation_names,
                        scoring=scoring_name,
                        n_repeats=int(permutation_repeats),
                        random_state=42,
                    )
    
                    grouped_table = pd.DataFrame()
                    if grouped_importance_enabled:
                        grouped_table = (
                            descriptor_grouped_permutation_importance(
                                model=importance_model,
                                X=X_eval_importance,
                                y=y_eval_importance,
                                feature_names=permutation_names,
                                threshold=float(correlation_threshold),
                                scoring=scoring_name,
                                n_repeats=int(permutation_repeats),
                                random_state=42,
                            )
                        )
    
                    st.session_state[
                        f"descriptor_importance_result_{model_name}"
                    ] = {
                        "permutation": permutation_table,
                        "grouped": grouped_table,
                        "source": permutation_source,
                        "scoring": scoring_name,
                    }
                except Exception as e:
                    st.error(
                        t('feature_importance.calculation_error', error=e)
                    )
    
            importance_result = st.session_state.get(
                f"descriptor_importance_result_{model_name}"
            )
            if not importance_validation_completed:
                importance_result = None
    
            if importance_result:
                permutation_table = importance_result["permutation"]
                grouped_table = importance_result.get(
                    "grouped", pd.DataFrame()
                )
    
                permutation_display = permutation_table.rename(columns={
                    "descriptor": t('feature_importance.col_descriptor'),
                    "importance_mean": t('feature_importance.col_mean'),
                    "importance_std": t('feature_importance.col_std'),
                    "positive_repeats_fraction": t(
                        'feature_importance.col_positive_fraction'
                    ),
                    "method": t('feature_importance.col_method'),
                })
                st.dataframe(
                    permutation_display,
                    width="stretch",
                    hide_index=True,
                )
    
                top_permutation = permutation_table.head(
                    int(permutation_top_n)
                ).iloc[::-1]
                fig_perm, ax_perm = plt.subplots(
                    figsize=(8, max(4, 0.30 * len(top_permutation)))
                )
                ax_perm.barh(
                    top_permutation["descriptor"],
                    top_permutation["importance_mean"],
                    xerr=top_permutation["importance_std"],
                    color="#3977a8",
                    alpha=0.85,
                )
                ax_perm.axvline(0.0, color="black", linewidth=0.8)
                ax_perm.set_xlabel(
                    t('feature_importance.permutation_axis')
                )
                ax_perm.set_title(
                    t('feature_importance.permutation_plot')
                )
                fig_perm.tight_layout()
                st.pyplot(fig_perm)
                plt.close(fig_perm)
    
                negative_count = int(
                    (permutation_table["importance_mean"] < 0).sum()
                )
                if negative_count:
                    st.info(
                        t(
                            'feature_importance.negative_info',
                            count=negative_count,
                        )
                    )
    
                if not grouped_table.empty:
                    st.markdown(
                        t('feature_importance.grouped_title')
                    )
                    grouped_display = grouped_table.rename(columns={
                        "group_id": t('feature_importance.col_group'),
                        "descriptors": t(
                            'feature_importance.col_descriptors'
                        ),
                        "n_descriptors": t(
                            'feature_importance.col_n_descriptors'
                        ),
                        "importance_mean": t(
                            'feature_importance.col_mean'
                        ),
                        "importance_std": t(
                            'feature_importance.col_std'
                        ),
                        "positive_repeats_fraction": t(
                            'feature_importance.col_positive_fraction'
                        ),
                        "method": t('feature_importance.col_method'),
                    })
                    st.dataframe(
                        grouped_display,
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        t('feature_importance.correlated_warning')
                    )
    
                st.download_button(
                    t('feature_importance.download_button'),
                    permutation_table.to_csv(index=False).encode("utf-8"),
                    f"descriptor_importance_{model_name}.csv",
                    "text/csv",
                    key=f"download_descriptor_importance_{model_name}",
                )
    
        with importance_tabs[2]:
            if not shap_available:
                st.info(t('feature_importance.shap_unavailable'))
            else:
                st.caption(t('feature_importance.shap_caption'))
                shap_sample_size = st.number_input(
                    t('feature_importance.shap_sample_size'),
                    min_value=5,
                    max_value=max(5, min(500, len(y_all_current))),
                    value=min(100, max(5, len(y_all_current))),
                    step=5,
                    key=f"descriptor_shap_sample_{model_name}",
                )
    
                if st.button(
                    t('feature_importance.shap_button'),
                    key=f"calculate_descriptor_shap_{model_name}",
                ):
                    try:
                        sample_size = min(
                            int(shap_sample_size),
                            len(y_all_current),
                        )
                        rng_shap = np.random.default_rng(42)
                        indices_sample = rng_shap.choice(
                            len(y_all_current),
                            sample_size,
                            replace=False,
                        )
    
                        if isinstance(model, Pipeline):
                            X_shap = np.asarray(
                                X_all_current, dtype=float
                            )[indices_sample]
                            for step_name, step in model.steps[:-1]:
                                X_shap = step.transform(X_shap)
                            shap_model = descriptor_importance_final_estimator(
                                model
                            )
                            shap_names = (
                                descriptor_importance_model_feature_names(
                                    model,
                                    desc_names_current,
                                )
                            )
                        else:
                            X_shap = np.asarray(
                                X_scaled, dtype=float
                            )[indices_sample]
                            shap_model = model
                            shap_names = native_feature_names
    
                        if hasattr(shap_model, "feature_importances_"):
                            explainer = shap.TreeExplainer(shap_model)
                        elif hasattr(shap_model, "coef_"):
                            background = X_shap[
                                :min(100, len(X_shap))
                            ]
                            explainer = shap.LinearExplainer(
                                shap_model, background
                            )
                        else:
                            X_shap = X_shap[:min(30, len(X_shap))]
                            background = shap.sample(
                                X_shap, min(50, len(X_shap))
                            )
                            explainer = shap.KernelExplainer(
                                shap_model.predict, background
                            )
    
                        if isinstance(explainer, shap.KernelExplainer):
                            shap_values = explainer.shap_values(
                                X_shap,
                                nsamples=min(
                                    200,
                                    2 * X_shap.shape[1] + 1,
                                ),
                            )
                        else:
                            shap_values = explainer.shap_values(X_shap)
    
                        if hasattr(shap_values, "values"):
                            shap_array = np.asarray(
                                shap_values.values, dtype=float
                            )
                        else:
                            shap_array = np.asarray(
                                shap_values, dtype=float
                            )
                        if shap_array.ndim == 3:
                            shap_array = shap_array[..., 0]
                        if shap_array.ndim == 1:
                            shap_array = shap_array.reshape(1, -1)
                        if shap_array.shape[1] != len(shap_names):
                            raise ValueError(
                                "Размерность SHAP не совпадает с числом "
                                "дескрипторов модели."
                            )
    
                        shap_table = pd.DataFrame({
                            "descriptor": list(shap_names),
                            "mean_abs_shap": np.mean(
                                np.abs(shap_array), axis=0
                            ),
                            "mean_shap": np.mean(shap_array, axis=0),
                            "shap_std": np.std(
                                shap_array, axis=0, ddof=0
                            ),
                            "method": "shap",
                        }).sort_values(
                            "mean_abs_shap", ascending=False
                        ).reset_index(drop=True)
                        st.session_state[
                            f"descriptor_shap_result_{model_name}"
                        ] = shap_table
    
                        shap.summary_plot(
                            shap_values,
                            X_shap,
                            feature_names=shap_names,
                            show=False,
                            plot_type="bar",
                            max_display=20,
                        )
                        fig_shap_bar = plt.gcf()
                        st.pyplot(fig_shap_bar)
                        plt.close(fig_shap_bar)
    
                        shap.summary_plot(
                            shap_values,
                            X_shap,
                            feature_names=shap_names,
                            show=False,
                            max_display=20,
                        )
                        fig_shap_summary = plt.gcf()
                        st.pyplot(fig_shap_summary)
                        plt.close(fig_shap_summary)
                    except Exception as e:
                        st.warning(
                            t('feature_importance.shap_error', error=e)
                        )
    
                shap_table = st.session_state.get(
                    f"descriptor_shap_result_{model_name}"
                )
                if isinstance(shap_table, pd.DataFrame) and not shap_table.empty:
                    shap_display = shap_table.rename(columns={
                        "descriptor": t('feature_importance.col_descriptor'),
                        "mean_abs_shap": t(
                            'feature_importance.col_mean_abs_shap'
                        ),
                        "mean_shap": t('feature_importance.col_mean_shap'),
                        "shap_std": t('feature_importance.col_std'),
                        "method": t('feature_importance.col_method'),
                    })
                    st.dataframe(
                        shap_display,
                        width="stretch",
                        hide_index=True,
                    )
    
        with importance_tabs[3]:
            if not importance_validation_completed:
                st.info(t('feature_importance.validation_required'))
            permutation_result = st.session_state.get(
                f"descriptor_importance_result_{model_name}",
                {},
            )
            permutation_table_summary = (
                permutation_result.get("permutation")
                if isinstance(permutation_result, dict)
                else None
            )
            shap_table_summary = st.session_state.get(
                f"descriptor_shap_result_{model_name}"
            )
            unified_importance = descriptor_unified_importance_table(
                coefficient_table=coefficient_table,
                native_table=native_importance_table,
                permutation_table=permutation_table_summary,
                shap_table=shap_table_summary,
            )
            if not importance_validation_completed:
                unified_importance = pd.DataFrame()
            st.session_state[
                f"descriptor_importance_unified_{model_name}"
            ] = unified_importance
    
            if unified_importance.empty:
                st.info(t('feature_importance.summary_empty'))
            else:
                stability_labels = {
                    "stable": t('feature_importance.stability_stable'),
                    "moderate": t('feature_importance.stability_moderate'),
                    "unstable": t('feature_importance.stability_unstable'),
                    "not_evaluated": t(
                        'feature_importance.stability_not_evaluated'
                    ),
                }
                unified_display = unified_importance.copy()
                unified_display["stability_status"] = (
                    unified_display["stability_status"]
                    .map(stability_labels)
                    .fillna(unified_display["stability_status"])
                )
                unified_display = unified_display.rename(columns={
                    "combined_rank": t('feature_importance.col_rank'),
                    "descriptor": t('feature_importance.col_descriptor'),
                    "combined_score": t(
                        'feature_importance.col_combined_score'
                    ),
                    "methods_available": t(
                        'feature_importance.col_methods_available'
                    ),
                    "stability_status": t(
                        'feature_importance.col_stability'
                    ),
                })
                st.caption(t('feature_importance.summary_caption'))
                st.dataframe(
                    unified_display,
                    width="stretch",
                    hide_index=True,
                )
    
                top_unified = unified_importance.head(20).iloc[::-1]
                fig_unified, ax_unified = plt.subplots(
                    figsize=(8, max(4, 0.30 * len(top_unified)))
                )
                ax_unified.barh(
                    top_unified["descriptor"],
                    top_unified["combined_score"],
                    color="#3f8f6b",
                    alpha=0.88,
                )
                ax_unified.set_xlabel(
                    t('feature_importance.col_combined_score')
                )
                ax_unified.set_title(
                    t('feature_importance.summary_plot')
                )
                fig_unified.tight_layout()
                st.pyplot(fig_unified)
                plt.close(fig_unified)
    
                st.download_button(
                    t('feature_importance.download_unified'),
                    unified_importance.to_csv(index=False).encode("utf-8"),
                    f"descriptor_importance_unified_{model_name}.csv",
                    "text/csv",
                    key=f"download_descriptor_importance_unified_{model_name}",
                )
    
