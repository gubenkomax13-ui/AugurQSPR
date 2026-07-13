# -*- coding: utf-8 -*-
"""Final QSPR statistics summary UI."""

import json
import sys

import numpy as np
import pandas as pd
import streamlit as st

from modules.i18n import t
from modules.module_explain_ui import render_module_explanation


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _fmt_value(value, digits=4):
    if value is None:
        return t("final_stats.na")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return t("final_stats.na") if not np.isfinite(value) else round(float(value), digits)
    return value


def qspr_safe_dataframe_for_streamlit(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    out.columns = [str(col) for col in out.columns]
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(_safe_cell_to_text).astype("string")
    return out


def _safe_cell_to_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _section_rows(section_key, rows):
    section = t(section_key)
    return [
        {
            t("final_stats.col_section"): section,
            t("final_stats.col_metric"): metric,
            t("final_stats.col_value"): _fmt_value(value),
            t("final_stats.col_comment"): comment,
        }
        for metric, value, comment in rows
    ]


def _unavailable_row(section_key):
    return _section_rows(
        section_key,
        [(t("final_stats.metric_status"), t("final_stats.unavailable"), t("final_stats.unavailable_comment"))],
    )


def calculate_target_statistics(data, target_col, smiles_col=None, valid_indices=None):
    if not isinstance(data, pd.DataFrame) or data.empty or not target_col or target_col not in data.columns:
        return pd.DataFrame(_unavailable_row("final_stats.section_dataset")), pd.DataFrame()

    y = pd.to_numeric(data[target_col], errors="coerce")
    y_valid = y.dropna()
    missing = int(y.isna().sum())
    valid_count = int(len(y_valid))

    invalid_smiles = np.nan
    duplicate_smiles = np.nan
    if smiles_col and smiles_col in data.columns:
        smiles = data[smiles_col].astype(str).str.strip()
        empty_smiles = smiles.isin(["", "nan", "None"])
        duplicate_smiles = int(smiles[~empty_smiles].duplicated().sum())
        try:
            from rdkit import Chem

            invalid_smiles = int(
                sum(1 for value in smiles[~empty_smiles] if Chem.MolFromSmiles(str(value)) is None)
            )
        except Exception:
            invalid_smiles = t("final_stats.not_calculated")

    q1 = _safe_float(y_valid.quantile(0.25)) if valid_count else np.nan
    q3 = _safe_float(y_valid.quantile(0.75)) if valid_count else np.nan
    iqr = q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else np.nan

    skew = _safe_float(y_valid.skew()) if valid_count >= 3 else np.nan
    kurt = _safe_float(y_valid.kurt()) if valid_count >= 4 else np.nan
    std = _safe_float(y_valid.std(ddof=1)) if valid_count >= 2 else np.nan
    prop_range = _safe_float(y_valid.max() - y_valid.min()) if valid_count else np.nan

    comments = []
    if np.isfinite(skew):
        comments.append(
            t("final_stats.interpret_skew_asym")
            if abs(skew) >= 1
            else t("final_stats.interpret_skew_sym")
        )
    if np.isfinite(prop_range) and np.isfinite(std):
        comments.append(
            t("final_stats.interpret_range_wide")
            if std > abs(_safe_float(y_valid.mean())) * 0.5
            else t("final_stats.interpret_range_regular")
        )
    if np.isfinite(iqr):
        outliers = int(((y_valid < q1 - 1.5 * iqr) | (y_valid > q3 + 1.5 * iqr)).sum())
        if outliers:
            comments.append(t("final_stats.interpret_target_outliers", count=outliers))
    else:
        outliers = np.nan

    rows = [
        (t("final_stats.metric_rows_total"), int(len(data)), ""),
        (t("final_stats.metric_valid_compounds"), len(valid_indices) if valid_indices is not None else valid_count, ""),
        (t("final_stats.metric_target_missing"), missing, ""),
        (t("final_stats.metric_invalid_smiles"), invalid_smiles, ""),
        (t("final_stats.metric_duplicate_smiles"), duplicate_smiles, ""),
        (t("final_stats.metric_target_min"), _safe_float(y_valid.min()) if valid_count else np.nan, ""),
        (t("final_stats.metric_target_max"), _safe_float(y_valid.max()) if valid_count else np.nan, ""),
        (t("final_stats.metric_target_mean"), _safe_float(y_valid.mean()) if valid_count else np.nan, ""),
        (t("final_stats.metric_target_median"), _safe_float(y_valid.median()) if valid_count else np.nan, ""),
        (t("final_stats.metric_target_std"), std, ""),
        ("Q1", q1, ""),
        ("Q3", q3, ""),
        ("IQR", iqr, ""),
        ("Skewness", skew, ""),
        ("Kurtosis", kurt, ""),
        (t("final_stats.metric_target_outliers"), outliers, ""),
    ]
    summary = pd.DataFrame(_section_rows("final_stats.section_dataset", rows))
    interpretation = pd.DataFrame(
        {
            t("final_stats.col_comment"): comments or [t("final_stats.interpret_not_enough")]
        }
    )
    return summary, interpretation


def calculate_descriptor_statistics(X, y=None, desc_names=None):
    if X is None:
        return (
            pd.DataFrame(_unavailable_row("final_stats.section_descriptors")),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    try:
        X_arr = np.asarray(X, dtype=float)
    except Exception:
        return (
            pd.DataFrame(_unavailable_row("final_stats.section_descriptors")),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if X_arr.ndim != 2 or X_arr.size == 0:
        return (
            pd.DataFrame(_unavailable_row("final_stats.section_descriptors")),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    n_rows, n_cols = X_arr.shape
    names = list(desc_names or [])
    if len(names) != n_cols:
        names = [f"x{i + 1}" for i in range(n_cols)]

    finite = np.isfinite(X_arr)
    nan_count = int(np.isnan(X_arr).sum())
    inf_count = int(np.isinf(X_arr).sum())
    stds = np.nanstd(np.where(finite, X_arr, np.nan), axis=0)
    constant_count = int(np.sum(np.nan_to_num(stds, nan=0.0) <= 1e-12))
    near_constant_count = int(np.sum((np.nan_to_num(stds, nan=0.0) > 1e-12) & (stds <= 1e-6)))

    rows = [
        (t("final_stats.metric_descriptor_rows"), n_rows, ""),
        (t("final_stats.metric_descriptor_initial"), n_cols, ""),
        (t("final_stats.metric_descriptor_final"), n_cols, ""),
        (t("final_stats.metric_nan_count"), nan_count, ""),
        (t("final_stats.metric_inf_count"), inf_count, ""),
        (t("final_stats.metric_constant_features"), constant_count, ""),
        (t("final_stats.metric_near_constant_features"), near_constant_count, ""),
    ]

    clean = pd.DataFrame(X_arr, columns=names).replace([np.inf, -np.inf], np.nan)
    desc_table = pd.DataFrame(
        {
            t("final_stats.col_descriptor"): names,
            "mean": clean.mean(numeric_only=True).values,
            "std": clean.std(numeric_only=True).values,
            "min": clean.min(numeric_only=True).values,
            "max": clean.max(numeric_only=True).values,
            t("final_stats.col_missing_percent"): clean.isna().mean().values * 100,
            t("final_stats.col_zero_percent"): (clean == 0).mean().values * 100,
            "skewness": clean.skew(numeric_only=True).values,
        }
    )

    corr_table = pd.DataFrame()
    if y is not None:
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == n_rows:
            corr_rows = []
            y_series = pd.Series(y_arr)
            for name in names:
                x_series = pd.to_numeric(clean[name], errors="coerce")
                joined = pd.concat([x_series, y_series], axis=1).dropna()
                if len(joined) >= 3 and joined.iloc[:, 0].nunique() > 1:
                    corr_rows.append(
                        {
                            t("final_stats.col_descriptor"): name,
                            "Pearson r": joined.iloc[:, 0].corr(joined.iloc[:, 1], method="pearson"),
                            "Spearman rho": joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman"),
                        }
                    )
            corr_table = pd.DataFrame(corr_rows)
            if not corr_table.empty:
                corr_table = corr_table.assign(abs_r=corr_table["Pearson r"].abs()).sort_values(
                    "abs_r", ascending=False
                ).drop(columns=["abs_r"]).head(40)

    multicol_rows = []
    max_corr_cols = min(n_cols, 250)
    if max_corr_cols >= 2:
        corr = clean.iloc[:, :max_corr_cols].corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for threshold in (0.90, 0.95, 0.99):
            multicol_rows.append(
                (
                    t("final_stats.metric_pairs_above", threshold=threshold),
                    int((upper > threshold).sum().sum()),
                    t("final_stats.multicol_limited", n=max_corr_cols) if n_cols > max_corr_cols else "",
                )
            )
    multicol_table = pd.DataFrame(_section_rows("final_stats.section_descriptors", multicol_rows))

    summary = pd.DataFrame(_section_rows("final_stats.section_descriptors", rows))
    return summary, desc_table, corr_table, multicol_table


def _metrics_row(method, metrics, spread=""):
    if not isinstance(metrics, dict):
        metrics = {}
    r2 = metrics.get("R2", metrics.get("test_R2_mean", np.nan))
    rmse = metrics.get("RMSE", metrics.get("test_RMSE_mean", np.nan))
    mae = metrics.get("MAE", metrics.get("test_MAE_mean", np.nan))
    return {
        t("final_stats.col_method"): method,
        "R2/Q2": _fmt_value(_safe_float(r2)),
        "RMSE": _fmt_value(_safe_float(rmse)),
        "MAE": _fmt_value(_safe_float(mae)),
        t("final_stats.col_spread"): spread,
        t("final_stats.col_interpretation"): _interpret_model_quality(_safe_float(r2), _safe_float(rmse)),
    }


def _interpret_model_quality(r2, rmse):
    if not np.isfinite(r2):
        return t("final_stats.quality_undefined")
    if r2 >= 0.85:
        return t("final_stats.quality_excellent")
    if r2 >= 0.70:
        return t("final_stats.quality_good")
    if r2 >= 0.50:
        return t("final_stats.quality_acceptable")
    return t("final_stats.quality_weak")


def calculate_validation_statistics_from_session(model_name):
    model_data = st.session_state.get("trained_models", {}).get(model_name)
    if not model_data:
        return pd.DataFrame(_unavailable_row("final_stats.section_model")), pd.DataFrame()

    selected = model_data.get("selected_desc_names") or st.session_state.get("model_used_descriptor_names", [])
    rows = [
        (t("final_stats.metric_model_name"), model_name, ""),
        (t("final_stats.metric_training_objects"), len(model_data.get("y_pred", [])), ""),
        (t("final_stats.metric_model_descriptors"), len(selected), ""),
    ]
    summary = pd.DataFrame(_section_rows("final_stats.section_model", rows))

    validation_rows = [_metrics_row(t("final_stats.method_training"), model_data.get("metrics", {}))]

    current_checker = getattr(
        sys.modules.get("__main__"),
        "qspr_validation_result_is_current",
        None,
    )

    def current_result(store_name, kind):
        result = st.session_state.get(store_name, {}).get(model_name)
        if not isinstance(result, dict):
            return None
        if callable(current_checker):
            try:
                if not current_checker(model_name, result, kind):
                    return None
            except Exception:
                return None
        return result

    holdout = current_result("holdout_results_dict", "holdout")
    if holdout:
        validation_rows.append(_metrics_row("Hold-out", holdout.get("metrics_test", {})))
    kfold = current_result("kfold_results_dict", "kfold")
    if kfold:
        validation_rows.append(_metrics_row("K-Fold", kfold.get("metrics", {})))
    loo = current_result("loo_results_dict", "loo")
    if loo:
        validation_rows.append(_metrics_row("LOO", loo.get("metrics", {})))

    repeated = st.session_state.get("montecarlo_results_dict", {}).get(model_name)
    if isinstance(repeated, dict):
        summary_dict = repeated.get("summary", {})
        spread = _mean_std(summary_dict.get("test_R2_mean"), summary_dict.get("test_R2_std"))
        validation_rows.append(_metrics_row(t("final_stats.method_repeated_holdout"), summary_dict, spread=spread))

    bootstrap = current_result("bootstrap_results_dict", "bootstrap")
    if isinstance(bootstrap, dict):
        summary_dict = bootstrap.get("summary", {})
        validation_rows.append(
            {
                t("final_stats.col_method"): "Bootstrap OOB",
                "R2/Q2": _fmt_value(_safe_float(summary_dict.get("R2_OOB_mean"))),
                "RMSE": _fmt_value(_safe_float(summary_dict.get("RMSE_OOB_mean"))),
                "MAE": _fmt_value(_safe_float(summary_dict.get("MAE_OOB_mean"))),
                t("final_stats.col_spread"): _mean_std(
                    summary_dict.get("RMSE_OOB_mean"), summary_dict.get("RMSE_OOB_std")
                ),
                t("final_stats.col_interpretation"): _interpret_model_quality(
                    _safe_float(summary_dict.get("R2_OOB_mean")),
                    _safe_float(summary_dict.get("RMSE_OOB_mean")),
                ),
            }
        )

    return summary, pd.DataFrame(validation_rows)


def _mean_std(mean, std):
    mean = _safe_float(mean)
    std = _safe_float(std)
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        return str(_fmt_value(mean))
    return f"{_fmt_value(mean)} +/- {_fmt_value(std)}"


def calculate_residual_statistics(y_true, y_pred):
    if y_true is None or y_pred is None:
        return pd.DataFrame(_unavailable_row("final_stats.section_residuals")), pd.DataFrame()

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return pd.DataFrame(_unavailable_row("final_stats.section_residuals")), pd.DataFrame()

    residual = y_true - y_pred
    abs_error = np.abs(residual)
    q1 = np.nanquantile(residual, 0.25)
    q3 = np.nanquantile(residual, 0.75)
    mean_res = _safe_float(np.nanmean(residual))
    bias = _bias_text(mean_res, np.nanstd(y_true) if len(y_true) > 1 else np.nan)
    rows = [
        (t("final_stats.metric_mean_residual"), mean_res, bias),
        (t("final_stats.metric_median_residual"), _safe_float(np.nanmedian(residual)), ""),
        (t("final_stats.metric_std_residual"), _safe_float(np.nanstd(residual, ddof=1)), ""),
        ("MAE", _safe_float(np.nanmean(abs_error)), ""),
        ("RMSE", _safe_float(np.sqrt(np.nanmean(residual ** 2))), ""),
        (t("final_stats.metric_max_abs_error"), _safe_float(np.nanmax(abs_error)), ""),
        (t("final_stats.metric_residual_q1"), _safe_float(q1), ""),
        (t("final_stats.metric_residual_q3"), _safe_float(q3), ""),
        ("Skewness", _safe_float(pd.Series(residual).skew()), ""),
        ("Kurtosis", _safe_float(pd.Series(residual).kurt()), ""),
    ]
    summary = pd.DataFrame(_section_rows("final_stats.section_residuals", rows))
    residual_table = pd.DataFrame(
        {
            t("final_stats.col_experimental"): y_true,
            t("final_stats.col_predicted"): y_pred,
            t("final_stats.col_residual"): residual,
            t("final_stats.col_abs_error"): abs_error,
        }
    )
    return summary, residual_table


def _bias_text(mean_residual, y_std):
    if not np.isfinite(mean_residual):
        return ""
    tolerance = 0.05 * y_std if np.isfinite(y_std) and y_std > 0 else 1e-9
    if abs(mean_residual) <= tolerance:
        return t("final_stats.bias_none")
    if mean_residual > 0:
        return t("final_stats.bias_under")
    return t("final_stats.bias_over")


def calculate_ad_statistics_from_session():
    ad_info = st.session_state.get("ad_info")
    if not isinstance(ad_info, dict) or "leverage" not in ad_info:
        return pd.DataFrame(_unavailable_row("final_stats.section_residuals"))

    leverage = np.asarray(ad_info.get("leverage", []), dtype=float)
    threshold = _safe_float(ad_info.get("threshold"))
    outside = leverage > threshold if np.isfinite(threshold) else np.full(len(leverage), False)
    rows = [
        (t("final_stats.metric_ad_inside"), int((~outside).sum()), ""),
        (t("final_stats.metric_ad_outside"), int(outside.sum()), ""),
        (t("final_stats.metric_ad_outside_percent"), float(outside.mean() * 100) if len(outside) else np.nan, ""),
        (t("final_stats.metric_ad_mean_h"), _safe_float(np.nanmean(leverage)), ""),
        (t("final_stats.metric_ad_threshold"), threshold, ""),
        (t("final_stats.metric_ad_max_h"), _safe_float(np.nanmax(leverage)) if len(leverage) else np.nan, ""),
    ]
    return pd.DataFrame(_section_rows("final_stats.section_residuals", rows))


def build_final_statistics_summary(context):
    data = context.get("data")
    target_col = context.get("target_col") or st.session_state.get("target_col")
    smiles_col = context.get("smiles_col_current") or context.get("smiles_col")
    valid_indices = context.get("valid_indices_current") or context.get("valid_indices")
    X = context.get("X_all_current", context.get("X_all"))
    y = context.get("y_all_current", context.get("y_all"))
    desc_names = context.get("desc_names_current", context.get("desc_names"))
    model_name = context.get("model_name") or st.session_state.get("last_model_algorithm", "")
    model_data = st.session_state.get("trained_models", {}).get(model_name, {})

    dataset_summary, dataset_interpretation = calculate_target_statistics(
        data, target_col, smiles_col=smiles_col, valid_indices=valid_indices
    )
    descriptor_summary, descriptor_table, correlation_table, multicol_table = calculate_descriptor_statistics(
        X, y=y, desc_names=desc_names
    )
    model_summary, validation_table = calculate_validation_statistics_from_session(model_name)
    residual_summary, residual_table = calculate_residual_statistics(y, model_data.get("y_pred"))
    ad_summary = calculate_ad_statistics_from_session()

    return {
        "dataset": {
            "summary": dataset_summary,
            "interpretation": dataset_interpretation,
        },
        "descriptors": {
            "summary": descriptor_summary,
            "descriptor_table": descriptor_table,
            "correlation_table": correlation_table,
            "multicollinearity": multicol_table,
        },
        "model_validation": {
            "summary": model_summary,
            "validation_table": validation_table,
        },
        "residuals_ad": {
            "summary": pd.concat([residual_summary, ad_summary], ignore_index=True),
            "residual_table": residual_table,
        },
    }


def final_statistics_to_flat_dataframe(summary):
    frames = []
    for section in summary.values():
        for value in section.values():
            if isinstance(value, pd.DataFrame) and not value.empty:
                frames.append(value)
    if not frames:
        return pd.DataFrame(
            columns=[
                t("final_stats.col_section"),
                t("final_stats.col_metric"),
                t("final_stats.col_value"),
                t("final_stats.col_comment"),
            ]
        )
    return qspr_safe_dataframe_for_streamlit(pd.concat(frames, ignore_index=True, sort=False))


def final_statistics_to_json_bytes(summary):
    def convert(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.replace([np.inf, -np.inf], np.nan).where(pd.notna(obj), None).to_dict(orient="records")
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        return obj

    return json.dumps(convert(summary), ensure_ascii=False, indent=2).encode("utf-8")


def _render_table(df, max_rows=None, **kwargs):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info(t("final_stats.unavailable_comment"))
        return
    view = df.head(max_rows).copy() if max_rows else df
    st.dataframe(qspr_safe_dataframe_for_streamlit(view), width="stretch", hide_index=True, **kwargs)


def render_final_statistics_summary(context):
    st.header(t("final_stats.header"))
    render_module_explanation("final_statistics")
    summary = build_final_statistics_summary(context)
    flat = final_statistics_to_flat_dataframe(summary)

    tabs = st.tabs(
        [
            t("final_stats.tab_dataset"),
            t("final_stats.tab_descriptors"),
            t("final_stats.tab_model"),
            t("final_stats.tab_residuals"),
            t("final_stats.tab_export"),
        ]
    )

    with tabs[0]:
        _render_table(summary["dataset"]["summary"])
        interpretation = summary["dataset"]["interpretation"]
        if isinstance(interpretation, pd.DataFrame) and not interpretation.empty:
            st.caption(" ".join(interpretation.iloc[:, 0].astype(str).tolist()))

    with tabs[1]:
        _render_table(summary["descriptors"]["summary"])
        if not summary["descriptors"]["descriptor_table"].empty:
            st.markdown(t("final_stats.descriptor_table_title"))
            _render_table(summary["descriptors"]["descriptor_table"], max_rows=100)
        if not summary["descriptors"]["correlation_table"].empty:
            st.markdown(t("final_stats.correlation_table_title"))
            _render_table(summary["descriptors"]["correlation_table"], max_rows=40)
        if not summary["descriptors"]["multicollinearity"].empty:
            st.markdown(t("final_stats.multicollinearity_title"))
            _render_table(summary["descriptors"]["multicollinearity"])

    with tabs[2]:
        _render_table(summary["model_validation"]["summary"])
        _render_table(summary["model_validation"]["validation_table"])

    with tabs[3]:
        _render_table(summary["residuals_ad"]["summary"])
        if not summary["residuals_ad"]["residual_table"].empty:
            st.markdown(t("final_stats.residual_table_title"))
            _render_table(summary["residuals_ad"]["residual_table"], max_rows=100)

    with tabs[4]:
        _render_table(flat, max_rows=300)
        col_csv, col_json = st.columns(2)
        with col_csv:
            st.download_button(
                t("final_stats.download_csv"),
                flat.to_csv(index=False).encode("utf-8-sig"),
                "qspr_final_statistics.csv",
                "text/csv",
                key="download_qspr_final_statistics_csv",
            )
        with col_json:
            st.download_button(
                t("final_stats.download_json"),
                final_statistics_to_json_bytes(summary),
                "qspr_final_statistics.json",
                "application/json",
                key="download_qspr_final_statistics_json",
            )

    return summary
