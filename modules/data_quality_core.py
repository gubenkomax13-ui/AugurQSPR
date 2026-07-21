# -*- coding: utf-8 -*-
"""Input data quality control for Augur QSPR.

The module is intentionally UI-free: it detects censored target values,
unit conversion issues, repeated measurements, and outliers, while keeping
an explicit audit trail for every changed or excluded row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


UNIT_TO_MOLAR = {
    "m": 1.0,
    "mol/l": 1.0,
    "mol_l": 1.0,
    "molar": 1.0,
    "mm": 1e-3,
    "mmol/l": 1e-3,
    "mmol_l": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "umol/l": 1e-6,
    "umol_l": 1e-6,
    "nm": 1e-9,
    "nmol/l": 1e-9,
    "nmol_l": 1e-9,
    "pm": 1e-12,
    "pmol/l": 1e-12,
    "pmol_l": 1e-12,
}

DISPLAY_UNITS = ["M", "mM", "uM", "nM", "pM"]


def normalize_unit(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("mol/l", "mol/l")
    return text


def unit_factor_to_molar(value) -> float | None:
    norm = normalize_unit(value)
    if not norm:
        return None
    return UNIT_TO_MOLAR.get(norm)


def display_unit_to_factor(unit: str) -> float:
    factor = unit_factor_to_molar(unit)
    if factor is None:
        raise ValueError(f"Unsupported unit: {unit}")
    return float(factor)


@dataclass
class ParsedTarget:
    value: float
    censor: str
    raw: str


def parse_target_value(value) -> ParsedTarget:
    raw = "" if value is None else str(value).strip()
    if raw.lower() in {"", "nan", "none", "null"}:
        return ParsedTarget(np.nan, "", raw)

    text = raw.replace(",", ".")
    text = text.replace("≤", "<=").replace("≥", ">=")
    censor = ""
    match = re.match(r"^\s*(<=|>=|<|>)\s*(.+?)\s*$", text)
    if match:
        censor = match.group(1)
        text = match.group(2)

    number_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not number_match:
        return ParsedTarget(np.nan, censor, raw)

    try:
        numeric = float(number_match.group(0))
    except Exception:
        numeric = np.nan
    return ParsedTarget(numeric, censor, raw)


def _safe_numeric_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed = series.map(parse_target_value)
    values = parsed.map(lambda item: item.value)
    censors = parsed.map(lambda item: item.censor)
    return pd.to_numeric(values, errors="coerce"), censors.astype(str)


def _append_audit(audit_rows: list[dict], row, row_index: int, action: str, reason: str, detail: str = ""):
    audit_rows.append(
        {
            "original_row": int(row_index) + 1,
            "action": action,
            "reason": reason,
            "detail": detail,
        }
    )


def _row_key_columns(df: pd.DataFrame, smiles_col: str, context_cols: Iterable[str] | None = None) -> list[str]:
    cols = [smiles_col] if smiles_col in df.columns else []
    for col in context_cols or []:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def _collapse_replicates(
    work: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    policy: str,
    audit_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not group_cols:
        return work, pd.DataFrame()

    replicate_rows = []
    output_rows = []
    policy = str(policy or "median").lower()

    for _, group in work.groupby(group_cols, dropna=False, sort=False):
        values = pd.to_numeric(group[target_col], errors="coerce")
        finite = values.dropna()
        if len(group) <= 1:
            output_rows.append(group.iloc[0].copy())
            continue

        mean_value = float(finite.mean()) if len(finite) else np.nan
        median_value = float(finite.median()) if len(finite) else np.nan
        std_value = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0 if len(finite) == 1 else np.nan
        min_value = float(finite.min()) if len(finite) else np.nan
        max_value = float(finite.max()) if len(finite) else np.nan
        range_value = float(max_value - min_value) if np.isfinite(min_value) and np.isfinite(max_value) else np.nan

        row = group.iloc[0].copy()
        if policy == "mean":
            merged_value = mean_value
        elif policy == "keep_all":
            for _, item in group.iterrows():
                output_rows.append(item.copy())
            merged_value = np.nan
        else:
            merged_value = median_value

        replicate_rows.append(
            {
                "group_key": " | ".join(str(group.iloc[0].get(c, "")) for c in group_cols),
                "n_measurements": int(len(group)),
                "mean": mean_value,
                "median": median_value,
                "std": std_value,
                "min": min_value,
                "max": max_value,
                "range": range_value,
                "merge_policy": policy,
                "source_rows": "; ".join(str(int(i) + 1) for i in group.index.tolist()),
            }
        )

        if policy != "keep_all":
            row[target_col] = merged_value
            row["_data_quality_replicates_merged"] = int(len(group))
            row["_data_quality_replicate_source_rows"] = "; ".join(str(int(i) + 1) for i in group.index.tolist())
            output_rows.append(row)
            for idx, item in group.iloc[1:].iterrows():
                _append_audit(
                    audit_rows,
                    item,
                    int(idx),
                    "excluded",
                    "repeated_measurement_merged",
                    f"merged into representative row by {policy}; group columns: {', '.join(group_cols)}",
                )

    if not replicate_rows:
        return work, pd.DataFrame()

    return pd.DataFrame(output_rows).reset_index(drop=True), pd.DataFrame(replicate_rows)


def run_input_data_quality_control(
    df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    unit_col: str | None = None,
    target_unit: str | None = None,
    convert_units: bool = True,
    censored_policy: str = "exclude",
    replicate_policy: str = "median",
    context_cols: Iterable[str] | None = None,
    outlier_iqr_multiplier: float = 3.0,
    technical_min: float | None = None,
    technical_max: float | None = None,
) -> dict:
    """Run mandatory input-QC checks and return cleaned data plus audit tables."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        empty = pd.DataFrame()
        return {
            "cleaned_df": empty,
            "audit_df": empty,
            "excluded_df": empty,
            "summary_df": empty,
            "replicates_df": empty,
            "conversions_df": empty,
            "outliers_df": empty,
        }
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    work = df.copy().reset_index(drop=True)
    audit_rows: list[dict] = []
    conversion_rows: list[dict] = []

    raw_target_col = f"{target_col}__raw_before_qc"
    censor_col = f"{target_col}__censor"
    work[raw_target_col] = work[target_col]
    numeric_values, censors = _safe_numeric_series(work[target_col])
    work[target_col] = numeric_values
    work[censor_col] = censors

    exclude_mask = pd.Series(False, index=work.index)
    exclude_reasons = pd.Series("", index=work.index, dtype=object)

    missing_mask = work[target_col].isna()
    for idx, row in work.loc[missing_mask].iterrows():
        exclude_mask.loc[idx] = True
        exclude_reasons.loc[idx] = "target_not_numeric_or_missing"
        _append_audit(audit_rows, row, int(idx), "excluded", "target_not_numeric_or_missing")

    censored_mask = work[censor_col].astype(str).str.strip() != ""
    if censored_mask.any():
        for idx, row in work.loc[censored_mask].iterrows():
            detail = f"{raw_target_col}={row.get(raw_target_col, '')}; censor={row.get(censor_col, '')}"
            if censored_policy == "exclude":
                exclude_mask.loc[idx] = True
                exclude_reasons.loc[idx] = _join_reason(exclude_reasons.loc[idx], "censored_value")
                _append_audit(audit_rows, row, int(idx), "excluded", "censored_value", detail)
            else:
                _append_audit(audit_rows, row, int(idx), "kept_with_flag", "censored_value", detail)

    if convert_units and unit_col and unit_col in work.columns:
        target_unit = target_unit or "uM"
        target_factor = display_unit_to_factor(target_unit)
        unit_norm = work[unit_col].map(normalize_unit)
        unit_factor = unit_norm.map(lambda u: UNIT_TO_MOLAR.get(u) if u else None)
        unknown_unit_mask = unit_norm.astype(str).str.strip().ne("") & unit_factor.isna()

        for idx, row in work.loc[unknown_unit_mask].iterrows():
            exclude_mask.loc[idx] = True
            exclude_reasons.loc[idx] = _join_reason(exclude_reasons.loc[idx], "unknown_or_unsupported_unit")
            _append_audit(
                audit_rows,
                row,
                int(idx),
                "excluded",
                "unknown_or_unsupported_unit",
                f"{unit_col}={row.get(unit_col, '')}",
            )

        convertible = work[target_col].notna() & unit_factor.notna()
        original_values = work.loc[convertible, target_col].copy()
        work.loc[convertible, target_col] = original_values * unit_factor.loc[convertible].astype(float) / target_factor
        converted_col = f"{target_col}__unit_after_qc"
        work[converted_col] = target_unit

        changed_units = convertible & (unit_norm != normalize_unit(target_unit))
        for idx, row in work.loc[changed_units].iterrows():
            before_value = original_values.loc[idx]
            after_value = work.loc[idx, target_col]
            conversion_rows.append(
                {
                    "original_row": int(idx) + 1,
                    "from_unit": row.get(unit_col, ""),
                    "to_unit": target_unit,
                    "value_before": before_value,
                    "value_after": after_value,
                    "target_column": target_col,
                }
            )
            _append_audit(
                audit_rows,
                row,
                int(idx),
                "converted",
                "unit_conversion",
                f"{before_value} {row.get(unit_col, '')} -> {after_value} {target_unit}",
            )

    technical_mask = pd.Series(False, index=work.index)
    if technical_min is not None and np.isfinite(float(technical_min)):
        technical_mask |= work[target_col] < float(technical_min)
    if technical_max is not None and np.isfinite(float(technical_max)):
        technical_mask |= work[target_col] > float(technical_max)

    for idx, row in work.loc[technical_mask & ~exclude_mask].iterrows():
        exclude_mask.loc[idx] = True
        exclude_reasons.loc[idx] = _join_reason(exclude_reasons.loc[idx], "technical_error_out_of_hard_range")
        _append_audit(
            audit_rows,
            row,
            int(idx),
            "excluded",
            "technical_error_out_of_hard_range",
            f"value={row.get(target_col)}; allowed=[{technical_min}, {technical_max}]",
        )

    kept_for_outlier = work.loc[~exclude_mask, target_col].dropna()
    outliers_df = pd.DataFrame()
    low_limit = np.nan
    high_limit = np.nan
    if len(kept_for_outlier) >= 4:
        q1 = float(kept_for_outlier.quantile(0.25))
        q3 = float(kept_for_outlier.quantile(0.75))
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            low_limit = q1 - float(outlier_iqr_multiplier) * iqr
            high_limit = q3 + float(outlier_iqr_multiplier) * iqr
            outlier_mask = (~exclude_mask) & ((work[target_col] < low_limit) | (work[target_col] > high_limit))
            outlier_rows = []
            for idx, row in work.loc[outlier_mask].iterrows():
                classification = "scientifically_valid_extreme_review_required"
                outlier_rows.append(
                    {
                        "original_row": int(idx) + 1,
                        target_col: row.get(target_col),
                        "outlier_classification": classification,
                        "rule": f"IQR x {float(outlier_iqr_multiplier)}",
                        "lower_limit": low_limit,
                        "upper_limit": high_limit,
                        "decision": "kept_for_review",
                    }
                )
                _append_audit(
                    audit_rows,
                    row,
                    int(idx),
                    "kept_with_flag",
                    classification,
                    f"value={row.get(target_col)}; IQR limits=[{low_limit}, {high_limit}]",
                )
            outliers_df = pd.DataFrame(outlier_rows)

    work["_data_quality_excluded"] = exclude_mask
    work["_data_quality_exclusion_reason"] = exclude_reasons
    excluded_df = work.loc[exclude_mask].copy().reset_index(drop=True)
    cleaned = work.loc[~exclude_mask].copy()

    replicate_group_cols = _row_key_columns(cleaned, smiles_col, context_cols=context_cols)
    cleaned, replicates_df = _collapse_replicates(
        cleaned,
        target_col=target_col,
        group_cols=replicate_group_cols,
        policy=replicate_policy,
        audit_rows=audit_rows,
    )

    cleaned = cleaned.drop(columns=["_data_quality_excluded", "_data_quality_exclusion_reason"], errors="ignore")
    summary_df = pd.DataFrame(
        [
            {"Проверка": "Строк исходно", "Значение": int(len(df)), "Комментарий": ""},
            {"Проверка": "Цензурированные значения", "Значение": int(censored_mask.sum()), "Комментарий": f"Политика: {censored_policy}"},
            {"Проверка": "Конверсии единиц", "Значение": int(len(conversion_rows)), "Комментарий": f"Целевая единица: {target_unit or ''}"},
            {"Проверка": "Повторные измерения", "Значение": int(len(replicates_df)), "Комментарий": f"Политика: {replicate_policy}"},
            {"Проверка": "Выбросы IQR", "Значение": int(len(outliers_df)), "Комментарий": "IQR-выбросы сохранены как научные экстремумы для ручной проверки"},
            {"Проверка": "Исключено строк", "Значение": int(len(excluded_df)), "Комментарий": "Каждая исключённая строка есть в export"},
            {"Проверка": "Осталось строк", "Значение": int(len(cleaned)), "Комментарий": ""},
        ]
    )

    audit_df = pd.DataFrame(audit_rows)
    conversions_df = pd.DataFrame(conversion_rows)
    return {
        "cleaned_df": cleaned.reset_index(drop=True),
        "audit_df": audit_df.reset_index(drop=True),
        "excluded_df": excluded_df.reset_index(drop=True),
        "summary_df": summary_df,
        "replicates_df": replicates_df.reset_index(drop=True),
        "conversions_df": conversions_df.reset_index(drop=True),
        "outliers_df": outliers_df.reset_index(drop=True),
        "limits": {"iqr_low": low_limit, "iqr_high": high_limit},
    }


def _join_reason(old: str, new: str) -> str:
    old = str(old or "").strip()
    if not old:
        return new
    parts = [item.strip() for item in old.split(";") if item.strip()]
    if new not in parts:
        parts.append(new)
    return "; ".join(parts)
