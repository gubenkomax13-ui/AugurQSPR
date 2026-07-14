# -*- coding: utf-8 -*-

"""
descriptor_bank_core.py

Простой CSV-банк дескрипторов для QSPR Forge.

Версия 1:
- локальный CSV;
- ключ: InChIKey, fallback canonical_smiles;
- профиль расчёта: default;
- хранит дескрипторы без target;
- возвращает кешированные строки с текущими row_index / _original_index.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from rdkit import Chem


DESCRIPTOR_BANK_DIR = "descriptor_bank"
DESCRIPTOR_BANK_FILE = os.path.join(DESCRIPTOR_BANK_DIR, "descriptor_bank.csv")

def descriptor_bank_get_file(descriptor_source, descriptor_profile="default"):
    """
    Возвращает путь к отдельному CSV-банку для конкретного источника дескрипторов.

    Примеры:
    descriptor_bank/xtb_descriptor_bank.csv
    descriptor_bank/morfeus_descriptor_bank.csv
    descriptor_bank/dscribe_descriptor_bank.csv
    """
    os.makedirs(DESCRIPTOR_BANK_DIR, exist_ok=True)

    source = str(descriptor_source).lower().strip()
    profile = str(descriptor_profile).lower().strip()

    safe_source = (
        source
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    safe_profile = (
        profile
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    if safe_profile == "default":
        filename = f"{safe_source}_descriptor_bank.csv"
    else:
        filename = f"{safe_source}_{safe_profile}_descriptor_bank.csv"

    return os.path.join(DESCRIPTOR_BANK_DIR, filename)

BANK_SERVICE_COLUMNS = [
    "bank_key",
    "inchikey",
    "canonical_smiles",
    "descriptor_family",
    "descriptor_source",
    "descriptor_profile",
    "created_at",
]


def descriptor_bank_load(bank_file=DESCRIPTOR_BANK_FILE):
    """
    Загружает CSV-банк дескрипторов.
    """
    os.makedirs(os.path.dirname(bank_file), exist_ok=True)

    if not os.path.exists(bank_file):
        return pd.DataFrame(columns=BANK_SERVICE_COLUMNS)

    try:
        bank = pd.read_csv(bank_file)
    except Exception:
        bank = pd.DataFrame(columns=BANK_SERVICE_COLUMNS)

    for col in BANK_SERVICE_COLUMNS:
        if col not in bank.columns:
            bank[col] = ""

    return bank


def descriptor_bank_save(bank_df, bank_file=DESCRIPTOR_BANK_FILE):
    """
    Сохраняет CSV-банк дескрипторов.
    """
    os.makedirs(os.path.dirname(bank_file), exist_ok=True)

    bank_df = bank_df.copy()
    bank_df = bank_df.loc[:, ~bank_df.columns.duplicated()].copy()

    bank_df.to_csv(bank_file, index=False)


def descriptor_bank_smiles_to_keys(smiles):
    """
    SMILES -> canonical_smiles, InChIKey, bank_key.
    bank_key = InChIKey, если он есть, иначе canonical_smiles.
    """
    try:
        smiles = str(smiles).strip()

        if not smiles or smiles.lower() in ["nan", "none"]:
            return "", "", "", "empty_smiles"

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return "", "", "", "invalid_smiles"

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = ""

        bank_key = inchikey if str(inchikey).strip() else canonical_smiles

        return canonical_smiles, inchikey, bank_key, "ok"

    except Exception as e:
        return "", "", "", f"key_error: {e}"


def descriptor_bank_make_input_key_table(df, smiles_col, max_molecules=None):
    """
    Делает таблицу ключей для текущего датасета.

    Возвращает:
    _original_index
    row_index
    canonical_smiles
    inchikey
    bank_key
    bank_key_status
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if smiles_col not in df.columns:
        raise ValueError(f"Колонка SMILES не найдена: {smiles_col}")

    work = df.copy()

    if max_molecules is not None:
        work = work.head(int(max_molecules)).copy()

    rows = []

    for idx, row in work.iterrows():
        canonical_smiles, inchikey, bank_key, status = descriptor_bank_smiles_to_keys(
            row.get(smiles_col, "")
        )

        rows.append({
            "_original_index": idx,
            "row_index": idx,
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "bank_key": bank_key,
            "bank_key_status": status,
        })

    return pd.DataFrame(rows)


def descriptor_bank_get_cached_and_missing(
    df,
    smiles_col,
    descriptor_family,
    descriptor_source,
    descriptor_profile="default",
    max_molecules=None,
    bank_file=DESCRIPTOR_BANK_FILE,
):
    """
    Возвращает:
    cached_df — строки из банка, адаптированные под текущие индексы;
    missing_df — строки исходного df, которые нужно досчитать;
    report — статистика.
    """
    key_table = descriptor_bank_make_input_key_table(
        df=df,
        smiles_col=smiles_col,
        max_molecules=max_molecules,
    )

    if key_table.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "bank_found": 0,
            "bank_missing": 0,
            "bank_total": 0,
        }

    valid_keys = key_table[key_table["bank_key"].astype(str).str.strip() != ""].copy()
    
    if bank_file == DESCRIPTOR_BANK_FILE:
        bank_file = descriptor_bank_get_file(
            descriptor_source=descriptor_source,
            descriptor_profile=descriptor_profile,
        )
        
    bank = descriptor_bank_load(bank_file)

    bank_part = bank[
        (bank["descriptor_family"].astype(str) == str(descriptor_family))
        & (bank["descriptor_source"].astype(str) == str(descriptor_source))
        & (bank["descriptor_profile"].astype(str) == str(descriptor_profile))
    ].copy()

    if not bank_part.empty:
        bank_part = bank_part.drop_duplicates(
            subset=[
                "bank_key",
                "descriptor_family",
                "descriptor_source",
                "descriptor_profile",
            ],
            keep="last",
        )

    cached_df = pd.DataFrame()

    if not valid_keys.empty and not bank_part.empty:
        cached_df = valid_keys.merge(
            bank_part,
            on="bank_key",
            how="inner",
            suffixes=("", "_bank"),
        )

        # Берём актуальные индексы текущего датасета.
        if "canonical_smiles_bank" in cached_df.columns:
            cached_df["canonical_smiles"] = cached_df["canonical_smiles"].where(
                cached_df["canonical_smiles"].astype(str).str.strip() != "",
                cached_df["canonical_smiles_bank"],
            )

        if "inchikey_bank" in cached_df.columns:
            cached_df["inchikey"] = cached_df["inchikey"].where(
                cached_df["inchikey"].astype(str).str.strip() != "",
                cached_df["inchikey_bank"],
            )

        # Удаляем дублирующиеся bank-колонки после merge.
        drop_cols = [
            c for c in cached_df.columns
            if c.endswith("_bank")
        ]

        cached_df = cached_df.drop(columns=drop_cols, errors="ignore")

    cached_keys = set(cached_df["bank_key"].astype(str)) if not cached_df.empty else set()

    missing_key_rows = key_table[
        ~key_table["bank_key"].astype(str).isin(cached_keys)
    ].copy()

    missing_indices = missing_key_rows["_original_index"].tolist()

    missing_df = df.loc[missing_indices].copy()

    report = {
        "bank_total": int(len(key_table)),
        "bank_found": int(len(cached_df)),
        "bank_missing": int(len(missing_df)),
        "descriptor_family": descriptor_family,
        "descriptor_source": descriptor_source,
        "descriptor_profile": descriptor_profile,
    }

    return cached_df.reset_index(drop=True), missing_df, report


def descriptor_bank_append(
    desc_df,
    descriptor_family,
    descriptor_source,
    descriptor_profile="default",
    target_col=None,
    bank_file=DESCRIPTOR_BANK_FILE,
):
    """
    Добавляет новые рассчитанные дескрипторы в банк.

    target_col НЕ сохраняется.
    row_index / _original_index НЕ сохраняются как ключевые поля банка.
    """
    if bank_file == DESCRIPTOR_BANK_FILE:
        bank_file = descriptor_bank_get_file(
            descriptor_source=descriptor_source,
            descriptor_profile=descriptor_profile,
        )

    if desc_df is None or not isinstance(desc_df, pd.DataFrame) or desc_df.empty:
        return descriptor_bank_load(bank_file)

    work = desc_df.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()

    status_col = None
    source_name = str(descriptor_source).lower().strip()

    if source_name == "xtb" and "xtb_status" in work.columns:
        status_col = "xtb_status"
    elif source_name == "morfeus" and "morfeus_status" in work.columns:
        status_col = "morfeus_status"
    elif source_name == "dscribe" and "dscribe_status" in work.columns:
        status_col = "dscribe_status"

    if status_col is not None:
        work = work[
            work[status_col].astype(str).str.lower().str.strip() == "ok"
        ].copy()

        if work.empty:
            return descriptor_bank_load(bank_file)

    if "canonical_smiles" not in work.columns or "inchikey" not in work.columns:
        raise ValueError(
            "Для сохранения в банк нужны колонки canonical_smiles и inchikey."
        )

    if "bank_key" not in work.columns:
        keys = []

        for _, row in work.iterrows():
            inchikey = str(row.get("inchikey", "")).strip()
            canonical_smiles = str(row.get("canonical_smiles", "")).strip()
            keys.append(inchikey if inchikey else canonical_smiles)

        work["bank_key"] = keys

    work = work[work["bank_key"].astype(str).str.strip() != ""].copy()

    if work.empty:
        return descriptor_bank_load(bank_file)

    work["descriptor_family"] = str(descriptor_family)
    work["descriptor_source"] = str(descriptor_source)
    work["descriptor_profile"] = str(descriptor_profile)
    work["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Не храним target и временные индексы датасета.
    drop_cols = [
        target_col,
        "_original_index",
        "row_index",
        "bank_key_status",
    ]

    work = work.drop(
        columns=[c for c in drop_cols if c is not None and c in work.columns],
        errors="ignore",
    )

    service_first = [
        c for c in BANK_SERVICE_COLUMNS
        if c in work.columns
    ]

    other_cols = [
        c for c in work.columns
        if c not in service_first
    ]

    work = work[service_first + other_cols].copy()
    
    if bank_file == DESCRIPTOR_BANK_FILE:
        bank_file = descriptor_bank_get_file(
            descriptor_source=descriptor_source,
            descriptor_profile=descriptor_profile,
        )
    
    bank = descriptor_bank_load(bank_file)

    bank = pd.concat([bank, work], ignore_index=True)
    bank = bank.loc[:, ~bank.columns.duplicated()].copy()

    bank = bank.drop_duplicates(
        subset=[
            "bank_key",
            "descriptor_family",
            "descriptor_source",
            "descriptor_profile",
        ],
        keep="last",
    )

    descriptor_bank_save(bank, bank_file)

    return bank


def descriptor_bank_attach_target(df, source_df, target_col):
    """
    Добавляет target_col к таблице дескрипторов по _original_index / row_index.
    Нужно только для текущего моделирования, в банк target не пишется.
    """
    if df is None or df.empty:
        return df

    if target_col is None or target_col not in source_df.columns:
        return df

    out = df.copy()

    index_col = None

    if "_original_index" in out.columns:
        index_col = "_original_index"
    elif "row_index" in out.columns:
        index_col = "row_index"

    if index_col is None:
        out[target_col] = pd.to_numeric(
            source_df[target_col].values[:len(out)],
            errors="coerce",
        )
        return out

    idx = pd.to_numeric(out[index_col], errors="coerce")
    valid = idx.notna()

    out[target_col] = np.nan

    if valid.any():
        idx_valid = idx.loc[valid].astype(int)

        out.loc[valid, target_col] = pd.to_numeric(
            source_df.loc[idx_valid, target_col].values,
            errors="coerce",
        )

    return out


def descriptor_bank_show_report(report, title="Банк дескрипторов"):
    """
    Короткий Streamlit-отчёт. Импорт streamlit делаем внутри,
    чтобы модуль можно было использовать и без UI.
    """
    try:
        import streamlit as st
    except Exception:
        return

    st.info(
        f"{title}: найдено в банке {report.get('bank_found', 0)}, "
        f"досчитано {report.get('bank_missing', 0)}, "
        f"всего {report.get('bank_total', 0)}."
    )