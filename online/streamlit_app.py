#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Lightweight public Augur QSPR app for Streamlit Cloud.

This entrypoint is intentionally independent from the full local qspr_app.py.
It keeps the public online build fast and stable while preserving RDKit-based
SMILES processing and basic QSPR modelling.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def configure_page() -> None:
    st.set_page_config(page_title="Augur QSPR online", layout="wide")


LANGUAGES = {
    "ru": "Русский",
    "kk": "Қазақша",
    "en": "English",
}


TEXT = {
    "ru": {
        "title": "Augur QSPR онлайн",
        "subtitle": "Лёгкая публичная версия: RDKit-дескрипторы, базовые модели и быстрый прогноз.",
        "full_note": (
            "Полная локальная версия содержит Mordred, PaDEL, xTB, PySR, SAOD, спектры, "
            "расширенные отчёты и тяжёлые проверки."
        ),
        "upload": "Загрузите CSV или XLSX",
        "example": "Файл должен содержать колонку SMILES и числовое целевое свойство.",
        "smiles": "Колонка SMILES",
        "target": "Целевое свойство",
        "model": "Модель",
        "test_size": "Тестовая выборка, %",
        "seed": "Seed",
        "descriptor_mode": "Дескрипторы",
        "rdkit_basic": "Базовые RDKit-дескрипторы",
        "uploaded_numeric": "Числовые колонки файла",
        "combined": "RDKit + числовые колонки файла",
        "train": "Обучить модель",
        "preview": "Предпросмотр данных",
        "invalid_smiles": "Невалидные SMILES: {count}. Эти строки исключены из обучения.",
        "not_enough": "Недостаточно строк для обучения: нужно минимум 5 объектов и не менее 2 разных значений свойства.",
        "no_features": "Не найдено числовых дескрипторов для обучения.",
        "metrics": "Метрики тестовой выборки",
        "predictions": "Прогнозы для тестовой выборки",
        "download": "Скачать прогнозы CSV",
        "diagnostics": "Диагностика установки",
        "rdkit_ok": "RDKit работает",
        "rdkit_failed": "RDKit недоступен: {error}",
        "local_link": "Для полного функционала используйте локальную версию из GitHub.",
    },
    "kk": {
        "title": "Augur QSPR онлайн",
        "subtitle": "Жеңіл жария нұсқа: RDKit дескрипторлары, базалық модельдер және жылдам болжам.",
        "full_note": (
            "Толық жергілікті нұсқада Mordred, PaDEL, xTB, PySR, SAOD, спектрлер, "
            "кеңейтілген есептер және ауыр тексерулер бар."
        ),
        "upload": "CSV немесе XLSX файлын жүктеңіз",
        "example": "Файлда SMILES бағаны және сандық мақсатты қасиет болуы керек.",
        "smiles": "SMILES бағаны",
        "target": "Мақсатты қасиет",
        "model": "Модель",
        "test_size": "Тест жиыны, %",
        "seed": "Seed",
        "descriptor_mode": "Дескрипторлар",
        "rdkit_basic": "Базалық RDKit дескрипторлары",
        "uploaded_numeric": "Файлдағы сандық бағандар",
        "combined": "RDKit + файлдағы сандық бағандар",
        "train": "Модельді оқыту",
        "preview": "Деректерді алдын ала қарау",
        "invalid_smiles": "Жарамсыз SMILES: {count}. Бұл жолдар оқытуға кірмейді.",
        "not_enough": "Оқытуға дерек жеткіліксіз: кемінде 5 объект және қасиеттің кемінде 2 түрлі мәні керек.",
        "no_features": "Оқытуға арналған сандық дескрипторлар табылмады.",
        "metrics": "Тест жиыны метрикалары",
        "predictions": "Тест жиыны болжамдары",
        "download": "Болжамдарды CSV ретінде жүктеу",
        "diagnostics": "Орнату диагностикасы",
        "rdkit_ok": "RDKit жұмыс істейді",
        "rdkit_failed": "RDKit қолжетімсіз: {error}",
        "local_link": "Толық функционал үшін GitHub-тағы жергілікті нұсқаны пайдаланыңыз.",
    },
    "en": {
        "title": "Augur QSPR online",
        "subtitle": "Light public version: RDKit descriptors, basic models, and fast prediction.",
        "full_note": (
            "The full local version includes Mordred, PaDEL, xTB, PySR, SAOD, spectra, "
            "advanced reports, and heavy diagnostics."
        ),
        "upload": "Upload CSV or XLSX",
        "example": "The file should contain a SMILES column and a numeric target property.",
        "smiles": "SMILES column",
        "target": "Target property",
        "model": "Model",
        "test_size": "Test set, %",
        "seed": "Seed",
        "descriptor_mode": "Descriptors",
        "rdkit_basic": "Basic RDKit descriptors",
        "uploaded_numeric": "Numeric columns from file",
        "combined": "RDKit + numeric columns from file",
        "train": "Train model",
        "preview": "Data preview",
        "invalid_smiles": "Invalid SMILES: {count}. These rows are excluded from training.",
        "not_enough": "Not enough data for training: at least 5 compounds and 2 distinct target values are required.",
        "no_features": "No numeric descriptors were found for training.",
        "metrics": "Test-set metrics",
        "predictions": "Test-set predictions",
        "download": "Download predictions CSV",
        "diagnostics": "Installation diagnostics",
        "rdkit_ok": "RDKit works",
        "rdkit_failed": "RDKit unavailable: {error}",
        "local_link": "Use the local GitHub version for the full feature set.",
    },
}


DESCRIPTOR_FUNCTIONS = {
    "RDKit::MolWt": Descriptors.MolWt,
    "RDKit::MolLogP": Descriptors.MolLogP,
    "RDKit::TPSA": Descriptors.TPSA,
    "RDKit::NumHDonors": Descriptors.NumHDonors,
    "RDKit::NumHAcceptors": Descriptors.NumHAcceptors,
    "RDKit::NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RDKit::RingCount": Descriptors.RingCount,
    "RDKit::HeavyAtomCount": Descriptors.HeavyAtomCount,
    "RDKit::FractionCSP3": Descriptors.FractionCSP3,
    "RDKit::BertzCT": Descriptors.BertzCT,
}


@dataclass
class DescriptorResult:
    frame: pd.DataFrame
    valid_mask: pd.Series
    invalid_count: int


def text(lang: str, key: str, **kwargs) -> str:
    value = TEXT.get(lang, TEXT["ru"]).get(key, key)
    try:
        return value.format(**kwargs)
    except Exception:
        return value


def detect_language() -> str:
    try:
        query_lang = st.query_params.get("lang")
        if query_lang in LANGUAGES:
            return query_lang
    except Exception:
        pass
    return "ru"


def remember_language(lang: str) -> None:
    try:
        st.query_params["lang"] = lang
    except Exception:
        pass


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = str(uploaded_file.name).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")


def find_smiles_columns(columns) -> list[str]:
    aliases = {"smiles", "canonical_smiles", "canonical smiles", "isomeric_smiles"}
    return [col for col in columns if str(col).strip().lower() in aliases]


def calculate_rdkit_descriptors(smiles_series: pd.Series) -> DescriptorResult:
    rows = []
    valid = []
    for smiles in smiles_series.astype(str):
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            valid.append(False)
            rows.append({name: np.nan for name in DESCRIPTOR_FUNCTIONS})
            continue
        valid.append(True)
        row = {}
        for name, func in DESCRIPTOR_FUNCTIONS.items():
            try:
                row[name] = float(func(mol))
            except Exception:
                row[name] = np.nan
        rows.append(row)
    valid_mask = pd.Series(valid, index=smiles_series.index)
    return DescriptorResult(
        frame=pd.DataFrame(rows, index=smiles_series.index),
        valid_mask=valid_mask,
        invalid_count=int((~valid_mask).sum()),
    )


def numeric_frame(data: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    for column in data.columns:
        if column in excluded:
            continue
        parsed = pd.to_numeric(
            data[column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if parsed.notna().sum() >= 3:
            result[str(column)] = parsed
    return result


def model_pipeline(model_name: str, seed: int) -> Pipeline:
    if model_name == "Random Forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=120,
                min_samples_leaf=2,
                random_state=int(seed),
                n_jobs=1,
            )),
        ])
    if model_name == "Linear Regression":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ])
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def main() -> None:
    lang = st.session_state.get("lang", detect_language())

    with st.sidebar:
        st.header("Augur QSPR")
        selected_lang = st.selectbox(
            "Language / Язык / Тіл",
            options=list(LANGUAGES),
            index=list(LANGUAGES).index(lang) if lang in LANGUAGES else 0,
            format_func=lambda code: LANGUAGES[code],
        )
        if selected_lang != lang:
            st.session_state.lang = selected_lang
            remember_language(selected_lang)
            st.rerun()
        lang = selected_lang
        st.info(text(lang, "full_note"))

        with st.expander(text(lang, "diagnostics"), expanded=False):
            try:
                mol = Chem.MolFromSmiles("CCO")
                if mol is None:
                    raise RuntimeError("CCO parse returned None")
                st.success(text(lang, "rdkit_ok"))
            except Exception as exc:
                st.error(text(lang, "rdkit_failed", error=exc))

    st.title("🧪 " + text(lang, "title"))
    st.write(text(lang, "subtitle"))
    st.caption(text(lang, "local_link"))

    uploaded_file = st.file_uploader(text(lang, "upload"), type=["csv", "xlsx", "xls"])
    st.caption(text(lang, "example"))
    if uploaded_file is None:
        return

    try:
        data = read_uploaded_table(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        return

    if data.empty:
        st.error(text(lang, "not_enough"))
        return

    st.subheader(text(lang, "preview"))
    st.dataframe(data.head(20), use_container_width=True)

    smiles_candidates = find_smiles_columns(data.columns)
    smiles_col = st.selectbox(
        text(lang, "smiles"),
        options=list(data.columns),
        index=list(data.columns).index(smiles_candidates[0]) if smiles_candidates else 0,
    )
    target_col = st.selectbox(text(lang, "target"), options=list(data.columns))

    descriptor_mode = st.radio(
        text(lang, "descriptor_mode"),
        options=["rdkit", "uploaded", "combined"],
        format_func=lambda code: {
            "rdkit": text(lang, "rdkit_basic"),
            "uploaded": text(lang, "uploaded_numeric"),
            "combined": text(lang, "combined"),
        }[code],
        horizontal=True,
    )

    rdkit_result = calculate_rdkit_descriptors(data[smiles_col])
    if rdkit_result.invalid_count:
        st.warning(text(lang, "invalid_smiles", count=rdkit_result.invalid_count))

    target = pd.to_numeric(
        data[target_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    uploaded_numeric = numeric_frame(data, excluded={smiles_col, target_col})

    feature_parts = []
    if descriptor_mode in {"rdkit", "combined"}:
        feature_parts.append(rdkit_result.frame)
    if descriptor_mode in {"uploaded", "combined"} and not uploaded_numeric.empty:
        feature_parts.append(uploaded_numeric)

    if not feature_parts:
        st.error(text(lang, "no_features"))
        return

    features = pd.concat(feature_parts, axis=1)
    valid_rows = rdkit_result.valid_mask & target.notna()
    if descriptor_mode == "uploaded":
        valid_rows = target.notna()
    X = features.loc[valid_rows]
    y = target.loc[valid_rows]

    if len(X) < 5 or y.nunique(dropna=True) < 2:
        st.error(text(lang, "not_enough"))
        return

    col_model, col_test, col_seed = st.columns(3)
    with col_model:
        model_name = st.selectbox(text(lang, "model"), ["Ridge", "Linear Regression", "Random Forest"])
    with col_test:
        test_percent = st.slider(text(lang, "test_size"), 10, 40, 20, 5)
    with col_seed:
        seed = st.number_input(text(lang, "seed"), min_value=0, max_value=100000, value=42, step=1)

    if not st.button("🚀 " + text(lang, "train"), type="primary"):
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_percent) / 100.0,
        random_state=int(seed),
    )
    model = model_pipeline(model_name, int(seed))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = pd.DataFrame([{
        "model": model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "R2": float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
    }])
    st.subheader(text(lang, "metrics"))
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    predictions = data.loc[X_test.index].copy()
    predictions["observed"] = y_test.to_numpy()
    predictions["predicted"] = y_pred
    predictions["residual"] = predictions["observed"] - predictions["predicted"]
    st.subheader(text(lang, "predictions"))
    st.dataframe(predictions, use_container_width=True)
    st.download_button(
        text(lang, "download"),
        data=predictions.to_csv(index=False).encode("utf-8-sig"),
        file_name="augur_qspr_online_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    configure_page()
    main()
