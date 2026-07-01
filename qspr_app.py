#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GenQSPR / QSPR Универсальный конструктор моделей

Главный Streamlit-интерфейс.
Расчётные ядра вынесены в модули:
- modules/qspr_core.py
- modules/spectra_core.py
- modules/saod2_core.py

В этом файле находятся:
- загрузка данных;
- интерфейс выбора дескрипторов;
- интерфейс spectra_bank;
- интерфейс SAOD;
- интерфейс обучения/валидации QSPR-моделей.
"""

import os
import glob
import re
import sys
import json
import io
import subprocess
import importlib
import hmac
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from sklearn.linear_model import LinearRegression
from modules.applicability_domain_core import *
from modules.methodology_generator import generate_methodology_text
from modules.report_generator import generate_full_report
from modules.save_model_ui import render_verified_model_save
from modules.validation_ui import render_validation_section
from modules.error_analysis_ui import render_error_analysis_section
from modules.comparison_ui import render_model_comparison_section
from modules.consensus_ui import render_consensus_section
from modules.diagnostics_ui import render_model_diagnostics_section
from modules.training_ui import render_training_section
from modules.report_ui import render_report_section
from modules.statistics_summary_ui import render_final_statistics_summary
import time

import streamlit as st
import streamlit.components.v1 as components


def _qspr_arrow_safe_table_data(data):
    """Return a display copy that PyArrow can serialize without noisy fallback logs."""
    try:
        import pandas as _pd
    except Exception:
        return data

    if not isinstance(data, _pd.DataFrame):
        return data

    out = data.copy()
    out.columns = [str(col) for col in out.columns]

    for col in out.columns:
        if out[col].dtype != "object":
            continue

        def _cell_to_text(value):
            if value is None:
                return ""
            try:
                if _pd.isna(value):
                    return ""
            except (TypeError, ValueError):
                pass
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        out[col] = out[col].map(_cell_to_text).astype("string")

    return out


def _install_qspr_arrow_safe_streamlit_tables():
    if getattr(st, "_qspr_arrow_safe_tables_installed", False):
        return

    original_dataframe = st.dataframe
    original_data_editor = st.data_editor

    def dataframe(data=None, *args, **kwargs):
        return original_dataframe(_qspr_arrow_safe_table_data(data), *args, **kwargs)

    def data_editor(data=None, *args, **kwargs):
        return original_data_editor(_qspr_arrow_safe_table_data(data), *args, **kwargs)

    st.dataframe = dataframe
    st.data_editor = data_editor
    st._qspr_arrow_safe_tables_installed = True


_install_qspr_arrow_safe_streamlit_tables()

from modules.i18n import (
    gettext,
    set_language,
    t,
    load_language,
    validate_translation_keys,
)

SUPPORTED_LANGS = ("ru", "en", "kk")
AUGUR_GITHUB_URL = "https://github.com/gubenkomax13-ui/AugurQSPR"
ONLINE_LOCK_MESSAGE = (
    "Эта функция показана как возможность полной локальной версии Augur QSPR, "
    "но в публичном онлайн-режиме она отключена для безопасности и стабильности."
)
ONLINE_MAX_UPLOAD_MB = 5
ONLINE_MAX_DATA_ROWS = 1000


def _normalize_lang_code(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""

    value = str(value or "").strip().lower()
    if not value:
        return None

    for item in value.split(","):
        code = item.split(";", 1)[0].strip().lower()
        base_code = code.split("-", 1)[0]
        if base_code in SUPPORTED_LANGS:
            return base_code

    return None


def _query_param_lang():
    try:
        return _normalize_lang_code(st.query_params.get("lang"))
    except Exception:
        return None


def _browser_lang():
    context = getattr(st, "context", None)
    if context is None:
        return None

    for attr_name in ("locale", "language"):
        lang = _normalize_lang_code(getattr(context, attr_name, None))
        if lang:
            return lang

    headers = getattr(context, "headers", {}) or {}
    try:
        accept_language = headers.get("accept-language") or headers.get("Accept-Language")
    except AttributeError:
        accept_language = None

    return _normalize_lang_code(accept_language)


def _remember_lang_in_url(lang):
    try:
        st.query_params["lang"] = lang
    except Exception:
        pass


def _qspr_bool_setting(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def qspr_is_online_streamlit_version():
    for source in (os.environ.get("AUGUR_SHOW_ONLINE_DEMO_NOTICE"),):
        parsed = _qspr_bool_setting(source)
        if parsed is not None:
            return parsed

    try:
        parsed = _qspr_bool_setting(st.secrets.get("AUGUR_SHOW_ONLINE_DEMO_NOTICE"))
        if parsed is not None:
            return parsed
    except Exception:
        pass

    streamlit_env_markers = [
        "STREAMLIT_CLOUD",
        "STREAMLIT_SHARING_MODE",
        "STREAMLIT_RUNTIME_ENV",
    ]
    for env_name in streamlit_env_markers:
        value = str(os.environ.get(env_name, "")).strip().lower()
        if value and value not in {"0", "false", "local", "development"}:
            return True

    context = getattr(st, "context", None)
    headers = getattr(context, "headers", {}) if context is not None else {}
    try:
        host = str(headers.get("host") or headers.get("Host") or "").lower()
    except AttributeError:
        host = ""

    try:
        url = str(getattr(context, "url", "") or "").lower()
    except Exception:
        url = ""

    online_hosts = ("streamlit.app", "share.streamlit.io")
    return any(marker in host or marker in url for marker in online_hosts)


def qspr_runtime_mode():
    """Return `online` or `local` for feature gating."""
    for source in (os.environ.get("AUGUR_MODE"), os.environ.get("AUGUR_RUNTIME_MODE")):
        value = str(source or "").strip().lower()
        if value in {"online", "demo", "cloud", "public"}:
            return "online"
        if value in {"local", "full", "desktop"}:
            return "local"

    try:
        value = str(st.secrets.get("AUGUR_MODE", "") or "").strip().lower()
        if value in {"online", "demo", "cloud", "public"}:
            return "online"
        if value in {"local", "full", "desktop"}:
            return "local"
    except Exception:
        pass

    return "online" if qspr_is_online_streamlit_version() else "local"


def qspr_is_online_mode():
    return qspr_runtime_mode() == "online"


def qspr_online_lock_notice(feature_name=""):
    if feature_name:
        st.info(f"{feature_name}: {ONLINE_LOCK_MESSAGE}")
    else:
        st.info(ONLINE_LOCK_MESSAGE)


def qspr_show_online_demo_notice():
    if not qspr_is_online_streamlit_version():
        return

    title = escape(t("online_demo_notice.title"))
    body = escape(t("online_demo_notice.body"))
    github_text = escape(t("online_demo_notice.github"))
    collapse_label = escape(t("online_demo_notice.collapse_label"))
    expand_label = escape(t("online_demo_notice.expand_label"))
    link_label = t("online_demo_notice.link_label")
    github_link = (
        f'<a href="{escape(AUGUR_GITHUB_URL, quote=True)}" '
        f'target="_blank" rel="noopener noreferrer">GitHub</a>'
    )
    if "GitHub" in link_label:
        link_label = escape(link_label).replace("GitHub", github_link)
    else:
        link_label = f"{escape(link_label)}: {github_link}"

    components.html(
        f"""
        <!doctype html>
        <html lang="{escape(st.session_state.get('lang', 'ru'))}">
        <head>
        <meta charset="utf-8">
        <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: "Source Sans Pro", sans-serif;
        }}
        .online-demo-notice {{
            box-sizing: border-box;
            margin: 0;
            border: 1px solid #9ec5fe;
            border-radius: 0.5rem;
            background: #eef6ff;
            color: #102a43;
            line-height: 1.45;
            overflow: hidden;
        }}
        .online-demo-notice summary {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.8rem 1rem;
            cursor: pointer;
            list-style: none;
            user-select: none;
        }}
        .online-demo-notice summary::-webkit-details-marker {{
            display: none;
        }}
        .online-demo-notice summary::after {{
            content: "{collapse_label}";
            flex: 0 0 auto;
            color: #315f8f;
            font-size: 0.86rem;
            font-weight: 600;
        }}
        .online-demo-notice:not([open]) summary::after {{
            content: "{expand_label}";
        }}
        .online-demo-notice-title {{
            font-weight: 700;
        }}
        .online-demo-notice-body {{
            padding: 0 1rem 0.95rem 1rem;
        }}
        .online-demo-notice a {{
            color: #0b5ed7;
            font-weight: 600;
            text-decoration: none;
        }}
        .online-demo-notice a:hover {{
            text-decoration: underline;
        }}
        @media (max-width: 640px) {{
            .online-demo-notice summary {{
                align-items: flex-start;
                flex-direction: column;
            }}
        }}
        </style>
        </head>
        <body>
        <details class="online-demo-notice" id="online-demo-notice" open>
          <summary><span class="online-demo-notice-title">{title}</span></summary>
          <div class="online-demo-notice-body">
            {body}<br><br>
            {github_text}<br><br>
            {link_label}
          </div>
        </details>
        <script>
        const notice = document.getElementById("online-demo-notice");
        const summary = notice.querySelector("summary");
        let userInteracted = false;

        function setFrameHeight() {{
            const height = Math.ceil(document.documentElement.scrollHeight);
            window.parent.postMessage({{
                isStreamlitMessage: true,
                type: "streamlit:setFrameHeight",
                height: height
            }}, "*");
        }}

        function refreshFrameHeight() {{
            setFrameHeight();
            window.setTimeout(setFrameHeight, 80);
            window.setTimeout(setFrameHeight, 250);
        }}

        summary.addEventListener("click", () => {{
            userInteracted = true;
            refreshFrameHeight();
        }});

        notice.addEventListener("toggle", refreshFrameHeight);
        window.addEventListener("load", refreshFrameHeight);
        window.setTimeout(() => {{
            if (!userInteracted && notice.open) {{
                notice.open = false;
                refreshFrameHeight();
            }}
        }}, 10000);
        refreshFrameHeight();
        </script>
        </body>
        </html>
        """,
        height=230,
        scrolling=False,
    )


query_lang = _query_param_lang()
if query_lang and st.session_state.get("lang") != query_lang:
    st.session_state.lang = query_lang
    st.session_state.lang_from_url = True
elif "lang" not in st.session_state:
    initial_lang = _browser_lang() or "ru"
    st.session_state.lang = initial_lang
    st.session_state.lang_auto_detected = initial_lang

set_language(st.session_state.lang)

translation_key_issues = validate_translation_keys(
    os.path.dirname(os.path.abspath(__file__))
)

# ------------------------------------------------------------------
# Проверка обязательных пакетов до тяжёлых импортов

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy<2": ("numpy", "numpy<2"),
    "matplotlib": "matplotlib",
    "rdkit-pypi": "rdkit",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "streamlit": "streamlit",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "Pillow": "PIL",
    "openpyxl": "openpyxl",
}

OPTIONAL_PACKAGES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "pysr": "pysr",
    "mordred": "mordred",
    "padelpy": "padelpy",
    "shap": "shap",
    "jcamp": "jcamp",
    "xtb": "xtb",
}


def check_packages(packages):
    missing = []

    for pkg_key, pkg_info in packages.items():
        if isinstance(pkg_info, tuple):
            import_name = pkg_info[0]
        else:
            import_name = pkg_info

        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg_key)

    return missing


missing_required = check_packages(REQUIRED_PACKAGES)

numpy_conflict = False

try:
    import numpy as _np_check

    if _np_check.__version__.startswith("2."):
        numpy_conflict = True
        if "numpy<2" not in missing_required:
            missing_required.append("numpy<2")
except ImportError:
    numpy_conflict = True
    if "numpy<2" not in missing_required:
        missing_required.append("numpy<2")

if missing_required or numpy_conflict:
    st.set_page_config(page_title=t('install.page_title'), layout="centered")
    st.title(t('install.title'))

    if missing_required:
        st.error(t('install.missing_packages', packages=', '.join(missing_required)))

    if numpy_conflict:
        st.error(t('install.numpy_conflict'))

    if st.button(t('install.install_button'), type="primary"):
        with st.spinner(t('install.spinner_text')):
            install_list = []

            for pkg_key, pkg_info in REQUIRED_PACKAGES.items():
                if isinstance(pkg_info, tuple):
                    install_list.append(pkg_info[1])
                else:
                    install_list.append(pkg_key)

            for pkg in OPTIONAL_PACKAGES:
                install_list.append(pkg)

            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--user"] + install_list
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                st.success(t('install.success'))
                st.info(t('install.java_required'))
            else:
                st.error(t('install.error_manual'))
                st.code(f"{sys.executable} -m pip install --user {' '.join(install_list)}")

    st.stop()

# ------------------------------------------------------------------
# Основные импорты

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings(
    "ignore",
    message="`sklearn\\.utils\\.parallel\\.delayed` should be used with "
    "`sklearn\\.utils\\.parallel\\.Parallel`.*",
    category=UserWarning,
)


def _streamlit_table_value_to_text(value):
    if value is None:
        return value

    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        for encoding in ("utf-8", "cp1251"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                pass
        return raw.hex()

    return str(value)


def _streamlit_safe_table_data(data):
    if not isinstance(data, pd.DataFrame):
        return data

    safe = data.copy()

    for col in safe.columns:
        series = safe[col]
        if series.dtype != "object":
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        type_names = {
            type(value).__name__
            for value in non_null.head(200).to_numpy(dtype=object)
        }
        has_bytes = any(
            isinstance(value, (bytes, bytearray, memoryview))
            for value in non_null.head(200).to_numpy(dtype=object)
        )

        if has_bytes or len(type_names) > 1:
            safe[col] = series.map(_streamlit_table_value_to_text)

    return safe


def safe_histplot(ax, data, bins=30, kde=False, **kwargs):
    """
    Безопасная гистограмма: очищает данные, ограничивает выбросы,
    строит через ax.hist с фиксированным числом бинов.
    """
    # Преобразуем в numpy массив и очищаем
    d = np.asarray(data)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return
    # Ограничиваем 1% и 99% перцентилями
    q01, q99 = np.percentile(d, [1, 99])
    if q99 > q01:
        d = d[(d >= q01) & (d <= q99)]
    if len(d) == 0:
        return
    # Строим гистограмму
    ax.hist(d, bins=bins, **kwargs)
    # Опционально добавляем KDE
    if kde:
        try:
            from scipy.stats import gaussian_kde
            kde_obj = gaussian_kde(d)
            x_grid = np.linspace(d.min(), d.max(), 200)
            # Масштабируем KDE к высоте гистограммы (приблизительно)
            bin_width = (d.max() - d.min()) / bins
            ax.plot(x_grid, kde_obj(x_grid) * len(d) * bin_width,
                    'r-', linewidth=2, label='KDE')
            ax.legend()
        except:
            pass

from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2, norm

from rdkit import Chem
try:
    from rdkit.Chem import Draw
    rdkit_draw_available = True
except Exception:
    Draw = None
    rdkit_draw_available = False
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import joblib

try:
    import shap

    shap_available = True
except Exception:
    shap_available = False

# ------------------------------------------------------------------
# Streamlit page config

st.set_page_config(
    page_title=t('main.page_title'),
    layout="wide"
)

if translation_key_issues:
    with st.sidebar.expander("⚠️ i18n", expanded=False):
        st.warning("Обнаружены отсутствующие ключи локализации.")
        for issue_lang, issue_keys in translation_key_issues.items():
            st.code(
                f"{issue_lang}: " + "\n".join(issue_keys),
                language=None,
            )

# ------------------------------------------------------------------
# Подключение модулей GenQSPR

try:
    for _secret_key in [
        "AUGUR_SPECTRA_INDEX_URL",
        "AUGUR_SPECTRA_INDEX_FILE_ID",
        "AUGUR_SPECTRA_MANIFEST_URL",
        "AUGUR_SPECTRA_MANIFEST_FILE_ID",
        "AUGUR_SPECTRA_SEARCH_CACHE_URL",
        "AUGUR_SPECTRA_SEARCH_CACHE_FILE_ID",
        "AUGUR_SPECTRA_BANK_FOLDER_URL",
        "AUGUR_SPECTRA_BANK_FOLDER_ID",
        "AUGUR_GOOGLE_DRIVE_API_KEY",
    ]:
        try:
            _secret_value = st.secrets.get(_secret_key, "")
        except Exception:
            _secret_value = ""

        if _secret_value:
            os.environ.setdefault(_secret_key, str(_secret_value))

    from modules.spectra_core import *
    from modules.qspr_core import *
    from modules.saod2_core import *
    from modules.structural_filter_core import *
    from modules.prognostic_model_core import *
    from modules.descriptor_bank_core import *
    from modules.descriptor_importance_core import *
    from modules.error_analysis_core import *


    try:
        from modules.morfeus_descriptor_core import *
        morfeus_available, morfeus_import_error = morfeus_is_available()
    except Exception as morfeus_e:
        morfeus_available = False
        morfeus_import_error = str(morfeus_e)
        
    try:
        from modules.dscribe_descriptor_core import *
        dscribe_available, dscribe_import_error = dscribe_is_available()
    except Exception as dscribe_e:
        dscribe_available = False
        dscribe_import_error = str(dscribe_e)

except Exception as e:
    st.error(t('modules.import_error'))
    st.exception(e)
    st.stop()

# ------------------------------------------------------------------
# Константы приложения

RESULTS_DIR = "results"
DATA_BANK_FILE = "data_bank.csv"
HELP_DIR = "help"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(HELP_DIR, exist_ok=True)

def load_help_markdown(filename):
    """
    Читает markdown-файл из папки help.
    """
    help_path = os.path.join(HELP_DIR, filename)

    if not os.path.exists(help_path):
        return t('help.file_not_found', path=help_path)

    with open(help_path, "r", encoding="utf-8") as f:
        return f.read()

# ------------------------------------------------------------------
# CSS

st.markdown(
    """
<style>
    button[data-testid="stTabButton"] {
        font-size: 22px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }

    button[data-baseweb="tab"] {
        font-size: 22px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }

    div[data-testid="stTabs"] button {
        font-size: 22px !important;
    }

    .stTabs [role="tab"] {
        font-size: 22px !important;
        font-weight: 600 !important;
    }

    .tool-badge {
        display: inline-block;
        background: linear-gradient(90deg, #064e3b, #022c22);
        color: white;
        border: 1px solid #047857;
        border-radius: 999px;
        padding: 6px 14px;
        font-weight: 700;
        font-size: 15px;
        margin-top: 14px;
        margin-bottom: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        letter-spacing: 0.2px;
    }
    .label-tooltip {
        display: inline-block;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 6px;
        color: inherit;
        cursor: help;
        border-bottom: 1px dotted rgba(180, 180, 180, 0.8);
    }

    .label-tooltip:hover {
        color: #7dd3fc;
        border-bottom-color: #7dd3fc;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# UI helpers

def show_molecule_viewer(data, target_col, smiles_col="SMILES"):
    """
    Просмотр молекулярных структур по SMILES.
    Показывает сетку молекул с подписями.
    """
    if data is None or data.empty:
        st.info(t('molecule_viewer.no_data'))
        return

    if smiles_col not in data.columns:
        st.warning(t('molecule_viewer.column_not_found', col=smiles_col))
        return

    with st.expander(t('molecule_viewer.title'), expanded=False):
        st.caption(t('molecule_viewer.caption'))

        max_available = len(data)

        st.write(t('molecule_viewer.molecules_count', count=max_available))

        col_mol_1, col_mol_2, col_mol_3, col_mol_4 = st.columns(4)

        with col_mol_1:
            n_show_mols = st.selectbox(t('molecule_viewer.molecules_per_page'), [20, 50, 100, 200], index=2, key="mol_view_n_per_page")

        with col_mol_2:
            total_pages = int(np.ceil(max_available / n_show_mols))

            page_number = st.number_input(t('molecule_viewer.page'), min_value=1, max_value=max(total_pages, 1), value=1, step=1, key="mol_view_page_number")

        with col_mol_3:
            mols_per_row = st.selectbox(t('molecule_viewer.molecules_per_row'), [3, 4, 5, 6], index=1, key="mol_view_per_row")

        with col_mol_4:
            sub_img_size = st.selectbox(t('molecule_viewer.image_size'), [t('molecule_viewer.size_small'), t('molecule_viewer.size_medium'), t('molecule_viewer.size_large')], index=1, key="mol_view_img_size")

        start_row = (int(page_number) - 1) * int(n_show_mols)
        end_row = min(start_row + int(n_show_mols), max_available)

        st.caption(t('molecule_viewer.range_info', start=start_row+1, end=end_row, total=max_available, page=page_number, pages=total_pages))

        size_map = {
            "Маленький": (180, 140),
            "Средний": (220, 180),
            "Крупный": (280, 220),
        }

        img_size = size_map.get(sub_img_size, (220, 180))

        show_invalid = st.checkbox(t('molecule_viewer.show_invalid'), value=False, key="mol_view_show_invalid")

        mols = []
        legends = []
        invalid_rows = []

        view_df = data.iloc[start_row:end_row].copy()

        for idx, row in view_df.iterrows():
            smiles = str(row.get(smiles_col, "")).strip()

            if not smiles or smiles.lower() in ["nan", "none"]:
                invalid_rows.append({
                    t('molecule_grid.table_row_number'): idx,
                    "SMILES": smiles
                })
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                mol = None

            if mol is None:
                invalid_rows.append({
                    t('molecule_grid.table_row_number'): idx,
                    "SMILES": smiles
                })
                continue

            mols.append(mol)

            legend_parts = [f"{idx}"]

            if target_col in row.index:
                value = row.get(target_col, "")
                legend_parts.append(f"{target_col}={value}")

            legends.append(" | ".join(legend_parts))

        if mols and Draw is None:
            st.warning("Молекулярные изображения недоступны: RDKit Draw не импортирован.")
        elif mols:
            try:
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=int(mols_per_row),
                    subImgSize=img_size,
                    legends=legends,
                    useSVG=False
                )

                st.image(img)

            except Exception as e:
                st.error(t('molecule_viewer.build_error', error=e))
        else:
            st.warning(t('molecule_viewer.no_molecules'))

        if show_invalid:
            if invalid_rows:
                st.warning(t('molecule_viewer.invalid_count', count=len(invalid_rows)))
                st.dataframe(
                    pd.DataFrame(invalid_rows),
                    width="stretch",
                    hide_index=True
                )
            else:
                st.success(t('molecule_viewer.no_invalid'))

def show_molecule_grid_from_table(
    table_df,
    title=t('molecule_grid.default_title'),
    target_col=None,
    smiles_col="SMILES",
    max_molecules=50,
    key_prefix="mol_grid"
):
    """
    Показывает структуры молекул из любой таблицы, где есть колонка SMILES.
    Удобно для выбросов, подозрительных веществ и ошибок модели.
    """
    if table_df is None or table_df.empty:
        return

    if smiles_col not in table_df.columns:
        possible_smiles_cols = [
            "SMILES",
            "input_smiles",
            "Исходный SMILES",
            "canonical_smiles",
            "Канонический SMILES",
            "smiles_a",
            "smiles_b",
            "SMILES A",
            "SMILES B",
        ]

        found_smiles_col = None

        for candidate in possible_smiles_cols:
            if candidate in table_df.columns:
                found_smiles_col = candidate
                break

        if found_smiles_col is None:
            st.info(t('molecule_grid.smiles_required', cols=', '.join(possible_smiles_cols)))
            return

        smiles_col = found_smiles_col

    with st.expander(title, expanded=False):
        n_available = len(table_df)

        st.write(t('molecule_grid.compounds_count', count=n_available))

        col_1, col_2, col_3 = st.columns(3)
      
        with col_1:

            max_allowed = min(
                max_molecules,
                n_available
            )

            if max_allowed <= 1:

                n_show = 1

                st.write(t('molecule_grid.showing_one'))

            else:

                n_show = st.slider(
                    t('molecule_grid.how_many_structures'),
                    min_value=1,
                    max_value=max_allowed,
                    value=min(20, max_allowed),
                    step=1,
                    key=f"{key_prefix}_n_show"
                )

        with col_2:
            mols_per_row = st.selectbox(
                t('molecule_grid.molecules_per_row'),
                [3, 4, 5, 6],
                index=1,
                key=f"{key_prefix}_per_row"
            )

        with col_3:
            sub_img_size = st.selectbox(
                t('molecule_grid.image_size'),
                [t('molecule_grid.size_small'), t('molecule_grid.size_medium'), t('molecule_grid.size_large')],
                index=1,
                key=f"{key_prefix}_img_size"
            )

        size_map = {
            "Маленький": (180, 140),
            "Средний": (220, 180),
            "Крупный": (280, 220),
        }

        img_size = size_map.get(sub_img_size, (220, 180))

        mols = []
        legends = []
        invalid_rows = []

        view_df = table_df.head(n_show).copy()

        mol_counter = 0

        for idx, row in view_df.iterrows():
            raw_smiles = str(row.get(smiles_col, "")).strip()

            # В SAOD иногда в одной ячейке может быть несколько SMILES через ;
            smiles_list = [
                s.strip()
                for s in raw_smiles.replace("\n", ";").split(";")
                if s.strip()
            ]

            if not smiles_list:
                invalid_rows.append({
                    t('molecule_grid.index'): idx,
                    t('molecule_grid.smiles'): raw_smiles,
                    t('molecule_grid.reason_empty'): "пустая строка"
                })
                continue

            for smi_i, smiles in enumerate(smiles_list, start=1):
                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    invalid_rows.append({
                        t('molecule_grid.index'): idx,
                        t('molecule_grid.smiles'): smiles,
                        t('molecule_grid.reason_rdkit'): "RDKit не распознал SMILES"
                    })
                    continue

                mols.append(mol)
                mol_counter += 1

                legend_parts = []

                # Нормальная нумерация в подписи
                if "№" in row.index:
                    legend_parts.append(f"№{row['№']}")
                elif "Номер в исходной таблице" in row.index:
                    legend_parts.append(f"№{row['Номер в исходной таблице']}")
                elif "ID вещества" in row.index:
                    legend_parts.append(f"ID {row['ID вещества']}")
                elif "compound_id" in row.index:
                    legend_parts.append(f"ID {row['compound_id']}")
                else:
                    legend_parts.append(f"№{mol_counter}")

                # Если из одной ячейки получилось несколько SMILES
                if len(smiles_list) > 1:
                    legend_parts.append(f"mol {smi_i}")

                # Роль A/B для поломок правил
                if "Роль" in row.index:
                    legend_parts.append(str(row["Роль"]))
                elif "role" in row.index:
                    legend_parts.append(str(row["role"]))

                # Значение свойства — коротко, без длинного названия колонки
                if target_col is not None and target_col in row.index:
                    try:
                        value = float(row[target_col])
                        legend_parts.append(f"y={value:.2f}")
                    except Exception:
                        legend_parts.append(f"y={row[target_col]}")

                # Махаланобис
                if "Расстояние Махаланобиса" in row.index:
                    try:
                        legend_parts.append(f"MD={float(row['Расстояние Махаланобиса']):.2f}")
                    except Exception:
                        pass

                # Ошибка прогноза
                if "Ошибка" in row.index:
                    try:
                        legend_parts.append(f"err={float(row['Ошибка']):.2f}")
                    except Exception:
                        pass

                # SAOD-статус, если есть
                if "final_status" in row.index:
                    status_text = str(row["final_status"])
                    if len(status_text) > 18:
                        status_text = status_text[:18] + "..."
                    legend_parts.append(status_text)

                legends.append(" | ".join(legend_parts))

        if mols and Draw is None:
            st.warning("Молекулярные изображения недоступны: RDKit Draw не импортирован.")
        elif mols:
            try:
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=int(mols_per_row),
                    subImgSize=img_size,
                    legends=legends,
                    useSVG=False
                )

                st.image(img)

            except Exception as e:
                st.error(t('molecule_grid.build_error', error=e))
        else:
            st.warning(t('molecule_grid.no_molecules'))

        if invalid_rows:
            st.warning(t('molecule_grid.invalid_count', count=len(invalid_rows)))
            st.dataframe(
                pd.DataFrame(invalid_rows),
                width="stretch",
                hide_index=True
            )
def show_dataset_change_report(
    before_df,
    after_df,
    target_col,
    smiles_col=None,
    title="📎 Эффект инструмента на датасет",
    removed_title="Удалённые / исключённые вещества",
    key_prefix="dataset_change"
):
    """
    Показывает отчёт до/после для инструмента, который меняет датасет:
    - сколько веществ было / осталось / удалено;
    - распределение свойства до и после;
    - boxplot до и после;
    - таблица удалённых веществ;
    - структуры удалённых веществ, если есть SMILES.
    """
    if before_df is None or after_df is None:
        return

    if before_df.empty:
        st.info(t('dataset_change.empty'))
        return

    if target_col not in before_df.columns:
        st.warning(t('dataset_change.target_not_found_original', col=target_col))
        return

    if target_col not in after_df.columns:
        st.warning(t('dataset_change.target_not_found_final', col=target_col))
        return

    before = before_df.copy().reset_index(drop=False).rename(columns={"index": "_original_index"})
    after = after_df.copy().reset_index(drop=False).rename(columns={"index": "_original_index"})

    before[target_col] = pd.to_numeric(
        before[target_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    after[target_col] = pd.to_numeric(
        after[target_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    n_before = len(before)
    n_after = len(after)
    n_removed = max(n_before - n_after, 0)
    keep_percent = n_after / n_before * 100 if n_before > 0 else 0.0

    st.subheader(title)

    col_1, col_2, col_3, col_4 = st.columns(4)

    with col_1:
        st.metric(t('dataset_change.initial_count'), n_before)

    with col_2:
        st.metric(t('dataset_change.remaining_count'), n_after)

    with col_3:
        st.metric(t('dataset_change.removed_count'), n_removed)

    with col_4:
        st.metric(t('dataset_change.kept_percent'), f"{keep_percent:.1f}%")

    before_values = before[target_col].dropna()
    after_values = after[target_col].dropna()

    if before_values.empty or after_values.empty:
        st.warning(t('dataset_change.insufficient_values'))
        return

    stats_before = before_values.describe()
    stats_after = after_values.describe()

    stats_compare = pd.DataFrame({
        t('dataset_change.param_count'): [
            t('dataset_change.stat_count'),
            t('dataset_change.stat_mean'),
            t('dataset_change.stat_std'),
            t('dataset_change.stat_min'),
            t('dataset_change.stat_25pct'),
            t('dataset_change.stat_median'),
            t('dataset_change.stat_75pct'),
            t('dataset_change.stat_max'),
        ],
        t('dataset_change.before'): [
            stats_before.get("count", np.nan),
            stats_before.get("mean", np.nan),
            stats_before.get("std", np.nan),
            stats_before.get("min", np.nan),
            stats_before.get("25%", np.nan),
            stats_before.get("50%", np.nan),
            stats_before.get("75%", np.nan),
            stats_before.get("max", np.nan),
        ],
        t('dataset_change.after'): [
            stats_after.get("count", np.nan),
            stats_after.get("mean", np.nan),
            stats_after.get("std", np.nan),
            stats_after.get("min", np.nan),
            stats_after.get("25%", np.nan),
            stats_after.get("50%", np.nan),
            stats_after.get("75%", np.nan),
            stats_after.get("max", np.nan),
        ],
    })

    stats_compare[t('dataset_change.change')] = stats_compare[t('dataset_change.after')] - stats_compare[t('dataset_change.before')]

    st.markdown(t('dataset_change.stats_title'))

    st.dataframe(
        stats_compare.round(4),
        width="stretch",
        hide_index=True
    )

    st.markdown(t('dataset_change.distribution_title'))

    col_hist_1, col_hist_2 = st.columns(2)

    with col_hist_1:
        fig_before, ax_before = plt.subplots(figsize=(5, 3.5))
        safe_histplot(ax_before, before_values, kde=True, color='steelblue', edgecolor='black', alpha=0.7)
        ax_before.set_title(t('dataset_change.hist_before', col=target_col))
        ax_before.set_xlabel(target_col)
        ax_before.set_ylabel(t('dataset_change.hist_count'))
        ax_before.grid(True, alpha=0.25)
        fig_before.tight_layout()
        st.pyplot(fig_before)

    with col_hist_2:
        fig_after, ax_after = plt.subplots(figsize=(5, 3.5))
        safe_histplot(ax_after, after_values, kde=True, color='steelblue', edgecolor='black', alpha=0.7)
        ax_after.set_title(t('dataset_change.hist_after', col=target_col))
        ax_after.set_xlabel(target_col)
        ax_after.set_ylabel(t('dataset_change.hist_count'))
        ax_after.grid(True, alpha=0.25)
        fig_after.tight_layout()
        st.pyplot(fig_after)

    st.markdown(t('dataset_change.boxplot_title'))

    compare_box_df = pd.DataFrame({
        target_col: pd.concat([before_values, after_values], ignore_index=True),
        t('dataset_change.state'): (
            [t('dataset_change.state_before')] * len(before_values)
            + [t('dataset_change.state_after')] * len(after_values)
        )
    })

    fig_box, ax_box = plt.subplots(figsize=(7, 3.5))
    sns.boxplot(
        data=compare_box_df,
        x=t('dataset_change.state'),
        y=target_col,
        ax=ax_box
    )
    ax_box.set_title(t('dataset_change.boxplot_title_compare', col=target_col))
    ax_box.grid(True, alpha=0.25)
    fig_box.tight_layout()
    st.pyplot(fig_box)

    # ------------------------------------------------------------
    # Таблица удалённых / исключённых веществ

    removed_df = pd.DataFrame()

    if "_original_index" in before.columns and "_original_index" in after.columns:
        after_indices = set(after["_original_index"].tolist())
        removed_df = before[~before["_original_index"].isin(after_indices)].copy()

    if removed_df.empty:
        st.success(t('dataset_change.no_removed'))
        return

    st.markdown(f"### {removed_title}")

    show_cols = []

    for col in [
        "_original_index",
        smiles_col,
        "SMILES",
        "smiles",
        "canonical_smiles",
        "name",
        "Name",
        "CAS",
        "cas",
        target_col,
        "SAOD_manual_decision",
        "SAOD_auto_recommendation",
        "final_status",
        "human_observation",
    ]:
        if col is not None and col in removed_df.columns and col not in show_cols:
            show_cols.append(col)

    removed_view = removed_df[show_cols].copy() if show_cols else removed_df.copy()

    if "_original_index" in removed_view.columns:
        removed_view = removed_view.rename(columns={"_original_index": t('dataset_change.original_index')})
        removed_view[t('dataset_change.original_index')] = removed_view[t('dataset_change.original_index')] + 1

    st.dataframe(
        removed_view.head(300),
        width="stretch",
        hide_index=True
    )

    csv_removed = removed_view.to_csv(index=False).encode("utf-8")

    st.download_button(
        t('dataset_change.download_removed'),
        csv_removed,
        f"{key_prefix}_removed_compounds.csv",
        "text/csv",
        key=f"{key_prefix}_download_removed"
    )

    # ------------------------------------------------------------
    # Структуры удалённых веществ

    smiles_for_removed = None

    possible_smiles_cols = [
        smiles_col,
        "SMILES",
        "smiles",
        "input_smiles",
        "canonical_smiles",
    ]

    for candidate in possible_smiles_cols:
        if candidate is not None and candidate in removed_df.columns:
            smiles_for_removed = candidate
            break

    if smiles_for_removed is not None:
        show_molecule_grid_from_table(
            table_df=removed_df,
            title=t('dataset_change.removed_structures_title'),
            target_col=target_col,
            smiles_col=smiles_for_removed,
            max_molecules=100,
            key_prefix=f"{key_prefix}_removed_structures"
        )

def incremental_to_numeric(series):
    """
    Безопасное преобразование в число.
    Поддерживает десятичную запятую.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def fit_incremental_contributions(
    data,
    target_col,
    increment_cols,
    use_intercept=True
):
    """
    Расчёт инкрементных вкладов методом наименьших квадратов.
    Аналог Excel-функции ЛИНЕЙН.

    Модель:
    y = b0 + a1*x1 + a2*x2 + ... + an*xn
    """
    if data is None or data.empty:
        raise ValueError(t('incremental.no_data'))

    if target_col not in data.columns:
        raise ValueError(t('incremental.target_not_found', col=target_col))

    if not increment_cols:
        raise ValueError(t('incremental.no_increments'))

    work = data.copy()

    work[target_col] = incremental_to_numeric(work[target_col])

    for col in increment_cols:
        if col not in work.columns:
            raise ValueError(t('incremental.increment_not_found', col=col))

        work[col] = incremental_to_numeric(work[col])

    valid_mask = work[target_col].notna()

    for col in increment_cols:
        valid_mask = valid_mask & work[col].notna()

    work_valid = work.loc[valid_mask].copy()

    if work_valid.empty:
        raise ValueError(t('incremental.empty_after_cleaning'))

    X = work_valid[increment_cols].values.astype(float)
    y = work_valid[target_col].values.astype(float)

    if len(y) <= len(increment_cols):
        raise ValueError(t('incremental.too_few_compounds'))

    model = LinearRegression(fit_intercept=use_intercept)
    model.fit(X, y)

    y_pred = np.ravel(model.predict(X))
    errors = y - y_pred

    metrics = qspr_metrics(y, y_pred)

    coef_table = pd.DataFrame({
        t('incremental.coef_table_group'): increment_cols,
        t('incremental.coef_table_contribution'): model.coef_
    })

    if use_intercept:
        intercept_row = pd.DataFrame({
            t('incremental.coef_table_group'): [t('incremental.intercept')],
            t('incremental.coef_table_contribution'): [model.intercept_]
        })

        coef_table = pd.concat(
            [intercept_row, coef_table],
            ignore_index=True
        )

    result_table = work_valid.copy()
    result_table[t('incremental.result_predicted')] = y_pred
    result_table[t('incremental.result_error')] = errors
    result_table[t('incremental.result_abs_error')] = np.abs(errors)

    equation_parts = []

    if use_intercept:
        equation_parts.append(f"{model.intercept_:.6g}")

    for name, coef in zip(increment_cols, model.coef_):
        sign = "+" if coef >= 0 else "-"
        equation_parts.append(f"{sign} {abs(coef):.6g}·{name}")

    equation = f"{target_col} = " + " ".join(equation_parts)

    return {
        "model": model,
        "X": X,
        "y": y,
        "y_pred": y_pred,
        "errors": errors,
        "coef_table": coef_table,
        "result_table": result_table,
        "metrics": metrics,
        "equation": equation,
        "valid_indices": work_valid.index.tolist(),
        "increment_cols": increment_cols,
        "target_col": target_col,
        "use_intercept": use_intercept
    }

def show_saod_molecule_grid(
    table_df,
    processed,
    title=t('saod_grid.default_title'),
    key_prefix="saod_molecules",
    max_molecules=100
):
    """
    Визуализация структур для SAOD-таблиц.
    Показывает структуры одной RDKit-сеткой с короткими подписями:
    ID/номер вещества + значение свойства.
    """
    if table_df is None or table_df.empty:
        return

    work = table_df.copy()

    # Добавляем SMILES, если в SAOD-таблице их нет.
    work = saod2_add_smiles_for_visualization(work, processed)

    if "SMILES" not in work.columns:
        st.info(t('saod_grid.no_smiles_column'))
        return

    # Подтягиваем служебные поля из processed, если возможно.
    if processed is not None and not processed.empty:
        proc = processed.copy()

        merge_key = None

        if "compound_id" in work.columns and "compound_id" in proc.columns:
            merge_key = "compound_id"
        elif "canonical_smiles" in work.columns and "canonical_smiles" in proc.columns:
            merge_key = "canonical_smiles"
        elif "inchikey" in work.columns and "inchikey" in proc.columns:
            merge_key = "inchikey"

        if merge_key is not None:
            useful_cols = [merge_key]

            for col in [
                "input_smiles",
                "canonical_smiles",
                "inchikey",
                "property_value",
                "compound_id",
                "name",
                "exact_pattern",
                "overall_checkability",
                "checkability_level",
                "final_status"
            ]:
                if col in proc.columns and col not in useful_cols:
                    useful_cols.append(col)

            proc_small = proc[useful_cols].drop_duplicates(
                subset=[merge_key],
                keep="first"
            )

            work = work.merge(
                proc_small,
                on=merge_key,
                how="left",
                suffixes=("", "_proc")
            )

    with st.expander(title, expanded=False):
        n_available = len(work)

        st.write(t('saod_grid.compounds_count', count=n_available))

        col_1, col_2, col_3 = st.columns(3)

        with col_1:
            n_show_max = min(max_molecules, n_available)

            if n_show_max > 1:
                n_show = st.slider(
                    t('saod_grid.how_many_structures'),
                    min_value=1,
                    max_value=n_show_max,
                    value=min(20, n_show_max),
                    step=1,
                    key=f"{key_prefix}_n_show"
                )
            else:
                n_show = max(0, n_show_max)
                st.metric(t('saod_grid.how_many_structures'), n_show)

        with col_2:
            mols_per_row = st.selectbox(
                t('saod_grid.molecules_per_row'),
                [3, 4, 5],
                index=1,
                key=f"{key_prefix}_per_row"
            )

        with col_3:
            sub_img_size = st.selectbox(
                t('saod_grid.image_size'),
                [t('saod_grid.size_small'), t('saod_grid.size_medium'), t('saod_grid.size_large')],
                index=1,
                key=f"{key_prefix}_img_size"
            )

        size_map = {
            "Маленький": (220, 200),
            "Средний": (280, 250),
            "Крупный": (340, 300),
        }

        img_size = size_map.get(sub_img_size, (280, 250))

        view_df = work.head(n_show).copy()

        mols = []
        legends = []
        invalid_rows = []

        for local_i, (idx, row) in enumerate(view_df.iterrows(), start=1):
            smiles = str(row.get("SMILES", "")).strip()

            if not smiles or smiles.lower() in ["nan", "none"]:
                invalid_rows.append({
                    t('saod_grid.index'): idx,
                    t('saod_grid.smiles'): smiles,
                    t('saod_grid.reason_empty'): "пустой SMILES"
                })
                continue

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                invalid_rows.append({
                    t('saod_grid.index'): idx,
                    t('saod_grid.smiles'): smiles,
                    t('saod_grid.reason_rdkit'): "RDKit не распознал SMILES"
                })
                continue

            mols.append(mol)

            # Короткий номер вещества.
            if "compound_id" in row.index and pd.notna(row.get("compound_id")):
                number_text = f"ID {row.get('compound_id')}"
            elif "ID вещества" in row.index and pd.notna(row.get("ID вещества")):
                number_text = f"ID {row.get('ID вещества')}"
            elif "Номер в исходной таблице" in row.index and pd.notna(row.get("Номер в исходной таблице")):
                number_text = f"№ {row.get('Номер в исходной таблице')}"
            elif "row_index" in row.index and pd.notna(row.get("row_index")):
                number_text = f"№ {int(row.get('row_index')) + 1}"
            else:
                number_text = f"№ {local_i}"

            # Значение свойства.
            y_value = None

            for y_col in [
                "property_value",
                "property_value_proc",
                "Значение свойства",
                "BoilingPoint"
            ]:
                if y_col in row.index and pd.notna(row.get(y_col)):
                    y_value = row.get(y_col)
                    break

            if y_value is not None:
                try:
                    y_text = f"y={float(y_value):.2f}"
                except Exception:
                    y_text = f"y={y_value}"
            else:
                y_text = ""

            legend_parts = [number_text]

            if y_text:
                legend_parts.append(y_text)

            legends.append("  ".join(legend_parts))

        if mols and Draw is None:
            st.warning("Молекулярные изображения недоступны: RDKit Draw не импортирован.")
        elif mols:
            try:
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=int(mols_per_row),
                    subImgSize=img_size,
                    legends=legends,
                    useSVG=False
                )

                st.image(img)

            except Exception as e:
                st.error(t('saod_grid.build_error', error=e))
        else:
            st.warning(t('saod_grid.no_molecules'))

        if invalid_rows:
            st.warning(t('saod_grid.invalid_count', count=len(invalid_rows)))
            st.dataframe(
                pd.DataFrame(invalid_rows),
                width="stretch",
                hide_index=True
            )
            
def saod2_add_smiles_for_visualization(table_df, processed):
    """
    Добавляет колонку SMILES в SAOD-таблицу для визуализации структур.
    Берёт SMILES из самой таблицы, а если их нет — подтягивает из processed.
    """
    if table_df is None or table_df.empty:
        return table_df

    out = table_df.copy()

    possible_existing = [
        "SMILES",
        "input_smiles",
        "canonical_smiles",
        "Исходный SMILES",
        "Канонический SMILES",
        "smiles_a",
        "smiles_b",
        "SMILES A",
        "SMILES B",
    ]

    for col in possible_existing:
        if col in out.columns:
            out["SMILES"] = out[col].astype(str)
            return out

    if processed is None or processed.empty:
        return out

    proc = processed.copy()

    smiles_source_col = None

    for col in ["input_smiles", "SMILES", "canonical_smiles"]:
        if col in proc.columns:
            smiles_source_col = col
            break

    if smiles_source_col is None:
        return out

    # Связь по compound_id
    if "compound_id" in out.columns and "compound_id" in proc.columns:
        proc_small = proc[["compound_id", smiles_source_col]].drop_duplicates(
            subset=["compound_id"],
            keep="first"
        )

        proc_small = proc_small.rename(columns={smiles_source_col: "SMILES"})

        out = out.merge(
            proc_small,
            on="compound_id",
            how="left"
        )

        return out

    # Связь по canonical_smiles
    if "canonical_smiles" in out.columns and "canonical_smiles" in proc.columns:
        proc_small = proc[["canonical_smiles", smiles_source_col]].drop_duplicates(
            subset=["canonical_smiles"],
            keep="first"
        )

        proc_small = proc_small.rename(columns={smiles_source_col: "SMILES"})

        out = out.merge(
            proc_small,
            on="canonical_smiles",
            how="left"
        )

        return out

    # Связь по inchikey
    if "inchikey" in out.columns and "inchikey" in proc.columns:
        proc_small = proc[["inchikey", smiles_source_col]].drop_duplicates(
            subset=["inchikey"],
            keep="first"
        )

        proc_small = proc_small.rename(columns={smiles_source_col: "SMILES"})

        out = out.merge(
            proc_small,
            on="inchikey",
            how="left"
        )

        return out

    return out
 
def qspr_standardize_molecule_dataset(
    input_df,
    smiles_col,
    target_col=None,
    remove_duplicates_by_inchikey=False
):
    """
    Стандартизация структур перед расчётом дескрипторов.

    Делает:
    - проверку SMILES;
    - выбор главного фрагмента;
    - очистку/нормализацию;
    - нейтрализацию зарядов;
    - canonical SMILES;
    - InChIKey;
    - диагностику смесей, неорганики, металлоорганики, полимерных/неполных записей;
    - удаление дубликатов по InChIKey.
    """
    if input_df is None or input_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def qspr_standardize_molecule_dataset(
    input_df,
    smiles_col,
    target_col=None,
    remove_duplicates_by_inchikey=False
):
    """
    Стандартизация структур перед расчётом дескрипторов.

    Делает:
    - проверку SMILES;
    - выбор главного фрагмента;
    - очистку/нормализацию;
    - нейтрализацию зарядов;
    - canonical SMILES;
    - InChIKey;
    - диагностику смесей, неорганики, металлоорганики, полимерных/неполных записей;
    - удаление дубликатов по InChIKey.
    """
    if input_df is None or input_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if smiles_col not in input_df.columns:
        raise ValueError(t('standardization.smiles_column_not_found', col=smiles_col))

    work = input_df.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()
    rows = []

    metal_atomic_numbers = {
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 55, 56, 57, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83
    }

    chooser = rdMolStandardize.LargestFragmentChooser()
    normalizer = rdMolStandardize.Normalizer()
    uncharger = rdMolStandardize.Uncharger()

    for idx, row in work.iterrows():
        raw_smiles = str(row.get(smiles_col, "")).strip()

        result = row.to_dict()
        result["Номер исходной строки"] = idx + 1
        result["input_smiles_original"] = raw_smiles
        result["standardized_smiles"] = ""
        result["canonical_smiles"] = ""
        result["inchikey"] = ""
        result["standardization_status"] = ""
        result["standardization_warnings"] = ""

        warnings = []

        if not raw_smiles or raw_smiles.lower() in ["nan", "none"]:
            result["standardization_status"] = "empty_smiles"
            rows.append(result)
            continue

        if "." in raw_smiles:
            warnings.append(t('standardization.warning_mix'))

        if "*" in raw_smiles or "[" in raw_smiles and "]n" in raw_smiles:
            warnings.append(t('standardization.warning_polymer'))

        try:
            mol = Chem.MolFromSmiles(raw_smiles, sanitize=True)

            if mol is None:
                result["standardization_status"] = "invalid_smiles"
                rows.append(result)
                continue

            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                result["standardization_status"] = f"valence_or_sanitize_error: {e}"
                rows.append(result)
                continue

            atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            carbon_count = atom_nums.count(6)
            has_metal = any(num in metal_atomic_numbers for num in atom_nums)

            if carbon_count == 0:
                warnings.append(t('standardization.warning_inorganic'))

            if has_metal and carbon_count > 0:
                warnings.append(t('standardization.warning_organometallic'))

            elif has_metal:
                warnings.append(t('standardization.warning_metal'))

            try:
                mol = chooser.choose(mol)
            except Exception:
                warnings.append(t('standardization.warning_fragment_fail'))

            try:
                mol = rdMolStandardize.Cleanup(mol)
            except Exception:
                warnings.append(t('standardization.warning_cleanup'))

            try:
                mol = normalizer.normalize(mol)
            except Exception:
                warnings.append(t('standardization.warning_normalization'))

            try:
                mol = uncharger.uncharge(mol)
            except Exception:
                warnings.append(t('standardization.warning_uncharge'))

            standardized_smiles = Chem.MolToSmiles(mol, canonical=False)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            try:
                inchikey = Chem.MolToInchiKey(mol)
            except Exception:
                inchikey = ""

            result["standardized_smiles"] = standardized_smiles
            result["canonical_smiles"] = canonical_smiles
            result["inchikey"] = inchikey
            result["standardization_status"] = "ok"
            result["standardization_warnings"] = "; ".join(warnings)

        except Exception as e:
            result["standardization_status"] = f"standardization_error: {e}"

        rows.append(result)

    standardized_df = pd.DataFrame(rows)

    duplicate_removed_df = pd.DataFrame()

    if (
        remove_duplicates_by_inchikey
        and "inchikey" in standardized_df.columns
        and not standardized_df.empty
    ):
        valid_inchikey_mask = standardized_df["inchikey"].astype(str).str.strip() != ""

        duplicate_mask = (
            valid_inchikey_mask
            & standardized_df.duplicated(subset=["inchikey"], keep="first")
        )

        duplicate_removed_df = standardized_df.loc[duplicate_mask].copy()
        standardized_df = standardized_df.loc[~duplicate_mask].copy()

    summary_df = pd.DataFrame([
        {t('standardization.summary_prompt'): t('standardization.summary_initial'), t('standardization.summary_value'): len(input_df)},
        {t('standardization.summary_prompt'): t('standardization.summary_success'), t('standardization.summary_value'): len(standardized_df)},
        {t('standardization.summary_prompt'): t('standardization.summary_invalid_smiles'), t('standardization.summary_value'): int((standardized_df["standardization_status"] != "ok").sum()) if not standardized_df.empty else 0},
        {t('standardization.summary_prompt'): t('standardization.summary_warnings'), t('standardization.summary_value'): int((standardized_df["standardization_warnings"].astype(str).str.strip() != "").sum()) if not standardized_df.empty else 0},
        {t('standardization.summary_prompt'): t('standardization.summary_duplicates'), t('standardization.summary_value'): len(duplicate_removed_df)},
    ])

    return standardized_df.reset_index(drop=True), duplicate_removed_df.reset_index(drop=True), summary_df

def qspr_classify_molecule_simple(mol):
    """
    Простая классификация вещества по основным структурным классам.
    Возвращает список классов: алканы, алкены, спирты, кислоты и т.п.
    """
    if mol is None:
        return [t('classify.invalid_structure')]

    classes = []

    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    has_c = 6 in atom_nums
    has_h_only_elements = all(num in [1, 6] for num in atom_nums)

    metal_atomic_numbers = {
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 55, 56, 57, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83
    }

    has_metal = any(num in metal_atomic_numbers for num in atom_nums)

    if not has_c:
        classes.append(t('classify.inorganic'))

    if has_metal and has_c:
        classes.append(t('classify.organometallic'))
    elif has_metal:
        classes.append(t('classify.metal_containing'))

    if "." in Chem.MolToSmiles(mol):
        classes.append(t('classify.mixture'))

    ring_info = mol.GetRingInfo()
    has_ring = ring_info.NumRings() > 0
    has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    has_double_cc = mol.HasSubstructMatch(Chem.MolFromSmarts("C=C"))
    has_triple_cc = mol.HasSubstructMatch(Chem.MolFromSmarts("C#C"))

    if has_h_only_elements:
        if not has_double_cc and not has_triple_cc and not has_aromatic:
            if has_ring:
                classes.append(t('classify.cycloalkanes'))
            else:
                classes.append(t('classify.alkanes'))
        if has_double_cc:
            classes.append(t('classify.alkenes'))
        if has_triple_cc:
            classes.append(t('classify.alkynes'))
        if has_aromatic:
            classes.append(t('classify.aromatic_hydrocarbons'))
    else:
        if has_aromatic:
            classes.append(t('classify.aromatic_compounds'))

    smarts_map = {
        t('classify.alcohols'): "[OX2H]",
        t('classify.phenols'): "c[OX2H]",
        t('classify.ethers'): "[OD2]([#6])[#6]",
        t('classify.aldehydes'): "[CX3H1](=O)[#6,#1]",
        t('classify.ketones'): "[#6][CX3](=O)[#6]",
        t('classify.carboxylic_acids'): "C(=O)[OX2H1]",
        t('classify.carboxylates'): "C(=O)[O-]",
        t('classify.esters'): "C(=O)O[#6]",
        t('classify.amides'): "C(=O)N",
        t('classify.amines'): "[NX3;H2,H1,H0;!$(NC=O)]",
        t('classify.nitriles'): "C#N",
        t('classify.nitro_compounds'): "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
        t('classify.halogenated'): "[F,Cl,Br,I]",
        t('classify.thiols'): "[SX2H]",
        t('classify.sulfides'): "[#6][SX2][#6]",
        t('classify.sulfones'): "S(=O)(=O)",
        t('classify.organophosphorus'): "P",
    }

    for class_name, smarts in smarts_map.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            classes.append(class_name)

    if not classes:
        classes.append(t('classify.other_organic'))

    return sorted(set(classes))

def qspr_analyze_dataset_for_qspr(data, smiles_col, target_col):
    """
    Диагностика пригодности датасета перед QSPR:
    - целевое свойство;
    - SMILES;
    - дубликаты;
    - подозрительность целевой колонки;
    - классы веществ.
    """
    work = data.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()

    y = pd.to_numeric(
        work[target_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    n_total = len(work)
    n_y_valid = int(y.notna().sum())
    n_y_missing = int(y.isna().sum())
    n_y_unique = int(y.dropna().nunique())

    y_min = float(y.min()) if n_y_valid else np.nan
    y_median = float(y.median()) if n_y_valid else np.nan
    y_max = float(y.max()) if n_y_valid else np.nan
    y_range = y_max - y_min if n_y_valid else np.nan

    q1 = float(y.quantile(0.25)) if n_y_valid else np.nan
    q3 = float(y.quantile(0.75)) if n_y_valid else np.nan
    iqr = q3 - q1 if n_y_valid else np.nan

    mad = float(np.median(np.abs(y.dropna() - y_median))) if n_y_valid else np.nan

    if n_y_valid and np.isfinite(iqr) and iqr > 0:
        outlier_mask = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)
        n_iqr_outliers = int(outlier_mask.sum())
    else:
        n_iqr_outliers = 0

    smiles_raw = work[smiles_col].astype(str).fillna("").str.strip()
    n_smiles_empty = int((smiles_raw == "").sum())

    valid_mols = []
    canonical_smiles = []
    inchikeys = []
    class_rows = []
    invalid_smiles_rows = []

    for idx, smi in smiles_raw.items():
        mol = None

        if smi:
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:
                mol = None

        if mol is None:
            invalid_smiles_rows.append({
                t('analyze.original_row'): idx + 1,
                smiles_col: smi,
                target_col: y.loc[idx] if idx in y.index else np.nan,
            })
            canonical_smiles.append("")
            inchikeys.append("")
            continue

        valid_mols.append(mol)

        try:
            can_smi = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            can_smi = ""

        try:
            ik = Chem.MolToInchiKey(mol)
        except Exception:
            ik = ""

        canonical_smiles.append(can_smi)
        inchikeys.append(ik)

        for cls in qspr_classify_molecule_simple(mol):
            class_rows.append({
                t('analyze.class'): cls,
                t('analyze.original_row'): idx + 1
            })

    n_valid_mols = len(valid_mols)
    n_invalid_smiles = len(invalid_smiles_rows)

    temp = work.copy()
    temp["_qspr_canonical_smiles"] = canonical_smiles
    temp["_qspr_inchikey"] = inchikeys
    temp["_target_numeric"] = y

    valid_inchikey = temp["_qspr_inchikey"].astype(str).str.strip() != ""

    n_unique_structures = int(temp.loc[valid_inchikey, "_qspr_inchikey"].nunique())
    n_duplicate_structures = int(valid_inchikey.sum() - n_unique_structures)

    duplicate_summary = (
        temp.loc[valid_inchikey]
        .groupby("_qspr_inchikey")["_target_numeric"]
        .agg(
            n_records="size",
            n_unique_values=lambda x: x.dropna().nunique(),
            min_value="min",
            max_value="max"
        )
        .reset_index()
    )

    conflict_duplicates = duplicate_summary[
        (duplicate_summary["n_records"] > 1)
        & (duplicate_summary["n_unique_values"] > 1)
    ].copy()

    n_conflict_duplicates = len(conflict_duplicates)

    target_name_lower = str(target_col).lower()

    suspicious_target_reasons = []

    if any(x in target_name_lower for x in ["id", "index", "номер", "row", "line", "source_line"]):
        suspicious_target_reasons.append(t('analyze.suspicious_id_like'))

    if n_y_valid > 0:
        unique_ratio = n_y_unique / n_y_valid

        if unique_ratio > 0.95 and n_y_valid > 100:
            suspicious_target_reasons.append(t('analyze.suspicious_unique'))

        y_valid = y.dropna().reset_index(drop=True)

        if len(y_valid) > 10:
            diffs = y_valid.diff().dropna()
            monotonic_ratio = float((diffs > 0).mean())

            if monotonic_ratio > 0.90:
                suspicious_target_reasons.append(t('analyze.suspicious_monotonic'))

    if suspicious_target_reasons:
        target_status = t('analyze.status_warning')
        target_comment = "; ".join(suspicious_target_reasons)
    else:
        target_status = t('analyze.status_ok')
        target_comment = t('analyze.comment_ok')

    diagnostics_table = pd.DataFrame([
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_column'), t('analyze.diag_value'): target_col, t('analyze.diag_comment'): target_comment},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_status'), t('analyze.diag_value'): target_status, t('analyze.diag_comment'): target_comment},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_total_rows'), t('analyze.diag_value'): n_total, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_numeric_values'), t('analyze.diag_value'): n_y_valid, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_missing_values'), t('analyze.diag_value'): n_y_missing, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_unique_values'), t('analyze.diag_value'): n_y_unique, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_min'), t('analyze.diag_value'): round(y_min, 6) if np.isfinite(y_min) else "", t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_median'), t('analyze.diag_value'): round(y_median, 6) if np.isfinite(y_median) else "", t('analyze.diag_comment'): t('analyze.comment_robust')},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_max'), t('analyze.diag_value'): round(y_max, 6) if np.isfinite(y_max) else "", t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_range'), t('analyze.diag_value'): round(y_range, 6) if np.isfinite(y_range) else "", t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_iqr'), t('analyze.diag_value'): round(iqr, 6) if np.isfinite(iqr) else "", t('analyze.diag_comment'): t('analyze.comment_iqr')},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_mad'), t('analyze.diag_value'): round(mad, 6) if np.isfinite(mad) else "", t('analyze.diag_comment'): t('analyze.comment_mad')},
        {t('analyze.diag_section'): t('analyze.section_target'), t('analyze.diag_prompt'): t('analyze.diag_iqr_outliers'), t('analyze.diag_value'): n_iqr_outliers, t('analyze.diag_comment'): t('analyze.comment_preliminary')},
        {t('analyze.diag_section'): t('analyze.section_structures'), t('analyze.diag_prompt'): t('analyze.diag_nonempty_smiles'), t('analyze.diag_value'): n_total - n_smiles_empty, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_structures'), t('analyze.diag_prompt'): t('analyze.diag_valid_rdkit'), t('analyze.diag_value'): n_valid_mols, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_structures'), t('analyze.diag_prompt'): t('analyze.diag_invalid_smiles'), t('analyze.diag_value'): n_invalid_smiles, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_structures'), t('analyze.diag_prompt'): t('analyze.diag_unique_inchikey'), t('analyze.diag_value'): n_unique_structures, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_duplicates'), t('analyze.diag_prompt'): t('analyze.diag_duplicates_inchikey'), t('analyze.diag_value'): n_duplicate_structures, t('analyze.diag_comment'): ""},
        {t('analyze.diag_section'): t('analyze.section_duplicates'), t('analyze.diag_prompt'): t('analyze.diag_conflict_duplicates'), t('analyze.diag_value'): n_conflict_duplicates, t('analyze.diag_comment'): t('analyze.comment_conflict')},
    ])

    if class_rows:
        class_summary = (
            pd.DataFrame(class_rows)
            .groupby(t('analyze.class'))
            .size()
            .reset_index(name=t('analyze.count'))
            .sort_values(t('analyze.count'), ascending=False)
            .reset_index(drop=True)
        )
    else:
        class_summary = pd.DataFrame(columns=[t('analyze.class'), t('analyze.count')])

    invalid_smiles_df = pd.DataFrame(invalid_smiles_rows)

    return diagnostics_table, class_summary, invalid_smiles_df, conflict_duplicates
 
def qspr_make_dataset_passport(
    data,
    smiles_col,
    target_col,
    source_filename="",
    suspicious_iqr_multiplier=1.5
):
    """
    Формирует краткий паспорт датасета перед QSPR-моделированием.

    Возвращает:
    - passport_df: компактная таблица показателей;
    - suspicious_values_df: строки с подозрительными значениями свойства;
    - duplicate_structures_df: сводка дубликатов по InChIKey;
    - conflict_duplicates_df: одинаковые InChIKey с разными значениями свойства.
    """
    if data is None or data.empty:
        empty_passport = pd.DataFrame([
            {t('passport.prompt'): t('passport.status'), t('passport.value'): t('passport.dataset_empty'), t('passport.comment'): ""}
        ])
        return empty_passport, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = data.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()

    if smiles_col not in work.columns:
        raise ValueError(t('passport.smiles_column_not_found', col=smiles_col))

    if target_col not in work.columns:
        raise ValueError(t('passport.target_column_not_found', col=target_col))

    y = pd.to_numeric(
        work[target_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    n_rows = len(work)
    n_target_missing = int(y.isna().sum())
    n_target_valid = int(y.notna().sum())

    y_min = float(y.min()) if n_target_valid else np.nan
    y_max = float(y.max()) if n_target_valid else np.nan
    y_median = float(y.median()) if n_target_valid else np.nan

    q1 = float(y.quantile(0.25)) if n_target_valid else np.nan
    q3 = float(y.quantile(0.75)) if n_target_valid else np.nan
    iqr = q3 - q1 if n_target_valid else np.nan

    suspicious_values_df = pd.DataFrame()

    if n_target_valid and np.isfinite(iqr) and iqr > 0:
        low_limit = q1 - suspicious_iqr_multiplier * iqr
        high_limit = q3 + suspicious_iqr_multiplier * iqr

        suspicious_mask = (y < low_limit) | (y > high_limit)

        suspicious_values_df = work.loc[suspicious_mask].copy()
        suspicious_values_df[t('passport.original_row')] = suspicious_values_df.index + 1
        suspicious_values_df[t('passport.property_value')] = y.loc[suspicious_mask]
        suspicious_values_df[t('passport.reason')] = t('passport.iqr_reason', low=low_limit, high=high_limit)
    else:
        low_limit = np.nan
        high_limit = np.nan

    smiles_values = work[smiles_col].astype(str).fillna("").str.strip()

    n_empty_smiles = int((smiles_values == "").sum())

    canonical_smiles = []
    inchikeys = []
    invalid_rows = []

    for idx, smi in smiles_values.items():
        if not smi or smi.lower() in ["nan", "none"]:
            canonical_smiles.append("")
            inchikeys.append("")
            invalid_rows.append({
                t('passport.original_row'): idx + 1,
                smiles_col: smi,
                target_col: y.loc[idx] if idx in y.index else np.nan,
                t('passport.reason'): t('passport.empty_smiles')
            })
            continue

        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None

        if mol is None:
            canonical_smiles.append("")
            inchikeys.append("")
            invalid_rows.append({
                t('passport.original_row'): idx + 1,
                smiles_col: smi,
                target_col: y.loc[idx] if idx in y.index else np.nan,
                t('passport.reason'): t('passport.rdkit_error')
            })
            continue

        try:
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        except Exception:
            canonical_smiles.append("")

        try:
            inchikeys.append(Chem.MolToInchiKey(mol))
        except Exception:
            inchikeys.append("")

    work["_passport_canonical_smiles"] = canonical_smiles
    work["_passport_inchikey"] = inchikeys
    work["_passport_target_numeric"] = y

    invalid_smiles_df = pd.DataFrame(invalid_rows)

    n_invalid_smiles = len(invalid_smiles_df)
    n_valid_smiles = n_rows - n_invalid_smiles

    valid_inchikey_mask = work["_passport_inchikey"].astype(str).str.strip() != ""

    n_unique_structures = int(
        work.loc[valid_inchikey_mask, "_passport_inchikey"].nunique()
    )

    n_duplicate_structures = int(valid_inchikey_mask.sum() - n_unique_structures)

    if valid_inchikey_mask.any():
        duplicate_structures_df = (
            work.loc[valid_inchikey_mask]
            .groupby("_passport_inchikey")["_passport_target_numeric"]
            .agg(
                n_records="size",
                n_unique_target_values=lambda x: x.dropna().nunique(),
                min_target="min",
                max_target="max"
            )
            .reset_index()
            .rename(columns={"_passport_inchikey": "InChIKey"})
        )

        duplicate_structures_df = duplicate_structures_df[
            duplicate_structures_df["n_records"] > 1
        ].copy()

        conflict_duplicates_df = duplicate_structures_df[
            duplicate_structures_df["n_unique_target_values"] > 1
        ].copy()
    else:
        duplicate_structures_df = pd.DataFrame()
        conflict_duplicates_df = pd.DataFrame()

    n_conflict_duplicates = len(conflict_duplicates_df)
    n_suspicious_values = len(suspicious_values_df)

    quality_warnings = []

    if n_invalid_smiles > 0:
        quality_warnings.append(t('passport.warning_invalid_smiles', count=n_invalid_smiles))

    if n_target_missing > 0:
        quality_warnings.append(t('passport.warning_missing_target', count=n_target_missing))

    if n_duplicate_structures > 0:
        quality_warnings.append(t('passport.warning_duplicates', count=n_duplicate_structures))

    if n_conflict_duplicates > 0:
        quality_warnings.append(t('passport.warning_conflict_duplicates', count=n_conflict_duplicates))

    if n_suspicious_values > 0:
        quality_warnings.append(t('passport.warning_suspicious_values', count=n_suspicious_values))

    target_name_lower = str(target_col).lower()

    if any(x in target_name_lower for x in ["id", "index", "номер", "row", "line", "source_line"]):
        quality_warnings.append(t('passport.warning_id_like'))

    if quality_warnings:
        quality_status = t('passport.status_warning')
        quality_comment = "; ".join(quality_warnings)
    else:
        quality_status = t('passport.status_ok')
        quality_comment = t('passport.comment_ok')

    passport_df = pd.DataFrame([
        {t('passport.prompt'): t('passport.file'), t('passport.value'): source_filename or t('passport.not_specified'), t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.rows'), t('passport.value'): n_rows, t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.smiles_col'), t('passport.value'): smiles_col, t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.valid_smiles'), t('passport.value'): n_valid_smiles, t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.invalid_smiles'), t('passport.value'): n_invalid_smiles, t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.empty_smiles_count'), t('passport.value'): n_empty_smiles, t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.unique_structures'), t('passport.value'): n_unique_structures, t('passport.comment'): t('passport.comment_inchikey')},
        {t('passport.prompt'): t('passport.duplicates'), t('passport.value'): n_duplicate_structures, t('passport.comment'): t('passport.comment_duplicates_inchikey')},
        {t('passport.prompt'): t('passport.conflict_duplicates'), t('passport.value'): n_conflict_duplicates, t('passport.comment'): t('passport.comment_conflict')},
        {t('passport.prompt'): t('passport.target_property'), t('passport.value'): target_col, t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.numeric_values'), t('passport.value'): n_target_valid, t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.missing_values'), t('passport.value'): n_target_missing, t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.min'), t('passport.value'): round(y_min, 6) if np.isfinite(y_min) else "", t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.median'), t('passport.value'): round(y_median, 6) if np.isfinite(y_median) else "", t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.max'), t('passport.value'): round(y_max, 6) if np.isfinite(y_max) else "", t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.suspicious_count'), t('passport.value'): n_suspicious_values, t('passport.comment'): t('passport.comment_iqr_rule')},
        {t('passport.prompt'): t('passport.iqr_lower'), t('passport.value'): round(low_limit, 6) if np.isfinite(low_limit) else "", t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.iqr_upper'), t('passport.value'): round(high_limit, 6) if np.isfinite(high_limit) else "", t('passport.comment'): ""},
        {t('passport.prompt'): t('passport.final_status'), t('passport.value'): quality_status, t('passport.comment'): quality_comment},
    ])

    return (
        passport_df,
        suspicious_values_df.reset_index(drop=True),
        duplicate_structures_df.reset_index(drop=True),
        conflict_duplicates_df.reset_index(drop=True)
    )

def qspr_make_dataset_passport_excel(
    passport_df,
    diagnostics_df=None,
    molecule_class_summary=None,
    suspicious_values_df=None,
    duplicate_structures_df=None,
    conflict_duplicates_df=None,
    invalid_smiles_df=None
):
    """
    Excel-отчёт по исходному датасету.
    Первый лист — паспорт датасета.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        passport_df.to_excel(writer, index=False, sheet_name=t('passport_excel.sheet_passport'))

        if isinstance(diagnostics_df, pd.DataFrame) and not diagnostics_df.empty:
            diagnostics_df.to_excel(writer, index=False, sheet_name=t('passport_excel.sheet_diagnostics'))

        if isinstance(molecule_class_summary, pd.DataFrame) and not molecule_class_summary.empty:
            molecule_class_summary.to_excel(writer, index=False, sheet_name=t('passport_excel.sheet_classes'))

        if isinstance(suspicious_values_df, pd.DataFrame) and not suspicious_values_df.empty:
            suspicious_values_df.to_excel(writer, index=False, sheet_name=t('passport_excel.sheet_suspicious'))

        if isinstance(duplicate_structures_df, pd.DataFrame) and not duplicate_structures_df.empty:
            duplicate_structures_df.to_excel(writer, index=False, sheet_name=t('passport_excel.sheet_duplicates'))

        if isinstance(conflict_duplicates_df, pd.DataFrame) and not conflict_duplicates_df.empty:
            conflict_duplicates_df.to_excel(writer, index=False, sheet_name=t('passport_excel.sheet_conflict'))

        if isinstance(invalid_smiles_df, pd.DataFrame) and not invalid_smiles_df.empty:
            invalid_smiles_df.to_excel(writer, index=False, sheet_name=t('passport_excel.sheet_invalid'))

        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"

            for column_cells in worksheet.columns:
                max_len = 0
                col_letter = column_cells[0].column_letter

                for cell in column_cells:
                    try:
                        value_len = len(str(cell.value))
                    except Exception:
                        value_len = 0

                    max_len = max(max_len, value_len)

                worksheet.column_dimensions[col_letter].width = min(max_len + 2, 55)

    output.seek(0)
    return output.getvalue()

def qspr_detect_data_leakage_columns(
    descriptor_cols,
    target_col,
    data=None,
    y=None,
    correlation_threshold=0.995
):
    """
    Ищет потенциальные колонки утечки данных.

    Проверяет:
    - имя дескриптора похоже на целевое свойство;
    - имя содержит target/experimental/source/value/property;
    - числовая колонка почти полностью совпадает с y;
    - числовая колонка почти идеально коррелирует с y.
    """
    descriptor_cols = list(descriptor_cols or [])
    target_col = str(target_col)

    target_lower = target_col.lower().strip()

    def normalize_name(name):
        x = str(name).lower().strip()
        for ch in [" ", "_", "-", ".", "/", "\\", "(", ")", "[", "]", "{", "}", "%"]:
            x = x.replace(ch, "")
        return x

    target_norm = normalize_name(target_col)

    suspicious_tokens = [
        "target",
        "y",
        "label",
        "property",
        "value",
        "source_value",
        "sourcevalue",
        "experimental",
        "experiment",
        "exp",
        "observed",
        "obs",
        "measured",
        "measurement",
        "real",
        "actual",
        "true",
        "truth",
        "response",
        "endpoint",
        "activity",
        "boiling",
        "boilingpoint",
        "boil",
        "bp",
        "melting",
        "meltingpoint",
        "mp",
        "logp_exp",
        "logpexp",
        "solubility_exp",
        "solubilityexp",
    ]

    target_parts = [
        p for p in target_lower
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .split("_")
        if len(p) >= 3
    ]

    rows = []

    for col in descriptor_cols:
        col_str = str(col)
        col_lower = col_str.lower().strip()
        col_norm = normalize_name(col_str)

        reasons = []

        if col_str == target_col:
            reasons.append(t('data_leakage.reason_identical'))

        if col_lower == target_lower or col_norm == target_norm:
            reasons.append(t('data_leakage.reason_name_match'))

        if target_norm and target_norm in col_norm:
            reasons.append(t('data_leakage.reason_contains_target'))

        for part in target_parts:
            if part in col_lower or part in col_norm:
                reasons.append(t('data_leakage.reason_contains_part', part=part))

        for token in suspicious_tokens:
            token_norm = normalize_name(token)

            if token_norm and token_norm in col_norm:
                reasons.append(t('data_leakage.reason_suspicious_token', token=token))

        if data is not None and y is not None and col in data.columns:
            try:
                x = pd.to_numeric(
                    data[col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )

                y_series = pd.Series(y, index=data.index)
                y_series = pd.to_numeric(
                    y_series.astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )

                mask = x.notna() & y_series.notna()

                if mask.sum() >= 5:
                    x_valid = x.loc[mask].astype(float)
                    y_valid = y_series.loc[mask].astype(float)

                    same_mask = np.isclose(
                        x_valid.values,
                        y_valid.values,
                        rtol=1e-10,
                        atol=1e-12
                    )

                    same_ratio = float(np.mean(same_mask))

                    if same_ratio >= 0.98:
                        reasons.append(t('data_leakage.reason_near_identical', ratio=same_ratio))

                    if x_valid.nunique(dropna=True) > 1 and y_valid.nunique(dropna=True) > 1:
                        corr = float(np.corrcoef(x_valid.values, y_valid.values)[0, 1])

                        if np.isfinite(corr) and abs(corr) >= correlation_threshold:
                            reasons.append(t('data_leakage.reason_high_correlation', corr=corr))

            except Exception:
                pass

        if reasons:
            rows.append({
                t('data_leakage.col'): col_str,
                t('data_leakage.reasons'): "; ".join(sorted(set(reasons))),
                t('data_leakage.recommendation'): t('data_leakage.recommendation_text')
            })

    return pd.DataFrame(rows)


def qspr_show_data_leakage_warning(leakage_df, title=t('data_leakage.warning_title')):
    """
    Показывает предупреждение об утечке данных.
    """
    if leakage_df is None or not isinstance(leakage_df, pd.DataFrame) or leakage_df.empty:
        return False

    st.warning(t('data_leakage.warning_text'))

    with st.expander(title, expanded=True):
        st.dataframe(
            leakage_df,
            width="stretch",
            hide_index=True
        )

    return True

def qspr_descriptor_name_match(name, contains=None, startswith=None, exact=None):
    """
    Проверяет, относится ли имя дескриптора к группе по простым правилам.
    """
    name_str = str(name)
    name_low = name_str.lower()

    contains = contains or []
    startswith = startswith or []
    exact = exact or []

    for x in exact:
        if name_str == x:
            return True

    for x in startswith:
        if name_str.startswith(x):
            return True

    for x in contains:
        if str(x).lower() in name_low:
            return True

    return False


def qspr_make_descriptor_groups(names, rules):
    """
    Делит список имён дескрипторов на группы.
    rules:
    {
        "Группа": {
            "contains": [...],
            "startswith": [...],
            "exact": [...]
        }
    }
    """
    names = list(names or [])
    groups = {}
    used = set()

    for group_name, rule in rules.items():
        selected = []

        for name in names:
            if qspr_descriptor_name_match(
                name,
                contains=rule.get("contains", []),
                startswith=rule.get("startswith", []),
                exact=rule.get("exact", [])
            ):
                selected.append(name)
                used.add(name)

        groups[group_name] = selected

    other = [name for name in names if name not in used]

    if other:
        groups[t('descriptor_groups.other')] = other

    return groups


def qspr_descriptor_group_checkboxes(
    title,
    groups,
    key_prefix,
    expanded=False
):
    """
    Рисует раскрывающийся список чекбоксов групп дескрипторов.
    Возвращает список выбранных имён дескрипторов.
    """
    selected_names = []

    total_count = sum(len(v) for v in groups.values())

    with st.expander(f"{title} ({total_count})", expanded=expanded):
        col_a, col_b = st.columns(2)

        with col_a:
            select_all = st.checkbox(
                t('descriptor_groups.select_all'),
                value=True,
                key=f"{key_prefix}_select_all"
            )

        with col_b:
            show_details = st.checkbox(
                t('descriptor_groups.show_details'),
                value=False,
                key=f"{key_prefix}_show_details"
            )

        for group_name, names in groups.items():
            if not names:
                continue

            use_group = st.checkbox(
                f"{group_name} ({len(names)})",
                value=select_all,
                key=f"{key_prefix}_{group_name}"
            )

            if use_group:
                selected_names.extend(names)

            if show_details:
                st.caption(", ".join([str(x) for x in names[:80]]))
                if len(names) > 80:
                    st.caption(t('descriptor_groups.more', count=len(names) - 80))

        st.caption(t('descriptor_groups.selected_count', selected=len(set(selected_names)), total=total_count))

    return sorted(set(selected_names))


def qspr_descriptor_group_selection_ui(mode, desc_lists):
    """
    UI выбора групп RDKit / Mordred / PaDEL под текущий режим расчёта.
    Возвращает allowed_rdkit_names, allowed_mordred_names, allowed_padel_names.

    Логика режимов:
    - RDKit: только RDKit-группы;
    - Mordred: RDKit + Mordred;
    - Умный: RDKit + Mordred + уникальные PaDEL;
    - Максимальная точность: RDKit + Mordred + все доступные PaDEL.
    """
    desc_lists = desc_lists or {}

    rdkit_all = desc_lists.get("rdkit_all", []) or []
    mordred_unique = desc_lists.get("mordred_unique", []) or []
    padel_unique = desc_lists.get("padel_unique", []) or []

    padel_all = (
        desc_lists.get("padel_all", [])
        or desc_lists.get("padel_descriptors_all", [])
        or desc_lists.get("padel_all_descriptors", [])
        or []
    )

    padel_fingerprints = desc_lists.get("padel_fingerprints", []) or []
    padel_1d2d = desc_lists.get("padel_1d2d", []) or []

    if mode == "max_accuracy":
        if padel_all:
            padel_source_names = padel_all
            padel_source_label = t('descriptor_groups.padel_all_label',
                total=len(padel_all),
                fp=len(padel_fingerprints),
                desc=len(padel_1d2d)
            )
        else:
            padel_source_names = padel_unique
            padel_source_label = t('descriptor_groups.padel_unique_only_label',
                total=len(padel_unique)
            )
    else:
        padel_source_names = padel_unique
        padel_source_label = t('descriptor_groups.padel_unique_rdkit_mordred_label',
            total=len(padel_unique)
        )

    rdkit_rules = {
        t('descriptor_groups.rdkit_mass'): {
            "exact": [
                "MolWt",
                "HeavyAtomMolWt",
                "ExactMolWt",
                "NumValenceElectrons",
                "NumRadicalElectrons",
                "HeavyAtomCount",
            ],
            "contains": ["Wt", "AtomCount"]
        },
        t('descriptor_groups.rdkit_lipinski'): {
            "exact": [
                "TPSA",
                "NumHAcceptors",
                "NumHDonors",
                "NumRotatableBonds",
                "NHOHCount",
                "NOCount",
                "RingCount",
                "FractionCSP3",
            ],
            "contains": ["Lipinski"]
        },
        t('descriptor_groups.rdkit_logp'): {
            "exact": ["MolLogP", "MolMR"],
            "contains": ["Crippen"]
        },
        t('descriptor_groups.rdkit_estate'): {
            "contains": ["EState"]
        },
        t('descriptor_groups.rdkit_bcut'): {
            "startswith": ["BCUT2D"]
        },
        t('descriptor_groups.rdkit_autocorr'): {
            "startswith": ["AUTOCORR2D"]
        },
        t('descriptor_groups.rdkit_chi_kappa'): {
            "startswith": ["Chi", "Kappa"]
        },
        t('descriptor_groups.rdkit_peoe_vsa'): {
            "startswith": ["PEOE_VSA"]
        },
        t('descriptor_groups.rdkit_smr_vsa'): {
            "startswith": ["SMR_VSA"]
        },
        t('descriptor_groups.rdkit_slogp_vsa'): {
            "startswith": ["SlogP_VSA"]
        },
        t('descriptor_groups.rdkit_estate_vsa'): {
            "startswith": ["EState_VSA", "VSA_EState"]
        },
        t('descriptor_groups.rdkit_fragments'): {
            "startswith": ["fr_"]
        },
        t('descriptor_groups.rdkit_topological'): {
            "exact": [
                "BertzCT",
                "BalabanJ",
                "Ipc",
                "HallKierAlpha",
                "LabuteASA",
            ],
            "contains": ["qed"]
        },
    }

    mordred_rules = {
        t('descriptor_groups.mordred_constitutional'): {
            "contains": ["Atom", "Count", "Weight", "Constitutional"]
        },
        t('descriptor_groups.mordred_topological'): {
            "contains": [
                "Topo",
                "Wiener",
                "Zagreb",
                "Kappa",
                "Chi",
                "Balaban",
                "Walk",
                "Path",
                "Distance",
                "Matrix",
            ]
        },
        t('descriptor_groups.mordred_ring_aromatic'): {
            "contains": ["Ring", "Aromatic"]
        },
        t('descriptor_groups.mordred_charge_estate'): {
            "contains": ["Charge", "EState", "Polar", "TopoPSA", "HydrogenBond"]
        },
        t('descriptor_groups.mordred_logp'): {
            "contains": ["LogP", "SLogP", "LogS"]
        },
        t('descriptor_groups.mordred_bcut_autocorr'): {
            "contains": ["BCUT", "Autocorrelation"]
        },
        t('descriptor_groups.mordred_3d_geometry'): {
            "contains": [
                "MoRSE",
                "Moment",
                "Inertia",
                "PBF",
                "Geometrical",
                "Gravitational",
            ]
        },
        t('descriptor_groups.mordred_information'): {
            "contains": ["Information", "Complexity"]
        },
        t('descriptor_groups.mordred_volume_surface'): {
            "contains": ["Volume", "Vdw", "Surface", "Area"]
        },
    }

    padel_rules = {
        t('descriptor_groups.padel_fingerprints'): {
            "contains": ["FP", "Fingerprint", "PubchemFP", "SubFP", "KRF", "MACCS"]
        },
        t('descriptor_groups.padel_topological'): {
            "contains": ["Topo", "ETA", "ATSC", "AATS", "MATS", "GATS"]
        },
        t('descriptor_groups.padel_charge_estate'): {
            "contains": ["EState", "Estate", "Charge"]
        },
        t('descriptor_groups.padel_vsa_surface'): {
            "contains": ["VSA", "Surface", "TPSA"]
        },
        t('descriptor_groups.padel_bcut_burden'): {
            "contains": ["BCUT", "Burden"]
        },
        t('descriptor_groups.padel_acid_base'): {
            "contains": ["nAcid", "nBase", "HBD", "HBA", "RotB"]
        },
        t('descriptor_groups.padel_logp'): {
            "contains": ["LogP", "ALogP", "XLogP", "MLogP"]
        },
    }

    rdkit_groups = qspr_make_descriptor_groups(rdkit_all, rdkit_rules)
    mordred_groups = qspr_make_descriptor_groups(mordred_unique, mordred_rules)
    padel_groups = qspr_make_descriptor_groups(padel_source_names, padel_rules)

    use_rdkit = mode in {
        "rdkit_fast",
        "mordred",
        "mordred_padel_unique",
        "max_accuracy",
    }

    use_mordred = mode in {
        "mordred",
        "mordred_padel_unique",
        "max_accuracy",
    }

    use_padel = mode in {
        "mordred_padel_unique",
        "max_accuracy",
    }

    st.markdown(t('descriptor_groups.molecular_groups_title'))

    allowed_rdkit_names = None
    allowed_mordred_names = None
    allowed_padel_names = None

    if use_rdkit:
        allowed_rdkit_names = qspr_descriptor_group_checkboxes(
            t('descriptor_groups.rdkit_checkbox_title'),
            rdkit_groups,
            key_prefix=f"rdkit_descriptor_groups_{mode}",
            expanded=False
        )

    if use_mordred:
        allowed_mordred_names = qspr_descriptor_group_checkboxes(
            t('descriptor_groups.mordred_checkbox_title'),
            mordred_groups,
            key_prefix=f"mordred_descriptor_groups_{mode}",
            expanded=False
        )

    if use_padel:
        if mode == "max_accuracy" and not padel_all:
            st.warning(t('descriptor_groups.warning_max_accuracy_no_padel_all'))

        if mode == "mordred_padel_unique":
            st.info(t('descriptor_groups.info_smart_mode', count=len(padel_unique)))

        if mode == "max_accuracy":
            st.info(t('descriptor_groups.info_max_accuracy_mode',
                total=len(padel_all),
                unique=len(padel_unique)
            ))

        allowed_padel_names = qspr_descriptor_group_checkboxes(
            padel_source_label,
            padel_groups,
            key_prefix=f"padel_descriptor_groups_{mode}",
            expanded=False
        )

    total_selected = 0

    for selected in [
        allowed_rdkit_names,
        allowed_mordred_names,
        allowed_padel_names,
    ]:
        if selected is not None:
            total_selected += len(selected)

    if total_selected == 0:
        st.warning(t('descriptor_groups.warning_no_groups_selected'))
    else:
        st.caption(t('descriptor_groups.caption_total_selected', count=total_selected))

    return allowed_rdkit_names, allowed_mordred_names, allowed_padel_names

def spectra_resolve_saved_files_from_message(result_row):
    """
    Пытается восстановить raw_file / processed_file по сообщению вида:
    'Спектр скачан и сохранён как IR_NIST_gas_XXXX_001'
    или
    'MoNA: масс-спектр скачан и сохранён как Mass_MASSBANK_gas_XXXX_001'

    Это нужно, если spectra_search_one_compound сообщает о сохранении,
    но не возвращает raw_file / processed_file.
    """
    import re
    import glob

    row = dict(result_row or {})

    raw_file = str(row.get("raw_file", "")).strip()
    processed_file = str(row.get("processed_file", "")).strip()

    if raw_file or processed_file:
        return row

    message = str(row.get("message", "")).strip()
    spectrum_type = str(row.get("spectrum_type", "")).strip()

    if not message:
        return row

    match = re.search(r"сохран[ёе]н(?:\s+как)?\s+([A-Za-z0-9_\-]+)", message)

    if not match:
        return row

    base_name = match.group(1).strip()

    if not base_name:
        return row

    candidate_dirs = []

    if spectrum_type == "IR":
        candidate_dirs.extend([
            SPECTRA_IR_RAW_DIR,
            SPECTRA_IR_PROCESSED_DIR,
        ])

    elif spectrum_type == "Mass":
        candidate_dirs.extend([
            SPECTRA_MASS_RAW_DIR,
            SPECTRA_MASS_PROCESSED_DIR,
        ])

    else:
        candidate_dirs.extend([
            SPECTRA_IR_RAW_DIR,
            SPECTRA_IR_PROCESSED_DIR,
            SPECTRA_MASS_RAW_DIR,
            SPECTRA_MASS_PROCESSED_DIR,
        ])

    found_raw = ""
    found_processed = ""

    for folder in candidate_dirs:
        if not os.path.isdir(folder):
            continue

        patterns = [
            os.path.join(folder, base_name + ".*"),
            os.path.join(folder, base_name + "_processed.*"),
            os.path.join(folder, "*" + base_name + "*"),
        ]

        for pattern in patterns:
            for path in glob.glob(pattern):
                norm_path = os.path.normpath(path)

                lower_path = norm_path.lower()

                if "raw_jdx" in lower_path and not found_raw:
                    found_raw = norm_path

                if "processed" in lower_path and not found_processed:
                    found_processed = norm_path

    if found_raw:
        row["raw_file"] = found_raw

    if found_processed:
        row["processed_file"] = found_processed

    return row

def spectra_build_search_cache_lookup():
    """
    Быстро строит словари журнала уже проверенных спектров.

    Логика:
    если InChIKey + тип спектра уже встречались в журнале,
    значит это вещество уже проверялось и не должно повторно уходить
    во внешний поиск, даже если спектр тогда не был найден.
    """
    try:
        cache_df = spectra_load_search_cache()
    except Exception:
        cache_df = pd.DataFrame()

    lookup_by_inchikey = {}
    lookup_by_smiles = {}

    if cache_df is None or cache_df.empty:
        return lookup_by_inchikey, lookup_by_smiles

    work = cache_df.copy()

    required_cols = [
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "final_status",
        "spectrum_status",
        "selected_source",
        "candidate_count",
        "spectrum_id",
        "raw_file",
        "processed_file",
        "message",
        "date_checked",
    ]

    for col in required_cols:
        if col not in work.columns:
            work[col] = ""

    work["inchikey"] = work["inchikey"].astype(str).str.strip()
    work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()
    work["_spectrum_type_norm"] = work["spectrum_type"].astype(str).apply(
        spectra_normalize_spectrum_type
    )

    # Более свежие записи должны перезаписывать старые.
    for _, row in work.iterrows():
        row_dict = row.to_dict()

        cache_type = str(row_dict.get("_spectrum_type_norm", "")).strip()
        cache_inchikey = str(row_dict.get("inchikey", "")).strip()
        cache_smiles = str(row_dict.get("canonical_smiles", "")).strip()

        if not row_dict.get("final_status", ""):
            row_dict["final_status"] = row_dict.get("spectrum_status", "")

        if cache_inchikey and cache_type:
            lookup_by_inchikey[(cache_inchikey, cache_type)] = row_dict

        if cache_smiles and cache_type:
            lookup_by_smiles[(cache_smiles, cache_type)] = row_dict

    return lookup_by_inchikey, lookup_by_smiles

def spectra_build_existing_bank_lookup():
    """
    Строит быстрые lookup-наборы по spectra_index.csv.

    Возвращает:
    - bank_by_inchikey: {(inchikey, spectrum_type): record}
    - bank_by_smiles: {(canonical_smiles, spectrum_type): record}

    Это отвечает на вопрос:
    есть ли уже спектр в локальной spectra_bank?
    """
    bank_by_inchikey = {}
    bank_by_smiles = {}

    try:
        index_df = spectra_load_index()
    except Exception:
        index_df = pd.DataFrame()

    if index_df is None or index_df.empty:
        return bank_by_inchikey, bank_by_smiles

    work = index_df.copy()

    required_cols = [
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "active",
        "source",
        "source_database",
        "spectrum_id",
        "raw_file",
        "processed_file",
        "status",
    ]

    for col in required_cols:
        if col not in work.columns:
            work[col] = ""

    work["inchikey"] = work["inchikey"].astype(str).str.strip()
    work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()
    work["_spectrum_type_norm"] = work["spectrum_type"].astype(str).apply(
        spectra_normalize_spectrum_type
    )

    active_values = ["true", "1", "yes", "y", "да", "active", ""]

    work["_active_norm"] = (
        work["active"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(active_values)
    )

    work = work[work["_active_norm"]].copy()

    for _, row in work.iterrows():
        row_dict = row.to_dict()

        spectrum_type = str(row_dict.get("_spectrum_type_norm", "")).strip()
        inchikey = str(row_dict.get("inchikey", "")).strip()
        canonical_smiles = str(row_dict.get("canonical_smiles", "")).strip()

        if inchikey and spectrum_type:
            bank_by_inchikey[(inchikey, spectrum_type)] = row_dict

        if canonical_smiles and spectrum_type:
            bank_by_smiles[(canonical_smiles, spectrum_type)] = row_dict

    return bank_by_inchikey, bank_by_smiles


def spectra_build_search_cache_lookup():
    """
    Строит быстрые lookup-словари журнала проверенных спектров.

    Важная логика:
    если InChIKey + тип спектра уже есть в журнале,
    значит это вещество уже проверялось.

    Не важно, был ли спектр найден или нет.
    Не важно, какой набор источников был выбран.
    Повторно гонять внешний поиск не нужно.
    """
    cache_by_inchikey = {}
    cache_by_smiles = {}

    try:
        cache_df = spectra_load_search_cache()
    except Exception:
        cache_df = pd.DataFrame()

    if cache_df is None or cache_df.empty:
        return cache_by_inchikey, cache_by_smiles

    work = cache_df.copy()

    required_cols = [
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "final_status",
        "spectrum_status",
        "selected_source",
        "candidate_count",
        "spectrum_id",
        "raw_file",
        "processed_file",
        "message",
        "date_checked",
    ]

    for col in required_cols:
        if col not in work.columns:
            work[col] = ""

    work["inchikey"] = work["inchikey"].astype(str).str.strip()
    work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()
    work["_spectrum_type_norm"] = work["spectrum_type"].astype(str).apply(
        spectra_normalize_spectrum_type
    )

    for _, row in work.iterrows():
        row_dict = row.to_dict()

        spectrum_type = str(row_dict.get("_spectrum_type_norm", "")).strip()
        inchikey = str(row_dict.get("inchikey", "")).strip()
        canonical_smiles = str(row_dict.get("canonical_smiles", "")).strip()

        if not row_dict.get("final_status", ""):
            row_dict["final_status"] = row_dict.get("spectrum_status", "")

        if inchikey and spectrum_type:
            cache_by_inchikey[(inchikey, spectrum_type)] = row_dict

        if canonical_smiles and spectrum_type:
            cache_by_smiles[(canonical_smiles, spectrum_type)] = row_dict

    return cache_by_inchikey, cache_by_smiles


def spectra_make_skipped_result_row(
    compound,
    spectrum_type,
    status,
    message,
    selected_source="",
    candidate_count=0,
    spectrum_id="",
    raw_file="",
    processed_file="",
):
    """
    Формирует строку результата для пропущенных задач:
    - уже есть в spectra_bank;
    - уже проверялось в журнале;
    - некорректная структура.
    """
    return {
        "source_line_number": compound.get(
            "source_line_number",
            compound.get("row_index", "")
        ),
        "compound_id": compound.get("compound_id", ""),
        "name": compound.get("name", ""),
        "cas": compound.get("cas", ""),
        "input_smiles": compound.get("input_smiles", ""),
        "canonical_smiles": compound.get("canonical_smiles", ""),
        "inchikey": compound.get("inchikey", ""),
        "structure_status": compound.get("structure_status", ""),
        "spectrum_type": spectra_normalize_spectrum_type(spectrum_type),
        "spectrum_status": status,
        "selected_source": selected_source,
        "candidate_count": candidate_count,
        "spectrum_id": spectrum_id,
        "raw_file": raw_file,
        "processed_file": processed_file,
        "message": message,
    }

LOG_STAGES = {
    "SYSTEM",
    "DATA_UPLOAD",
    "DATA_VALIDATION",
    "SAOD",
    "DESCRIPTORS",
    "FEATURE_SELECTION",
    "MODEL_TRAINING",
    "VALIDATION",
    "MODEL_COMPARISON",
    "Y_RANDOMIZATION",
    "BOOTSTRAP",
    "APPLICABILITY_DOMAIN",
    "RESIDUAL_DIAGNOSTICS",
    "DESCRIPTOR_IMPORTANCE",
    "PREDICTION",
    "UNCERTAINTY",
    "SPECTRAL_SEARCH",
    "REPORT",
    "EXPORT",
}

LOG_STAGE_ALIASES = {
    "system": "SYSTEM",
    "data_upload": "DATA_UPLOAD",
    "data_validation": "DATA_VALIDATION",
    "saod": "SAOD",
    "descriptor_calculation": "DESCRIPTORS",
    "descriptor_filtering": "FEATURE_SELECTION",
    "model_training": "MODEL_TRAINING",
    "model_validation": "VALIDATION",
    "model_comparison": "MODEL_COMPARISON",
    "error_analysis": "RESIDUAL_DIAGNOSTICS",
    "prediction": "PREDICTION",
    "applicability_domain": "APPLICABILITY_DOMAIN",
    "uncertainty": "UNCERTAINTY",
    "spectral_search": "SPECTRAL_SEARCH",
    "export": "EXPORT",
    "report": "REPORT",
}

LOG_LEVELS = {"INFO", "WARNING", "ERROR", "DEBUG"}


def _log_normalize_level(level):
    normalized = str(level or "INFO").upper().strip()
    return normalized if normalized in LOG_LEVELS else "INFO"


def _log_normalize_stage(stage):
    raw = str(stage or "SYSTEM").strip()
    alias = LOG_STAGE_ALIASES.get(raw.lower())
    if alias:
        return alias
    normalized = raw.upper()
    return normalized if normalized in LOG_STAGES else "SYSTEM"


def _log_compact_mapping(mapping):
    if not mapping:
        return ""
    parts = []
    for key, value in dict(mapping).items():
        if value is None or value == "":
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _log_format_event(event, include_details=True):
    event_details = dict(event.get("details", {}) or {})
    for legacy_key in ("counts", "metrics", "context"):
        if event.get(legacy_key):
            event_details.update(dict(event.get(legacy_key, {})))
    detail_text = _log_compact_mapping(event_details) if include_details else ""
    suffix = f" ({detail_text})" if detail_text else ""
    event_time = event.get("time") or str(event.get("timestamp", ""))[-8:] or "--:--:--"
    return (
        f"[{event_time}] {_log_normalize_level(event.get('level'))} | "
        f"{_log_normalize_stage(event.get('stage'))} | {event.get('message', '')}{suffix}"
    )


def add_event_log(stage, message, level="info", details=None, event=None):
    """Adds one structured research-journal event."""
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "log_events" not in st.session_state:
        st.session_state.log_events = []

    level = _log_normalize_level(level)
    stage = _log_normalize_stage(stage)
    now = datetime.now()

    entry = {
        "timestamp": now.isoformat(timespec="seconds"),
        "time": now.strftime("%H:%M:%S"),
        "stage": stage,
        "event": str(event or "event"),
        "level": level,
        "message": str(message),
        "details": dict(details or {}),
    }


def render_spectra_search_results_if_available():
    results = st.session_state.get("spectra_search_results")
    if not isinstance(results, pd.DataFrame):
        return False

    search_results_df = results.copy()
    if search_results_df.empty:
        return False

    st.subheader(t('spectra.results_subheader'))

    status = st.session_state.get("spectra_search_status", "")
    if status == "stopped_by_user":
        st.info("Поиск спектров остановлен пользователем. Показаны результаты уже выполненной работы.")

    if is_admin() and st.button(t('spectra.clear_results_button'), key="clear_spectra_search_results_global"):
        del st.session_state.spectra_search_results
        st.session_state.pop("spectra_search_status", None)
        st.session_state.pop("spectra_search_total_tasks", None)
        st.rerun()

    if "spectrum_status" not in search_results_df.columns:
        st.warning(t('spectra.old_or_invalid_result'))
        st.dataframe(search_results_df, width="stretch")
        return True

    for col in [
        "spectrum_status",
        "spectrum_type",
        "inchikey",
        "canonical_smiles",
        "source_line_number",
        "name",
        "input_smiles",
        "_from_real_search",
    ]:
        if col not in search_results_df.columns:
            search_results_df[col] = False if col == "_from_real_search" else ""

    status_norm = search_results_df["spectrum_status"].astype(str).str.strip().str.lower()
    type_norm = search_results_df["spectrum_type"].astype(str).str.strip().apply(spectra_normalize_spectrum_type)
    real_search_mask = (
        search_results_df["_from_real_search"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y"])
    )
    summary_scope_mask = real_search_mask if bool(real_search_mask.any()) else pd.Series(True, index=search_results_df.index)
    found_mask = status_norm.isin(["found_downloaded", "already_in_bank"]) & summary_scope_mask
    found_ir_count = int((found_mask & (type_norm == "IR")).sum())
    found_mass_count = int((found_mask & (type_norm == "Mass")).sum())
    not_found_count = int((status_norm == "not_found_in_all_sources").sum())
    api_error_count = int(status_norm.isin(["api_error", "search_error", "download_error"]).sum())
    parse_error_count = int(status_norm.isin(["parse_error", "no_numeric_spectrum"]).sum())
    processed_count = int(len(search_results_df))
    total_expected = st.session_state.get("spectra_search_total_tasks")
    if total_expected is None:
        total_expected = processed_count

    last_row = search_results_df.iloc[-1]
    last_line = last_row.get("source_line_number", "")
    last_compound = (
        str(last_row.get("name", "")).strip()
        or str(last_row.get("canonical_smiles", "")).strip()
        or str(last_row.get("input_smiles", "")).strip()
        or "—"
    )

    summary_df = pd.DataFrame({
        "Показатель": [
            "Обработано / всего",
            "Найдено",
            "Не найдено",
            "Ошибки API",
            "Ошибки парсинга",
            "Последняя обработанная строка",
            "Последнее вещество",
        ],
        "Значение": [
            f"{processed_count} / {total_expected}",
            int(found_mask.sum()),
            not_found_count,
            api_error_count,
            parse_error_count,
            last_line,
            last_compound,
        ],
    })
    summary_df = pd.concat([
        summary_df.iloc[:2],
        pd.DataFrame({
            summary_df.columns[0]: ["Найдено IR", "Найдено Mass"],
            summary_df.columns[1]: [found_ir_count, found_mass_count],
        }),
        summary_df.iloc[2:],
    ], ignore_index=True)
    st.dataframe(summary_df, width="stretch", hide_index=True)

    search_results_df["_spectrum_status_norm"] = status_norm
    search_results_df["_spectrum_type_norm"] = (
        search_results_df["spectrum_type"]
        .astype(str)
        .str.strip()
        .apply(spectra_normalize_spectrum_type)
    )

    status_summary = (
        search_results_df
        .groupby("spectrum_status")
        .size()
        .reset_index(name=t('spectra.status_count'))
        .rename(columns={"spectrum_status": t('spectra.status_label')})
        .sort_values(t('spectra.status_count'), ascending=False)
    )
    st.dataframe(status_summary, width="stretch", hide_index=True)

    with st.expander(t('spectra.show_full_table_expander'), expanded=True):
        st.dataframe(
            search_results_df.drop(
                columns=["_from_real_search", "_from_real_search_bool"],
                errors="ignore",
            ),
            width="stretch",
            hide_index=True,
        )

    csv_search = search_results_df.drop(
        columns=[
            "_spectrum_status_norm",
            "_spectrum_type_norm",
            "_from_real_search",
            "_from_real_search_bool",
        ],
        errors="ignore",
    ).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Скачать текущие результаты CSV",
        csv_search,
        "spectra_search_results_current.csv",
        "text/csv",
        key="download_current_spectra_search_results",
    )
    return True

    st.session_state.log_events.append(entry)
    st.session_state.logs.append(_log_format_event(entry))

    if len(st.session_state.log_events) > 500:
        st.session_state.log_events = st.session_state.log_events[-500:]
    if len(st.session_state.logs) > 500:
        st.session_state.logs = st.session_state.logs[-500:]


def add_log(
    message,
    level="INFO",
    *,
    stage="SYSTEM",
    event="event",
    details=None,
    counts=None,
    metrics=None,
    context=None,
    warnings=None,
    debug=None,
):
    """Compatibility wrapper for old add_log calls."""
    merged_details = {}
    for value in (details, counts, metrics, context):
        if value:
            merged_details.update(dict(value))
    if warnings:
        merged_details["warnings"] = "; ".join(map(str, warnings))
    if debug:
        merged_details.update({f"debug_{k}": v for k, v in dict(debug).items()})
    add_event_log(stage=stage, message=message, level=level, details=merged_details, event=event)


def log_streamlit_message(stage, message, level="warning", details=None, event=None):
    add_event_log(stage=stage, message=message, level=level, details=details, event=event)


VALIDATION_LABELS_RU = {
    "excellent": "отличное",
    "good": "хорошее",
    "acceptable": "приемлемое",
    "weak": "слабое",
    "bad": "плохое",
    "unstable": "нестабильное",
    "undefined": "не определено",
}


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _fmt_metric(value, digits=2):
    value = _safe_float(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def _fmt_mean_std(mean_value, std_value, digits=2):
    return f"{_fmt_metric(mean_value, digits)}±{_fmt_metric(std_value, digits)}"


def interpret_validation_quality(
    r2=None,
    q2=None,
    rmse=None,
    mae=None,
    y_std=None,
    metric_std=None,
    method=None,
):
    score = _safe_float(q2 if q2 is not None else r2)
    rmse_value = _safe_float(rmse)
    y_std_value = _safe_float(y_std)
    metric_std_value = _safe_float(metric_std)
    reasons = []

    if score is None:
        label = "undefined"
        level = "warning"
        reasons.append("метрика R²/Q² не рассчитана")
    elif score >= 0.85 and (
        rmse_value is None or y_std_value in (None, 0) or rmse_value / y_std_value < 0.30
    ):
        label = "excellent"
        level = "info"
        reasons.append("R²/Q² >= 0.85")
    elif score >= 0.70:
        label = "good"
        level = "info"
        reasons.append("R²/Q² >= 0.70")
    elif score >= 0.50:
        label = "acceptable"
        level = "info"
        reasons.append("R²/Q² >= 0.50")
    elif score >= 0.30:
        label = "weak"
        level = "warning"
        reasons.append("R²/Q² < 0.50")
    else:
        label = "bad"
        level = "error"
        reasons.append("R²/Q² < 0.30")

    if rmse_value is not None and y_std_value not in (None, 0):
        rmse_ratio = rmse_value / y_std_value
        if rmse_ratio > 0.50:
            level = "warning" if level == "info" else level
            reasons.append("RMSE/std(y) > 0.50")
        if score is not None and score >= 0.85 and rmse_ratio >= 0.30 and label == "excellent":
            label = "good"
            reasons.append("RMSE/std(y) не подтверждает отличный результат")

    if metric_std_value is not None and metric_std_value > 0.20:
        label = "unstable"
        level = "warning"
        reasons.append("разброс метрики высокий")

    if method and str(method).lower().startswith("bootstrap") and metric_std_value is not None and metric_std_value > 0.15:
        label = "unstable"
        level = "warning"
        reasons.append("bootstrap показывает нестабильность")

    return {"label": label, "level": level, "reasons": reasons}


def validation_quality_text(quality):
    label = quality.get("label", "undefined") if isinstance(quality, dict) else "undefined"
    return VALIDATION_LABELS_RU.get(label, "не определено")


def log_validation_result(method, model_name, result, y_values=None, details=None):
    details = dict(details or {})
    y_array = np.asarray(y_values, dtype=float) if y_values is not None else np.asarray([], dtype=float)
    y_std = float(np.nanstd(y_array, ddof=1)) if len(y_array) > 1 else None

    if method == "holdout":
        metrics = result.get("metrics_test", {})
        quality = interpret_validation_quality(
            r2=metrics.get("R2"),
            rmse=metrics.get("RMSE"),
            mae=metrics.get("MAE"),
            y_std=y_std,
            method="Hold-out",
        )
        message = (
            f"Hold-out {model_name}: train={len(result.get('y_train', []))}, "
            f"test={len(result.get('y_test', []))}, R²test={_fmt_metric(metrics.get('R2'))}, "
            f"RMSEtest={_fmt_metric(metrics.get('RMSE'))}, MAEtest={_fmt_metric(metrics.get('MAE'))}; "
            f"качество {validation_quality_text(quality)}."
        )
        event = "holdout_completed"
    elif method == "kfold":
        metrics = result.get("metrics", {})
        errors = np.asarray(result.get("y", []), dtype=float) - np.asarray(result.get("y_pred_cv", []), dtype=float)
        error_std = float(np.nanstd(errors, ddof=1)) if len(errors) > 1 else None
        quality = interpret_validation_quality(
            q2=metrics.get("R2"),
            rmse=metrics.get("RMSE"),
            mae=metrics.get("MAE"),
            y_std=y_std,
            metric_std=details.get("metric_std"),
            method="K-Fold",
        )
        message = (
            f"{result.get('k', details.get('folds', 'K'))}-fold CV {model_name}: "
            f"Q²={_fmt_metric(metrics.get('R2'))}, RMSE={_fmt_metric(metrics.get('RMSE'))}, "
            f"MAE={_fmt_metric(metrics.get('MAE'))}, n={len(result.get('y', []))}; "
            f"качество {validation_quality_text(quality)}."
        )
        details["residual_std"] = error_std
        event = "kfold_completed"
    elif method == "loo":
        metrics = result.get("metrics", {})
        quality = interpret_validation_quality(
            q2=metrics.get("R2"),
            rmse=metrics.get("RMSE"),
            mae=metrics.get("MAE"),
            y_std=y_std,
            method="LOO",
        )
        message = (
            f"LOO {model_name}: Q²LOO={_fmt_metric(metrics.get('R2'))}, "
            f"RMSE={_fmt_metric(metrics.get('RMSE'))}, MAE={_fmt_metric(metrics.get('MAE'))}, "
            f"n={len(result.get('y', []))}; качество {validation_quality_text(quality)}."
        )
        event = "loo_completed"
    else:
        quality = {"level": "info", "label": "undefined", "reasons": []}
        message = f"{method} {model_name}: результат валидации сохранён; метрики недоступны для этого метода."
        event = "validation_completed"

    details.update({"model": model_name, "method": method, "quality": validation_quality_text(quality)})
    if quality.get("reasons"):
        details["quality_reasons"] = "; ".join(quality["reasons"])
    add_event_log("VALIDATION", message, level=quality.get("level", "info"), details=details, event=event)


def log_repeated_holdout_result(model_name, result):
    quality = interpret_validation_quality(
        r2=result.get("test_r2_mean"),
        rmse=result.get("test_rmse_mean"),
        metric_std=result.get("test_r2_std"),
        method="Repeated Hold-out",
    )
    message = (
        f"Repeated Hold-out {model_name}: {result.get('n_repeats')} повторов, "
        f"test={float(result.get('test_size', 0)):.0%}, "
        f"R²={_fmt_mean_std(result.get('test_r2_mean'), result.get('test_r2_std'))}, "
        f"RMSE={_fmt_mean_std(result.get('test_rmse_mean'), result.get('test_rmse_std'))}; "
        f"качество {validation_quality_text(quality)}."
    )
    add_event_log(
        "VALIDATION",
        message,
        level=quality["level"],
        event="repeated_holdout_completed",
        details={
            "model": model_name,
            "successful_repeats": result.get("n_ok"),
            "failed_repeats": result.get("n_failed"),
            "conclusion": result.get("conclusion"),
        },
    )


def log_bootstrap_result(model_name, result):
    summary = result.get("summary", {})
    quality = interpret_validation_quality(
        r2=summary.get("r2_oob_mean"),
        rmse=summary.get("rmse_oob_mean"),
        metric_std=summary.get("r2_oob_std"),
        method="Bootstrap",
    )
    iterations = result.get("iterations_table", pd.DataFrame())
    rmse_values = pd.to_numeric(iterations.get("RMSE OOB", pd.Series(dtype=float)), errors="coerce").dropna()
    p95_rmse = float(rmse_values.quantile(0.95)) if not rmse_values.empty else None
    median_rmse = float(rmse_values.median()) if not rmse_values.empty else None
    if p95_rmse is not None and median_rmse not in (None, 0) and p95_rmse > 2.0 * median_rmse:
        quality = {"label": "unstable", "level": "warning", "reasons": ["P95 RMSE сильно выше медианы"]}

    message = (
        f"Bootstrap {model_name}: {summary.get('n_iterations_requested')} итераций, "
        f"успешных {summary.get('n_iterations_successful')}, "
        f"OOB R²={_fmt_mean_std(summary.get('r2_oob_mean'), summary.get('r2_oob_std'))}, "
        f"RMSE={_fmt_mean_std(summary.get('rmse_oob_mean'), summary.get('rmse_oob_std'))}, "
        f"P95 RMSE={_fmt_metric(p95_rmse)}; результат {validation_quality_text(quality)}."
    )
    add_event_log(
        "BOOTSTRAP",
        message,
        level=quality["level"],
        event="bootstrap_completed",
        details={
            "model": model_name,
            "failed_iterations": summary.get("n_iterations_skipped_or_failed"),
            "mae_oob": _fmt_mean_std(summary.get("mae_oob_mean"), summary.get("mae_oob_std")),
            "quality_reasons": "; ".join(quality.get("reasons", [])),
        },
    )


def log_y_randomization_result(model_name, result):
    summary = result.get("summary", {})
    original_q2 = _safe_float(summary.get("original_q2"))
    mean_perm = _safe_float(summary.get("mean_q2_permuted"))
    std_perm = _safe_float(summary.get("std_q2_permuted"))
    gap = original_q2 - mean_perm if original_q2 is not None and mean_perm is not None else None
    risky = gap is None or gap < 0.10 or (std_perm is not None and gap < std_perm)
    level = "warning" if risky else "info"
    status = "возможна случайная корреляция" if risky else "риск случайной корреляции низкий"
    message = (
        f"Y-randomization {model_name}: {summary.get('n_permutations')} перестановок, "
        f"Q² исходной модели={_fmt_metric(original_q2)}, "
        f"Q² случайных моделей={_fmt_mean_std(mean_perm, std_perm)}; {status}."
    )
    add_event_log(
        "Y_RANDOMIZATION",
        message,
        level=level,
        event="y_randomization_completed",
        details={
            "model": model_name,
            "validation_method": summary.get("validation_method"),
            "q2_gap": _fmt_metric(gap),
            "p_value": _fmt_metric(summary.get("p_value"), digits=3),
            "conclusion": summary.get("conclusion"),
        },
    )


def build_log_txt():
    events = st.session_state.get("log_events", [])
    lines = [
        "Augur QSPR — журнал исследования",
        f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "Режим: QSPR-моделирование свойств молекул",
        "",
    ]

    if events:
        lines.extend(_log_format_event(event, include_details=True) for event in events)
    else:
        lines.append("Журнал пуст.")

    return "\n".join(lines) + "\n"


def show_logs():
    if "logs" in st.session_state and st.session_state.logs:
        st.subheader(t('logs.title'))

        events = st.session_state.get("log_events", [])
        if events:
            visible_events = [
                event for event in events
                if _log_normalize_level(event.get("level")) != "DEBUG"
            ][-20:]
            for event in visible_events:
                line = _log_format_event(event, include_details=False)
                level = _log_normalize_level(event.get("level"))
                if level == "ERROR":
                    st.error(line)
                elif level == "WARNING":
                    st.warning(line)
                else:
                    st.caption(line)
        else:
            for log in st.session_state.logs[-20:]:
                st.caption(log)

        st.download_button(
            "Скачать log.txt",
            build_log_txt().encode("utf-8"),
            "log.txt",
            "text/plain",
            key="download_research_log_txt",
        )
    else:
        st.info(t('logs.empty'))

def load_markdown_help(help_path):
    if not os.path.exists(help_path):
        return t('help.file_not_found', path=help_path)

    try:
        with open(help_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return t('help.read_error', path=help_path, error=e)


def show_markdown_help(title, help_path, expanded=False):
    with st.expander(title, expanded=expanded):
        st.markdown(load_markdown_help(help_path))


def get_admin_password_secret():
    try:
        return st.secrets.get("ADMIN_PASSWORD", "")
    except Exception:
        return ""


def get_user_role():
    if bool(st.session_state.get("admin_authenticated", False)):
        return "admin"
    return "user"


def is_admin():
    return get_user_role() == "admin"


def is_user():
    return get_user_role() == "user"


def log_access_control(feature):
    message = f"INFO | ACCESS_CONTROL | User attempted to open admin-only feature: {feature}."
    add_log(message, stage="access_control", event="admin_only_blocked")


def show_admin_only_notice(feature):
    log_access_control(feature)
    st.info("Эта функция доступна только администратору.")


def render_admin_login_controls():
    with st.sidebar.expander("Администратор", expanded=False):
        if qspr_is_online_mode():
            st.info(ONLINE_LOCK_MESSAGE)
            st.text_input(
                "Admin password",
                type="password",
                disabled=True,
                key="admin_password_input_online_disabled",
            )
            st.button("Login", disabled=True, key="admin_login_button_online_disabled")
            return

        if is_admin():
            st.success("Режим администратора активен.")
            if st.button("Выйти из режима администратора", key="admin_logout_button"):
                st.session_state.admin_authenticated = False
                st.rerun()
            return

        admin_password = get_admin_password_secret()
        entered_password = st.text_input(
            "Пароль администратора",
            type="password",
            key="admin_password_input",
        )

        if st.button("Войти", key="admin_login_button"):
            if (
                admin_password
                and entered_password
                and hmac.compare_digest(str(entered_password), str(admin_password))
            ):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.session_state.admin_authenticated = False
                st.caption("Обычный пользовательский режим.")

        st.caption("Без корректного пароля приложение работает в обычном режиме.")


def show_compact_matplotlib_plot(fig, width=850, dpi=140):
    """
    Показывает matplotlib-график контролируемого размера.
    """
    current_w, current_h = fig.get_size_inches()
    aspect = current_h / current_w if current_w > 0 else 0.7

    new_w_inch = width / dpi
    new_h_inch = new_w_inch * aspect

    fig.set_size_inches(new_w_inch, new_h_inch)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)

    st.image(buf, width=width)
    plt.close(fig)


def tooltip(text, explanation):
    return f"{text} <span style='cursor: help; border-bottom: 1px dotted gray;' title='{explanation}'>ℹ️</span>"

def spectral_selection_reason_to_text(reason):
    """
    Человекочитаемое объяснение причины выбора спектра.
    """
    reason = str(reason).strip()

    if reason == "selected_gas_by_priority":
        return t('spectra.reason_gas_priority')
    if reason == "selected_only_gas":
        return t('spectra.reason_only_gas')
    if reason.startswith("gas_not_available_selected_"):
        phase = reason.replace("gas_not_available_selected_", "")
        return t('spectra.reason_gas_not_available', phase=phase)
    if reason.startswith("selected_from_manual_phases_"):
        phase = reason.replace("selected_from_manual_phases_", "")
        return t('spectra.reason_manual_phases', phase=phase)
    if reason.startswith("selected_any_active_"):
        phase = reason.replace("selected_any_active_", "")
        return t('spectra.reason_any_active', phase=phase)
    if reason.lower() in ["", "nan", "none", "unknown"]:
        return t('spectra.reason_unknown')

    return reason


def spectral_selection_reason_to_ru(reason):
    return spectral_selection_reason_to_text(reason)


def qspr_connect_spectral_descriptors_to_session(
    current_df,
    target_col,
    smiles_col_current,
    spectral_df,
    use_molecular_for_qspr=False
):
    """
    Подключает рассчитанные спектральные дескрипторы к QSPR-модулю.
    """
    if spectral_df is None or not isinstance(spectral_df, pd.DataFrame) or spectral_df.empty:
        raise ValueError(t('spectra.no_spectral_data'))

    spectral_work = spectral_df.copy()

    sparring_cols = [
        c for c in spectral_work.columns
        if str(c).startswith("spectral_sparring_")
        or str(c).startswith("SPEC_spectral_sparring_")
    ]

    if sparring_cols:
        id_cols = [
            c for c in [
                "row_index",
                "compound_id",
                "name",
                "input_smiles",
                "canonical_smiles",
                "inchikey",
                "spectrum_type",
                "spectrum_id",
            ]
            if c in spectral_work.columns
        ]

        st.session_state.spectral_sparring_control_df = spectral_work[
            id_cols + sparring_cols
        ].copy()

        spectral_work = spectral_work.drop(columns=sparring_cols, errors="ignore")
    else:
        st.session_state.spectral_sparring_control_df = pd.DataFrame()

    bundle = qspr_build_descriptor_matrix_from_sources(
        current_df=current_df,
        target_col=target_col,
        use_molecular=use_molecular_for_qspr,
        molecular_desc_df=st.session_state.get("df_desc"),
        molecular_valid_indices=st.session_state.get("valid_indices"),
        use_spectral=True,
        spectral_desc_df=spectral_work,
        smiles_col=smiles_col_current,
        restrict_to_spectral_subset=True,
    )

    store_descriptor_bundle(
        bundle,
        bundle["report"]["descriptor_source"]
    )

    st.session_state.descriptor_source_mode = t('spectra.source_mode_calc')
    st.session_state.descriptor_calculation_mode = "spectral_or_combined"

    st.session_state.desc_calculated = True
    st.session_state.X_all = bundle["X_all"]
    st.session_state.y_all = bundle["y_all"]
    st.session_state.valid_indices = bundle["valid_indices"]
    st.session_state.desc_names = bundle["desc_names"]
    st.session_state.df_desc = bundle["df_desc"]
    st.session_state.custom_descriptor_source = bundle["report"]["descriptor_source"]
    st.session_state.custom_descriptors_used = True

    st.session_state.spectral_qspr_match_info = bundle.get(
        "match_info",
        pd.DataFrame()
    )

    excluded_text = ""

    if sparring_cols:
        excluded_text = t('spectra.sparring_excluded_text', cols=', '.join(sparring_cols))

    st.session_state.qspr_descriptor_matrix_ready_message = (
        t('spectra.matrix_ready_message',
            rows=bundle['X_all'].shape[0],
            cols=bundle['X_all'].shape[1],
            source=bundle['report']['descriptor_source']
        ) + excluded_text
    )

    qspr_show_descriptor_meaning_table(
        desc_names=bundle["desc_names"],
        title=t('spectra.descriptor_meanings_title'),
        status_label=t('spectra.status_calculated_included'),
        expanded=False,
        key_prefix="spectral_descriptor_meanings"
    )    

    add_log(
        t('spectra.log_matrix_connected',
            source=bundle['report']['descriptor_source'],
            rows=bundle['X_all'].shape[0],
            cols=bundle['X_all'].shape[1],
            sparring_count=len(sparring_cols)
        )
    )

    return bundle

def qspr_make_morfeus_bundle_from_dataframe(morfeus_df, target_col):
    """
    Делает bundle из morfeus-таблицы в формате, совместимом со store_descriptor_bundle().
    """
    if morfeus_df is None or not isinstance(morfeus_df, pd.DataFrame) or morfeus_df.empty:
        raise ValueError(t('morfeus.bundle_empty'))

    work = morfeus_df.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()

    if target_col not in work.columns:
        raise ValueError(t('morfeus.bundle_target_missing', col=target_col))

    if "row_index" not in work.columns:
        work["_original_index"] = work.index.astype(int)
    else:
        work["_original_index"] = pd.to_numeric(
            work["row_index"],
            errors="coerce"
        )

    if "morfeus_status" in work.columns:
        status_counts = work["morfeus_status"].astype(str).value_counts().to_dict()
        work = work[work["morfeus_status"].astype(str) == "ok"].copy()
    else:
        status_counts = {}

    if work.empty:
        raise ValueError(t('morfeus.bundle_no_ok_rows', statuses=status_counts))

    service_cols = {
        "row_index",
        "_original_index",
        "compound_id",
        "input_smiles",
        "morfeus_status",
        "morfeus_error",
        "morfeus_traceback",
        "morfeus_3d_status",
        "morfeus_3d_message",
        "morfeus_sasa_status",
        "morfeus_dispersion_status",
        "morfeus_xtb_status",
        target_col,
    }

    descriptor_cols = [
        c for c in work.columns
        if str(c).startswith("morfeus_")
        and c not in service_cols
    ]

    work[target_col] = qspr_to_numeric(work[target_col])

    for col in descriptor_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    descriptor_cols = [
        col for col in descriptor_cols
        if work[col].notna().any()
    ]

    if not descriptor_cols:
        raise ValueError(t('morfeus.bundle_all_descriptors_empty', statuses=status_counts))

    valid_mask = work[target_col].notna()
    valid_mask = valid_mask & work[descriptor_cols].notna().any(axis=1)

    work = work.loc[valid_mask].copy()

    if work.empty:
        raise ValueError(t('morfeus.bundle_no_valid_rows', statuses=status_counts))

    df_desc = work[descriptor_cols].copy()

    # Заполняем оставшиеся NaN медианами колонок.
    for col in descriptor_cols:
        median_value = df_desc[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        df_desc[col] = df_desc[col].fillna(median_value)

    return {
        "X_all": df_desc.values.astype(float),
        "y_all": work[target_col].values.astype(float),
        "valid_indices": work["_original_index"].astype(int).tolist(),
        "desc_names": descriptor_cols,
        "df_desc": df_desc.reset_index(drop=True),
        "report": {
            "descriptor_source": "morfeus_3d_descriptors",
            "morfeus_status_counts": status_counts,
            "n_morfeus_valid_rows": len(work),
            "n_morfeus_descriptors": len(descriptor_cols),
        }
    }

def qspr_append_morfeus_to_bundle(base_bundle, morfeus_df, target_col):
    """
    Добавляет morfeus-дескрипторы к уже рассчитанному bundle.
    Сшивка идёт по _original_index.
    """
    if base_bundle is None:
        return qspr_make_morfeus_bundle_from_dataframe(morfeus_df, target_col)

    morfeus_bundle = qspr_make_morfeus_bundle_from_dataframe(
        morfeus_df=morfeus_df,
        target_col=target_col
    )

    base_desc = base_bundle["df_desc"].copy()
    base_desc["_original_index"] = list(base_bundle["valid_indices"])
    base_desc[target_col] = list(base_bundle["y_all"])

    morfeus_desc = morfeus_bundle["df_desc"].copy()
    morfeus_desc["_original_index"] = list(morfeus_bundle["valid_indices"])

    merged = base_desc.merge(
        morfeus_desc,
        on="_original_index",
        how="inner"
    )

    if merged.empty:
        raise ValueError(t('morfeus.append_merge_failed'))

    desc_names = list(base_bundle["desc_names"]) + list(morfeus_bundle["desc_names"])
    desc_names = [c for c in desc_names if c in merged.columns]

    y = merged[target_col].values.astype(float)
    X = merged[desc_names].values.astype(float)

    source_a = base_bundle.get("report", {}).get("descriptor_source", "base")
    source_label = f"{source_a}_plus_morfeus"

    return {
        "X_all": X,
        "y_all": y,
        "valid_indices": merged["_original_index"].astype(int).tolist(),
        "desc_names": desc_names,
        "df_desc": merged[desc_names].reset_index(drop=True),
        "report": {
            "descriptor_source": source_label,
            "n_objects": len(merged),
            "n_descriptors": len(desc_names),
            "morfeus_report": morfeus_bundle.get("report", {}),
            "base_report": base_bundle.get("report", {}),
        }
    }

def qspr_make_dscribe_bundle_from_dataframe(dscribe_df, target_col):
    """
    Делает bundle из DScribe-таблицы в формате, совместимом со store_descriptor_bundle().
    """
    if dscribe_df is None or not isinstance(dscribe_df, pd.DataFrame) or dscribe_df.empty:
        raise ValueError(t('dscribe.bundle_empty'))

    work = dscribe_df.copy()
    work = work.loc[:, ~work.columns.duplicated()].copy()

    if target_col not in work.columns:
        raise ValueError(t('dscribe.bundle_target_missing', col=target_col))

    if "row_index" not in work.columns:
        work["_original_index"] = work.index.astype(int)
    else:
        work["_original_index"] = pd.to_numeric(
            work["row_index"],
            errors="coerce"
        )

    if "dscribe_status" in work.columns:
        status_counts = work["dscribe_status"].astype(str).value_counts().to_dict()
        work = work[work["dscribe_status"].astype(str) == "ok"].copy()
    else:
        status_counts = {}

    if work.empty:
        raise ValueError(t('dscribe.bundle_no_ok_rows', statuses=status_counts))

    service_cols = {
        "row_index",
        "_original_index",
        "compound_id",
        "input_smiles",
        "dscribe_status",
        "dscribe_error",
        "dscribe_traceback",
        "dscribe_3d_status",
        "dscribe_3d_message",
        "dscribe_coulomb_status",
        target_col,
    }

    descriptor_cols = [
        c for c in work.columns
        if str(c).startswith("dscribe_")
        and c not in service_cols
    ]

    work[target_col] = qspr_to_numeric(work[target_col])

    for col in descriptor_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    descriptor_cols = [
        col for col in descriptor_cols
        if work[col].notna().any()
    ]

    if not descriptor_cols:
        raise ValueError(t('dscribe.bundle_all_descriptors_empty', statuses=status_counts))

    valid_mask = work[target_col].notna()
    valid_mask = valid_mask & work[descriptor_cols].notna().any(axis=1)

    work = work.loc[valid_mask].copy()

    if work.empty:
        raise ValueError(t('dscribe.bundle_no_valid_rows', statuses=status_counts))

    df_desc = work[descriptor_cols].copy()

    for col in descriptor_cols:
        median_value = df_desc[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        df_desc[col] = df_desc[col].fillna(median_value)

    return {
        "X_all": df_desc.values.astype(float),
        "y_all": work[target_col].values.astype(float),
        "valid_indices": work["_original_index"].astype(int).tolist(),
        "desc_names": descriptor_cols,
        "df_desc": df_desc.reset_index(drop=True),
        "report": {
            "descriptor_source": "dscribe_atomistic_descriptors",
            "dscribe_status_counts": status_counts,
            "n_dscribe_valid_rows": len(work),
            "n_dscribe_descriptors": len(descriptor_cols),
        }
    }

def qspr_append_dscribe_to_bundle(base_bundle, dscribe_df, target_col):
    """
    Добавляет DScribe-дескрипторы к уже рассчитанному bundle.
    Сшивка идёт по _original_index.
    """
    if base_bundle is None:
        return qspr_make_dscribe_bundle_from_dataframe(dscribe_df, target_col)

    dscribe_bundle = qspr_make_dscribe_bundle_from_dataframe(
        dscribe_df=dscribe_df,
        target_col=target_col
    )

    base_desc = base_bundle["df_desc"].copy()
    base_desc["_original_index"] = list(base_bundle["valid_indices"])
    base_desc[target_col] = list(base_bundle["y_all"])

    dscribe_desc = dscribe_bundle["df_desc"].copy()
    dscribe_desc["_original_index"] = list(dscribe_bundle["valid_indices"])

    merged = base_desc.merge(
        dscribe_desc,
        on="_original_index",
        how="inner"
    )

    if merged.empty:
        raise ValueError(t('dscribe.append_merge_failed'))

    desc_names = list(base_bundle["desc_names"]) + list(dscribe_bundle["desc_names"])
    desc_names = [c for c in desc_names if c in merged.columns]

    y = merged[target_col].values.astype(float)
    X = merged[desc_names].values.astype(float)

    source_a = base_bundle.get("report", {}).get("descriptor_source", "base")
    source_label = f"{source_a}_plus_dscribe"

    return {
        "X_all": X,
        "y_all": y,
        "valid_indices": merged["_original_index"].astype(int).tolist(),
        "desc_names": desc_names,
        "df_desc": merged[desc_names].reset_index(drop=True),
        "report": {
            "descriptor_source": source_label,
            "n_objects": len(merged),
            "n_descriptors": len(desc_names),
            "dscribe_report": dscribe_bundle.get("report", {}),
            "base_report": base_bundle.get("report", {}),
        }
    }

def qspr_make_xtb_bundle_from_dataframe(xtb_df, target_col):
    """
    Делает bundle из xTB-таблицы в формате, совместимом со store_descriptor_bundle().

    Логика:
    - берём только строки xtb_status == ok;
    - целевое свойство должно быть числовым;
    - xTB-дескрипторы переводим в числа;
    - полностью пустые xTB-колонки удаляем;
    - строка сохраняется, если есть хотя бы один валидный xTB-дескриптор.
    """
    if xtb_df is None or not isinstance(xtb_df, pd.DataFrame) or xtb_df.empty:
        raise ValueError(t('xtb.bundle_empty'))

    work = xtb_df.copy()

    if "_original_index" not in work.columns:
        raise ValueError(t('xtb.bundle_no_original_index'))

    if target_col not in work.columns:
        raise ValueError(t('xtb.bundle_target_missing', col=target_col))

    if "xtb_status" not in work.columns:
        raise ValueError(t('xtb.bundle_no_status'))

    descriptor_cols = [
        c for c in work.columns
        if str(c).startswith("xtb_") and c != "xtb_status"
    ]

    if not descriptor_cols:
        raise ValueError(t('xtb.bundle_no_descriptors'))

    status_counts = (
        work["xtb_status"]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )

    work = work[work["xtb_status"].astype(str) == "ok"].copy()

    if work.empty:
        raise ValueError(t('xtb.bundle_no_ok_rows', statuses=status_counts))

    work[target_col] = qspr_to_numeric(work[target_col])

    for col in descriptor_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    descriptor_cols = [
        col for col in descriptor_cols
        if work[col].notna().any()
    ]

    if not descriptor_cols:
        raise ValueError(t('xtb.bundle_all_descriptors_empty', statuses=status_counts))

    valid_mask = work[target_col].notna()
    valid_mask = valid_mask & work[descriptor_cols].notna().any(axis=1)

    work = work.loc[valid_mask].copy()

    if work.empty:
        raise ValueError(t('xtb.bundle_no_valid_rows', statuses=status_counts))

    df_desc = work[descriptor_cols].copy()

    # Заполняем оставшиеся NaN медианами колонок, чтобы модель могла обучаться.
    for col in descriptor_cols:
        median_value = df_desc[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        df_desc[col] = df_desc[col].fillna(median_value)

    return {
        "X_all": df_desc.values.astype(float),
        "y_all": work[target_col].values.astype(float),
        "valid_indices": work["_original_index"].astype(int).tolist(),
        "desc_names": descriptor_cols,
        "df_desc": df_desc.reset_index(drop=True),
        "report": {
            "descriptor_source": "xtb_quantum_descriptors",
            "xtb_status_counts": status_counts,
            "n_xtb_valid_rows": len(work),
            "n_xtb_descriptors": len(descriptor_cols),
        }
    }

def qspr_append_xtb_to_bundle(base_bundle, xtb_df, target_col):
    """
    Добавляет xTB-дескрипторы к уже рассчитанному molecular bundle.
    Сшивка идёт по _original_index.
    """
    if base_bundle is None:
        return qspr_make_xtb_bundle_from_dataframe(xtb_df, target_col)

    xtb_bundle = qspr_make_xtb_bundle_from_dataframe(xtb_df, target_col)

    base_desc = base_bundle["df_desc"].copy()
    base_desc["_original_index"] = list(base_bundle["valid_indices"])
    base_desc[target_col] = list(base_bundle["y_all"])

    xtb_desc = xtb_bundle["df_desc"].copy()
    xtb_desc["_original_index"] = list(xtb_bundle["valid_indices"])

    merged = base_desc.merge(
        xtb_desc,
        on="_original_index",
        how="inner"
    )

    if merged.empty:
        raise ValueError(t('xtb.append_merge_failed'))

    desc_names = list(base_bundle["desc_names"]) + list(xtb_bundle["desc_names"])

    merged[target_col] = qspr_to_numeric(merged[target_col])

    merged[target_col] = qspr_to_numeric(merged[target_col])

    for col in desc_names:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    valid_mask = merged[target_col].notna()
    valid_mask = valid_mask & merged[desc_names].notna().any(axis=1)

    merged = merged.loc[valid_mask].copy()

    if merged.empty:
        raise ValueError(t('xtb.append_no_valid_rows'))

    for col in desc_names:
        median_value = merged[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        merged[col] = merged[col].fillna(median_value)

    return {
        "X_all": merged[desc_names].values.astype(float),
        "y_all": merged[target_col].values.astype(float),
        "valid_indices": merged["_original_index"].astype(int).tolist(),
        "desc_names": desc_names,
        "df_desc": merged[desc_names].copy(),
        "report": {
            "descriptor_source": "molecular_plus_xtb"
        }
    }

def get_model_params_from_session():
    """Параметры моделей из session_state для qspr_core."""
    return {
        "pls_components": st.session_state.get("pls_components", 2),
        "ridge_alpha": st.session_state.get("ridge_alpha", 1.0),
        "lasso_alpha": st.session_state.get("lasso_alpha", 0.01),
        "elastic_alpha": st.session_state.get("elastic_alpha", 0.01),
        "elastic_l1_ratio": st.session_state.get("elastic_l1_ratio", 0.5),
        "rf_n_estimators": st.session_state.get("rf_n_estimators", 300),
        "xgb_n_estimators": st.session_state.get("xgb_n_estimators", 300),
        "xgb_learning_rate": st.session_state.get("xgb_learning_rate", 0.05),
        "xgb_max_depth": st.session_state.get("xgb_max_depth", 4),
        "lightgbm_n_estimators": st.session_state.get("lightgbm_n_estimators", 300),
        "lightgbm_learning_rate": st.session_state.get("lightgbm_learning_rate", 0.05),
        "lightgbm_num_leaves": st.session_state.get("lightgbm_num_leaves", 31),
        "catboost_iterations": st.session_state.get("catboost_iterations", 300),
        "catboost_learning_rate": st.session_state.get("catboost_learning_rate", 0.05),
        "catboost_depth": st.session_state.get("catboost_depth", 6),
        "stacking_cv": st.session_state.get("stacking_cv", 5),
        "stacking_passthrough": st.session_state.get("stacking_passthrough", False),
        "svr_c": st.session_state.get("svr_c", 10.0),
        "svr_epsilon": st.session_state.get("svr_epsilon", 0.1),
        "svr_gamma": st.session_state.get("svr_gamma", "scale"),
        "gpr_alpha": st.session_state.get("gpr_alpha", 0.000001),
        "gpr_length_scale": st.session_state.get("gpr_length_scale", 1.0),
        "gpr_noise_level": st.session_state.get("gpr_noise_level", 0.1),
        "knn_n_neighbors": st.session_state.get("knn_n_neighbors", 5),
        "knn_weights": st.session_state.get("knn_weights", "distance"),
        "cart_max_depth": st.session_state.get("cart_max_depth", 5),
        "cart_min_samples_leaf": st.session_state.get("cart_min_samples_leaf", 2),
        "mars_degree": st.session_state.get("mars_degree", 2),
        "mars_alpha": st.session_state.get("mars_alpha", 1.0),
        "spline_n_knots": st.session_state.get("spline_n_knots", 5),
        "spline_degree": st.session_state.get("spline_degree", 3),
        "spline_alpha": st.session_state.get("spline_alpha", 1.0),
        "gam_n_splines": st.session_state.get("gam_n_splines", 6),
        "gam_degree": st.session_state.get("gam_degree", 3),
        "gam_alpha": st.session_state.get("gam_alpha", 1.0),
        "gep_population_size": st.session_state.get("gep_population_size", 500),
        "gep_generations": st.session_state.get("gep_generations", 20),
        "gep_max_depth": st.session_state.get("gep_max_depth", 4),
        "gp_population_size": st.session_state.get("gp_population_size", 500),
        "gp_generations": st.session_state.get("gp_generations", 20),
        "gp_max_depth": st.session_state.get("gp_max_depth", 4),
        "pysr_niterations": st.session_state.get("pysr_niterations", 40),
        "pysr_populations": st.session_state.get("pysr_populations", 8),
        "pysr_maxsize": st.session_state.get("pysr_maxsize", 20),
        "voting_rf_weight": st.session_state.get("voting_rf_weight", 1.0),
        "voting_extra_trees_weight": st.session_state.get("voting_extra_trees_weight", 1.0),
        "voting_ridge_weight": st.session_state.get("voting_ridge_weight", 1.0),
        "et_n_estimators": st.session_state.get("et_n_estimators", 300),
        "et_max_depth": st.session_state.get("et_max_depth", None),
        "et_min_samples_split": st.session_state.get("et_min_samples_split", 2),
        "et_min_samples_leaf": st.session_state.get("et_min_samples_leaf", 1),
        "et_max_features": st.session_state.get("et_max_features", "sqrt"),
    }


QSPR_IS_CLOUD_RUNTIME = (
    os.environ.get("AUGUR_QSPR_CLOUD_GUARD", "").lower() in {"1", "true", "yes", "y"}
    or os.getcwd().replace("\\", "/").startswith("/mount/src/")
    or os.path.abspath(__file__).replace("\\", "/").startswith("/mount/src/")
)
QSPR_LOO_MAX_SAMPLES = int(os.environ.get("AUGUR_QSPR_LOO_MAX_SAMPLES", "150"))
QSPR_LOO_HEAVY_MAX_SAMPLES = int(os.environ.get("AUGUR_QSPR_LOO_HEAVY_MAX_SAMPLES", "40"))
QSPR_ALLOW_EXPENSIVE_LOO = os.environ.get("AUGUR_QSPR_ALLOW_EXPENSIVE_LOO", "").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
QSPR_LOO_HEAVY_MODELS = {
    "Random Forest",
    "Extra Trees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "MLP Regression",
    "Gaussian Process Regression (GPR)",
    "AdaBoost Regressor",
    "HistGradientBoosting Regressor",
    "Stacking",
    "Voting Regressor",
    "GEP Symbolic Regression",
    "Genetic Programming Regression",
    "PySR",
}


def qspr_loo_skip_reason(model_name, n_samples):
    if QSPR_ALLOW_EXPENSIVE_LOO or not QSPR_IS_CLOUD_RUNTIME:
        return None

    model_key = normalize_runtime_name(model_name)

    if int(n_samples) > QSPR_LOO_MAX_SAMPLES:
        return t(
            "loo_guard.skip_general",
            n=int(n_samples),
            limit=QSPR_LOO_MAX_SAMPLES,
        )

    if model_key in QSPR_LOO_HEAVY_MODELS and int(n_samples) > QSPR_LOO_HEAVY_MAX_SAMPLES:
        return t(
            "loo_guard.skip_heavy",
            model=model_name,
            n=int(n_samples),
            limit=QSPR_LOO_HEAVY_MAX_SAMPLES,
        )

    return None


def qspr_metric_from_result(result, metric_name, metrics_key="metrics"):
    """Безопасно достаёт метрику из результата валидации."""
    if result is None:
        return np.nan

    metrics = result.get(metrics_key, {}) if isinstance(result, dict) else {}

    try:
        value = metrics.get(metric_name, np.nan)
        return float(value)
    except Exception:
        return np.nan


def qspr_model_group_for_name(model_name, model_groups=None):
    """Находит группу модели по её названию."""
    catalog_group = get_model_group(model_name)
    if catalog_group:
        return catalog_group

    if model_groups is None:
        try:
            model_groups = qspr_available_model_options()
        except Exception:
            model_groups = {}

    for group_name, models in model_groups.items():
        if model_name in models:
            return group_name

    return ""


def qspr_count_outside_ad_for_model(model_data):
    """Считает число обучающих веществ вне leverage AD для сохранённой модели."""
    try:
        X_model = model_data.get("X_scaled", None)

        if X_model is None:
            return np.nan, np.nan

        ad = qspr_calculate_leverage_ad(X_train=X_model)
        leverage = np.asarray(ad["leverage"], dtype=float)
        threshold = float(ad["threshold"])
        n_out = int(np.sum(leverage > threshold))
        percent_out = n_out / len(leverage) * 100 if len(leverage) else np.nan
        return n_out, percent_out
    except Exception:
        return np.nan, np.nan


def qspr_make_model_comment(row):
    """Короткий экспертный комментарий к строке сравнения моделей."""
    comments = []

    trained_r2 = row.get("Train R²", np.nan)
    holdout_r2 = row.get("Hold-out R²", np.nan)
    kfold_q2 = row.get("K-Fold Q²", np.nan)
    loo_q2 = row.get("LOO Q²", np.nan)
    rmse = row.get("RMSE", np.nan)
    outside_ad = row.get("Вне AD", np.nan)

    if pd.isna(holdout_r2) and pd.isna(kfold_q2) and pd.isna(loo_q2):
        comments.append(t('model_comment.no_validation'))

    cv_values = [v for v in [kfold_q2, loo_q2] if pd.notna(v)]

    if cv_values:
        best_cv = max(cv_values)
        if best_cv >= 0.75:
            comments.append(t('model_comment.cv_strong'))
        elif best_cv >= 0.50:
            comments.append(t('model_comment.cv_moderate'))
        elif best_cv >= 0.20:
            comments.append(t('model_comment.cv_weak'))
        else:
            comments.append(t('model_comment.cv_random'))

    if pd.notna(trained_r2) and pd.notna(kfold_q2):
        if trained_r2 - kfold_q2 > 0.30:
            comments.append(t('model_comment.overfitting'))

    if pd.notna(trained_r2) and pd.notna(loo_q2):
        if trained_r2 - loo_q2 > 0.30 and t('model_comment.overfitting') not in comments:
            comments.append(t('model_comment.overfitting'))

    if pd.notna(outside_ad) and outside_ad > 0:
        outside_ad_count = int(outside_ad)
        comments.append(t(
            'model_comment.outside_ad',
            n=outside_ad_count,
            count=outside_ad_count
        ))

    if pd.notna(rmse):
        comments.append(t('model_comment.rmse_in_rating'))

    if not comments:
        comments.append(t('model_comment.default'))

    return "; ".join(comments)


def qspr_build_model_comparison_table():
    """Формирует сводную таблицу сравнения обученных и валидированных моделей."""
    trained = st.session_state.get("trained_models", {}) or {}
    holdouts = st.session_state.get("holdout_results_dict", {}) or {}
    kfolds = st.session_state.get("kfold_results_dict", {}) or {}
    loos = st.session_state.get("loo_results_dict", {}) or {}

    model_names = sorted(set(trained.keys()) | set(holdouts.keys()) | set(kfolds.keys()) | set(loos.keys()))

    rows = []

    for name in model_names:
        model_data = trained.get(name, {}) or {}
        hold = holdouts.get(name)
        kfold = kfolds.get(name)
        loo = loos.get(name)

        n_out_ad, pct_out_ad = qspr_count_outside_ad_for_model(model_data) if model_data else (np.nan, np.nan)

        candidate_rmse = [
            qspr_metric_from_result(hold, "RMSE", metrics_key="metrics_test"),
            qspr_metric_from_result(kfold, "RMSE"),
            qspr_metric_from_result(loo, "RMSE"),
        ]
        candidate_mae = [
            qspr_metric_from_result(hold, "MAE", metrics_key="metrics_test"),
            qspr_metric_from_result(kfold, "MAE"),
            qspr_metric_from_result(loo, "MAE"),
        ]

        rmse_available = [x for x in candidate_rmse if pd.notna(x)]
        mae_available = [x for x in candidate_mae if pd.notna(x)]

        row = {
            t('comparison.model'): name,
            t('comparison.group'): qspr_model_group_for_name(name),
            "Train R²": float(model_data.get("metrics", {}).get("R2", np.nan)) if model_data else np.nan,
            "Hold-out R²": qspr_metric_from_result(hold, "R2", metrics_key="metrics_test"),
            "K-Fold Q²": qspr_metric_from_result(kfold, "R2"),
            "LOO Q²": qspr_metric_from_result(loo, "R2"),
            "RMSE": float(np.nanmean(rmse_available)) if rmse_available else np.nan,
            "MAE": float(np.nanmean(mae_available)) if mae_available else np.nan,
            t('comparison.outside_ad'): n_out_ad,
            t('comparison.outside_ad_percent'): pct_out_ad,
            t('comparison.checks'): ", ".join([
                label for label, res in [
                    ("Hold-out", hold),
                    ("K-Fold", kfold),
                    ("LOO", loo),
                ]
                if res is not None
            ]) or t('comparison.only_training'),
        }

        row[t('comparison.comment')] = qspr_make_model_comment(row)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    rank_parts = []

    for col in ["K-Fold Q²", "LOO Q²", "Hold-out R²"]:
        if table[col].notna().any():
            rank_parts.append(table[col].rank(ascending=False, na_option="bottom"))

    for col in ["RMSE", "MAE", t('comparison.outside_ad')]:
        if table[col].notna().any():
            rank_parts.append(table[col].rank(ascending=True, na_option="bottom"))

    if rank_parts:
        rank_sum = np.zeros(len(table), dtype=float)
        for r in rank_parts:
            rank_sum += r.values.astype(float)
        table[t('comparison.rating')] = rank_sum / len(rank_parts)
    else:
        table[t('comparison.rating')] = np.nan

    table = table.sort_values(
        by=[t('comparison.rating'), "K-Fold Q²", "LOO Q²", "Hold-out R²"],
        ascending=[True, False, False, False],
        na_position="last"
    ).reset_index(drop=True)

    table.insert(0, t('comparison.place'), range(1, len(table) + 1))

    return table


def qspr_run_missing_validation_for_models(
    model_names,
    X,
    y,
    valid_indices,
    smiles,
    target_col,
    run_holdout=False,
    run_kfold=True,
    run_loo=False,
    holdout_test_size=0.2,
    holdout_random_state=42,
    kfold_k=5,
):
    """Досчитывает выбранные типы валидации для списка обученных моделей."""
    messages = []
    params = get_model_params_from_session()

    for candidate_model in model_names:
        if run_holdout and candidate_model not in st.session_state.holdout_results_dict:
            res_hold = qspr_holdout_validation(
                X=X,
                y=y,
                model_name=candidate_model,
                valid_indices=valid_indices,
                smiles=smiles,
                test_size=holdout_test_size,
                random_state=holdout_random_state,
                use_random=True,
                manual_indices=None,
                params=params,
                scale=True,
            )
            st.session_state.holdout_results_dict[candidate_model] = res_hold
            combined_df = pd.concat([res_hold["train_table"], res_hold["test_table"]], ignore_index=True)
            qspr_save_results_auto(combined_df, "holdout", target_col, len(y))
            messages.append(f"Hold-out: {candidate_model}")

        if run_kfold and candidate_model not in st.session_state.kfold_results_dict:
            res_kfold = qspr_kfold_validation(
                X=X,
                y=y,
                model_name=candidate_model,
                valid_indices=valid_indices,
                smiles=smiles,
                k=kfold_k,
                params=params,
                scale=True,
                shuffle=True,
                random_state=42,
            )
            st.session_state.kfold_results_dict[candidate_model] = res_kfold
            qspr_save_results_auto(res_kfold["result_table"], "kfold", target_col, len(y))
            messages.append(f"K-Fold: {candidate_model}")

        if run_loo and candidate_model not in st.session_state.loo_results_dict:
            loo_skip_reason = qspr_loo_skip_reason(candidate_model, len(y))
            if loo_skip_reason:
                messages.append(loo_skip_reason)
                continue

            res_loo = qspr_loo_validation(
                X=X,
                y=y,
                model_name=candidate_model,
                valid_indices=valid_indices,
                smiles=smiles,
                params=params,
                scale=True,
            )
            st.session_state.loo_results_dict[candidate_model] = res_loo
            qspr_save_results_auto(res_loo["result_table"], "loo", target_col, len(y))
            messages.append(f"LOO: {candidate_model}")

    return messages



def qspr_auto_train_validate_models_for_comparison(
    model_names,
    X,
    y,
    valid_indices,
    smiles,
    target_col,
    run_holdout=True,
    run_kfold=True,
    run_loo=False,
    holdout_test_size=0.2,
    holdout_random_state=42,
    kfold_k=5,
    force_retrain=True,
):
    """
    Настоящее сравнение моделей-кандидатов:
    1) обучает каждую выбранную модель на тех же дескрипторах;
    2) считает выбранные схемы валидации;
    3) сохраняет результаты в session_state;
    4) возвращает сообщения и ошибки по моделям.
    """
    messages = []
    errors = []
    params = get_model_params_from_session()

    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "holdout_results_dict" not in st.session_state:
        st.session_state.holdout_results_dict = {}
    if "kfold_results_dict" not in st.session_state:
        st.session_state.kfold_results_dict = {}
    if "loo_results_dict" not in st.session_state:
        st.session_state.loo_results_dict = {}

    for candidate_model in model_names:
        try:
            if force_retrain or candidate_model not in st.session_state.trained_models:
                train_res = qspr_train_analysis_model(
                    X=X,
                    y=y,
                    model_name=candidate_model,
                    params=params,
                    scale=True,
                )
                st.session_state.trained_models[candidate_model] = train_res
                messages.append(t('training.trained_model', model=candidate_model))

            if run_holdout and (force_retrain or candidate_model not in st.session_state.holdout_results_dict):
                res_hold = qspr_holdout_validation(
                    X=X,
                    y=y,
                    model_name=candidate_model,
                    valid_indices=valid_indices,
                    smiles=smiles,
                    test_size=holdout_test_size,
                    random_state=holdout_random_state,
                    use_random=True,
                    manual_indices=None,
                    params=params,
                    scale=True,
                )
                st.session_state.holdout_results_dict[candidate_model] = res_hold
                combined_df = pd.concat([res_hold["train_table"], res_hold["test_table"]], ignore_index=True)
                qspr_save_results_auto(combined_df, "holdout", target_col, len(y))

            if run_kfold and (force_retrain or candidate_model not in st.session_state.kfold_results_dict):
                res_kfold = qspr_kfold_validation(
                    X=X,
                    y=y,
                    model_name=candidate_model,
                    valid_indices=valid_indices,
                    smiles=smiles,
                    k=kfold_k,
                    params=params,
                    scale=True,
                    shuffle=True,
                    random_state=42,
                )
                st.session_state.kfold_results_dict[candidate_model] = res_kfold
                qspr_save_results_auto(res_kfold["result_table"], "kfold", target_col, len(y))

            if run_loo and (force_retrain or candidate_model not in st.session_state.loo_results_dict):
                loo_skip_reason = qspr_loo_skip_reason(candidate_model, len(y))
                if loo_skip_reason:
                    errors.append({
                        t('training.model'): candidate_model,
                        t('training.group'): qspr_model_group_for_name(candidate_model),
                        t('training.error'): loo_skip_reason,
                    })
                    continue

                res_loo = qspr_loo_validation(
                    X=X,
                    y=y,
                    model_name=candidate_model,
                    valid_indices=valid_indices,
                    smiles=smiles,
                    params=params,
                    scale=True,
                )
                st.session_state.loo_results_dict[candidate_model] = res_loo
                qspr_save_results_auto(res_loo["result_table"], "loo", target_col, len(y))

        except Exception as e:
            errors.append({
                t('training.model'): candidate_model,
                t('training.group'): qspr_model_group_for_name(candidate_model),
                t('training.error'): str(e),
            })
            continue

    return messages, pd.DataFrame(errors)

def qspr_bootstrap_validation(
    X,
    y,
    model_name,
    valid_indices,
    smiles,
    target_col,
    params=None,
    n_iterations=200,
    sample_fraction=1.0,
    random_state=42,
    scale=True,
    min_oob_size=2,
    progress_callback=None
):
    """
    Bootstrap / out-of-bag validation для QSPR-модели.

    Логика:
    - на каждой итерации берём bootstrap-выборку с возвращением;
    - обучаем модель на bootstrap-выборке;
    - проверяем на out-of-bag объектах, которые не попали в bootstrap train;
    - считаем R2, RMSE, MAE, MAPE;
    - возвращаем mean ± std по итерациям.

    Это полезно для малых выборок, где одиночный Hold-out может быть нестабилен.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(y)

    if n < 5:
        raise ValueError(t('bootstrap.min_compounds'))

    if params is None:
        params = {}

    if valid_indices is None:
        valid_indices = list(range(n))
    else:
        valid_indices = list(valid_indices)

    if smiles is None:
        smiles = [""] * n
    else:
        smiles = list(smiles)

    n_iterations = int(n_iterations)
    n_iterations = max(1, n_iterations)

    sample_fraction = float(sample_fraction)

    if sample_fraction <= 0:
        raise ValueError(t('bootstrap.sample_fraction_positive'))

    boot_size = int(round(n * sample_fraction))
    boot_size = max(2, boot_size)

    rng = np.random.default_rng(int(random_state))

    iteration_rows = []
    oob_prediction_rows = []
    skipped_iterations = 0

    for i in range(n_iterations):
        try:
            bootstrap_idx = rng.integers(0, n, size=boot_size)
            bootstrap_unique = set(bootstrap_idx.tolist())

            oob_idx = np.array(
                [j for j in range(n) if j not in bootstrap_unique],
                dtype=int
            )

            if len(oob_idx) < int(min_oob_size):
                skipped_iterations += 1

                iteration_rows.append({
                    t('bootstrap.iteration'): i + 1,
                    t('bootstrap.status'): "skipped_no_oob",
                    "Train n": len(bootstrap_idx),
                    "Train unique n": len(bootstrap_unique),
                    "OOB n": len(oob_idx),
                    "R2 OOB": np.nan,
                    "RMSE OOB": np.nan,
                    "MAE OOB": np.nan,
                    "MAPE OOB, %": np.nan,
                })

                if progress_callback is not None:
                    progress_callback(i + 1, n_iterations)

                continue

            X_train = X[bootstrap_idx]
            y_train = y[bootstrap_idx]

            X_oob = X[oob_idx]
            y_oob = y[oob_idx]

            if scale:
                scaler = StandardScaler()
                X_train_model = scaler.fit_transform(X_train)
                X_oob_model = scaler.transform(X_oob)
            else:
                X_train_model = X_train
                X_oob_model = X_oob

            model = qspr_create_regression_model(
                model_name,
                n_samples=X_train_model.shape[0],
                n_features=X_train_model.shape[1],
                params=params
            )

            model.fit(X_train_model, y_train)

            y_oob_pred = np.ravel(model.predict(X_oob_model))

            metrics_oob = qspr_metrics(y_oob, y_oob_pred)

            iteration_rows.append({
                t('bootstrap.iteration'): i + 1,
                t('bootstrap.status'): "ok",
                "Train n": len(bootstrap_idx),
                "Train unique n": len(bootstrap_unique),
                "OOB n": len(oob_idx),
                "R2 OOB": metrics_oob.get("R2", np.nan),
                "RMSE OOB": metrics_oob.get("RMSE", np.nan),
                "MAE OOB": metrics_oob.get("MAE", np.nan),
                "MAPE OOB, %": metrics_oob.get("MAPE_percent", np.nan),
            })

            for local_pos, idx_oob in enumerate(oob_idx):
                oob_prediction_rows.append({
                    t('bootstrap.iteration'): i + 1,
                    t('bootstrap.original_row'): int(valid_indices[idx_oob]) + 1,
                    "SMILES": smiles[idx_oob] if idx_oob < len(smiles) else "",
                    t('bootstrap.experimental'): float(y_oob[local_pos]),
                    t('bootstrap.predicted'): float(y_oob_pred[local_pos]),
                    t('bootstrap.error'): float(y_oob[local_pos] - y_oob_pred[local_pos]),
                    t('bootstrap.abs_error'): float(abs(y_oob[local_pos] - y_oob_pred[local_pos])),
                })

        except Exception as e:
            skipped_iterations += 1

            iteration_rows.append({
                t('bootstrap.iteration'): i + 1,
                t('bootstrap.status'): f"error: {e}",
                "Train n": np.nan,
                "Train unique n": np.nan,
                "OOB n": np.nan,
                "R2 OOB": np.nan,
                "RMSE OOB": np.nan,
                "MAE OOB": np.nan,
                "MAPE OOB, %": np.nan,
            })

        if progress_callback is not None:
            progress_callback(i + 1, n_iterations)

    iterations_table = pd.DataFrame(iteration_rows)
    oob_predictions_table = pd.DataFrame(oob_prediction_rows)

    ok_table = iterations_table[
        iterations_table[t('bootstrap.status')] == "ok"
    ].copy()

    def _mean_std(col):
        if ok_table.empty or col not in ok_table.columns:
            return np.nan, np.nan

        values = pd.to_numeric(ok_table[col], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if values.empty:
            return np.nan, np.nan

        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        return mean_value, std_value

    r2_mean, r2_std = _mean_std("R2 OOB")
    rmse_mean, rmse_std = _mean_std("RMSE OOB")
    mae_mean, mae_std = _mean_std("MAE OOB")
    mape_mean, mape_std = _mean_std("MAPE OOB, %")

    summary = {
        "model_name": model_name,
        "n_iterations_requested": int(n_iterations),
        "n_iterations_successful": int(len(ok_table)),
        "n_iterations_skipped_or_failed": int(skipped_iterations),
        "sample_fraction": float(sample_fraction),
        "bootstrap_train_size": int(boot_size),
        "r2_oob_mean": r2_mean,
        "r2_oob_std": r2_std,
        "rmse_oob_mean": rmse_mean,
        "rmse_oob_std": rmse_std,
        "mae_oob_mean": mae_mean,
        "mae_oob_std": mae_std,
        "mape_oob_mean": mape_mean,
        "mape_oob_std": mape_std,
    }

    summary_table = pd.DataFrame([
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_model'), t('bootstrap.summary_value'): model_name},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_iterations_requested'), t('bootstrap.summary_value'): int(n_iterations)},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_successful'), t('bootstrap.summary_value'): int(len(ok_table))},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_skipped'), t('bootstrap.summary_value'): int(skipped_iterations)},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_sample_fraction'), t('bootstrap.summary_value'): float(sample_fraction)},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_train_size'), t('bootstrap.summary_value'): int(boot_size)},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_r2_mean'), t('bootstrap.summary_value'): r2_mean},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_r2_std'), t('bootstrap.summary_value'): r2_std},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_rmse_mean'), t('bootstrap.summary_value'): rmse_mean},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_rmse_std'), t('bootstrap.summary_value'): rmse_std},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_mae_mean'), t('bootstrap.summary_value'): mae_mean},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_mae_std'), t('bootstrap.summary_value'): mae_std},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_mape_mean'), t('bootstrap.summary_value'): mape_mean},
        {t('bootstrap.summary_prompt'): t('bootstrap.summary_mape_std'), t('bootstrap.summary_value'): mape_std},
    ])

    return {
        "summary": summary,
        "summary_table": summary_table,
        "iterations_table": iterations_table,
        "oob_predictions_table": oob_predictions_table,
    }

def qspr_y_randomization_test(
    X,
    y,
    model_name,
    valid_indices,
    smiles,
    params=None,
    method="K-Fold",
    n_permutations=100,
    k=5,
    random_state=42,
    scale=True,
    progress_callback=None
):
    """
    Y-randomization / permutation test для QSPR-модели.

    Идея:
    - сначала считаем исходный Q² модели на настоящем y;
    - затем много раз перемешиваем y;
    - каждый раз заново обучаем модель и считаем CV Q²;
    - если случайные Q² редко достигают исходного Q², модель неслучайна.

    p-value:
    (число перестановок с Q²_perm >= Q²_original + 1) / (N + 1)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_indices = list(valid_indices)
    smiles = list(smiles)

    if params is None:
        params = {}

    method = str(method).strip()
    n_permutations = int(n_permutations)
    n_permutations = max(1, n_permutations)

    rng = np.random.default_rng(int(random_state))

    if method == "LOO":
        original_result = qspr_loo_validation(
            X=X,
            y=y,
            model_name=model_name,
            valid_indices=valid_indices,
            smiles=smiles,
            params=params,
            scale=scale,
        )
        original_metrics = original_result.get("metrics", {})
        original_table = original_result.get("result_table", pd.DataFrame())
        validation_label = "Leave-One-Out"
    else:
        k = int(k)
        k = max(2, min(k, len(y)))

        original_result = qspr_kfold_validation(
            X=X,
            y=y,
            model_name=model_name,
            valid_indices=valid_indices,
            smiles=smiles,
            k=k,
            params=params,
            scale=scale,
            shuffle=True,
            random_state=int(random_state),
        )
        original_metrics = original_result.get("metrics", {})
        original_table = original_result.get("result_table", pd.DataFrame())
        validation_label = f"{k}-Fold CV"

    original_q2 = float(original_metrics.get("R2", np.nan))
    original_rmse = float(original_metrics.get("RMSE", np.nan))
    original_mae = float(original_metrics.get("MAE", np.nan))

    rows = []

    for i in range(n_permutations):
        y_perm = rng.permutation(y)

        try:
            if method == "LOO":
                perm_result = qspr_loo_validation(
                    X=X,
                    y=y_perm,
                    model_name=model_name,
                    valid_indices=valid_indices,
                    smiles=smiles,
                    params=params,
                    scale=scale,
                )
            else:
                perm_result = qspr_kfold_validation(
                    X=X,
                    y=y_perm,
                    model_name=model_name,
                    valid_indices=valid_indices,
                    smiles=smiles,
                    k=k,
                    params=params,
                    scale=scale,
                    shuffle=True,
                    random_state=int(random_state) + i + 1,
                )

            perm_metrics = perm_result.get("metrics", {})

            q2_perm = float(perm_metrics.get("R2", np.nan))
            rmse_perm = float(perm_metrics.get("RMSE", np.nan))
            mae_perm = float(perm_metrics.get("MAE", np.nan))
            mape_perm = float(perm_metrics.get("MAPE_percent", np.nan))

            status = "ok"

        except Exception as e:
            q2_perm = np.nan
            rmse_perm = np.nan
            mae_perm = np.nan
            mape_perm = np.nan
            status = f"error: {e}"

        rows.append({
            t('y_randomization.permutation'): i + 1,
            t('y_randomization.q2_perm'): q2_perm,
            t('y_randomization.rmse_perm'): rmse_perm,
            t('y_randomization.mae_perm'): mae_perm,
            t('y_randomization.mape_perm_pct'): mape_perm,
            t('y_randomization.status'): status,
        })

        if progress_callback is not None:
            progress_callback(i + 1, n_permutations)

    permutation_table = pd.DataFrame(rows)

    q2_values = permutation_table[t('y_randomization.q2_perm')].replace([np.inf, -np.inf], np.nan).dropna()

    if len(q2_values) > 0 and np.isfinite(original_q2):
        mean_q2_perm = float(q2_values.mean())
        median_q2_perm = float(q2_values.median())
        max_q2_perm = float(q2_values.max())
        min_q2_perm = float(q2_values.min())
        std_q2_perm = float(q2_values.std(ddof=1)) if len(q2_values) > 1 else 0.0

        p_value = float((np.sum(q2_values >= original_q2) + 1) / (len(q2_values) + 1))
    else:
        mean_q2_perm = np.nan
        median_q2_perm = np.nan
        max_q2_perm = np.nan
        min_q2_perm = np.nan
        std_q2_perm = np.nan
        p_value = np.nan

    if np.isfinite(p_value) and np.isfinite(original_q2):
        if p_value <= 0.01 and original_q2 > mean_q2_perm:
            conclusion = t('y_randomization.conclusion_strong')
        elif p_value <= 0.05 and original_q2 > mean_q2_perm:
            conclusion = t('y_randomization.conclusion_significant')
        elif p_value <= 0.10 and original_q2 > mean_q2_perm:
            conclusion = t('y_randomization.conclusion_moderate')
        else:
            conclusion = t('y_randomization.conclusion_not_confirmed')
    else:
        conclusion = t('y_randomization.conclusion_undefined')

    summary = {
        "model_name": model_name,
        "validation_method": validation_label,
        "n_permutations": int(n_permutations),
        "original_q2": original_q2,
        "original_rmse": original_rmse,
        "original_mae": original_mae,
        "mean_q2_permuted": mean_q2_perm,
        "median_q2_permuted": median_q2_perm,
        "max_q2_permuted": max_q2_perm,
        "min_q2_permuted": min_q2_perm,
        "std_q2_permuted": std_q2_perm,
        "p_value": p_value,
        "conclusion": conclusion,
    }

    summary_table = pd.DataFrame([
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_model'), t('y_randomization.summary_value'): model_name},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_method'), t('y_randomization.summary_value'): validation_label},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_permutations'), t('y_randomization.summary_value'): n_permutations},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_original_q2'), t('y_randomization.summary_value'): original_q2},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_original_rmse'), t('y_randomization.summary_value'): original_rmse},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_original_mae'), t('y_randomization.summary_value'): original_mae},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_mean_q2'), t('y_randomization.summary_value'): mean_q2_perm},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_median_q2'), t('y_randomization.summary_value'): median_q2_perm},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_max_q2'), t('y_randomization.summary_value'): max_q2_perm},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_min_q2'), t('y_randomization.summary_value'): min_q2_perm},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_std_q2'), t('y_randomization.summary_value'): std_q2_perm},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_p_value'), t('y_randomization.summary_value'): p_value},
        {t('y_randomization.summary_prompt'): t('y_randomization.summary_conclusion'), t('y_randomization.summary_value'): conclusion},
    ])

    return {
        "summary": summary,
        "summary_table": summary_table,
        "permutation_table": permutation_table,
        "original_result": original_result,
        "original_table": original_table,
    }

def qspr_repeated_holdout_validation(
    X,
    y,
    model_name,
    valid_indices,
    smiles,
    target_col,
    params=None,
    n_repeats=100,
    test_size=0.2,
    random_state=42,
    scale=True,
    progress_callback=None
):
    """
    Repeated Hold-out / Monte-Carlo cross-validation.

    Логика:
    - много раз случайно делим данные на train/test;
    - каждый раз обучаем ту же модель;
    - собираем train/test метрики;
    - считаем mean ± std для R², RMSE, MAE, MAPE.

    Это проверяет устойчивость модели к случайному разбиению выборки.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_indices = list(valid_indices)
    smiles = list(smiles)

    if params is None:
        params = {}

    n_repeats = int(n_repeats)
    n_repeats = max(1, n_repeats)

    test_size = float(test_size)

    if test_size <= 0 or test_size >= 1:
        raise ValueError(t('repeated_holdout.test_size_error'))

    if len(y) < 5:
        raise ValueError(t('repeated_holdout.min_compounds'))

    repeat_rows = []
    all_train_tables = []
    all_test_tables = []

    for i in range(n_repeats):
        current_seed = int(random_state) + i

        try:
            res_hold = qspr_holdout_validation(
                X=X,
                y=y,
                model_name=model_name,
                valid_indices=valid_indices,
                smiles=smiles,
                test_size=test_size,
                random_state=current_seed,
                use_random=True,
                manual_indices=None,
                params=params,
                scale=scale,
            )

            metrics_train = res_hold.get("metrics_train", {})
            metrics_test = res_hold.get("metrics_test", {})

            repeat_rows.append({
                t('repeated_holdout.repeat'): i + 1,
                "random_state": current_seed,

                "Train R²": metrics_train.get("R2", np.nan),
                "Train RMSE": metrics_train.get("RMSE", np.nan),
                "Train MAE": metrics_train.get("MAE", np.nan),
                "Train MAPE, %": metrics_train.get("MAPE_percent", np.nan),

                "Test R²": metrics_test.get("R2", np.nan),
                "Test RMSE": metrics_test.get("RMSE", np.nan),
                "Test MAE": metrics_test.get("MAE", np.nan),
                "Test MAPE, %": metrics_test.get("MAPE_percent", np.nan),

                "Train n": len(res_hold.get("y_train", [])),
                "Test n": len(res_hold.get("y_test", [])),
                t('repeated_holdout.status'): "ok",
            })

            train_table_i = res_hold.get("train_table", pd.DataFrame()).copy()
            test_table_i = res_hold.get("test_table", pd.DataFrame()).copy()

            if isinstance(train_table_i, pd.DataFrame) and not train_table_i.empty:
                train_table_i.insert(0, t('repeated_holdout.repeat'), i + 1)
                train_table_i.insert(1, "random_state", current_seed)
                all_train_tables.append(train_table_i)

            if isinstance(test_table_i, pd.DataFrame) and not test_table_i.empty:
                test_table_i.insert(0, t('repeated_holdout.repeat'), i + 1)
                test_table_i.insert(1, "random_state", current_seed)
                all_test_tables.append(test_table_i)

        except Exception as e:
            repeat_rows.append({
                t('repeated_holdout.repeat'): i + 1,
                "random_state": current_seed,
                "Train R²": np.nan,
                "Train RMSE": np.nan,
                "Train MAE": np.nan,
                "Train MAPE, %": np.nan,
                "Test R²": np.nan,
                "Test RMSE": np.nan,
                "Test MAE": np.nan,
                "Test MAPE, %": np.nan,
                "Train n": np.nan,
                "Test n": np.nan,
                t('repeated_holdout.status'): f"error: {e}",
            })

        if progress_callback is not None:
            progress_callback(i + 1, n_repeats)

    repeats_table = pd.DataFrame(repeat_rows)

    ok_table = repeats_table[repeats_table[t('repeated_holdout.status')] == "ok"].copy()

    metric_cols = [
        "Train R²",
        "Train RMSE",
        "Train MAE",
        "Train MAPE, %",
        "Test R²",
        "Test RMSE",
        "Test MAE",
        "Test MAPE, %",
    ]

    summary_rows = []

    for col in metric_cols:
        values = pd.to_numeric(ok_table[col], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(values) > 0:
            summary_rows.append({
                t('repeated_holdout.summary_metric'): col,
                "Mean": float(values.mean()),
                "Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "Median": float(values.median()),
                "Min": float(values.min()),
                "Max": float(values.max()),
                "N valid": int(len(values)),
            })
        else:
            summary_rows.append({
                t('repeated_holdout.summary_metric'): col,
                "Mean": np.nan,
                "Std": np.nan,
                "Median": np.nan,
                "Min": np.nan,
                "Max": np.nan,
                "N valid": 0,
            })

    summary_table = pd.DataFrame(summary_rows)

    test_r2_values = pd.to_numeric(
        ok_table["Test R²"],
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    test_rmse_values = pd.to_numeric(
        ok_table["Test RMSE"],
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(test_r2_values) > 0:
        test_r2_mean = float(test_r2_values.mean())
        test_r2_std = float(test_r2_values.std(ddof=1)) if len(test_r2_values) > 1 else 0.0
    else:
        test_r2_mean = np.nan
        test_r2_std = np.nan

    if len(test_rmse_values) > 0:
        test_rmse_mean = float(test_rmse_values.mean())
        test_rmse_std = float(test_rmse_values.std(ddof=1)) if len(test_rmse_values) > 1 else 0.0
    else:
        test_rmse_mean = np.nan
        test_rmse_std = np.nan

    if pd.notna(test_r2_mean):
        if test_r2_mean >= 0.7 and test_r2_std <= 0.15:
            conclusion = t('repeated_holdout.conclusion_stable')
        elif test_r2_mean >= 0.5 and test_r2_std <= 0.25:
            conclusion = t('repeated_holdout.conclusion_moderate')
        elif test_r2_mean >= 0.3:
            conclusion = t('repeated_holdout.conclusion_weak')
        else:
            conclusion = t('repeated_holdout.conclusion_low')
    else:
        conclusion = t('repeated_holdout.conclusion_undefined')

    if len(ok_table) == 0:
        conclusion = t('repeated_holdout.conclusion_all_failed')

    combined_train_table = (
        pd.concat(all_train_tables, ignore_index=True)
        if all_train_tables
        else pd.DataFrame()
    )

    combined_test_table = (
        pd.concat(all_test_tables, ignore_index=True)
        if all_test_tables
        else pd.DataFrame()
    )

    result = {
        "model_name": model_name,
        "target_col": target_col,
        "n_repeats": int(n_repeats),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "n_ok": int(len(ok_table)),
        "n_failed": int(n_repeats - len(ok_table)),
        "test_r2_mean": test_r2_mean,
        "test_r2_std": test_r2_std,
        "test_rmse_mean": test_rmse_mean,
        "test_rmse_std": test_rmse_std,
        "conclusion": conclusion,
        "summary_table": summary_table,
        "repeats_table": repeats_table,
        "combined_train_table": combined_train_table,
        "combined_test_table": combined_test_table,
    }

    return result

def store_descriptor_bundle(bundle, source_label):
    """Единая запись дескрипторов в session_state."""
    st.session_state.X_all = bundle["X_all"]
    st.session_state.y_all = bundle["y_all"]
    st.session_state.valid_indices = bundle["valid_indices"]
    st.session_state.desc_names = bundle["desc_names"]
    st.session_state.df_desc = bundle["df_desc"]

    st.session_state.desc_calculated = True
    st.session_state.custom_descriptors_used = source_label not in ["molecular_calculated"]
    st.session_state.custom_descriptor_source = source_label

    st.session_state.validation_done = False
    st.session_state.holdout_results_dict = {}
    st.session_state.kfold_results_dict = {}
    st.session_state.loo_results_dict = {}
    st.session_state.trained_models = {}

def qspr_calculate_leverage_ad(X_train, X_query=None, desc_names=None):
    """
    Applicability Domain через leverage.

    X_train — матрица обучающих дескрипторов.
    X_query — матрица веществ, для которых считаем leverage.
              Если None, считаем для самой обучающей выборки.

    Возвращает:
    {
        "leverage": ndarray,
        "threshold": float,
        "p": int,
        "n": int,
        "status": list[str]
    }
    """
    X_train = np.asarray(X_train, dtype=float)

    if X_query is None:
        X_query = X_train
    else:
        X_query = np.asarray(X_query, dtype=float)

    if X_train.ndim != 2:
        raise ValueError(t('ad_leverage.train_2d'))

    if X_query.ndim != 2:
        raise ValueError(t('ad_leverage.query_2d'))

    n, p = X_train.shape

    if n < 2:
        raise ValueError(t('ad_leverage.min_compounds'))

    if X_query.shape[1] != p:
        raise ValueError(t('ad_leverage.dim_mismatch', query_dim=X_query.shape[1], train_dim=p))

    # Добавляем свободный член, поэтому p_eff = p + 1.
    X_train_aug = np.column_stack([
        np.ones(n),
        X_train
    ])

    X_query_aug = np.column_stack([
        np.ones(X_query.shape[0]),
        X_query
    ])

    xtx_inv = np.linalg.pinv(X_train_aug.T @ X_train_aug)

    leverage = np.sum(
        (X_query_aug @ xtx_inv) * X_query_aug,
        axis=1
    )

    p_eff = p + 1
    threshold = 3.0 * p_eff / n

    # Формально h не может быть больше 1 для обучающей OLS-H,
    # но при p >> n и псевдообратной матрице порог может быть > 1.
    # Для отображения оставляем классическую формулу.
    status = [
        t('ad_leverage.in_ad') if h <= threshold else t('ad_leverage.out_ad')
        for h in leverage
    ]

    return {
        "leverage": leverage,
        "threshold": float(threshold),
        "p": int(p),
        "n": int(n),
        "status": status
    }


def qspr_make_ad_table(
    X_train,
    smiles,
    y=None,
    original_indices=None,
    desc_names=None
):
    """
    Таблица Applicability Domain для обучающей выборки.
    """
    ad = qspr_calculate_leverage_ad(
        X_train=X_train,
        X_query=None,
        desc_names=desc_names
    )

    n = len(ad["leverage"])

    table = pd.DataFrame({
        "№": range(1, n + 1),
        "Leverage h": ad["leverage"],
        t('ad_table.col_threshold'): ad["threshold"],
        t('ad_table.col_status'): ad["status"],
    })

    if original_indices is not None:
        table.insert(1, t('ad_table.col_original_index'), [int(i) + 1 for i in original_indices])

    if smiles is not None:
        insert_pos = 2 if original_indices is not None else 1
        table.insert(insert_pos, "SMILES", list(smiles))

    if y is not None:
        table[t('ad_table.col_property')] = y

    table[t('ad_table.col_reliability')] = table[t('ad_table.col_status')].map({
        t('ad_leverage.in_ad'): t('ad_table.reliability_in'),
        t('ad_leverage.out_ad'): t('ad_table.reliability_out')
    })

    return table, ad

def descriptor_source_message():
    source = st.session_state.get("custom_descriptor_source", "molecular_calculated")
    n_desc = len(st.session_state.get("desc_names", []))
    n_obj = len(st.session_state.get("y_all", [])) if st.session_state.get("y_all") is not None else 0

    if source == "spectral_only":
        st.info(t('desc_source.spectral_only', n_desc=n_desc, n_obj=n_obj))
    elif source == "molecular_plus_spectral":
        st.info(t('desc_source.molecular_plus_spectral', n_desc=n_desc, n_obj=n_obj))
    elif source == "molecular_only":
        st.info(t('desc_source.molecular_only', n_desc=n_desc, n_obj=n_obj))
    elif source == "xtb_quantum_descriptors":
        st.info(t('desc_source.xtb_quantum', n_desc=n_desc, n_obj=n_obj))
    elif source == "molecular_plus_xtb":
        st.info(t('desc_source.molecular_plus_xtb', n_desc=n_desc, n_obj=n_obj))
    elif source == "molecular_xtb_plus_spectral":
        st.info(t('desc_source.molecular_xtb_spectral', n_desc=n_desc, n_obj=n_obj))
    elif source == "morfeus_3d_descriptors":
        st.info(t('desc_source.morfeus_3d', n_desc=n_desc, n_obj=n_obj))
    elif "morfeus" in str(source):
        st.info(t('desc_source.morfeus_combined', n_desc=n_desc, n_obj=n_obj))
    elif source == "custom_descriptors":
        st.info(t('desc_source.custom', n_desc=n_desc, n_obj=n_obj))
    else:
        st.info(t('desc_source.default', n_desc=n_desc, n_obj=n_obj))
    
def qspr_guess_descriptor_source(desc_name):
    """
    Грубо определяет источник дескриптора по имени.
    """
    name = str(desc_name)

    if name.startswith("xtb_"):
        return "xTB"
        
    if name.startswith("morfeus_"):
        return "morfeus"
        
    if name.startswith("SPEC_") or name.startswith("spectral_"):
        return "Spectra"

    if name.startswith("IR_") or name.startswith("Mass_"):
        return "Spectra"

    if name.startswith("MACCS") or name.startswith("PubchemFP") or name.startswith("SubFP"):
        return "PaDEL"

    return "Molecular"


def qspr_make_descriptor_meaning_table(desc_names, status_label=""):
    """
    Делает таблицу:
    дескриптор -> расшифровка -> источник -> статус.
    Использует descriptor_meanings.json через qspr_load_descriptor_meanings().
    """
    desc_names = list(desc_names or [])

    meanings = qspr_load_descriptor_meanings()

    rows = []

    for i, desc_name in enumerate(desc_names, start=1):
        desc_str = str(desc_name)

        rows.append({
            "№": i,
            t('desc_meaning_table.col_descriptor'): desc_str,
            t('desc_meaning_table.col_meaning'): meanings.get(desc_str, t('desc_meaning_table.no_meaning')),
            t('desc_meaning_table.col_source'): qspr_guess_descriptor_source(desc_str),
            t('desc_meaning_table.col_status'): status_label,
        })

    return pd.DataFrame(rows)


def qspr_show_descriptor_meaning_table(
    desc_names,
    title=t('desc_meaning_table.default_title'),
    status_label="",
    expanded=False,
    key_prefix="descriptor_meanings"
):
    """
    Показывает таблицу расшифровки дескрипторов и кнопку скачивания CSV.
    """
    desc_names = list(desc_names or [])

    if not desc_names:
        return

    meaning_df = qspr_make_descriptor_meaning_table(
        desc_names=desc_names,
        status_label=status_label
    )

    with st.expander(title, expanded=expanded):
        st.caption(t('desc_meaning_table.caption'))

        st.dataframe(
            meaning_df,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            t('desc_meaning_table.download_button'),
            meaning_df.to_csv(index=False).encode("utf-8-sig"),
            f"{key_prefix}.csv",
            "text/csv",
            key=f"download_{key_prefix}"
        )

# ------------------------------------------------------------------
# Data bank


def load_data_bank():
    if os.path.exists(DATA_BANK_FILE):
        df = pd.read_csv(DATA_BANK_FILE)

        if "SMILES" not in df.columns:
            df["SMILES"] = ""

        df = df.drop_duplicates(subset=["SMILES"], keep="first")
        return df

    return pd.DataFrame(columns=["SMILES"])


def save_data_bank(df):
    df.to_csv(DATA_BANK_FILE, index=False)
    st.success(t('data_bank.saved', file=DATA_BANK_FILE))


def manage_data_bank():
    st.subheader(t('data_bank.header'))
    st.markdown(t('data_bank.description'))

    bank_df = load_data_bank()

    col1, col2 = st.columns([3, 1])

    with col1:
        props = [c for c in bank_df.columns if c != "SMILES"]
        st.write(t('data_bank.current', n=len(bank_df), props=', '.join(props) or t('data_bank.none')))

    with col2:
        if st.button(t('data_bank.refresh_button')):
            st.rerun()

    uploaded_bank = st.file_uploader(
        t('data_bank.upload_prompt'),
        type=["csv"],
        key="bank_upload"
    )

    if uploaded_bank is not None:
        new_data = pd.read_csv(uploaded_bank)
        new_data.columns = new_data.columns.str.strip()

        if "SMILES" not in new_data.columns:
            st.error(t('data_bank.no_smiles_column'))
        else:
            new_data["SMILES"] = new_data["SMILES"].astype(str).str.strip()
            new_data = new_data[new_data["SMILES"] != ""]

            bank_df_idx = bank_df.set_index("SMILES")
            new_data_idx = new_data.set_index("SMILES")

            for col in new_data_idx.columns:
                if col in bank_df_idx.columns:
                    common_idx = bank_df_idx.index.intersection(new_data_idx.index)
                    bank_df_idx.loc[common_idx, col] = new_data_idx.loc[common_idx, col]

                    new_idx = new_data_idx.index.difference(bank_df_idx.index)

                    if len(new_idx) > 0:
                        bank_df_idx = pd.concat([bank_df_idx, new_data_idx.loc[new_idx, [col]]], axis=0)
                else:
                    bank_df_idx[col] = new_data_idx[col]

            bank_df_idx.reset_index(inplace=True)
            bank_df_idx = bank_df_idx.drop_duplicates(subset=["SMILES"], keep="last")
            save_data_bank(bank_df_idx)
            st.success(t('data_bank.updated_success'))
            st.rerun()

    if st.checkbox(t('data_bank.show_content_checkbox')):
        st.dataframe(bank_df.reset_index(drop=True), width="stretch")
        csv_bank = bank_df.to_csv(index=False).encode("utf-8")
        st.download_button(t('data_bank.download_button'), csv_bank, "data_bank.csv", "text/csv")

    return bank_df


# ------------------------------------------------------------------
# SAOD UI table helpers


def saod2_ru_table(df):
    if df is None or df.empty:
        return df

    rename_map = {
        "compound_id": t('saod2_ru.compound_id'),
        "name": t('saod2_ru.name'),
        "input_smiles": t('saod2_ru.input_smiles'),
        "canonical_smiles": t('saod2_ru.canonical_smiles'),
        "inchikey": "InChIKey",
        "structure_status": t('saod2_ru.structure_status'),
        "valid_structure": t('saod2_ru.valid_structure'),
        "molecular_formula": t('saod2_ru.molecular_formula'),
        "molecular_weight": t('saod2_ru.molecular_weight'),
        "carbon_count": t('saod2_ru.carbon_count'),
        "atom_count": t('saod2_ru.atom_count'),
        "ring_count": t('saod2_ru.ring_count'),
        "is_hydrocarbon": t('saod2_ru.is_hydrocarbon'),
        "is_acyclic": t('saod2_ru.is_acyclic'),
        "is_saturated": t('saod2_ru.is_saturated'),
        "is_acyclic_alkane": t('saod2_ru.is_acyclic_alkane'),
        "exact_pattern": t('saod2_ru.exact_pattern'),
        "substituent_summary": t('saod2_ru.substituent_summary'),
        "longest_carbon_chain": t('saod2_ru.longest_carbon_chain'),
        "branch_count": t('saod2_ru.branch_count'),
        "branching_index": t('saod2_ru.branching_index'),
        "property_value": t('saod2_ru.property_value'),
        "duplicate_structure": t('saod2_ru.duplicate_structure'),
        "duplicate_conflict": t('saod2_ru.duplicate_conflict'),
        "series_size": t('saod2_ru.series_size'),
        "formula_group_size": t('saod2_ru.formula_group_size'),
        "raw_edges_total": t('saod2_ru.raw_edges_total'),
        "trusted_edges_total": t('saod2_ru.trusted_edges_total'),
        "broken_trusted_edges": t('saod2_ru.broken_trusted_edges'),
        "has_same_pattern_neighbors": t('saod2_ru.has_same_pattern_neighbors'),
        "has_formula_isomers": t('saod2_ru.has_formula_isomers'),
        "has_positional_analogs": t('saod2_ru.has_positional_analogs'),
        "series_checkability": t('saod2_ru.series_checkability'),
        "formula_checkability": t('saod2_ru.formula_checkability'),
        "network_checkability": t('saod2_ru.network_checkability'),
        "overall_checkability": t('saod2_ru.overall_checkability'),
        "checkability_score": t('saod2_ru.checkability_score'),
        "checkability_level": t('saod2_ru.checkability_level'),
        "dataset_noise_risk": t('saod2_ru.dataset_noise_risk'),
        "checkability_comment": t('saod2_ru.checkability_comment'),
        "checkability_recommendation": t('saod2_ru.checkability_recommendation'),
        "edge_label": t('saod2_ru.edge_label'),
        "formula": t('saod2_ru.formula'),
        "pattern_a": t('saod2_ru.pattern_a'),
        "pattern_b": t('saod2_ru.pattern_b'),
        "family_a": t('saod2_ru.family_a'),
        "family_b": t('saod2_ru.family_b'),
        "compound_a_id": t('saod2_ru.compound_a_id'),
        "compound_b_id": t('saod2_ru.compound_b_id'),
        "name_a": t('saod2_ru.name_a'),
        "name_b": t('saod2_ru.name_b'),
        "smiles_a": "SMILES A",
        "smiles_b": "SMILES B",
        "value_a": t('saod2_ru.value_a'),
        "value_b": t('saod2_ru.value_b'),
        "delta_a_minus_b": "Δ = A − B",
        "expected_delta": t('saod2_ru.expected_delta'),
        "delta_residual": t('saod2_ru.delta_residual'),
        "delta_change_to_previous": t('saod2_ru.delta_change_to_previous'),
        "delta_change_observation": t('saod2_ru.delta_change_observation'),
        "residual_score": t('saod2_ru.residual_score'),
        "change_score": t('saod2_ru.change_score'),
        "combined_edge_score": t('saod2_ru.combined_edge_score'),
        "edge_level": t('saod2_ru.edge_level'),
        "main_direction": t('saod2_ru.main_direction'),
        "sign_break": t('saod2_ru.sign_break'),
        "edge_status": t('saod2_ru.edge_status'),
        "n_points": t('saod2_ru.n_points'),
        "carbon_range": t('saod2_ru.carbon_range'),
        "mean_delta": t('saod2_ru.mean_delta'),
        "median_delta": t('saod2_ru.median_delta'),
        "sign_consistency": t('saod2_ru.sign_consistency'),
        "trend_slope": t('saod2_ru.trend_slope'),
        "trend_method": t('saod2_ru.trend_method'),
        "max_edge_score": t('saod2_ru.max_edge_score'),
        "break_count": t('saod2_ru.break_count'),
        "sign_break_count": t('saod2_ru.sign_break_count'),
        "rule_status": t('saod2_ru.rule_status'),
        "can_be_used_for_checking": t('saod2_ru.can_be_used_for_checking'),
        "broken_formulas": t('saod2_ru.broken_formulas'),
        "formula_examples": t('saod2_ru.formula_examples'),
        "property_value_example": t('saod2_ru.property_value_example'),
        "edges_total": t('saod2_ru.edges_total'),
        "edges_broken": t('saod2_ru.edges_broken'),
        "broken_fraction": t('saod2_ru.broken_fraction'),
        "broken_edge_labels": t('saod2_ru.broken_edge_labels'),
        "final_status": t('saod2_ru.final_status'),
        "human_observation": t('saod2_ru.human_observation'),
        "previous_formula": t('saod2_ru.previous_formula'),
        "previous_delta": t('saod2_ru.previous_delta'),
        "delta_delta_to_previous_recalculated": t('saod2_ru.delta_delta_to_previous_recalculated'),
        "mean_delta_within_formula": t('saod2_ru.mean_delta_within_formula'),
        "min_delta_within_formula": t('saod2_ru.min_delta_within_formula'),
        "max_delta_within_formula": t('saod2_ru.max_delta_within_formula'),
        "delta_spread_within_formula": t('saod2_ru.delta_spread_within_formula'),
        "n_raw_comparisons": t('saod2_ru.n_raw_comparisons'),
        "ambiguity_note": t('saod2_ru.ambiguity_note'),
    }

    return df.rename(columns=rename_map)


def saod2_show_table(df, height=None):
    if df is None or df.empty:
        st.info(t('saod2_show_table.empty'))
        return

    display_df = saod2_ru_table(df).copy()

    for bad_col in ["index", "level_0"]:
        if bad_col in display_df.columns:
            display_df = display_df.drop(columns=[bad_col])

    if "№" not in display_df.columns:
        display_df.insert(0, "№", range(1, len(display_df) + 1))

    styled = (
        display_df
        .style
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ])
    )

    try:
        styled = styled.hide(axis="index")
    except Exception:
        pass

    if height is None:
        st.dataframe(styled, width="stretch")
    else:
        st.dataframe(styled, width="stretch", height=height)

def saod2_make_review_dataset(original_df, processed, checkability, suspicion):
    """
    Формирует главный SAOD-файл для ручной проверки:
    все вещества исходного файла + статусы SAOD + колонки для решения.
    """
    review = original_df.copy().reset_index(drop=True)
    review["SAOD_row_id"] = range(len(review))

    proc = processed.copy().reset_index(drop=True)

    useful_processed_cols = [
        "canonical_smiles",
        "inchikey",
        "structure_status",
        "valid_structure",
        "molecular_formula",
        "carbon_count",
        "is_acyclic_alkane",
        "exact_pattern",
        "property_value",
        "duplicate_structure",
        "duplicate_conflict",
    ]

    useful_processed_cols = [
        c for c in useful_processed_cols
        if c in proc.columns
    ]

    proc_small = proc[useful_processed_cols].copy()
    proc_small["SAOD_row_id"] = range(len(proc_small))

    review = review.merge(
        proc_small,
        on="SAOD_row_id",
        how="left",
        suffixes=("", "_saod")
    )

    if checkability is not None and not checkability.empty:
        chk = checkability.copy()

        chk_cols = [
            "canonical_smiles",
            "overall_checkability",
            "checkability_score",
            "checkability_level",
            "dataset_noise_risk",
            "checkability_comment",
            "checkability_recommendation",
        ]

        chk_cols = [c for c in chk_cols if c in chk.columns]

        if "canonical_smiles" in chk_cols:
            chk_small = chk[chk_cols].drop_duplicates(
                subset=["canonical_smiles"],
                keep="first"
            )

            review = review.merge(
                chk_small,
                on="canonical_smiles",
                how="left",
                suffixes=("", "_checkability")
            )

    if suspicion is not None and not suspicion.empty:
        susp = suspicion.copy()

        susp_cols = [
            "canonical_smiles",
            "final_status",
            "edges_total",
            "edges_broken",
            "broken_fraction",
            "broken_edge_labels",
            "human_observation",
        ]

        susp_cols = [c for c in susp_cols if c in susp.columns]

        if "canonical_smiles" in susp_cols:
            susp_small = susp[susp_cols].drop_duplicates(
                subset=["canonical_smiles"],
                keep="first"
            )

            review = review.merge(
                susp_small,
                on="canonical_smiles",
                how="left",
                suffixes=("", "_suspicion")
            )

    def auto_decision(row):
        if row.get("valid_structure") is False:
            return t('saod2_review.auto_exclude_invalid')

        if bool(row.get("duplicate_conflict", False)):
            return t('saod2_review.auto_check_exclude_conflict')

        status = str(row.get("final_status", "")).strip().lower()

        if status in [
            t('saod2_review.status_critical'),
            t('saod2_review.status_highly_suspicious'),
            t('saod2_review.status_needs_check'),
        ]:
            return t('saod2_review.auto_manual_check')

        return t('saod2_review.auto_keep')

    review["SAOD_auto_recommendation"] = review.apply(auto_decision, axis=1)

    review["SAOD_manual_decision"] = review["SAOD_auto_recommendation"].map({
        t('saod2_review.auto_keep'): t('saod2_review.manual_keep'),
        t('saod2_review.auto_manual_check'): t('saod2_review.manual_keep'),
        t('saod2_review.auto_check_exclude_conflict'): t('saod2_review.manual_exclude'),
        t('saod2_review.auto_exclude_invalid'): t('saod2_review.manual_exclude'),
    }).fillna(t('saod2_review.manual_keep'))

    review["SAOD_manual_comment"] = ""

    first_cols = [
        "SAOD_row_id",
        "SAOD_auto_recommendation",
        "SAOD_manual_decision",
        "SAOD_manual_comment",
    ]

    other_cols = [c for c in review.columns if c not in first_cols]

    return review[first_cols + other_cols]


def saod2_excel_bytes(report_tables):
    """
    Создаёт Excel-файл SAOD в памяти.
    report_tables = dict(sheet_name -> DataFrame)
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in report_tables.items():
            if df_sheet is None:
                continue

            safe_sheet = str(sheet_name)[:31]

            if isinstance(df_sheet, pd.DataFrame):
                df_sheet.to_excel(writer, sheet_name=safe_sheet, index=False)

    output.seek(0)

    return output.getvalue()


def saod2_filter_dataset_by_review(review_df, decision_col="SAOD_manual_decision"):
    """
    Возвращает очищенный датасет после SAOD.
    Оставляет строки, где решение не начинается с 'Исключить'.
    """
    work = review_df.copy()

    if decision_col not in work.columns:
        raise ValueError(t('saod2_filter.no_decision_column', col=decision_col))

    exclude_keyword = t('saod2_review.manual_exclude').lower()
    keep_mask = ~work[decision_col].astype(str).str.lower().str.startswith(exclude_keyword)

    cleaned = work.loc[keep_mask].copy()

    technical_cols = [
        "SAOD_auto_recommendation",
        "SAOD_manual_decision",
        "SAOD_manual_comment",
        "canonical_smiles",
        "inchikey",
        "structure_status",
        "valid_structure",
        "molecular_formula",
        "carbon_count",
        "is_acyclic_alkane",
        "exact_pattern",
        "property_value",
        "duplicate_structure",
        "duplicate_conflict",
        "overall_checkability",
        "checkability_score",
        "checkability_level",
        "dataset_noise_risk",
        "checkability_comment",
        "checkability_recommendation",
        "final_status",
        "edges_total",
        "edges_broken",
        "broken_fraction",
        "broken_edge_labels",
        "human_observation",
    ]

    # SAOD_row_id оставим как служебную связь с отчётом, но можно удалить.
    cols_to_drop = [c for c in technical_cols if c in cleaned.columns]
    cleaned = cleaned.drop(columns=cols_to_drop, errors="ignore")

    return cleaned.reset_index(drop=True)

# ------------------------------------------------------------------
# Session state

SESSION_DEFAULTS = {
    "data": None,
    "target_col": None,
    "desc_calculated": False,
    "X_all": None,
    "y_all": None,
    "valid_indices": None,
    "desc_names": None,
    "df_desc": None,
    "logs": [],
    "log_events": [],
    "desc_lists": None,
    "descriptor_source_mode": "calculate",
    "custom_descriptor_cols": [],
    "custom_descriptors_used": False,
    "custom_descriptor_source": "molecular_calculated",
    "descriptor_calculation_mode": "👁️‍🗨️ Расширенный (Mordred)",
    "molecular_descriptor_calculation_mode": "👁️‍🗨️ Расширенный (Mordred)",
    "validation_done": False,
    "last_model_group": MODEL_GROUP_LINEAR,
    "last_model_algorithm": "Random Forest",
    "pls_components": 2,
    "ridge_alpha": 1.0,
    "lasso_alpha": 0.01,
    "elastic_alpha": 0.01,
    "elastic_l1_ratio": 0.5,
    "rf_n_estimators": 300,
    "lightgbm_n_estimators": 300,
    "lightgbm_learning_rate": 0.05,
    "lightgbm_num_leaves": 31,
    "catboost_iterations": 300,
    "catboost_learning_rate": 0.05,
    "catboost_depth": 6,
    "stacking_cv": 5,
    "stacking_passthrough": False,
    "trained_models": {},
    "model_used_descriptor_names": [],
    "model_used_descriptor_model_name": "",
    "holdout_results_dict": {},
    "kfold_results_dict": {},
    "loo_results_dict": {},
    "repeated_holdout_results_dict": {},
    "y_randomization_results_dict": {},
    "svr_c": 10.0,
    "svr_epsilon": 0.1,
    "svr_gamma": "scale",
    "gpr_alpha": 0.000001,
    "gpr_length_scale": 1.0,
    "gpr_noise_level": 0.1,
    "knn_n_neighbors": 5,
    "knn_weights": "distance",
    "mlp_hidden_layer_sizes": "100,50",
    "mlp_activation": "relu",
    "mlp_alpha": 0.0001,
    "mlp_learning_rate_init": 0.001,
    "mlp_max_iter": 2000,
    "cart_max_depth": 5,
    "cart_min_samples_leaf": 2,
    "mars_degree": 2,
    "mars_alpha": 1.0,
    "spline_n_knots": 5,
    "spline_degree": 3,
    "spline_alpha": 1.0,
    "gam_n_splines": 6,
    "gam_degree": 3,
    "gam_alpha": 1.0,
    "gep_population_size": 500,
    "gep_generations": 20,
    "gep_max_depth": 4,
    "gp_population_size": 500,
    "gp_generations": 20,
    "gp_max_depth": 4,
    "pysr_niterations": 40,
    "pysr_populations": 8,
    "pysr_maxsize": 20,
    "voting_rf_weight": 1.0,
    "voting_extra_trees_weight": 1.0,
    "voting_ridge_weight": 1.0,
    "auto_feature_selection": False,
    "auto_hyperparameter_optimization": False,
    "auto_feature_selection_method": "fast",
    "auto_max_features": 50,
    "auto_cv": 5,
    "auto_search_method": "grid",
    "auto_remove_constant_descriptors": True,
    "auto_remove_correlated_descriptors": True,
    "auto_corr_threshold": 0.95,
    "auto_lasso_selection_alpha": 0.01,
    "auto_rf_selection_estimators": 300,
    "auto_rfe_step": 0.2,
    "auto_tuning_result": None,
    "model_comparison_df": None,
    "best_model_from_comparison": None,
    "pending_selected_model": None,
    "saod2_review_df": None,
    "saod2_cleaned_df": None,
    "saod2_cleaning_applied": False,
    "saod2_original_before_cleaning": None,
    "data_source_note": "",
    "incremental_result": None,
    "incremental_cols": [],
    "incremental_use_intercept": True,
    "struct_filter_result_df": None,
    "struct_filter_applied": False,
    "struct_filter_note": "",
    "xtb_descriptors_df": None,
    "xtb_descriptors_report": None,
    "et_n_estimators": 300,
    "et_max_depth": None,
    "et_min_samples_split": 2,
    "et_min_samples_leaf": 1,
    "et_max_features": "sqrt",
    "ext_validation_result": None,
    "adaboost_n_estimators": 300,
    "adaboost_learning_rate": 1.0,
    "methodology_history": [],
    "methodology_current_index": 0,
    "methodology_language": "ru",
    "methodology_style": "full",
    "ad_info": None,   # если ещё нет, добавим для хранения данных AD
    "report_full_history": [],
    "report_full_current_index": 0,
    "report_language": "ru",
    "hgb_max_iter": 300,
    "hgb_learning_rate": 0.1,
    "hgb_max_depth": None,
    "hgb_min_samples_leaf": 20,
    "hgb_l2_regularization": 0.0
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

def reset_project_state_for_new_file():
    """
    Полный сброс состояния приложения при загрузке нового файла.
    Оставляет только настройки интерфейса и сам новый файл.
    """

    keys_to_reset = [
        # Данные и цель
        "target_col",
        "data_source_note",

        # Дескрипторы
        "desc_calculated",
        "X_all",
        "y_all",
        "valid_indices",
        "desc_names",
        "df_desc",
        "custom_descriptors_used",
        "custom_descriptor_source",
        "custom_descriptor_cols",

        # Модели и валидация
        "validation_done",
        "trained_models",
        "model_used_descriptor_names",
        "model_used_descriptor_model_name",
        "holdout_results_dict",
        "kfold_results_dict",
        "loo_results_dict",
        "repeated_holdout_results_dict",
        "y_randomization_results_dict",
        "auto_tuning_result",
        "model_comparison_df",
        "best_model_from_comparison",
        "pending_selected_model",

        # Spectra
        "spectra_search_results",
        "spectral_descriptors_df",
        "spectral_descriptors_report",
        "spectral_descriptors_saved_path",
        "spectral_qspr_match_info",

        # SAOD
        "saod2_result",
        "saod2_review_df",
        "saod2_cleaned_df",
        "saod2_cleaning_applied",
        "saod2_original_before_cleaning",

        # Структурный фильтр
        "struct_filter_result_df",
        "struct_filter_applied",
        "struct_filter_note",

        # Пользовательские дескрипторы / МНК
        "incremental_result",
        "incremental_cols",
        
        # xTB
        "xtb_descriptors_df",
        "xtb_descriptors_report",

        # Прогностическая модель
        "prog_model",
        "prog_scaler",
        "prog_predictions",
        "prediction_uncertainty_result",
        "standalone_prediction_uncertainty_result",
    ]

    for key in keys_to_reset:
        if key in SESSION_DEFAULTS:
            st.session_state[key] = SESSION_DEFAULTS[key]
        elif key in st.session_state:
            del st.session_state[key]
            
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Sidebar

st.title(t('app_title'))

qspr_show_online_demo_notice()

render_admin_login_controls()

legacy_app_mode = st.session_state.get(
    "main_app_mode_code",
    st.session_state.get("main_app_mode"),
)

legacy_qspr_labels = {
    load_language(lang).get("mode", {}).get("qspr")
    for lang in ("ru", "en", "kk")
}
legacy_prediction_labels = {
    load_language(lang).get("mode", {}).get("prediction")
    for lang in ("ru", "en", "kk")
}

if legacy_app_mode in legacy_qspr_labels:
    st.session_state["main_app_mode_code"] = "qspr"
elif legacy_app_mode in legacy_prediction_labels:
    st.session_state["main_app_mode_code"] = "prediction"
elif legacy_app_mode not in {"qspr", "prediction"}:
    st.session_state["main_app_mode_code"] = "qspr"
else:
    st.session_state["main_app_mode_code"] = legacy_app_mode

if (not qspr_is_online_mode()) and not is_admin() and st.session_state.get("main_app_mode_code") == "prediction":
    show_admin_only_notice("prediction_mode")
    st.session_state["main_app_mode_code"] = "qspr"

app_mode_options = ["qspr", "prediction"] if (qspr_is_online_mode() or is_admin()) else ["qspr"]

app_mode = st.radio(
    t('mode.select'),
    app_mode_options,
    format_func=lambda mode: {
        "qspr": t('mode.qspr'),
        "prediction": t('mode.prediction'),
    }[mode],
    horizontal=True,
    key="main_app_mode_code",
)

if not is_admin() and not qspr_is_online_mode():
    st.caption(t("mode.prediction_admin_only"))

with st.sidebar:
    st.header(t('sidebar.title'))
    
    # --- Переключатель языка ---
    lang = st.selectbox(
        t('sidebar.language_select'),
        options=['ru', 'en', 'kk'],
        format_func=lambda x: {'ru': 'Русский', 'en': 'English', 'kk': 'Қазақша'}[x],
        index=['ru','en','kk'].index(st.session_state.lang),
        key='lang_selector'
    )
    if lang != st.session_state.lang:
        st.session_state.lang = lang
        st.session_state.lang_manual = True
        _remember_lang_in_url(lang)
        set_language(lang)
        st.rerun()
    # --- конец переключателя ---    

    if app_mode != "prediction":
        padel_unique_count = len(qspr_load_padel_unique_from_file())

        if padel_unique_count:
            st.success(t('sidebar.padel_unique_count', count=padel_unique_count))
        else:
            st.warning(t('sidebar.padel_warning'))

        if is_admin() and st.button(t('sidebar.update_desc_lists')):
            with st.spinner(t('sidebar.spinner_computing')):
                lists = qspr_compute_descriptor_lists()
                qspr_save_descriptor_lists(lists)
                st.session_state.desc_lists = lists
                add_log(t('sidebar.log_lists_updated',
                    rdkit=len(lists.get('rdkit_all', [])),
                    mordred=len(lists.get('mordred_unique', [])),
                    padel_all=len(lists.get('padel_all', [])),
                    padel_unique=len(lists.get('padel_unique', []))
                ))
                st.success(t('sidebar.lists_updated',
                    rdkit=len(lists.get('rdkit_all', [])),
                    mordred=len(lists.get('mordred_unique', [])),
                    padel_all=len(lists.get('padel_all', [])),
                    padel_unique=len(lists.get('padel_unique', []))
                ))

        if st.session_state.desc_lists is None:
            st.session_state.desc_lists = qspr_load_descriptor_lists()

        if st.session_state.desc_lists:
            st.info(t('sidebar.lists_loaded',
                rdkit=len(st.session_state.desc_lists.get('rdkit_all', [])),
                mordred=len(st.session_state.desc_lists.get('mordred_unique', [])),
                padel_fp=len(st.session_state.desc_lists.get('padel_fingerprints', [])),
                padel_1d2d=len(st.session_state.desc_lists.get('padel_1d2d', [])),
                padel_all=len(st.session_state.desc_lists.get('padel_all', [])),
                padel_unique=len(st.session_state.desc_lists.get('padel_unique', []))
            ))
        else:
            st.warning(t('sidebar.lists_not_created'))

        if is_admin():
            with st.expander(t('sidebar.show_log')):
                show_logs()

        st.divider()

        if is_admin():
            st.toggle(
                t('sidebar.show_data_bank'),
                value=False,
                key="show_data_bank_panel"
            )
        else:
            st.session_state.show_data_bank_panel = False

if app_mode == "prediction" and qspr_is_online_mode():
    st.header(t('mode.prediction'))
    qspr_online_lock_notice("Standalone prediction and model package loading")
    st.stop()

if app_mode == "prediction":
    qspr_show_standalone_prediction_page()
    st.stop()

st.header(t('header.qspr'))
st.markdown(t('data_upload.instruction'))

if app_mode != "prediction" and is_admin() and st.session_state.get("show_data_bank_panel", False):
    with st.expander(t('sidebar.manage_data_bank'), expanded=True):
        manage_data_bank()

# ------------------------------------------------------------------
# Data upload

uploaded_file = st.file_uploader(
    t('upload.prompt'),
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.lower().split(".")[-1]

    if file_ext == "csv":
        delimiter = st.radio(t('upload.delimiter_radio'), [",", ";"], index=1, horizontal=True)
    else:
        delimiter = None
        st.info(t('upload.xlsx_info'))

    uploaded_file_size = uploaded_file.size
    if qspr_is_online_mode() and uploaded_file_size > ONLINE_MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"Online demo accepts files up to {ONLINE_MAX_UPLOAD_MB} MB. "
            "Use the local version for larger datasets."
        )
        st.stop()

    file_changed = (
        st.session_state.get("uploaded_file_name") != uploaded_file.name or
        st.session_state.get("uploaded_file_size") != uploaded_file_size or
        st.session_state.get("delimiter") != delimiter
    )

    if file_changed:
        if file_ext == "csv":
            delimiter_label = st.selectbox(
                t('upload.delimiter_select'),
                [
                    t('upload.delimiter_auto'),
                    t('upload.delimiter_comma'),
                    t('upload.delimiter_semicolon'),
                    t('upload.delimiter_tab'),
                ],
                key="csv_delimiter_label"
            )

            delimiter_map = {
                t('upload.delimiter_comma'): ",",
                t('upload.delimiter_semicolon'): ";",
                t('upload.delimiter_tab'): "\t",
            }

            if delimiter_label == t('upload.delimiter_auto'):
                sep_value = None
            else:
                sep_value = delimiter_map[delimiter_label]
            try:
                if sep_value is None:
                    data = pd.read_csv(
                        uploaded_file,
                        sep=None,
                        engine="python",
                        on_bad_lines="skip"
                    )
                else:
                    data = pd.read_csv(
                        uploaded_file,
                        sep=sep_value,
                        engine="python",
                        on_bad_lines="skip"
                    )
            except Exception as e:
                st.error(t('upload.csv_read_error', error=e))
                st.stop()
    
        elif file_ext == "xlsx":
            data = pd.read_excel(uploaded_file)
        else:
            st.error(t('upload.unsupported_format'))
            st.stop()

        data.columns = data.columns.str.strip()

        if qspr_is_online_mode() and len(data) > ONLINE_MAX_DATA_ROWS:
            st.error(
                f"Online demo accepts up to {ONLINE_MAX_DATA_ROWS} rows. "
                "Use the local version for larger datasets."
            )
            st.stop()

        reset_project_state_for_new_file()

        st.session_state.data = data
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_size = uploaded_file_size
        st.session_state.delimiter = delimiter

        add_log(
            t('upload.file_loaded', name=uploaded_file.name, rows=len(data)),
            stage="data_upload",
            event="data_loaded",
            counts={
                "rows": len(data),
                "columns": len(data.columns),
            },
            context={
                "file": uploaded_file.name,
                "format": file_ext,
                "smiles_candidates": ", ".join([
                    c for c in data.columns
                    if str(c).lower() in {"smiles", "canonical_smiles"}
                ]),
            },
        )

        st.rerun()
        
if st.session_state.data is None:
    st.info(t('upload.example_header'))
    st.code("SMILES;BoilingPoint\nC;-161.5\nCC;-88.6\nCCC;-42.1", language="csv")
    sample_alkanes_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "файлы для проб",
        "153.xlsx",
    )
    if os.path.exists(sample_alkanes_path):
        with open(sample_alkanes_path, "rb") as sample_file:
            st.download_button(
                t("upload.sample_alkanes_download"),
                data=sample_file,
                file_name="153.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    st.stop()
# ------------------------------------------------------------------
# Basic data preparation

# ------------------------------------------------------------------
# Data is loaded: show data-preparation module header

st.header(t('data_prep.header'))

st.caption(t('data_prep.caption'))

data = st.session_state.data.copy()

possible_smiles_cols = [
    "SMILES",
    "smiles",
    "canonical_smiles",
    "Canonical SMILES",
    "canonical_SMILES",
    "can_smiles",
    "mol_smiles",
]

available_smiles_cols = [
    col for col in possible_smiles_cols
    if col in data.columns
]

if len(available_smiles_cols) == 0:
    st.error(t('data_prep.no_smiles_column'))
    st.stop()

if "canonical_smiles" in available_smiles_cols:
    default_smiles_col = "canonical_smiles"
else:
    default_smiles_col = available_smiles_cols[0]

st.markdown(
    f"""
    <span class="label-tooltip"
          title="{t('data_prep.smiles_tooltip')}">
        {t('data_prep.smiles_label')}
    </span>
    """,
    unsafe_allow_html=True
)

smiles_col_current = st.selectbox(
    t('data_prep.smiles_label'),
    options=available_smiles_cols,
    index=available_smiles_cols.index(default_smiles_col),
    key="smiles_col_current",
    label_visibility="collapsed"
)

st.session_state.data = data

smiles_values_current = data[smiles_col_current].astype(str).fillna("")
data = data[smiles_values_current.str.strip() != ""].copy()

if data.empty:
    st.error(t('data_prep.no_nonempty_smiles'))
    st.stop()

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

if "SMILES" in numeric_cols:
    numeric_cols.remove("SMILES")

if not numeric_cols:
    # Попробуем найти числовые строки с запятой
    numeric_like = []

    for col in data.columns:
        if col == "SMILES":
            continue

        converted = pd.to_numeric(
            data[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

        if converted.notna().mean() >= 0.7:
            numeric_like.append(col)

    numeric_cols = numeric_like

if not numeric_cols:
    st.error(t('data_prep.no_numeric_columns'))
    st.stop()

st.markdown(
    f"""
    <span class="label-tooltip"
          title="{t('data_prep.target_tooltip')}">
        {t('data_prep.target_label')}
    </span>
    """,
    unsafe_allow_html=True
)

# Если target_col ещё не задан или отсутствует в новом датасете
if (
    "target_col" not in st.session_state
    or st.session_state.target_col not in numeric_cols
):
    st.session_state.target_col = numeric_cols[0]

# Синхронизация ключа виджета
if (
    "target_col_select" not in st.session_state
    or st.session_state.target_col_select not in numeric_cols
):
    st.session_state.target_col_select = st.session_state.target_col

old_target = st.session_state.target_col

target_col = st.selectbox(
    t('data_prep.target_label'),
    numeric_cols,
    index=numeric_cols.index(st.session_state.target_col_select),
    key="target_col_select",
    label_visibility="collapsed"
)

if target_col != old_target:
    st.session_state.target_col = target_col

    st.session_state.desc_calculated = False
    st.session_state.validation_done = False
    st.session_state.X_all = None
    st.session_state.y_all = None
    st.session_state.valid_indices = None
    st.session_state.desc_names = None
    st.session_state.df_desc = None
    st.session_state.trained_models = {}
    st.session_state.holdout_results_dict = {}
    st.session_state.kfold_results_dict = {}
    st.session_state.loo_results_dict = {}

    add_log(
        t('data_prep.target_selected_log', target=target_col)
    )

    st.rerun()
    st.stop()
else:
    st.session_state.target_col = target_col

data[target_col] = pd.to_numeric(
    data[target_col].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

st.subheader(t('data_prep.data_subheader'))

if st.session_state.get("data_source_note"):
    st.success(st.session_state.data_source_note)

st.write(t('data_prep.current_dataset', count=len(data)))

possible_smiles_cols = [
    "SMILES",
    "smiles",
    "canonical_smiles",
    "Canonical SMILES",
    "canonical_SMILES",
    "can_smiles",
    "mol_smiles",
]

available_smiles_cols = [
    c for c in possible_smiles_cols
    if c in data.columns
]

if not available_smiles_cols:
    st.error(t('data_prep.no_smiles_column'))
    st.stop()

st.dataframe(data.head(), width="stretch")

show_molecule_viewer(
    data=data,
    target_col=target_col,
    smiles_col=smiles_col_current
)

st.markdown(
    '<div class="tool-badge">' + t('data_prep.tool_badge') + '</div>',
    unsafe_allow_html=True
)
st.markdown(t('data_prep.standardization_title'))

st.info(t('data_prep.standardization_info'))
    
st.caption(t('data_prep.standardization_caption'))

if st.button(
    t('standardization_ui.normalize_button'),
    type="primary",
    key="run_molecule_standardization"
):
    try:
        std_df, duplicate_removed_df, std_summary_df = qspr_standardize_molecule_dataset(
            input_df=data,
            smiles_col=smiles_col_current,
            target_col=target_col,
            remove_duplicates_by_inchikey=False
        )

        st.session_state.standardized_molecule_df = std_df
        st.session_state.standardized_duplicates_removed_df = duplicate_removed_df
        st.session_state.standardization_summary_df = std_summary_df

        add_log(
            t('standardization_ui.log_normalization',
            initial=len(data),
            final=len(std_df),
            duplicates=len(duplicate_removed_df)
        ))

        st.success(t('standardization_ui.success_normalization'))

    except Exception as e:
        st.error(t('standardization_ui.error_normalization', error=e))
        st.exception(e)

if isinstance(st.session_state.get("standardization_summary_df"), pd.DataFrame):
    std_summary_df = st.session_state.standardization_summary_df

    if not std_summary_df.empty:
        st.dataframe(
            std_summary_df,
            width="stretch",
            hide_index=True
        )

if isinstance(st.session_state.get("standardized_molecule_df"), pd.DataFrame):
    std_df = st.session_state.standardized_molecule_df
    std_df = std_df.loc[:, ~std_df.columns.duplicated()].copy()

    if not std_df.empty:
        preview_cols_raw = [
            "Номер исходной строки",
            smiles_col_current,
            "input_smiles_original",
            "standardized_smiles",
            "canonical_smiles",
            "inchikey",
            "standardization_status",
            "standardization_warnings",
            target_col,
        ]

        preview_cols = []

        for c in preview_cols_raw:
            if c in std_df.columns and c not in preview_cols:
                preview_cols.append(c)

        with st.expander(t('standardization_ui.preview_title'), expanded=True):
            st.dataframe(
                std_df[preview_cols].head(200),
                width="stretch",
                hide_index=True
            )

        warning_df = std_df[
            (std_df["standardization_status"] != "ok")
            | (std_df["standardization_warnings"].astype(str).str.strip() != "")
        ].copy()

        if not warning_df.empty:
            with st.expander(t('standardization_ui.warning_title'), expanded=False):
                st.dataframe(
                    warning_df[preview_cols].head(300),
                    width="stretch",
                    hide_index=True
                )

        if isinstance(st.session_state.get("standardized_duplicates_removed_df"), pd.DataFrame):
            std_duplicates_df = st.session_state.standardized_duplicates_removed_df

            if not std_duplicates_df.empty:
                duplicate_preview_cols = []

                for c in preview_cols_raw:
                    if c in std_duplicates_df.columns and c not in duplicate_preview_cols:
                        duplicate_preview_cols.append(c)
                st.caption(t('standardization_ui.duplicates_caption'))
                with st.expander(t('standardization_ui.duplicates_title'), expanded=False):
                    st.dataframe(
                        std_duplicates_df[duplicate_preview_cols].head(300),
                        width="stretch",
                        hide_index=True
                    )

        csv_std = std_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            t('standardization_ui.download_button'),
            csv_std,
            "standardized_molecules.csv",
            "text/csv",
            key="download_standardized_molecules"
        )

        if st.button(
            t('standardization_ui.apply_button'),
            type="primary",
            key="apply_standardized_molecules"
        ):
            try:
                # std_df — это отчёт стандартизации.
                # Его нельзя целиком делать рабочим датасетом.
                std_work = std_df.copy()
                std_work = std_work.loc[:, ~std_work.columns.duplicated()].copy()

                required_std_cols = [
                    "canonical_smiles",
                    "standardization_status",
                    "Номер исходной строки",
                ]

                missing_std_cols = [
                    c for c in required_std_cols
                    if c not in std_work.columns
                ]

                if missing_std_cols:
                    st.error(t('standardization_ui.error_missing_cols', cols=', '.join(missing_std_cols)))
                    st.stop()

                # Берём именно исходный рабочий датасет.
                # Он может содержать уже рассчитанные/загруженные дескрипторы.
                applied_df = data.copy()
                applied_df = applied_df.loc[:, ~applied_df.columns.duplicated()].copy()
                applied_df = applied_df.reset_index(drop=True)

                if smiles_col_current not in applied_df.columns:
                    st.error(t('standardization_ui.error_no_smiles_col', col=smiles_col_current))
                    st.stop()

                # В std_df номер исходной строки записан как 1, 2, 3...
                # Переводим его в позицию pandas: 0, 1, 2...
                std_work["_original_pos"] = pd.to_numeric(
                    std_work["Номер исходной строки"],
                    errors="coerce"
                ) - 1

                std_work = std_work[
                    std_work["_original_pos"].notna()
                ].copy()

                std_work["_original_pos"] = std_work["_original_pos"].astype(int)

                # Берём только успешно стандартизированные структуры.
                ok_std = std_work[
                    std_work["standardization_status"]
                    .astype(str)
                    .str.strip()
                    .eq("ok")
                ].copy()

                canonical_map = (
                    ok_std
                    .set_index("_original_pos")["canonical_smiles"]
                    .astype(str)
                    .to_dict()
                )

                # Заменяем SMILES только там, где есть успешная стандартизация.
                # Строки НЕ удаляем. Все дескрипторные колонки сохраняются.
                valid_positions = [
                    pos for pos in canonical_map.keys()
                    if 0 <= int(pos) < len(applied_df)
                ]

                if valid_positions:
                    applied_df.loc[valid_positions, smiles_col_current] = [
                        canonical_map[pos] for pos in valid_positions
                    ]

                # Финальная защита от дублей имён колонок.
                applied_df = applied_df.loc[:, ~applied_df.columns.duplicated()].copy()
                applied_df = applied_df.reset_index(drop=True)

                # Отчёт стандартизации сохраняем отдельно.
                # Он не должен попадать в рабочий data.
                st.session_state.standardized_molecules_df = std_work.copy()

                # Рабочий датасет:
                # все исходные колонки + заменённый SMILES.
                st.session_state.data = applied_df.copy()

                st.session_state.data_source_note = t('standardization_ui.data_source_note', count=len(applied_df))

                # Сбрасываем только результаты, которые зависят от структуры/модели.
                st.session_state.desc_calculated = False
                st.session_state.validation_done = False
                st.session_state.X_all = None
                st.session_state.y_all = None
                st.session_state.valid_indices = None
                st.session_state.desc_names = None
                st.session_state.df_desc = None
                st.session_state.trained_models = {}
                st.session_state.holdout_results_dict = {}
                st.session_state.kfold_results_dict = {}
                st.session_state.loo_results_dict = {}

                add_log(t('standardization_ui.apply_log', rows=len(applied_df), cols=len(applied_df.columns)))

                st.success(t('standardization_ui.apply_success'))

                st.rerun()

            except Exception as e:
                st.error(t('standardization_ui.apply_error', error=e))
                st.exception(e)

## ------------------------------------------------------------------
# Dataset passport

st.markdown(t('dataset_passport.title'))

try:
    source_filename = ""

    try:
        if uploaded_file is not None:
            source_filename = uploaded_file.name
    except Exception:
        source_filename = st.session_state.get("data_source_note", "")

    (
        dataset_passport_df,
        suspicious_values_df,
        duplicate_structures_df,
        passport_conflict_duplicates_df
    ) = qspr_make_dataset_passport(
        data=data,
        smiles_col=smiles_col_current,
        target_col=target_col,
        source_filename=source_filename
    )

    status_value = ""

    try:
        status_value = str(
            dataset_passport_df.loc[
                dataset_passport_df[t('passport.prompt')] == t('passport.final_status'),
                t('passport.value')
            ].iloc[0]
        )
    except Exception:
        status_value = ""

    if status_value.startswith("✅"):
        st.success(status_value)
    elif status_value.startswith("⚠️"):
        st.warning(status_value)

    passport_metric_map = {}

    for _, row in dataset_passport_df.iterrows():
        passport_metric_map[str(row[t('passport.prompt')])] = row[t('passport.value')]

    pass_col_1, pass_col_2, pass_col_3, pass_col_4 = st.columns(4)

    with pass_col_1:
        st.metric(t('passport.rows'), passport_metric_map.get(t('passport.rows'), "—"))

    with pass_col_2:
        st.metric(t('passport.valid_smiles'), passport_metric_map.get(t('passport.valid_smiles'), "—"))

    with pass_col_3:
        st.metric(t('passport.unique_structures'), passport_metric_map.get(t('passport.unique_structures'), "—"))

    with pass_col_4:
        st.metric(t('passport.conflict_duplicates'), passport_metric_map.get(t('passport.conflict_duplicates'), "—"))

    pass_col_5, pass_col_6, pass_col_7, pass_col_8 = st.columns(4)

    with pass_col_5:
        st.metric(t('passport.missing_values'), passport_metric_map.get(t('passport.missing_values'), "—"))

    with pass_col_6:
        st.metric(t('passport.min'), passport_metric_map.get(t('passport.min'), "—"))

    with pass_col_7:
        st.metric(t('passport.max'), passport_metric_map.get(t('passport.max'), "—"))

    with pass_col_8:
        st.metric(t('passport.suspicious_count'), passport_metric_map.get(t('passport.suspicious_count'), "—"))

    with st.expander(t('dataset_passport.full_passport'), expanded=True):
        st.dataframe(
            dataset_passport_df,
            width="stretch",
            hide_index=True
        )

    if isinstance(suspicious_values_df, pd.DataFrame) and not suspicious_values_df.empty:
        with st.expander(t('dataset_passport.suspicious_values'), expanded=False):
            st.dataframe(
                suspicious_values_df.head(300),
                width="stretch",
                hide_index=True
            )

    if isinstance(duplicate_structures_df, pd.DataFrame) and not duplicate_structures_df.empty:
        with st.expander(t('dataset_passport.duplicate_structures'), expanded=False):
            st.dataframe(
                duplicate_structures_df.head(300),
                width="stretch",
                hide_index=True
            )

    if isinstance(passport_conflict_duplicates_df, pd.DataFrame) and not passport_conflict_duplicates_df.empty:
        with st.expander(t('dataset_passport.conflict_duplicates'), expanded=False):
            st.dataframe(
                passport_conflict_duplicates_df.head(300),
                width="stretch",
                hide_index=True
            )

    # Эти переменные могут уже существовать, если выше вставлена расширенная диагностика.
    # Если их нет, Excel всё равно будет создан с паспортом и доступными листами.
    diagnostics_for_excel = locals().get("dataset_diagnostics_table", pd.DataFrame())
    classes_for_excel = locals().get("molecule_class_summary", pd.DataFrame())
    invalid_smiles_for_excel = locals().get("invalid_smiles_df", pd.DataFrame())

    passport_excel = qspr_make_dataset_passport_excel(
        passport_df=dataset_passport_df,
        diagnostics_df=diagnostics_for_excel,
        molecule_class_summary=classes_for_excel,
        suspicious_values_df=suspicious_values_df,
        duplicate_structures_df=duplicate_structures_df,
        conflict_duplicates_df=passport_conflict_duplicates_df,
        invalid_smiles_df=invalid_smiles_for_excel
    )

    st.download_button(
        t('dataset_passport.download_excel'),
        passport_excel,
        "dataset_passport.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_dataset_passport_excel"
    )

except Exception as e:
    st.warning(t('dataset_passport.error', error=e))

col_hist, col_box = st.columns(2)

with col_hist:
    fig_hist, ax_hist = plt.subplots(figsize=(4, 3))
    safe_histplot(ax_hist, data[target_col], kde=True, color='steelblue', edgecolor='black', alpha=0.7)
    ax_hist.set_title(t('dataset_passport.hist_title', col=target_col))
    st.pyplot(fig_hist)

with col_box:
    fig_box, ax_box = plt.subplots(figsize=(4, 2.5))
    sns.boxplot(x=data[target_col].dropna(), ax=ax_box)
    ax_box.set_title(t('dataset_passport.boxplot_title', col=target_col))
    st.pyplot(fig_box)

st.subheader(t('dataset_passport.diagnostics_title'))

diagnostics_found = False

duplicate_summary = (
    data
    .groupby(smiles_col_current)[target_col]
    .agg(
        n_records="size",
        n_unique_values=lambda x: x.dropna().nunique(),
        min_value="min",
        max_value="max"
    )
    .reset_index()
)

conflict_smiles = duplicate_summary[
    (duplicate_summary["n_records"] > 1) &
    (duplicate_summary["n_unique_values"] > 1)
].copy()

if not conflict_smiles.empty:
    diagnostics_found = True

    conflict_rows = data[
        data[smiles_col_current].isin(conflict_smiles[smiles_col_current])
    ].copy()

    conflict_rows.insert(
        0,
        t('dataset_passport.original_row'),
        conflict_rows.index + 1
    )

    st.warning(t('dataset_passport.conflict_warning', count=len(conflict_smiles)))

    st.dataframe(
        conflict_rows,
        width="stretch",
        hide_index=True
    )

    conflict_rows_for_view = conflict_rows.copy()
    conflict_rows_for_view["SMILES"] = conflict_rows_for_view[smiles_col_current]

    show_molecule_grid_from_table(
        table_df=conflict_rows_for_view,
        title=t('dataset_passport.conflict_structures_title'),
        target_col=target_col,
        smiles_col="SMILES",
        max_molecules=100,
        key_prefix="duplicate_smiles_conflicts"
    )

else:
    st.success(t('dataset_passport.no_conflict_found'))

target_values = pd.to_numeric(data[target_col], errors="coerce")
valid_target = target_values.dropna()

target_outlier_rows = pd.DataFrame()

if len(valid_target) >= 4:
    q1 = valid_target.quantile(0.25)
    q3 = valid_target.quantile(0.75)
    iqr = q3 - q1

    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr

    mean_y = valid_target.mean()
    std_y = valid_target.std()

    if std_y > 1e-12:
        z_scores_all = (target_values - mean_y).abs() / std_y
    else:
        z_scores_all = pd.Series(0.0, index=data.index)

    outlier_mask = (
        (target_values < lower_iqr) |
        (target_values > upper_iqr) |
        (z_scores_all > 3)
    )

    target_outlier_rows = data.loc[outlier_mask].copy()

    if not target_outlier_rows.empty:
        diagnostics_found = True

        target_outlier_rows.insert(
            0,
            t('outliers.col_original_row'),
            target_outlier_rows.index + 1
        )

        target_outlier_rows[t('outliers.col_reason')] = ""

        target_outlier_rows.loc[
            target_values.loc[target_outlier_rows.index] < lower_iqr,
            t('outliers.col_reason')
        ] += t('outliers.reason_below_iqr') + "; "

        target_outlier_rows.loc[
            target_values.loc[target_outlier_rows.index] > upper_iqr,
            t('outliers.col_reason')
        ] += t('outliers.reason_above_iqr') + "; "

        target_outlier_rows.loc[
            z_scores_all.loc[target_outlier_rows.index] > 3,
            t('outliers.col_reason')
        ] += t('outliers.reason_zscore') + "; "

        target_outlier_rows[t('outliers.col_iqr_lower')] = lower_iqr
        target_outlier_rows[t('outliers.col_iqr_upper')] = upper_iqr
        target_outlier_rows[t('outliers.col_zscore')] = z_scores_all.loc[target_outlier_rows.index].values

        st.warning(t('outliers.count_warning', count=len(target_outlier_rows)))
        st.info(t('outliers.diagnostic_info'))

        show_molecule_grid_from_table(
            table_df=target_outlier_rows,
            title=t('outliers.structures_title'),
            target_col=target_col,
            smiles_col=smiles_col_current,
            max_molecules=100,
            key_prefix="target_property_outliers"
        )
        st.dataframe(
            target_outlier_rows,
            width="stretch",
            hide_index=True
        )
    else:
        st.success(t('outliers.no_outliers'))
else:
    st.info(t('outliers.insufficient_data'))

if not diagnostics_found:
    st.info(t('outliers.no_gross_issues'))

st.info(t('outliers.multivariate_info'))

# ------------------------------------------------------------------
# Structural dataset filter
st.markdown(
    '<div class="tool-badge">' + t('struct_filter.tool_badge') + '</div>',
    unsafe_allow_html=True
)
with st.expander(t('struct_filter.expander_title'), expanded=False):
    show_markdown_help(
        t('struct_filter.help_title_structural'),
        os.path.join(HELP_DIR, "structural_filter_help.md"),
        expanded=False
    )

    st.info(t('struct_filter.smiles_col_info', col=smiles_col_current))

    show_markdown_help(
        t('struct_filter.help_title_smarts'),
        os.path.join(HELP_DIR, "smarts_help.md"),
        expanded=False
    )

    st.markdown(t('struct_filter.elemental_composition_title'))

    common_elements = [
        "C", "H", "N", "O", "S", "P",
        "F", "Cl", "Br", "I",
        "Si", "B", "Na", "K", "Ca", "Mg"
    ]

    col_el_1, col_el_2 = st.columns(2)

    with col_el_1:
        selected_elements = st.multiselect(
            t('struct_filter.select_elements'),
            options=common_elements,
            default=[],
            key="struct_filter_elements"
        )

    with col_el_2:
        element_mode = st.radio(
            t('struct_filter.element_logic'),
            [t('struct_filter.any_selected'), t('struct_filter.all_selected')],
            horizontal=True,
            key="struct_filter_element_mode"
        )

    st.markdown(t('struct_filter.functional_groups_title'))

    group_options = structural_filter_get_group_options()

    col_gr_1, col_gr_2 = st.columns(2)

    with col_gr_1:
        selected_groups = st.multiselect(
            t('struct_filter.select_groups'),
            options=group_options,
            default=[],
            key="struct_filter_groups"
        )

    with col_gr_2:
        group_mode = st.radio(
            t('struct_filter.group_logic'),
            [t('struct_filter.any_selected'), t('struct_filter.all_selected')],
            horizontal=True,
            key="struct_filter_group_mode"
        )

    st.markdown(t('struct_filter.custom_smarts_title'))

    custom_smarts_text = st.text_area(
        t('struct_filter.custom_smarts_label'),
        value="",
        height=90,
        key="struct_filter_custom_smarts",
        placeholder=t('struct_filter.custom_smarts_placeholder')
    )

    custom_smarts_mode = st.radio(
        t('struct_filter.custom_smarts_logic'),
        [t('struct_filter.any_smarts'), t('struct_filter.all_smarts')],
        horizontal=True,
        key="struct_filter_custom_smarts_mode"
    )

    st.markdown(t('struct_filter.simple_constraints_title'))

    col_sf_1, col_sf_2, col_sf_3 = st.columns(3)

    with col_sf_1:
        require_aromatic = st.selectbox(
            t('struct_filter.aromaticity'),
            [t('struct_filter.aromaticity_any'), t('struct_filter.aromaticity_only'), t('struct_filter.aromaticity_non')],
            index=0,
            key="struct_filter_aromatic"
        )

    with col_sf_2:
        require_only_ch = st.checkbox(
            t('struct_filter.only_ch'),
            value=False,
            key="struct_filter_only_ch"
        )

    with col_sf_3:
        combine_mode = st.radio(
            t('struct_filter.combine_mode'),
            [t('struct_filter.combine_all'), t('struct_filter.combine_any')],
            horizontal=False,
            key="struct_filter_combine_mode"
        )

    col_c_1, col_c_2, col_h_1, col_h_2 = st.columns(4)

    with col_c_1:
        use_carbon_min = st.checkbox(
            t('struct_filter.use_c_min'),
            value=False,
            key="struct_filter_use_c_min"
        )

        carbon_min = st.number_input(
            t('struct_filter.c_min'),
            min_value=0,
            max_value=200,
            value=0,
            step=1,
            key="struct_filter_c_min"
        ) if use_carbon_min else None

    with col_c_2:
        use_carbon_max = st.checkbox(
            t('struct_filter.use_c_max'),
            value=False,
            key="struct_filter_use_c_max"
        )

        carbon_max = st.number_input(
            t('struct_filter.c_max'),
            min_value=0,
            max_value=200,
            value=20,
            step=1,
            key="struct_filter_c_max"
        ) if use_carbon_max else None

    with col_h_1:
        use_hetero_min = st.checkbox(
            t('struct_filter.use_hetero_min'),
            value=False,
            key="struct_filter_use_hetero_min"
        )

        hetero_min = st.number_input(
            t('struct_filter.hetero_min'),
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            key="struct_filter_hetero_min"
        ) if use_hetero_min else None

    with col_h_2:
        use_hetero_max = st.checkbox(
            t('struct_filter.use_hetero_max'),
            value=False,
            key="struct_filter_use_hetero_max"
        )

        hetero_max = st.number_input(
            t('struct_filter.hetero_max'),
            min_value=0,
            max_value=100,
            value=5,
            step=1,
            key="struct_filter_hetero_max"
        ) if use_hetero_max else None

    st.markdown(t('struct_filter.text_search_title'))

    text_candidate_cols = [
        c for c in data.columns
        if data[c].dtype == "object" or str(data[c].dtype).startswith("string")
    ]

    text_col = None
    text_query = ""

    if text_candidate_cols:
        col_txt_1, col_txt_2 = st.columns(2)

        with col_txt_1:
            text_col = st.selectbox(
                t('struct_filter.text_column'),
                options=[t('struct_filter.text_column_none')] + text_candidate_cols,
                index=0,
                key="struct_filter_text_col"
            )

            if text_col == t('struct_filter.text_column_none'):
                text_col = None

        with col_txt_2:
            text_query = st.text_input(
                t('struct_filter.text_query'),
                value="",
                key="struct_filter_text_query",
                placeholder=t('struct_filter.text_query_placeholder')
            )
    else:
        st.caption(t('struct_filter.no_text_columns'))

    st.divider()

    col_run_filter_1, col_run_filter_2, col_run_filter_3 = st.columns([1.5, 1.5, 5])

    with col_run_filter_1:
        run_struct_filter = st.button(
            t('struct_filter.run_button'),
            type="primary",
            key="run_structural_filter"
        )

    with col_run_filter_2:
        reset_struct_filter = st.button(
            t('struct_filter.reset_button'),
            key="reset_structural_filter"
        )

    if reset_struct_filter:
        st.session_state.struct_filter_result_df = None
        st.session_state.struct_filter_note = ""
        st.success(t('struct_filter.reset_success'))

    if run_struct_filter:
        try:
            filtered_df, filter_report = structural_filter_apply(
                data=data,
                smiles_col=smiles_col_current,
                selected_elements=selected_elements,
                element_mode=element_mode,
                selected_groups=selected_groups,
                group_mode=group_mode,
                custom_smarts_text=custom_smarts_text,
                custom_smarts_mode=custom_smarts_mode,
                require_aromatic=require_aromatic,
                require_only_ch=require_only_ch,
                carbon_min=carbon_min,
                carbon_max=carbon_max,
                hetero_min=hetero_min,
                hetero_max=hetero_max,
                text_col=text_col,
                text_query=text_query,
                combine_mode=combine_mode
            )

            st.session_state.struct_filter_result_df = filtered_df.copy()

            st.session_state.struct_filter_note = t('struct_filter.filter_note', found=len(filtered_df), total=len(data))

            st.success(st.session_state.struct_filter_note)

        except Exception as e:
            st.error(t('struct_filter.filter_error', error=e))

filtered_result = st.session_state.get("struct_filter_result_df")

if filtered_result is not None:
    st.subheader(t('struct_filter_result.subheader'))

    col_res_1, col_res_2, col_res_3 = st.columns(3)

    with col_res_1:
        st.metric(t('struct_filter_result.metric_initial'), len(data))

    with col_res_2:
        st.metric(t('struct_filter_result.metric_found'), len(filtered_result))

    with col_res_3:
        percent_found = len(filtered_result) / len(data) * 100 if len(data) > 0 else 0
        st.metric(t('struct_filter_result.metric_percent'), f"{percent_found:.1f}%")

    if filtered_result.empty:
        st.warning(t('struct_filter_result.no_results'))
    else:
        show_cols = [
            c for c in [
                smiles_col_current,
                target_col,
                "name",
                "CAS",
                "molecular_formula",
                "formula_rdkit",
                "carbon_count_rdkit",
                "heteroatom_count_rdkit",
                "ring_count_rdkit",
                "aromatic_atom_count_rdkit",
                "Найденные группы",
                "Найденные SMARTS",
                ]
                if c in filtered_result.columns
            ]

        st.dataframe(
            filtered_result[show_cols].head(200),
            width="stretch",
            hide_index=True
        )

        with st.expander(t('struct_filter_result.effect_expander'), expanded=False):
            show_dataset_change_report(
                before_df=data,
                after_df=filtered_result,
                target_col=target_col,
                smiles_col=smiles_col_current,
                title=t('struct_filter_result.effect_title'),
                removed_title=t('struct_filter_result.removed_title'),
                key_prefix="structural_filter_effect"
            )

        csv_filtered = filtered_result.to_csv(index=False).encode("utf-8")

        st.download_button(
            t('struct_filter_result.download_button'),
            csv_filtered,
            "structural_filtered_dataset.csv",
            "text/csv"
        )

        filtered_view = filtered_result.copy()
        filtered_view["SMILES"] = filtered_view[smiles_col_current]

        show_molecule_grid_from_table(
            table_df=filtered_view,
            title=t('struct_filter_result.molecule_grid_title'),
            target_col=target_col,
            smiles_col="SMILES",
            max_molecules=200,
            key_prefix="structural_filter_molecules"
        )

        if st.button(
            t('struct_filter_result.use_button'),
            key="use_structural_filter_as_dataset"
        ):
            try:
                if filtered_result is None or filtered_result.empty:
                    st.error(t('struct_filter_result.cannot_apply_empty'))
                    st.stop()

                st.session_state.data = filtered_result.copy()
                st.session_state.struct_filter_applied = True

                st.session_state.data_source_note = t('struct_filter_result.data_source_note',
                    found=len(filtered_result),
                    total=len(data)
                )

                st.session_state.desc_calculated = False
                st.session_state.validation_done = False
                st.session_state.X_all = None
                st.session_state.y_all = None
                st.session_state.valid_indices = None
                st.session_state.desc_names = None
                st.session_state.df_desc = None
                st.session_state.trained_models = {}
                st.session_state.holdout_results_dict = {}
                st.session_state.kfold_results_dict = {}
                st.session_state.loo_results_dict = {}

                st.success(t('struct_filter_result.apply_success', count=len(filtered_result)))

                st.rerun()

            except Exception as e:
                st.error(t('struct_filter_result.apply_error', error=e))

# ------------------------------------------------------------------
# SAOD UI
st.markdown(
    '<div class="tool-badge">' + t('saod_ui.tool_badge') + '</div>',
    unsafe_allow_html=True
)
with st.expander(t('saod_ui.expander_title'), expanded=False):
    st.markdown(t('saod_ui.description'))

    saod2_source = st.radio(
        t('saod_ui.source_radio'),
        [t('saod_ui.source_main'), t('saod_ui.source_upload')],
        horizontal=True,
        key="saod2_source"
    )

    saod2_df = None

    if saod2_source == t('saod_ui.source_main'):
        saod2_df = data.copy()
        st.success(t('saod_ui.source_main_success', count=len(saod2_df)))
    else:
        saod2_file = st.file_uploader(t('saod_ui.upload_prompt'), type=["csv", "xlsx"], key="saod2_file_upload")

        if saod2_file is not None:
            try:
                ext = saod2_file.name.lower().split(".")[-1]

                if ext == "csv":
                    saod2_sep = st.radio(t('saod_ui.csv_sep'), [",", ";"], index=1, horizontal=True, key="saod2_csv_sep")
                    saod2_df = pd.read_csv(saod2_file, sep=saod2_sep)
                else:
                    saod2_df = pd.read_excel(saod2_file)

                saod2_df.columns = saod2_df.columns.str.strip()
                st.success(t('saod_ui.upload_success', count=len(saod2_df)))
                st.dataframe(saod2_df.head(), width="stretch")
            except Exception as e:
                st.error(t('saod_ui.upload_error', error=e))

    if saod2_df is not None:
        saod2_columns = saod2_df.columns.tolist()

        smiles_index = 0
        for i, col in enumerate(saod2_columns):
            if col.lower() == "smiles":
                smiles_index = i
                break

        saod2_smiles_col = st.selectbox(t('saod_ui.smiles_col'), saod2_columns, index=smiles_index, key="saod2_smiles_col")

        if target_col in saod2_columns:
            default_prop = target_col
        else:
            numeric_candidates = [
                c for c in saod2_df.select_dtypes(include=[np.number]).columns.tolist()
                if str(c).strip().lower() not in ["№", "no", "id", "index", "номер"]
            ]
            default_prop = numeric_candidates[0] if numeric_candidates else saod2_columns[0]

        saod2_property_state = st.session_state.get("saod2_property_col")
        invalid_property_state = (
            saod2_property_state not in saod2_columns
            or str(saod2_property_state).strip().lower() in ["№", "no", "id", "index", "номер"]
        )

        if invalid_property_state and "saod2_property_col" in st.session_state:
            del st.session_state["saod2_property_col"]
            saod2_property_state = None

        prop_value = saod2_property_state if saod2_property_state in saod2_columns else default_prop
        prop_index = saod2_columns.index(prop_value)

        saod2_property_col = st.selectbox(
            t('saod_ui.property_col'),
            saod2_columns,
            index=prop_index,
            key="saod2_property_col"
        )

        saod2_min_rule_points = st.slider(t('saod_ui.min_points'), min_value=3, max_value=10, value=3, step=1, key="saod2_min_rule_points")

        if st.button(t('saod_ui.run_button'), type="primary", key="run_saod2"):
            with st.spinner(t('saod_ui.run_spinner')):
                st.session_state.saod2_result = run_saod2_analysis(
                    input_df=saod2_df,
                    smiles_col=saod2_smiles_col,
                    property_col=saod2_property_col,
                    min_rule_points=saod2_min_rule_points,
                )

        if "saod2_result" in st.session_state:
            saod2_result_data = st.session_state.saod2_result

            if not isinstance(saod2_result_data, dict):
                st.warning(t('saod_ui.invalid_result'))

                if st.button(t('saod_ui.clear_button'), key="clear_saod2_result"):
                    del st.session_state.saod2_result
                    st.rerun()

                st.stop()

            if saod2_result_data["errors"]:
                for err in saod2_result_data["errors"]:
                    st.error(err)
            else:
                for warn in saod2_result_data["warnings"]:
                    st.warning(warn)

                processed = saod2_result_data["processed"]
                checkability = saod2_result_data["checkability"]
                raw_edge_table = saod2_result_data.get("raw_edge_table", pd.DataFrame())
                edge_table = saod2_result_data["edge_table"]
                rules = saod2_result_data["rules"]
                edge_details = saod2_result_data["edge_details"]
                broken_edges = saod2_result_data["broken_edges"]
                suspicion = saod2_result_data["suspicion"]
                summary = saod2_result_data["summary"]

                st.subheader(t('saod_ui.summary_subheader'))
                saod2_show_table(summary)
                
                st.subheader(t('saod_ui.review_subheader'))

                review_df = saod2_make_review_dataset(
                    original_df=saod2_df,
                    processed=processed,
                    checkability=checkability,
                    suspicion=suspicion
                )

                st.session_state.saod2_review_df = review_df

                col_saod_review_1, col_saod_review_2, col_saod_review_3 = st.columns(3)

                with col_saod_review_1:
                    st.metric(t('saod_ui.metric_total'), len(review_df))

                with col_saod_review_2:
                    n_auto_exclude = int(
                        review_df["SAOD_manual_decision"]
                        .astype(str)
                        .str.lower()
                        .str.startswith(t('saod2_review.manual_exclude').lower())
                        .sum()
                    )
                    st.metric(t('saod_ui.metric_auto_exclude'), n_auto_exclude)

                with col_saod_review_3:
                    st.metric(t('saod_ui.metric_remaining'), len(review_df) - n_auto_exclude)

                with st.expander(t('saod_ui.review_expander'), expanded=False):
                    edited_review_df = st.data_editor(
                        review_df,
                        column_config={
                            "SAOD_manual_decision": st.column_config.SelectboxColumn(
                                t('saod_ui.column_decision'),
                                options=[t('saod2_review.manual_keep'), t('saod2_review.manual_exclude'), t('saod2_review.auto_manual_check')],
                                required=True
                            ),
                            "SAOD_manual_comment": st.column_config.TextColumn(
                                t('saod_ui.column_comment')
                            ),
                        },
                        disabled=[
                            c for c in review_df.columns
                            if c not in ["SAOD_manual_decision", "SAOD_manual_comment"]
                        ],
                        hide_index=True,
                        width="stretch",
                        key="saod2_review_editor"
                    )

                    st.session_state.saod2_review_df = edited_review_df

                report_tables = {
                    "SAOD_review": st.session_state.saod2_review_df,
                    "Original_data": saod2_df.reset_index(drop=True),
                    "Processed": processed,
                    "Checkability": checkability,
                    "Suspicion": suspicion,
                    "Broken_edges": broken_edges,
                    "Rules": rules,
                    "Edge_details": edge_details,
                    "Summary": summary,
                }

                excel_data = saod2_excel_bytes(report_tables)

                st.download_button(
                    t('saod_ui.download_review'),
                    excel_data,
                    "saod2_review_dataset.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                col_saod_use_1, col_saod_use_2 = st.columns(2)
                cleaning_already_applied = st.session_state.get("saod2_cleaning_applied", False)

                if cleaning_already_applied:
                    st.success(
                        st.session_state.get(
                            "data_source_note",
                            t('saod_ui.cleaning_already_applied')
                        )
                    )
                    st.info(t('saod_ui.reapply_info'))

                col_saod_use_1, col_saod_use_2 = st.columns(2)

                with col_saod_use_1:
                    if st.button(
                        t('saod_ui.use_manual_button'),
                        key="use_saod_manual_cleaned",
                        disabled=cleaning_already_applied
                    ):
                        try:
                            # Сохраняем датасет ДО очистки, чтобы можно было вернуть назад
                            st.session_state.saod2_original_before_cleaning = data.copy()

                            cleaned_df = saod2_filter_dataset_by_review(
                                st.session_state.saod2_review_df,
                                decision_col="SAOD_manual_decision"
                            )

                            # Главное: делаем очищенный датасет текущим рабочим датасетом
                            st.session_state.data = cleaned_df.copy()
                            st.session_state.saod2_cleaned_df = cleaned_df.copy()

                            # Ставим флаг, чтобы повторно не резать уже очищенный датасет
                            st.session_state.saod2_cleaning_applied = True

                            st.session_state.data_source_note = t('saod_ui.manual_cleaned_note', count=len(cleaned_df))

                            # Сбрасываем старые дескрипторы и модели
                            st.session_state.desc_calculated = False
                            st.session_state.validation_done = False
                            st.session_state.X_all = None
                            st.session_state.y_all = None
                            st.session_state.valid_indices = None
                            st.session_state.desc_names = None
                            st.session_state.df_desc = None
                            st.session_state.trained_models = {}
                            st.session_state.holdout_results_dict = {}
                            st.session_state.kfold_results_dict = {}
                            st.session_state.loo_results_dict = {}

                            st.success(t('saod_ui.manual_cleaned_success', count=len(cleaned_df)))

                            st.rerun()
                            st.stop()

                        except Exception as e:
                            st.error(t('saod_ui.manual_cleaned_error', error=e))

                with col_saod_use_2:
                    if st.button(
                        t('saod_ui.use_auto_button'),
                        key="use_saod_auto_cleaned",
                        disabled=cleaning_already_applied
                    ):
                        try:
                            # Сохраняем датасет ДО очистки
                            st.session_state.saod2_original_before_cleaning = data.copy()

                            auto_review_df = review_df.copy()

                            auto_review_df["SAOD_manual_decision"] = auto_review_df[
                                "SAOD_auto_recommendation"
                            ].apply(
                                lambda x: t('saod2_review.manual_exclude')
                                if str(x).lower().startswith(t('saod2_review.manual_exclude').lower())
                                or t('saod2_review.auto_check_exclude_conflict').lower() in str(x).lower()
                                else t('saod2_review.manual_keep')
                            )

                            cleaned_df = saod2_filter_dataset_by_review(
                                auto_review_df,
                                decision_col="SAOD_manual_decision"
                            )

                            # Главное: делаем очищенный датасет текущим рабочим датасетом
                            st.session_state.data = cleaned_df.copy()
                            st.session_state.saod2_review_df = auto_review_df.copy()
                            st.session_state.saod2_cleaned_df = cleaned_df.copy()

                            # Флаг защиты от повторного удаления
                            st.session_state.saod2_cleaning_applied = True

                            st.session_state.data_source_note = t('saod_ui.auto_cleaned_note',
                                count=len(cleaned_df),
                                total=len(auto_review_df)
                            )

                            # Сбрасываем старые дескрипторы и модели
                            st.session_state.desc_calculated = False
                            st.session_state.validation_done = False
                            st.session_state.X_all = None
                            st.session_state.y_all = None
                            st.session_state.valid_indices = None
                            st.session_state.desc_names = None
                            st.session_state.df_desc = None
                            st.session_state.trained_models = {}
                            st.session_state.holdout_results_dict = {}
                            st.session_state.kfold_results_dict = {}
                            st.session_state.loo_results_dict = {}

                            st.success(t('saod_ui.auto_cleaned_success',
                                count=len(cleaned_df),
                                total=len(auto_review_df)
                            ))

                            st.rerun()
                            st.stop()

                        except Exception as e:
                            st.error(t('saod_ui.auto_cleaned_error', error=e))

                tab_v2_1, tab_v2_2, tab_v2_3, tab_v2_4, tab_v2_5, tab_v2_6, tab_v2_7, tab_v2_8 = st.tabs([
                    t('saod_tabs.tab_processed'),
                    t('saod_tabs.tab_checkability'),
                    t('saod_tabs.tab_rules'),
                    t('saod_tabs.tab_kitchen'),
                    t('saod_tabs.tab_edge_details'),
                    t('saod_tabs.tab_broken_edges'),
                    t('saod_tabs.tab_suspicious'),
                    t('saod_tabs.tab_uncheckable'),
                ])

                with tab_v2_1:
                    show_markdown_help(t('saod_tabs.help_processed'), os.path.join(HELP_DIR, "saod2_processed.md"), expanded=False)
                    saod2_show_table(processed)
                    st.download_button(t('saod_tabs.download_processed'), processed.to_csv(index=False).encode("utf-8"), "saod2_processed.csv", "text/csv")

                with tab_v2_2:
                    show_markdown_help(t('saod_tabs.help_checkability'), os.path.join(HELP_DIR, "saod2_checkability.md"), expanded=False)
                    saod2_show_table(checkability)
                    
                    if not checkability.empty:
                        low_checkability = checkability.copy()

                        if "overall_checkability" in low_checkability.columns:
                            low_checkability = low_checkability[
                                low_checkability["overall_checkability"].isin([
                                    t('saod_checkability.almost_uncheckable'),
                                    t('saod_checkability.weakly_checkable'),
                                    t('saod_checkability.unique_pattern_partially_checkable')
                                ])
                            ].copy()
                        elif "checkability_level" in low_checkability.columns:
                            low_checkability = low_checkability[
                                low_checkability["checkability_level"].isin([
                                    t('saod_checkability.almost_uncheckable'),
                                    t('saod_checkability.weakly_checkable')
                                ])
                            ].copy()

                        low_checkability_vis = saod2_add_smiles_for_visualization(
                            low_checkability,
                            processed
                        )

                        show_saod_molecule_grid(
                            table_df=low_checkability,
                            processed=processed,
                            title=t('saod_tabs.low_checkability_title'),
                            key_prefix="saod_low_checkability_molecules",
                            max_molecules=100
                        )
                    if not checkability.empty:
                        if "overall_checkability" in checkability.columns:
                            check_summary = checkability.groupby("overall_checkability").size().reset_index(name="n_compounds").sort_values("n_compounds", ascending=False)
                        else:
                            check_summary = checkability.groupby("checkability_level").size().reset_index(name="n_compounds").sort_values("n_compounds", ascending=False)
                        st.subheader(t('saod_tabs.checkability_summary'))
                        saod2_show_table(check_summary)

                with tab_v2_3:
                    show_markdown_help(t('saod_tabs.help_rules'), os.path.join(HELP_DIR, "saod2_rules.md"), expanded=False)
                    saod2_show_table(rules)

                    if not rules.empty and "can_be_used_for_checking" in rules.columns:
                        usable_rules = rules[rules["can_be_used_for_checking"] == True].copy()
                        st.subheader(t('saod_tabs.usable_rules'))
                        saod2_show_table(usable_rules)

                with tab_v2_4:
                    show_markdown_help(t('saod_tabs.help_kitchen'), os.path.join(HELP_DIR, "saod2_kitchen.md"), expanded=False)

                    if edge_details.empty:
                        st.info(t('saod_tabs.no_edge_details'))
                    else:
                        edge_counts = edge_details.groupby("edge_label").size().reset_index(name="n").sort_values("n", ascending=False)
                        selected_edge = st.selectbox(t('saod_tabs.select_edge'), edge_counts["edge_label"].tolist(), key="saod2_kitchen_edge_select")

                        with st.expander(t('saod_tabs.edge_explanation_expander'), expanded=False):
                            st.markdown(saod2_edge_kitchen_explanation(edge_details=edge_details, edge_label=selected_edge))

                        st.markdown(t('saod_tabs.delta_plots_title'))
                        col_delta_1, col_delta_2 = st.columns(2)

                        with col_delta_1:
                            fig_delta = saod2_plot_edge_delta(edge_details=edge_details, edge_label=selected_edge)
                            if fig_delta is not None:
                                st.pyplot(fig_delta)
                            else:
                                st.info(t('saod_tabs.insufficient_points_delta'))

                        with col_delta_2:
                            fig_delta_change = saod2_plot_edge_delta_change(edge_details=edge_details, edge_label=selected_edge)
                            if fig_delta_change is not None:
                                st.pyplot(fig_delta_change)
                            else:
                                st.info(t('saod_tabs.insufficient_points_delta_delta'))

                        st.markdown(t('saod_tabs.kitchen_table_title'))
                        kitchen_table = saod2_make_edge_kitchen_table(edge_details=edge_details, edge_label=selected_edge)
                        saod2_show_table(kitchen_table)

                        st.markdown(t('saod_tabs.edge_list_title'))
                        saod2_show_table(edge_counts)

                with tab_v2_5:
                    show_markdown_help(t('saod_tabs.help_edge_details'), os.path.join(HELP_DIR, "saod2_edge_details.md"), expanded=False)
                    saod2_show_table(edge_details)

                    with st.expander(t('saod_tabs.aggregated_edges_expander')):
                        st.info(t('saod_tabs.aggregated_edges_info'))
                        saod2_show_table(edge_table)

                    with st.expander(t('saod_tabs.raw_comparisons_expander')):
                        st.warning(t('saod_tabs.raw_comparisons_info'))
                        saod2_show_table(raw_edge_table)

                with tab_v2_6:
                    show_markdown_help(t('saod_tabs.help_broken_edges'), os.path.join(HELP_DIR, "saod2_broken_edges.md"), expanded=False)
                    if broken_edges.empty:
                        st.success(t('saod_tabs.no_broken_edges'))
                    else:
                        saod2_show_table(broken_edges)
                        
                        broken_mols_rows = []

                        def add_broken_smiles_rows(row, smiles_value, name_value, prop_value, role_value):
                            smiles_text = str(smiles_value).strip()

                            if not smiles_text:
                                return

                            smiles_parts = [
                                s.strip()
                                for s in smiles_text.replace("\n", ";").split(";")
                                if s.strip()
                            ]

                            for smi in smiles_parts:
                                broken_mols_rows.append({
                                    "SMILES": smi,
                                    t('saod_tabs.name_label'): name_value,
                                    t('saod_tabs.property_label'): prop_value,
                                    t('saod_tabs.edge_label_col'): row.get("edge_label", ""),
                                    t('saod_tabs.role_label'): role_value,
                                    t('saod_tabs.break_level_label'): row.get("edge_level", ""),
                                })

                        for _, row in broken_edges.iterrows():
                            add_broken_smiles_rows(
                                row=row,
                                smiles_value=row.get("smiles_a", ""),
                                name_value=row.get("name_a", ""),
                                prop_value=row.get("value_a", np.nan),
                                role_value="A"
                            )

                            add_broken_smiles_rows(
                                row=row,
                                smiles_value=row.get("smiles_b", ""),
                                name_value=row.get("name_b", ""),
                                prop_value=row.get("value_b", np.nan),
                                role_value="B"
                            )

                        broken_mols_df = pd.DataFrame(broken_mols_rows)

                        if not broken_mols_df.empty:
                            broken_mols_df = broken_mols_df.drop_duplicates(
                                subset=["SMILES", t('saod_tabs.role_label'), t('saod_tabs.property_label')],
                                keep="first"
                            ).reset_index(drop=True)

                            broken_mols_df.insert(0, t('saod_tabs.number_col'), range(1, len(broken_mols_df) + 1))

                            show_molecule_grid_from_table(
                                table_df=broken_mols_df,
                                title=t('saod_tabs.broken_structures_title'),
                                target_col=t('saod_tabs.property_label'),
                                smiles_col="SMILES",
                                max_molecules=100,
                                key_prefix="saod_broken_rule_molecules"
                            )

                with tab_v2_7:
                    show_markdown_help(t('saod_tabs.help_suspicious'), os.path.join(HELP_DIR, "saod2_suspicion.md"), expanded=False)
                    saod2_show_table(suspicion)

                    if not suspicion.empty:
                        suspicious_only = suspicion[suspicion["final_status"].isin([
                            t('saod_suspicion.status_needs_check'),
                            t('saod_suspicion.status_highly_suspicious'),
                            t('saod_suspicion.status_critical')
                        ])].copy()
                        if not suspicious_only.empty:
                            st.subheader(t('saod_tabs.manual_check_subheader'))
                            saod2_show_table(suspicious_only)
                            
                            suspicious_vis = saod2_add_smiles_for_visualization(
                                suspicious_only,
                                processed
                            )

                            show_saod_molecule_grid(
                                table_df=suspicious_only,
                                processed=processed,
                                title=t('saod_tabs.suspicious_structures_title'),
                                key_prefix="saod_suspicious_molecules",
                                max_molecules=100
                            )

                with tab_v2_8:
                    show_markdown_help(t('saod_tabs.help_uncheckable'), os.path.join(HELP_DIR, "saod2_uncheckable.md"), expanded=False)

                    if checkability.empty:
                        st.info(t('saod_tabs.no_data'))
                    else:
                        if "overall_checkability" in checkability.columns:
                            isolated = checkability[checkability["overall_checkability"].isin([
                                t('saod_checkability.almost_uncheckable'),
                                t('saod_checkability.unique_pattern_partially_checkable'),
                                t('saod_checkability.weakly_checkable')
                            ])].copy()
                        else:
                            isolated = checkability[checkability["checkability_level"] == t('saod_checkability.almost_uncheckable')].copy()

                        if isolated.empty:
                            st.success(t('saod_tabs.no_isolated'))
                        else:
                            st.warning(t('saod_tabs.isolated_warning'))
                            saod2_show_table(isolated)
                            
                            isolated_vis = saod2_add_smiles_for_visualization(
                                isolated,
                                processed
                            )

                            show_saod_molecule_grid(
                                table_df=isolated,
                                processed=processed,
                                title=t('saod_tabs.isolated_structures_title'),
                                key_prefix="saod_uncheckable_molecules",
                                max_molecules=100
                            )

# ------------------------------------------------------------------
# Spectra module UI
spectra_expander_should_be_open = (
    st.session_state.get("keep_spectra_expander_open", False)
    or isinstance(st.session_state.get("spectral_descriptors_df"), pd.DataFrame)
    or st.session_state.get("spectral_descriptors_transferred_ready", False)
    or st.session_state.get("pending_qspr_descriptor_bundle_ready", False)
)

# ------------------------------------------------------------------
# Spectra module UI

spectra_expander_should_be_open = (
    st.session_state.get("keep_spectra_expander_open", False)
    or isinstance(st.session_state.get("spectral_descriptors_df"), pd.DataFrame)
    or st.session_state.get("spectral_descriptors_transferred_ready", False)
    or st.session_state.get("pending_qspr_descriptor_bundle_ready", False)
)

st.markdown(
    '<div class="tool-badge">' + t('spectra.tool_badge') + '</div>',
    unsafe_allow_html=True
)

with st.expander(
    t('spectra.expander_title'),
    expanded=spectra_expander_should_be_open
):
    show_markdown_help(
        t('spectra.help_search'),
        os.path.join(HELP_DIR, "spectra_search_help.md"),
        expanded=False
    )

    if qspr_is_online_mode():
        with st.expander(t('spectra.import_expander'), expanded=False):
            qspr_online_lock_notice("Local spectral bank import")
            st.file_uploader(
                t('spectra.import_file_uploader'),
                type=["jdx", "dx", "json", "txt", "csv"],
                accept_multiple_files=True,
                disabled=True,
                key="local_spectra_import_files_online_disabled"
            )
            st.button(
                t('spectra.import_button'),
                key="run_local_spectra_import_online_disabled",
                disabled=True,
            )

    elif is_admin():
        with st.expander(t('spectra.import_expander'), expanded=False):
            st.markdown(t('spectra.import_section1_title'))

            st.caption(t('spectra.import_caption'))

            uploaded_spectrum_files = st.file_uploader(
                t('spectra.import_file_uploader'),
                type=["jdx", "dx", "json", "txt", "csv"],
                accept_multiple_files=True,
                key="local_spectra_import_files"
            )

            local_import_type = st.selectbox(
                t('spectra.import_type_select'),
                ["auto", "IR", "Mass"],
                index=0,
                key="local_spectra_import_type"
            )

            overwrite_existing = st.checkbox(
                t('spectra.import_overwrite_checkbox'),
                value=False,
                key="local_spectra_overwrite_existing"
            )

            st.caption(t('spectra.import_caption2'))

            if uploaded_spectrum_files and st.button(
                t('spectra.import_button'),
                key="run_local_spectra_import"
            ):
                import_rows = []

                temp_import_dir = os.path.join(
                    SPECTRA_BANK_DIR,
                    "_tmp_local_import"
                )

                os.makedirs(temp_import_dir, exist_ok=True)

                for uploaded_file in uploaded_spectrum_files:
                    safe_uploaded_name = spectra_safe_filename_part(
                        uploaded_file.name
                    )

                    temp_path = os.path.join(
                        temp_import_dir,
                        safe_uploaded_name
                    )

                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    result = spectra_import_local_spectrum_file(
                        filepath=temp_path,
                        spectrum_type=local_import_type,
                        compound_hint=None,
                        overwrite_existing=overwrite_existing
                    )

                    record = result.get("record") or {}

                    import_rows.append({
                        t('spectra.import_col_file'): uploaded_file.name,
                        t('spectra.import_col_status'): result.get("status", ""),
                        t('spectra.import_col_message'): result.get("message", ""),
                        "spectrum_id": record.get("spectrum_id", ""),
                        t('spectra.import_col_spectrum_type'): record.get("spectrum_type", ""),
                        "InChIKey": record.get("inchikey", ""),
                        t('spectra.import_col_name'): record.get("name", ""),
                        t('spectra.import_col_points'): record.get("n_points_processed", ""),
                        "processed_file": record.get("processed_file", ""),
                    })

                import_report_df = pd.DataFrame(import_rows)

                st.dataframe(
                    import_report_df,
                    width="stretch",
                    hide_index=True
                )

                st.download_button(
                    t('spectra.import_download_report'),
                    import_report_df.to_csv(index=False).encode("utf-8-sig"),
                    "local_spectra_import_report.csv",
                    "text/csv",
                    key="download_local_spectra_import_report"
                )

            st.divider()

            st.markdown(t('spectra.reindex_section_title'))

            st.caption(t('spectra.reindex_caption'))

            st.code(
                f"IR:   {SPECTRA_IR_RAW_DIR}\n"
                f"Mass: {SPECTRA_MASS_RAW_DIR}",
                language="text"
            )

            col_reindex_1, col_reindex_2, col_reindex_3 = st.columns(3)

            with col_reindex_1:
                reindex_ir = st.checkbox(
                    t('spectra.reindex_ir_checkbox'),
                    value=True,
                    key="reindex_existing_ir_raw"
                )

            with col_reindex_2:
                reindex_mass = st.checkbox(
                    t('spectra.reindex_mass_checkbox'),
                    value=True,
                    key="reindex_existing_mass_raw"
                )

            with col_reindex_3:
                reindex_recursive = st.checkbox(
                    t('spectra.reindex_recursive_checkbox'),
                    value=False,
                    key="reindex_existing_raw_recursive"
                )

            if st.button(
                t('spectra.reindex_button'),
                key="run_reindex_existing_raw_spectra"
            ):
                reindex_report_df = spectra_reindex_existing_raw_spectra(
                    scan_ir=reindex_ir,
                    scan_mass=reindex_mass,
                    overwrite_existing=overwrite_existing,
                    recursive=reindex_recursive
                )

                st.dataframe(
                    reindex_report_df,
                    width="stretch",
                    hide_index=True
                )

                n_added = int(
                    (reindex_report_df["status"] == "imported_local").sum()
                )

                n_existing = int(
                    (
                        (reindex_report_df["status"] == "already_registered")
                        | (reindex_report_df["status"] == "already_in_bank")
                    ).sum()
                )

                n_failed = len(reindex_report_df) - n_added - n_existing

                col_r_1, col_r_2, col_r_3 = st.columns(3)

                with col_r_1:
                    st.metric(t('spectra.reindex_metric_added'), n_added)

                with col_r_2:
                    st.metric(t('spectra.reindex_metric_existing'), n_existing)

                with col_r_3:
                    st.metric(t('spectra.reindex_metric_failed'), n_failed)

                st.download_button(
                    t('spectra.reindex_download_report'),
                    reindex_report_df.to_csv(index=False).encode("utf-8-sig"),
                    "raw_jdx_reindex_report.csv",
                    "text/csv",
                    key="download_raw_jdx_reindex_report"
                )
               

    else:
        st.info(
            "Спектральная база Augur подключается автоматически. "
            "Импорт, переиндексация и ручное пополнение базы скрыты в обычном режиме."
        )

    spectra_index = spectra_load_index()

    if spectra_index.empty:
        st.info(t('spectra.bank_empty'))
    else:
        idx = spectra_index.copy()

        required_spectra_cols = [
            "spectrum_type",
            "phase",
            "source",
            "source_database",
            "status",
            "active",
            "intensity_type",
            "sample_type",
            "spectrum_id",
            "inchikey",
            "canonical_smiles",
            "processed_file",
        ]

        for col in required_spectra_cols:
            if col not in idx.columns:
                idx[col] = ""

        idx["spectrum_type"] = idx["spectrum_type"].astype(str).str.strip()
        idx["phase"] = idx["phase"].astype(str).str.strip()
        idx["source"] = idx["source"].astype(str).str.strip()
        idx["source_database"] = idx["source_database"].astype(str).str.strip()
        idx["status"] = idx["status"].astype(str).str.strip()
        idx["active"] = idx["active"].astype(str).str.strip().str.lower()
        idx["intensity_type"] = idx["intensity_type"].astype(str).str.strip()
        idx["sample_type"] = idx["sample_type"].astype(str).str.strip()

        # Нормализация пустых значений.
        idx["spectrum_type"] = idx["spectrum_type"].replace("", "unknown")
        idx["phase"] = idx["phase"].replace("", "unknown")
        idx["source"] = idx["source"].replace("", "unknown")
        idx["source_database"] = idx["source_database"].replace("", "unknown")
        idx["status"] = idx["status"].replace("", "unknown")
        idx["intensity_type"] = idx["intensity_type"].replace("", "unknown")
        idx["sample_type"] = idx["sample_type"].replace("", "unknown")

        # Нормализация типа спектра.
        idx["_spectrum_type_norm"] = idx["spectrum_type"].apply(
            spectra_normalize_spectrum_type
        )

        # Нормализация фазы / состояния образца.
        # Функция spectral_normalize_phase_value уже есть в spectra_core.py.
        try:
            idx["_phase_norm"] = idx["phase"].apply(spectral_normalize_phase_value)
        except Exception:
            idx["_phase_norm"] = idx["phase"].astype(str).str.lower().replace("", "unknown")

        active_values = ["true", "1", "yes", "y", "да", "active", ""]
        active_mask = idx["active"].isin(active_values)
        idx_active = idx[active_mask].copy()

        ir_count = int((idx_active["_spectrum_type_norm"] == "IR").sum())
        mass_count = int((idx_active["_spectrum_type_norm"] == "Mass").sum())
        other_count = int(len(idx_active) - ir_count - mass_count)

        st.write(t('spectra.bank_summary',
            total=len(idx),
            active=len(idx_active),
            ir=ir_count,
            mass=mass_count,
            other=other_count
        ))

        if not is_admin():
            st.subheader("Спектральная база Augur")
            status_text = "подключена" if len(idx_active) > 0 else "не подключена"
            st.markdown(
                "Доступно спектров: "
                f"**{len(idx_active)}**\n\n"
                f"Статус: **{status_text}**\n\n"
                "База используется автоматически для проверки ваших SMILES. "
                "Служебное пополнение, переиндексация и внешние загрузки не показываются."
            )

        # ------------------------------------------------------------
        # Объединённая сводка spectra_bank:
        # фазы, типы спектров, источники, интенсивность, статусы

        with st.expander(t('spectra.summary_expander'), expanded=False):
            st.caption(t('spectra.summary_caption'))

            if idx_active.empty:
                st.info(t('spectra.no_active_spectra'))
            else:
                phase_order = [
                    "gas",
                    "liquid",
                    "solution",
                    "film",
                    "solid",
                    "kbr",
                    "nujol",
                    "unknown",
                ]

                # ----------------------------------------------------
                # 1. Быстрые счётчики фаз

                phase_counts = (
                    idx_active
                    .groupby("_phase_norm")
                    .size()
                    .reset_index(name=t('spectra.phase_count_col'))
                    .rename(columns={"_phase_norm": t('spectra.phase_state_col')})
                )

                phase_pivot = (
                    idx_active
                    .pivot_table(
                        index="_phase_norm",
                        columns="_spectrum_type_norm",
                        values="spectrum_id",
                        aggfunc="count",
                        fill_value=0
                    )
                    .reset_index()
                    .rename(columns={"_phase_norm": t('spectra.phase_state_col')})
                )

                phase_summary = phase_counts.merge(
                    phase_pivot,
                    on=t('spectra.phase_state_col'),
                    how="left"
                )

                for col in ["IR", "Mass"]:
                    if col not in phase_summary.columns:
                        phase_summary[col] = 0

                known_cols = [t('spectra.phase_state_col'), t('spectra.phase_count_col'), "IR", "Mass"]
                extra_type_cols = [
                    c for c in phase_summary.columns
                    if c not in known_cols
                ]

                if extra_type_cols:
                    phase_summary[t('spectra.phase_other_col')] = phase_summary[extra_type_cols].sum(axis=1)
                else:
                    phase_summary[t('spectra.phase_other_col')] = 0

                phase_summary["_sort"] = phase_summary[t('spectra.phase_state_col')].apply(
                    lambda x: phase_order.index(x) if x in phase_order else 999
                )

                phase_summary = (
                    phase_summary
                    .sort_values(["_sort", t('spectra.phase_count_col')], ascending=[True, False])
                    .drop(columns=["_sort"] + extra_type_cols, errors="ignore")
                    .reset_index(drop=True)
                )

                phase_metric_values = {
                    row[t('spectra.phase_state_col')]: int(row[t('spectra.phase_count_col')])
                    for _, row in phase_summary.iterrows()
                }

                st.markdown(t('spectra.phase_title'))

                col_phase_1, col_phase_2, col_phase_3, col_phase_4 = st.columns(4)

                with col_phase_1:
                    st.metric("Gas", phase_metric_values.get("gas", 0))

                with col_phase_2:
                    st.metric("Liquid", phase_metric_values.get("liquid", 0))

                with col_phase_3:
                    st.metric(
                        "Solution / Film",
                        phase_metric_values.get("solution", 0)
                        + phase_metric_values.get("film", 0)
                    )

                with col_phase_4:
                    st.metric(
                        "Solid / KBr / Nujol",
                        (
                            phase_metric_values.get("solid", 0)
                            + phase_metric_values.get("kbr", 0)
                            + phase_metric_values.get("nujol", 0)
                        )
                    )

                st.dataframe(
                    phase_summary,
                    width="stretch",
                    hide_index=True
                )

                # ----------------------------------------------------
                # 2. Тип спектра × фаза

                st.markdown(t('spectra.type_phase_title'))

                phase_type_summary = (
                    idx_active
                    .groupby(["_spectrum_type_norm", "_phase_norm"])
                    .size()
                    .reset_index(name=t('spectra.count_col'))
                    .rename(columns={
                        "_spectrum_type_norm": t('spectra.type_col'),
                        "_phase_norm": t('spectra.phase_state_col')
                    })
                    .sort_values([t('spectra.type_col'), t('spectra.count_col')], ascending=[True, False])
                )

                st.dataframe(
                    phase_type_summary,
                    width="stretch",
                    hide_index=True
                )

                # ----------------------------------------------------
                # 3. Типы спектров

                st.markdown(t('spectra.types_title'))

                type_summary = (
                    idx_active
                    .groupby("_spectrum_type_norm")
                    .size()
                    .reset_index(name=t('spectra.count_col'))
                    .rename(columns={"_spectrum_type_norm": t('spectra.type_col')})
                    .sort_values(t('spectra.count_col'), ascending=False)
                )

                st.dataframe(
                    type_summary,
                    width="stretch",
                    hide_index=True
                )

                # ----------------------------------------------------
                # 4. Источник × тип спектра

                st.markdown(t('spectra.sources_title'))

                source_type_summary = (
                    idx_active
                    .groupby(["source", "_spectrum_type_norm"])
                    .size()
                    .reset_index(name=t('spectra.count_col'))
                    .rename(columns={
                        "source": t('spectra.source_col'),
                        "_spectrum_type_norm": t('spectra.type_col')
                    })
                    .sort_values([t('spectra.source_col'), t('spectra.count_col')], ascending=[True, False])
                )

                st.dataframe(
                    source_type_summary,
                    width="stretch",
                    hide_index=True
                )

                # ----------------------------------------------------
                # 5. База-источник × тип спектра

                st.markdown(t('spectra.source_databases_title'))

                source_database_summary = (
                    idx_active
                    .groupby(["source_database", "_spectrum_type_norm"])
                    .size()
                    .reset_index(name=t('spectra.count_col'))
                    .rename(columns={
                        "source_database": t('spectra.source_database_col'),
                        "_spectrum_type_norm": t('spectra.type_col')
                    })
                    .sort_values([t('spectra.source_database_col'), t('spectra.count_col')], ascending=[True, False])
                )

                st.dataframe(
                    source_database_summary,
                    width="stretch",
                    hide_index=True
                )

                # ----------------------------------------------------
                # 6. Тип интенсивности × тип спектра

                st.markdown(t('spectra.intensity_title'))

                intensity_summary = (
                    idx_active
                    .groupby(["intensity_type", "_spectrum_type_norm"])
                    .size()
                    .reset_index(name=t('spectra.count_col'))
                    .rename(columns={
                        "intensity_type": t('spectra.intensity_col'),
                        "_spectrum_type_norm": t('spectra.type_col')
                    })
                    .sort_values([t('spectra.intensity_col'), t('spectra.count_col')], ascending=[True, False])
                )

                st.dataframe(
                    intensity_summary,
                    width="stretch",
                    hide_index=True
                )

                # ----------------------------------------------------
                # 7. Sample type × тип спектра

                st.markdown(t('spectra.sample_type_title'))

                sample_type_summary = (
                    idx_active
                    .groupby(["sample_type", "_spectrum_type_norm"])
                    .size()
                    .reset_index(name=t('spectra.count_col'))
                    .rename(columns={
                        "sample_type": t('spectra.sample_type_col'),
                        "_spectrum_type_norm": t('spectra.type_col')
                    })
                    .sort_values([t('spectra.sample_type_col'), t('spectra.count_col')], ascending=[True, False])
                )

                st.dataframe(
                    sample_type_summary,
                    width="stretch",
                    hide_index=True
                )

                # ----------------------------------------------------
                # 8. Статусы всех записей, не только активных

                st.markdown(t('spectra.status_title'))

                status_summary = (
                    idx
                    .groupby(["status", "_spectrum_type_norm"])
                    .size()
                    .reset_index(name=t('spectra.count_col'))
                    .rename(columns={
                        "status": t('spectra.status_col'),
                        "_spectrum_type_norm": t('spectra.type_col')
                    })
                    .sort_values([t('spectra.status_col'), t('spectra.count_col')], ascending=[True, False])
                )

                st.dataframe(
                    status_summary,
                    width="stretch",
                    hide_index=True
                )

                # ----------------------------------------------------
                # 9. Скачать все сводки одним CSV

                combined_summary_tables = []

                phase_summary_export = phase_summary.copy()
                phase_summary_export.insert(0, t('spectra.export_section'), t('spectra.export_phases'))
                combined_summary_tables.append(phase_summary_export)

                phase_type_export = phase_type_summary.copy()
                phase_type_export.insert(0, t('spectra.export_section'), t('spectra.export_type_phase'))
                combined_summary_tables.append(phase_type_export)

                type_summary_export = type_summary.copy()
                type_summary_export.insert(0, t('spectra.export_section'), t('spectra.export_types'))
                combined_summary_tables.append(type_summary_export)

                source_type_export = source_type_summary.copy()
                source_type_export.insert(0, t('spectra.export_section'), t('spectra.export_sources'))
                combined_summary_tables.append(source_type_export)

                source_database_export = source_database_summary.copy()
                source_database_export.insert(0, t('spectra.export_section'), t('spectra.export_databases'))
                combined_summary_tables.append(source_database_export)

                intensity_export = intensity_summary.copy()
                intensity_export.insert(0, t('spectra.export_section'), t('spectra.export_intensity'))
                combined_summary_tables.append(intensity_export)

                sample_type_export = sample_type_summary.copy()
                sample_type_export.insert(0, t('spectra.export_section'), t('spectra.export_sample_type'))
                combined_summary_tables.append(sample_type_export)

                status_export = status_summary.copy()
                status_export.insert(0, t('spectra.export_section'), t('spectra.export_status'))
                combined_summary_tables.append(status_export)

                spectra_summary_export = pd.concat(
                    combined_summary_tables,
                    ignore_index=True,
                    sort=False
                )

                csv_spectra_summary = spectra_summary_export.to_csv(index=False).encode("utf-8")

                st.download_button(
                    t('spectra.download_combined_summary'),
                    csv_spectra_summary,
                    "spectra_bank_combined_summary.csv",
                    "text/csv",
                    key="download_spectra_combined_summary"
                )

    if is_admin() and st.checkbox(t('spectra.show_index_checkbox'), key="show_spectra_index"):
        if spectra_index.empty:
            st.info(t('spectra.index_empty'))
        else:
            st.dataframe(spectra_index, width="stretch")

    current_df = data.copy()

    compounds_for_search = spectra_prepare_compounds_from_df(
        current_df,
        smiles_col=smiles_col_current
    )

    if is_admin():
        st.subheader(t('spectra.search_settings'))
        st.markdown(t('spectra.spectrum_types_title'))

        col_type_1, col_type_2, col_type_spacer = st.columns([1, 1, 6])

        with col_type_1:
            search_ir = st.checkbox(t('spectra.checkbox_ir'), value=True, key="spectra_type_ir")

        with col_type_2:
            search_mass = st.checkbox(
                t('spectra.checkbox_mass'),
                value=True,
                key="spectra_type_mass",
            )

        selected_spectrum_types = []

        if search_ir:
            selected_spectrum_types.append("IR")

        if search_mass:
            selected_spectrum_types.append("Mass")

        if not selected_spectrum_types:
            st.warning(t('spectra.warning_select_type'))
        else:
            st.caption(t('spectra.types_for_search', types=" + ".join(selected_spectrum_types)))
    else:
        search_ir = True
        search_mass = True
        selected_spectrum_types = ["IR", "Mass"]
        st.subheader("Подходящие спектры для ваших веществ")
        st.caption("Проверяются IR и Mass спектры из автоматически подключенной базы Augur.")

    # Быстрая проверка spectra_bank без тысяч повторных чтений spectra_index.csv

    ir_statuses = []
    mass_statuses = []
    overall_statuses = []

    index_fast = spectra_index.copy()

    if index_fast is None or index_fast.empty:
        index_fast = pd.DataFrame(columns=[
            "inchikey",
            "canonical_smiles",
            "spectrum_type",
            "active"
        ])

    for col in ["inchikey", "canonical_smiles", "spectrum_type", "active"]:
        if col not in index_fast.columns:
            index_fast[col] = ""

    index_fast["inchikey"] = index_fast["inchikey"].astype(str).str.strip()
    index_fast["canonical_smiles"] = index_fast["canonical_smiles"].astype(str).str.strip()
    index_fast["spectrum_type"] = index_fast["spectrum_type"].astype(str).str.strip()
    index_fast["active"] = index_fast["active"].astype(str).str.strip()

    index_fast["_spectrum_type_norm"] = index_fast["spectrum_type"].apply(
        spectra_normalize_spectrum_type
    )

    active_values = ["true", "1", "yes", "y", "да", "active", ""]
    index_fast["_active_norm"] = (
        index_fast["active"]
        .astype(str)
        .str.lower()
        .isin(active_values)
    )

    index_fast = index_fast[index_fast["_active_norm"]].copy()

    ir_bank = index_fast[index_fast["_spectrum_type_norm"] == "IR"].copy()
    mass_bank = index_fast[index_fast["_spectrum_type_norm"] == "Mass"].copy()

    ir_inchikey_set = set(ir_bank["inchikey"].dropna().astype(str).str.strip())
    ir_smiles_set = set(ir_bank["canonical_smiles"].dropna().astype(str).str.strip())

    mass_inchikey_set = set(mass_bank["inchikey"].dropna().astype(str).str.strip())
    mass_smiles_set = set(mass_bank["canonical_smiles"].dropna().astype(str).str.strip())

    for _, row in compounds_for_search.iterrows():
        inchikey_value = str(row.get("inchikey", "")).strip()
        canonical_smiles_value = str(row.get("canonical_smiles", "")).strip()

        existing_ir = (
            inchikey_value in ir_inchikey_set
            or canonical_smiles_value in ir_smiles_set
        )

        existing_mass = (
            inchikey_value in mass_inchikey_set
            or canonical_smiles_value in mass_smiles_set
        )

        ir_status = t('spectra.status_has_ir') if existing_ir else t('spectra.status_no_ir')
        mass_status = t('spectra.status_has_mass') if existing_mass else t('spectra.status_no_mass')

        ir_statuses.append(ir_status)
        mass_statuses.append(mass_status)

        if existing_ir and existing_mass:
            overall_statuses.append(t('spectra.status_both'))
        elif existing_ir:
            overall_statuses.append(t('spectra.status_only_ir'))
        elif existing_mass:
            overall_statuses.append(t('spectra.status_only_mass'))
        else:
            overall_statuses.append(t('spectra.status_none'))

    compounds_for_search["IR_status"] = ir_statuses
    compounds_for_search["Mass_status"] = mass_statuses
    compounds_for_search["spectrum_bank_status"] = overall_statuses

    st.subheader(t('spectra.coverage_subheader'))

    ir_available = int((compounds_for_search["IR_status"] == t('spectra.status_has_ir')).sum())
    mass_available = int((compounds_for_search["Mass_status"] == t('spectra.status_has_mass')).sum())

    if selected_spectrum_types == ["IR"]:
        spectra_available = ir_available
        spectra_missing = len(compounds_for_search) - ir_available
    elif selected_spectrum_types == ["Mass"]:
        spectra_available = mass_available
        spectra_missing = len(compounds_for_search) - mass_available
    else:
        both_available = int(
            (
                (compounds_for_search["IR_status"] == t('spectra.status_has_ir')) &
                (compounds_for_search["Mass_status"] == t('spectra.status_has_mass'))
            ).sum()
        )
        spectra_available = both_available
        spectra_missing = len(compounds_for_search) - both_available

    coverage_percent = (
        spectra_available / len(compounds_for_search) * 100
        if len(compounds_for_search) > 0
        else 0
    )

    col_cov1, col_cov2, col_cov3, col_cov4 = st.columns(4)

    with col_cov1:
        st.metric(t('spectra.metric_total'), len(compounds_for_search))

    with col_cov2:
        st.metric(t('spectra.metric_has_ir'), ir_available)

    with col_cov3:
        st.metric(t('spectra.metric_has_mass'), mass_available)

    if selected_spectrum_types == ["IR"]:
        missing_label = t('spectra.missing_ir')
    elif selected_spectrum_types == ["Mass"]:
        missing_label = t('spectra.missing_mass')
    else:
        missing_label = t('spectra.missing_both')

    with col_cov4:
        st.metric(missing_label, spectra_missing)

    st.progress(min(coverage_percent / 100, 1.0))
    if selected_spectrum_types == ["IR"]:
        coverage_label = t('spectra.coverage_ir')
    elif selected_spectrum_types == ["Mass"]:
        coverage_label = t('spectra.coverage_mass')
    else:
        coverage_label = t('spectra.coverage_both')

    st.caption(t('spectra.coverage_percent', label=coverage_label, percent=coverage_percent))

    with st.expander(t('spectra.show_compounds_expander'), expanded=False):
        st.caption(t('spectra.compact_caption'))

        compact_cols = [
            "row_index",
            "input_smiles",
            "structure_status",
            "IR_status",
            "Mass_status",
            "spectrum_bank_status",
        ]

        compact_cols = [
            col for col in compact_cols
            if col in compounds_for_search.columns
        ]

        compact_view = compounds_for_search[compact_cols].copy()

        if "row_index" in compact_view.columns:
            compact_view = compact_view.rename(columns={"row_index": t('spectra.col_row')})
            compact_view[t('spectra.col_row')] = compact_view[t('spectra.col_row')] + 1

        compact_view = compact_view.rename(columns={
            "input_smiles": t('spectra.col_smiles'),
            "structure_status": t('spectra.col_structure'),
            "IR_status": t('spectra.col_ir'),
            "Mass_status": t('spectra.col_mass'),
            "spectrum_bank_status": t('spectra.col_spectrum_status'),
        })

        st.dataframe(
            compact_view,
            width="stretch",
            hide_index=True
        )

        details_df = compounds_for_search.copy()

        try:
            debug_index = spectra_load_index()

            for col in [
                "inchikey",
                "canonical_smiles",
                "spectrum_type",
                "active",
                "raw_file",
                "processed_file",
            ]:
                if col not in debug_index.columns:
                    debug_index[col] = ""

            debug_index["inchikey"] = (
                debug_index["inchikey"]
                .astype(str)
                .str.strip()
            )

            debug_index["canonical_smiles"] = (
                debug_index["canonical_smiles"]
                .astype(str)
                .str.strip()
            )

            debug_index["_spectrum_type_norm"] = (
                debug_index["spectrum_type"]
                .astype(str)
                .apply(spectra_normalize_spectrum_type)
            )

            active_values_debug = [
                "true",
                "1",
                "yes",
                "y",
                "да",
                "active",
                "",
                "nan",
                "none",
            ]

            debug_index["_active_norm"] = (
                debug_index["active"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(active_values_debug)
            )

            debug_index_active = debug_index[
                debug_index["_active_norm"]
            ].copy()

            debug_ir = debug_index_active[
                debug_index_active["_spectrum_type_norm"] == "IR"
            ].copy()

            debug_mass = debug_index_active[
                debug_index_active["_spectrum_type_norm"] == "Mass"
            ].copy()

            ir_inchikeys = set(
                debug_ir["inchikey"]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .tolist()
            )

            ir_smiles = set(
                debug_ir["canonical_smiles"]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .tolist()
            )

            mass_inchikeys = set(
                debug_mass["inchikey"]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .tolist()
            )

            mass_smiles = set(
                debug_mass["canonical_smiles"]
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .tolist()
            )

            if "inchikey" in details_df.columns:
                details_df["matches_IR_by_inchikey"] = (
                    details_df["inchikey"]
                    .astype(str)
                    .str.strip()
                    .isin(ir_inchikeys)
                )

                details_df["matches_Mass_by_inchikey"] = (
                    details_df["inchikey"]
                    .astype(str)
                    .str.strip()
                    .isin(mass_inchikeys)
                )
            else:
                details_df["matches_IR_by_inchikey"] = False
                details_df["matches_Mass_by_inchikey"] = False

            if "canonical_smiles" in details_df.columns:
                details_df["matches_IR_by_smiles"] = (
                    details_df["canonical_smiles"]
                    .astype(str)
                    .str.strip()
                    .isin(ir_smiles)
                )

                details_df["matches_Mass_by_smiles"] = (
                    details_df["canonical_smiles"]
                    .astype(str)
                    .str.strip()
                    .isin(mass_smiles)
                )
            else:
                details_df["matches_IR_by_smiles"] = False
                details_df["matches_Mass_by_smiles"] = False

            details_df["spectra_index_active_IR_records"] = len(debug_ir)
            details_df["spectra_index_active_Mass_records"] = len(debug_mass)

        except Exception as e:
            details_df["spectra_index_diagnostics_error"] = str(e)

        csv_details = details_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            t('spectra.download_details_csv'),
            csv_details,
            "current_dataset_spectra_status_details.csv",
            "text/csv",
            key="download_current_dataset_spectra_status_details"
        )

    matched_spectra_rows = []

    if not index_fast.empty and not compounds_for_search.empty:
        matched_index = index_fast.copy()

        for col in ["spectrum_id", "phase", "source", "source_database"]:
            if col not in matched_index.columns:
                matched_index[col] = ""

        for _, compound_row in compounds_for_search.iterrows():
            input_smiles = str(compound_row.get("input_smiles", "")).strip()
            canonical_smiles_value = str(compound_row.get("canonical_smiles", "")).strip()
            inchikey_value = str(compound_row.get("inchikey", "")).strip()

            try:
                row_number = int(compound_row.get("row_index", 0)) + 1
            except Exception:
                row_number = ""

            match_mask = pd.Series(False, index=matched_index.index)

            if inchikey_value:
                match_mask = match_mask | (
                    matched_index["inchikey"].astype(str).str.strip() == inchikey_value
                )

            if canonical_smiles_value:
                match_mask = match_mask | (
                    matched_index["canonical_smiles"].astype(str).str.strip() == canonical_smiles_value
                )

            compound_matches = matched_index[
                match_mask
                & matched_index["_spectrum_type_norm"].isin(selected_spectrum_types)
            ].copy()

            for _, spectrum_row in compound_matches.iterrows():
                source_value = str(spectrum_row.get("source_database", "")).strip()

                if not source_value or source_value.lower() in ["nan", "none", "unknown"]:
                    source_value = str(spectrum_row.get("source", "")).strip()

                matched_spectra_rows.append({
                    "Строка": row_number,
                    "SMILES": input_smiles or canonical_smiles_value,
                    "InChIKey": inchikey_value,
                    "Тип спектра": spectrum_row.get("_spectrum_type_norm", ""),
                    "ID спектра": spectrum_row.get("spectrum_id", ""),
                    "Фаза": spectrum_row.get("phase", ""),
                    "Источник": source_value,
                })

    matched_spectra_df = pd.DataFrame(matched_spectra_rows)

    if not is_admin():
        st.subheader("Найденные спектры")

        if matched_spectra_df.empty:
            st.info("Для загруженных веществ совпавшие спектры в подключенной базе Augur не найдены.")
        else:
            st.dataframe(
                matched_spectra_df,
                width="stretch",
                hide_index=True
            )

            matched_csv = matched_spectra_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Скачать найденные спектры CSV",
                matched_csv,
                "matched_augur_spectra.csv",
                "text/csv",
                key="download_matched_augur_spectra_csv"
            )

    if is_admin():
        st.markdown(t('spectra.search_in_title'))

        st.caption(t('spectra.search_cascade'))

        st.markdown(t('spectra.main_sources_title'))

        col_src_1, col_src_2 = st.columns(2)

        with col_src_1:
            use_local_bank = st.checkbox(
                t('spectra.source_local_bank'),
                value=True,
                key="spectra_use_local_bank"
            )

            use_nist = st.checkbox(
                t('spectra.source_nist'),
                value=True,
                key="spectra_use_nist"
            )

            use_mona_mass = st.checkbox(
                t('spectra.source_mona'),
                value=True,
                key="spectra_use_mona_mass",
                help=t('spectra.source_mona_help')
            )

        with st.expander(t('spectra.extra_sources_title'), expanded=False):
            use_local_jdx_folder = st.checkbox(
                t('spectra.source_local_jdx'),
                value=False,
                key="spectra_use_local_jdx_folder",
                disabled=True,
                help=t('spectra.source_local_jdx_help')
            )

            use_spectralbench = st.checkbox(
                t('spectra.source_spectralbench'),
                value=False,
                key="spectra_use_spectralbench",
                disabled=True,
                help=t('spectra.source_spectralbench_help')
            )

            use_nist_epa_srd35 = st.checkbox(
                t('spectra.source_nist_epa'),
                value=False,
                key="spectra_use_nist_epa_srd35",
                disabled=True,
                help=t('spectra.source_nist_epa_help')
            )


    else:
        use_local_bank = True
        use_nist = False
        use_mona_mass = False
        st.caption("Используется только автоматически подключенная спектральная база Augur.")

    selected_sources = []

    if use_local_bank:
        selected_sources.append("local_bank")

    if search_ir and use_nist:
        selected_sources.append("nist_webbook")

    if search_mass and use_mona_mass:
        selected_sources.append("mona_mass")

    if not selected_sources:
        st.warning(t('spectra.warning_select_source'))
    else:
        source_labels = {
            "local_bank": t('spectra.source_label_local_bank'),
            "nist_webbook": t('spectra.source_label_nist'),
            "mona_mass": t('spectra.source_label_mona'),
        }

        if is_admin():
            st.caption(t('spectra.search_order', order=" → ".join([source_labels.get(s, s) for s in selected_sources])))

        if is_admin():
            with st.expander(t('spectra.cache_expander'), expanded=False):
                st.caption(t('spectra.cache_caption'))

                cache_df = spectra_load_search_cache()

                if cache_df.empty:
                    st.info(t('spectra.cache_empty'))
                else:
                    cache_view = cache_df.copy()

                    if "spectrum_type" not in cache_view.columns:
                        cache_view["spectrum_type"] = ""

                    cache_view["_spectrum_type_norm"] = cache_view["spectrum_type"].apply(
                        spectra_normalize_spectrum_type
                    )

                    n_cache_total = len(cache_view)
                    n_cache_ir = int((cache_view["_spectrum_type_norm"] == "IR").sum())
                    n_cache_mass = int((cache_view["_spectrum_type_norm"] == "Mass").sum())

                    cache_col_1, cache_col_2, cache_col_3 = st.columns(3)

                    with cache_col_1:
                        st.metric(t('spectra.cache_total'), n_cache_total)

                    with cache_col_2:
                        st.metric(t('spectra.cache_ir'), n_cache_ir)

                    with cache_col_3:
                        st.metric(t('spectra.cache_mass'), n_cache_mass)

                    st.dataframe(
                        cache_df.tail(100),
                        width="stretch",
                        hide_index=True
                    )

                st.markdown(t('spectra.clear_cache_title'))

                clear_col_1, clear_col_2, clear_col_3 = st.columns(3)

                with clear_col_1:
                    if st.button(t('spectra.clear_ir_button'), key="clear_spectra_search_cache_ir"):
                        try:
                            cache_df = spectra_load_search_cache()

                            if not cache_df.empty and "spectrum_type" in cache_df.columns:
                                work = cache_df.copy()
                                work["_spectrum_type_norm"] = work["spectrum_type"].apply(
                                    spectra_normalize_spectrum_type
                                )
                                cache_df = work[
                                    work["_spectrum_type_norm"] != "IR"
                                ].drop(columns=["_spectrum_type_norm"], errors="ignore")

                                spectra_save_search_cache(cache_df)

                            st.success(t('spectra.clear_ir_success'))
                            st.rerun()

                        except Exception as e:
                            st.error(t('spectra.clear_error', error=e))

                with clear_col_2:
                    if st.button(t('spectra.clear_mass_button'), key="clear_spectra_search_cache_mass"):
                        try:
                            cache_df = spectra_load_search_cache()

                            if not cache_df.empty and "spectrum_type" in cache_df.columns:
                                work = cache_df.copy()
                                work["_spectrum_type_norm"] = work["spectrum_type"].apply(
                                    spectra_normalize_spectrum_type
                                )
                                cache_df = work[
                                    work["_spectrum_type_norm"] != "Mass"
                                ].drop(columns=["_spectrum_type_norm"], errors="ignore")

                                spectra_save_search_cache(cache_df)

                            st.success(t('spectra.clear_mass_success'))
                            st.rerun()

                        except Exception as e:
                            st.error(t('spectra.clear_error', error=e))

                with clear_col_3:
                    if st.button(t('spectra.clear_all_button'), key="clear_spectra_search_cache_all"):
                        try:
                            cache_df = spectra_load_search_cache()
                            spectra_save_search_cache(pd.DataFrame(columns=cache_df.columns))

                            st.success(t('spectra.clear_all_success'))
                            st.rerun()

                        except Exception as e:
                            st.error(t('spectra.clear_error', error=e))

                st.markdown(t('spectra.clear_current_file_title'))

                if st.button(
                    t('spectra.clear_current_file_button'),
                    key="clear_spectra_search_cache_current_file"
                ):
                    try:
                        cache_df = spectra_load_search_cache()

                        if cache_df.empty:
                            st.info(t('spectra.cache_already_empty'))
                        else:
                            if "inchikey" not in cache_df.columns:
                                cache_df["inchikey"] = ""

                            if "spectrum_type" not in cache_df.columns:
                                cache_df["spectrum_type"] = ""

                            current_inchikeys = set()

                            if (
                                "compounds_for_search" in locals()
                                and compounds_for_search is not None
                                and not compounds_for_search.empty
                                and "inchikey" in compounds_for_search.columns
                            ):
                                current_inchikeys = set(
                                    compounds_for_search["inchikey"]
                                    .astype(str)
                                    .str.strip()
                                    .replace("", np.nan)
                                    .dropna()
                                    .tolist()
                                )

                            if not current_inchikeys:
                                st.warning(t('spectra.no_inchikey_warning'))
                            else:
                                work_cache = cache_df.copy()

                                work_cache["inchikey"] = (
                                    work_cache["inchikey"]
                                    .astype(str)
                                    .str.strip()
                                )

                                work_cache["_spectrum_type_norm"] = (
                                    work_cache["spectrum_type"]
                                    .astype(str)
                                    .apply(spectra_normalize_spectrum_type)
                                )

                                remove_mask = work_cache["inchikey"].isin(current_inchikeys)

                                if (
                                    "selected_spectrum_types" in locals()
                                    and selected_spectrum_types
                                ):
                                    selected_types_norm = [
                                        spectra_normalize_spectrum_type(x)
                                        for x in selected_spectrum_types
                                    ]

                                    remove_mask = remove_mask & work_cache[
                                        "_spectrum_type_norm"
                                    ].isin(selected_types_norm)

                                removed_count = int(remove_mask.sum())

                                new_cache_df = work_cache.loc[~remove_mask].copy()
                                new_cache_df = new_cache_df.drop(
                                    columns=["_spectrum_type_norm"],
                                    errors="ignore"
                                )

                                spectra_save_search_cache(new_cache_df)

                                st.success(t('spectra.clear_current_file_success',
                                    removed=removed_count,
                                    remaining=len(new_cache_df)
                                ))

                                add_log(t('spectra.clear_current_file_log', removed=removed_count))

                                st.rerun()

                    except Exception as e:
                        st.error(t('spectra.clear_current_file_error', error=e))


    if is_admin():
        only_missing = st.checkbox(
            t('spectra.only_missing_checkbox'),
            value=True,
            key="spectra_only_missing"
        )

        search_cache_mode = st.radio(
            t('spectra.cache_mode_radio'),
            [
                t('spectra.cache_mode_use'),
                t('spectra.cache_mode_ignore'),
                t('spectra.cache_mode_not_use')
            ],
            index=0,
            key="spectra_search_cache_mode",
            horizontal=False
        )

        use_search_cache = search_cache_mode == t('spectra.cache_mode_use')
        ignore_search_cache = search_cache_mode == t('spectra.cache_mode_ignore')

        if search_cache_mode == t('spectra.cache_mode_use'):
            st.caption(t('spectra.cache_mode_use_caption'))
        elif search_cache_mode == t('spectra.cache_mode_ignore'):
            st.caption(t('spectra.cache_mode_ignore_caption'))
        else:
            st.caption(t('spectra.cache_mode_not_use_caption'))

        total_compounds_for_spectra = len(compounds_for_search)

        if selected_spectrum_types == ["IR"]:
            missing_spectra_count = int((compounds_for_search["IR_status"] == t('spectra.status_no_ir')).sum())
        elif selected_spectrum_types == ["Mass"]:
            missing_spectra_count = int((compounds_for_search["Mass_status"] == t('spectra.status_no_mass')).sum())
        else:
            missing_spectra_count = int(
                (
                    (compounds_for_search["IR_status"] == t('spectra.status_no_ir')) |
                    (compounds_for_search["Mass_status"] == t('spectra.status_no_mass'))
                ).sum()
            )

        default_search_count = total_compounds_for_spectra
        default_search_count = max(1, default_search_count)

        max_to_search = st.number_input(
            t('spectra.max_to_search'),
            min_value=1,
            max_value=max(total_compounds_for_spectra, 1),
            value=default_search_count,
            step=1,
            key="spectra_max_to_search"
        )

        delay_seconds = st.number_input(
            t('spectra.delay_seconds'),
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key="spectra_delay_seconds"
        )

        if st.button(t('spectra.stop_search_button'), key="stop_spectra_search"):
            st.session_state.stop_spectra_search_requested = True
            st.session_state.spectra_search_status = "stopped_by_user"
            spectra_request_stop()

            stop_msg_placeholder = st.empty()

            stop_msg_placeholder.warning(t('spectra.stop_search_warning'))

            time.sleep(5)
            stop_msg_placeholder.empty()

        spectra_results_rendered_this_run = False

        if qspr_is_online_mode():
            qspr_online_lock_notice("Online spectral search and local spectral bank writes")

        if st.button(
            t('spectra.search_button'),
            type="primary",
            key="run_spectra_search",
            disabled=qspr_is_online_mode(),
        ):
            st.session_state.stop_spectra_search_requested = False
            st.session_state.spectra_search_status = "running"
            spectra_clear_stop()

            if not selected_sources:
                st.error(t('spectra.error_no_source'))
                st.stop()

            if not selected_spectrum_types:
                st.error(t('spectra.error_no_type'))
                st.stop()

            search_df = compounds_for_search.copy()

            # Лимит применяется к веществам, а не к задачам.
            # Если выбраны IR + Mass, 116 веществ должны дать до 232 задач.
            search_df = search_df.head(int(max_to_search)).copy()

            # ------------------------------------------------------------
            # Нормальная очередь поиска:
            # одна задача = одна структура + один тип спектра.
            #
            # Это защищает от ошибок вида:
            # - IR уже есть, но при выборе IR+Mass снова запускается IR;
            # - вещество уже проверялось и не найдено, но снова уходит в поиск;
            # - одинаковый InChIKey несколько раз скачивается в одном запуске.

            bank_by_inchikey = {}
            bank_by_smiles = {}

            try:
                bank_index_df = spectra_load_index()
            except Exception:
                bank_index_df = pd.DataFrame()

            if bank_index_df is not None and not bank_index_df.empty:
                bank_work = bank_index_df.copy()

                for col in [
                    "inchikey",
                    "canonical_smiles",
                    "spectrum_type",
                    "active",
                    "source",
                    "source_database",
                    "spectrum_id",
                    "raw_file",
                    "processed_file",
                ]:
                    if col not in bank_work.columns:
                        bank_work[col] = ""

                bank_work["inchikey"] = bank_work["inchikey"].astype(str).str.strip()
                bank_work["canonical_smiles"] = bank_work["canonical_smiles"].astype(str).str.strip()
                bank_work["_spectrum_type_norm"] = bank_work["spectrum_type"].astype(str).apply(
                    spectra_normalize_spectrum_type
                )

                active_values = ["true", "1", "yes", "y", "да", "active", ""]
                bank_work["_active_norm"] = (
                    bank_work["active"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(active_values)
                )

                bank_work = bank_work[bank_work["_active_norm"]].copy()

                for _, bank_row in bank_work.iterrows():
                    row_dict = bank_row.to_dict()
                    bank_type = str(row_dict.get("_spectrum_type_norm", "")).strip()
                    bank_inchikey = str(row_dict.get("inchikey", "")).strip()
                    bank_smiles = str(row_dict.get("canonical_smiles", "")).strip()

                    if bank_inchikey and bank_type:
                        bank_by_inchikey[(bank_inchikey, bank_type)] = row_dict

                    if bank_smiles and bank_type:
                        bank_by_smiles[(bank_smiles, bank_type)] = row_dict

            cache_by_inchikey = {}
            cache_by_smiles = {}

            if use_search_cache and not ignore_search_cache:
                try:
                    cache_df = spectra_load_search_cache()
                except Exception:
                    cache_df = pd.DataFrame()

                if cache_df is not None and not cache_df.empty:
                    cache_work = cache_df.copy()

                    for col in [
                        "inchikey",
                        "canonical_smiles",
                        "spectrum_type",
                        "final_status",
                        "selected_sources_key",
                        "selected_sources",
                        "selected_source",
                        "candidate_count",
                        "spectrum_id",
                        "raw_file",
                        "processed_file",
                        "message",
                        "date_checked",
                    ]:
                        if col not in cache_work.columns:
                            cache_work[col] = ""

                    cache_work["inchikey"] = cache_work["inchikey"].astype(str).str.strip()
                    cache_work["canonical_smiles"] = cache_work["canonical_smiles"].astype(str).str.strip()
                    cache_work["_spectrum_type_norm"] = cache_work["spectrum_type"].astype(str).apply(
                        spectra_normalize_spectrum_type
                    )

                    for _, cache_row in cache_work.iterrows():
                        row_dict = cache_row.to_dict()
                        cache_type = str(row_dict.get("_spectrum_type_norm", "")).strip()
                        cache_inchikey = str(row_dict.get("inchikey", "")).strip()
                        cache_smiles = str(row_dict.get("canonical_smiles", "")).strip()
                        
                        if not str(row_dict.get("selected_sources_key", "")).strip():
                            row_dict["selected_sources_key"] = (
                                str(row_dict.get("selected_sources", ""))
                                .replace(" | ", "|")
                                .strip()
                            )
                        
                        if cache_inchikey and cache_type:
                            cache_by_inchikey[(cache_inchikey, cache_type)] = row_dict

                        if cache_smiles and cache_type:
                            cache_by_smiles[(cache_smiles, cache_type)] = row_dict

            search_results = []
            skipped_results = []
            tasks = []
            queued_task_keys = set()

            for _, compound_row in search_df.iterrows():
                compound = compound_row.to_dict()

                inchikey_value = str(compound.get("inchikey", "")).strip()
                canonical_smiles_value = str(compound.get("canonical_smiles", "")).strip()
                structure_status_value = str(compound.get("structure_status", "")).strip()

                for spectrum_type in selected_spectrum_types:
                    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

                    base_result = {
                        "source_line_number": compound.get("source_line_number", compound.get("row_index", "")),
                        "compound_id": compound.get("compound_id", ""),
                        "name": compound.get("name", ""),
                        "cas": compound.get("cas", ""),
                        "input_smiles": compound.get("input_smiles", ""),
                        "canonical_smiles": canonical_smiles_value,
                        "inchikey": inchikey_value,
                        "structure_status": structure_status_value,
                        "spectrum_type": spectrum_type_norm,
                        "spectrum_status": "",
                        "selected_source": "",
                        "candidate_count": 0,
                        "spectrum_id": "",
                        "raw_file": "",
                        "processed_file": "",
                        "message": "",
                        "_from_real_search": False,
                    }

                    if structure_status_value != "ok":
                        base_result["spectrum_status"] = "invalid_structure"
                        base_result["message"] = t('spectra.message_invalid_structure')
                        skipped_results.append(base_result)
                        continue

                    bank_record = None

                    if inchikey_value:
                        bank_record = bank_by_inchikey.get(
                            (inchikey_value, spectrum_type_norm)
                        )

                    if bank_record is None and canonical_smiles_value:
                        bank_record = bank_by_smiles.get(
                            (canonical_smiles_value, spectrum_type_norm)
                        )

                    if bank_record is not None:
                        selected_source_value = (
                            bank_record.get("source_database", "")
                            or bank_record.get("source", "")
                            or "local_bank"
                        )

                        base_result.update({
                            "spectrum_status": "already_in_bank",
                            "selected_source": selected_source_value,
                            "candidate_count": 1,
                            "spectrum_id": bank_record.get("spectrum_id", ""),
                            "raw_file": bank_record.get("raw_file", ""),
                            "processed_file": bank_record.get("processed_file", ""),
                            "message": t('spectra.message_skipped_in_bank', spectrum_type=spectrum_type_norm),
                        })
                        skipped_results.append(base_result)
                        continue

                    cache_record = None

                    if use_search_cache and not ignore_search_cache:
                        if inchikey_value:
                            cache_record = cache_by_inchikey.get(
                                (inchikey_value, spectrum_type_norm)
                            )

                        if cache_record is None and canonical_smiles_value:
                            cache_record = cache_by_smiles.get(
                                (canonical_smiles_value, spectrum_type_norm)
                            )

                    if cache_record is not None:
                        cached_final_status = (
                            cache_record.get("final_status", "")
                            or cache_record.get("spectrum_status", "")
                            or "already_checked"
                        )

                        cached_final_status_norm = str(cached_final_status).strip()
                        current_sources_key = spectra_make_sources_key(selected_sources)
                        cached_sources_key = str(
                            cache_record.get("selected_sources_key", "")
                        ).strip()

                        if not cached_sources_key:
                            cached_sources_key = (
                                str(cache_record.get("selected_sources", ""))
                                .replace(" | ", "|")
                                .strip()
                            )

                        # Найденные спектры можно безопасно брать из журнала/банка.
                        # Отрицательные и ошибочные результаты для Mass/MoNA не должны
                        # блокировать повторный поиск: MoNA обновляется, а старый баг
                        # HTTP-функции мог уже записать ложный not_found_in_all_sources.
                        cache_success_statuses = {
                            "found_downloaded",
                            "already_in_bank",
                        }
                        retryable_cache_statuses = {
                            "not_found_in_all_sources",
                            "candidate_link_found",
                            "parse_error",
                            "download_error",
                            "no_numeric_spectrum",
                            "search_error",
                            "mona_records_found_but_not_parsed",
                        }

                        should_skip_by_cache = cached_final_status_norm in cache_success_statuses

                        if not should_skip_by_cache:
                            same_sources = cached_sources_key == current_sources_key
                            is_mona_mass_retry = False

                            should_skip_by_cache = same_sources and not is_mona_mass_retry

                        if should_skip_by_cache:
                            extra_msg = cache_record.get('message', '')
                            base_result.update({
                                "spectrum_status": "skipped_already_checked",
                                "selected_source": cache_record.get("selected_source", ""),
                                "candidate_count": cache_record.get("candidate_count", 0),
                                "spectrum_id": cache_record.get("spectrum_id", ""),
                                "raw_file": cache_record.get("raw_file", ""),
                                "processed_file": cache_record.get("processed_file", ""),
                                "message": t('spectra.message_skipped_by_cache',
                                    status=cached_final_status,
                                    extra_message=extra_msg
                                ),
                            })
                            skipped_results.append(base_result)
                            continue

                    structure_key = inchikey_value or canonical_smiles_value

                    if not structure_key:
                        structure_key = str(compound.get("row_index", ""))

                    queue_key = (
                        str(structure_key).strip(),
                        str(spectrum_type_norm).strip()
                    )

                    if queue_key in queued_task_keys:
                        base_result.update({
                            "spectrum_status": "skipped_duplicate_in_current_queue",
                            "selected_source": "current_queue",
                            "message": t('spectra.message_skipped_duplicate'),
                        })
                        skipped_results.append(base_result)
                        continue

                    queued_task_keys.add(queue_key)
                    tasks.append((compound, spectrum_type_norm))

            skipped_df = pd.DataFrame(skipped_results)

            n_skipped_bank = 0
            n_skipped_cache = 0
            n_skipped_duplicates = 0
            n_skipped_invalid = 0

            if not skipped_df.empty and "spectrum_status" in skipped_df.columns:
                n_skipped_bank = int((skipped_df["spectrum_status"] == "already_in_bank").sum())
                n_skipped_cache = int((skipped_df["spectrum_status"] == "skipped_already_checked").sum())
                n_skipped_duplicates = int((skipped_df["spectrum_status"] == "skipped_duplicate_in_current_queue").sum())
                n_skipped_invalid = int((skipped_df["spectrum_status"] == "invalid_structure").sum())

            st.success(t('spectra.success_tasks_sent', tasks=len(tasks)))

            with st.expander(t('spectra.expander_skipped_before_search'), expanded=False):
                n_skipped_total = (
                    n_skipped_bank
                    + n_skipped_cache
                    + n_skipped_duplicates
                    + n_skipped_invalid
                )

                col_skip_0, col_skip_1, col_skip_2, col_skip_3, col_skip_4 = st.columns(5)

                with col_skip_0:
                    st.metric(t('spectra.metric_total_skipped'), n_skipped_total)

                with col_skip_1:
                    st.metric(t('spectra.metric_skipped_in_bank'), n_skipped_bank)

                with col_skip_2:
                    st.metric(t('spectra.metric_skipped_in_cache'), n_skipped_cache)

                with col_skip_3:
                    st.metric(t('spectra.metric_skipped_duplicates'), n_skipped_duplicates)

                with col_skip_4:
                    st.metric(t('spectra.metric_skipped_invalid'), n_skipped_invalid)

                if not skipped_df.empty:
                    show_skip_cols = [
                        "source_line_number",
                        "name",
                        "canonical_smiles",
                        "inchikey",
                        "spectrum_type",
                        "spectrum_status",
                        "message",
                    ]

                    show_skip_cols = [
                        c for c in show_skip_cols
                        if c in skipped_df.columns
                    ]

                    st.dataframe(
                        skipped_df[show_skip_cols].head(300),
                        width="stretch",
                        hide_index=True
                    )

            if not tasks:
                result_df = pd.DataFrame(skipped_results)
                st.session_state.spectra_search_results = result_df
                st.session_state.spectra_search_status = "completed"
                st.session_state.spectra_search_total_tasks = len(result_df)

                no_tasks_msg_placeholder = st.empty()

                no_tasks_msg_placeholder.success(t('spectra.no_tasks_message'))

                time.sleep(5)
                no_tasks_msg_placeholder.empty()

            if tasks:
                preview_rows = []

                for compound, spectrum_type in tasks:
                    row = dict(compound)
                    row[t('spectra.col_what_to_search')] = spectrum_type
                    preview_rows.append(row)

                preview_df = pd.DataFrame(preview_rows)

                with st.expander(t('spectra.expander_check_queue'), expanded=False):
                    preview_cols = [
                        t('spectra.col_what_to_search'),
                        "source_line_number",
                        "cas",
                        "name",
                        "input_smiles",
                        "canonical_smiles",
                        "inchikey",
                        "IR_status",
                        "Mass_status",
                    ]

                    preview_cols = [
                        c for c in preview_cols
                        if c in preview_df.columns
                    ]

                    st.dataframe(
                        preview_df[preview_cols].head(100),
                        width="stretch",
                        hide_index=True
                    )

                progress = st.progress(0)
                col_status_ir, col_status_ms = st.columns(2)

                with col_status_ir:
                    status_box_ir = st.empty()

                with col_status_ms:
                    status_box_ms = st.empty()

                def _normalize_search_result(compound, spectrum_type_norm, result):
                    normalized = {
                        "source_line_number": compound.get("source_line_number", compound.get("row_index", "")),
                        "compound_id": compound.get("compound_id", ""),
                        "name": compound.get("name", ""),
                        "cas": compound.get("cas", ""),
                        "input_smiles": compound.get("input_smiles", ""),
                        "canonical_smiles": compound.get("canonical_smiles", ""),
                        "inchikey": compound.get("inchikey", ""),
                        "structure_status": compound.get("structure_status", ""),
                        "spectrum_type": spectrum_type_norm,
                        "spectrum_status": "",
                        "selected_source": "",
                        "candidate_count": 0,
                        "spectrum_id": "",
                        "raw_file": "",
                        "processed_file": "",
                        "candidate_url": "",
                        "message": "",
                        "_from_real_search": True,
                    }

                    if not isinstance(result, dict):
                        normalized["spectrum_status"] = "search_error"
                        normalized["message"] = t('spectra.search_error_result')
                        return normalized

                    normalized["spectrum_status"] = (
                        result.get("spectrum_status", "")
                        or result.get("status", "")
                        or result.get("final_status", "")
                    )

                    normalized["message"] = (
                        result.get("message", "")
                        or result.get("status_message", "")
                        or result.get("error", "")
                    )

                    normalized["selected_source"] = (
                        result.get("selected_source", "")
                        or result.get("source_database", "")
                        or result.get("source", "")
                    )

                    normalized["candidate_count"] = (
                        result.get("candidate_count", 0)
                        or result.get("n_candidates", 0)
                        or result.get("candidates_count", 0)
                    )

                    normalized["spectrum_id"] = result.get("spectrum_id", "") or result.get("id", "")
                    normalized["raw_file"] = (
                        result.get("raw_file", "")
                        or result.get("raw_path", "")
                        or result.get("raw_jdx_path", "")
                        or result.get("downloaded_file", "")
                        or result.get("file_path", "")
                    )
                    normalized["processed_file"] = (
                        result.get("processed_file", "")
                        or result.get("processed_path", "")
                        or result.get("processed_csv", "")
                    )
                    normalized["candidate_url"] = result.get("candidate_url", "")

                    for record_key in ["record", "spectrum_record", "index_record", "saved_record"]:
                        record = result.get(record_key, None)

                        if isinstance(record, dict):
                            normalized["selected_source"] = (
                                normalized["selected_source"]
                                or record.get("selected_source", "")
                                or record.get("source_database", "")
                                or record.get("source", "")
                            )
                            normalized["spectrum_id"] = (
                                normalized["spectrum_id"]
                                or record.get("spectrum_id", "")
                                or record.get("id", "")
                            )
                            normalized["raw_file"] = (
                                normalized["raw_file"]
                                or record.get("raw_file", "")
                                or record.get("raw_path", "")
                                or record.get("raw_jdx_path", "")
                                or record.get("downloaded_file", "")
                                or record.get("file_path", "")
                            )
                            normalized["processed_file"] = (
                                normalized["processed_file"]
                                or record.get("processed_file", "")
                                or record.get("processed_path", "")
                                or record.get("processed_csv", "")
                            )

                    return normalized

                def _spectra_search_task(compound, spectrum_type_norm):
                    try:
                        result = spectra_search_one_compound(
                            compound=compound,
                            spectrum_type=spectrum_type_norm,
                            selected_sources=selected_sources,
                            delay_seconds=delay_seconds
                        )

                        return _normalize_search_result(compound, spectrum_type_norm, result)

                    except Exception as e:
                        return {
                            "source_line_number": compound.get("source_line_number", compound.get("row_index", "")),
                            "compound_id": compound.get("compound_id", ""),
                            "name": compound.get("name", ""),
                            "cas": compound.get("cas", ""),
                            "input_smiles": compound.get("input_smiles", ""),
                            "canonical_smiles": compound.get("canonical_smiles", ""),
                            "inchikey": compound.get("inchikey", ""),
                            "structure_status": compound.get("structure_status", ""),
                            "spectrum_type": spectrum_type_norm,
                            "spectrum_status": "search_error",
                            "selected_source": "",
                            "candidate_count": 0,
                            "spectrum_id": "",
                            "raw_file": "",
                            "processed_file": "",
                            "message": str(e),
                            "_from_real_search": True,
                        }

                parallel_search_enabled = (
                    len(selected_spectrum_types) > 1
                    and "IR" in selected_spectrum_types
                    and "Mass" in selected_spectrum_types
                )

                max_workers = 2 if parallel_search_enabled else 1

                ir_total_tasks = sum(1 for _, spectrum_type in tasks if spectrum_type == "IR")
                ms_total_tasks = sum(1 for _, spectrum_type in tasks if spectrum_type == "Mass")

                ir_done_tasks = 0
                ms_done_tasks = 0
                done_tasks = 0
                total_tasks = len(tasks)
                st.session_state.spectra_search_total_tasks = len(skipped_results) + total_tasks

                if ir_total_tasks > 0:
                    status_box_ir.info(t('spectra.ir_search_started', total=ir_total_tasks))
                else:
                    status_box_ir.info(t('spectra.ir_search_not_selected'))

                if ms_total_tasks > 0:
                    status_box_ms.info(t('spectra.mass_search_started', total=ms_total_tasks))
                else:
                    status_box_ms.info(t('spectra.mass_search_not_selected'))

                stop_requested = False

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    task_iter = iter(tasks)
                    future_to_task = {}

                    def submit_next_task():
                        if (
                            st.session_state.get("stop_spectra_search_requested", False)
                            or spectra_is_stop_requested()
                        ):
                            return False

                        try:
                            compound_next, spectrum_type_next = next(task_iter)
                        except StopIteration:
                            return False

                        future = executor.submit(
                            _spectra_search_task,
                            compound_next,
                            spectrum_type_next
                        )

                        future_to_task[future] = (
                            compound_next,
                            spectrum_type_next
                        )

                        return True

                    # Стартуем только max_workers задач, а не всю очередь сразу.
                    for _ in range(max_workers):
                        if (
                            st.session_state.get("stop_spectra_search_requested", False)
                            or spectra_is_stop_requested()
                        ):
                            stop_requested = True
                            break

                        submitted = submit_next_task()

                        if not submitted:
                            break

                    while future_to_task:
                        for future in as_completed(list(future_to_task.keys())):
                            compound, spectrum_type = future_to_task.pop(future)

                            try:
                                result_row = future.result()
                            except Exception as e:
                                result_row = {
                                    "source_line_number": compound.get("source_line_number", compound.get("row_index", "")),
                                    "compound_id": compound.get("compound_id", ""),
                                    "name": compound.get("name", ""),
                                    "cas": compound.get("cas", ""),
                                    "input_smiles": compound.get("input_smiles", ""),
                                    "canonical_smiles": compound.get("canonical_smiles", ""),
                                    "inchikey": compound.get("inchikey", ""),
                                    "structure_status": compound.get("structure_status", ""),
                                    "spectrum_type": spectrum_type,
                                    "spectrum_status": "search_error",
                                    "selected_source": "",
                                    "candidate_count": 0,
                                    "spectrum_id": "",
                                    "raw_file": "",
                                    "processed_file": "",
                                    "message": str(e),
                                    "_from_real_search": True,
                                }

                            search_results.append(result_row)
                            partial_results = []
                            if skipped_results:
                                partial_results.extend(skipped_results)
                            partial_results.extend(search_results)
                            partial_df = pd.DataFrame(partial_results)
                            st.session_state.spectra_search_results = partial_df
                            st.session_state.spectra_search_status = "running"

                            # ------------------------------------------------------------
                            # Сразу пишем результат реального поиска в spectra_search_cache.
                            # Это важно: если поиск остановлен или Streamlit перезапустился,
                            # уже проверенные not_found_in_all_sources не должны потеряться.

                            result_status = str(result_row.get("spectrum_status", "")).strip()
                            result_message = str(result_row.get("message", "")).strip()
                            from_real_search = bool(result_row.get("_from_real_search", False))

                            if from_real_search:
                                should_cache_result = result_status in [
                                    "found_downloaded",
                                    "not_found_in_all_sources",
                                    "candidate_link_found",
                                    "already_in_bank",
                                    "parse_error",
                                    "download_error",
                                    "no_numeric_spectrum",
                                ]

                                if result_status == "search_error":
                                    if (
                                        "is not defined" not in result_message
                                        and "NameError" not in result_message
                                    ):
                                        should_cache_result = True

                                if result_status == "stopped_by_user":
                                    should_cache_result = False

                                if should_cache_result:
                                    try:
                                        spectra_add_to_search_cache(
                                            result_row,
                                            selected_sources
                                        )
                                    except Exception as cache_error:
                                        result_row["message"] = (
                                            str(result_row.get("message", ""))
                                            + t('spectra.cache_write_error', error=cache_error)
                                        )

                            done_tasks += 1

                            if spectrum_type == "IR":
                                ir_done_tasks += 1
                            elif spectrum_type == "Mass":
                                ms_done_tasks += 1

                            progress.progress(done_tasks / total_tasks)

                            compound_name = str(result_row.get("name", "")).strip() or t('spectra.unnamed')
                            compound_smiles = str(result_row.get("canonical_smiles", "")).strip()

                            if not compound_smiles:
                                compound_smiles = str(result_row.get("input_smiles", "")).strip()

                            compound_inchikey = str(result_row.get("inchikey", "")).strip()
                            compound_line = result_row.get("source_line_number", "")
                            compound_status = str(result_row.get("spectrum_status", "")).strip()
                            compound_source = str(result_row.get("selected_source", "")).strip() or "—"
                            compound_message = str(result_row.get("message", "")).strip() or "—"
                            compound_raw_file = str(result_row.get("raw_file", "")).strip() or "—"
                            compound_processed_file = str(result_row.get("processed_file", "")).strip() or "—"

                            details_lines = [
                                t('spectra.detail_last_compound') + f" {compound_name}",
                                t('spectra.detail_line') + f" {compound_line}",
                                t('spectra.detail_smiles') + f" `{compound_smiles}`",
                                t('spectra.detail_inchikey') + f" `{compound_inchikey}`",
                                t('spectra.detail_source') + f" {compound_source}",
                                t('spectra.detail_status') + f" {compound_status}",
                                t('spectra.detail_message') + f" {compound_message}",
                                t('spectra.detail_raw_file') + f" `{compound_raw_file}`",
                                t('spectra.detail_processed_file') + f" `{compound_processed_file}`"
                            ]
                            details = "\n\n".join(details_lines)

                            if spectrum_type == "IR":
                                status_box_ir.info(t('spectra.ir_search_progress', done=ir_done_tasks, total=ir_total_tasks) + "\n\n" + details)

                            elif spectrum_type == "Mass":
                                status_box_ms.info(t('spectra.mass_search_progress', done=ms_done_tasks, total=ms_total_tasks) + "\n\n" + details)

                            # После завершения задачи проверяем флаг остановки.
                            # Если остановка запрошена — новые задачи не ставим.
                            if (
                                st.session_state.get("stop_spectra_search_requested", False)
                                or spectra_is_stop_requested()
                            ):
                                stop_requested = True
                            else:
                                submit_next_task()

                            break

                all_results = []

                if skipped_results:
                    all_results.extend(skipped_results)

                if search_results:
                    all_results.extend(search_results)

                result_df = pd.DataFrame(all_results)
                st.session_state.spectra_search_results = result_df

                if stop_requested:
                    st.session_state.spectra_search_status = "stopped_by_user"
                    st.warning(t('spectra.stop_warning'))
                else:
                    st.session_state.spectra_search_status = "completed"
                    st.success(t('spectra.search_completed'))

                if "spectra_search_results" in st.session_state:
                    st.subheader(t('spectra.results_subheader'))

                    if st.button(t('spectra.clear_results_button'), key="clear_spectra_search_results"):
                        del st.session_state.spectra_search_results
                        st.rerun()

                    search_results_df = st.session_state.spectra_search_results.copy()

                    if search_results_df.empty:
                        st.info(t('spectra.no_results_yet'))
                    elif "spectrum_status" not in search_results_df.columns:
                        st.warning(t('spectra.old_or_invalid_result'))
                        st.dataframe(search_results_df, width="stretch")
                    else:
                        total_checked = len(search_results_df)
            

                for col in [
                    "spectrum_status",
                    "spectrum_type",
                    "inchikey",
                    "canonical_smiles",
                ]:
                    if col not in search_results_df.columns:
                        search_results_df[col] = ""

                search_results_df["_spectrum_status_norm"] = (
                    search_results_df["spectrum_status"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                search_results_df["_spectrum_type_norm"] = (
                    search_results_df["spectrum_type"]
                    .astype(str)
                    .str.strip()
                    .apply(spectra_normalize_spectrum_type)
                )

                search_results_df["_compound_key"] = (
                    search_results_df["inchikey"]
                    .astype(str)
                    .str.strip()
                )

                empty_key_mask = search_results_df["_compound_key"] == ""

                search_results_df.loc[empty_key_mask, "_compound_key"] = (
                    search_results_df.loc[empty_key_mask, "canonical_smiles"]
                    .astype(str)
                    .str.strip()
                )

                found_statuses = [
                    "found_downloaded",
                    "already_in_bank",
                ]

                found_rows_df = search_results_df[
                    search_results_df["_spectrum_status_norm"].isin(found_statuses)
                ].copy()

                found_ir_rows = found_rows_df[
                    found_rows_df["_spectrum_type_norm"] == "IR"
                ].copy()

                found_mass_rows = found_rows_df[
                    found_rows_df["_spectrum_type_norm"] == "Mass"
                ].copy()

                found_ir_spectra = int(len(found_ir_rows))
                found_mass_spectra = int(len(found_mass_rows))

                found_ir_keys = set(
                    found_ir_rows["_compound_key"]
                    .astype(str)
                    .str.strip()
                    .replace("", np.nan)
                    .dropna()
                )

                found_mass_keys = set(
                    found_mass_rows["_compound_key"]
                    .astype(str)
                    .str.strip()
                    .replace("", np.nan)
                    .dropna()
                )

                found_both_keys = found_ir_keys & found_mass_keys
                found_any_keys = found_ir_keys | found_mass_keys

                found_downloaded = int(
                    (search_results_df["_spectrum_status_norm"] == "found_downloaded").sum()
                )

                already_in_bank = int(
                    (search_results_df["_spectrum_status_norm"] == "already_in_bank").sum()
                )

                not_found = int(
                    (search_results_df["_spectrum_status_norm"] == "not_found_in_all_sources").sum()
                )

                skipped_checked = int(
                    (search_results_df["_spectrum_status_norm"] == "skipped_already_checked").sum()
                )

                candidate_links = int(
                    (search_results_df["_spectrum_status_norm"] == "candidate_link_found").sum()
                )

                parse_errors = int(
                    (search_results_df["_spectrum_status_norm"] == "parse_error").sum()
                )

                download_errors = int(
                    (search_results_df["_spectrum_status_norm"] == "download_error").sum()
                )

                invalid_structures = int(
                    (search_results_df["_spectrum_status_norm"] == "invalid_structure").sum()
                )

                no_numeric_spectrum = int(
                    (search_results_df["_spectrum_status_norm"] == "no_numeric_spectrum").sum()
                )

                summary_cards = pd.DataFrame({
                    t('spectra.summary_indicator'): [
                        t('spectra.summary_checked_rows'),
                        t('spectra.summary_downloaded_total'),
                        t('spectra.summary_downloaded_ir'),
                        t('spectra.summary_downloaded_mass'),
                        t('spectra.summary_compounds_both'),
                        t('spectra.summary_compounds_any'),
                        t('spectra.summary_compounds_ir_only'),
                        t('spectra.summary_compounds_mass_only'),
                        t('spectra.summary_already_in_bank'),
                        t('spectra.summary_candidate_links'),
                        t('spectra.summary_not_found'),
                        t('spectra.summary_no_numeric'),
                        t('spectra.summary_parse_errors'),
                        t('spectra.summary_download_errors'),
                        t('spectra.summary_invalid_structures'),
                        t('spectra.summary_skipped_checked'),
                    ],
                    t('spectra.summary_value'): [
                        total_checked,
                        found_downloaded,
                        found_ir_spectra,
                        found_mass_spectra,
                        len(found_both_keys),
                        len(found_any_keys),
                        len(found_ir_keys - found_mass_keys),
                        len(found_mass_keys - found_ir_keys),
                        already_in_bank,
                        candidate_links,
                        not_found,
                        no_numeric_spectrum,
                        parse_errors,
                        download_errors,
                        invalid_structures,
                        skipped_checked,
                    ]
                })

                card_cols = st.columns(4)

                for i, row in summary_cards.iterrows():
                    with card_cols[i % 4]:
                        st.markdown(
                            f"""
                            <div style="
                                border: 1px solid rgba(128,128,128,0.35);
                                border-radius: 10px;
                                padding: 12px 14px;
                                margin-bottom: 10px;
                                min-height: 82px;
                                background: rgba(255,255,255,0.03);
                            ">
                                <div style="font-size: 13px; opacity: 0.78; min-height: 32px;">
                                    {row[t('spectra.summary_indicator')]}
                                </div>
                                <div style="font-size: 28px; font-weight: 700; margin-top: 4px;">
                                    {row[t('spectra.summary_value')]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.subheader(t('spectra.status_summary_subheader'))

                status_summary = (
                    search_results_df
                    .groupby("spectrum_status")
                    .size()
                    .reset_index(name=t('spectra.status_count'))
                    .rename(columns={"spectrum_status": t('spectra.status_label')})
                    .sort_values(t('spectra.status_count'), ascending=False)
                )

                st.dataframe(status_summary, width="stretch", hide_index=True)

                if "spectrum_type" in search_results_df.columns:
                    st.subheader(t('spectra.type_summary_subheader'))
                    type_summary = (
                        search_results_df
                        .groupby(["spectrum_type", "spectrum_status"])
                        .size()
                        .reset_index(name=t('spectra.status_count'))
                        .sort_values(["spectrum_type", t('spectra.status_count')], ascending=[True, False])
                    )
                    st.dataframe(type_summary, width="stretch", hide_index=True)

                with st.expander(t('spectra.show_full_table_expander')):
                    st.dataframe(
                        search_results_df.drop(
                            columns=["_from_real_search", "_from_real_search_bool"],
                            errors="ignore",
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                csv_search = search_results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    t('spectra.download_search_results'),
                    csv_search,
                    "spectra_search_results.csv",
                    "text/csv"
                )

                if st.button(t('spectra.refresh_spectra_bank_button'), key="refresh_spectra_status_after_search"):
                    st.rerun()

                spectra_results_rendered_this_run = True

        if not spectra_results_rendered_this_run:
            render_spectra_search_results_if_available()

        st.divider()



    else:
        st.info(
            "Используется автоматически подключенная спектральная база Augur. "
            "Внешний поиск, управление cache и очистка служебных результатов скрыты в обычном режиме."
        )

    st.subheader(t('spectra_desc.subheader'))

    st.markdown(t('spectra_desc.description'))

    spectral_admin_mode = is_admin()
    spectrum_type_options = ["IR", "Mass"] if spectral_admin_mode else ["IR", "Mass", "IR + Mass"]
    descriptor_spectrum_selection = st.selectbox(
        t('spectra_desc.spectrum_type_select'),
        spectrum_type_options,
        index=0,
        key="spectral_descriptor_type"
    )
    descriptor_spectrum_types = (
        ["IR", "Mass"]
        if descriptor_spectrum_selection == "IR + Mass"
        else [descriptor_spectrum_selection]
    )
    descriptor_spectrum_type = descriptor_spectrum_types[0]
    descriptor_spectrum_label = (
        "IR + Mass"
        if len(descriptor_spectrum_types) > 1
        else descriptor_spectrum_type
    )

    # ------------------------------------------------------------
    # Методический режим

    if spectral_admin_mode:
        descriptor_method_mode = st.radio(
            t('spectra_desc.method_mode_radio'),
            [
                t('spectra_desc.mode_vazhev'),
                t('spectra_desc.mode_compressed'),
                t('spectra_desc.mode_custom'),
            ],
            index=0,
            key="spectral_descriptor_method_mode",
            help=t('spectra_desc.method_mode_help')
        )

        if descriptor_method_mode == t('spectra_desc.mode_vazhev'):
            st.info(t('spectra_desc.info_vazhev'))
        elif descriptor_method_mode == t('spectra_desc.mode_compressed'):
            st.info(t('spectra_desc.info_compressed'))
        else:
            st.info(t('spectra_desc.info_custom'))
    else:
        descriptor_method_mode = t('spectra_desc.mode_vazhev')

    # ------------------------------------------------------------
    # Отбор спектров

    st.markdown(t('spectra_desc.selection_title'))

    if descriptor_method_mode == t('spectra_desc.mode_vazhev'):
        default_phase_mode_label = t('spectra_desc.phase_only_gas')
    else:
        default_phase_mode_label = t('spectra_desc.phase_prefer_gas')

    phase_options = [
        t('spectra_desc.phase_prefer_gas'),
        t('spectra_desc.phase_only_gas'),
        t('spectra_desc.phase_any_active'),
        t('spectra_desc.phase_manual'),
    ]

    spectral_phase_mode_label = st.radio(
        t('spectra_desc.phase_radio_label'),
        phase_options,
        index=phase_options.index(default_phase_mode_label),
        key="spectral_phase_mode_label",
        help=t('spectra_desc.phase_radio_help')
    )

    phase_mode_map = {
        t('spectra_desc.phase_prefer_gas'): "prefer_gas",
        t('spectra_desc.phase_only_gas'): "only_gas",
        t('spectra_desc.phase_any_active'): "any",
        t('spectra_desc.phase_manual'): "manual",
    }

    spectral_phase_mode = phase_mode_map.get(
        spectral_phase_mode_label,
        "prefer_gas"
    )

    allowed_phases_for_desc = None
    phase_checkbox_options = [
        ("gas", "gas"),
        ("liquid", "liquid"),
        ("solid", "solid"),
        ("solution", "solution"),
        ("film", "film"),
        ("kbr", "KBr"),
        ("nujol", "Nujol"),
        ("unknown", "unknown"),
    ]

    if spectral_phase_mode in ["manual", "any"]:
        st.markdown("Разрешённые фазы / состояния образца")
        st.caption(
            "Этот выбор применяется и к готовому файлу спектральных дескрипторов: "
            "будут взяты только строки с отмеченными фазами."
        )

        allowed_phases_for_desc = []
        phase_cols = st.columns(4)
        default_all_phases = spectral_phase_mode == "any"

        for phase_idx, (phase_value, phase_label) in enumerate(phase_checkbox_options):
            with phase_cols[phase_idx % len(phase_cols)]:
                phase_checked = st.checkbox(
                    phase_label,
                    value=default_all_phases or phase_value == "gas",
                    key=f"spectral_allowed_phase_checkbox_{phase_value}_{spectral_phase_mode}"
                )

            if phase_checked:
                allowed_phases_for_desc.append(phase_value)

        spectral_phase_mode = "manual"

        if not allowed_phases_for_desc:
            st.warning(t('spectra_desc.phase_manual_warning'))

    elif spectral_phase_mode == "only_gas":
        allowed_phases_for_desc = ["gas"]

    if spectral_phase_mode == "manual":
        st.info(
            "Будут использоваться только спектры или готовые дескрипторы отмеченных фаз: "
            + ", ".join(allowed_phases_for_desc or [])
        )
    elif spectral_phase_mode == "prefer_gas":
        st.info(t('spectra_desc.phase_prefer_info'))
    elif spectral_phase_mode == "only_gas":
        st.info(t('spectra_desc.phase_only_info'))

    # ------------------------------------------------------------
    # Источники и типы интенсивности

    idx_for_desc_filters = spectra_load_index()

    available_spectral_sources = []
    available_intensity_types = []

    if idx_for_desc_filters is not None and not idx_for_desc_filters.empty:
        temp_idx = idx_for_desc_filters.copy()

        if "spectrum_type" not in temp_idx.columns:
            temp_idx["spectrum_type"] = ""

        temp_idx["_spectrum_type_norm"] = temp_idx["spectrum_type"].apply(
            spectra_normalize_spectrum_type
        )

        selected_type_norms = [
            spectra_normalize_spectrum_type(x)
            for x in descriptor_spectrum_types
        ]
        temp_idx = temp_idx[
            temp_idx["_spectrum_type_norm"].isin(selected_type_norms)
        ].copy()

        if "source" in temp_idx.columns:
            available_spectral_sources = (
                temp_idx["source"]
                .astype(str)
                .str.strip()
                .replace("", "unknown")
                .dropna()
                .unique()
                .tolist()
            )

        if "intensity_type" in temp_idx.columns:
            available_intensity_types = (
                temp_idx["intensity_type"]
                .astype(str)
                .str.strip()
                .replace("", "unknown")
                .dropna()
                .unique()
                .tolist()
            )

    available_spectral_sources = sorted([
        x for x in available_spectral_sources
        if str(x).strip() != ""
    ])

    available_intensity_types = sorted([
        x for x in available_intensity_types
        if str(x).strip() != ""
    ])

    if not available_spectral_sources:
        available_spectral_sources = ["unknown"]

    if not available_intensity_types:
        available_intensity_types = ["unknown"]

    if spectral_admin_mode:
        with st.expander(t('spectra_desc.extended_filters_expander'), expanded=False):
            selected_sources_for_desc = st.multiselect(
                t('spectra_desc.allowed_sources_label'),
                options=available_spectral_sources,
                default=available_spectral_sources,
                key="spectral_selected_sources_for_desc"
            )

            selected_intensity_types_for_desc = st.multiselect(
                t('spectra_desc.allowed_intensity_types_label'),
                options=available_intensity_types,
                default=available_intensity_types,
                key="spectral_selected_intensity_types_for_desc"
            )

            experimental_only_for_desc = st.checkbox(
                t('spectra_desc.experimental_only_checkbox'),
                value=True,
                key="spectral_experimental_only_for_desc"
            )

            prefer_quantitative_for_desc = st.checkbox(
                t('spectra_desc.prefer_quantitative_checkbox'),
                value=False,
                key="spectral_prefer_quantitative_for_desc",
                help=t('spectra_desc.prefer_quantitative_help')
            )
    else:
        selected_sources_for_desc = available_spectral_sources
        selected_intensity_types_for_desc = available_intensity_types
        experimental_only_for_desc = True
        prefer_quantitative_for_desc = False
        
    # ------------------------------------------------------------
    # Сетка и нормировка

    if descriptor_spectrum_type == "IR":
        axis_label = t('spectra_desc.axis_label_ir')
        prefix_label = "IR"
        min_allowed = 100
        max_allowed = 5000

        if descriptor_method_mode == t('spectra_desc.mode_vazhev'):
            default_min = 550
            default_max = 3798
            default_step = 4
            default_norm = "sum"
            default_invert = False
        elif descriptor_method_mode == t('spectra_desc.mode_compressed'):
            default_min = 550
            default_max = 3800
            default_step = 8
            default_norm = "vector"
            default_invert = False
        else:
            default_min = 550
            default_max = 3798
            default_step = 4
            default_norm = "min-max"
            default_invert = False

    else:
        axis_label = t('spectra_desc.axis_label_mass')
        prefix_label = "Mass"
        min_allowed = 1
        max_allowed = 2000

        if descriptor_method_mode == t('spectra_desc.mode_vazhev'):
            default_min = 1
            default_max = 300
            default_step = 1
            default_norm = "sum"
            default_invert = False
        elif descriptor_method_mode == t('spectra_desc.mode_compressed'):
            default_min = 1
            default_max = 500
            default_step = 1
            default_norm = "vector"
            default_invert = False
        else:
            default_min = 1
            default_max = 500
            default_step = 1
            default_norm = "min-max"
            default_invert = False

    if spectral_admin_mode:
        st.markdown(t('spectra_desc.grid_title'))

        col_grid_1, col_grid_2, col_grid_3 = st.columns(3)

        with col_grid_1:
            wn_min = st.number_input(
                t('spectra_desc.axis_min', label=axis_label),
                min_value=min_allowed,
                max_value=max_allowed,
                value=default_min,
                step=1 if descriptor_spectrum_type == "Mass" else 10,
                key=f"{descriptor_spectrum_type.lower()}_desc_axis_min"
            )

        with col_grid_2:
            wn_max = st.number_input(
                t('spectra_desc.axis_max', label=axis_label),
                min_value=min_allowed,
                max_value=max_allowed,
                value=default_max,
                step=1 if descriptor_spectrum_type == "Mass" else 10,
                key=f"{descriptor_spectrum_type.lower()}_desc_axis_max"
            )

        with col_grid_3:
            wn_step = st.number_input(
                t('spectra_desc.axis_step', label=axis_label),
                min_value=1,
                max_value=100,
                value=default_step,
                step=1,
                key=f"{descriptor_spectrum_type.lower()}_desc_axis_step"
            )

        norm_options = ["min-max", "sum", "vector", "none"]

        normalization = st.selectbox(
            t('spectra_desc.normalization_label'),
            norm_options,
            index=norm_options.index(default_norm),
            key=f"{descriptor_spectrum_type.lower()}_desc_normalization"
        )

        invert_signal = st.checkbox(
            t('spectra_desc.invert_signal_label'),
            value=default_invert,
            key=f"{descriptor_spectrum_type.lower()}_desc_invert_signal",
            help=t('spectra_desc.invert_signal_help')
        )

        show_markdown_help(
            t('spectra_desc.method_help_title'),
            os.path.join(HELP_DIR, "spectral_descriptor_method_help.md"),
            expanded=False
        )
    else:
        wn_min = default_min
        wn_max = default_max
        wn_step = default_step
        normalization = default_norm
        invert_signal = default_invert

    # ------------------------------------------------------------
    # Типы создаваемых дескрипторов

    if descriptor_method_mode == t('spectra_desc.mode_vazhev'):
        default_use_grid = True
        default_use_binary = True
        default_use_bands = True
        default_use_svd = True
    elif descriptor_method_mode == t('spectra_desc.mode_compressed'):
        default_use_grid = True
        default_use_binary = True
        default_use_bands = True
        default_use_svd = True
    else:
        default_use_grid = True
        default_use_binary = True
        default_use_bands = True
        default_use_svd = True

    def spectral_ready_defaults_for_type(spectrum_type_name):
        spectrum_type_name = spectra_normalize_spectrum_type(spectrum_type_name)

        if spectrum_type_name == "Mass":
            return {
                "wn_min": 1,
                "wn_max": 300,
                "wn_step": 1,
                "normalization": "sum",
                "invert_signal": False,
                "binary_window": 2,
                "binary_threshold": 0.10,
                "numeric_window": 10,
                "svd_components": 20,
                "axis_label": t('spectra_desc.axis_label_mass'),
                "window_unit": t('spectra_desc.window_unit_mass'),
            }

        return {
            "wn_min": 550,
            "wn_max": 3798,
            "wn_step": 4,
            "normalization": "sum",
            "invert_signal": False,
            "binary_window": 20,
            "binary_threshold": 0.10,
            "numeric_window": 100,
            "svd_components": 20,
            "axis_label": t('spectra_desc.axis_label_ir'),
            "window_unit": t('spectra_desc.window_unit_ir'),
        }

    spectral_descriptor_configs = {}
    descriptor_section_start = 2

    for spectrum_type_idx, spectrum_type_name in enumerate(descriptor_spectrum_types):
        prefix_label = spectra_normalize_spectrum_type(spectrum_type_name)
        type_defaults = spectral_ready_defaults_for_type(prefix_label)
        section_number = descriptor_section_start + spectrum_type_idx

        if spectral_admin_mode:
            section_title = t('spectra_desc.descriptor_types_title')
        elif len(descriptor_spectrum_types) > 1:
            section_title = f"{section_number}. Какие {prefix_label}-дескрипторы использовать"
        else:
            section_title = "2. Какие дескрипторы использовать"

        st.markdown(section_title)

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            cfg_use_grid = st.checkbox(
                t('spectra_desc.checkbox_grid', prefix=prefix_label),
                value=default_use_grid,
                key=f"{prefix_label.lower()}_ready_desc_use_grid"
            )

            cfg_use_binary = st.checkbox(
                t('spectra_desc.checkbox_binary', prefix=prefix_label),
                value=default_use_binary,
                key=f"{prefix_label.lower()}_ready_desc_use_binary"
            )

        with col_d2:
            cfg_use_bands = st.checkbox(
                t('spectra_desc.checkbox_bands', prefix=prefix_label),
                value=default_use_bands,
                key=f"{prefix_label.lower()}_ready_desc_use_bands"
            )

            cfg_use_svd = st.checkbox(
                t('spectra_desc.checkbox_svd', prefix=prefix_label),
                value=default_use_svd,
                key=f"{prefix_label.lower()}_ready_desc_use_svd"
            )

        if not any([cfg_use_grid, cfg_use_binary, cfg_use_bands, cfg_use_svd]):
            st.warning(t('spectra_desc.warning_select_descriptor'))

        cfg = {
            "spectrum_type": prefix_label,
            "wn_min": type_defaults["wn_min"],
            "wn_max": type_defaults["wn_max"],
            "wn_step": type_defaults["wn_step"],
            "normalization": type_defaults["normalization"],
            "invert_signal": type_defaults["invert_signal"],
            "use_grid": cfg_use_grid,
            "use_binary": cfg_use_binary,
            "use_bands": cfg_use_bands,
            "use_svd": cfg_use_svd,
            "binary_window": type_defaults["binary_window"],
            "binary_threshold": type_defaults["binary_threshold"],
            "numeric_window": type_defaults["numeric_window"],
            "svd_components": type_defaults["svd_components"],
            "axis_label": type_defaults["axis_label"],
        }

        if spectral_admin_mode and len(descriptor_spectrum_types) == 1:
            cfg["wn_min"] = wn_min
            cfg["wn_max"] = wn_max
            cfg["wn_step"] = wn_step
            cfg["normalization"] = normalization
            cfg["invert_signal"] = invert_signal

            col_param_1, col_param_2, col_param_3 = st.columns(3)

            with col_param_1:
                cfg["binary_window"] = st.number_input(
                    t('spectra_desc.binary_window_label', unit=type_defaults["window_unit"]),
                    min_value=1,
                    max_value=500,
                    value=type_defaults["binary_window"],
                    step=1,
                    key=f"{prefix_label.lower()}_desc_binary_window"
                )

            with col_param_2:
                cfg["binary_threshold"] = st.number_input(
                    t('spectra_desc.binary_threshold_label'),
                    min_value=0.01,
                    max_value=1.00,
                    value=0.10,
                    step=0.01,
                    key=f"{prefix_label.lower()}_desc_binary_threshold"
                )

            with col_param_3:
                cfg["numeric_window"] = st.number_input(
                    t('spectra_desc.numeric_window_label', unit=type_defaults["window_unit"]),
                    min_value=1,
                    max_value=1000,
                    value=type_defaults["numeric_window"],
                    step=1,
                    key=f"{prefix_label.lower()}_desc_numeric_window"
                )

            cfg["svd_components"] = st.number_input(
                t('spectra_desc.svd_components_label'),
                min_value=1,
                max_value=100,
                value=10 if descriptor_method_mode != t('spectra_desc.mode_vazhev') else 20,
                step=1,
                key=f"{prefix_label.lower()}_desc_svd_components"
            )

        spectral_descriptor_configs[prefix_label] = cfg

    primary_config = spectral_descriptor_configs[descriptor_spectrum_type]
    use_grid_desc = primary_config["use_grid"]
    use_binary_fp = primary_config["use_binary"]
    use_binned_numeric = primary_config["use_bands"]
    use_svd_desc = primary_config["use_svd"]
    binary_window = primary_config["binary_window"]
    binary_threshold = primary_config["binary_threshold"]
    numeric_window = primary_config["numeric_window"]
    svd_components = primary_config["svd_components"]

    # ------------------------------------------------------------
    # Спарринг-свойство / контрольные колонки

    sparring_section_number = 2 + len(descriptor_spectrum_types)

    if spectral_admin_mode:
        st.markdown(t('spectra_desc.sparring_title'))
    else:
        st.markdown(f"{sparring_section_number}. Контрольные признаки для спарринг-проверки")

    if spectral_admin_mode:
        add_sparring_columns = st.checkbox(
            t('spectra_desc.sparring_checkbox'),
            value=True if descriptor_method_mode == t('spectra_desc.mode_vazhev') else False,
            key=f"{descriptor_spectrum_type.lower()}_desc_add_sparring_columns",
            help=t('spectra_desc.sparring_help')
        )
    else:
        add_sparring_columns = True

    st.caption(t('spectra_desc.sparring_caption'))

    # ------------------------------------------------------------
    # Запуск расчёта

    run_section_number = sparring_section_number + 1

    if spectral_admin_mode:
        st.markdown(t('spectra_desc.run_title'))
    else:
        st.markdown(f"{run_section_number}. Расчёт")

    run_admin_cache_all_spectra = False

    if is_admin():
        st.markdown("#### Админ: готовые спектральные дескрипторы для GitHub")
        st.caption(
            "Рассчитывает дескрипторы для всех локально имеющихся processed-спектров "
            "выбранного типа и сохраняет CSV-кэш в папке проекта. SVD сюда не входит, "
            "потому что SVD зависит от конкретного датасета пользователя."
        )
        admin_skip_existing_inchikey = st.checkbox(
            "Пропускать InChIKey, уже имеющиеся в файле-банке дескрипторов",
            value=True,
            key=f"admin_skip_existing_inchikey_{descriptor_spectrum_type.lower()}",
            help=(
                "Если включено, программа сначала читает существующий CSV-банк "
                "спектральных дескрипторов и считает только те spectra processed-файлы, "
                "чьего InChIKey ещё нет в банке при текущих настройках дескрипторов."
            )
        )
        admin_autosave_every = st.number_input(
            "Автосохранять файл-банк каждые N новых спектров",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key=f"admin_descriptor_cache_autosave_every_{descriptor_spectrum_type.lower()}",
            help="После каждого такого количества новых рассчитанных спектров CSV в корне проекта будет сразу обновлён."
        )
        run_admin_cache_all_spectra = st.button(
            f"Рассчитать готовые {descriptor_spectrum_type}-дескрипторы для всех имеющихся спектров",
            key=f"admin_cache_all_spectra_{descriptor_spectrum_type.lower()}",
            type="secondary"
        )

    run_ready_spectral_descriptors = st.button(
        "Использовать готовые спектральные дескрипторы",
        type="primary",
        key=f"use_ready_spectral_descriptors_{descriptor_spectrum_label.lower().replace(' ', '_').replace('+', 'plus')}",
        disabled=qspr_is_online_mode(),
        help=(
            "Берёт готовую таблицу спектральных дескрипторов из локального кэша "
            "или из GitHub raw URL, без скачивания файлов спектров."
        )
    )

    if run_admin_cache_all_spectra:
        if not any([use_grid_desc, use_binary_fp, use_binned_numeric]):
            st.error("Для готового кэша выберите хотя бы GRID, BIN или BAND. SVD отдельно в кэш не сохраняется.")
        elif wn_min >= wn_max:
            st.error(t('spectra_desc.error_axis_min_max'))
        else:
            admin_progress_bar = st.progress(0.0)
            admin_progress_status = st.empty()

            def update_admin_cache_progress(current, total, stage, payload):
                total = max(int(total or 0), 1)
                current = max(0, min(int(current or 0), total))
                admin_progress_bar.progress(current / total)

                payload = payload or {}
                spectrum_id_text = str(payload.get("spectrum_id", "")).strip()

                if stage == "loading_spectrum":
                    msg = f"Чтение локального processed-спектра: {current + 1}/{total}"
                elif stage == "skipped_existing_inchikey":
                    msg = f"InChIKey уже есть в банке, пропуск: {current}/{total}"
                elif stage == "done":
                    msg = f"Готовый дескриптор сохранён: {current}/{total}"
                elif stage == "autosaved":
                    msg = f"Файл-банк автосохранён: {current}/{total}"
                else:
                    msg = f"Подготовка кэша спектральных дескрипторов: {current}/{total}"

                if spectrum_id_text:
                    msg += " | " + spectrum_id_text

                admin_progress_status.info(msg)

            with st.spinner("Рассчитываем готовые спектральные дескрипторы для локального банка..."):
                cache_df_all, cache_report = spectral_build_descriptor_cache_for_all_indexed_spectra(
                    spectrum_type=descriptor_spectrum_type,
                    wn_min=wn_min,
                    wn_max=wn_max,
                    step=wn_step,
                    normalization=normalization,
                    invert_signal=invert_signal,
                    use_grid=use_grid_desc,
                    use_binary_fp=use_binary_fp,
                    use_binned_numeric=use_binned_numeric,
                    binary_window=binary_window,
                    binary_threshold=binary_threshold,
                    numeric_window=numeric_window,
                    active_only=True,
                    skip_existing_inchikey=admin_skip_existing_inchikey,
                    autosave_every=admin_autosave_every,
                    progress_callback=update_admin_cache_progress,
                )

            admin_progress_bar.progress(1.0)
            admin_progress_status.success("Подготовка готовых спектральных дескрипторов завершена.")

            st.session_state.spectral_admin_cache_report = cache_report

            st.success(
                f"Файл-банк спектральных дескрипторов обновлён сразу. "
                f"Было строк: {cache_report.get('cache_rows_before', 0)}; "
                f"рассчитано сейчас: {cache_report.get('cache_rows_added', 0)}; "
                f"стало строк после объединения: {cache_report.get('cache_rows_after', 0)}. "
                f"Файл: {cache_report.get('cache_file', '')}"
            )

            summary_rows = [
                {"Показатель": "Строк в индексе выбранного типа", "Значение": cache_report.get("total_index_rows", 0)},
                {"Показатель": "Успешно рассчитано", "Значение": cache_report.get("processed", 0)},
                {"Показатель": "Строк в банке до расчёта", "Значение": cache_report.get("cache_rows_before", 0)},
                {"Показатель": "Строк рассчитано сейчас", "Значение": cache_report.get("cache_rows_added", 0)},
                {"Показатель": "Строк в банке после объединения", "Значение": cache_report.get("cache_rows_after", 0)},
                {"Показатель": "Автосохранений файла", "Значение": cache_report.get("autosave_count", 0)},
                {"Показатель": "Интервал автосохранения", "Значение": cache_report.get("autosave_every", 0)},
                {"Показатель": "Пропущено: InChIKey уже есть", "Значение": cache_report.get("skipped_existing_inchikey", 0)},
                {"Показатель": "Нет локального processed_file", "Значение": cache_report.get("missing_processed_file", 0)},
                {"Показатель": "Ошибки чтения/парсинга", "Значение": cache_report.get("parse_errors", 0)},
                {"Показатель": "Пустая сетка", "Значение": cache_report.get("empty_grid", 0)},
                {"Показатель": "Файл кэша", "Значение": cache_report.get("cache_file", "")},
            ]
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

            if cache_df_all is not None and not cache_df_all.empty:
                st.dataframe(cache_df_all.head(50), width="stretch", hide_index=True)

    if run_ready_spectral_descriptors:
        use_ready_descriptor_cache_only = True
        selected_descriptor_configs = list(spectral_descriptor_configs.values())

        if any(
            not any([
                cfg["use_grid"],
                cfg["use_binary"],
                cfg["use_bands"],
                cfg["use_svd"],
            ])
            for cfg in selected_descriptor_configs
        ):
            st.error(t('spectra_desc.error_no_descriptor_type'))
        elif any(cfg["wn_min"] >= cfg["wn_max"] for cfg in selected_descriptor_configs):
            st.error(t('spectra_desc.error_axis_min_max'))
        else:
            progress_bar = st.progress(0.0)
            progress_status = st.empty()

            def update_spectral_progress(current, total, stage, payload):
                total = max(int(total or 0), 1)
                current = max(0, min(int(current or 0), total))
                progress_bar.progress(current / total)

                payload = payload or {}
                inchikey_text = str(payload.get("inchikey", "")).strip()
                spectrum_id_text = str(payload.get("spectrum_id", "")).strip()

                payload_spectrum_type = str(payload.get("spectrum_type", descriptor_spectrum_label)).strip()

                if stage == "loading_ready_descriptors":
                    msg = f"Поиск готовых {payload_spectrum_type}-дескрипторов: {current + 1}/{total}"
                elif stage == "no_ready_descriptors":
                    msg = f"Готовые дескрипторы не найдены: {current}/{total}"
                elif stage == "searching":
                    msg = f"Поиск подходящего {payload_spectrum_type}-спектра: {current + 1}/{total}"
                elif stage == "loading_spectrum":
                    msg = f"Скачивание/чтение спектра из банка: {current + 1}/{total}"
                elif stage == "done":
                    msg = f"Спектр обработан: {current}/{total}"
                elif stage == "no_spectrum":
                    msg = f"Подходящий спектр не найден: {current}/{total}"
                elif stage in ["parse_error", "empty_grid"]:
                    msg = f"Спектр найден, но не удалось подготовить дескрипторы: {current}/{total}"
                else:
                    msg = f"Обработка спектров: {current}/{total}"

                details = []

                if inchikey_text:
                    details.append(inchikey_text)

                if spectrum_id_text:
                    details.append(spectrum_id_text)

                if details:
                    msg += " | " + " | ".join(details)

                progress_status.info(msg)

            spinner_text = (
                "Подключаем готовые спектральные дескрипторы..."
            )

            with st.spinner(spinner_text):
                def run_one_spectral_descriptor_builder(cfg):
                    spectral_builder = spectral_build_descriptors_from_ready_cache_for_dataset

                    def update_typed_progress(current, total, stage, payload):
                        payload = dict(payload or {})
                        payload["spectrum_type"] = cfg["spectrum_type"]
                        update_spectral_progress(current, total, stage, payload)

                    return spectral_builder(
                        input_df=current_df,
                        smiles_col=smiles_col_current,
                        spectrum_type=cfg["spectrum_type"],
                        wn_min=cfg["wn_min"],
                        wn_max=cfg["wn_max"],
                        step=cfg["wn_step"],
                        normalization=cfg["normalization"],
                        invert_signal=cfg["invert_signal"],
                        use_grid=cfg["use_grid"],
                        use_binary_fp=cfg["use_binary"],
                        use_binned_numeric=cfg["use_bands"],
                        binary_window=cfg["binary_window"],
                        binary_threshold=cfg["binary_threshold"],
                        numeric_window=cfg["numeric_window"],
                        use_svd=cfg["use_svd"],
                        svd_components=cfg["svd_components"],
                        spectrum_phase_mode=spectral_phase_mode,
                        allowed_phases=allowed_phases_for_desc,
                        allowed_sources=selected_sources_for_desc,
                        allowed_intensity_types=selected_intensity_types_for_desc,
                        prefer_quantitative=prefer_quantitative_for_desc,
                        experimental_only=experimental_only_for_desc,
                        progress_callback=update_typed_progress,
                    )

                def prepare_spectral_df_for_merge(df, spectrum_type_name):
                    if df is None or df.empty:
                        return pd.DataFrame()

                    key_cols = [
                        c for c in [
                            "row_index",
                            "compound_id",
                            "name",
                            "input_smiles",
                            "canonical_smiles",
                            "inchikey",
                        ]
                        if c in df.columns
                    ]
                    descriptor_prefixes = (
                        f"{spectrum_type_name}_GRID_",
                        f"{spectrum_type_name}_BIN_",
                        f"{spectrum_type_name}_BAND_",
                        f"{spectrum_type_name}_SVD_",
                    )
                    rename_cols = {
                        col: f"{spectrum_type_name}_{col}"
                        for col in df.columns
                        if col not in key_cols
                        and not any(str(col).startswith(prefix) for prefix in descriptor_prefixes)
                    }

                    return df.rename(columns=rename_cols)

                if len(selected_descriptor_configs) == 1:
                    descriptors_df, spectral_report = run_one_spectral_descriptor_builder(
                        selected_descriptor_configs[0]
                    )
                else:
                    partial_results = []
                    partial_reports = {}

                    for cfg in selected_descriptor_configs:
                        part_df, part_report = run_one_spectral_descriptor_builder(cfg)
                        partial_reports[cfg["spectrum_type"]] = part_report
                        partial_results.append(
                            (
                                cfg["spectrum_type"],
                                prepare_spectral_df_for_merge(part_df, cfg["spectrum_type"]),
                            )
                        )

                    key_cols = [
                        "row_index",
                        "compound_id",
                        "name",
                        "input_smiles",
                        "canonical_smiles",
                        "inchikey",
                    ]
                    descriptors_df = None

                    for spectrum_type_name, part_df in partial_results:
                        if part_df is None or part_df.empty:
                            descriptors_df = pd.DataFrame()
                            break

                        merge_keys = [
                            c for c in key_cols
                            if c in part_df.columns
                            and (
                                descriptors_df is None
                                or c in descriptors_df.columns
                            )
                        ]

                        if descriptors_df is None:
                            descriptors_df = part_df.copy()
                        else:
                            descriptors_df = descriptors_df.merge(
                                part_df,
                                on=merge_keys,
                                how="inner"
                            )

                    if descriptors_df is None:
                        descriptors_df = pd.DataFrame()

                    first_report = next(iter(partial_reports.values()), {})
                    spectral_report = {
                        "total_compounds": int(first_report.get("total_compounds", len(current_df))),
                        "with_spectrum": int(len(descriptors_df)),
                        "without_spectrum": max(
                            int(first_report.get("total_compounds", len(current_df))) - int(len(descriptors_df)),
                            0
                        ),
                        "parse_errors": sum(int(r.get("parse_errors", 0)) for r in partial_reports.values()),
                        "used_spectra": [],
                        "used_phases": {},
                        "spectrum_selection_reasons": {},
                        "descriptor_cache_hits": int(len(descriptors_df)),
                        "descriptor_cache_misses": max(
                            int(first_report.get("total_compounds", len(current_df))) - int(len(descriptors_df)),
                            0
                        ),
                        "ready_descriptor_mode": use_ready_descriptor_cache_only,
                        "combined_spectrum_types": list(partial_reports.keys()),
                        "combined_intersection_only": True,
                        "partial_reports": partial_reports,
                        "spectra_bank_status": first_report.get("spectra_bank_status", {}),
                    }

                    for spectrum_type_name, part_report in partial_reports.items():
                        spectral_report["used_spectra"].extend(part_report.get("used_spectra", []))

                        for phase_key, phase_count in part_report.get("used_phases", {}).items():
                            combined_key = f"{spectrum_type_name}: {phase_key}"
                            spectral_report["used_phases"][combined_key] = int(phase_count)

                        for reason_key, reason_count in part_report.get("spectrum_selection_reasons", {}).items():
                            combined_key = f"{spectrum_type_name}: {reason_key}"
                            spectral_report["spectrum_selection_reasons"][combined_key] = int(reason_count)

                progress_bar.progress(1.0)
                if use_ready_descriptor_cache_only:
                    progress_status.success("Готовые спектральные дескрипторы подключены.")
                else:
                    progress_status.success("Загрузка спектров и расчёт спектральных дескрипторов завершены.")

                if add_sparring_columns and descriptors_df is not None and not descriptors_df.empty:
                    molwt_rows = []

                    for _, row_desc in descriptors_df.iterrows():
                        smi = str(row_desc.get("canonical_smiles", "")).strip()

                        molwt = np.nan
                        exact_molwt = np.nan

                        try:
                            mol = Chem.MolFromSmiles(smi)

                            if mol is not None:
                                from rdkit.Chem import Descriptors

                                molwt = float(Descriptors.MolWt(mol))
                                exact_molwt = float(Descriptors.ExactMolWt(mol))
                        except Exception:
                            pass

                        molwt_rows.append({
                            "inchikey": row_desc.get("inchikey", ""),
                            "spectral_sparring_MolWt": molwt,
                            "spectral_sparring_ExactMolWt": exact_molwt,
                        })

                    molwt_df = pd.DataFrame(molwt_rows)

                    descriptors_df = descriptors_df.merge(
                        molwt_df,
                        on="inchikey",
                        how="left"
                    )

                    spectral_report["sparring_columns_added"] = True
                    spectral_report["sparring_columns"] = [
                        "spectral_sparring_MolWt",
                        "spectral_sparring_ExactMolWt",
                    ]
                else:
                    spectral_report["sparring_columns_added"] = False

                spectral_report["descriptor_method_mode"] = descriptor_method_mode
                spectral_report["descriptor_spectrum_type"] = descriptor_spectrum_label
                spectral_report["descriptor_type_configs"] = spectral_descriptor_configs
                spectral_report["axis_min"] = float(primary_config["wn_min"])
                spectral_report["axis_max"] = float(primary_config["wn_max"])
                spectral_report["axis_step"] = float(primary_config["wn_step"])
                spectral_report["axis_label"] = primary_config["axis_label"]
                spectral_report["normalization"] = primary_config["normalization"]
                spectral_report["invert_signal"] = bool(primary_config["invert_signal"])
                spectral_report["descriptor_input_mode"] = (
                    "ready_descriptor_cache"
                    if use_ready_descriptor_cache_only
                    else "spectra_bank_processed"
                )

                st.session_state.spectral_descriptors_df = descriptors_df
                st.session_state.spectral_descriptors_report = spectral_report
                st.session_state.keep_spectra_expander_open = True

                if descriptors_df is not None and not descriptors_df.empty:
                    saved_path = spectral_save_descriptors(
                        descriptors_df,
                        spectrum_type=descriptor_spectrum_label
                    )
                    st.session_state.spectral_descriptors_saved_path = saved_path

                    st.success(t('spectra_desc.success_calculation', count=len(descriptors_df)))
                else:
                    st.warning(t('spectra_desc.warning_empty_table'))

    if "spectral_descriptors_report" in st.session_state:
        st.subheader(t('spectra_report.subheader'))
        rep = st.session_state.spectral_descriptors_report

        col_rep1, col_rep2, col_rep3 = st.columns(3)

        with col_rep1:
            st.metric(t('spectra_report.total_compounds'), rep.get("total_compounds", 0))

        with col_rep2:
            st.metric(t('spectra_report.with_spectrum'), rep.get("with_spectrum", 0))

        with col_rep3:
            st.metric(t('spectra_report.without_spectrum'), rep.get("without_spectrum", 0))

        if rep.get("descriptor_cache_hits", 0) or rep.get("ready_descriptor_mode", False):
            col_cache_1, col_cache_2 = st.columns(2)

            with col_cache_1:
                st.metric("Готовые дескрипторы найдены", rep.get("descriptor_cache_hits", 0))

            with col_cache_2:
                st.metric("Готовые дескрипторы не найдены", rep.get("descriptor_cache_misses", 0))

        if rep.get("combined_intersection_only"):
            partial_reports = rep.get("partial_reports", {})
            detail_parts = []

            if isinstance(partial_reports, dict):
                for spectrum_type_name, part_report in partial_reports.items():
                    detail_parts.append(
                        f"{spectrum_type_name}: {part_report.get('descriptor_cache_hits', 0)}"
                    )

            detail_text = "; ".join(detail_parts)
            st.info(
                "Режим IR + Mass: в итоговую таблицу включены только вещества, "
                "для которых найдены оба набора готовых дескрипторов."
                + (f" Найдено по отдельности: {detail_text}." if detail_text else "")
            )

        bank_status = rep.get("spectra_bank_status", {})

        if isinstance(bank_status, dict) and rep.get("with_spectrum", 0) == 0:
            st.warning(
                "Спектральные дескрипторы не созданы, потому что для текущих веществ "
                "не удалось получить подходящие активные спектры из базы."
            )

            with st.expander("Диагностика подключения спектральной базы", expanded=True):
                diagnostic_rows = [
                    {
                        "Параметр": "spectra_index.csv найден",
                        "Значение": "да" if bank_status.get("index_exists") else "нет",
                    },
                    {
                        "Параметр": "Строк в spectra_index.csv",
                        "Значение": bank_status.get("index_rows", 0),
                    },
                    {
                        "Параметр": "IR-записей в индексе",
                        "Значение": bank_status.get("index_ir_rows", 0),
                    },
                    {
                        "Параметр": "Mass-записей в индексе",
                        "Значение": bank_status.get("index_mass_rows", 0),
                    },
                    {
                        "Параметр": "spectra_manifest.csv найден",
                        "Значение": "да" if bank_status.get("manifest_exists") else "нет",
                    },
                    {
                        "Параметр": "Строк в spectra_manifest.csv",
                        "Значение": bank_status.get("manifest_rows", 0),
                    },
                    {
                        "Параметр": "Processed-файлов в manifest",
                        "Значение": bank_status.get("manifest_processed_rows", 0),
                    },
                    {
                        "Параметр": "spectra_search_cache.csv найден",
                        "Значение": "да" if bank_status.get("search_cache_exists") else "нет",
                    },
                    {
                        "Параметр": "Строк в spectra_search_cache.csv",
                        "Значение": bank_status.get("search_cache_rows", 0),
                    },
                    {
                        "Параметр": "AUGUR_SPECTRA_INDEX_* настроен",
                        "Значение": "да" if bank_status.get("remote_index_configured") else "нет",
                    },
                    {
                        "Параметр": "AUGUR_SPECTRA_MANIFEST_* настроен",
                        "Значение": "да" if bank_status.get("remote_manifest_configured") else "нет",
                    },
                    {
                        "Параметр": "AUGUR_SPECTRA_SEARCH_CACHE_* настроен",
                        "Значение": "да" if bank_status.get("remote_search_cache_configured") else "нет",
                    },
                    {
                        "Параметр": "AUGUR_SPECTRA_*_DESCRIPTOR_CACHE_URL настроен",
                        "Значение": "да" if bank_status.get("remote_descriptor_cache_configured") else "нет",
                    },
                    {
                        "Параметр": "GitHub/raw spectral_descriptor_shards доступен",
                        "Значение": "да" if bank_status.get("remote_descriptor_shards_configured") else "нет",
                    },
                    {
                        "Параметр": "AUGUR_SPECTRA_BANK_FOLDER_* настроен",
                        "Значение": "да" if bank_status.get("remote_bank_folder_configured") else "нет",
                    },
                    {
                        "Параметр": "AUGUR_GOOGLE_DRIVE_API_KEY настроен",
                        "Значение": "да" if bank_status.get("google_drive_api_key_configured") else "нет",
                    },
                    {
                        "Параметр": "Локальных IR processed CSV",
                        "Значение": bank_status.get("local_ir_processed_files", 0),
                    },
                    {
                        "Параметр": "Локальных Mass processed CSV",
                        "Значение": bank_status.get("local_mass_processed_files", 0),
                    },
                    {
                        "Параметр": "Строк в IR descriptor cache",
                        "Значение": bank_status.get("local_ir_descriptor_cache_rows", 0),
                    },
                    {
                        "Параметр": "IR shard-файлов дескрипторов",
                        "Значение": bank_status.get("local_ir_descriptor_shard_files", 0),
                    },
                    {
                        "Параметр": "Строк в IR descriptor shards",
                        "Значение": bank_status.get("local_ir_descriptor_shard_rows", 0),
                    },
                    {
                        "Параметр": "Строк в Mass descriptor cache",
                        "Значение": bank_status.get("local_mass_descriptor_cache_rows", 0),
                    },
                    {
                        "Параметр": "Mass shard-файлов дескрипторов",
                        "Значение": bank_status.get("local_mass_descriptor_shard_files", 0),
                    },
                    {
                        "Параметр": "Строк в Mass descriptor shards",
                        "Значение": bank_status.get("local_mass_descriptor_shard_rows", 0),
                    },
                ]

                st.dataframe(
                    pd.DataFrame(diagnostic_rows),
                    width="stretch",
                    hide_index=True
                )

                if not bank_status.get("index_exists") or int(bank_status.get("index_rows", 0) or 0) == 0:
                    st.info(
                        "На Streamlit Cloud не найден рабочий `spectra_index.csv`. "
                        "Добавьте в Secrets `AUGUR_SPECTRA_INDEX_FILE_ID` или "
                        "`AUGUR_SPECTRA_INDEX_URL` для файла `spectra_index.csv`."
                    )
                elif not bank_status.get("manifest_exists") or int(bank_status.get("manifest_rows", 0) or 0) == 0:
                    st.info(
                        "Индекс найден, но нет `spectra_manifest.csv`. "
                        "Без manifest приложение видит записи спектров, но не знает, "
                        "какой файл Google Drive скачать для конкретного спектра."
                    )
                elif int(bank_status.get("manifest_processed_rows", 0) or 0) == 0:
                    st.info(
                        "`spectra_manifest.csv` найден, но в нём нет processed-файлов спектров. "
                        "Добавьте в Secrets `AUGUR_SPECTRA_BANK_FOLDER_ID` и "
                        "`AUGUR_GOOGLE_DRIVE_API_KEY`, чтобы приложение само обошло "
                        "Google Drive-папку и построило manifest по файлам `IR/processed` "
                        "и `Mass/processed`."
                    )
                else:
                    st.info(
                        "Индекс и manifest найдены. Если совпадений всё равно 0, "
                        "проверьте выбранный тип спектра, режим фазы, источники, "
                        "экспериментальность и наличие этих InChIKey/SMILES в индексе."
                    )
                    
        if rep.get("used_phases"):
            st.markdown(t('spectra_report.used_phases_title'))

            used_phases_df = pd.DataFrame([
                {
                    t('spectra_report.phase_state'): k,
                    t('spectra_report.count'): v
                }
                for k, v in rep.get("used_phases", {}).items()
            ])

            if not used_phases_df.empty:
                used_phases_df = used_phases_df.sort_values(
                    t('spectra_report.count'),
                    ascending=False
                )

                st.dataframe(
                    used_phases_df,
                    width="stretch",
                    hide_index=True
                )
            used_phases_dict = rep.get("used_phases", {})

            if isinstance(used_phases_dict, dict) and len(used_phases_dict) > 1:
                st.warning(t('spectra_report.warning_mixed_phases'))
                    
        if rep.get("spectrum_selection_reasons"):
            st.markdown(t('spectra_report.reasons_title'))

            reason_df = pd.DataFrame([
                {
                    t('spectra_report.reason'): spectral_selection_reason_to_ru(k),
                    t('spectra_report.technical_code'): k,
                    t('spectra_report.count'): v
                }
                for k, v in rep.get("spectrum_selection_reasons", {}).items()
            ])

            if not reason_df.empty:
                reason_df = reason_df.sort_values(
                    t('spectra_report.count'),
                    ascending=False
                )

                st.dataframe(
                    reason_df,
                    width="stretch",
                    hide_index=True
                )

        if "svd_components_created" in rep:
            st.info(t('spectra_report.svd_info',
                created=rep.get('svd_components_created'),
                variance=rep.get('svd_explained_variance_sum', 0)
            ))

        if "svd_error" in rep:
            st.warning(t('spectra_report.svd_error', error=rep['svd_error']))

    if "spectral_descriptors_df" in st.session_state:
        spectral_df = st.session_state.spectral_descriptors_df

        if spectral_df.empty:
            st.warning(t('spectra_report.no_descriptors'))
        else:
            st.subheader(t('spectra_report.table_subheader'))
            st.dataframe(spectral_df.head(100), width="stretch", hide_index=True)
            st.caption(t('spectra_report.table_size',
                rows=spectral_df.shape[0],
                cols=spectral_df.shape[1]
            ))

            csv_spec_desc = spectral_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                t('spectra_report.download_csv'),
                csv_spec_desc,
                "spectral_descriptors.csv",
                "text/csv"
            )

            if "spectral_descriptors_saved_path" in st.session_state:
                st.info(t('spectra_report.file_saved',
                    path=st.session_state.spectral_descriptors_saved_path
                ))

            st.markdown(t('spectra_report.transfer_title'))

            st.caption(t('spectra_report.transfer_caption'))

            if st.button(
                t('spectra_report.transfer_button'),
                type="primary",
                key="use_selected_descriptors_for_qspr"
            ):          
                try:
                    bundle = qspr_build_descriptor_matrix_from_sources(
                        current_df=current_df,
                        target_col=target_col,
                        use_molecular=False,
                        molecular_desc_df=None,
                        molecular_valid_indices=None,
                        use_spectral=True,
                        spectral_desc_df=spectral_df,
                        smiles_col=smiles_col_current,
                        restrict_to_spectral_subset=True,
                    )

                    store_descriptor_bundle(
                        bundle,
                        bundle["report"]["descriptor_source"]
                    )

                    st.session_state.desc_calculated = True
                    st.session_state.X_all = bundle["X_all"]
                    st.session_state.y_all = bundle["y_all"]
                    st.session_state.valid_indices = bundle["valid_indices"]
                    st.session_state.desc_names = bundle["desc_names"]
                    st.session_state.df_desc = bundle["df_desc"]
                    st.session_state.custom_descriptor_source = bundle["report"]["descriptor_source"]
                    st.session_state.custom_descriptors_used = True

                    st.session_state.pending_qspr_descriptor_bundle = bundle
                    st.session_state.pending_qspr_descriptor_bundle_ready = False
                    st.session_state.spectral_descriptors_transferred_ready = True
                    st.session_state.use_spectral_descriptors_source = True
                    st.session_state.keep_spectra_expander_open = True
                    st.session_state.descriptor_calculation_mode = "spectral_or_combined"

                    st.session_state.qspr_descriptor_matrix_ready_message = t('spectra_report.matrix_ready',
                        rows=bundle['X_all'].shape[0],
                        cols=bundle['X_all'].shape[1],
                        source=bundle['report']['descriptor_source']
                    )

                    st.session_state.pending_qspr_descriptor_bundle_message = (
                        st.session_state.qspr_descriptor_matrix_ready_message
                    )

                    st.session_state.spectral_qspr_match_info = bundle.get(
                        "match_info",
                        pd.DataFrame()
                    )

                    add_log(t('spectra_report.matrix_log',
                        source=bundle['report']['descriptor_source'],
                        rows=bundle['X_all'].shape[0],
                        cols=bundle['X_all'].shape[1]
                    ))

                    st.success(st.session_state.qspr_descriptor_matrix_ready_message)
                    st.rerun()

                except Exception as e:
                    st.error(t('spectra_report.matrix_error', error=e))

st.header(t('modeling.header'))
if st.session_state.get("qspr_descriptor_matrix_ready_message"):
    st.success(st.session_state.qspr_descriptor_matrix_ready_message)
st.caption(t('modeling.caption'))

# ------------------------------------------------------------------
# Descriptor source

st.subheader(t('descriptor_source.subheader'))

DESCRIPTOR_SOURCE_OPTIONS = {
    "calculate": "descriptor_source.option_calculate",
    "file": "descriptor_source.option_file",
}

# миграция старых значений из старых сессий
old_source_mode = st.session_state.get("descriptor_source_mode", "calculate")

if old_source_mode not in DESCRIPTOR_SOURCE_OPTIONS:
    old_text = str(old_source_mode)

    if old_text in {
        "Рассчитать дескрипторы программой",
        t("descriptor_source.option_calculate"),
        t("spectra.source_mode_calc"),
    }:
        st.session_state.descriptor_source_mode = "calculate"
    elif old_text in {
        "Использовать дескрипторы из файла",
        t("descriptor_source.option_file"),
    }:
        st.session_state.descriptor_source_mode = "file"
    else:
        st.session_state.descriptor_source_mode = "calculate"

descriptor_source_mode = st.radio(
    t("descriptor_source.radio_label"),
    options=list(DESCRIPTOR_SOURCE_OPTIONS.keys()),
    index=list(DESCRIPTOR_SOURCE_OPTIONS.keys()).index(
        st.session_state.get("descriptor_source_mode", "calculate")
    ),
    format_func=lambda key: t(DESCRIPTOR_SOURCE_OPTIONS[key]),
    key="descriptor_source_mode_radio_v2"
)

if descriptor_source_mode != st.session_state.descriptor_source_mode:
    previous_descriptor_source_mode = st.session_state.descriptor_source_mode

    keep_connected_matrix = (
        st.session_state.get("desc_calculated", False)
        and st.session_state.get("descriptor_calculation_mode", "") == "spectral_or_combined"
        and st.session_state.get("X_all") is not None
        and st.session_state.get("y_all") is not None
        and st.session_state.get("desc_names") is not None
    )

    st.session_state.descriptor_source_mode = descriptor_source_mode

    if keep_connected_matrix:
        add_log(t("descriptor_source.log_matrix_kept"))
    else:
        st.session_state.desc_calculated = False
        st.session_state.validation_done = False
        st.session_state.trained_models = {}
        st.session_state.holdout_results_dict = {}
        st.session_state.kfold_results_dict = {}
        st.session_state.loo_results_dict = {}

        add_log(t(
            "descriptor_source.log_source_changed",
            old=previous_descriptor_source_mode,
            new=descriptor_source_mode
        ))

if descriptor_source_mode == "file":
    st.info(t('descriptor_source.file_info'))

    excluded_cols = ["SMILES", target_col]
    optional_meta_cols = ["compound_id", "name", "CAS", "cas", "source", "units"]
    excluded_cols += [c for c in optional_meta_cols if c in data.columns]

    candidate_descriptor_cols = [c for c in data.columns if c not in excluded_cols]

    numeric_like_cols = []

    for col in candidate_descriptor_cols:
        test_series = pd.to_numeric(data[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        if test_series.notna().mean() >= 0.7:
            numeric_like_cols.append(col)

    st.write(t('descriptor_source.found_descriptors', count=len(numeric_like_cols)))

    custom_descriptor_cols = st.multiselect(
        t('descriptor_source.select_columns'),
        numeric_like_cols,
        default=numeric_like_cols,
        key="custom_descriptor_cols_multiselect"
    )

    st.session_state.custom_descriptor_cols = custom_descriptor_cols

    leakage_df = qspr_detect_data_leakage_columns(
        descriptor_cols=custom_descriptor_cols,
        target_col=target_col,
        data=data,
        y=pd.to_numeric(
            data[target_col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
    )

    qspr_show_data_leakage_warning(
        leakage_df,
        title=t('descriptor_source.leakage_warning_title')
    )

    if st.button(t('descriptor_source.use_button'), type="primary"):
        try:
            leakage_df = qspr_detect_data_leakage_columns(
                descriptor_cols=st.session_state.custom_descriptor_cols,
                target_col=target_col,
                data=data,
                y=pd.to_numeric(
                    data[target_col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )
            )

            if not leakage_df.empty:
                qspr_show_data_leakage_warning(
                    leakage_df,
                    title=t('descriptor_source.leakage_stopped_title')
                )
                st.error(t('descriptor_source.leakage_stopped_error'))
                st.stop()

            bundle = qspr_prepare_custom_descriptors_from_file(
                data=data,
                target_col=target_col,
                descriptor_cols=st.session_state.custom_descriptor_cols,
                smiles_col=smiles_col_current
            )

            store_descriptor_bundle(bundle, "custom_descriptors")
            st.session_state.descriptor_calculation_mode = "custom_descriptors"

            descriptors_df = bundle["df_desc"].copy()
            descriptors_df["SMILES"] = data[smiles_col_current].iloc[bundle["valid_indices"]].values
            descriptors_df[target_col] = bundle["y_all"]
            cols = ["SMILES", target_col] + [c for c in descriptors_df.columns if c not in ["SMILES", target_col]]
            descriptors_df = descriptors_df[cols]

            qspr_save_results_auto(descriptors_df, "custom_descriptors", target_col, len(bundle["y_all"]))

            st.success(t('descriptor_source.use_success',
                n_desc=len(bundle['desc_names']),
                n_compounds=len(bundle['y_all'])
            ))
            add_log(t('descriptor_source.log_used',
                n_desc=len(bundle['desc_names'])
            ))

        except Exception as e:
            st.error(t('descriptor_source.use_error', error=e))

else:
    st.subheader(t('descriptor_settings.subheader'))

    spectral_ready_for_source = (
        st.session_state.get("spectral_descriptors_transferred_ready", False)
        or (
            st.session_state.get("desc_calculated", False)
            and st.session_state.get("descriptor_calculation_mode", "") == "spectral_or_combined"
        )
        or isinstance(st.session_state.get("spectral_descriptors_df"), pd.DataFrame)
    )

    col_desc_source_1, col_desc_source_2, col_desc_source_3 = st.columns(3)

    with col_desc_source_1:
        use_molecular_descriptors_source = st.checkbox(
            t('descriptor_settings.checkbox_molecular'),
            value=True,
            key="use_molecular_descriptors_source"
        )

    with col_desc_source_2:
        use_quantum_descriptors_source = st.checkbox(
            t('descriptor_settings.checkbox_quantum'),
            value=False,
            disabled=qspr_is_online_mode(),
            key="use_quantum_descriptors_source"
        )
        if qspr_is_online_mode():
            st.caption(ONLINE_LOCK_MESSAGE)

    with col_desc_source_3:
        if not spectral_ready_for_source:
            st.session_state.use_spectral_descriptors_source = False

        use_spectral_descriptors_source = st.checkbox(
            t('descriptor_settings.checkbox_spectral'),
            disabled=(not spectral_ready_for_source) or qspr_is_online_mode(),
            key="use_spectral_descriptors_source"
        )
        if qspr_is_online_mode():
            st.caption(ONLINE_LOCK_MESSAGE)

    # По умолчанию подисточники квантово-химического блока выключены.
    use_xtb_descriptors_source = False
    use_morfeus_descriptors_source = False
    use_dscribe_descriptors_source = False

    if qspr_is_online_mode():
        use_quantum_descriptors_source = False
        use_spectral_descriptors_source = False

    if use_morfeus_descriptors_source and not morfeus_available:
        st.warning(t('descriptor_settings.morfeus_not_available', error=morfeus_import_error))

    if use_quantum_descriptors_source:
        with st.expander(t('descriptor_settings.quantum_expander'), expanded=True):
            st.markdown(t('descriptor_settings.quantum_sources_title'))

            col_q_1, col_q_2, col_q_3 = st.columns(3)

            with col_q_1:
                use_xtb_descriptors_source = st.checkbox(
                    t('descriptor_settings.xtb_checkbox'),
                    value=True,
                    key="use_xtb_descriptors_source"
                )

            with col_q_2:
                use_morfeus_descriptors_source = st.checkbox(
                    t('descriptor_settings.morfeus_checkbox'),
                    value=False,
                    key="use_morfeus_descriptors_source"
                )

            with col_q_3:
                use_dscribe_descriptors_source = st.checkbox(
                    t('descriptor_settings.dscribe_checkbox'),
                    value=False,
                    key="use_dscribe_descriptors_source"
                )
                
            if use_morfeus_descriptors_source and not morfeus_available:
                st.warning(t('descriptor_settings.morfeus_not_available', error=morfeus_import_error))
            if use_dscribe_descriptors_source and not dscribe_available:
                st.warning(t('descriptor_settings.dscribe_not_available', error=dscribe_import_error))

            with st.expander(t('descriptor_settings.bank_expander'), expanded=False):
                use_descriptor_bank = st.checkbox(
                    t('descriptor_settings.bank_use_checkbox'),
                    value=True,
                    key="use_descriptor_bank",
                )

                save_descriptor_bank = st.checkbox(
                    t('descriptor_settings.bank_save_checkbox'),
                    value=True,
                    key="save_descriptor_bank",
                )

                st.caption(t('descriptor_settings.bank_caption'))

            # ------------------------------------------------------------
            # xTB settings

            if use_xtb_descriptors_source:
                st.markdown(t('descriptor_settings.xtb_title'))

                col_xtb_1, col_xtb_2, col_xtb_3 = st.columns(3)

                with col_xtb_1:
                    xtb_method = st.selectbox(
                        t('descriptor_settings.xtb_method_label'),
                        ["GFN2-xTB", "GFN1-xTB", "GFN0-xTB"],
                        index=0,
                        key="xtb_method"
                    )

                    xtb_accuracy = st.number_input(
                        t('descriptor_settings.xtb_accuracy_label'),
                        min_value=0.01,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        key="xtb_accuracy"
                    )

                with col_xtb_2:
                    xtb_charge = st.number_input(
                        t('descriptor_settings.xtb_charge_label'),
                        min_value=-10,
                        max_value=10,
                        value=0,
                        step=1,
                        key="xtb_charge"
                    )

                    xtb_etemp = st.number_input(
                        t('descriptor_settings.xtb_temp_label'),
                        min_value=0.0,
                        max_value=10000.0,
                        value=300.0,
                        step=50.0,
                        key="xtb_etemp"
                    )

                with col_xtb_3:
                    xtb_uhf = st.number_input(
                        t('descriptor_settings.xtb_uhf_label'),
                        min_value=0,
                        max_value=20,
                        value=0,
                        step=1,
                        key="xtb_uhf"
                    )

                    xtb_max_iter = st.number_input(
                        t('descriptor_settings.xtb_iter_label'),
                        min_value=50,
                        max_value=2000,
                        value=250,
                        step=50,
                        key="xtb_max_iter"
                    )

                xtb_optimize_rdkit = st.checkbox(
                    t('descriptor_settings.xtb_optimize_checkbox'),
                    value=True,
                    key="xtb_optimize_rdkit"
                )

                xtb_limit = st.number_input(
                    t('descriptor_settings.xtb_limit_label'),
                    min_value=0,
                    value=0,
                    step=10,
                    key="xtb_limit"
                )

                show_markdown_help(
                    t('descriptor_settings.xtb_help_title'),
                    os.path.join(HELP_DIR, "xtb_settings_help.md"),
                    expanded=False
                )

            # ------------------------------------------------------------
            # morfeus settings

            if use_morfeus_descriptors_source:
                st.markdown(t('descriptor_settings.morfeus_title'))

                morfeus_calc_sasa = st.checkbox(
                    t('descriptor_settings.morfeus_sasa_checkbox'),
                    value=True,
                    key="morfeus_calc_sasa"
                )

                morfeus_calc_dispersion = st.checkbox(
                    t('descriptor_settings.morfeus_dispersion_checkbox'),
                    value=True,
                    key="morfeus_calc_dispersion"
                )

                morfeus_calc_xtb = st.checkbox(
                    t('descriptor_settings.morfeus_xtb_checkbox'),
                    value=True,
                    key="morfeus_calc_xtb"
                )

                morfeus_optimize_3d = st.checkbox(
                    t('descriptor_settings.morfeus_optimize_checkbox'),
                    value=True,
                    key="morfeus_optimize_3d"
                )

                morfeus_limit = st.number_input(
                    t('descriptor_settings.morfeus_limit_label'),
                    min_value=0,
                    value=0,
                    step=10,
                    key="morfeus_limit"
                )

                st.caption(t('descriptor_settings.morfeus_caption'))

            # ------------------------------------------------------------
            # DScribe settings

            if use_dscribe_descriptors_source:
                st.markdown(t('descriptor_settings.dscribe_title'))

                dscribe_descriptor_type = st.selectbox(
                    t('descriptor_settings.dscribe_type_label'),
                    [t('descriptor_settings.dscribe_type_coulomb')],
                    index=0,
                    key="dscribe_descriptor_type"
                )

                dscribe_max_atoms = st.number_input(
                    t('descriptor_settings.dscribe_max_atoms_label'),
                    min_value=5,
                    max_value=300,
                    value=60,
                    step=5,
                    key="dscribe_max_atoms"
                )

                dscribe_optimize_3d = st.checkbox(
                    t('descriptor_settings.dscribe_optimize_checkbox'),
                    value=True,
                    key="dscribe_optimize_3d"
                )

                dscribe_limit = st.number_input(
                    t('descriptor_settings.dscribe_limit_label'),
                    min_value=0,
                    value=0,
                    step=10,
                    key="dscribe_limit"
                )

                st.caption(t('descriptor_settings.dscribe_caption'))
                
    if not spectral_ready_for_source:
        st.caption(t('descriptor_settings.spectral_not_ready_caption'))

    if (
        not use_molecular_descriptors_source
        and not use_xtb_descriptors_source
        and not use_spectral_descriptors_source
    ):
        st.warning(t('descriptor_settings.warning_select_descriptor_type'))

    mode = None
    allowed_rdkit_names = []
    allowed_mordred_names = []
    allowed_padel_names = []

if use_molecular_descriptors_source:
    DESCRIPTOR_MODE_OPTIONS = {
        "rdkit_fast": "descriptor_mode.speed",
        "mordred": "descriptor_mode.extended",
        "mordred_padel_unique": "descriptor_mode.smart",
        "max_accuracy": "descriptor_mode.accuracy",
    }

    DESCRIPTOR_MODE_HELP = {
        "rdkit_fast": "descriptor_mode.help_speed",
        "mordred": "descriptor_mode.help_extended",
        "mordred_padel_unique": "descriptor_mode.help_smart",
        "max_accuracy": "descriptor_mode.help_accuracy",
    }

    mode = st.radio(
        t("descriptor_mode.radio_label"),
        options=list(DESCRIPTOR_MODE_OPTIONS.keys()),
        index=0 if qspr_is_online_mode() else 1,
        format_func=lambda key: t(DESCRIPTOR_MODE_OPTIONS[key]),
        key="descriptor_mode_radio_v2"
    )

    st.caption(
        t("descriptor_mode.help_prefix") + " " + t(DESCRIPTOR_MODE_HELP[mode])
    )

    online_heavy_descriptor_mode = qspr_is_online_mode() and mode != "rdkit_fast"
    if online_heavy_descriptor_mode:
        qspr_online_lock_notice("Mordred, PaDEL and maximum-accuracy descriptor modes")

    (
        allowed_rdkit_names,
        allowed_mordred_names,
        allowed_padel_names
    ) = qspr_descriptor_group_selection_ui(
        mode=mode,
        desc_lists=st.session_state.desc_lists
    )

    if (
        use_molecular_descriptors_source
        or use_xtb_descriptors_source
        or use_morfeus_descriptors_source
        or use_dscribe_descriptors_source
    ):

        if st.button(
            t('descriptor_calc.calculate_button'),
            type="primary",
            disabled=online_heavy_descriptor_mode,
        ):
            try:
                bundle = None
                source_label = ""
                selected_descriptor_sources = []

                if use_molecular_descriptors_source:
                    selected_descriptor_sources.append(t('descriptor_calc.source_molecular'))

                if use_xtb_descriptors_source:
                    selected_descriptor_sources.append(t('descriptor_calc.source_xtb'))

                if use_morfeus_descriptors_source:
                    selected_descriptor_sources.append(t('descriptor_calc.source_morfeus'))

                if use_dscribe_descriptors_source:
                    selected_descriptor_sources.append(t('descriptor_calc.source_dscribe'))

                total_descriptor_sources = len(selected_descriptor_sources)
                current_descriptor_source_step = [0]

                overall_progress_bar = st.progress(0)
                overall_progress_text = st.empty()

                def update_overall_descriptor_progress(source_name):
                    current_descriptor_source_step[0] += 1

                    if total_descriptor_sources > 0:
                        overall_progress_bar.progress(
                            min(
                                current_descriptor_source_step[0] / total_descriptor_sources,
                                1.0
                            )
                        )

                    overall_progress_text.caption(t('descriptor_calc.progress_text',
                        done=current_descriptor_source_step[0],
                        total=total_descriptor_sources,
                        source=source_name
                    ))
                    

                if use_molecular_descriptors_source:
                    with st.spinner(t('descriptor_calc.spinner_molecular')):
                        bundle = qspr_calculate_molecular_descriptors(
                            data=data,
                            smiles_col=smiles_col_current,
                            target_col=target_col,
                            mode=mode,
                            desc_lists=st.session_state.desc_lists,
                            allowed_rdkit_names=allowed_rdkit_names,
                            allowed_mordred_names=allowed_mordred_names,
                            allowed_padel_names=allowed_padel_names
                        )

                    source_label = "molecular_calculated"
                    update_overall_descriptor_progress(t('descriptor_calc.source_molecular'))

                if use_xtb_descriptors_source:
                    if not xtb_python_available:
                        st.error(t('descriptor_calc.error_xtb_not_available'))
                        st.stop()

                    xtb_max_molecules = None if int(xtb_limit) <= 0 else int(xtb_limit)

                    use_descriptor_bank = bool(st.session_state.get("use_descriptor_bank", True))
                    save_descriptor_bank = bool(st.session_state.get("save_descriptor_bank", True))

                    xtb_use_bank_default_profile = (
                        use_descriptor_bank
                        and str(xtb_method) == "GFN2-xTB"
                        and int(xtb_charge) == 0
                        and int(xtb_uhf) == 0
                        and abs(float(xtb_accuracy) - 1.0) < 1e-12
                        and abs(float(xtb_etemp) - 300.0) < 1e-12
                        and int(xtb_max_iter) == 250
                        and bool(xtb_optimize_rdkit) is True
                    )

                    if use_descriptor_bank and not xtb_use_bank_default_profile:
                        st.warning(t('descriptor_calc.warning_xtb_bank_profile'))

                        if xtb_use_bank_default_profile:
                            cached_xtb_df, xtb_missing_df, bank_report = descriptor_bank_get_cached_and_missing(
                                df=data,
                                smiles_col=smiles_col_current,
                                descriptor_family="quantum",
                                descriptor_source="xtb",
                                descriptor_profile="default",
                                max_molecules=xtb_max_molecules,
                            )

                            descriptor_bank_show_report(
                                bank_report,
                                title=t('descriptor_calc.bank_xtb_title'),
                            )

                            calculated_xtb_df = pd.DataFrame()
                            calculated_xtb_report = {}

                            if not xtb_missing_df.empty:
                                with st.spinner(t('descriptor_calc.spinner_xtb_missing')):                                    
                                    calculated_xtb_df, calculated_xtb_report = qspr_calc_xtb_descriptors_dataframe(
                                        data=xtb_missing_df,
                                        smiles_col=smiles_col_current,
                                        target_col=target_col,
                                        method=xtb_method,
                                        charge=int(xtb_charge),
                                        uhf=int(xtb_uhf),
                                        accuracy=float(xtb_accuracy),
                                        electronic_temperature=float(xtb_etemp),
                                        max_iterations=int(xtb_max_iter),
                                        random_seed=1,
                                        optimize_with_rdkit=bool(xtb_optimize_rdkit),
                                        max_molecules=None,
                                    )

                                if save_descriptor_bank and calculated_xtb_df is not None and not calculated_xtb_df.empty:
                                    try:
                                        calculated_xtb_df["descriptor_profile"] = "default"
                                        calculated_xtb_df["xtb_method"] = str(xtb_method)
                                        calculated_xtb_df["xtb_charge_setting"] = int(xtb_charge)
                                        calculated_xtb_df["xtb_uhf_setting"] = int(xtb_uhf)
                                        calculated_xtb_df["xtb_accuracy_setting"] = float(xtb_accuracy)
                                        calculated_xtb_df["xtb_electronic_temperature_setting"] = float(xtb_etemp)
                                        calculated_xtb_df["xtb_max_iterations_setting"] = int(xtb_max_iter)
                                        calculated_xtb_df["xtb_rdkit_optimize_setting"] = bool(xtb_optimize_rdkit)
                                        calculated_xtb_df["xtb_random_seed_setting"] = 1
                                        
                                        descriptor_bank_append(
                                            desc_df=calculated_xtb_df,
                                            descriptor_family="quantum",
                                            descriptor_source="xtb",
                                            descriptor_profile="default",
                                            target_col=target_col,
                                        )
                                        st.success(t('descriptor_calc.xtb_bank_success'))
                                    except Exception as e:
                                        st.warning(t('descriptor_calc.xtb_bank_error', error=e))

                            if cached_xtb_df is not None and not cached_xtb_df.empty:
                                cached_xtb_df = descriptor_bank_attach_target(
                                    df=cached_xtb_df,
                                    source_df=data,
                                    target_col=target_col,
                                )

                            if calculated_xtb_df is not None and not calculated_xtb_df.empty:
                                calculated_xtb_df = descriptor_bank_attach_target(
                                    df=calculated_xtb_df,
                                    source_df=data,
                                    target_col=target_col,
                                )

                            xtb_parts = []

                            if cached_xtb_df is not None and not cached_xtb_df.empty:
                                xtb_parts.append(cached_xtb_df)

                            if calculated_xtb_df is not None and not calculated_xtb_df.empty:
                                xtb_parts.append(calculated_xtb_df)

                            if xtb_parts:
                                xtb_df = pd.concat(xtb_parts, ignore_index=True, sort=False)

                                if "_original_index" in xtb_df.columns:
                                    xtb_df = xtb_df.sort_values("_original_index").reset_index(drop=True)

                                xtb_report = dict(calculated_xtb_report or {})
                                xtb_report["descriptor_bank"] = bank_report
                            else:
                                xtb_df = pd.DataFrame()
                                xtb_report = {"descriptor_bank": bank_report}

                        else:
                            with st.spinner(t('descriptor_calc.spinner_xtb_calculating')):
                                xtb_df, xtb_report = qspr_calc_xtb_descriptors_dataframe(
                                    data=data,
                                    smiles_col=smiles_col_current,
                                    target_col=target_col,
                                    method=xtb_method,
                                    charge=int(xtb_charge),
                                    uhf=int(xtb_uhf),
                                    accuracy=float(xtb_accuracy),
                                    electronic_temperature=float(xtb_etemp),
                                    max_iterations=int(xtb_max_iter),
                                    random_seed=1,
                                    optimize_with_rdkit=bool(xtb_optimize_rdkit),
                                    max_molecules=xtb_max_molecules,
                                )

                        st.session_state.xtb_descriptors_df = xtb_df
                        st.session_state.xtb_descriptors_report = xtb_report
                        with st.expander(t('descriptor_calc.xtb_table_expander'), expanded=True):
                            st.caption(t('descriptor_calc.xtb_table_caption'))

                            st.dataframe(
                                xtb_df,
                                width="stretch",
                                hide_index=True
                            )

                        if use_molecular_descriptors_source:
                            bundle = qspr_append_xtb_to_bundle(
                                base_bundle=bundle,
                                xtb_df=xtb_df,
                                target_col=target_col
                            )
                            source_label = "molecular_plus_xtb"
                        else:
                            bundle = qspr_make_xtb_bundle_from_dataframe(
                                xtb_df=xtb_df,
                                target_col=target_col
                            )
                            source_label = "xtb_quantum_descriptors"

                        st.download_button(
                            t('descriptor_calc.xtb_download_button'),
                            xtb_df.to_csv(index=False).encode("utf-8-sig"),
                            "xtb_descriptors.csv",
                            "text/csv",
                            key="download_xtb_descriptors_csv_after_calc"
                        )
                        update_overall_descriptor_progress(t('descriptor_calc.source_xtb'))
                        
                if use_morfeus_descriptors_source:
                    if not morfeus_available:
                        st.error(t('descriptor_calc.morfeus_not_available', error=morfeus_import_error))
                        st.stop()

                    morfeus_max_molecules = (
                        None if int(morfeus_limit) <= 0 else int(morfeus_limit)
                    )

                    use_descriptor_bank = bool(st.session_state.get("use_descriptor_bank", True))
                    save_descriptor_bank = bool(st.session_state.get("save_descriptor_bank", True))

                    morfeus_use_bank_default_profile = (
                        use_descriptor_bank
                        and bool(morfeus_calc_sasa) is True
                        and bool(morfeus_calc_dispersion) is True
                        and bool(morfeus_calc_xtb) is True
                        and bool(morfeus_optimize_3d) is True
                    )

                    if use_descriptor_bank and not morfeus_use_bank_default_profile:
                        st.warning(t('descriptor_calc.morfeus_bank_profile_warning'))

                    if morfeus_use_bank_default_profile:
                        cached_morfeus_df, morfeus_missing_df, bank_report = descriptor_bank_get_cached_and_missing(
                            df=data,
                            smiles_col=smiles_col_current,
                            descriptor_family="3d",
                            descriptor_source="morfeus",
                            descriptor_profile="default",
                            max_molecules=morfeus_max_molecules,
                        )

                        descriptor_bank_show_report(
                            bank_report,
                            title=t('descriptor_calc.morfeus_bank_title'),
                        )

                        calculated_morfeus_df = pd.DataFrame()

                        if not morfeus_missing_df.empty:
                            progress_bar = st.progress(0)
                            progress_text = st.empty()

                            def morfeus_progress(done, total, message):
                                if total > 0:
                                    progress_bar.progress(min(done / total, 1.0))
                                progress_text.caption(message)

                            with st.spinner(t('descriptor_calc.spinner_morfeus_missing')):
                                calculated_morfeus_df = calculate_morfeus_descriptors_for_dataframe(
                                    df=morfeus_missing_df,
                                    smiles_col=smiles_col_current,
                                    id_col=None,
                                    random_seed=42,
                                    optimize=bool(morfeus_optimize_3d),
                                    calc_sasa=bool(morfeus_calc_sasa),
                                    calc_dispersion=bool(morfeus_calc_dispersion),
                                    calc_xtb=bool(morfeus_calc_xtb),
                                    max_molecules=None,
                                    progress_callback=morfeus_progress
                                )

                            progress_bar.empty()
                            progress_text.empty()

                            if save_descriptor_bank and calculated_morfeus_df is not None and not calculated_morfeus_df.empty:
                                try:
                                    _morfeus_key_table = descriptor_bank_make_input_key_table(
                                        df=morfeus_missing_df,
                                        smiles_col=smiles_col_current,
                                        max_molecules=None,
                                    )

                                    calculated_morfeus_df = calculated_morfeus_df.copy()

                                    if "row_index" in calculated_morfeus_df.columns:
                                        calculated_morfeus_df["_original_index"] = pd.to_numeric(
                                            calculated_morfeus_df["row_index"],
                                            errors="coerce",
                                        ).astype("Int64")
                                    else:
                                        calculated_morfeus_df["_original_index"] = calculated_morfeus_df.index.astype("int64")

                                    calculated_morfeus_df = calculated_morfeus_df.merge(
                                        _morfeus_key_table[
                                            [
                                                "_original_index",
                                                "canonical_smiles",
                                                "inchikey",
                                                "bank_key",
                                                "bank_key_status",
                                            ]
                                        ],
                                        on="_original_index",
                                        how="left",
                                    )

                                    calculated_morfeus_df["descriptor_profile"] = "default"
                                    calculated_morfeus_df["morfeus_calc_sasa_setting"] = bool(morfeus_calc_sasa)
                                    calculated_morfeus_df["morfeus_calc_dispersion_setting"] = bool(morfeus_calc_dispersion)
                                    calculated_morfeus_df["morfeus_calc_xtb_setting"] = bool(morfeus_calc_xtb)
                                    calculated_morfeus_df["morfeus_optimize_3d_setting"] = bool(morfeus_optimize_3d)
                                    calculated_morfeus_df["morfeus_random_seed_setting"] = 42

                                    descriptor_bank_append(
                                        desc_df=calculated_morfeus_df,
                                        descriptor_family="3d",
                                        descriptor_source="morfeus",
                                        descriptor_profile="default",
                                        target_col=target_col,
                                    )

                                    st.success(t('descriptor_calc.morfeus_bank_success'))

                                except Exception as e:
                                    st.warning(t('descriptor_calc.morfeus_bank_error', error=e))

                        if cached_morfeus_df is not None and not cached_morfeus_df.empty:
                            cached_morfeus_df = descriptor_bank_attach_target(
                                df=cached_morfeus_df,
                                source_df=data,
                                target_col=target_col,
                            )

                        if calculated_morfeus_df is not None and not calculated_morfeus_df.empty:
                            calculated_morfeus_df = descriptor_bank_attach_target(
                                df=calculated_morfeus_df,
                                source_df=data,
                                target_col=target_col,
                            )

                        morfeus_parts = []

                        if cached_morfeus_df is not None and not cached_morfeus_df.empty:
                            morfeus_parts.append(cached_morfeus_df)

                        if calculated_morfeus_df is not None and not calculated_morfeus_df.empty:
                            morfeus_parts.append(calculated_morfeus_df)

                        if morfeus_parts:
                            morfeus_df = pd.concat(morfeus_parts, ignore_index=True, sort=False)

                            if "_original_index" in morfeus_df.columns:
                                morfeus_df = morfeus_df.sort_values("_original_index").reset_index(drop=True)
                            elif "row_index" in morfeus_df.columns:
                                morfeus_df = morfeus_df.sort_values("row_index").reset_index(drop=True)
                        else:
                            morfeus_df = pd.DataFrame()

                    else:
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        def morfeus_progress(done, total, message):
                            if total > 0:
                                progress_bar.progress(min(done / total, 1.0))
                            progress_text.caption(message)

                        with st.spinner(t('descriptor_calc.spinner_morfeus_3d')):
                            morfeus_df = calculate_morfeus_descriptors_for_dataframe(
                                df=data,
                                smiles_col=smiles_col_current,
                                id_col=None,
                                random_seed=42,
                                optimize=bool(morfeus_optimize_3d),
                                calc_sasa=bool(morfeus_calc_sasa),
                                calc_dispersion=bool(morfeus_calc_dispersion),
                                calc_xtb=bool(morfeus_calc_xtb),
                                max_molecules=morfeus_max_molecules,
                                progress_callback=morfeus_progress
                            )

                        progress_bar.empty()
                        progress_text.empty()

                    # Добавляем целевое свойство в morfeus_df.
                    # row_index хранит исходный индекс строки data.
                    if "row_index" in morfeus_df.columns:
                        morfeus_row_indices = pd.to_numeric(
                            morfeus_df["row_index"],
                            errors="coerce"
                        )

                        valid_row_mask = morfeus_row_indices.notna()
                        morfeus_df = morfeus_df.loc[valid_row_mask].copy()
                        morfeus_row_indices = morfeus_row_indices.loc[valid_row_mask].astype(int)

                        morfeus_df[target_col] = pd.to_numeric(
                            data.loc[morfeus_row_indices, target_col].values,
                            errors="coerce"
                        )
                    else:
                        morfeus_df[target_col] = pd.to_numeric(
                            data[target_col].values[:len(morfeus_df)],
                            errors="coerce"
                        )

                    st.session_state.morfeus_descriptors_df = morfeus_df

                    with st.expander(t('descriptor_calc.morfeus_table_expander'), expanded=True):
                        st.caption(t('descriptor_calc.morfeus_table_caption'))

                        st.dataframe(
                            morfeus_df,
                            width="stretch",
                            hide_index=True
                        )

                        status_cols = [
                            c for c in [
                                "morfeus_status",
                                "morfeus_3d_status",
                                "morfeus_sasa_status",
                                "morfeus_dispersion_status",
                                "morfeus_xtb_status",
                            ]
                            if c in morfeus_df.columns
                        ]

                        if status_cols:
                            st.markdown(t('descriptor_calc.morfeus_status_title'))

                            for status_col in status_cols:
                                st.write(f"**{status_col}**")
                                st.dataframe(
                                    morfeus_df[status_col].astype(str).value_counts().reset_index().rename(
                                        columns={
                                            "index": t('descriptor_calc.status_column'),
                                            status_col: t('descriptor_calc.count_column')
                                        }
                                    ),
                                    width="stretch",
                                    hide_index=True
                                )

                    # Собираем или расширяем общий descriptor bundle.
                    if bundle is None:
                        bundle = qspr_make_morfeus_bundle_from_dataframe(
                            morfeus_df=morfeus_df,
                            target_col=target_col
                        )
                        source_label = "morfeus_3d_descriptors"
                    else:
                        bundle = qspr_append_morfeus_to_bundle(
                            base_bundle=bundle,
                            morfeus_df=morfeus_df,
                            target_col=target_col
                        )
                        source_label = bundle["report"]["descriptor_source"]

                    st.download_button(
                        t('descriptor_calc.morfeus_download_button'),
                        morfeus_df.to_csv(index=False).encode("utf-8-sig"),
                        "morfeus_descriptors.csv",
                        "text/csv",
                        key="download_morfeus_descriptors_csv_after_calc"
                    )
                    update_overall_descriptor_progress(t('descriptor_calc.source_morfeus'))
                    
                if use_dscribe_descriptors_source:
                    if not dscribe_available:
                        st.error(t('descriptor_calc.dscribe_not_available', error=dscribe_import_error))
                        st.stop()

                    dscribe_max_molecules = (
                        None if int(dscribe_limit) <= 0 else int(dscribe_limit)
                    )

                    use_descriptor_bank = bool(st.session_state.get("use_descriptor_bank", True))
                    save_descriptor_bank = bool(st.session_state.get("save_descriptor_bank", True))

                    dscribe_use_bank_default_profile = (
                        use_descriptor_bank
                        and str(dscribe_descriptor_type) == "Coulomb Matrix eigenspectrum"
                        and bool(dscribe_optimize_3d) is True
                        and int(dscribe_max_atoms) == 60
                    )

                    if use_descriptor_bank and not dscribe_use_bank_default_profile:
                        st.warning(t('descriptor_calc.dscribe_bank_profile_warning'))

                    if dscribe_use_bank_default_profile:
                        cached_dscribe_df, dscribe_missing_df, bank_report = descriptor_bank_get_cached_and_missing(
                            df=data,
                            smiles_col=smiles_col_current,
                            descriptor_family="atomistic",
                            descriptor_source="dscribe",
                            descriptor_profile="default",
                            max_molecules=dscribe_max_molecules,
                        )

                        descriptor_bank_show_report(
                            bank_report,
                            title=t('descriptor_calc.dscribe_bank_title'),
                        )

                        calculated_dscribe_df = pd.DataFrame()

                        if not dscribe_missing_df.empty:
                            progress_bar = st.progress(0)
                            progress_text = st.empty()

                            def dscribe_progress(done, total, message):
                                if total > 0:
                                    progress_bar.progress(min(done / total, 1.0))
                                progress_text.caption(message)

                            with st.spinner(t('descriptor_calc.spinner_dscribe_missing')):
                                calculated_dscribe_df = calculate_dscribe_descriptors_for_dataframe(
                                    df=dscribe_missing_df,
                                    smiles_col=smiles_col_current,
                                    id_col=None,
                                    random_seed=42,
                                    optimize=bool(dscribe_optimize_3d),
                                    max_atoms=int(dscribe_max_atoms),
                                    calc_coulomb=True,
                                    max_molecules=None,
                                    progress_callback=dscribe_progress
                                )

                            progress_bar.empty()
                            progress_text.empty()

                            if save_descriptor_bank and calculated_dscribe_df is not None and not calculated_dscribe_df.empty:
                                try:
                                    _dscribe_key_table = descriptor_bank_make_input_key_table(
                                        df=dscribe_missing_df,
                                        smiles_col=smiles_col_current,
                                        max_molecules=None,
                                    )

                                    calculated_dscribe_df = calculated_dscribe_df.copy()

                                    if "row_index" in calculated_dscribe_df.columns:
                                        calculated_dscribe_df["_original_index"] = pd.to_numeric(
                                            calculated_dscribe_df["row_index"],
                                            errors="coerce",
                                        ).astype("Int64")

                                    calculated_dscribe_df = calculated_dscribe_df.merge(
                                        _dscribe_key_table[
                                            [
                                                "_original_index",
                                                "canonical_smiles",
                                                "inchikey",
                                                "bank_key",
                                                "bank_key_status",
                                            ]
                                        ],
                                        on="_original_index",
                                        how="left",
                                    )
                                    calculated_dscribe_df["descriptor_profile"] = "default"
                                    calculated_dscribe_df["dscribe_descriptor_type_setting"] = str(dscribe_descriptor_type)
                                    calculated_dscribe_df["dscribe_optimize_3d_setting"] = bool(dscribe_optimize_3d)
                                    calculated_dscribe_df["dscribe_max_atoms_setting"] = int(dscribe_max_atoms)
                                    calculated_dscribe_df["dscribe_calc_coulomb_setting"] = True
                                    calculated_dscribe_df["dscribe_random_seed_setting"] = 42

                                    descriptor_bank_append(
                                        desc_df=calculated_dscribe_df,
                                        descriptor_family="atomistic",
                                        descriptor_source="dscribe",
                                        descriptor_profile="default",
                                        target_col=target_col,
                                    )
                                    st.success(t('descriptor_calc.dscribe_bank_success'))
                                except Exception as e:
                                    st.warning(t('descriptor_calc.dscribe_bank_error', error=e))

                        if cached_dscribe_df is not None and not cached_dscribe_df.empty:
                            cached_dscribe_df = descriptor_bank_attach_target(
                                df=cached_dscribe_df,
                                source_df=data,
                                target_col=target_col,
                            )

                        if calculated_dscribe_df is not None and not calculated_dscribe_df.empty:
                            calculated_dscribe_df = descriptor_bank_attach_target(
                                df=calculated_dscribe_df,
                                source_df=data,
                                target_col=target_col,
                            )

                        dscribe_parts = []

                        if cached_dscribe_df is not None and not cached_dscribe_df.empty:
                            dscribe_parts.append(cached_dscribe_df)

                        if calculated_dscribe_df is not None and not calculated_dscribe_df.empty:
                            dscribe_parts.append(calculated_dscribe_df)

                        if dscribe_parts:
                            dscribe_df = pd.concat(dscribe_parts, ignore_index=True, sort=False)

                            if "_original_index" in dscribe_df.columns:
                                dscribe_df = dscribe_df.sort_values("_original_index").reset_index(drop=True)
                            elif "row_index" in dscribe_df.columns:
                                dscribe_df = dscribe_df.sort_values("row_index").reset_index(drop=True)
                        else:
                            dscribe_df = pd.DataFrame()

                    else:
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        def dscribe_progress(done, total, message):
                            if total > 0:
                                progress_bar.progress(min(done / total, 1.0))
                            progress_text.caption(message)

                        with st.spinner(t('descriptor_calc.spinner_dscribe_atomistic')):
                            dscribe_df = calculate_dscribe_descriptors_for_dataframe(
                                df=data,
                                smiles_col=smiles_col_current,
                                id_col=None,
                                random_seed=42,
                                optimize=bool(dscribe_optimize_3d),
                                max_atoms=int(dscribe_max_atoms),
                                calc_coulomb=True,
                                max_molecules=dscribe_max_molecules,
                                progress_callback=dscribe_progress
                            )

                    progress_bar.empty()
                    progress_text.empty()

                    if "row_index" in dscribe_df.columns:
                        dscribe_row_indices = pd.to_numeric(
                            dscribe_df["row_index"],
                            errors="coerce"
                        )

                        valid_row_mask = dscribe_row_indices.notna()

                        dscribe_df = dscribe_df.loc[valid_row_mask].copy()
                        dscribe_row_indices = (
                            dscribe_row_indices
                            .loc[valid_row_mask]
                            .astype(int)
                        )

                        dscribe_df[target_col] = pd.to_numeric(
                            data.loc[dscribe_row_indices, target_col].values,
                            errors="coerce"
                        )
                    else:
                        dscribe_df[target_col] = pd.to_numeric(
                            data[target_col].values[:len(dscribe_df)],
                            errors="coerce"
                        )

                    st.session_state.dscribe_descriptors_df = dscribe_df

                    with st.expander(t('descriptor_calc.dscribe_table_expander'), expanded=True):
                        st.caption(t('descriptor_calc.dscribe_table_caption'))

                        st.dataframe(
                            dscribe_df,
                            width="stretch",
                            hide_index=True
                        )

                        status_cols = [
                            c for c in [
                                "dscribe_status",
                                "dscribe_3d_status",
                                "dscribe_coulomb_status",
                            ]
                            if c in dscribe_df.columns
                        ]

                        if status_cols:
                            st.markdown(t('descriptor_calc.dscribe_status_title'))

                            for status_col in status_cols:
                                status_table = (
                                    dscribe_df[status_col]
                                    .astype(str)
                                    .value_counts()
                                    .reset_index()
                                )

                                status_table.columns = [
                                    t('descriptor_calc.status_column'),
                                    t('descriptor_calc.count_column')
                                ]

                                st.write(f"**{status_col}**")
                                st.dataframe(
                                    status_table,
                                    width="stretch",
                                    hide_index=True
                                )

                    if bundle is None:
                        bundle = qspr_make_dscribe_bundle_from_dataframe(
                            dscribe_df=dscribe_df,
                            target_col=target_col
                        )
                        source_label = "dscribe_atomistic_descriptors"
                    else:
                        bundle = qspr_append_dscribe_to_bundle(
                            base_bundle=bundle,
                            dscribe_df=dscribe_df,
                            target_col=target_col
                        )
                        source_label = bundle["report"]["descriptor_source"]

                    st.download_button(
                        t('descriptor_calc.dscribe_download_button'),
                        dscribe_df.to_csv(index=False).encode("utf-8-sig"),
                        "dscribe_descriptors.csv",
                        "text/csv",
                        key="download_dscribe_descriptors_csv_after_calc"
                    )    
                    update_overall_descriptor_progress(t('descriptor_calc.source_dscribe'))

                # ------------------------------------------------------------------
                # КЛЮЧЕВОЕ МЕСТО: выбор ветки (спектральная или нет)
                if bundle is None:
                    st.warning(t('descriptor_calc.no_descriptor_sources'))
                    st.stop()

                if use_spectral_descriptors_source:
                    # --- Ветка со спектральными дескрипторами ---
                    spectral_df_for_bundle = st.session_state.get("spectral_descriptors_df")

                    if (
                        spectral_df_for_bundle is None
                        or not isinstance(spectral_df_for_bundle, pd.DataFrame)
                        or spectral_df_for_bundle.empty
                    ):
                        st.warning(t('descriptor_calc.spectral_not_found_warning'))
                        
                        if bundle is None:
                            st.error("bundle is None – расчёт дескрипторов не удался!")
                            st.stop()
                        
                        store_descriptor_bundle(bundle, source_label)
                    else:
                        bundle = qspr_build_descriptor_matrix_from_sources(
                            current_df=current_df,
                            target_col=target_col,
                            use_molecular=True,
                            molecular_desc_df=bundle["df_desc"],
                            molecular_valid_indices=bundle["valid_indices"],
                            use_spectral=True,
                            spectral_desc_df=spectral_df_for_bundle,
                            smiles_col=smiles_col_current,
                            restrict_to_spectral_subset=True,
                        )

                        if source_label == "molecular_plus_xtb":
                            bundle["report"]["descriptor_source"] = "molecular_xtb_plus_spectral"
                        elif source_label == "xtb_quantum_descriptors":
                            bundle["report"]["descriptor_source"] = "xtb_plus_spectral"
                        elif source_label == "morfeus_3d_descriptors":
                            bundle["report"]["descriptor_source"] = "morfeus_plus_spectral"
                        elif "morfeus" in str(source_label):
                            bundle["report"]["descriptor_source"] = f"{source_label}_plus_spectral"

                        source_label = bundle["report"]["descriptor_source"]
                        store_descriptor_bundle(bundle, source_label)

                        st.session_state.descriptor_calculation_mode = "spectral_or_combined"
                        st.session_state.custom_descriptor_source = source_label
                        st.session_state.custom_descriptors_used = True
                    pass
                else:
                    # --- ОСНОВНОЙ ПУТЬ (без спектральных дескрипторов) ---
                    if bundle is None:
                        st.error("Ошибка: bundle is None – расчёт дескрипторов не удался!")
                        st.stop()


                    store_descriptor_bundle(bundle, source_label)
                    st.session_state.molecular_descriptors_ready = True
                    st.session_state.molecular_df_desc = bundle["df_desc"].copy()
                    st.session_state.molecular_valid_indices = list(bundle["valid_indices"])
                    st.session_state.molecular_desc_names = list(bundle["desc_names"])
                    st.session_state.molecular_X_all = np.array(bundle["X_all"], copy=True)
                    st.session_state.molecular_y_all = np.array(bundle["y_all"], copy=True)
                    st.session_state.molecular_descriptor_source = source_label
                    
                    st.session_state.descriptor_calculation_mode = source_label
                    st.session_state.molecular_descriptor_calculation_mode = (
                        mode if mode is not None else source_label
                    )
                    
                    # Проверка, что флаг установлен (дополнительная защита)
                    if not st.session_state.desc_calculated:
                        st.error("Не удалось установить флаг desc_calculated. Проверьте функцию store_descriptor_bundle.")
                        st.stop()

                # ---- ОБЩИЙ КОД (выполняется в обеих ветках) ----
                descriptors_df = bundle["df_desc"].copy()
                descriptors_df["SMILES"] = data[smiles_col_current].iloc[
                    bundle["valid_indices"]
                ].values
                descriptors_df[target_col] = bundle["y_all"]

                cols = ["SMILES", target_col] + [
                    c for c in descriptors_df.columns
                    if c not in ["SMILES", target_col]
                ]

                descriptors_df = descriptors_df[cols]

                qspr_save_results_auto(
                    descriptors_df,
                    "descriptors",
                    target_col,
                    len(bundle["y_all"])
                )
                overall_progress_bar.progress(1.0)
                overall_progress_text.caption(t('descriptor_calc.progress_all_done'))

                st.success(t('descriptor_calc.success_collected',
                    n_desc=len(bundle['desc_names']),
                    n_compounds=len(bundle['y_all']),
                    source=source_label
                ))

                qspr_show_descriptor_meaning_table(
                    desc_names=bundle["desc_names"],
                    title=t('descriptor_calc.meaning_table_title'),
                    status_label=t('descriptor_calc.meaning_table_status'),
                    expanded=False,
                    key_prefix="calculated_descriptor_meanings"
                )

                add_log(t('descriptor_calc.log_collected',
                    source=source_label,
                    n_desc=len(bundle['desc_names'])
                ))

            except Exception as e:
                st.error(t('descriptor_calc.error_calculation', error=e))


# ------------------------------------------------------------------
# Пользовательские дескрипторы / МНК

with st.expander(t('incremental.expander_title'), expanded=False):
    st.markdown(t('incremental.description'))

    if data is None or data.empty:
        st.info(t('incremental.no_data'))

    else:
        # ------------------------------------------------------------
        # 1. Поиск числовых колонок-кандидатов

        numeric_candidate_cols = []

        for col in data.columns:
            if col == target_col:
                continue

            if col == smiles_col_current:
                continue

            try:
                test_series = incremental_to_numeric(data[col])

                if test_series.notna().sum() > 0:
                    numeric_candidate_cols.append(col)

            except Exception:
                pass

        if not numeric_candidate_cols:
            st.warning(t('incremental.no_candidates'))

        else:
            st.caption(t('incremental.found_candidates_caption'))

            # ------------------------------------------------------------
            # 2. Автоматическое предложение исключить служебные колонки

            auto_exclude_cols = []

            for col in numeric_candidate_cols:
                col_lower = str(col).lower()

                if col == target_col:
                    auto_exclude_cols.append(col)

                elif col_lower in [
                    "№",
                    "index",
                    "id",
                    "compound_id",
                    "source_line_number",
                    "pubchem_cid",
                    "cas"
                ]:
                    auto_exclude_cols.append(col)

                elif "id" in col_lower:
                    auto_exclude_cols.append(col)

                elif "index" in col_lower:
                    auto_exclude_cols.append(col)

                elif "number" in col_lower and "carbon" not in col_lower:
                    auto_exclude_cols.append(col)

                elif col_lower in [
                    "mw",
                    "molecular_weight",
                    "molecular_weight_g_mol"
                ]:
                    auto_exclude_cols.append(col)

            exclude_table = pd.DataFrame({
                t('incremental.column'): numeric_candidate_cols,
                t('incremental.exclude'): [
                    c in auto_exclude_cols
                    for c in numeric_candidate_cols
                ]
            })

            edited_exclude_table = st.data_editor(
                exclude_table,
                column_config={
                    t('incremental.column'): st.column_config.TextColumn(
                        t('incremental.column'),
                        disabled=True
                    ),
                    t('incremental.exclude'): st.column_config.CheckboxColumn(
                        t('incremental.exclude_from_calculation'),
                        default=False
                    ),
                },
                disabled=[t('incremental.column')],
                hide_index=True,
                width="stretch",
                height=420,
                key="incremental_exclude_table_editor"
            )

            exclude_increment_cols = (
                edited_exclude_table
                .loc[edited_exclude_table[t('incremental.exclude')] == True, t('incremental.column')]
                .tolist()
            )

            increment_cols = [
                c for c in numeric_candidate_cols
                if c not in exclude_increment_cols
            ]

            st.session_state.incremental_cols = increment_cols

            st.info(t('incremental.used_features_info',
                count=len(increment_cols),
                total=len(numeric_candidate_cols)
            ))

            with st.expander(t('incremental.show_features_expander'), expanded=False):
                st.dataframe(
                    pd.DataFrame({t('incremental.feature_label'): increment_cols}),
                    width="stretch",
                    hide_index=True
                )

            # ------------------------------------------------------------
            # 3. Выбор режима работы

            use_mnk_here = st.checkbox(
                t('incremental.calc_mnk_checkbox'),
                value=True,
                key="custom_desc_use_mnk",
                help=t('incremental.calc_mnk_help')
            )

            if use_mnk_here:
                use_intercept = st.checkbox(
                    t('incremental.use_intercept_checkbox'),
                    value=bool(st.session_state.get("incremental_use_intercept", True)),
                    key="incremental_use_intercept_checkbox"
                )

                st.session_state.incremental_use_intercept = use_intercept

                with st.expander(t('incremental.intercept_help_expander'), expanded=False):
                    st.markdown(t('incremental.intercept_help_text'))

            else:
                use_intercept = True

                st.info(t('incremental.mnk_disabled_info'))

            # ------------------------------------------------------------
            # 4. Кнопка запуска в зависимости от режима

            if use_mnk_here:
                run_custom_desc_action = st.button(
                    t('incremental.calc_mnk_button'),
                    key="run_incremental_model"
                )
            else:
                run_custom_desc_action = st.button(
                    t('incremental.transfer_button'),
                    key="send_custom_descriptors_to_qspr_direct"
                )

            # ------------------------------------------------------------
            # 5. Выполнение действия

            if run_custom_desc_action:
                if not increment_cols:
                    st.error(t('incremental.error_no_columns'))
                    st.stop()

                if use_mnk_here:
                    try:
                        inc_result = fit_incremental_contributions(
                            data=data,
                            target_col=target_col,
                            increment_cols=increment_cols,
                            use_intercept=use_intercept
                        )

                        st.session_state.incremental_result = inc_result

                        add_log(t('incremental.log_mnk_calculated', n=len(increment_cols)))

                        st.success(t('incremental.success_mnk_calculated'))

                    except Exception as e:
                        st.error(t('incremental.error_mnk_calculation', error=e))

                else:
                    try:
                        leakage_df = qspr_detect_data_leakage_columns(
                            descriptor_cols=inc_result["increment_cols"],
                            target_col=target_col,
                            data=data,
                            y=pd.to_numeric(
                                data[target_col].astype(str).str.replace(",", ".", regex=False),
                                errors="coerce"
                            )
                        )

                        if not leakage_df.empty:
                            qspr_show_data_leakage_warning(
                                leakage_df,
                                title=t('incremental.leakage_stopped_title')
                            )
                            st.error(t('incremental.leakage_stopped_error'))
                            st.stop()
                        prepared = qspr_prepare_custom_descriptors_from_file(
                            data=data,
                            target_col=target_col,
                            descriptor_cols=increment_cols,
                            smiles_col=smiles_col_current
                        )

                        store_descriptor_bundle(
                            prepared,
                            "custom_descriptors"
                        )
                        st.session_state.descriptor_calculation_mode = "custom_descriptors"

                        add_log(t('incremental.log_transferred', n=len(increment_cols)))

                        st.success(t('incremental.success_transferred',
                            n_desc=len(prepared['desc_names']),
                            n_compounds=len(prepared['y_all'])
                        ))

                        st.rerun()

                    except Exception as e:
                        st.error(t('incremental.error_transfer', error=e))

            # ------------------------------------------------------------
            # 6. Показ результатов МНК

            inc_result = st.session_state.get("incremental_result")

            if use_mnk_here and inc_result is not None:
                st.subheader(t('incremental.equation_subheader'))
                st.code(inc_result["equation"])

                st.subheader(t('incremental.coef_table_subheader'))

                st.dataframe(
                    inc_result["coef_table"],
                    width="stretch",
                    hide_index=True
                )

                col_inc_1, col_inc_2, col_inc_3, col_inc_4 = st.columns(4)

                with col_inc_1:
                    st.metric(t('incremental.metric_r2'), f"{inc_result['metrics']['R2']:.3f}")

                with col_inc_2:
                    st.metric(t('incremental.metric_rmse'), f"{inc_result['metrics']['RMSE']:.3f}")

                with col_inc_3:
                    st.metric(t('incremental.metric_mae'), f"{inc_result['metrics']['MAE']:.3f}")

                with col_inc_4:
                    st.metric(t('incremental.metric_mape'), f"{inc_result['metrics']['MAPE_percent']:.2f}")

                st.subheader(t('incremental.plots_subheader'))

                col_inc_plot_1, col_inc_plot_2 = st.columns(2)

                with col_inc_plot_1:
                    fig_inc, ax_inc = plt.subplots(figsize=(4, 4))

                    y = inc_result["y"]
                    y_pred = inc_result["y_pred"]

                    min_y = min(np.nanmin(y), np.nanmin(y_pred))
                    max_y = max(np.nanmax(y), np.nanmax(y_pred))
                    pad_y = (max_y - min_y) * 0.05 if max_y > min_y else 1.0

                    ax_inc.scatter(
                        y,
                        y_pred,
                        alpha=0.7,
                        s=35
                    )

                    ax_inc.plot(
                        [min_y - pad_y, max_y + pad_y],
                        [min_y - pad_y, max_y + pad_y],
                        "r--",
                        lw=1.5
                    )

                    ax_inc.set_xlabel(t('incremental.plot_exp_label'))
                    ax_inc.set_ylabel(t('incremental.plot_pred_label'))
                    ax_inc.set_title(t('incremental.plot_title'))
                    ax_inc.grid(True, alpha=0.3)

                    fig_inc.tight_layout()
                    st.pyplot(fig_inc)

                with col_inc_plot_2:
                    fig_inc_err, ax_inc_err = plt.subplots(figsize=(4, 3))

                    errors = inc_result["errors"]

                    safe_histplot(ax_inc_err, errors, bins=30, kde=True, color='steelblue', edgecolor='black', alpha=0.7)

                    ax_inc_err.set_xlabel(t('incremental.plot_error_label'))
                    ax_inc_err.set_ylabel(t('incremental.plot_count_label'))
                    ax_inc_err.set_title(t('incremental.plot_error_title'))
                    ax_inc_err.grid(True, alpha=0.3)

                    fig_inc_err.tight_layout()
                    st.pyplot(fig_inc_err)

                st.subheader(t('incremental.result_table_subheader'))

                st.dataframe(
                    inc_result["result_table"],
                    width="stretch",
                    hide_index=True
                )

                coef_csv = inc_result["coef_table"].to_csv(index=False).encode("utf-8")
                result_csv = inc_result["result_table"].to_csv(index=False).encode("utf-8")

                col_inc_download_1, col_inc_download_2 = st.columns(2)

                with col_inc_download_1:
                    st.download_button(
                        t('incremental.download_coef_button'),
                        coef_csv,
                        "mnk_coefficients.csv",
                        "text/csv"
                    )

                with col_inc_download_2:
                    st.download_button(
                        t('incremental.download_result_button'),
                        result_csv,
                        "mnk_predictions.csv",
                        "text/csv"
                    )

                # ------------------------------------------------------------
                # 7. Передача этих же признаков в основной QSPR

                if st.button(
                    t('incremental.transfer_these_descriptors_button'),
                    key="use_incremental_as_descriptors"
                ):
                    try:
                        prepared = qspr_prepare_custom_descriptors_from_file(
                            data=data,
                            target_col=target_col,
                            descriptor_cols=inc_result["increment_cols"],
                            smiles_col=smiles_col_current
                        )

                        store_descriptor_bundle(
                            prepared,
                            "custom_descriptors"
                        )
                        st.session_state.descriptor_calculation_mode = "custom_descriptors"

                        add_log(t('incremental.log_transferred_from_mnk', n=len(inc_result['increment_cols'])))

                        st.success(t('incremental.success_transferred_from_mnk'))

                        st.rerun()

                    except Exception as e:
                        st.error(t('incremental.error_transfer_from_mnk', error=e))

# ------------------------------------------------------------------
# QSPR modelling

if not st.session_state.desc_calculated:
    if st.session_state.get("pending_qspr_descriptor_bundle_ready", False):
        st.info(t('desc_calc.pending_bundle_info'))
    elif isinstance(st.session_state.get("spectral_descriptors_df"), pd.DataFrame):
        st.info(t('desc_calc.spectral_ready_info'))
    else:
        st.info(t('desc_calc.calc_or_connect_info'))
    st.stop()

X_all = st.session_state.X_all
y_all = st.session_state.y_all
valid_indices = st.session_state.valid_indices
desc_names = st.session_state.desc_names
df_desc = st.session_state.df_desc
desc_names_current = st.session_state.desc_names

descriptor_source_message()

# ------------------------------------------------------------------
# Data leakage control

st.subheader(t('leakage_control.title'))

try:
    leakage_df = qspr_detect_data_leakage_columns(
        descriptor_cols=desc_names_current,
        target_col=target_col,
        data=df_desc if isinstance(df_desc, pd.DataFrame) else None,
        y=y_all
    )

    if leakage_df.empty:
        st.success(t('leakage_control.success'))
    else:
        qspr_show_data_leakage_warning(
            leakage_df,
            title=t('leakage_control.warning_title')
        )
        st.error(t('leakage_control.error_text'))

        allow_leakage_training = st.checkbox(
            t('leakage_control.checkbox_label'),
            value=False,
            key="allow_training_with_possible_leakage"
        )

        if not allow_leakage_training:
            st.stop()

except Exception as e:
    st.warning(t('leakage_control.failed', error=e))

# ------------------------------------------------------------------
# Descriptor diagnostics

st.subheader(t('descriptor_diagnostics.title'))

show_markdown_help(
    t('mahalanobis.help_title'),
    os.path.join(HELP_DIR, "mahalanobis_help.md"),
    expanded=False
)

fig_mahal = None
fig_corr_heat = None
mahal_table_for_view = None
corr_table = pd.DataFrame()

mahal_message = None
mahal_message_type = "info"

# ------------------------------------------------------------
# 1. Расчёт Махаланобиса

if len(y_all) > X_all.shape[1] + 10:
    try:
        scaler_maha = StandardScaler()
        X_scaled_maha = scaler_maha.fit_transform(X_all)

        cov = np.cov(X_scaled_maha.T)
        cov_inv = np.linalg.pinv(cov)
        mean_vec = np.mean(X_scaled_maha, axis=0)

        mahal_dist = np.array([
            mahalanobis(x, mean_vec, cov_inv)
            for x in X_scaled_maha
        ])

        dof = X_all.shape[1]
        thresh = np.sqrt(chi2.ppf(0.95, dof))
        outliers_mahal = np.where(mahal_dist > thresh)[0]

        fig_mahal, ax_mahal = plt.subplots(figsize=(6, 4))

        ax_mahal.scatter(
            range(len(mahal_dist)),
            mahal_dist,
            alpha=0.65,
            s=22
        )

        ax_mahal.axhline(
            y=thresh,
            color="r",
            linestyle="--",
            label=t('mahalanobis.threshold_label')
        )

        ax_mahal.set_xlabel(t('mahalanobis.xlabel'))
        ax_mahal.set_ylabel(t('mahalanobis.ylabel'))
        ax_mahal.set_title(t('mahalanobis.title'))
        ax_mahal.legend(fontsize=8)
        ax_mahal.grid(True, alpha=0.25)
        fig_mahal.tight_layout()

        if len(outliers_mahal) > 0:
            outlier_rows = []

            possible_name_cols = [
                "Name",
                "name",
                "Название",
                "Compound",
                "compound",
                "Molecule",
                "molecule"
            ]

            name_col_for_outliers = None

            for name_candidate in possible_name_cols:
                if name_candidate in data.columns:
                    name_col_for_outliers = name_candidate
                    break

            for local_pos in outliers_mahal:
                original_index = valid_indices[local_pos]
                smiles_value = str(data[smiles_col_current].iloc[original_index])

                if name_col_for_outliers is not None:
                    compound_name = data[name_col_for_outliers].iloc[original_index]
                else:
                    compound_name = ""

                try:
                    mol_for_inchi = Chem.MolFromSmiles(smiles_value)

                    if mol_for_inchi is not None:
                        inchikey_value = Chem.MolToInchiKey(mol_for_inchi)
                    else:
                        inchikey_value = ""

                except Exception:
                    inchikey_value = ""

                mahal_value = float(mahal_dist[local_pos])

                reason_text = t('mahalanobis.reason_text', mahal=mahal_value, thresh=thresh)

                outlier_rows.append({
                    t('mahalanobis.col_num'): len(outlier_rows) + 1,
                    t('mahalanobis.col_original_row'): int(original_index) + 1,
                    t('mahalanobis.col_smiles'): smiles_value,
                    t('mahalanobis.col_name'): compound_name,
                    t('mahalanobis.col_inchikey'): inchikey_value,
                    t('mahalanobis.col_property', col=target_col): y_all[local_pos],
                    t('mahalanobis.col_distance'): mahal_value,
                    t('mahalanobis.col_threshold'): float(thresh),
                    t('mahalanobis.col_reason'): reason_text
                })

            mahal_table_for_view = pd.DataFrame(outlier_rows)

            mahal_message = t('mahalanobis.outliers_found', count=len(outliers_mahal))
            mahal_message_type = "warning"

        else:
            mahal_message = t('mahalanobis.no_outliers')
            mahal_message_type = "success"

    except Exception as e:
        mahal_message = t('mahalanobis.error', error=e)
        mahal_message_type = "warning"
else:
    mahal_message = t('mahalanobis.insufficient_data')
    mahal_message_type = "info"

# ------------------------------------------------------------
# 2. Расчёт корреляций

corr_values = []

y_series_for_corr = pd.Series(y_all).reset_index(drop=True)

df_desc_corr = df_desc.copy().reset_index(drop=True)

for col in df_desc_corr.columns:
    try:
        x = pd.to_numeric(df_desc_corr[col], errors="coerce")

        valid_mask = x.notna() & y_series_for_corr.notna()

        if valid_mask.sum() < 3:
            continue

        corr = x.loc[valid_mask].corr(y_series_for_corr.loc[valid_mask])

        if np.isfinite(corr):
            corr_values.append((col, abs(corr), corr))

    except Exception:
        pass

corr_values.sort(key=lambda x: x[1], reverse=True)
top_corr = corr_values[:20]

corr_table = pd.DataFrame({
    "№": range(1, len(top_corr) + 1),
    t('corr_table.descriptor'): [c[0] for c in top_corr],
    t('corr_table.abs_corr'): [round(c[1], 3) for c in top_corr],
    t('corr_table.corr'): [round(c[2], 3) for c in top_corr],
    t('corr_table.sign'): ["+" if c[2] >= 0 else "-" for c in top_corr]
})

# ------------------------------------------------------------
# 2.1. Отбор дескрипторов для тепловой карты
# Берём сильные по связи со свойством, но убираем почти дублирующие друг друга.

candidate_desc = [c[0] for c in corr_values]

selected_heatmap_desc = []
max_inter_descriptor_corr = 0.95

for desc in candidate_desc:
    if len(selected_heatmap_desc) >= 15:
        break

    if desc not in df_desc_corr.columns:
        continue

    if not selected_heatmap_desc:
        selected_heatmap_desc.append(desc)
        continue

    is_too_similar = False

    for selected_desc in selected_heatmap_desc:
        try:
            c_pair = df_desc_corr[desc].corr(df_desc_corr[selected_desc])

            if np.isfinite(c_pair) and abs(c_pair) >= max_inter_descriptor_corr:
                is_too_similar = True
                break

        except Exception:
            pass

    if not is_too_similar:
        selected_heatmap_desc.append(desc)

# Если после фильтра осталось слишком мало, fallback: обычный топ-15.
if len(selected_heatmap_desc) < 2:
    selected_heatmap_desc = candidate_desc[:15]

fig_corr_heat = None

if len(selected_heatmap_desc) >= 2:
    heat_df = df_desc_corr[selected_heatmap_desc].copy()
    heat_df[target_col] = y_series_for_corr.values

    heat_df = heat_df.apply(pd.to_numeric, errors="coerce")
    heat_df = heat_df.replace([np.inf, -np.inf], np.nan)

    corr_matrix = heat_df.corr(method="pearson", numeric_only=True)

    # Для компактности: свойство ставим первым.
    ordered_cols = [target_col] + [c for c in corr_matrix.columns if c != target_col]
    corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]

    fig_corr_heat, ax_corr_heat = plt.subplots(figsize=(7, 5.5))

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "r"},
        ax=ax_corr_heat
    )

    ax_corr_heat.set_title(t('corr_heatmap.title'))
    ax_corr_heat.tick_params(axis="x", labelrotation=90)
    ax_corr_heat.tick_params(axis="y", labelrotation=0)

    fig_corr_heat.tight_layout()
# ------------------------------------------------------------
# 3. Первая строка: два графика

col_plot_1, col_plot_2 = st.columns(2)

with col_plot_1:
    st.markdown(t('mahalanobis.plot_section_title'))

    if fig_mahal is not None:
        st.pyplot(fig_mahal)
    else:
        st.info(t('mahalanobis.plot_not_built'))

    if mahal_message_type == "warning":
        st.warning(mahal_message)
    elif mahal_message_type == "success":
        st.success(mahal_message)
    else:
        st.info(mahal_message)

with col_plot_2:
    st.markdown(t('corr_heatmap.section_title'))

    if fig_corr_heat is not None:
        st.pyplot(fig_corr_heat)
        st.caption(t('corr_heatmap.caption'))
    else:
        st.info(t('corr_heatmap.insufficient_descriptors'))

# ------------------------------------------------------------
# 4. Вторая строка: две таблицы

col_table_1, col_table_2 = st.columns(2)

with col_table_1:
    st.markdown(t('mahalanobis.outliers_section_title'))

    if mahal_table_for_view is not None and not mahal_table_for_view.empty:
        st.dataframe(
            mahal_table_for_view,
            width="stretch",
            hide_index=True
        )

        with st.expander(t('mahalanobis.interpret_expander'), expanded=False):
            st.markdown(t('mahalanobis.interpret_text'))
    else:
        st.info(t('mahalanobis.outliers_table_empty'))

with col_table_2:
    st.markdown(t('corr_table.section_title'))

    if corr_table is not None and not corr_table.empty:
        st.dataframe(
            corr_table,
            width="stretch",
            hide_index=True
        )
    else:
        st.info(t('corr_table.not_calculated'))

# ------------------------------------------------------------
# 5. Ниже на всю ширину: структуры веществ

if mahal_table_for_view is not None and not mahal_table_for_view.empty:
    show_molecule_grid_from_table(
        table_df=mahal_table_for_view,
        title=t('mahalanobis.structures_title'),
        target_col=t('mahalanobis.col_property', col=target_col),
        smiles_col="SMILES",
        max_molecules=100,
        key_prefix="mahalanobis_outliers_full_width"
    )
    
remove_outliers_choice = st.checkbox(
    t('outliers_remove.checkbox_label'),
    help=t('outliers_remove.help_text')
)

X_all_current = X_all
y_all_current = y_all
valid_indices_current = list(valid_indices)
df_desc_current = df_desc

if remove_outliers_choice:
    q1, q3 = np.percentile(y_all_current, [25, 75])
    iqr = q3 - q1
    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr

    std_y = np.std(y_all_current)

    if std_y <= 1e-12:
        z_scores = np.zeros_like(y_all_current)
    else:
        z_scores = np.abs((y_all_current - np.mean(y_all_current)) / std_y)

    outliers_iqr = np.where((y_all_current < lower_iqr) | (y_all_current > upper_iqr))[0]
    outliers_z = np.where(z_scores > 3)[0]
    outliers_all = np.unique(np.concatenate([outliers_iqr, outliers_z]))

    if len(outliers_all) > 0:
        keep = np.ones(len(y_all_current), dtype=bool)
        keep[outliers_all] = False

        X_all_current = X_all_current[keep]
        y_all_current = y_all_current[keep]
        valid_indices_current = [valid_indices_current[i] for i in range(len(valid_indices_current)) if keep[i]]
        df_desc_current = df_desc_current.iloc[keep].reset_index(drop=True)

        st.success(t('outliers_remove.removed_success', removed=len(outliers_all), remaining=len(y_all_current)))
    else:
        st.info(t('outliers_remove.no_outliers_to_remove'))

# ------------------------------------------------------------------
# Model training

training_context = render_training_section({**globals(), **locals()})
globals().update(training_context)

# ------------------------------------------------------------------
# Validation

render_validation_section({**globals(), **locals()})
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Model diagnostics

if model_name in st.session_state.trained_models:
    render_model_diagnostics_section({**globals(), **locals()})

# Chemical error profiling

render_error_analysis_section({**globals(), **locals()})

# Save verified model

if model_name in st.session_state.trained_models:
    verified_model_data = st.session_state.trained_models[model_name]
    verified_validation_completed = any([
        model_name in st.session_state.get("holdout_results_dict", {}),
        model_name in st.session_state.get("kfold_results_dict", {}),
        model_name in st.session_state.get("loo_results_dict", {}),
        bool(st.session_state.get("ext_validation_result")),
    ])
    render_verified_model_save(
        model_name=model_name,
        model_data=verified_model_data,
        target_col=target_col,
        smiles_col=smiles_col_current,
        descriptor_names=desc_names_current,
        X_train=X_all_current,
        y_train=y_all_current,
        train_smiles=(
            data[smiles_col_current]
            .iloc[list(valid_indices_current)]
            .astype(str)
            .tolist()
        ),
        validation_completed=verified_validation_completed,
        add_log=add_log,
    )

# ------------------------------------------------------------------
# Model comparison

render_model_comparison_section({**globals(), **locals()})

# ------------------------------------------------------------------
# Consensus prediction

render_consensus_section({**globals(), **locals()})

# ------------------------------------------------------------------
# Prognostic model

qspr_show_prognostic_training_section(
    data=data,
    model_name=model_name,
    target_col=target_col,
    smiles_col_current=smiles_col_current,
    desc_names_current=desc_names_current,
    X_all_current=X_all_current,
    y_all_current=y_all_current,
    valid_indices_current=valid_indices_current,
    get_model_params_from_session=get_model_params_from_session,
    add_log=add_log,
)

# ------------------------------------------------------------------
# Final statistics

with st.expander(t("final_stats.expander_title"), expanded=False):
    render_final_statistics_summary({**globals(), **locals()})

# ------------------------------------------------------------------
# Report

render_report_section({**globals(), **locals()})




