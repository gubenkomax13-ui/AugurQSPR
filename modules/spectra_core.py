# -*- coding: utf-8 -*-

"""
spectra_core.py

Ядро работы со спектральной базой GenQSPR:
- подготовка веществ из SMILES;
- spectra_bank;
- поиск IR-спектров в NIST Chemistry WebBook;
- скачивание JCAMP-DX;
- чтение JDX/DX;
- сохранение raw/processed спектров;
- построение спектральных дескрипторов:
  IR_GRID, IR_BIN, IR_BAND, IR_SVD.
"""

import os
import re
import json
import time
import uuid
import shutil
import urllib.parse
import urllib.request
import threading
from datetime import datetime
import requests

import numpy as np
import pandas as pd

from rdkit import Chem
from sklearn.decomposition import TruncatedSVD

SPECTRA_HTTP_TIMEOUT = 20

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Папки spectra_bank
# Важно: пути считаются от папки проекта, а не от os.getcwd().
# Если файл лежит в E:\QSPR Forge\modules\spectra_core.py,
# банк будет здесь: E:\QSPR Forge\spectra_bank

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(MODULE_DIR)

SPECTRA_BANK_DIR = os.path.join(PROJECT_ROOT_DIR, "spectra_bank")

SPECTRA_INDEX_FILE = os.path.join(
    SPECTRA_BANK_DIR,
    "spectra_index.csv"
)

SPECTRA_SEARCH_CACHE_FILE = os.path.join(
    SPECTRA_BANK_DIR,
    "spectra_search_cache.csv"
)

SPECTRA_REMOTE_INDEX_URL = os.environ.get("AUGUR_SPECTRA_INDEX_URL", "").strip()
SPECTRA_REMOTE_INDEX_FILE_ID = os.environ.get("AUGUR_SPECTRA_INDEX_FILE_ID", "").strip()
SPECTRA_REMOTE_MANIFEST_URL = os.environ.get("AUGUR_SPECTRA_MANIFEST_URL", "").strip()
SPECTRA_REMOTE_MANIFEST_FILE_ID = os.environ.get("AUGUR_SPECTRA_MANIFEST_FILE_ID", "").strip()
SPECTRA_REMOTE_SEARCH_CACHE_URL = os.environ.get("AUGUR_SPECTRA_SEARCH_CACHE_URL", "").strip()
SPECTRA_REMOTE_SEARCH_CACHE_FILE_ID = os.environ.get("AUGUR_SPECTRA_SEARCH_CACHE_FILE_ID", "").strip()
SPECTRA_REMOTE_IR_DESCRIPTOR_CACHE_URL = (
    os.environ.get("AUGUR_SPECTRA_IR_DESCRIPTOR_CACHE_URL", "").strip()
    or os.environ.get("AUGUR_IR_SPECTRAL_DESCRIPTOR_CACHE_URL", "").strip()
)
SPECTRA_REMOTE_MASS_DESCRIPTOR_CACHE_URL = (
    os.environ.get("AUGUR_SPECTRA_MASS_DESCRIPTOR_CACHE_URL", "").strip()
    or os.environ.get("AUGUR_MASS_SPECTRAL_DESCRIPTOR_CACHE_URL", "").strip()
)
SPECTRA_REMOTE_DESCRIPTOR_CACHE_URL = os.environ.get(
    "AUGUR_SPECTRA_DESCRIPTOR_CACHE_URL",
    ""
).strip()
SPECTRA_REMOTE_DESCRIPTOR_SHARDS_BASE_URL = os.environ.get(
    "AUGUR_SPECTRA_DESCRIPTOR_SHARDS_BASE_URL",
    ""
).strip()
SPECTRA_DEFAULT_DESCRIPTOR_SHARDS_BASE_URL = (
    "https://raw.githubusercontent.com/gubenkomax13-ui/"
    "AugurQSPR/main/spectral_descriptor_shards"
)
SPECTRA_REMOTE_BANK_FOLDER_URL = os.environ.get("AUGUR_SPECTRA_BANK_FOLDER_URL", "").strip()
SPECTRA_REMOTE_BANK_FOLDER_ID = os.environ.get("AUGUR_SPECTRA_BANK_FOLDER_ID", "").strip()
SPECTRA_GOOGLE_DRIVE_API_KEY = os.environ.get("AUGUR_GOOGLE_DRIVE_API_KEY", "").strip()
SPECTRA_REMOTE_MANIFEST_FILE = os.path.join(
    SPECTRA_BANK_DIR,
    "spectra_manifest.csv"
)
SPECTRA_DESCRIPTOR_SHARDS_DIR = os.path.join(
    PROJECT_ROOT_DIR,
    "spectral_descriptor_shards"
)

SPECTRA_IR_DIR = os.path.join(SPECTRA_BANK_DIR, "IR")
SPECTRA_MASS_DIR = os.path.join(SPECTRA_BANK_DIR, "Mass")

# IR
SPECTRA_IR_RAW_DIR = os.path.join(SPECTRA_IR_DIR, "raw_jdx")
SPECTRA_IR_PROCESSED_DIR = os.path.join(SPECTRA_IR_DIR, "processed")
SPECTRA_IR_DESCRIPTORS_DIR = os.path.join(SPECTRA_IR_DIR, "descriptors")
SPECTRA_IR_LOG_DIR = os.path.join(SPECTRA_IR_DIR, "search_log")
SPECTRA_IR_DESCRIPTOR_CACHE_LEGACY_FILE = os.path.join(
    SPECTRA_IR_DESCRIPTORS_DIR,
    "IR_spectral_descriptor_cache.csv"
)
SPECTRA_IR_DESCRIPTOR_CACHE_FILE = os.path.join(
    PROJECT_ROOT_DIR,
    "spectral_descriptor_bank_IR.csv"
)

# Mass
SPECTRA_MASS_RAW_DIR = os.path.join(SPECTRA_MASS_DIR, "raw_jdx")
SPECTRA_MASS_PROCESSED_DIR = os.path.join(SPECTRA_MASS_DIR, "processed")
SPECTRA_MASS_DESCRIPTORS_DIR = os.path.join(SPECTRA_MASS_DIR, "descriptors")
SPECTRA_MASS_LOG_DIR = os.path.join(SPECTRA_MASS_DIR, "search_log")
SPECTRA_MASS_DESCRIPTOR_CACHE_LEGACY_FILE = os.path.join(
    SPECTRA_MASS_DESCRIPTORS_DIR,
    "Mass_spectral_descriptor_cache.csv"
)
SPECTRA_MASS_DESCRIPTOR_CACHE_FILE = os.path.join(
    PROJECT_ROOT_DIR,
    "spectral_descriptor_bank_Mass.csv"
)

# Таблицы неудачных поисков
SPECTRA_IR_NOT_FOUND_FILE = os.path.join(
    SPECTRA_IR_DIR,
    "ir_not_found_searches.csv"
)

SPECTRA_MASS_NOT_FOUND_FILE = os.path.join(
    SPECTRA_MASS_DIR,
    "mass_not_found_searches.csv"
)

# Подробные JSON-логи в search_log оставляем только как отладочный режим.
# Для обычной работы лучше False.
SPECTRA_WRITE_JSON_SEARCH_LOGS = False

for _d in [
    SPECTRA_BANK_DIR,

    SPECTRA_IR_DIR,
    SPECTRA_IR_RAW_DIR,
    SPECTRA_IR_PROCESSED_DIR,
    SPECTRA_IR_DESCRIPTORS_DIR,
    SPECTRA_IR_LOG_DIR,

    SPECTRA_MASS_DIR,
    SPECTRA_MASS_RAW_DIR,
    SPECTRA_MASS_PROCESSED_DIR,
    SPECTRA_MASS_DESCRIPTORS_DIR,
    SPECTRA_MASS_LOG_DIR,
]:
    os.makedirs(_d, exist_ok=True)


def spectra_google_drive_download_url(file_id):
    """
    Формирует прямой download URL для публичного файла Google Drive.
    """
    file_id = str(file_id or "").strip()

    if not file_id:
        return ""

    return (
        "https://drive.google.com/uc?"
        + urllib.parse.urlencode({"export": "download", "id": file_id})
    )


def spectra_google_drive_file_id_from_url(url):
    """
    Достаёт file_id из обычной ссылки Google Drive на файл.
    """
    url = str(url or "").strip()

    if not url or "drive.google.com" not in url:
        return ""

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)

    if query.get("id"):
        return str(query["id"][0]).strip()

    parts = [p for p in parsed.path.split("/") if p]

    if "d" in parts:
        d_idx = parts.index("d")

        if d_idx + 1 < len(parts):
            return parts[d_idx + 1].strip()

    return ""


def spectra_google_drive_folder_id_from_url(url):
    """
    Достаёт folder_id из ссылки Google Drive на папку.
    """
    url = str(url or "").strip()

    if not url or "drive.google.com" not in url:
        return ""

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)

    if query.get("id"):
        return str(query["id"][0]).strip()

    parts = [p for p in parsed.path.split("/") if p]

    if "folders" in parts:
        folder_idx = parts.index("folders")

        if folder_idx + 1 < len(parts):
            return parts[folder_idx + 1].strip()

    return ""


def spectra_normalize_download_url(url):
    """
    Превращает обычную Drive file-ссылку в прямую download-ссылку.
    """
    url = str(url or "").strip()

    if not url:
        return ""

    file_id = spectra_google_drive_file_id_from_url(url)

    if file_id:
        return spectra_google_drive_download_url(file_id)

    return url


def spectra_download_public_file(url, filepath, timeout=60):
    """
    Скачивает небольшой публичный файл в локальный runtime cache.
    """
    url = spectra_normalize_download_url(url)

    if not url:
        return False

    headers = {
        "User-Agent": "Mozilla/5.0 Augur-QSPR-SpectraBank/1.0",
        "Accept": "text/csv,application/json,text/plain,application/octet-stream,*/*",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
        allow_redirects=True,
    )

    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


def spectra_download_public_drive_file(file_id, filepath, timeout=60):
    """
    Скачивает публичный файл Google Drive по file_id.
    """
    url = spectra_google_drive_download_url(file_id)

    if not url:
        return False

    return spectra_download_public_file(url, filepath, timeout=timeout)


def spectra_get_remote_bank_folder_id():
    """
    Возвращает ID корневой Google Drive-папки spectra_bank.
    """
    folder_id = str(SPECTRA_REMOTE_BANK_FOLDER_ID or "").strip()

    if folder_id:
        return folder_id

    return spectra_google_drive_folder_id_from_url(SPECTRA_REMOTE_BANK_FOLDER_URL)


def spectra_drive_api_list_children(folder_id):
    """
    Возвращает список прямых детей Google Drive-папки через Drive API.
    Требуется публичная папка и AUGUR_GOOGLE_DRIVE_API_KEY.
    """
    folder_id = str(folder_id or "").strip()

    if not folder_id or not SPECTRA_GOOGLE_DRIVE_API_KEY:
        return []

    files = []
    page_token = ""

    while True:
        params = {
            "key": SPECTRA_GOOGLE_DRIVE_API_KEY,
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            "pageSize": 1000,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            params=params,
            timeout=60,
        )

        response.raise_for_status()
        payload = response.json()
        files.extend(payload.get("files", []))

        page_token = payload.get("nextPageToken", "")

        if not page_token:
            break

    return files


def spectra_build_manifest_from_drive_folder():
    """
    Обходит Google Drive-папку и строит manifest path -> file_id.
    Спектры не скачиваются, читаются только метаданные файлов.
    """
    root_folder_id = spectra_get_remote_bank_folder_id()

    if not root_folder_id or not SPECTRA_GOOGLE_DRIVE_API_KEY:
        return pd.DataFrame()

    folder_mime = "application/vnd.google-apps.folder"
    rows = []
    stack = [(root_folder_id, "")]

    while stack:
        folder_id, rel_prefix = stack.pop()

        try:
            children = spectra_drive_api_list_children(folder_id)
        except Exception:
            continue

        for item in children:
            name = str(item.get("name", "")).strip()
            file_id = str(item.get("id", "")).strip()
            mime_type = str(item.get("mimeType", "")).strip()

            if not name or not file_id:
                continue

            child_rel = f"{rel_prefix}/{name}".strip("/")
            child_rel = spectra_normalize_bank_relative_path(child_rel)

            if mime_type == folder_mime:
                stack.append((file_id, child_rel))
                continue

            rows.append({
                "path": child_rel,
                "file_id": file_id,
                "name": name,
                "mime_type": mime_type,
                "size": item.get("size", ""),
                "modified_time": item.get("modifiedTime", ""),
            })

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        return manifest

    manifest = manifest.drop_duplicates(subset=["path"], keep="last")
    manifest = manifest.sort_values("path").reset_index(drop=True)

    return manifest


def spectra_save_remote_manifest(manifest_df):
    """
    Сохраняет сгенерированный manifest в runtime spectra_bank.
    """
    if manifest_df is None or manifest_df.empty:
        return False

    os.makedirs(SPECTRA_BANK_DIR, exist_ok=True)
    manifest_df.to_csv(SPECTRA_REMOTE_MANIFEST_FILE, index=False, encoding="utf-8-sig")

    return True


def spectra_manifest_needs_drive_rebuild():
    """
    Проверяет, похоже ли, что manifest не содержит файлов спектров.
    """
    if not os.path.exists(SPECTRA_REMOTE_MANIFEST_FILE):
        return True

    try:
        manifest = pd.read_csv(SPECTRA_REMOTE_MANIFEST_FILE)
    except Exception:
        return True

    if manifest.empty:
        return True

    if len(manifest) < 10:
        return True

    path_cols = [
        c for c in ["path", "relative_path", "bank_path", "processed_file"]
        if c in manifest.columns
    ]

    if not path_cols:
        return True

    path_series = pd.Series("", index=manifest.index, dtype=str)

    for col in path_cols:
        normalized = manifest[col].astype(str).apply(spectra_normalize_bank_relative_path)
        path_series = path_series.where(path_series.astype(str).str.strip() != "", normalized)

    processed_mask = (
        path_series.str.contains("/processed/", case=False, regex=False)
        & path_series.str.lower().str.endswith(".csv")
    )

    return int(processed_mask.sum()) == 0


def spectra_ensure_remote_index():
    """
    Подтягивает только spectra_index.csv, если локального индекса ещё нет.
    """
    if os.path.exists(SPECTRA_INDEX_FILE) and os.path.getsize(SPECTRA_INDEX_FILE) > 0:
        return True

    try:
        if SPECTRA_REMOTE_INDEX_URL:
            return spectra_download_public_file(
                SPECTRA_REMOTE_INDEX_URL,
                SPECTRA_INDEX_FILE,
                timeout=60
            )

        if SPECTRA_REMOTE_INDEX_FILE_ID:
            return spectra_download_public_drive_file(
                SPECTRA_REMOTE_INDEX_FILE_ID,
                SPECTRA_INDEX_FILE,
                timeout=60
            )
    except Exception:
        return False

    return False


def spectra_ensure_remote_manifest():
    """
    Подтягивает маленький manifest path -> file_id для ленивой загрузки спектров.

    Ожидаемые колонки:
    - path / relative_path / processed_file / raw_file
    - file_id / drive_file_id / google_drive_file_id
    Дополнительно можно использовать download_url / url.
    """
    drive_folder_configured = bool(
        spectra_get_remote_bank_folder_id() and SPECTRA_GOOGLE_DRIVE_API_KEY
    )

    if (
        os.path.exists(SPECTRA_REMOTE_MANIFEST_FILE)
        and os.path.getsize(SPECTRA_REMOTE_MANIFEST_FILE) > 0
        and not (drive_folder_configured and spectra_manifest_needs_drive_rebuild())
    ):
        return True

    try:
        if drive_folder_configured:
            manifest = spectra_build_manifest_from_drive_folder()

            if spectra_save_remote_manifest(manifest):
                return True

        if SPECTRA_REMOTE_MANIFEST_URL and not os.path.exists(SPECTRA_REMOTE_MANIFEST_FILE):
            return spectra_download_public_file(
                SPECTRA_REMOTE_MANIFEST_URL,
                SPECTRA_REMOTE_MANIFEST_FILE,
                timeout=60
            )

        if SPECTRA_REMOTE_MANIFEST_FILE_ID and not os.path.exists(SPECTRA_REMOTE_MANIFEST_FILE):
            return spectra_download_public_drive_file(
                SPECTRA_REMOTE_MANIFEST_FILE_ID,
                SPECTRA_REMOTE_MANIFEST_FILE,
                timeout=60
            )
    except Exception:
        return False

    return False


def spectra_ensure_remote_search_cache():
    """
    Подтягивает маленький spectra_search_cache.csv, если он настроен и локального файла нет.
    """
    if (
        os.path.exists(SPECTRA_SEARCH_CACHE_FILE)
        and os.path.getsize(SPECTRA_SEARCH_CACHE_FILE) > 0
    ):
        return True

    try:
        if SPECTRA_REMOTE_SEARCH_CACHE_URL:
            return spectra_download_public_file(
                SPECTRA_REMOTE_SEARCH_CACHE_URL,
                SPECTRA_SEARCH_CACHE_FILE,
                timeout=60
            )

        if SPECTRA_REMOTE_SEARCH_CACHE_FILE_ID:
            return spectra_download_public_drive_file(
                SPECTRA_REMOTE_SEARCH_CACHE_FILE_ID,
                SPECTRA_SEARCH_CACHE_FILE,
                timeout=60
            )
    except Exception:
        return False

    return False


def spectra_normalize_bank_relative_path(path):
    """
    Приводит путь к виду относительно spectra_bank с прямыми слэшами.
    """
    value = str(path or "").strip().strip('"').strip("'")

    if not value:
        return ""

    value = value.replace("\\", "/")
    bank_root = os.path.abspath(SPECTRA_BANK_DIR).replace("\\", "/")

    try:
        abs_value = os.path.abspath(value).replace("\\", "/")

        if abs_value.startswith(bank_root + "/"):
            value = abs_value[len(bank_root) + 1:]
        else:
            marker = "/spectra_bank/"
            marker_idx = value.lower().find(marker)

            if marker_idx >= 0:
                value = value[marker_idx + len(marker):]
    except Exception:
        pass

    value = value.lstrip("./")

    if value.lower().startswith("spectra_bank/"):
        value = value[len("spectra_bank/"):]

    return value


def spectra_load_remote_manifest():
    """
    Загружает manifest удалённой Google Drive-базы.
    """
    if not spectra_ensure_remote_manifest():
        return pd.DataFrame()

    try:
        manifest = pd.read_csv(SPECTRA_REMOTE_MANIFEST_FILE)
    except Exception:
        return pd.DataFrame()

    if manifest.empty:
        return pd.DataFrame()

    return manifest


def spectra_remote_bank_status():
    """
    Возвращает диагностическую сводку по локальной/удалённой спектральной базе.
    """
    status = {
        "spectra_bank_dir": SPECTRA_BANK_DIR,
        "index_file": SPECTRA_INDEX_FILE,
        "index_exists": os.path.exists(SPECTRA_INDEX_FILE),
        "index_rows": 0,
        "index_ir_rows": 0,
        "index_mass_rows": 0,
        "manifest_file": SPECTRA_REMOTE_MANIFEST_FILE,
        "manifest_exists": os.path.exists(SPECTRA_REMOTE_MANIFEST_FILE),
        "manifest_rows": 0,
        "manifest_processed_rows": 0,
        "search_cache_file": SPECTRA_SEARCH_CACHE_FILE,
        "search_cache_exists": os.path.exists(SPECTRA_SEARCH_CACHE_FILE),
        "search_cache_rows": 0,
        "remote_bank_folder_configured": bool(spectra_get_remote_bank_folder_id()),
        "google_drive_api_key_configured": bool(SPECTRA_GOOGLE_DRIVE_API_KEY),
        "remote_index_configured": bool(SPECTRA_REMOTE_INDEX_URL or SPECTRA_REMOTE_INDEX_FILE_ID),
        "remote_manifest_configured": bool(SPECTRA_REMOTE_MANIFEST_URL or SPECTRA_REMOTE_MANIFEST_FILE_ID),
        "remote_search_cache_configured": bool(
            SPECTRA_REMOTE_SEARCH_CACHE_URL or SPECTRA_REMOTE_SEARCH_CACHE_FILE_ID
        ),
        "remote_descriptor_cache_configured": bool(
            SPECTRA_REMOTE_IR_DESCRIPTOR_CACHE_URL
            or SPECTRA_REMOTE_MASS_DESCRIPTOR_CACHE_URL
            or SPECTRA_REMOTE_DESCRIPTOR_CACHE_URL
        ),
        "remote_descriptor_shards_configured": bool(
            SPECTRA_REMOTE_DESCRIPTOR_SHARDS_BASE_URL
            or SPECTRA_DEFAULT_DESCRIPTOR_SHARDS_BASE_URL
        ),
        "descriptor_shards_dir": SPECTRA_DESCRIPTOR_SHARDS_DIR,
        "local_ir_descriptor_shard_files": 0,
        "local_mass_descriptor_shard_files": 0,
        "local_ir_descriptor_shard_rows": 0,
        "local_mass_descriptor_shard_rows": 0,
        "local_ir_processed_files": 0,
        "local_mass_processed_files": 0,
        "local_ir_descriptor_cache_exists": os.path.exists(SPECTRA_IR_DESCRIPTOR_CACHE_FILE),
        "local_mass_descriptor_cache_exists": os.path.exists(SPECTRA_MASS_DESCRIPTOR_CACHE_FILE),
        "local_ir_descriptor_cache_rows": 0,
        "local_mass_descriptor_cache_rows": 0,
    }

    try:
        spectra_ensure_remote_index()
        status["index_exists"] = os.path.exists(SPECTRA_INDEX_FILE)

        if status["index_exists"]:
            index_df = pd.read_csv(SPECTRA_INDEX_FILE)
            status["index_rows"] = int(len(index_df))

            if "spectrum_type" in index_df.columns:
                type_norm = index_df["spectrum_type"].astype(str).apply(
                    spectra_normalize_spectrum_type
                )
                status["index_ir_rows"] = int((type_norm == "IR").sum())
                status["index_mass_rows"] = int((type_norm == "Mass").sum())
    except Exception as e:
        status["index_error"] = str(e)

    try:
        if (
            status["remote_manifest_configured"]
            or os.path.exists(SPECTRA_REMOTE_MANIFEST_FILE)
        ):
            spectra_ensure_remote_manifest()

        status["manifest_exists"] = os.path.exists(SPECTRA_REMOTE_MANIFEST_FILE)

        if status["manifest_exists"]:
            manifest_df = pd.read_csv(SPECTRA_REMOTE_MANIFEST_FILE)
            status["manifest_rows"] = int(len(manifest_df))

            manifest_path_cols = [
                c for c in ["path", "relative_path", "bank_path", "processed_file"]
                if c in manifest_df.columns
            ]

            if manifest_path_cols:
                path_series = pd.Series("", index=manifest_df.index, dtype=str)

                for col in manifest_path_cols:
                    normalized = manifest_df[col].astype(str).apply(
                        spectra_normalize_bank_relative_path
                    )
                    path_series = path_series.where(
                        path_series.astype(str).str.strip() != "",
                        normalized
                    )

                processed_mask = (
                    path_series.str.contains("/processed/", case=False, regex=False)
                    & path_series.str.lower().str.endswith(".csv")
                )
                status["manifest_processed_rows"] = int(processed_mask.sum())
    except Exception as e:
        status["manifest_error"] = str(e)

    try:
        if (
            status["remote_search_cache_configured"]
            or os.path.exists(SPECTRA_SEARCH_CACHE_FILE)
        ):
            spectra_ensure_remote_search_cache()

        status["search_cache_exists"] = os.path.exists(SPECTRA_SEARCH_CACHE_FILE)

        if status["search_cache_exists"]:
            cache_df = pd.read_csv(SPECTRA_SEARCH_CACHE_FILE)
            status["search_cache_rows"] = int(len(cache_df))
    except Exception as e:
        status["search_cache_error"] = str(e)

    try:
        if os.path.isdir(SPECTRA_IR_PROCESSED_DIR):
            status["local_ir_processed_files"] = len([
                name for name in os.listdir(SPECTRA_IR_PROCESSED_DIR)
                if name.lower().endswith(".csv")
            ])
    except Exception:
        pass

    try:
        if os.path.isdir(SPECTRA_MASS_PROCESSED_DIR):
            status["local_mass_processed_files"] = len([
                name for name in os.listdir(SPECTRA_MASS_PROCESSED_DIR)
                if name.lower().endswith(".csv")
            ])
    except Exception:
        pass

    for spectrum_type, exists_key, rows_key in [
        ("IR", "local_ir_descriptor_cache_exists", "local_ir_descriptor_cache_rows"),
        ("Mass", "local_mass_descriptor_cache_exists", "local_mass_descriptor_cache_rows"),
    ]:
        try:
            cache_file = spectral_descriptor_cache_file(spectrum_type)
            status[exists_key] = os.path.exists(cache_file)

            if status[exists_key]:
                cache_df = pd.read_csv(cache_file, low_memory=False)
                status[rows_key] = int(len(cache_df))
        except Exception:
            pass

    for spectrum_type, files_key, rows_key in [
        ("IR", "local_ir_descriptor_shard_files", "local_ir_descriptor_shard_rows"),
        ("Mass", "local_mass_descriptor_shard_files", "local_mass_descriptor_shard_rows"),
    ]:
        try:
            shard_dir = os.path.join(SPECTRA_DESCRIPTOR_SHARDS_DIR, spectrum_type)

            if os.path.isdir(shard_dir):
                status[files_key] = len([
                    name for name in os.listdir(shard_dir)
                    if name.lower().endswith(".csv")
                ])

            manifest_file = os.path.join(
                SPECTRA_DESCRIPTOR_SHARDS_DIR,
                f"spectral_descriptor_manifest_{spectrum_type}.csv"
            )

            if os.path.exists(manifest_file):
                manifest_df = pd.read_csv(manifest_file)

                if "rows" in manifest_df.columns:
                    status[rows_key] = int(pd.to_numeric(
                        manifest_df["rows"],
                        errors="coerce"
                    ).fillna(0).sum())
        except Exception:
            pass

    return status


def spectra_find_remote_manifest_record(local_or_relative_path, spectrum_record=None):
    """
    Находит строку manifest для локального/относительного пути.
    """
    requested_rel = spectra_normalize_bank_relative_path(local_or_relative_path)
    requested_base = os.path.basename(str(local_or_relative_path or ""))

    if spectrum_record is not None:
        direct_file_id = (
            spectrum_record.get("drive_file_id", "")
            or spectrum_record.get("google_drive_file_id", "")
            or spectrum_record.get("file_id", "")
        )

        direct_url = (
            spectrum_record.get("download_url", "")
            or spectrum_record.get("remote_url", "")
        )

        if direct_file_id or direct_url:
            return {
                "file_id": direct_file_id,
                "download_url": direct_url,
                "path": requested_rel,
            }

    manifest = spectra_load_remote_manifest()

    if manifest.empty:
        return None

    path_cols = [
        "path",
        "relative_path",
        "bank_path",
        "processed_file",
        "raw_file",
        "filename",
        "name",
    ]

    id_cols = [
        "file_id",
        "drive_file_id",
        "google_drive_file_id",
        "id",
    ]

    url_cols = [
        "download_url",
        "url",
        "remote_url",
    ]

    for col in path_cols + id_cols + url_cols:
        if col not in manifest.columns:
            manifest[col] = ""

    manifest["_bank_rel_path"] = ""

    for col in path_cols:
        col_paths = manifest[col].astype(str).apply(spectra_normalize_bank_relative_path)
        manifest["_bank_rel_path"] = manifest["_bank_rel_path"].where(
            manifest["_bank_rel_path"].astype(str).str.strip() != "",
            col_paths
        )

    manifest["_bank_base_name"] = manifest["_bank_rel_path"].astype(str).apply(os.path.basename)

    found = manifest[manifest["_bank_rel_path"] == requested_rel].copy()

    if found.empty and requested_base:
        found = manifest[manifest["_bank_base_name"] == requested_base].copy()

    if found.empty:
        return None

    row = found.iloc[0].to_dict()

    file_id = ""
    download_url = ""

    for col in id_cols:
        file_id = str(row.get(col, "")).strip()

        if file_id:
            break

    for col in url_cols:
        download_url = str(row.get(col, "")).strip()

        if download_url:
            break

    if not file_id and not download_url:
        return None

    return {
        "file_id": file_id,
        "download_url": download_url,
        "path": row.get("_bank_rel_path", requested_rel),
    }


def spectra_local_path_for_bank_relative(relative_path):
    """
    Возвращает безопасный локальный путь внутри spectra_bank для relative_path.
    """
    relative_path = spectra_normalize_bank_relative_path(relative_path)

    if not relative_path:
        return ""

    local_path = os.path.abspath(os.path.join(SPECTRA_BANK_DIR, relative_path))
    bank_root = os.path.abspath(SPECTRA_BANK_DIR)

    if not (local_path == bank_root or local_path.startswith(bank_root + os.sep)):
        return ""

    return local_path


def spectra_materialize_remote_bank_file(local_or_relative_path, spectrum_record=None):
    """
    Если файла нет локально, скачивает только этот файл из Google Drive.
    """
    remote_record = spectra_find_remote_manifest_record(
        local_or_relative_path,
        spectrum_record=spectrum_record
    )

    if remote_record is None:
        return ""

    target_rel = remote_record.get("path", "") or spectra_normalize_bank_relative_path(
        local_or_relative_path
    )
    target_path = spectra_local_path_for_bank_relative(target_rel)

    if not target_path:
        base_name = os.path.basename(str(local_or_relative_path or ""))

        if not base_name:
            return ""

        target_path = os.path.join(SPECTRA_BANK_DIR, "remote_cache", base_name)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    try:
        download_url = str(remote_record.get("download_url", "")).strip()
        file_id = str(remote_record.get("file_id", "")).strip()

        if download_url:
            ok = spectra_download_public_file(download_url, target_path, timeout=90)
        else:
            ok = spectra_download_public_drive_file(file_id, target_path, timeout=90)

        if ok:
            return target_path
    except Exception:
        return ""

    return ""

# Защита CSV-файлов от одновременной записи при параллельном IR/Mass-поиске.
SPECTRA_INDEX_LOCK = threading.RLock()
SPECTRA_SEARCH_CACHE_LOCK = threading.RLock()
SPECTRA_NOT_FOUND_LOCK = threading.RLock()
SPECTRA_DESCRIPTOR_CACHE_LOCK = threading.RLock()
SPECTRA_STOP_FILE = os.path.join(
    SPECTRA_BANK_DIR,
    "spectra_search_stop.flag"
)


def spectra_request_stop():
    """
    Создаёт stop-файл. Его видят qspr_app.py и worker-функции spectra_core.py.
    """
    os.makedirs(SPECTRA_BANK_DIR, exist_ok=True)

    with open(SPECTRA_STOP_FILE, "w", encoding="utf-8") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def spectra_clear_stop():
    """
    Удаляет stop-файл перед новым запуском поиска.
    """
    try:
        if os.path.exists(SPECTRA_STOP_FILE):
            os.remove(SPECTRA_STOP_FILE)
    except Exception:
        pass


def spectra_is_stop_requested():
    """
    Проверяет, запрошена ли остановка поиска.
    """
    return os.path.exists(SPECTRA_STOP_FILE)
# ------------------------------------------------------------------
# Общие служебные функции

def spectra_safe_filename_part(value):
    """
    Делает безопасный фрагмент имени файла:
    оставляет буквы, цифры, дефис и подчёркивание.
    """
    value = str(value).strip()

    if value == "" or value.lower() in ["nan", "none"]:
        return "unknown"

    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    if value == "":
        return "unknown"

    return value


def spectra_source_code(source_name):
    """
    Короткое имя источника для имени файла.
    """
    source_name = str(source_name).lower()

    if "nist" in source_name:
        return "NIST"

    if "massbank" in source_name:
        return "MASSBANK"

    if "user" in source_name or "local" in source_name:
        return "LOCAL"

    return "SRC"


def spectra_normalize_spectrum_type(spectrum_type):
    """
    Нормализует тип спектра.
    """
    x = str(spectrum_type).strip().lower()

    if x in ["ir", "infrared", "infrared spectrum", "ик", "ик-спектр"]:
        return "IR"

    if x in ["mass", "ms", "mass spectrum", "mass_spectrum", "масс", "масс-спектр"]:
        return "Mass"

    return str(spectrum_type).strip()


def spectra_get_dirs_by_type(spectrum_type):
    """
    Возвращает папки raw/processed/descriptors/log для заданного типа спектра.

    Поддерживаем:
    - IR
    - Mass
    """
    spectrum_type = spectra_normalize_spectrum_type(spectrum_type)

    if spectrum_type == "Mass":
        return {
            "base": SPECTRA_MASS_DIR,
            "raw": SPECTRA_MASS_RAW_DIR,
            "processed": SPECTRA_MASS_PROCESSED_DIR,
            "descriptors": SPECTRA_MASS_DESCRIPTORS_DIR,
            "log": SPECTRA_MASS_LOG_DIR,
        }

    return {
        "base": SPECTRA_IR_DIR,
        "raw": SPECTRA_IR_RAW_DIR,
        "processed": SPECTRA_IR_PROCESSED_DIR,
        "descriptors": SPECTRA_IR_DESCRIPTORS_DIR,
        "log": SPECTRA_IR_LOG_DIR,
    }


# ------------------------------------------------------------------
# Подготовка веществ

def spectra_make_compound_key_from_mol(mol):
    """
    Возвращает canonical_smiles и InChIKey.
    Главный ключ для спектральной базы — InChIKey.
    """
    if mol is None:
        return "", ""

    try:
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical_smiles = ""

    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = ""

    return canonical_smiles, inchikey


def spectra_prepare_compounds_from_df(input_df, smiles_col):
    """
    Готовит таблицу веществ для поиска спектров.

    Главная логика:
    - структуру берём только из выбранной SMILES-колонки;
    - canonical_smiles и InChIKey рассчитываем через RDKit;
    - поиск спектров идёт по InChIKey;
    - name/CAS сохраняются только как справочные поля.
    """
    if input_df is None or input_df.empty:
        return pd.DataFrame()

    if smiles_col not in input_df.columns:
        raise ValueError(f"Колонка со SMILES не найдена: {smiles_col}")

    rows = []

    for idx, row in input_df.iterrows():
        input_smiles = str(row.get(smiles_col, "")).strip()

        base_row = {
            "row_index": idx,
            "compound_id": row.get("compound_id", ""),
            "input_smiles": input_smiles,
            "canonical_smiles": "",
            "inchikey": "",
            "structure_status": "",
            "valid_structure": False,
            "name": str(row.get("name", row.get("Name", ""))),
            "cas": str(row.get("CAS", row.get("cas", ""))),
        }

        if not input_smiles or input_smiles.lower() in ["nan", "none", ""]:
            base_row["structure_status"] = "empty_smiles"
            rows.append(base_row)
            continue

        mol = Chem.MolFromSmiles(input_smiles)

        if mol is None:
            base_row["structure_status"] = "invalid_smiles"
            rows.append(base_row)
            continue

        canonical_smiles, inchikey = spectra_make_compound_key_from_mol(mol)

        base_row["canonical_smiles"] = canonical_smiles
        base_row["inchikey"] = inchikey
        base_row["structure_status"] = "ok"
        base_row["valid_structure"] = True

        rows.append(base_row)

    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# spectra_index.csv

def spectra_load_index():
    """
    Загружает spectra_index.csv.
    """
    spectra_ensure_remote_index()

    if os.path.exists(SPECTRA_INDEX_FILE):
        try:
            return pd.read_csv(SPECTRA_INDEX_FILE)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame(columns=[
        "spectrum_id",
        "compound_id",
        "name",
        "cas",
        "canonical_smiles",
        "inchikey",
        "source",
        "source_database",
        "source_url",
        "spectrum_type",
        "phase",
        "intensity_type",
        "sample_type",
        "is_experimental",
        "is_quantitative",
        "download_mode",
        "confidence_level",
        "format",
        "raw_file",
        "processed_file",
        "wavenumber_min",
        "wavenumber_max",
        "n_points_raw",
        "n_points_processed",
        "status",
        "date_downloaded",
        "comment",
        "active",
    ])


def spectra_save_index(index_df):
    """
    Сохраняет индекс спектральной базы.
    """
    os.makedirs(SPECTRA_BANK_DIR, exist_ok=True)
    index_df.to_csv(SPECTRA_INDEX_FILE, index=False)


def spectra_add_to_index(record):
    """
    Добавляет запись в spectra_index.csv.

    Важно:
    функция может вызываться из параллельных потоков IR/Mass,
    поэтому запись защищена lock-ом.
    """
    if record is None or not isinstance(record, dict):
        return

    with SPECTRA_INDEX_LOCK:
        index_df = spectra_load_index()

        index_df = pd.concat(
            [index_df, pd.DataFrame([record])],
            ignore_index=True
        )

        # Если spectrum_id уже существует, оставляем последнюю запись.
        if "spectrum_id" in index_df.columns:
            index_df["spectrum_id"] = index_df["spectrum_id"].astype(str)

            non_empty_id = index_df["spectrum_id"].astype(str).str.strip() != ""

            with_id = index_df.loc[non_empty_id].drop_duplicates(
                subset=["spectrum_id"],
                keep="last"
            )

            without_id = index_df.loc[~non_empty_id]

            index_df = pd.concat(
                [without_id, with_id],
                ignore_index=True
            )

        spectra_save_index(index_df)

# ------------------------------------------------------------------
# Журнал уже проверенных спектров

def spectra_make_sources_key(selected_sources):
    """
    Делает стабильный текстовый ключ из списка источников.
    Храним его для истории, но повторность поиска определяем по:
    InChIKey/canonical_smiles + spectrum_type.
    """
    if selected_sources is None:
        selected_sources = []

    return "|".join([str(x).strip() for x in selected_sources])


def spectra_load_search_cache():
    """
    Загружает журнал уже выполненных поисков спектров.

    Журнал хранит все попытки поиска:
    - found_downloaded;
    - already_in_bank;
    - not_found_in_all_sources;
    - candidate_link_found;
    - download_error / parse_error / search_error;
    - invalid_structure.

    Отрицательный результат тоже важен: он исключает повторные пустые запросы.
    """
    required_cols = [
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "selected_sources_key",
        "selected_sources",
        "final_status",
        "selected_source",
        "candidate_count",
        "spectrum_id",
        "raw_file",
        "processed_file",
        "message",
        "date_checked",
    ]

    spectra_ensure_remote_search_cache()

    if os.path.exists(SPECTRA_SEARCH_CACHE_FILE):
        try:
            cache_df = pd.read_csv(SPECTRA_SEARCH_CACHE_FILE)
        except Exception:
            cache_df = pd.DataFrame(columns=required_cols)
    else:
        cache_df = pd.DataFrame(columns=required_cols)

    for col in required_cols:
        if col not in cache_df.columns:
            cache_df[col] = ""

    # Совместимость со старым журналом: раньше источники могли храниться
    # в selected_sources, а не selected_sources_key.
    old_mask = cache_df["selected_sources_key"].astype(str).str.strip() == ""

    cache_df.loc[old_mask, "selected_sources_key"] = (
        cache_df.loc[old_mask, "selected_sources"]
        .astype(str)
        .str.replace(" | ", "|", regex=False)
    )

    return cache_df[required_cols].copy()


def spectra_save_search_cache(cache_df):
    """
    Сохраняет журнал уже проверенных спектров.
    """
    os.makedirs(SPECTRA_BANK_DIR, exist_ok=True)

    if cache_df is None:
        cache_df = pd.DataFrame()

    with SPECTRA_SEARCH_CACHE_LOCK:
        cache_df.to_csv(
            SPECTRA_SEARCH_CACHE_FILE,
            index=False,
            encoding="utf-8-sig"
        )


def spectra_find_in_search_cache(inchikey, spectrum_type, selected_sources=None, canonical_smiles=""):
    """
    Проверяет журнал уже выполненных поисков.

    Важно:
    - найденный спектр / already_in_bank можно использовать независимо от источников;
    - отрицательный результат можно использовать только если набор источников совпадает;
    - если пользователь добавил новый источник, например MoNA, старый not_found
      не должен блокировать новый онлайн-поиск.
    """
    cache_df = spectra_load_search_cache()

    if cache_df.empty:
        return None

    work = cache_df.copy()

    for col in [
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "selected_sources_key",
        "selected_sources",
        "final_status",
        "selected_source",
        "candidate_count",
        "spectrum_id",
        "raw_file",
        "processed_file",
        "message",
        "date_checked",
    ]:
        if col not in work.columns:
            work[col] = ""

    inchikey = str(inchikey).strip()
    canonical_smiles = str(canonical_smiles).strip()
    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

    selected_sources = selected_sources or []
    current_sources_key = spectra_make_sources_key(selected_sources)

    work["inchikey"] = work["inchikey"].astype(str).str.strip()
    work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()
    work["selected_sources_key"] = work["selected_sources_key"].astype(str).str.strip()

    old_mask = work["selected_sources_key"].astype(str).str.strip() == ""

    if "selected_sources" in work.columns:
        work.loc[old_mask, "selected_sources_key"] = (
            work.loc[old_mask, "selected_sources"]
            .astype(str)
            .str.replace(" | ", "|", regex=False)
        )

    work["_spectrum_type_norm"] = work["spectrum_type"].astype(str).apply(
        spectra_normalize_spectrum_type
    )

    found = pd.DataFrame()

    if inchikey:
        found = work[
            (work["inchikey"] == inchikey)
            & (work["_spectrum_type_norm"] == spectrum_type_norm)
        ].copy()

    if found.empty and canonical_smiles:
        found = work[
            (work["canonical_smiles"] == canonical_smiles)
            & (work["_spectrum_type_norm"] == spectrum_type_norm)
        ].copy()

    if found.empty:
        return None

    found = found.sort_values("date_checked").reset_index(drop=True)

    positive_statuses = {
        "found_downloaded",
        "already_in_bank",
    }

    soft_statuses = {
        "candidate_link_found",
        "candidate_found",
    }

    negative_statuses = {
        "not_found_in_all_sources",
        "not_found",
        "invalid_structure",
    }

    error_statuses = {
        "search_error",
        "download_error",
        "parse_error",
        "no_numeric_spectrum",
        "search_timeout",
    }

    # 1. Положительный результат можно использовать сразу.
    positive = found[
        found["final_status"].astype(str).str.strip().str.lower().isin(positive_statuses)
    ].copy()

    if not positive.empty:
        return positive.iloc[-1].to_dict()

    # 2. Если раньше были кандидаты, но не удалось скачать/распарсить,
    # не называем это not_found и не блокируем новый поиск.
    candidate_or_error = found[
        found["final_status"].astype(str).str.strip().str.lower().isin(
            soft_statuses | error_statuses
        )
    ].copy()

    if not candidate_or_error.empty:
        return None

    # 3. Отрицательный результат используем только для того же набора источников.
    negative = found[
        found["final_status"].astype(str).str.strip().str.lower().isin(negative_statuses)
    ].copy()

    if negative.empty:
        return None

    if current_sources_key:
        negative_same_sources = negative[
            negative["selected_sources_key"].astype(str).str.strip() == current_sources_key
        ].copy()

        if negative_same_sources.empty:
            return None

        return negative_same_sources.iloc[-1].to_dict()

    return None


def spectra_add_to_search_cache(result_row, selected_sources=None):
    """
    Добавляет результат проверки спектра в журнал.

    Сохраняем любой статус:
    - found_downloaded;
    - already_in_bank;
    - not_found_in_all_sources;
    - candidate_link_found;
    - download_error;
    - parse_error;
    - invalid_structure;
    - no_numeric_spectrum;
    - search_timeout;
    - search_error.

    Логика дедупликации:
    одна актуальная запись на InChIKey/canonical_smiles + spectrum_type.
    Источники сохраняются справочно, но не мешают исключать повторный поиск.
    """
    if result_row is None or not isinstance(result_row, dict):
        return

    inchikey = str(result_row.get("inchikey", "")).strip()
    canonical_smiles = str(result_row.get("canonical_smiles", "")).strip()
    spectrum_type_norm = spectra_normalize_spectrum_type(
        result_row.get("spectrum_type", "")
    )

    if not inchikey and not canonical_smiles:
        return

    selected_sources = selected_sources or []
    selected_sources_key = spectra_make_sources_key(selected_sources)

    with SPECTRA_SEARCH_CACHE_LOCK:
        cache_df = spectra_load_search_cache()

        required_cols = [
            "inchikey",
            "canonical_smiles",
            "spectrum_type",
            "selected_sources_key",
            "selected_sources",
            "final_status",
            "selected_source",
            "candidate_count",
            "spectrum_id",
            "raw_file",
            "processed_file",
            "message",
            "date_checked",
        ]

        for col in required_cols:
            if col not in cache_df.columns:
                cache_df[col] = ""

        final_status = (
            result_row.get("spectrum_status", "")
            or result_row.get("final_status", "")
            or result_row.get("status", "")
        )

        new_record = {
            "inchikey": inchikey,
            "canonical_smiles": canonical_smiles,
            "spectrum_type": spectrum_type_norm,
            "selected_sources_key": selected_sources_key,
            "selected_sources": " | ".join(selected_sources),
            "final_status": final_status,
            "selected_source": result_row.get("selected_source", ""),
            "candidate_count": result_row.get("candidate_count", 0),
            "spectrum_id": result_row.get("spectrum_id", ""),
            "raw_file": result_row.get("raw_file", ""),
            "processed_file": result_row.get("processed_file", ""),
            "message": result_row.get("message", ""),
            "date_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        work = cache_df.copy()

        work["inchikey"] = work["inchikey"].astype(str).str.strip()
        work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()
        work["_spectrum_type_norm"] = work["spectrum_type"].astype(str).apply(
            spectra_normalize_spectrum_type
        )

        remove_mask = pd.Series(False, index=work.index)

        if inchikey:
            remove_mask = remove_mask | (
                (work["inchikey"] == inchikey)
                & (work["_spectrum_type_norm"] == spectrum_type_norm)
            )

        if canonical_smiles:
            remove_mask = remove_mask | (
                (work["canonical_smiles"] == canonical_smiles)
                & (work["_spectrum_type_norm"] == spectrum_type_norm)
            )

        work = work.loc[~remove_mask].copy()
        work = work.drop(columns=["_spectrum_type_norm"], errors="ignore")

        work = pd.concat(
            [work, pd.DataFrame([new_record])],
            ignore_index=True
        )

        work[required_cols].copy().to_csv(
            SPECTRA_SEARCH_CACHE_FILE,
            index=False,
            encoding="utf-8-sig"
        )


def spectra_clear_search_cache_by_type(spectrum_type=None, selected_sources=None):
    """
    Очищает журнал уже проверенных спектров.

    spectrum_type:
    - None: очистить весь журнал;
    - "IR": очистить только ИК;
    - "Mass": очистить только масс-спектры.

    selected_sources оставлен для совместимости интерфейса.
    Если selected_sources=None, очищается весь выбранный тип.
    """
    cache_df = spectra_load_search_cache()

    if cache_df is None or cache_df.empty:
        return {
            "removed": 0,
            "left": 0,
            "message": "Журнал уже пуст."
        }

    before_n = len(cache_df)

    if spectrum_type is None:
        spectra_save_search_cache(pd.DataFrame(columns=cache_df.columns))
        return {
            "removed": before_n,
            "left": 0,
            "message": "Журнал поиска спектров полностью очищен."
        }

    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

    work = cache_df.copy()

    if "spectrum_type" not in work.columns:
        work["spectrum_type"] = ""

    work["_spectrum_type_norm"] = work["spectrum_type"].astype(str).apply(
        spectra_normalize_spectrum_type
    )

    remove_mask = work["_spectrum_type_norm"] == spectrum_type_norm

    if selected_sources is not None:
        selected_sources_key = spectra_make_sources_key(selected_sources)

        if "selected_sources_key" not in work.columns:
            work["selected_sources_key"] = ""

        if "selected_sources" in work.columns:
            old_mask = work["selected_sources_key"].astype(str).str.strip() == ""
            work.loc[old_mask, "selected_sources_key"] = (
                work.loc[old_mask, "selected_sources"]
                .astype(str)
                .str.replace(" | ", "|", regex=False)
            )

        remove_mask = remove_mask & (
            work["selected_sources_key"].astype(str).str.strip()
            == selected_sources_key
        )

    new_cache_df = cache_df.loc[~remove_mask].copy()
    new_cache_df = new_cache_df.drop(
        columns=["_spectrum_type_norm"],
        errors="ignore"
    )

    removed_n = before_n - len(new_cache_df)

    spectra_save_search_cache(new_cache_df)

    return {
        "removed": int(removed_n),
        "left": int(len(new_cache_df)),
        "message": (
            f"Удалено записей журнала для типа {spectrum_type_norm}: {removed_n}. "
            f"Осталось записей: {len(new_cache_df)}."
        )
    }


def spectra_find_in_bank(inchikey, canonical_smiles="", spectrum_type=None):
    """
    Проверяет, есть ли активный спектр в локальной базе.

    spectrum_type:
    - None: любой спектр;
    - "IR": только ИК;
    - "Mass": только масс-спектры.
    """
    index_df = spectra_load_index()

    if index_df is None or index_df.empty:
        return None

    work = index_df.copy()

    for col in [
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "active",
        "processed_file",
        "raw_file",
        "source",
        "source_database",
        "spectrum_id",
    ]:
        if col not in work.columns:
            work[col] = ""

    work["inchikey"] = work["inchikey"].astype(str).str.strip()
    work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()
    work["spectrum_type"] = work["spectrum_type"].astype(str).str.strip()
    work["active"] = work["active"].astype(str).str.strip()

    work["_spectrum_type_norm"] = work["spectrum_type"].apply(
        spectra_normalize_spectrum_type
    )

    if spectrum_type is not None:
        requested_type = spectra_normalize_spectrum_type(spectrum_type)

        work = work[
            work["_spectrum_type_norm"] == requested_type
        ].copy()

    if work.empty:
        return None

    active_values = ["true", "1", "yes", "y", "да", "active", ""]

    work["_active_norm"] = (
        work["active"]
        .astype(str)
        .str.lower()
        .isin(active_values)
    )

    work = work[work["_active_norm"]].copy()

    if work.empty:
        return None

    inchikey = str(inchikey).strip()
    canonical_smiles = str(canonical_smiles).strip()

    if inchikey:
        found = work[
            work["inchikey"] == inchikey
        ].copy()

        if not found.empty:
            return found.iloc[0].to_dict()

    if canonical_smiles:
        found = work[
            work["canonical_smiles"] == canonical_smiles
        ].copy()

        if not found.empty:
            return found.iloc[0].to_dict()

    return None

# ------------------------------------------------------------------
# Таблицы не найденных / неуспешных поисков по источникам

def spectra_get_not_found_file(spectrum_type):
    """
    Возвращает CSV-файл таблицы неудачных поисков для IR или Mass.
    """
    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

    if spectrum_type_norm == "Mass":
        return SPECTRA_MASS_NOT_FOUND_FILE

    return SPECTRA_IR_NOT_FOUND_FILE


def spectra_get_source_columns_for_type(spectrum_type):
    """
    Возвращает колонки-источники для таблицы not-found.

    В таблице одна строка = одна структура + один тип спектра.
    Каждый источник имеет свою колонку:
    - not_found
    - search_error
    - candidate_link_found
    - download_error
    - parse_error
    - no_numeric_spectrum
    - empty, если этот источник не проверялся
    """
    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

    source_cols = []

    try:
        for source_key, source_info in SPECTRA_SOURCE_REGISTRY.items():
            source_type = spectra_normalize_spectrum_type(
                source_info.get("spectrum_type", "")
            )

            if source_type == spectrum_type_norm:
                source_cols.append(str(source_key).strip())
    except Exception:
        source_cols = []

    # Запасной вариант, если реестр ещё не определён в момент вызова.
    if not source_cols:
        if spectrum_type_norm == "Mass":
            source_cols = [
                "mona_mass",
            ]
        else:
            source_cols = [
                "nist_webbook",                
            ]

    return source_cols


def spectra_not_found_base_columns():
    """
    Общие колонки таблиц IR/Mass not-found.
    """
    return [
        "date_checked",
        "last_update",
        "spectrum_type",
        "compound_id",
        "name",
        "cas",
        "input_smiles",
        "canonical_smiles",
        "inchikey",
        "structure_status",
        "checked_sources",
        "last_source",
        "last_status",
        "last_source_url",
        "last_search_url",
        "last_message",
        "last_error",
        "total_not_found",
        "total_errors",
        "total_candidate_links",
    ]


def spectra_not_found_columns(spectrum_type):
    """
    Полная структура таблицы:
    базовые колонки + колонки по источникам + служебные колонки по источникам.
    """
    source_cols = spectra_get_source_columns_for_type(spectrum_type)

    status_cols = source_cols
    url_cols = [f"{src}_url" for src in source_cols]
    message_cols = [f"{src}_message" for src in source_cols]
    date_cols = [f"{src}_date" for src in source_cols]

    return (
        spectra_not_found_base_columns()
        + status_cols
        + url_cols
        + message_cols
        + date_cols
    )


def spectra_load_not_found_table(spectrum_type):
    """
    Загружает wide-таблицу неудачных поисков для IR или Mass.

    Одна строка = одна структура + один тип спектра.
    Источники — отдельные колонки.
    """
    filepath = spectra_get_not_found_file(spectrum_type)
    required_cols = spectra_not_found_columns(spectrum_type)

    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
        except Exception:
            df = pd.DataFrame(columns=required_cols)
    else:
        df = pd.DataFrame(columns=required_cols)

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    return df[required_cols].copy()


def spectra_save_not_found_table(spectrum_type, df):
    """
    Сохраняет wide-таблицу неудачных поисков.
    """
    filepath = spectra_get_not_found_file(spectrum_type)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    required_cols = spectra_not_found_columns(spectrum_type)

    if df is None:
        df = pd.DataFrame(columns=required_cols)

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df[required_cols].to_csv(
        filepath,
        index=False,
        encoding="utf-8-sig"
    )


def spectra_recalculate_not_found_summary(row, source_cols):
    """
    Пересчитывает сводные счётчики по колонкам источников.
    """
    total_not_found = 0
    total_errors = 0
    total_candidate_links = 0
    checked_sources = []

    for src in source_cols:
        value = str(row.get(src, "")).strip()

        if value:
            checked_sources.append(src)

        if value == "not_found":
            total_not_found += 1

        if value in [
            "search_error",
            "download_error",
            "parse_error",
            "no_numeric_spectrum",
        ]:
            total_errors += 1

        if value == "candidate_link_found":
            total_candidate_links += 1

    row["checked_sources"] = " | ".join(checked_sources)
    row["total_not_found"] = total_not_found
    row["total_errors"] = total_errors
    row["total_candidate_links"] = total_candidate_links

    return row


def spectra_add_to_not_found_table(
    compound,
    spectrum_type,
    source_key,
    source_label="",
    status="not_found",
    candidate_count=0,
    source_url="",
    search_url="",
    message="",
    error="",
):
    """
    Обновляет wide-таблицу not-found.

    Одна строка = InChIKey/canonical_smiles + spectrum_type.
    Конкретный источник записывается в свою колонку:
    nist_webbook = not_found
    и т.д.

    Итоговый статус not_found_in_all_sources сюда НЕ пишется как общий статус.
    Он остаётся только в spectra_search_cache.csv.
    """
    if compound is None:
        compound = {}

    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

    inchikey = str(compound.get("inchikey", "")).strip()
    canonical_smiles = str(compound.get("canonical_smiles", "")).strip()

    if not inchikey and not canonical_smiles:
        return

    source_key = str(source_key).strip()

    if not source_key:
        source_key = "unknown_source"

    source_label = str(source_label).strip() or source_key
    status = str(status).strip() or "not_found"

    allowed_statuses = [
        "not_found",
        "search_error",
        "candidate_link_found",
        "download_error",
        "parse_error",
        "no_numeric_spectrum",
        "not_found_after_candidate",
        "source_not_connected",
        "search_function_missing",
    ]

    if status not in allowed_statuses:
        status = "not_found"

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_cols = spectra_get_source_columns_for_type(spectrum_type_norm)

    if source_key not in source_cols:
        source_cols.append(source_key)

    with SPECTRA_NOT_FOUND_LOCK:
        df = spectra_load_not_found_table(spectrum_type_norm)

        required_cols = spectra_not_found_columns(spectrum_type_norm)

        # Если source_key динамически добавился, нужно добавить его колонки.
        for dynamic_col in [
            source_key,
            f"{source_key}_url",
            f"{source_key}_message",
            f"{source_key}_date",
        ]:
            if dynamic_col not in required_cols:
                required_cols.append(dynamic_col)

        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        work = df.copy()

        # ВАЖНО:
        # CSV с пустыми колонками pandas иногда читает как float64.
        # Потом запись строк/URL в такие колонки даёт FutureWarning.
        # Поэтому все текстовые служебные колонки заранее приводим к object/string.
        text_cols = [
            "date_checked",
            "last_update",
            "spectrum_type",
            "compound_id",
            "name",
            "cas",
            "input_smiles",
            "canonical_smiles",
            "inchikey",
            "structure_status",
            "checked_sources",
            "last_source",
            "last_status",
            "last_source_url",
            "last_search_url",
            "last_message",
            "last_error",
            source_key,
            f"{source_key}_url",
            f"{source_key}_message",
            f"{source_key}_date",
        ]

        for col in text_cols:
            if col not in work.columns:
                work[col] = ""
            work[col] = work[col].astype("object").where(work[col].notna(), "")

        numeric_cols = [
            "total_not_found",
            "total_errors",
            "total_candidate_links",
        ]

        for col in numeric_cols:
            if col not in work.columns:
                work[col] = 0
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

        work["inchikey"] = work["inchikey"].astype(str).str.strip()
        work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()
        work["spectrum_type"] = work["spectrum_type"].astype(str).apply(
            spectra_normalize_spectrum_type
        )

        match_mask = pd.Series(False, index=work.index)

        if inchikey:
            match_mask = match_mask | (
                (work["inchikey"] == inchikey)
                & (work["spectrum_type"] == spectrum_type_norm)
            )

        if canonical_smiles:
            match_mask = match_mask | (
                (work["canonical_smiles"] == canonical_smiles)
                & (work["spectrum_type"] == spectrum_type_norm)
            )

        if match_mask.any():
            idx = work.index[match_mask][0]
        else:
            new_row = {col: "" for col in required_cols}
            new_row.update({
                "date_checked": now_text,
                "last_update": now_text,
                "spectrum_type": spectrum_type_norm,
                "compound_id": compound.get("compound_id", ""),
                "name": compound.get("name", ""),
                "cas": compound.get("cas", ""),
                "input_smiles": compound.get("input_smiles", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "structure_status": compound.get("structure_status", ""),
                "checked_sources": "",
                "last_source": "",
                "last_status": "",
                "last_source_url": "",
                "last_search_url": "",
                "last_message": "",
                "last_error": "",
                "total_not_found": 0,
                "total_errors": 0,
                "total_candidate_links": 0,
            })

            work = pd.concat(
                [work, pd.DataFrame([new_row])],
                ignore_index=True
            )
            idx = work.index[-1]

        work.loc[idx, "last_update"] = now_text
        work.loc[idx, "last_source"] = source_key
        work.loc[idx, "last_status"] = status
        work.loc[idx, "last_source_url"] = source_url
        work.loc[idx, "last_search_url"] = search_url
        work.loc[idx, "last_message"] = message
        work.loc[idx, "last_error"] = error

        work.loc[idx, source_key] = status
        work.loc[idx, f"{source_key}_url"] = source_url or search_url
        work.loc[idx, f"{source_key}_message"] = message or error
        work.loc[idx, f"{source_key}_date"] = now_text

        row_dict = work.loc[idx].to_dict()
        row_dict = spectra_recalculate_not_found_summary(
            row_dict,
            source_cols
        )

        for key, value in row_dict.items():
            if key not in work.columns:
                work[key] = ""
            work.loc[idx, key] = value

        # Сохраняем с актуальным набором колонок.
        for col in required_cols:
            if col in [
                "total_not_found",
                "total_errors",
                "total_candidate_links",
            ]:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)
            else:
                work[col] = work[col].astype("object").where(work[col].notna(), "")

        work[required_cols].to_csv(
            spectra_get_not_found_file(spectrum_type_norm),
            index=False,
            encoding="utf-8-sig"
        )


def spectra_clear_not_found_table(spectrum_type):
    """
    Очищает таблицу неудачных поисков для IR или Mass.
    """
    spectra_save_not_found_table(
        spectrum_type,
        pd.DataFrame(columns=spectra_not_found_columns(spectrum_type))
    )
    
# ------------------------------------------------------------------
# HTTP / NIST

def spectra_urlopen_text(url, timeout=20):
    """
    Аккуратное чтение текста по URL.

    Используем requests вместо urllib, потому что urllib на Windows
    иногда падает с SSL:
    [ASN1: NOT_ENOUGH_DATA] not enough data (_ssl.c:4040)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 QSPR-Forge-SpectraSearch/0.5",
        "Accept": "text/html,application/xhtml+xml,application/xml,text/plain,*/*",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

    last_error = None

    for verify_ssl in [True, False]:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                verify=verify_ssl,
            )

            response.raise_for_status()

            if response.encoding:
                return response.text

            return response.content.decode("utf-8", errors="replace")

        except requests.exceptions.SSLError as e:
            last_error = e

            if verify_ssl:
                continue

            raise

        except Exception:
            raise

    if last_error is not None:
        raise last_error

    return ""

def spectra_download_binary(url, filepath, timeout=30):
    """
    Скачивает файл по URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 QSPR-Forge-SpectraSearch/0.5",
        "Accept": "chemical/x-jcamp-dx,text/plain,application/octet-stream,*/*",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    last_error = None

    for verify_ssl in [True, False]:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                verify=verify_ssl,
            )

            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            return

        except requests.exceptions.SSLError as e:
            last_error = e

            if verify_ssl:
                continue

            raise

        except Exception:
            raise

    if last_error is not None:
        raise last_error

def spectra_is_jcamp_text(text):
    """
    Проверяет, похож ли текст на JCAMP-DX.
    """
    if not isinstance(text, str):
        return False

    t = text[:3000].upper()

    return (
        "##JCAMP-DX" in t or
        "##DATA TYPE" in t or
        "##XYDATA" in t or
        "##PEAK TABLE" in t
    )


def spectra_extract_phase_from_jcamp_text(raw_text):
    """
    Извлекает агрегатное состояние / фазу из JCAMP-DX.

    Ищет:
    ##STATE=gas
    ##PHASE=liquid
    """
    if not isinstance(raw_text, str):
        return "unknown"

    phase = "unknown"

    for line in raw_text.splitlines():
        s = line.strip()

        if not s.startswith("##"):
            continue

        upper = s.upper()

        if upper.startswith("##STATE="):
            value = s.split("=", 1)[1].strip()
            if value:
                phase = value
                break

        if upper.startswith("##PHASE="):
            value = s.split("=", 1)[1].strip()
            if value:
                phase = value
                break

    phase = str(phase).strip().lower()

    phase_map = {
        "gas": "gas",
        "vapor": "gas",
        "vapour": "gas",
        "liquid": "liquid",
        "solid": "solid",
        "solution": "solution",
        "film": "film",
        "nujol": "nujol",
        "kbr": "kbr",
    }

    return phase_map.get(phase, spectra_safe_filename_part(phase).lower())


def spectra_extract_nist_jcamp_links(html_text):
    """
    Ищет прямые численные ссылки на ИК-спектры NIST.

    Поддерживает:
    - прямые .jdx / .dx / .jcamp ссылки;
    - готовые ссылки с JCAMP=...;
    - ссылки на страницу Type=IR-SPEC, которые превращаем в JCAMP;
    - download-ссылки, если NIST отдаёт их прямо на странице.
    """
    if not html_text:
        return []

    links = []
    base_url = "https://webbook.nist.gov/cgi/"

    hrefs = re.findall(
        r'href=[\'"]([^\'"]+)[\'"]',
        html_text,
        flags=re.IGNORECASE
    )

    for href in hrefs:
        href = href.replace("&amp;", "&").strip()

        if not href:
            continue

        if href.startswith("http"):
            url = href
        else:
            url = urllib.parse.urljoin(base_url, href)

        url_low = url.lower()

        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)

        # 1. Прямая ссылка на файл.
        if (
            url_low.endswith(".jdx")
            or url_low.endswith(".dx")
            or url_low.endswith(".jcamp")
        ):
            if url not in links:
                links.append(url)
            continue

        # 2. Уже готовая JCAMP-ссылка.
        if "JCAMP" in qs or "jcamp" in qs:
            type_value = (
                qs.get("Type", [""])[0]
                or qs.get("type", [""])[0]
            )

            # Для ИК обычно Type=IR, но иногда тип может быть не указан.
            if (
                not type_value
                or str(type_value).upper().startswith("IR")
                or "ir" in url_low
            ):
                if url not in links:
                    links.append(url)

            continue

        # 3. Страница ИК-спектра Type=IR-SPEC.
        type_value = qs.get("Type", qs.get("type", [""]))[0]
        type_upper = str(type_value).upper()

        if type_upper == "IR-SPEC" or "IR-SPEC" in url.upper():
            nist_id = qs.get("ID", qs.get("id", [""]))[0]
            index = qs.get("Index", qs.get("index", ["0"]))[0]

            if nist_id:
                jcamp_url = (
                    "https://webbook.nist.gov/cgi/cbook.cgi?"
                    + urllib.parse.urlencode({
                        "JCAMP": nist_id,
                        "Index": index,
                        "Type": "IR",
                    })
                )

                if jcamp_url not in links:
                    links.append(jcamp_url)

            continue

        # 4. Прямой download endpoint, если на странице он уже есть.
        if (
            "download" in url_low
            and (
                "ir" in url_low
                or "jcamp" in url_low
                or "spectrum" in url_low
            )
        ):
            if url not in links:
                links.append(url)

    # 5. Fallback: ссылки могли быть не в href, а просто текстом.
    direct_patterns = [
        r'(https?://[^\s"\']+\.(?:jdx|dx|jcamp))',
        r'(cbook\.cgi\?[^"\'>\s]*JCAMP=[^"\'>\s]*)',
        r'(cbook\.cgi\?[^"\'>\s]*Type=IR-SPEC[^"\'>\s]*)',
    ]

    for pattern in direct_patterns:
        for raw in re.findall(pattern, html_text, flags=re.IGNORECASE):
            raw = raw.replace("&amp;", "&").strip()

            if raw.startswith("http"):
                url = raw
            else:
                url = urllib.parse.urljoin(base_url, raw)

            parsed = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed.query)

            if "JCAMP" in qs or "jcamp" in qs:
                if url not in links:
                    links.append(url)
                continue

            type_value = qs.get("Type", qs.get("type", [""]))[0]

            if str(type_value).upper() == "IR-SPEC":
                nist_id = qs.get("ID", qs.get("id", [""]))[0]
                index = qs.get("Index", qs.get("index", ["0"]))[0]

                if nist_id:
                    jcamp_url = (
                        "https://webbook.nist.gov/cgi/cbook.cgi?"
                        + urllib.parse.urlencode({
                            "JCAMP": nist_id,
                            "Index": index,
                            "Type": "IR",
                        })
                    )

                    if jcamp_url not in links:
                        links.append(jcamp_url)
                continue

            if url.lower().endswith((".jdx", ".dx", ".jcamp")):
                if url not in links:
                    links.append(url)

    return links


def spectra_nist_search_urls_for_compound(compound):
    """
    Формирует список поисковых URL для NIST.

    Главная логика:
    SMILES -> RDKit -> InChIKey -> NIST

    CAS/name не используем для основного поиска, чтобы не ловить
    неправильные совпадения из текстовых полей.
    """
    urls = []

    base = "https://webbook.nist.gov/cgi/cbook.cgi"

    inchikey = str(compound.get("inchikey", "")).strip()

    if inchikey and inchikey.lower() not in ["nan", "none"]:
        params = {
            "InChI": inchikey,
            "Units": "SI",
            "Mask": "80",
        }

        urls.append(base + "?" + urllib.parse.urlencode(params))

    return urls

def spectra_clean_candidate_list(candidates):
    """
    Удаляет дубликаты кандидатов.
    Диагностические записи тоже сохраняет, если у них есть source_url,
    candidate_url, search_url, accession, error или message.
    """
    cleaned = []
    seen = set()

    for c in candidates or []:
        if not isinstance(c, dict):
            continue

        url = (
            str(c.get("source_url", "")).strip()
            or str(c.get("candidate_url", "")).strip()
            or str(c.get("search_url", "")).strip()
            or str(c.get("accession", "")).strip()
            or str(c.get("error", "")).strip()
            or str(c.get("message", "")).strip()
        )

        key = (
            url,
            str(c.get("source", "")),
            str(c.get("source_database", "")),
            str(c.get("spectrum_type", "")),
            str(c.get("format", "")),
            str(c.get("confidence_level", "")),
            str(c.get("download_mode", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(c)

    return cleaned

def spectra_search_nist_ir_candidates(compound):
    """
    Ищет кандидаты ИК JCAMP в NIST.

    Схема NIST:
    1) Страница вещества по InChIKey.
    2) Ссылка IR Spectrum / Mask=80.
    3) Конкретные страницы Type=IR-SPEC с Index.
    4) Download JCAMP-DX:
       cbook.cgi?Index=...&JCAMP=...&Type=IR
    """
    candidates = []
    search_urls = spectra_nist_search_urls_for_compound(compound)

    for search_url in search_urls:
        try:
            html = spectra_urlopen_text(search_url, timeout=20)

            # 1. Сначала пробуем вытащить прямые JCAMP / IR-SPEC ссылки
            links = spectra_extract_nist_jcamp_links(html)

            # 2. Если на первой странице прямых ссылок нет,
            # ищем ссылку на IR Spectrum / Mask=80 и открываем её.
            if not links:
                ir_page_urls = []

                hrefs = re.findall(
                    r'href=[\'"]([^\'"]+)[\'"]',
                    html,
                    flags=re.IGNORECASE
                )

                for href in hrefs:
                    href = href.replace("&amp;", "&").strip()

                    if not href:
                        continue

                    url = urllib.parse.urljoin(
                        "https://webbook.nist.gov/cgi/",
                        href
                    )

                    url_upper = url.upper()

                    if (
                        "MASK=80" in url_upper
                        or "TYPE=IR-SPEC" in url_upper
                        or "#IR" in url_upper
                    ):
                        if url not in ir_page_urls:
                            ir_page_urls.append(url)

                for ir_page_url in ir_page_urls[:5]:
                    try:
                        ir_html = spectra_urlopen_text(ir_page_url, timeout=20)
                        more_links = spectra_extract_nist_jcamp_links(ir_html)

                        for link in more_links:
                            if link not in links:
                                links.append(link)

                    except Exception:
                        pass

            for link in links:
                candidates.append({
                    "source": "NIST Chemistry WebBook",
                    "source_database": "NIST Chemistry WebBook",
                    "source_url": link,
                    "search_url": search_url,
                    "spectrum_type": "IR",
                    "format": "JCAMP-DX",
                    "phase": "",
                    "intensity_type": "",
                    "sample_type": "pure compound",
                    "is_experimental": True,
                    "is_quantitative": False,
                    "download_mode": "auto",
                    "confidence_level": "high",
                    "candidate_score": 100,
                })

            if not links:
                candidates.append({
                    "source": "NIST Chemistry WebBook",
                    "source_database": "NIST Chemistry WebBook",
                    "source_url": search_url,
                    "search_url": search_url,
                    "spectrum_type": "IR",
                    "format": "HTML",
                    "phase": "",
                    "intensity_type": "",
                    "sample_type": "pure compound",
                    "is_experimental": True,
                    "is_quantitative": False,
                    "download_mode": "no_jcamp_link_found",
                    "confidence_level": "not_found",
                    "candidate_score": 0,
                    "error": "NIST page opened, but no IR JCAMP-DX download link was found.",
                })

        except Exception as e:
            candidates.append({
                "source": "NIST Chemistry WebBook",
                "source_database": "NIST Chemistry WebBook",
                "source_url": search_url,
                "search_url": search_url,
                "spectrum_type": "IR",
                "format": "JCAMP-DX",
                "phase": "",
                "intensity_type": "",
                "sample_type": "pure compound",
                "is_experimental": True,
                "is_quantitative": False,
                "download_mode": "search_error",
                "confidence_level": "error",
                "candidate_score": 0,
                "error": str(e),
            })

    return spectra_clean_candidate_list(candidates)
    
# ------------------------------------------------------------------
# MoNA / MassBank of North America Mass spectra
# Программно повторяет ручной путь:
# search -> display full record -> download spectrum JSON/MSP -> save peaks

MONA_BASE_URL = "https://mona.fiehnlab.ucdavis.edu"
MONA_REST_SEARCH_URL = MONA_BASE_URL + "/rest/spectra/search"


def spectra_mona_http_headers(accept="application/json,text/plain,*/*", referer=None):
    """
    Единые HTTP-заголовки для запросов к MoNA.
    Referer нужен для endpoint-ов, которые вызываются со страницы Display Full Record.
    """
    headers = {
        "User-Agent": "QSPR-Forge-MoNA-MassSearch/0.4",
        "Accept": str(accept or "application/json,text/plain,*/*"),
    }

    if referer:
        headers["Referer"] = str(referer)

    return headers


def spectra_urlopen_json_get_params(url, params, timeout=40, referer=None):
    """
    GET JSON с query-параметрами.
    Для MoNA:
    /rest/spectra/search?query=...&size=...&page=...
    """
    headers = spectra_mona_http_headers(
        accept="application/json,text/plain,*/*",
        referer=referer,
    )

    last_error = None

    for verify_ssl in [True, False]:
        try:
            response = requests.get(
                url,
                params=params or {},
                headers=headers,
                timeout=timeout,
                verify=verify_ssl,
            )

            response.raise_for_status()

            text = response.text.strip()

            if not text:
                return {}

            return response.json()

        except requests.exceptions.SSLError as e:
            last_error = e

            if verify_ssl:
                continue

            raise

        except Exception:
            raise

    if last_error is not None:
        raise last_error

    return {}

def spectra_urlopen_text_get(url, timeout=40, referer=None):
    """
    GET text. Нужен для MSP fallback.
    """
    headers = spectra_mona_http_headers(
        accept="text/plain,application/json,*/*",
        referer=referer,
    )

    last_error = None

    for verify_ssl in [True, False]:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                verify=verify_ssl,
            )

            response.raise_for_status()
            return response.text

        except requests.exceptions.SSLError as e:
            last_error = e

            if verify_ssl:
                continue

            raise

        except Exception:
            raise

    if last_error is not None:
        raise last_error

    return ""


def spectra_urlopen_json_get(url, timeout=40, referer=None):
    """
    GET JSON без параметров.

    Для MoNA используем requests, потому что urllib на Windows иногда падает
    с SSL-ошибкой:
    [ASN1: NOT_ENOUGH_DATA] not enough data (_ssl.c:4040)
    """
    headers = spectra_mona_http_headers(
        accept="application/json,text/plain,*/*",
        referer=referer,
    )

    last_error = None

    for verify_ssl in [True, False]:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                verify=verify_ssl,
            )

            response.raise_for_status()

            text = response.text.strip()

            if not text:
                return {}

            return response.json()

        except requests.exceptions.SSLError as e:
            last_error = e

            if verify_ssl:
                continue

            raise

        except Exception:
            raise

    if last_error is not None:
        raise last_error

    return {}

def spectra_mona_fetch_spectrum_record_by_accession(accession):
    """
    GET полной записи MoNA по accession:
    https://mona.fiehnlab.ucdavis.edu/rest/spectra/<accession>
    """
    accession = str(accession).strip()

    if not accession:
        return {}

    display_url = f"{MONA_BASE_URL}/spectra/display/{urllib.parse.quote(accession)}"
    record_url = f"{MONA_BASE_URL}/rest/spectra/{urllib.parse.quote(accession)}"

    try:
        try:
            data = spectra_urlopen_json_get(
                record_url,
                timeout=40,
                referer=display_url,
            )
        except TypeError:
            data = spectra_urlopen_json_get(
                record_url,
                timeout=40,
            )

        if isinstance(data, dict):
            return data

        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]

    except Exception:
        return {}

    return {}


def spectra_mona_flatten_records(response_json):
    """
    Приводит разные ответы MoNA к списку записей.
    """
    if response_json is None:
        return []

    if isinstance(response_json, list):
        return response_json

    if isinstance(response_json, dict):
        for key in ["content", "results", "spectra", "data", "items"]:
            value = response_json.get(key)
            if isinstance(value, list):
                return value

        if "spectrum" in response_json or "compound" in response_json or "id" in response_json:
            return [response_json]

    return []


def spectra_mona_get_nested(obj, path, default=""):
    cur = obj

    for p in path:
        try:
            if isinstance(p, int):
                if isinstance(cur, list) and len(cur) > p:
                    cur = cur[p]
                else:
                    return default
            else:
                if isinstance(cur, dict):
                    cur = cur.get(p, default)
                else:
                    return default
        except Exception:
            return default

    if cur is None:
        return default

    return cur


def spectra_mona_find_metadata_value(record, keys):
    """
    Ищет значение в metaData MoNA.
    """
    keys_norm = [str(k).strip().lower() for k in keys]

    def scan_metadata_list(meta_list):
        if not isinstance(meta_list, list):
            return ""

        for item in meta_list:
            if not isinstance(item, dict):
                continue

            name = str(
                item.get("name", item.get("label", item.get("key", "")))
            ).strip().lower()

            if name in keys_norm:
                value = item.get("value", item.get("text", ""))
                if value is not None:
                    return str(value)

        return ""

    value = scan_metadata_list(record.get("metaData", []))
    if value:
        return value

    compounds = record.get("compound", [])
    if isinstance(compounds, dict):
        compounds = [compounds]

    if isinstance(compounds, list):
        for comp in compounds:
            if isinstance(comp, dict):
                value = scan_metadata_list(comp.get("metaData", []))
                if value:
                    return value

    return ""


def spectra_mona_record_accession(record):
    """
    Достаёт id/accession записи MoNA.
    """
    if not isinstance(record, dict):
        return ""

    for key in ["id", "accession", "spectrumId", "spectrum_id"]:
        value = str(record.get(key, "")).strip()
        if value:
            return value

    value = spectra_mona_find_metadata_value(
        record,
        ["accession", "spectrum id", "id"]
    )

    return str(value).strip()


def spectra_mona_record_inchikey(record):
    """
    Достаёт InChIKey из записи MoNA.
    """
    paths = [
        ["compound", 0, "inchiKey"],
        ["compound", 0, "inchi_key"],
        ["compound", 0, "inchikey"],
        ["compound", "inchiKey"],
        ["compound", "inchi_key"],
        ["compound", "inchikey"],
        ["inchiKey"],
        ["inchi_key"],
        ["inchikey"],
    ]

    for path in paths:
        value = str(spectra_mona_get_nested(record, path, "")).strip()
        if value:
            return value

    value = spectra_mona_find_metadata_value(
        record,
        ["InChIKey", "InChI Key", "inchi key", "inchikey"]
    )

    return str(value).strip()


def spectra_mona_record_name(record):
    paths = [
        ["compound", 0, "names", 0, "name"],
        ["compound", 0, "name"],
        ["compound", "names", 0, "name"],
        ["compound", "name"],
        ["name"],
    ]

    for path in paths:
        value = str(spectra_mona_get_nested(record, path, "")).strip()
        if value:
            return value

    return spectra_mona_find_metadata_value(
        record,
        ["name", "compound name", "common name"]
    )


def spectra_mona_record_ionization(record):
    return spectra_mona_find_metadata_value(
        record,
        ["ionization mode", "ion mode", "polarity"]
    )


def spectra_mona_record_ms_level(record):
    return spectra_mona_find_metadata_value(
        record,
        ["ms level", "ms_type", "spectrum type"]
    )


def spectra_mona_record_instrument(record):
    return spectra_mona_find_metadata_value(
        record,
        ["instrument", "instrument type", "ms type"]
    )


def spectra_mona_build_inchikey_query(inchikey):
    """
    Запрос как при ручном поиске по InChIKey на MoNA.
    """
    inchikey = str(inchikey).strip()

    return (
        "exists(compound.metaData.name:'InChIKey' "
        f"and compound.metaData.value:'{inchikey}')"
    )


def spectra_mona_search_by_inchikey(inchikey, size=10):
    """
    Шаг 1 ручного пути:
    поиск MoNA по InChIKey.
    """
    query = spectra_mona_build_inchikey_query(inchikey)

    response_json = spectra_urlopen_json_get_params(
        MONA_REST_SEARCH_URL,
        params={
            "query": query,
            "size": int(size),
            "page": 0,
        },
        timeout=40
    )

    records = spectra_mona_flatten_records(response_json)

    return records, query


def spectra_mona_fetch_full_record(accession):
    """
    Шаг 2 ручного пути:
    Display Full Record.

    Пробуем несколько REST-адресов, потому что у MoNA встречались разные
    варианты endpoint для полной записи.
    """
    accession = str(accession).strip()

    if not accession:
        return {}

    candidate_urls = [
        f"{MONA_BASE_URL}/rest/spectra/{urllib.parse.quote(accession)}",
        f"{MONA_BASE_URL}/rest/spectra/{urllib.parse.quote(accession)}/download",
        f"{MONA_BASE_URL}/rest/spectra/download/{urllib.parse.quote(accession)}",
    ]

    for url in candidate_urls:
        try:
            data = spectra_urlopen_json_get(url, timeout=40)

            if isinstance(data, dict) and data:
                return data

            if isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    return data[0]

        except Exception:
            pass

    return {}


def spectra_mona_download_record_json(accession):
    """
    Шаг 3 ручного пути:
    Download Spectrum -> JSON.

    Если direct download endpoint не сработает, вернём полную запись.
    """
    accession = str(accession).strip()

    candidate_urls = [
        f"{MONA_BASE_URL}/rest/spectra/{urllib.parse.quote(accession)}?format=json",
        f"{MONA_BASE_URL}/rest/spectra/download/{urllib.parse.quote(accession)}?format=json",
        f"{MONA_BASE_URL}/rest/spectra/{urllib.parse.quote(accession)}/download?format=json",
    ]

    for url in candidate_urls:
        try:
            data = spectra_urlopen_json_get(url, timeout=40)

            if isinstance(data, dict) and data:
                return data, url

            if isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    return data[0], url

        except Exception:
            pass

    full_record = spectra_mona_fetch_full_record(accession)

    if full_record:
        return full_record, "full_record_fallback"

    return {}, ""


def spectra_mona_download_record_msp(accession):
    """
    Альтернативный путь:
    Download Spectrum -> MSP.
    """
    accession = str(accession).strip()

    candidate_urls = [
        f"{MONA_BASE_URL}/rest/spectra/{urllib.parse.quote(accession)}?format=msp",
        f"{MONA_BASE_URL}/rest/spectra/download/{urllib.parse.quote(accession)}?format=msp",
        f"{MONA_BASE_URL}/rest/spectra/{urllib.parse.quote(accession)}/download?format=msp",
    ]

    for url in candidate_urls:
        try:
            text = spectra_urlopen_text_get(url, timeout=40)

            if text and ("Num Peaks" in text or "NumPeaks" in text or ":" in text):
                return text, url

        except Exception:
            pass

    return "", ""


def spectra_mona_parse_peak_string(spectrum_value):
    """
    Парсит peak list MoNA.

    Основной формат MoNA:
    "13:1.16 14:3.35 15:4.57 28:99.99"

    Также поддерживает:
    - "mz intensity" по строкам;
    - list[dict];
    - list[list].
    """
    rows = []

    if spectrum_value is None:
        return pd.DataFrame(columns=["mz", "intensity"])

    if isinstance(spectrum_value, list):
        for item in spectrum_value:
            if isinstance(item, dict):
                mz = item.get("mz", item.get("mass", item.get("m/z", None)))
                intensity = item.get("intensity", item.get("abundance", None))

                try:
                    rows.append((float(mz), float(intensity)))
                except Exception:
                    pass

            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    rows.append((float(item[0]), float(item[1])))
                except Exception:
                    pass

    else:
        text = str(spectrum_value).strip()

        # Главный формат MoNA: mz:intensity
        pairs = re.findall(
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[:;,]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            text
        )

        if pairs:
            for mz, intensity in pairs:
                try:
                    rows.append((float(mz), float(intensity)))
                except Exception:
                    pass
        else:
            # fallback: строки вида "mz intensity"
            lines = text.replace(";", "\n").splitlines()

            for line in lines:
                nums = re.findall(
                    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                    line
                )

                if len(nums) >= 2:
                    try:
                        rows.append((float(nums[0]), float(nums[1])))
                    except Exception:
                        pass

    if not rows:
        return pd.DataFrame(columns=["mz", "intensity"])

    df = pd.DataFrame(rows, columns=["mz", "intensity"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mz", "intensity"])
    df = df.drop_duplicates(subset=["mz"], keep="last")
    df = df.sort_values("mz").reset_index(drop=True)

    return df

def spectra_mona_parse_msp_peaks(msp_text):
    """
    Парсит MSP.
    После строки Num Peaks идут пары mz intensity.
    """
    if not msp_text:
        return pd.DataFrame(columns=["mz", "intensity"])

    lines = msp_text.splitlines()
    start_i = None

    for i, line in enumerate(lines):
        if line.strip().lower().replace(" ", "") in ["numpeaks:", "numpeaks"]:
            start_i = i + 1
            break

        if line.strip().lower().startswith("num peaks"):
            start_i = i + 1
            break

    if start_i is None:
        # fallback: пробуем все строки
        start_i = 0

    rows = []

    for line in lines[start_i:]:
        s = line.strip()

        if not s:
            continue

        nums = re.findall(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
            s
        )

        if len(nums) >= 2:
            try:
                rows.append((float(nums[0]), float(nums[1])))
            except Exception:
                pass

    if not rows:
        return pd.DataFrame(columns=["mz", "intensity"])

    df = pd.DataFrame(rows, columns=["mz", "intensity"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates(subset=["mz"], keep="max")
    df = df.sort_values("mz").reset_index(drop=True)

    return df

def spectra_score_mona_mass_candidate(candidate):
    """
    Оценка MoNA/Mass-кандидата для выбора спектра в spectra_bank.

    Приоритет:
    1. точное совпадение InChIKey;
    2. MS1;
    3. positive;
    4. EI / EI-B;
    5. MassBank/MoNA clean-подобные записи;
    6. больше пиков;
    7. стабильный accession.
    """
    if candidate is None or not isinstance(candidate, dict):
        return (-9999, 0, "")

    score = float(candidate.get("candidate_score", 0) or 0)

    confidence = str(candidate.get("confidence_level", "")).lower().strip()
    ionization_mode = str(candidate.get("ionization_mode", "")).lower().strip()
    ms_level = str(candidate.get("ms_level", "")).lower().strip()
    instrument = str(candidate.get("instrument", "")).lower().strip()
    accession = str(candidate.get("accession", "")).strip()
    source_database = str(candidate.get("source_database", "")).lower().strip()
    source = str(candidate.get("source", "")).lower().strip()
    source_url = str(candidate.get("source_url", "")).lower().strip()

    peak_count = int(candidate.get("_peak_count", 0) or 0)

    # 1. Совпадение структуры.
    if confidence == "exact_inchikey":
        score += 1000
    elif confidence == "same_connectivity_inchikey":
        score += 700
    elif confidence == "query_match_no_record_inchikey":
        score += 300
    elif "mismatch" in confidence or "rejected" in confidence:
        score -= 5000

    # 2. Только обычные MS1 предпочтительнее для универсального банка.
    if ms_level in ["ms1", "ms", ""]:
        score += 300
    elif "ms2" in ms_level or "ms/ms" in ms_level or "msn" in ms_level:
        score -= 500

    # 3. Для EI библиотек обычно positive.
    if ionization_mode in ["positive", "pos", "+", ""]:
        score += 150
    elif ionization_mode in ["negative", "neg", "-"]:
        score -= 150

    # 4. EI / EI-B предпочтительнее для классической масс-библиотеки.
    if "ei-b" in instrument:
        score += 250
    elif "ei" in instrument:
        score += 200
    elif "ci" in instrument:
        score -= 50

    # 5. Предпочтение MassBank/MoNA.
    if "massbank" in source_database or "massbank" in source or "massbank" in source_url:
        score += 100

    if "mona" in source_database or "mona" in source or "mona" in source_url:
        score += 50

    # 6. Запись без пиков нельзя брать.
    if peak_count <= 0:
        score -= 10000

    # 7. Больше пиков — лучше, но это вторичный критерий.
    score += min(peak_count, 200) * 0.5

    return (
        score,
        peak_count,
        accession,
    )

# ------------------------------------------------------------------
# MoNA Mass override: stable search/download logic


def spectra_mona_record_to_peak_df(record):
    """
    Достаёт mz/intensity из записи MoNA.

    Поддерживает:
    - spectrum
    - peaks
    - peakList
    - вложенные варианты этих полей
    """
    if not isinstance(record, dict):
        return pd.DataFrame(columns=["mz", "intensity"])

    direct_keys = [
        "spectrum",
        "peaks",
        "peakList",
        "peak_list",
        "peaklist",
    ]

    for key in direct_keys:
        value = record.get(key)

        if value:
            peak_df = spectra_mona_parse_peak_string(value)

            if peak_df is not None and not peak_df.empty:
                return peak_df

    def _scan(obj, depth=0):
        if depth > 5:
            return pd.DataFrame(columns=["mz", "intensity"])

        if isinstance(obj, dict):
            for key, value in obj.items():
                key_low = str(key).lower().strip()

                if key_low in [
                    "spectrum",
                    "peaks",
                    "peaklist",
                    "peak_list",
                ]:
                    peak_df = spectra_mona_parse_peak_string(value)

                    if peak_df is not None and not peak_df.empty:
                        return peak_df

                if isinstance(value, (dict, list)):
                    peak_df = _scan(value, depth + 1)

                    if peak_df is not None and not peak_df.empty:
                        return peak_df

        elif isinstance(obj, list):
            for item in obj:
                peak_df = _scan(item, depth + 1)

                if peak_df is not None and not peak_df.empty:
                    return peak_df

        return pd.DataFrame(columns=["mz", "intensity"])

    return _scan(record)


def spectra_search_mona_mass_candidates(compound):
    """
    Поиск Mass-спектров в MoNA.

    Старая рабочая схема:
    search по InChIKey -> accession -> /rest/spectra/<accession>
    -> сразу парсим поле spectrum.
    """
    candidates = []

    inchikey_query = str(compound.get("inchikey", "")).strip().upper()

    if not inchikey_query:
        return []

    query_string = (
        "exists(compound.metaData.name:'InChIKey' "
        f"and compound.metaData.value:'{inchikey_query}')"
    )

    search_url = (
        MONA_REST_SEARCH_URL
        + "?"
        + urllib.parse.urlencode({
            "query": query_string,
            "size": 10,
            "page": 0,
        })
    )

    try:
        search_json = spectra_urlopen_json_get(
            search_url,
            timeout=40
        )

        search_records = spectra_mona_flatten_records(search_json)

    except Exception as e:
        return [{
            "source": "MoNA / MassBank of North America",
            "source_database": "MoNA",
            "source_url": MONA_BASE_URL,
            "search_url": search_url,
            "spectrum_type": "Mass",
            "format": "MoNA_JSON",
            "phase": "unknown",
            "intensity_type": "",
            "download_mode": "error",
            "confidence_level": "error",
            "candidate_score": 0,
            "error": f"MoNA search error: {e}",
        }]

    if not search_records:
        return [{
            "source": "MoNA / MassBank of North America",
            "source_database": "MoNA",
            "source_url": MONA_BASE_URL,
            "search_url": search_url,
            "spectrum_type": "Mass",
            "format": "MoNA_JSON",
            "phase": "unknown",
            "intensity_type": "",
            "download_mode": "not_found",
            "confidence_level": "not_found",
            "candidate_score": 0,
            "error": "MoNA search returned 0 records",
        }]

    debug_errors = []

    for record_i, search_record in enumerate(search_records, start=1):
        if not isinstance(search_record, dict):
            continue

        accession = spectra_mona_record_accession(search_record)

        if not accession:
            debug_errors.append(
                f"record {record_i}: accession/id не найден; "
                f"keys={list(search_record.keys())[:20]}"
            )
            continue

        full_record = spectra_mona_fetch_spectrum_record_by_accession(accession)

        if not full_record:
            debug_errors.append(
                f"record {record_i}, accession={accession}: "
                f"/rest/spectra/{accession} не вернул JSON"
            )
            continue

        peak_df = spectra_mona_record_to_peak_df(full_record)

        if peak_df is None or peak_df.empty:
            debug_errors.append(
                f"record {record_i}, accession={accession}: "
                f"поле spectrum не прочитано; keys={list(full_record.keys())[:20]}"
            )
            continue

        rec_inchikey = spectra_mona_record_inchikey(full_record).strip().upper()

        confidence = "candidate"
        score = 50

        if rec_inchikey:
            if rec_inchikey == inchikey_query:
                confidence = "exact_inchikey"
                score = 100
            elif rec_inchikey[:14] == inchikey_query[:14]:
                confidence = "same_connectivity_inchikey"
                score = 80
            else:
                debug_errors.append(
                    f"accession={accession}: InChIKey mismatch: "
                    f"{rec_inchikey} != {inchikey_query}"
                )
                continue
        else:
            confidence = "query_match_no_record_inchikey"
            score = 65

        record_url = f"{MONA_BASE_URL}/rest/spectra/{urllib.parse.quote(accession)}"

        candidates.append({
            "source": "MoNA / MassBank of North America",
            "source_database": "MoNA",
            "source_url": f"{MONA_BASE_URL}/spectra/display/{accession}",
            "search_url": search_url,
            "download_json_url": record_url,
            "spectrum_type": "Mass",
            "format": "MoNA_REST_JSON",
            "phase": "gas",
            "intensity_type": "relative abundance",
            "sample_type": "pure compound",
            "is_experimental": True,
            "is_quantitative": False,
            "download_mode": "mona_rest_spectra_accession",
            "confidence_level": confidence,
            "candidate_score": score,
            "accession": accession,
            "mona_inchikey": rec_inchikey,
            "compound_name": spectra_mona_record_name(full_record),
            "ionization_mode": spectra_mona_record_ionization(full_record),
            "ms_level": spectra_mona_record_ms_level(full_record),
            "instrument": spectra_mona_record_instrument(full_record),
            "_mona_record": full_record,
            "_mona_msp": "",
            "_peak_df": peak_df,
            "_peak_count": len(peak_df),
        })

    if candidates:
        return spectra_clean_candidate_list(candidates)

    return [{
        "source": "MoNA / MassBank of North America",
        "source_database": "MoNA",
        "source_url": MONA_BASE_URL,
        "search_url": search_url,
        "spectrum_type": "Mass",
        "format": "MoNA_REST_JSON",
        "phase": "unknown",
        "intensity_type": "",
        "download_mode": "parse_error",
        "confidence_level": "error",
        "candidate_score": 0,
        "error": (
            "MoNA: записи найдены, но пригодный спектр не получен. "
            + " | ".join(debug_errors[:5])
        ),
    }]

def spectra_download_best_mona_mass_candidate(compound, candidates):
    """
    Сохраняет лучший MoNA Mass-кандидат в spectra_bank/Mass.

    В raw_file сохраняется JSON/MSP payload.
    В processed_file сохраняется mz/intensity как wavenumber/intensity.
    """
    if not candidates:
        return {
            "status": "not_found",
            "record": None,
            "message": "MoNA: масс-спектры не найдены.",
            "candidate_url": "",
        }

    # Сначала проверяем локальный банк.
    existing = spectra_find_in_bank(
        inchikey=compound.get("inchikey", ""),
        canonical_smiles=compound.get("canonical_smiles", ""),
        spectrum_type="Mass"
    )

    if existing is not None:
        return {
            "status": "already_in_bank",
            "record": existing,
            "message": "Mass-спектр уже есть в локальной базе. Повторное скачивание не выполнялось.",
            "candidate_url": existing.get("source_url", ""),
        }

    original_candidates = list(candidates)

    valid_candidates = [
        c for c in original_candidates
        if str(c.get("confidence_level", "")).strip().lower()
        not in [
            "error",
            "rejected_inchikey_mismatch",
        ]
        and int(c.get("_peak_count", 0) or 0) > 0
    ]

    if not valid_candidates:
        first = original_candidates[0] if original_candidates else {}

        first_error = (
            first.get("error", "")
            or first.get("message", "")
            or "MoNA: записи найдены, но пригодный численный спектр не получен."
        )

        first_download_mode = str(first.get("download_mode", "")).strip().lower()
        first_confidence = str(first.get("confidence_level", "")).strip().lower()

        candidate_url = (
            first.get("source_url", "")
            or first.get("candidate_url", "")
            or first.get("search_url", "")
        )

        # Если MoNA реально ничего не нашла, это not_found,
        # а не "records found but not parsed".
        if (
            first_download_mode == "not_found"
            or first_confidence == "not_found"
            or "0 records" in str(first_error).lower()
            or "по exact inchikey записей не найдено" in str(first_error).lower()
        ):
            return {
                "status": "not_found",
                "record": None,
                "message": first_error,
                "candidate_url": candidate_url,
            }

        return {
            "status": "mona_records_found_but_not_parsed",
            "record": None,
            "message": first_error,
            "candidate_url": candidate_url,
        }

    candidates = sorted(
        valid_candidates,
        key=spectra_score_mona_mass_candidate,
        reverse=True,
    )

    best = candidates[0]

    record_json = best.get("_mona_record", {})
    peak_df = best.get("_peak_df", pd.DataFrame())

    if peak_df is None or not isinstance(peak_df, pd.DataFrame) or peak_df.empty:
        peak_df = spectra_mona_record_to_peak_df(record_json)

    if peak_df is None or not isinstance(peak_df, pd.DataFrame) or peak_df.empty:
        return {
            "status": "parse_error",
            "record": None,
            "message": (
                "MoNA: запись найдена, но peak list не прочитан. "
                f"accession={best.get('accession', '')}; "
                f"json_url={best.get('download_json_url', '')}; "
                f"msp_url={best.get('download_msp_url', '')}; "
                f"source_url={best.get('source_url', '')}"
            ),
            "candidate_url": best.get("source_url", ""),
        }

    existing = spectra_find_in_bank(
        inchikey=compound.get("inchikey", ""),
        canonical_smiles=compound.get("canonical_smiles", ""),
        spectrum_type="Mass"
    )

    if existing is not None:
        return {
            "status": "already_in_bank",
            "record": existing,
            "message": "Mass-спектр уже есть в локальной базе. Повторное скачивание не выполнялось.",
            "candidate_url": existing.get("source_url", ""),
        }

    spectrum_id, raw_file, processed_file = spectra_make_flat_spectrum_paths(
        compound=compound,
        candidate=best
    )

    try:
        os.makedirs(os.path.dirname(raw_file), exist_ok=True)
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)

        raw_payload = {
            "mona_record": record_json,
            "mona_msp": best.get("_mona_msp", ""),
            "source_url": best.get("source_url", ""),
            "search_url": best.get("search_url", ""),
            "download_json_url": best.get("download_json_url", ""),
            "download_msp_url": best.get("download_msp_url", ""),
        }

        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(raw_payload, f, ensure_ascii=False, indent=2)

        processed_df = peak_df.rename(columns={"mz": "wavenumber"}).copy()

        processed_df["wavenumber"] = pd.to_numeric(
            processed_df["wavenumber"],
            errors="coerce"
        )

        processed_df["intensity"] = pd.to_numeric(
            processed_df["intensity"],
            errors="coerce"
        )

        processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
        processed_df = processed_df.dropna(subset=["wavenumber", "intensity"])
        processed_df = processed_df.sort_values("wavenumber").reset_index(drop=True)

        if processed_df.empty:
            return {
                "status": "parse_error",
                "record": None,
                "message": "MoNA: после очистки peak table не осталось точек.",
                "candidate_url": best.get("source_url", ""),
            }

        spectra_save_processed_spectrum(
            spectrum_df=processed_df,
            processed_file=processed_file
        )

        record = {
            "spectrum_id": spectrum_id,
            "compound_id": compound.get("compound_id", ""),
            "name": compound.get("name", "") or best.get("compound_name", ""),
            "cas": compound.get("cas", ""),
            "canonical_smiles": compound.get("canonical_smiles", ""),
            "inchikey": compound.get("inchikey", "") or best.get("mona_inchikey", ""),
            "source": "MoNA / MassBank of North America",
            "source_database": "MoNA",
            "source_url": best.get("source_url", ""),
            "spectrum_type": "Mass",
            "phase": best.get("phase", "gas"),
            "intensity_type": best.get("intensity_type", "relative abundance"),
            "sample_type": best.get("sample_type", "pure compound"),
            "is_experimental": best.get("is_experimental", True),
            "is_quantitative": best.get("is_quantitative", False),
            "download_mode": best.get("download_mode", "mona_rest_json_or_msp"),
            "confidence_level": best.get("confidence_level", ""),
            "format": best.get("format", "MoNA_REST_JSON_OR_MSP"),
            "raw_file": raw_file,
            "processed_file": processed_file,
            "wavenumber_min": float(processed_df["wavenumber"].min()),
            "wavenumber_max": float(processed_df["wavenumber"].max()),
            "n_points_raw": len(processed_df),
            "n_points_processed": len(processed_df),
            "status": "found_downloaded",
            "date_downloaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comment": (
                f"accession={best.get('accession', '')}; "
                f"ionization_mode={best.get('ionization_mode', '')}; "
                f"ms_level={best.get('ms_level', '')}; "
                f"instrument={best.get('instrument', '')}; "
                f"peak_count={best.get('_peak_count', '')}; "
                f"candidate_score={best.get('candidate_score', '')}; "
                f"selection_score={spectra_score_mona_mass_candidate(best)[0]}; "
                f"download_json_url={best.get('download_json_url', '')}; "
                f"download_msp_url={best.get('download_msp_url', '')}; "
                f"confidence={best.get('confidence_level', '')}"
            ),
            "active": True,
        }

        spectra_add_to_index(record)

        return {
            "status": "found_downloaded",
            "record": record,
            "message": (
                f"MoNA: масс-спектр скачан и сохранён. "
                f"Spectrum ID: {spectrum_id}. "
                f"Точек: {len(processed_df)}."
            ),
            "candidate_url": best.get("source_url", ""),
        }

    except Exception as e:
        return {
            "status": "download_error",
            "record": None,
            "message": f"MoNA download error: {e}",
            "candidate_url": best.get("source_url", ""),
        }

SPECTRA_SOURCE_REGISTRY = {
    "nist_webbook": {
        "label": "NIST Chemistry WebBook",
        "spectrum_type": "IR",
        "mode": "download",
        "search_function": spectra_search_nist_ir_candidates,
    },
    "mona_mass": {
        "label": "MoNA / MassBank of North America",
        "spectrum_type": "Mass",
        "mode": "mona_download",
        "search_function": spectra_search_mona_mass_candidates,
    },
}


# ------------------------------------------------------------------
# JCAMP / JDX / DX parsing

def spectra_parse_simple_jcamp_xy(raw_text):
    """
    Минимальный fallback-парсер JCAMP-DX.

    Читает простые числовые XYDATA/PEAK TABLE.
    Сжатые JCAMP-форматы может читать неполно.
    """
    lines = raw_text.splitlines()

    metadata = {}
    inside_data = False
    xy_rows = []

    for line in lines:
        s = line.strip()

        if not s:
            continue

        if s.startswith("##"):
            inside_data = False

            if "=" in s:
                key, value = s[2:].split("=", 1)
                key = key.strip().upper()
                metadata[key] = value.strip()

                if key in ["XYDATA", "PEAK TABLE"]:
                    inside_data = True

            continue

        if "XYDATA" in metadata or "PEAK TABLE" in metadata or inside_data:
            nums = re.findall(
                r"[-+]?\d*\.\d+|[-+]?\d+",
                s
            )

            if len(nums) >= 2:
                vals = [float(x) for x in nums]
                x0 = vals[0]

                if len(vals) == 2:
                    xy_rows.append((vals[0], vals[1]))
                else:
                    for i, y in enumerate(vals[1:]):
                        xy_rows.append((x0 + i, y))

    if not xy_rows:
        return pd.DataFrame(), metadata

    spectrum_df = pd.DataFrame(
        xy_rows,
        columns=["wavenumber", "intensity"]
    )

    spectrum_df = spectrum_df.dropna()
    spectrum_df = spectrum_df.drop_duplicates(subset=["wavenumber"])
    spectrum_df = spectrum_df.sort_values("wavenumber").reset_index(drop=True)

    return spectrum_df, metadata


def spectra_read_jdx_file(filepath):
    """
    Читает JDX/DX.

    Сначала пробует пакет jcamp, если он установлен.
    Если нет — использует простой fallback-парсер.
    """
    with open(filepath, "r", encoding="latin-1", errors="replace") as f:
        raw_text = f.read()

    try:
        import jcamp

        parsed = jcamp.JCAMP_reader(filepath)

        x = parsed.get("x", None)
        y = parsed.get("y", None)

        if x is not None and y is not None:
            spectrum_df = pd.DataFrame({
                "wavenumber": np.array(x, dtype=float),
                "intensity": np.array(y, dtype=float),
            })

            spectrum_df = spectrum_df.dropna()
            spectrum_df = spectrum_df.drop_duplicates(subset=["wavenumber"])
            spectrum_df = spectrum_df.sort_values("wavenumber").reset_index(drop=True)

            metadata = {
                "parser": "jcamp",
                "title": parsed.get("title", ""),
                "data_type": parsed.get("data type", ""),
                "state": spectra_extract_phase_from_jcamp_text(raw_text),
            }

            return spectrum_df, metadata

    except Exception:
        pass

    spectrum_df, metadata = spectra_parse_simple_jcamp_xy(raw_text)
    metadata["parser"] = "fallback_simple"
    metadata["state"] = spectra_extract_phase_from_jcamp_text(raw_text)

    return spectrum_df, metadata


def spectra_save_processed_spectrum(spectrum_df, processed_file):
    """
    Сохраняет обработанный спектр.
    """
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    spectrum_df.to_csv(processed_file, index=False)

    return processed_file

# ------------------------------------------------------------------
# Импорт локальных спектральных файлов

def spectra_detect_local_file_format(filepath):
    """
    Определяет формат локального спектрального файла.
    Важно: не доверяем расширению файла, сначала смотрим содержимое.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(5000)

    head_strip = head.lstrip()

    if head_strip.startswith("{") or head_strip.startswith("["):
        return "MASSBANK_JSON"

    upper = head.upper()

    if "##TITLE" in upper or "##XYDATA" in upper or "##PEAK TABLE" in upper:
        return "JCAMP_DX"

    if "," in head or ";" in head or "\t" in head:
        return "NUMERIC_TABLE"

    return "UNKNOWN"


def spectra_extract_inchikey_from_filename(filename):
    """
    Ищет InChIKey в имени файла.
    Полный InChIKey надёжен:
    AAAQKTZKLRYKHR-UHFFFAOYSA-N

    Короткий фрагмент InChIKey не считаем точным совпадением.
    """
    base = os.path.basename(str(filename)).upper()

    full_pattern = r"[A-Z]{14}-[A-Z]{10}-[A-Z]"
    short_pattern = r"[A-Z]{14}"

    full_match = re.search(full_pattern, base)

    if full_match:
        return full_match.group(0), "full"

    short_match = re.search(short_pattern, base)

    if short_match:
        return short_match.group(0), "short"

    return "", ""


def spectra_parse_massbank_json_file(filepath):
    """
    Читает локальный MassBank/MoNA JSON.
    Поддерживает файлы, которые могут иметь расширение .jdx,
    но внутри содержат JSON.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)

    compound = {}

    compounds = obj.get("compound", [])

    if isinstance(compounds, list) and compounds:
        compound = compounds[0]
    elif isinstance(compounds, dict):
        compound = compounds

    inchikey = (
        compound.get("inchiKey", "")
        or compound.get("inchikey", "")
        or compound.get("InChIKey", "")
        or ""
    )

    inchi = (
        compound.get("inchi", "")
        or compound.get("InChI", "")
        or ""
    )

    name = ""

    names = compound.get("names", [])

    if isinstance(names, list) and names:
        first_name = names[0]

        if isinstance(first_name, dict):
            name = first_name.get("name", "")
        else:
            name = str(first_name)

    cas = ""
    smiles = ""
    formula = ""

    metadata_list = compound.get("metaData", [])

    if isinstance(metadata_list, list):
        for item in metadata_list:
            if not isinstance(item, dict):
                continue

            item_name = str(item.get("name", "")).strip().lower()
            item_value = str(item.get("value", "")).strip()

            if item_name == "cas":
                cas = item_value
            elif item_name == "smiles" and not smiles:
                smiles = item_value
            elif item_name == "molecular formula" and not formula:
                formula = item_value
            elif item_name == "inchikey" and not inchikey:
                inchikey = item_value
            elif item_name == "inchi" and not inchi:
                inchi = item_value

    canonical_smiles = ""

    if smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

                if not inchikey:
                    try:
                        inchikey = Chem.MolToInchiKey(mol)
                    except Exception:
                        pass

        except Exception:
            pass

    spectrum_text = str(obj.get("spectrum", "")).strip()

    rows = []

    for token in spectrum_text.split():
        if ":" not in token:
            continue

        x_text, y_text = token.split(":", 1)

        try:
            rows.append((float(x_text), float(y_text)))
        except Exception:
            continue

    spectrum_df = pd.DataFrame(
        rows,
        columns=["wavenumber", "intensity"]
    )

    if not spectrum_df.empty:
        spectrum_df = (
            spectrum_df
            .dropna()
            .drop_duplicates(subset=["wavenumber"])
            .sort_values("wavenumber")
            .reset_index(drop=True)
        )

    source_database = ""

    library = obj.get("library", {})

    if isinstance(library, dict):
        source_database = library.get("library", "") or library.get("name", "")

    if not source_database:
        source_database = "MassBank/MoNA"

    source_url = ""

    if isinstance(library, dict):
        source_url = library.get("link", "")

    splash = ""

    splash_obj = obj.get("splash", "")

    if isinstance(splash_obj, dict):
        splash = splash_obj.get("splash", "")
    else:
        splash = str(splash_obj)

    metadata = {
        "parser": "massbank_json",
        "source_database": source_database,
        "source_url": source_url,
        "inchikey": inchikey,
        "inchi": inchi,
        "name": name,
        "cas": cas,
        "smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "splash": splash,
        "spectrum_type": "Mass",
        "phase": "gas",
        "intensity_type": "relative abundance",
    }

    return spectrum_df, metadata


def spectra_read_numeric_table_file(filepath):
    """
    Читает простой числовой файл:
    первая колонка — ось спектра,
    вторая колонка — интенсивность.
    """
    temp = pd.read_csv(
        filepath,
        sep=None,
        engine="python",
        comment="#",
        header=None
    )

    numeric = temp.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.dropna(axis=0, how="all")

    if numeric.shape[1] < 2:
        return pd.DataFrame(), {"parser": "numeric_table"}

    spectrum_df = numeric.iloc[:, :2].copy()
    spectrum_df.columns = ["wavenumber", "intensity"]

    spectrum_df = (
        spectrum_df
        .dropna()
        .drop_duplicates(subset=["wavenumber"])
        .sort_values("wavenumber")
        .reset_index(drop=True)
    )

    metadata = {
        "parser": "numeric_table",
        "source_database": "LOCAL",
        "source_url": "",
        "spectrum_type": "",
        "phase": "unknown",
        "intensity_type": "",
    }

    return spectrum_df, metadata

def spectra_import_local_spectrum_file(
    filepath,
    spectrum_type="auto",
    compound_hint=None,
    overwrite_existing=False
):
    """
    Импортирует локальный спектральный файл в spectra_bank.

    Главное правило:
    - сначала ищем InChIKey / SMILES внутри файла;
    - имя файла используем только как запасной вариант.
    """
    compound_hint = compound_hint or {}

    detected_format = spectra_detect_local_file_format(filepath)

    spectrum_df = pd.DataFrame()
    metadata = {}

    if detected_format == "MASSBANK_JSON":
        spectrum_df, metadata = spectra_parse_massbank_json_file(filepath)

        if spectrum_type == "auto":
            spectrum_type_norm = "Mass"
        else:
            spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

    elif detected_format == "JCAMP_DX":
        spectrum_df, metadata = spectra_read_jdx_file(filepath)

        if spectrum_type == "auto":
            spectrum_type_norm = "IR"
        else:
            spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

        metadata["phase"] = metadata.get("state", metadata.get("phase", "unknown"))
        metadata["source_database"] = "LOCAL"
        metadata["source_url"] = ""
        metadata["intensity_type"] = metadata.get("yunits", "")

    elif detected_format == "NUMERIC_TABLE":
        spectrum_df, metadata = spectra_read_numeric_table_file(filepath)

        if spectrum_type == "auto":
            spectrum_type_norm = "IR"
        else:
            spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

        metadata["spectrum_type"] = spectrum_type_norm

    else:
        return {
            "status": "unsupported_format",
            "record": None,
            "message": f"Неизвестный формат файла: {os.path.basename(filepath)}"
        }

    if spectrum_df is None or spectrum_df.empty:
        return {
            "status": "parse_error",
            "record": None,
            "message": f"Не удалось прочитать численные точки спектра: {os.path.basename(filepath)}"
        }

    inchikey = (
        str(metadata.get("inchikey", "")).strip()
        or str(compound_hint.get("inchikey", "")).strip()
    )

    canonical_smiles = (
        str(metadata.get("canonical_smiles", "")).strip()
        or str(compound_hint.get("canonical_smiles", "")).strip()
    )

    filename_inchikey, filename_inchikey_mode = spectra_extract_inchikey_from_filename(filepath)

    if not inchikey and filename_inchikey_mode == "full":
        inchikey = filename_inchikey

    if not inchikey:
        return {
            "status": "unmatched_compound",
            "record": None,
            "message": (
                "Спектр прочитан, но вещество не определено. "
                "Нужен InChIKey внутри файла, полный InChIKey в имени файла "
                "или ручное сопоставление."
            ),
            "metadata": metadata,
        }

    source_database = str(metadata.get("source_database", "")).strip()
    source_url = str(metadata.get("source_url", "")).strip()
    splash = str(metadata.get("splash", "")).strip()

    if not source_database:
        source_database = "LOCAL"

    if not overwrite_existing:
        duplicate_record = spectra_find_duplicate_spectrum_record(
            inchikey=inchikey,
            canonical_smiles=canonical_smiles,
            spectrum_type=spectrum_type_norm,
            source_database=source_database,
            source_url=source_url,
            splash=splash,
            wavenumber_min=float(spectrum_df["wavenumber"].min()),
            wavenumber_max=float(spectrum_df["wavenumber"].max()),
            n_points=len(spectrum_df)
        )

        if duplicate_record is not None:
            return {
                "status": "already_registered",
                "record": duplicate_record,
                "message": "Такой же спектр уже зарегистрирован в spectra_index.csv."
            }

    compound = {
        "compound_id": compound_hint.get("compound_id", ""),
        "name": compound_hint.get("name", "") or metadata.get("name", ""),
        "cas": compound_hint.get("cas", "") or metadata.get("cas", ""),
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
    }

    candidate = {
        "source": "User local import",
        "source_database": source_database,
        "source_url": source_url,
        "spectrum_type": spectrum_type_norm,
        "phase": metadata.get("phase", "unknown"),
        "intensity_type": metadata.get("intensity_type", ""),
        "sample_type": "pure compound",
        "is_experimental": True,
        "is_quantitative": False,
        "download_mode": "local_import",
        "confidence_level": "metadata_in_file" if metadata.get("inchikey") else "filename_or_manual",
    }

    spectrum_id, raw_file, processed_file = spectra_make_flat_spectrum_paths(
        compound=compound,
        candidate=candidate
    )

    os.makedirs(os.path.dirname(raw_file), exist_ok=True)

    src_abs = os.path.abspath(filepath)
    dst_abs = os.path.abspath(raw_file)

    if src_abs != dst_abs:
        shutil.copyfile(filepath, raw_file)
    else:
        raw_file = filepath

    spectra_save_processed_spectrum(
        spectrum_df=spectrum_df,
        processed_file=processed_file
    )

    record = {
        "spectrum_id": spectrum_id,
        "compound_id": compound.get("compound_id", ""),
        "name": compound.get("name", ""),
        "cas": compound.get("cas", ""),
        "canonical_smiles": compound.get("canonical_smiles", ""),
        "inchikey": compound.get("inchikey", ""),
        "source": "User local import",
        "source_database": candidate.get("source_database", ""),
        "source_url": candidate.get("source_url", ""),
        "spectrum_type": spectrum_type_norm,
        "phase": candidate.get("phase", ""),
        "intensity_type": candidate.get("intensity_type", ""),
        "sample_type": candidate.get("sample_type", "pure compound"),
        "is_experimental": True,
        "is_quantitative": False,
        "download_mode": "local_import",
        "confidence_level": candidate.get("confidence_level", ""),
        "format": detected_format,
        "raw_file": raw_file,
        "processed_file": processed_file,
        "wavenumber_min": float(spectrum_df["wavenumber"].min()),
        "wavenumber_max": float(spectrum_df["wavenumber"].max()),
        "n_points_raw": len(spectrum_df),
        "n_points_processed": len(spectrum_df),
        "status": "imported_local",
        "date_downloaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "comment": (
            f"local_file={os.path.basename(filepath)}; "
            f"parser={metadata.get('parser', '')}; "
            f"filename_inchikey={filename_inchikey}; "
            f"filename_inchikey_mode={filename_inchikey_mode}; "
            f"splash={metadata.get('splash', '')}"
        ),
        "active": True,
    }

    spectra_add_to_index(record)

    return {
        "status": "imported_local",
        "record": record,
        "message": f"Файл импортирован в spectra_bank как {spectrum_id}."
    }

def spectra_find_duplicate_spectrum_record(
    inchikey,
    canonical_smiles,
    spectrum_type,
    source_database="",
    source_url="",
    splash="",
    wavenumber_min=None,
    wavenumber_max=None,
    n_points=None
):
    """
    Ищет уже зарегистрированный такой же спектр в spectra_index.csv.

    Дубль считаем по:
    - InChIKey или canonical_smiles;
    - типу спектра IR/Mass;
    - источнику;
    - source_url или splash, если они есть;
    - числу точек и диапазону оси, если source_url/splash нет.
    """
    index_df = spectra_load_index()

    if index_df is None or index_df.empty:
        return None

    work = index_df.copy()

    for col in [
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "source_database",
        "source_url",
        "comment",
        "wavenumber_min",
        "wavenumber_max",
        "n_points_processed",
        "active",
    ]:
        if col not in work.columns:
            work[col] = ""

    work["_spectrum_type_norm"] = (
        work["spectrum_type"]
        .astype(str)
        .apply(spectra_normalize_spectrum_type)
    )

    active_values = ["true", "1", "yes", "y", "да", "active", "", "nan", "none"]

    work["_active_norm"] = (
        work["active"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(active_values)
    )

    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

    mask = (
        (work["_spectrum_type_norm"] == spectrum_type_norm)
        & (work["_active_norm"])
    )

    inchikey = str(inchikey).strip()
    canonical_smiles = str(canonical_smiles).strip()

    if inchikey:
        mask = mask & (
            work["inchikey"].astype(str).str.strip() == inchikey
        )
    elif canonical_smiles:
        mask = mask & (
            work["canonical_smiles"].astype(str).str.strip() == canonical_smiles
        )
    else:
        return None

    candidates = work.loc[mask].copy()

    if candidates.empty:
        return None

    source_database = str(source_database).strip()
    source_url = str(source_url).strip()
    splash = str(splash).strip()

    if source_database:
        candidates = candidates[
            candidates["source_database"].astype(str).str.strip().str.lower()
            == source_database.lower()
        ].copy()

        if candidates.empty:
            return None

    if source_url:
        same_url = candidates[
            candidates["source_url"].astype(str).str.strip() == source_url
        ].copy()

        if not same_url.empty:
            return same_url.iloc[0].to_dict()

    if splash:
        same_splash = candidates[
            candidates["comment"].astype(str).str.contains(
                splash,
                regex=False,
                na=False
            )
        ].copy()

        if not same_splash.empty:
            return same_splash.iloc[0].to_dict()

    if (
        wavenumber_min is not None
        and wavenumber_max is not None
        and n_points is not None
    ):
        temp = candidates.copy()

        temp["_wmin"] = pd.to_numeric(
            temp["wavenumber_min"],
            errors="coerce"
        )

        temp["_wmax"] = pd.to_numeric(
            temp["wavenumber_max"],
            errors="coerce"
        )

        temp["_n"] = pd.to_numeric(
            temp["n_points_processed"],
            errors="coerce"
        )

        same_numeric = temp[
            (np.isclose(temp["_wmin"], float(wavenumber_min), rtol=0, atol=1e-6))
            & (np.isclose(temp["_wmax"], float(wavenumber_max), rtol=0, atol=1e-6))
            & (temp["_n"] == int(n_points))
        ].copy()

        if not same_numeric.empty:
            return same_numeric.iloc[0].to_dict()

    return None

def spectra_reindex_existing_raw_spectra(
    scan_ir=True,
    scan_mass=True,
    overwrite_existing=False,
    recursive=False
):
    """
    Сканирует штатные папки spectra_bank:

    - spectra_bank/IR/raw_jdx
    - spectra_bank/Mass/raw_jdx

    Файлы не считаются новыми спектрами сами по себе.
    Функция читает их, создаёт processed CSV и добавляет записи в spectra_index.csv.

    Повторно одинаковые спектры не добавляются.
    """
    folders = []

    if scan_ir:
        folders.append(("IR", SPECTRA_IR_RAW_DIR))

    if scan_mass:
        folders.append(("Mass", SPECTRA_MASS_RAW_DIR))

    allowed_ext = {
        ".jdx",
        ".dx",
        ".json",
        ".txt",
        ".csv",
    }

    rows = []

    for spectrum_type, folder_path in folders:
        if not os.path.exists(folder_path):
            rows.append({
                "folder": folder_path,
                "file": "",
                "status": "folder_not_found",
                "message": "Папка не найдена.",
                "spectrum_id": "",
                "spectrum_type": spectrum_type,
                "inchikey": "",
                "name": "",
                "raw_file": "",
                "processed_file": "",
            })
            continue

        file_paths = []

        if recursive:
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()

                    if ext in allowed_ext:
                        file_paths.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(folder_path):
                path = os.path.join(folder_path, filename)

                if not os.path.isfile(path):
                    continue

                ext = os.path.splitext(filename)[1].lower()

                if ext in allowed_ext:
                    file_paths.append(path)

        for filepath in sorted(file_paths):
            try:
                result = spectra_import_local_spectrum_file(
                    filepath=filepath,
                    spectrum_type=spectrum_type,
                    compound_hint=None,
                    overwrite_existing=overwrite_existing
                )

                record = result.get("record") or {}

                rows.append({
                    "folder": folder_path,
                    "file": os.path.basename(filepath),
                    "status": result.get("status", ""),
                    "message": result.get("message", ""),
                    "spectrum_id": record.get("spectrum_id", ""),
                    "spectrum_type": record.get("spectrum_type", spectrum_type),
                    "inchikey": record.get("inchikey", ""),
                    "name": record.get("name", ""),
                    "n_points": record.get("n_points_processed", ""),
                    "raw_file": record.get("raw_file", ""),
                    "processed_file": record.get("processed_file", ""),
                })

            except Exception as e:
                rows.append({
                    "folder": folder_path,
                    "file": os.path.basename(filepath),
                    "status": "reindex_error",
                    "message": str(e),
                    "spectrum_id": "",
                    "spectrum_type": spectrum_type,
                    "inchikey": "",
                    "name": "",
                    "n_points": "",
                    "raw_file": filepath,
                    "processed_file": "",
                })

    if not rows:
        rows.append({
            "folder": "",
            "file": "",
            "status": "no_files",
            "message": "В штатных папках raw_jdx не найдено спектральных файлов.",
            "spectrum_id": "",
            "spectrum_type": "",
            "inchikey": "",
            "name": "",
            "n_points": "",
            "raw_file": "",
            "processed_file": "",
        })

    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# Логи поиска

def spectra_make_json_safe(obj):
    """
    Делает объект безопасным для json.dump.

    В MoNA-кандидатах могут быть pandas DataFrame, numpy-типы
    и служебные поля вроде _peak_df. В JSON-журнал это напрямую
    записывать нельзя.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.head(50).to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        safe = {}

        for key, value in obj.items():
            key_str = str(key)

            # В журнал не пишем тяжёлые внутренние поля.
            # Они нужны только внутри алгоритма.
            if key_str.startswith("_"):
                continue

            safe[key_str] = spectra_make_json_safe(value)

        return safe

    if isinstance(obj, list):
        return [spectra_make_json_safe(x) for x in obj]

    if isinstance(obj, tuple):
        return [spectra_make_json_safe(x) for x in obj]

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def spectra_write_search_log(*args, **kwargs):
    """
    Записывает подробный JSON-журнал поиска спектра.

    Сделано через *args/**kwargs, чтобы не зависеть от того,
    как именно функция вызывается в spectra_search_one_compound.
    """
    if not SPECTRA_WRITE_JSON_SEARCH_LOGS:
        return ""    
    try:
        # ------------------------------------------------------------
        # 1. Собираем данные журнала

        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            log_data = args[0]
        else:
            log_data = {
                "args": args,
                "kwargs": kwargs,
            }

        # ------------------------------------------------------------
        # 2. Определяем папку журнала

        log_dir = None

        for candidate_dir_name in [
            "SPECTRA_MASS_LOG_DIR",
            "SPECTRA_SEARCH_LOG_DIR",
            "SPECTRA_LOG_DIR",
        ]:
            if candidate_dir_name in globals():
                candidate_dir = globals().get(candidate_dir_name)

                if candidate_dir:
                    log_dir = candidate_dir
                    break

        if log_dir is None:
            log_dir = os.path.join("spectra_bank", "Mass", "search_log")

        os.makedirs(log_dir, exist_ok=True)

        # ------------------------------------------------------------
        # 3. Создаём имя файла

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        inchikey_for_name = ""

        try:
            if isinstance(log_data, dict):
                if "compound" in log_data and isinstance(log_data["compound"], dict):
                    inchikey_for_name = str(
                        log_data["compound"].get("inchikey", "")
                    ).strip()

                if not inchikey_for_name and "kwargs" in log_data:
                    kw = log_data.get("kwargs", {})

                    if isinstance(kw, dict):
                        compound_kw = kw.get("compound", {})

                        if isinstance(compound_kw, dict):
                            inchikey_for_name = str(
                                compound_kw.get("inchikey", "")
                            ).strip()
        except Exception:
            inchikey_for_name = ""

        safe_inchikey = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            inchikey_for_name
        )[:80]

        if safe_inchikey:
            filename = f"spectra_search_log_{safe_inchikey}_{timestamp}.json"
        else:
            filename = f"spectra_search_log_{timestamp}.json"

        full_log_file = os.path.join(log_dir, filename)

        # ------------------------------------------------------------
        # 4. Делаем объект JSON-безопасным и сохраняем

        safe_log_data = spectra_make_json_safe(log_data)

        with open(full_log_file, "w", encoding="utf-8") as f:
            json.dump(
                safe_log_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        return full_log_file

    except Exception:
        # Журнал не должен ломать поиск спектров.
        return None

# ------------------------------------------------------------------
# Имена файлов spectra_bank

def spectra_next_file_number(inchikey, spectrum_type="IR", source_code="NIST", phase="unknown"):
    """
    Возвращает следующий номер файла для данного вещества и типа спектра.

    Пример:
    IR_NIST_gas_XXXX_001.jdx
    IR_NIST_gas_XXXX_002.jdx
    """
    index_df = spectra_load_index()

    spectrum_type = spectra_normalize_spectrum_type(spectrum_type)

    if index_df.empty or "inchikey" not in index_df.columns:
        return 1

    same = index_df[
        (index_df["inchikey"].astype(str) == str(inchikey)) &
        (index_df["spectrum_type"].astype(str).apply(spectra_normalize_spectrum_type) == spectrum_type)
    ]

    if same.empty:
        return 1

    return len(same) + 1


def spectra_make_flat_spectrum_paths(compound, candidate):
    """
    Формирует spectrum_id, raw_file, processed_file по плоской схеме.

    Формат:
    IR_NIST_gas_InChIKey_001.jdx
    IR_NIST_gas_InChIKey_001_processed.csv
    """
    spectrum_type = spectra_normalize_spectrum_type(
        candidate.get("spectrum_type", "IR")
    )

    source_code = spectra_source_code(
        candidate.get("source", "")
    )

    phase = spectra_safe_filename_part(
        candidate.get("phase", "unknown")
    ).lower()

    inchikey = str(compound.get("inchikey", "")).strip()

    if not inchikey:
        inchikey = "NO_INCHIKEY_" + spectra_safe_filename_part(
            compound.get("canonical_smiles", "")
        )[:40]

    inchikey_safe = spectra_safe_filename_part(inchikey)

    number = spectra_next_file_number(
        inchikey=inchikey,
        spectrum_type=spectrum_type,
        source_code=source_code,
        phase=phase
    )

    short_name = f"{spectrum_type}_{source_code}_{phase}_{inchikey_safe}_{number:03d}"

    spectrum_id = short_name

    dirs = spectra_get_dirs_by_type(spectrum_type)

    raw_file = os.path.join(
        dirs["raw"],
        f"{short_name}.jdx"
    )

    processed_file = os.path.join(
        dirs["processed"],
        f"{short_name}_processed.csv"
    )

    return spectrum_id, raw_file, processed_file


def spectra_make_flat_spectrum_paths_with_phase(compound, candidate, phase):
    """
    То же, что spectra_make_flat_spectrum_paths,
    но фаза уже известна после чтения JCAMP.
    """
    candidate2 = dict(candidate)
    candidate2["phase"] = phase

    return spectra_make_flat_spectrum_paths(
        compound=compound,
        candidate=candidate2
    )


# ------------------------------------------------------------------
# Скачивание спектров

def spectra_download_best_candidate(compound, candidates):
    """
    Скачивает лучший подходящий кандидат.

    Для IR:
    - обязательно предпочитает gas/vapor/vapour;
    - не сохраняет первый попавшийся liquid/solid, если среди кандидатов есть gas;
    - сначала скачивает кандидаты во временные файлы, определяет фазу,
      потом сохраняет только лучший.

    Для Mass:
    - оставляет прежнюю логику: первый успешно скачанный численный JCAMP
      по candidate_score.
    """
    if not candidates:
        return {
            "status": "not_found",
            "record": None,
            "message": "Кандидаты не найдены.",
            "candidate_url": "",
        }

    candidates = sorted(
        candidates,
        key=lambda x: float(x.get("candidate_score", 0) or 0),
        reverse=True
    )

    first_candidate = candidates[0]
    spectrum_type_norm = spectra_normalize_spectrum_type(
        first_candidate.get("spectrum_type", "IR")
    )

    existing = spectra_find_in_bank(
        inchikey=compound.get("inchikey", ""),
        canonical_smiles=compound.get("canonical_smiles", ""),
        spectrum_type=spectrum_type_norm
    )

    if existing is not None:
        return {
            "status": "already_in_bank",
            "record": existing,
            "message": "Спектр уже есть в локальной базе. Повторное скачивание не выполнялось.",
            "candidate_url": existing.get("source_url", ""),
        }

    last_error_message = ""
    downloaded_options = []

    for cand in candidates:
        url = cand.get("source_url", "")

        if not url:
            continue

        tmp_raw_file = None

        try:
            cand_type_norm = spectra_normalize_spectrum_type(
                cand.get("spectrum_type", spectrum_type_norm)
            )

            dirs = spectra_get_dirs_by_type(cand_type_norm)

            os.makedirs(dirs["raw"], exist_ok=True)
            os.makedirs(dirs["processed"], exist_ok=True)

            tmp_name = (
                "tmp_"
                + spectra_safe_filename_part(str(compound.get("inchikey", "")))
                + "_"
                + uuid.uuid4().hex
                + ".jdx"
            )

            tmp_raw_file = os.path.join(dirs["raw"], tmp_name)

            spectra_download_binary(url, tmp_raw_file, timeout=15)

            with open(tmp_raw_file, "r", encoding="latin-1", errors="replace") as f:
                raw_text_full = f.read()

            if not spectra_is_jcamp_text(raw_text_full[:3000]):
                try:
                    os.remove(tmp_raw_file)
                except Exception:
                    pass

                last_error_message = "Скачанный файл не похож на JCAMP-DX."
                continue

            detected_phase = spectra_extract_phase_from_jcamp_text(raw_text_full)

            spectrum_df, metadata = spectra_read_jdx_file(tmp_raw_file)

            if spectrum_df.empty:
                try:
                    os.remove(tmp_raw_file)
                except Exception:
                    pass

                last_error_message = "JCAMP скачан, но точки спектра не прочитаны."
                continue

            cand_with_phase = dict(cand)
            cand_with_phase["phase"] = detected_phase

            downloaded_options.append({
                "candidate": cand_with_phase,
                "url": url,
                "tmp_raw_file": tmp_raw_file,
                "spectrum_df": spectrum_df,
                "metadata": metadata,
                "detected_phase": detected_phase,
                "phase_priority": spectra_phase_priority(detected_phase),
                "candidate_score": float(cand.get("candidate_score", 0) or 0),
                "spectrum_type": cand_type_norm,
            })

            # Для Mass можно оставить старое поведение:
            # первый валидный спектр достаточно.
            if spectrum_type_norm != "IR":
                break

        except Exception as e:
            last_error_message = str(e)

            try:
                if tmp_raw_file and os.path.exists(tmp_raw_file):
                    os.remove(tmp_raw_file)
            except Exception:
                pass

            continue

    if not downloaded_options:
        return {
            "status": "no_numeric_spectrum",
            "record": None,
            "message": (
                "Кандидаты найдены, но ни один численный JCAMP-DX не скачан. "
                f"{last_error_message}"
            ),
            "candidate_url": "",
        }

    # Для IR выбираем строго по фазе, затем по candidate_score.
    if spectrum_type_norm == "IR":
        downloaded_options = sorted(
            downloaded_options,
            key=lambda x: (
                x.get("phase_priority", 0),
                x.get("candidate_score", 0)
            ),
            reverse=True
        )

    best = downloaded_options[0]

    # Удаляем временные файлы всех проигравших кандидатов.
    for option in downloaded_options[1:]:
        try:
            loser_tmp = option.get("tmp_raw_file", "")
            if loser_tmp and os.path.exists(loser_tmp):
                os.remove(loser_tmp)
        except Exception:
            pass

    cand = best["candidate"]
    url = best["url"]
    tmp_raw_file = best["tmp_raw_file"]
    spectrum_df = best["spectrum_df"]
    metadata = best["metadata"]
    detected_phase = best["detected_phase"]

    existing = spectra_find_in_bank(
        inchikey=compound.get("inchikey", ""),
        canonical_smiles=compound.get("canonical_smiles", ""),
        spectrum_type=spectrum_type_norm
    )

    if existing is not None:
        try:
            os.remove(tmp_raw_file)
        except Exception:
            pass

        return {
            "status": "already_in_bank",
            "record": existing,
            "message": "Спектр уже был добавлен в банк параллельно. Повторный файл удалён.",
            "candidate_url": existing.get("source_url", ""),
        }

    spectrum_id, raw_file, processed_file = spectra_make_flat_spectrum_paths(
        compound=compound,
        candidate=cand
    )

    if os.path.exists(raw_file):
        try:
            os.remove(tmp_raw_file)
        except Exception:
            pass

        existing = spectra_find_in_bank(
            inchikey=compound.get("inchikey", ""),
            canonical_smiles=compound.get("canonical_smiles", ""),
            spectrum_type=spectrum_type_norm
        )

        if existing is not None:
            return {
                "status": "already_in_bank",
                "record": existing,
                "message": "Файл спектра уже существует. Повторное скачивание не выполнялось.",
                "candidate_url": existing.get("source_url", ""),
            }

        return {
            "status": "already_in_bank",
            "record": None,
            "message": f"Файл уже существует: {raw_file}. Повторное скачивание не выполнялось.",
            "candidate_url": url,
        }

    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    os.replace(tmp_raw_file, raw_file)

    spectra_save_processed_spectrum(
        spectrum_df=spectrum_df,
        processed_file=processed_file
    )

    record = {
        "spectrum_id": spectrum_id,
        "compound_id": compound.get("compound_id", ""),
        "name": compound.get("name", ""),
        "cas": compound.get("cas", ""),
        "canonical_smiles": compound.get("canonical_smiles", ""),
        "inchikey": compound.get("inchikey", ""),
        "source": cand.get("source", ""),
        "source_database": cand.get("source_database", cand.get("source", "")),
        "source_url": url,
        "spectrum_type": cand.get("spectrum_type", spectrum_type_norm),
        "phase": detected_phase,
        "intensity_type": cand.get("intensity_type", ""),
        "sample_type": cand.get("sample_type", "pure compound"),
        "is_experimental": cand.get("is_experimental", True),
        "is_quantitative": cand.get("is_quantitative", False),
        "download_mode": cand.get("download_mode", "auto"),
        "confidence_level": cand.get("confidence_level", ""),
        "format": "JCAMP-DX",
        "raw_file": raw_file,
        "processed_file": processed_file,
        "wavenumber_min": float(spectrum_df["wavenumber"].min()),
        "wavenumber_max": float(spectrum_df["wavenumber"].max()),
        "n_points_raw": len(spectrum_df),
        "n_points_processed": len(spectrum_df),
        "status": "found_downloaded",
        "date_downloaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "comment": (
            f"parser={metadata.get('parser', '')}; "
            f"state={detected_phase}; "
            f"phase_priority={spectra_phase_priority(detected_phase)}; "
            f"ir_gas_phase_priority={'yes' if spectrum_type_norm == 'IR' else 'no'}"
        ),
        "active": True,
    }

    spectra_add_to_index(record)

    return {
        "status": "found_downloaded",
        "record": record,
        "message": (
            f"Спектр скачан и сохранён как {spectrum_id}. "
            f"Выбрана фаза: {detected_phase}. "
            f"Приоритет IR gas-phase применён."
        ),
        "candidate_url": url,
    }

def spectra_phase_priority(phase):
    """
    Приоритет фаз для выбора спектра.
    Больше число = выше приоритет.
    """
    phase = str(phase).strip().lower()

    if phase in ["gas", "vapor", "vapour"]:
        return 100

    if phase in ["liquid", "film"]:
        return 60

    if phase in ["solution"]:
        return 40

    if phase in ["solid", "kbr", "nujol"]:
        return 30

    if phase in ["unknown", "", "nan", "none"]:
        return 10

    return 20

def spectra_search_one_compound(compound, spectrum_type="IR", selected_sources=None, delay_seconds=1.0):
    """
    Каскадный поиск одного вещества.

    Логика:
    1. Проверяем локальный spectra_bank.
    2. Если спектра нет, идём по выбранным источникам.
    3. Для каждого источника пишем not-found/error в wide-таблицу по базам.
    4. Если спектр найден и скачан, возвращаем found_downloaded.
    5. Если нигде не найден, возвращаем not_found_in_all_sources.
    """
    if compound is None:
        compound = {}

    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)

    if selected_sources is None:
        selected_sources = []

    inchikey = str(compound.get("inchikey", "")).strip()
    canonical_smiles = str(compound.get("canonical_smiles", "")).strip()

    if spectra_is_stop_requested():
        return {
            "compound_id": compound.get("compound_id", ""),
            "name": compound.get("name", ""),
            "cas": compound.get("cas", ""),
            "input_smiles": compound.get("input_smiles", ""),
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "structure_status": compound.get("structure_status", ""),
            "spectrum_type": spectrum_type_norm,
            "spectrum_status": "stopped_by_user",
            "selected_source": "",
            "candidate_count": 0,
            "spectrum_id": "",
            "raw_file": "",
            "processed_file": "",
            "message": "Поиск остановлен пользователем до начала внешнего запроса.",
        }

    log_data = {
        "compound": compound,
        "spectrum_type": spectrum_type_norm,
        "sources_checked": [],
        "final_status": "",
        "selected_record": None,
    }

    def _safe_add_not_found(
        source_key,
        source_label,
        status,
        candidate_count=0,
        source_url="",
        search_url="",
        message="",
        error="",
    ):
        """
        Безопасная запись в таблицу not-found.
        Если таблицы ещё не добавлены или функция отсутствует,
        поиск не должен падать.
        """
        try:
            spectra_add_to_not_found_table(
                compound=compound,
                spectrum_type=spectrum_type_norm,
                source_key=source_key,
                source_label=source_label,
                status=status,
                candidate_count=candidate_count,
                source_url=source_url,
                search_url=search_url,
                message=message,
                error=error,
            )
        except Exception:
            pass

    # ------------------------------------------------------------
    # 0. Некорректная структура

    if not inchikey and not canonical_smiles:
        return {
            "compound_id": compound.get("compound_id", ""),
            "name": compound.get("name", ""),
            "cas": compound.get("cas", ""),
            "input_smiles": compound.get("input_smiles", ""),
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "structure_status": compound.get("structure_status", ""),
            "spectrum_type": spectrum_type_norm,
            "spectrum_status": "invalid_structure",
            "selected_source": "",
            "candidate_count": 0,
            "spectrum_id": "",
            "raw_file": "",
            "processed_file": "",
            "message": "Нет InChIKey/canonical_smiles. Поиск спектра невозможен.",
        }

    # ------------------------------------------------------------
    # 1. Локальная база

    existing = spectra_find_in_bank(
        inchikey=inchikey,
        canonical_smiles=canonical_smiles,
        spectrum_type=spectrum_type_norm
    )

    if existing is not None:
        return {
            "compound_id": compound.get("compound_id", ""),
            "name": compound.get("name", ""),
            "cas": compound.get("cas", ""),
            "input_smiles": compound.get("input_smiles", ""),
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "structure_status": compound.get("structure_status", ""),
            "spectrum_type": spectrum_type_norm,
            "spectrum_status": "already_in_bank",
            "selected_source": (
                existing.get("source_database", "")
                or existing.get("source", "")
                or "local_bank"
            ),
            "candidate_count": 1,
            "spectrum_id": existing.get("spectrum_id", ""),
            "raw_file": existing.get("raw_file", ""),
            "processed_file": existing.get("processed_file", ""),
            "message": "Спектр уже есть в локальной базе.",
            "record": existing,
        }

    # ------------------------------------------------------------
    # 2. Внешние источники

    for source_key in selected_sources:
        if spectra_is_stop_requested():
            return {
                "compound_id": compound.get("compound_id", ""),
                "name": compound.get("name", ""),
                "cas": compound.get("cas", ""),
                "input_smiles": compound.get("input_smiles", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "structure_status": compound.get("structure_status", ""),
                "spectrum_type": spectrum_type_norm,
                "spectrum_status": "stopped_by_user",
                "selected_source": "",
                "candidate_count": 0,
                "spectrum_id": "",
                "raw_file": "",
                "processed_file": "",
                "message": "Поиск остановлен пользователем перед проверкой следующего источника.",
            }

        source_key = str(source_key).strip()

        if not source_key:
            continue

        if source_key not in SPECTRA_SOURCE_REGISTRY:
            log_data["sources_checked"].append({
                "source": source_key,
                "label": source_key,
                "status": "source_not_connected",
                "candidate_count": 0,
            })

            _safe_add_not_found(
                source_key=source_key,
                source_label=source_key,
                status="source_not_connected",
                candidate_count=0,
                message="Источник не подключён в SPECTRA_SOURCE_REGISTRY.",
            )

            continue

        source_info = SPECTRA_SOURCE_REGISTRY[source_key]

        source_label = source_info.get("label", source_key)
        source_mode = source_info.get("mode", "download")
        source_spectrum_type = source_info.get("spectrum_type", "IR")
        search_function = source_info.get("search_function", None)

        if spectra_normalize_spectrum_type(source_spectrum_type) != spectrum_type_norm:
            log_data["sources_checked"].append({
                "source": source_key,
                "label": source_label,
                "status": "wrong_spectrum_type",
                "candidate_count": 0,
            })
            continue

        if search_function is None:
            log_data["sources_checked"].append({
                "source": source_key,
                "label": source_label,
                "status": "search_function_missing",
                "candidate_count": 0,
            })

            _safe_add_not_found(
                source_key=source_key,
                source_label=source_label,
                status="search_function_missing",
                candidate_count=0,
                message="Для источника не задана search_function.",
            )

            continue

        try:
            time.sleep(float(delay_seconds))
        except Exception:
            pass
        if spectra_is_stop_requested():
            return {
                "compound_id": compound.get("compound_id", ""),
                "name": compound.get("name", ""),
                "cas": compound.get("cas", ""),
                "input_smiles": compound.get("input_smiles", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "structure_status": compound.get("structure_status", ""),
                "spectrum_type": spectrum_type_norm,
                "spectrum_status": "stopped_by_user",
                "selected_source": source_label,
                "candidate_count": 0,
                "spectrum_id": "",
                "raw_file": "",
                "processed_file": "",
                "message": "Поиск остановлен пользователем после паузы перед запросом к источнику.",
            }
        # --------------------------------------------------------
        # 2a. Поиск кандидатов в конкретном источнике

        try:
            candidates = search_function(compound)
        except Exception as e:
            candidates = []

            log_data["sources_checked"].append({
                "source": source_key,
                "label": source_label,
                "status": "search_error",
                "candidate_count": 0,
                "error": str(e),
            })

            _safe_add_not_found(
                source_key=source_key,
                source_label=source_label,
                status="search_error",
                candidate_count=0,
                message="Ошибка при поиске в источнике.",
                error=str(e),
            )

            continue

        candidates = spectra_clean_candidate_list(candidates)

        source_log = {
            "source": source_key,
            "label": source_label,
            "mode": source_mode,
            "status": "found" if candidates else "not_found",
            "candidate_count": len(candidates),
            "candidates": candidates[:5],
        }

        log_data["sources_checked"].append(source_log)

        if not candidates:
            _safe_add_not_found(
                source_key=source_key,
                source_label=source_label,
                status="not_found",
                candidate_count=0,
                message="Спектр не найден в этом источнике.",
            )

            continue

        # --------------------------------------------------------
        # 2b. Источники только со ссылками

        if source_mode == "link_only":
            best = sorted(
                candidates,
                key=lambda x: x.get("candidate_score", 0),
                reverse=True
            )[0]

            best_url = (
                best.get("source_url", "")
                or best.get("candidate_url", "")
                or best.get("search_url", "")
            )

            _safe_add_not_found(
                source_key=source_key,
                source_label=source_label,
                status="candidate_link_found",
                candidate_count=len(candidates),
                source_url=best_url,
                search_url=best.get("search_url", best_url),
                message=(
                    "Найдена ссылка-кандидат, автоматическое скачивание "
                    "для этого источника не включено."
                ),
            )

            log_data["final_status"] = "candidate_link_found"
            log_data["selected_record"] = best

            spectra_write_search_log(
                inchikey,
                log_data,
                spectrum_type=spectrum_type_norm
            )

            return {
                "compound_id": compound.get("compound_id", ""),
                "name": compound.get("name", ""),
                "cas": compound.get("cas", ""),
                "input_smiles": compound.get("input_smiles", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "structure_status": compound.get("structure_status", ""),
                "spectrum_type": spectrum_type_norm,
                "spectrum_status": "candidate_link_found",
                "selected_source": source_label,
                "candidate_count": len(candidates),
                "candidate_url": best_url,
                "spectrum_id": "",
                "raw_file": "",
                "processed_file": "",
                "message": (
                    f"Найдена ссылка-кандидат в {source_label}. "
                    "Автоматическое скачивание для этого источника пока не включено."
                ),
            }

        # --------------------------------------------------------
        # 2c. Источники с автоскачиванием

        if spectra_is_stop_requested():
            return {
                "compound_id": compound.get("compound_id", ""),
                "name": compound.get("name", ""),
                "cas": compound.get("cas", ""),
                "input_smiles": compound.get("input_smiles", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "structure_status": compound.get("structure_status", ""),
                "spectrum_type": spectrum_type_norm,
                "spectrum_status": "stopped_by_user",
                "selected_source": source_label,
                "candidate_count": len(candidates),
                "spectrum_id": "",
                "raw_file": "",
                "processed_file": "",
                "message": "Поиск остановлен пользователем перед скачиванием файла.",
            }

        if source_mode == "download_if_numeric_link_exists":
            downloaded = spectra_download_best_sdbs_candidate(
                compound=compound,
                candidates=candidates
            )
        elif source_mode == "mona_download":
            downloaded = spectra_download_best_mona_mass_candidate(
                compound=compound,
                candidates=candidates
            )
        else:
            downloaded = spectra_download_best_candidate(
                compound=compound,
                candidates=candidates
            )

        if downloaded is None:
            downloaded = {
                "status": "download_error",
                "record": None,
                "message": "Функция скачивания вернула None.",
                "candidate_url": "",
            }

        # --------------------------------------------------------
        # 2d. Спектр реально скачан и обработан

        if downloaded.get("record") is not None:
            record = downloaded.get("record", {}) or {}

            log_data["final_status"] = downloaded.get("status", "found_downloaded")
            log_data["selected_record"] = record

            spectra_write_search_log(
                inchikey,
                log_data,
                spectrum_type=spectrum_type_norm
            )

            return {
                "compound_id": compound.get("compound_id", ""),
                "name": compound.get("name", ""),
                "cas": compound.get("cas", ""),
                "input_smiles": compound.get("input_smiles", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "structure_status": compound.get("structure_status", ""),
                "spectrum_type": spectrum_type_norm,
                "spectrum_status": downloaded.get("status", "found_downloaded"),
                "selected_source": source_label,
                "candidate_count": len(candidates),
                "candidate_url": downloaded.get("candidate_url", ""),
                "spectrum_id": record.get("spectrum_id", ""),
                "raw_file": record.get("raw_file", ""),
                "processed_file": record.get("processed_file", ""),
                "message": downloaded.get("message", ""),
                "record": record,
            }

        # --------------------------------------------------------
        # 2e. Кандидаты были, но скачать/прочитать не удалось

        downloaded_status = str(downloaded.get("status", "")).strip()
        downloaded_message = str(downloaded.get("message", "")).strip()
        downloaded_candidate_url = str(downloaded.get("candidate_url", "")).strip()

        if downloaded_status == "candidate_page_no_numeric_data":
            status_for_table = "no_numeric_spectrum"
        elif downloaded_status:
            status_for_table = downloaded_status
        else:
            status_for_table = "not_found_after_candidate"

        _safe_add_not_found(
            source_key=source_key,
            source_label=source_label,
            status=status_for_table,
            candidate_count=len(candidates),
            source_url=downloaded_candidate_url,
            search_url=downloaded_candidate_url,
            message=downloaded_message,
            error=downloaded.get("error", ""),
        )

        # Для MoNA: если записи найдены, но не удалось получить пригодный peak list,
        # это не "not_found". Возвращаем честный статус пользователю и в журнал.
        if source_mode == "mona_download" and str(downloaded.get("status", "")).strip() in [
            "mona_records_found_but_not_parsed",
            "parse_error",
            "download_error",
            "search_error",
            "no_numeric_spectrum",
        ]:
            log_data["final_status"] = str(downloaded.get("status", "mona_records_found_but_not_parsed"))
            log_data["selected_record"] = {
                "source": source_key,
                "label": source_label,
                "candidate_count": len(candidates),
                "message": downloaded_message,
                "candidate_url": downloaded_candidate_url,
            }

            spectra_write_search_log(
                inchikey,
                log_data,
                spectrum_type=spectrum_type_norm
            )

            return {
                "compound_id": compound.get("compound_id", ""),
                "name": compound.get("name", ""),
                "cas": compound.get("cas", ""),
                "input_smiles": compound.get("input_smiles", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "structure_status": compound.get("structure_status", ""),
                "spectrum_type": spectrum_type_norm,
                "spectrum_status": str(downloaded.get("status", "mona_records_found_but_not_parsed")),
                "selected_source": source_label,
                "candidate_count": len(candidates),
                "candidate_url": downloaded_candidate_url,
                "spectrum_id": "",
                "raw_file": "",
                "processed_file": "",
                "message": downloaded_message,
            }

        # Для остальных источников продолжаем каскадный поиск.
        continue

    # ------------------------------------------------------------
    # 3. Нигде не найдено

    log_data["final_status"] = "not_found_in_all_sources"

    spectra_write_search_log(
        inchikey,
        log_data,
        spectrum_type=spectrum_type_norm
    )


    return {
        "compound_id": compound.get("compound_id", ""),
        "name": compound.get("name", ""),
        "cas": compound.get("cas", ""),
        "input_smiles": compound.get("input_smiles", ""),
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        "structure_status": compound.get("structure_status", ""),
        "spectrum_type": spectrum_type_norm,
        "spectrum_status": "not_found_in_all_sources",
        "selected_source": "",
        "candidate_count": 0,
        "spectrum_id": "",
        "raw_file": "",
        "processed_file": "",
        "message": "Спектр не найден во всех выбранных источниках.",
    }


# ------------------------------------------------------------------
# Спектральные дескрипторы

def spectral_resolve_processed_spectrum_path(processed_file, spectrum_record=None, allow_remote=True):
    """
    Находит processed CSV локально; при необходимости лениво скачивает из remote bank.
    """
    if processed_file is None:
        return ""

    processed_file = str(processed_file).strip()

    if not processed_file:
        return ""

    possible_paths = []

    possible_paths.append(processed_file)
    possible_paths.append(os.path.join(os.getcwd(), processed_file))

    bank_relative_path = spectra_normalize_bank_relative_path(processed_file)

    if bank_relative_path:
        bank_local_path = spectra_local_path_for_bank_relative(bank_relative_path)

        if bank_local_path:
            possible_paths.append(bank_local_path)

    base_name = os.path.basename(processed_file)

    possible_paths.append(os.path.join(SPECTRA_IR_PROCESSED_DIR, base_name))
    possible_paths.append(os.path.join(SPECTRA_MASS_PROCESSED_DIR, base_name))
    possible_paths.append(os.path.join(SPECTRA_BANK_DIR, "processed", base_name))

    for p in possible_paths:
        if p and os.path.exists(p):
            return p

    if allow_remote:
        remote_path = spectra_materialize_remote_bank_file(
            processed_file,
            spectrum_record=spectrum_record
        )

        if remote_path and os.path.exists(remote_path):
            return remote_path

    return ""


def spectral_load_processed_spectrum(processed_file, spectrum_record=None):
    """
    Загружает processed CSV спектра.
    Ожидаются колонки:
    wavenumber, intensity

    Дополнительно пытается найти файл, если путь в spectra_index.csv
    старый или относительный.
    """
    if processed_file is None:
        return pd.DataFrame()

    processed_file = str(processed_file).strip()

    real_path = spectral_resolve_processed_spectrum_path(
        processed_file,
        spectrum_record=spectrum_record,
        allow_remote=True
    )

    if not real_path:
        return pd.DataFrame()

    try:
        df = pd.read_csv(real_path)
    except Exception:
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.lower()

    if "wavenumber" not in df.columns or "intensity" not in df.columns:
        return pd.DataFrame()

    df["wavenumber"] = pd.to_numeric(df["wavenumber"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")

    df = df.dropna(subset=["wavenumber", "intensity"])
    df = df.drop_duplicates(subset=["wavenumber"])
    df = df.sort_values("wavenumber").reset_index(drop=True)

    return df


def spectral_interpolate_to_grid(spectrum_df, wn_min=550, wn_max=3798, step=4):
    """
    Интерполирует спектр на единую сетку волновых чисел.

    Возвращает:
    grid, values
    """
    if spectrum_df.empty:
        return np.array([]), np.array([])

    x = spectrum_df["wavenumber"].values.astype(float)
    y = spectrum_df["intensity"].values.astype(float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.array([]), np.array([])

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    grid = np.arange(float(wn_min), float(wn_max) + float(step), float(step))

    values = np.interp(grid, x, y)
    values[grid < x.min()] = np.nan
    values[grid > x.max()] = np.nan

    return grid, values


def spectral_normalize_values(values, method="min-max", invert=False):
    """
    Нормировка спектра.

    invert=True полезно для спектров пропускания:
    минимум пропускания соответствует максимуму поглощения.
    """
    values = np.asarray(values, dtype=float)

    if np.all(~np.isfinite(values)):
        return values

    med = np.nanmedian(values)
    values = np.where(np.isfinite(values), values, med)

    if invert:
        values = np.nanmax(values) - values

    if method == "none":
        return values

    if method == "min-max":
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        denom = vmax - vmin

        if denom <= 1e-12:
            return np.zeros_like(values)

        return (values - vmin) / denom

    if method == "sum":
        s = np.nansum(np.abs(values))

        if s <= 1e-12:
            return values

        return values / s

    if method == "vector":
        norm = np.sqrt(np.nansum(values ** 2))

        if norm <= 1e-12:
            return values

        return values / norm

    return values


def spectral_make_grid_descriptors(grid, values, prefix="IR_GRID"):
    """
    Каждая точка спектра становится числовым дескриптором.
    """
    desc = {}

    for wn, val in zip(grid, values):
        wn_int = int(round(wn))
        desc[f"{prefix}_{wn_int}"] = float(val)

    return desc


def spectral_make_binary_fp(grid, values, window_size=20, threshold=0.10, prefix="IR_BIN"):
    """
    SpectraFP-подобная бинаризация.

    Делим спектр на окна.
    В окне ставим 1, если максимум интенсивности >= threshold * global_max.
    """
    desc = {}

    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(grid) == 0 or len(values) == 0:
        return desc

    global_max = np.nanmax(values)

    if not np.isfinite(global_max) or global_max <= 1e-12:
        global_max = 1.0

    start = int(np.nanmin(grid))
    end = int(np.nanmax(grid))

    left = start

    while left < end:
        right = left + int(window_size)

        mask = (grid >= left) & (grid < right)

        if mask.any():
            local_max = np.nanmax(values[mask])
            bit = 1 if local_max >= threshold * global_max else 0
        else:
            bit = 0

        desc[f"{prefix}_{left}_{right}"] = bit

        left = right

    return desc


def spectral_make_binned_numeric_descriptors(grid, values, window_size=100, prefix="IR_BAND"):
    """
    Числовые дескрипторы по диапазонам:
    mean, max, area.
    """
    desc = {}

    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(grid) == 0 or len(values) == 0:
        return desc

    start = int(np.nanmin(grid))
    end = int(np.nanmax(grid))

    left = start

    while left < end:
        right = left + int(window_size)

        mask = (grid >= left) & (grid < right)

        if mask.any():
            local_values = values[mask]
            local_grid = grid[mask]

            desc[f"{prefix}_{left}_{right}_mean"] = float(np.nanmean(local_values))
            desc[f"{prefix}_{left}_{right}_max"] = float(np.nanmax(local_values))

            if len(local_values) >= 2:
                desc[f"{prefix}_{left}_{right}_area"] = float(np.trapz(local_values, local_grid))
            else:
                desc[f"{prefix}_{left}_{right}_area"] = 0.0

        left = right

    return desc


def spectral_descriptor_cache_file(spectrum_type="IR"):
    """
    Returns the per-spectrum descriptor cache file for a spectrum type.
    """
    spectrum_type = spectra_normalize_spectrum_type(spectrum_type)

    if spectrum_type == "Mass":
        return SPECTRA_MASS_DESCRIPTOR_CACHE_FILE

    return SPECTRA_IR_DESCRIPTOR_CACHE_FILE


def spectral_legacy_descriptor_cache_file(spectrum_type="IR"):
    """
    Returns the previous descriptor cache location for backward compatibility.
    """
    spectrum_type = spectra_normalize_spectrum_type(spectrum_type)

    if spectrum_type == "Mass":
        return SPECTRA_MASS_DESCRIPTOR_CACHE_LEGACY_FILE

    return SPECTRA_IR_DESCRIPTOR_CACHE_LEGACY_FILE


def spectral_remote_descriptor_cache_url(spectrum_type="IR"):
    """
    Returns a GitHub/raw URL for the ready descriptor cache, if configured.
    """
    spectrum_type = spectra_normalize_spectrum_type(spectrum_type)

    if spectrum_type == "Mass":
        return SPECTRA_REMOTE_MASS_DESCRIPTOR_CACHE_URL or SPECTRA_REMOTE_DESCRIPTOR_CACHE_URL

    return SPECTRA_REMOTE_IR_DESCRIPTOR_CACHE_URL or SPECTRA_REMOTE_DESCRIPTOR_CACHE_URL


def spectral_descriptor_shard_type_dir(spectrum_type="IR"):
    spectrum_type = spectra_normalize_spectrum_type(spectrum_type)
    return os.path.join(SPECTRA_DESCRIPTOR_SHARDS_DIR, spectrum_type)


def spectral_descriptor_shard_key(inchikey):
    value = str(inchikey or "").strip().upper()

    if value in ["", "NAN", "NONE", "NULL"]:
        return ""

    value = re.sub(r"[^A-Z0-9]+", "", value)

    if len(value) < 2:
        return ""

    return value[:2]


def spectral_descriptor_shard_file(spectrum_type="IR", shard_key=""):
    shard_key = str(shard_key or "").strip().upper()

    if not shard_key:
        return ""

    return os.path.join(
        spectral_descriptor_shard_type_dir(spectrum_type),
        f"{shard_key}.csv"
    )


def spectral_remote_descriptor_shard_url(spectrum_type="IR", shard_key=""):
    base_url = str(
        SPECTRA_REMOTE_DESCRIPTOR_SHARDS_BASE_URL
        or SPECTRA_DEFAULT_DESCRIPTOR_SHARDS_BASE_URL
        or ""
    ).strip().rstrip("/")
    shard_key = str(shard_key or "").strip().upper()

    if not base_url or not shard_key:
        return ""

    spectrum_type = spectra_normalize_spectrum_type(spectrum_type)

    return f"{base_url}/{spectrum_type}/{shard_key}.csv"


def spectral_descriptor_cache_settings(
    spectrum_type="IR",
    wn_min=550,
    wn_max=3798,
    step=4,
    normalization="min-max",
    invert_signal=False,
    use_grid=True,
    use_binary_fp=True,
    use_binned_numeric=True,
    binary_window=20,
    binary_threshold=0.10,
    numeric_window=100,
):
    """
    Settings that make cached per-spectrum descriptors comparable.

    SVD is intentionally not included: SVD descriptors are dataset-level
    components and cannot be reused as a property of one spectrum.
    """
    return {
        "descriptor_spectrum_type": spectra_normalize_spectrum_type(spectrum_type),
        "descriptor_wn_min": float(wn_min),
        "descriptor_wn_max": float(wn_max),
        "descriptor_step": float(step),
        "descriptor_normalization": str(normalization),
        "descriptor_invert_signal": bool(invert_signal),
        "descriptor_use_grid": bool(use_grid),
        "descriptor_use_binary_fp": bool(use_binary_fp),
        "descriptor_use_binned_numeric": bool(use_binned_numeric),
        "descriptor_binary_window": float(binary_window),
        "descriptor_binary_threshold": float(binary_threshold),
        "descriptor_numeric_window": float(numeric_window),
    }


def spectral_descriptor_cache_required_cols():
    return [
        "spectrum_id",
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "descriptor_spectrum_type",
        "descriptor_wn_min",
        "descriptor_wn_max",
        "descriptor_step",
        "descriptor_normalization",
        "descriptor_invert_signal",
        "descriptor_use_grid",
        "descriptor_use_binary_fp",
        "descriptor_use_binned_numeric",
        "descriptor_binary_window",
        "descriptor_binary_threshold",
        "descriptor_numeric_window",
        "descriptor_cached_at",
    ]


def spectral_load_descriptor_cache(spectrum_type="IR"):
    """
    Loads ready per-spectrum descriptors.
    """
    cache_file = spectral_descriptor_cache_file(spectrum_type)

    if not os.path.exists(cache_file):
        legacy_cache_file = spectral_legacy_descriptor_cache_file(spectrum_type)

        if os.path.exists(legacy_cache_file):
            cache_file = legacy_cache_file

    if not os.path.exists(cache_file):
        remote_url = spectral_remote_descriptor_cache_url(spectrum_type)

        if remote_url:
            try:
                spectra_download_public_file(remote_url, cache_file, timeout=90)
            except Exception:
                pass

    if not os.path.exists(cache_file):
        rel_path = spectra_normalize_bank_relative_path(cache_file)
        remote_path = spectra_materialize_remote_bank_file(rel_path)

        if remote_path and os.path.exists(remote_path):
            cache_file = remote_path

    if os.path.exists(cache_file):
        try:
            cache_df = pd.read_csv(cache_file, low_memory=False)
        except Exception:
            cache_df = pd.DataFrame()
    else:
        cache_df = pd.DataFrame()

    for col in spectral_descriptor_cache_required_cols():
        if col not in cache_df.columns:
            cache_df[col] = ""

    return cache_df


def spectral_load_descriptor_cache_for_inchikeys(spectrum_type="IR", inchikeys=None):
    """
    Loads only descriptor shards needed for a set of InChIKey values.
    Falls back to the full descriptor cache if no local/remote shards are found.
    """
    inchikeys = list(inchikeys or [])
    shard_keys = sorted({
        spectral_descriptor_shard_key(x)
        for x in inchikeys
        if spectral_descriptor_shard_key(x)
    })

    if not shard_keys:
        return pd.DataFrame()

    frames = []
    attempted_shards = 0

    for shard_key in shard_keys:
        shard_file = spectral_descriptor_shard_file(spectrum_type, shard_key)
        attempted_shards += 1

        if not os.path.exists(shard_file):
            shard_url = spectral_remote_descriptor_shard_url(spectrum_type, shard_key)

            if shard_url:
                try:
                    spectra_download_public_file(shard_url, shard_file, timeout=90)
                except Exception:
                    pass

        if not os.path.exists(shard_file):
            continue

        try:
            shard_df = pd.read_csv(shard_file, low_memory=False)
        except Exception:
            continue

        if shard_df.empty:
            continue

        frames.append(shard_df)

    if frames:
        cache_df = pd.concat(frames, ignore_index=True)

        for col in spectral_descriptor_cache_required_cols():
            if col not in cache_df.columns:
                cache_df[col] = ""

        return cache_df

    if attempted_shards > 0:
        return spectral_load_descriptor_cache(spectrum_type)

    return pd.DataFrame()


def spectral_save_descriptor_cache(cache_df, spectrum_type="IR"):
    """
    Saves the per-spectrum descriptor cache.
    """
    if cache_df is None:
        cache_df = pd.DataFrame()

    cache_file = spectral_descriptor_cache_file(spectrum_type)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    with SPECTRA_DESCRIPTOR_CACHE_LOCK:
        cache_df.to_csv(cache_file, index=False, encoding="utf-8-sig")


def spectral_cache_value_matches(actual, expected):
    if isinstance(expected, bool):
        actual_norm = str(actual).strip().lower()
        return actual_norm in ["true", "1", "yes", "y"] if expected else actual_norm in ["false", "0", "no", "n", ""]

    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= 1e-9
        except Exception:
            return False

    return str(actual).strip() == str(expected).strip()


def spectral_filter_descriptor_cache_by_settings(cache_df, settings):
    if cache_df is None or cache_df.empty:
        return pd.DataFrame()

    work = cache_df.copy()

    for col, expected in settings.items():
        if col not in work.columns:
            return pd.DataFrame()

        mask = work[col].apply(lambda actual: spectral_cache_value_matches(actual, expected))
        work = work[mask].copy()

        if work.empty:
            return pd.DataFrame()

    return work


def spectral_find_cached_descriptor_row(spectrum_record, descriptor_settings=None):
    """
    Finds a cached descriptor row for a spectrum record and calculation settings.
    """
    if spectrum_record is None:
        return None

    spectrum_type = spectra_normalize_spectrum_type(
        spectrum_record.get("spectrum_type", "")
        or spectrum_record.get("descriptor_spectrum_type", "")
        or "IR"
    )

    spectrum_id = str(spectrum_record.get("spectrum_id", "")).strip()
    inchikey = str(spectrum_record.get("inchikey", "")).strip()
    canonical_smiles = str(spectrum_record.get("canonical_smiles", "")).strip()

    cache_df = spectral_load_descriptor_cache_for_inchikeys(
        spectrum_type,
        [inchikey]
    )

    if cache_df.empty:
        return None

    if descriptor_settings:
        cache_df = spectral_filter_descriptor_cache_by_settings(
            cache_df,
            descriptor_settings
        )

        if cache_df.empty:
            return None

    found = pd.DataFrame()

    if spectrum_id and "spectrum_id" in cache_df.columns:
        found = cache_df[cache_df["spectrum_id"].astype(str).str.strip() == spectrum_id].copy()

    if found.empty and inchikey and "inchikey" in cache_df.columns:
        found = cache_df[cache_df["inchikey"].astype(str).str.strip() == inchikey].copy()

    if found.empty and canonical_smiles and "canonical_smiles" in cache_df.columns:
        found = cache_df[
            cache_df["canonical_smiles"].astype(str).str.strip() == canonical_smiles
        ].copy()

    if found.empty:
        return None

    return found.iloc[-1].to_dict()


def spectral_make_descriptor_row_metadata(compound, spectrum_record, processed_file):
    return {
        "row_index": compound.get("row_index", ""),
        "compound_id": compound.get("compound_id", ""),
        "name": compound.get("name", ""),
        "input_smiles": compound.get("input_smiles", ""),
        "canonical_smiles": compound.get("canonical_smiles", ""),
        "inchikey": compound.get("inchikey", ""),
        "spectrum_type": spectra_normalize_spectrum_type(
            spectrum_record.get("spectrum_type", "")
        ),
        "spectrum_id": spectrum_record.get("spectrum_id", ""),
        "spectrum_source": spectrum_record.get("source", ""),
        "spectrum_source_database": spectrum_record.get("source_database", ""),
        "spectrum_phase": spectrum_record.get("phase", ""),
        "spectrum_phase_norm": spectrum_record.get("spectrum_phase_norm", ""),
        "spectrum_intensity_type": spectrum_record.get("intensity_type", ""),
        "spectrum_is_experimental": spectrum_record.get("is_experimental", ""),
        "spectrum_is_quantitative": spectrum_record.get("is_quantitative", ""),
        "spectrum_selection_reason": spectrum_record.get("spectrum_selection_reason", ""),
        "processed_file": processed_file,
    }


def spectral_descriptor_columns_from_row(row, spectrum_type="IR"):
    prefix = spectra_normalize_spectrum_type(spectrum_type)
    descriptor_prefixes = (
        f"{prefix}_GRID_",
        f"{prefix}_BIN_",
        f"{prefix}_BAND_",
    )

    return [
        c for c in row.keys()
        if any(str(c).startswith(p) for p in descriptor_prefixes)
    ]


def spectral_prepare_cached_descriptor_row(cached_row, compound, spectrum_record, processed_file):
    """
    Converts a cached per-spectrum descriptor row to the current dataset row.
    """
    if cached_row is None:
        return None

    spectrum_type = spectra_normalize_spectrum_type(
        spectrum_record.get("spectrum_type", "")
    )
    metadata = spectral_make_descriptor_row_metadata(
        compound,
        spectrum_record,
        processed_file
    )

    desc = dict(metadata)

    for col in spectral_descriptor_columns_from_row(cached_row, spectrum_type=spectrum_type):
        value = cached_row.get(col, np.nan)
        desc[col] = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if len(desc) == len(metadata):
        return None

    desc["spectral_descriptor_source"] = "cached_descriptor"

    return desc


def spectral_values_norm_from_descriptor_row(desc, spectrum_type="IR"):
    """
    Rebuilds a normalized grid vector from GRID descriptors when available.
    """
    prefix = f"{spectra_normalize_spectrum_type(spectrum_type)}_GRID_"
    pairs = []

    for col, value in desc.items():
        col_str = str(col)

        if not col_str.startswith(prefix):
            continue

        try:
            wn = int(col_str.replace(prefix, "", 1))
            val = float(value)
        except Exception:
            continue

        pairs.append((wn, val))

    if not pairs:
        return None

    pairs = sorted(pairs, key=lambda x: x[0])
    return np.asarray([p[1] for p in pairs], dtype=float)


def spectral_prepare_descriptor_cache_row(row, descriptor_settings, spectrum_type="IR"):
    """
    Prepares one row for the per-spectrum descriptor cache.
    """
    if row is None or not isinstance(row, dict):
        return None

    descriptor_cols = spectral_descriptor_columns_from_row(row, spectrum_type=spectrum_type)

    if not descriptor_cols:
        return None

    cache_row = dict(row)
    cache_row.update(descriptor_settings or {})
    cache_row["descriptor_cached_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    drop_cols = ["row_index", "compound_id", "name", "input_smiles"]

    for col in drop_cols:
        cache_row.pop(col, None)

    return cache_row


def spectral_descriptor_cache_dedupe_columns(cache_df):
    return [
        c for c in [
            "spectrum_id",
            "inchikey",
            "descriptor_spectrum_type",
            "descriptor_wn_min",
            "descriptor_wn_max",
            "descriptor_step",
            "descriptor_normalization",
            "descriptor_invert_signal",
            "descriptor_use_grid",
            "descriptor_use_binary_fp",
            "descriptor_use_binned_numeric",
            "descriptor_binary_window",
            "descriptor_binary_threshold",
            "descriptor_numeric_window",
        ]
        if c in cache_df.columns
    ]


def spectral_merge_descriptor_cache_rows(rows, descriptor_settings, spectrum_type="IR"):
    """
    Merges ready descriptor rows into the existing cache and saves it immediately.
    """
    if rows is None:
        rows = []

    prepared_rows = []

    for row in rows:
        prepared = spectral_prepare_descriptor_cache_row(
            row,
            descriptor_settings,
            spectrum_type=spectrum_type
        )

        if prepared is not None:
            prepared_rows.append(prepared)

    cache_df = spectral_load_descriptor_cache(spectrum_type)

    before_rows = int(len(cache_df))

    if not prepared_rows:
        return cache_df, {
            "cache_rows_before": before_rows,
            "cache_rows_added": 0,
            "cache_rows_after": before_rows,
        }

    cache_row_df = pd.DataFrame(prepared_rows)

    if cache_df.empty:
        cache_df = cache_row_df
    else:
        cache_df = pd.concat([cache_df, cache_row_df], ignore_index=True)

    dedupe_cols = spectral_descriptor_cache_dedupe_columns(cache_df)

    if dedupe_cols:
        cache_df = cache_df.drop_duplicates(subset=dedupe_cols, keep="last")

    spectral_save_descriptor_cache(cache_df, spectrum_type=spectrum_type)

    after_rows = int(len(cache_df))

    return cache_df, {
        "cache_rows_before": before_rows,
        "cache_rows_added": len(prepared_rows),
        "cache_rows_after": after_rows,
    }


def spectral_update_descriptor_cache(row, descriptor_settings, spectrum_type="IR"):
    """
    Adds or replaces one ready descriptor row in the per-spectrum cache.
    """
    spectral_merge_descriptor_cache_rows(
        [row],
        descriptor_settings,
        spectrum_type=spectrum_type
    )


def spectral_spectrum_record_from_cached_descriptor_row(cached_row, spectrum_type="IR"):
    """
    Builds a spectrum-like record from a cached descriptor row.
    """
    cached_row = dict(cached_row or {})

    phase = (
        cached_row.get("spectrum_phase", "")
        or cached_row.get("phase", "")
        or cached_row.get("spectrum_phase_norm", "")
    )

    source = (
        cached_row.get("spectrum_source", "")
        or cached_row.get("source", "")
    )

    source_database = (
        cached_row.get("spectrum_source_database", "")
        or cached_row.get("source_database", "")
    )

    intensity_type = (
        cached_row.get("spectrum_intensity_type", "")
        or cached_row.get("intensity_type", "")
    )

    return {
        "spectrum_id": cached_row.get("spectrum_id", ""),
        "inchikey": cached_row.get("inchikey", ""),
        "canonical_smiles": cached_row.get("canonical_smiles", ""),
        "spectrum_type": spectra_normalize_spectrum_type(
            cached_row.get("spectrum_type", "")
            or cached_row.get("descriptor_spectrum_type", "")
            or spectrum_type
        ),
        "source": source,
        "source_database": source_database,
        "phase": phase,
        "spectrum_phase_norm": (
            cached_row.get("spectrum_phase_norm", "")
            or spectral_normalize_phase_value(phase)
        ),
        "intensity_type": intensity_type,
        "is_experimental": (
            cached_row.get("spectrum_is_experimental", "")
            or cached_row.get("is_experimental", "")
        ),
        "is_quantitative": (
            cached_row.get("spectrum_is_quantitative", "")
            or cached_row.get("is_quantitative", "")
        ),
        "spectrum_selection_reason": (
            cached_row.get("spectrum_selection_reason", "")
            or "selected_ready_descriptor"
        ),
        "processed_file": cached_row.get("processed_file", ""),
    }


def spectral_filter_ready_descriptor_candidates(
    cache_df,
    phase_mode="prefer_gas",
    allowed_phases=None,
    allowed_sources=None,
    allowed_intensity_types=None,
    prefer_quantitative=False,
    experimental_only=True,
):
    """
    Applies the same practical filters to ready descriptors where metadata exists.
    """
    if cache_df is None or cache_df.empty:
        return pd.DataFrame()

    work = cache_df.copy()

    for col in [
        "spectrum_phase",
        "phase",
        "spectrum_source",
        "source",
        "spectrum_intensity_type",
        "intensity_type",
        "spectrum_is_experimental",
        "is_experimental",
        "spectrum_is_quantitative",
        "is_quantitative",
        "spectrum_id",
    ]:
        if col not in work.columns:
            work[col] = ""

    phase_values = work["spectrum_phase"].where(
        work["spectrum_phase"].astype(str).str.strip() != "",
        work["phase"]
    )
    source_values = work["spectrum_source"].where(
        work["spectrum_source"].astype(str).str.strip() != "",
        work["source"]
    )
    intensity_values = work["spectrum_intensity_type"].where(
        work["spectrum_intensity_type"].astype(str).str.strip() != "",
        work["intensity_type"]
    )
    experimental_values = work["spectrum_is_experimental"].where(
        work["spectrum_is_experimental"].astype(str).str.strip() != "",
        work["is_experimental"]
    )
    quantitative_values = work["spectrum_is_quantitative"].where(
        work["spectrum_is_quantitative"].astype(str).str.strip() != "",
        work["is_quantitative"]
    )

    work["_phase_norm"] = phase_values.apply(spectral_normalize_phase_value)
    work["_phase_priority"] = work["_phase_norm"].apply(spectral_phase_priority_score)
    work["_source_norm"] = source_values.apply(spectral_normalize_source_value)
    work["_intensity_type_norm"] = (
        intensity_values.astype(str).str.strip().str.lower().replace("", "unknown")
    )

    if experimental_only:
        exp_norm = experimental_values.astype(str).str.lower().str.strip()
        experimental_ok = ["true", "1", "yes", "y", "РґР°", "experimental", ""]
        work = work[exp_norm.isin(experimental_ok)].copy()

        if work.empty:
            return pd.DataFrame()

    if allowed_sources is not None:
        allowed_sources_norm = [
            spectral_normalize_source_value(x)
            for x in allowed_sources
        ]
        work = work[work["_source_norm"].isin(allowed_sources_norm)].copy()

        if work.empty:
            return pd.DataFrame()

    if allowed_intensity_types is not None:
        allowed_intensity_types_norm = [
            str(x).strip().lower()
            for x in allowed_intensity_types
        ]
        work = work[work["_intensity_type_norm"].isin(allowed_intensity_types_norm)].copy()

        if work.empty:
            return pd.DataFrame()

    phase_mode = str(phase_mode).strip().lower()

    if phase_mode == "only_gas":
        work = work[work["_phase_norm"] == "gas"].copy()

        if work.empty:
            return pd.DataFrame()

    elif phase_mode == "manual":
        if allowed_phases is None or len(allowed_phases) == 0:
            return pd.DataFrame()

        allowed_phases_norm = [
            spectral_normalize_phase_value(x)
            for x in allowed_phases
        ]
        work = work[work["_phase_norm"].isin(allowed_phases_norm)].copy()

        if work.empty:
            return pd.DataFrame()

    if "confidence_level" in work.columns:
        conf = work["confidence_level"].astype(str).str.lower().str.strip()
        confidence_priority_map = {
            "high": 0,
            "spectral_page_found": 10,
            "medium": 20,
            "low": 40,
            "error": 90,
            "": 50,
        }
        work["_confidence_priority"] = conf.map(confidence_priority_map).fillna(50)
    else:
        work["_confidence_priority"] = 50

    quant_norm = quantitative_values.astype(str).str.lower().str.strip()
    work["_quant_priority"] = np.where(
        quant_norm.isin(["true", "1", "yes", "y", "РґР°"]),
        0 if prefer_quantitative else 1,
        1 if prefer_quantitative else 0
    )

    return work.sort_values(
        by=[
            "_phase_priority",
            "_quant_priority",
            "_confidence_priority",
            "spectrum_id",
        ],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)


def spectral_build_descriptors_from_ready_cache_for_dataset(
    input_df,
    smiles_col="SMILES",
    spectrum_type="IR",
    wn_min=550,
    wn_max=3798,
    step=4,
    normalization="min-max",
    invert_signal=False,
    use_grid=True,
    use_binary_fp=True,
    use_binned_numeric=True,
    binary_window=20,
    binary_threshold=0.10,
    numeric_window=100,
    use_svd=True,
    svd_components=10,
    spectrum_phase_mode="prefer_gas",
    allowed_phases=None,
    allowed_sources=None,
    allowed_intensity_types=None,
    prefer_quantitative=False,
    experimental_only=True,
    progress_callback=None
):
    """
    Builds a dataset descriptor table only from ready per-spectrum descriptors.
    No processed spectra are downloaded or read.
    """
    compounds = spectra_prepare_compounds_from_df(input_df, smiles_col=smiles_col)
    descriptor_settings = spectral_descriptor_cache_settings(
        spectrum_type=spectrum_type,
        wn_min=wn_min,
        wn_max=wn_max,
        step=step,
        normalization=normalization,
        invert_signal=invert_signal,
        use_grid=use_grid,
        use_binary_fp=use_binary_fp,
        use_binned_numeric=use_binned_numeric,
        binary_window=binary_window,
        binary_threshold=binary_threshold,
        numeric_window=numeric_window,
    )

    target_inchikeys = []

    if "inchikey" in compounds.columns:
        target_inchikeys = compounds["inchikey"].astype(str).str.strip().tolist()

    cache_df = spectral_load_descriptor_cache_for_inchikeys(
        spectrum_type,
        target_inchikeys
    )
    cache_df = spectral_filter_descriptor_cache_by_settings(
        cache_df,
        descriptor_settings
    )
    cache_df = spectral_filter_ready_descriptor_candidates(
        cache_df,
        phase_mode=spectrum_phase_mode,
        allowed_phases=allowed_phases,
        allowed_sources=allowed_sources,
        allowed_intensity_types=allowed_intensity_types,
        prefer_quantitative=prefer_quantitative,
        experimental_only=experimental_only,
    )

    rows = []
    matrix_rows = []
    matrix_meta = []

    report = {
        "total_compounds": len(compounds),
        "with_spectrum": 0,
        "without_spectrum": 0,
        "parse_errors": 0,
        "used_spectra": [],
        "used_phases": {},
        "spectrum_selection_reasons": {},
        "descriptor_cache_hits": 0,
        "descriptor_cache_misses": 0,
        "ready_descriptor_cache_rows": int(len(cache_df)),
        "ready_descriptor_mode": True,
        "spectra_bank_status": spectra_remote_bank_status(),
    }

    total_compounds = len(compounds)

    for processed_count, (_, compound) in enumerate(compounds.iterrows(), start=1):
        inchikey = str(compound.get("inchikey", "")).strip()
        canonical_smiles = str(compound.get("canonical_smiles", "")).strip()

        if progress_callback is not None:
            try:
                progress_callback(
                    processed_count - 1,
                    total_compounds,
                    "loading_ready_descriptors",
                    {
                        "inchikey": inchikey,
                        "canonical_smiles": canonical_smiles,
                    }
                )
            except Exception:
                pass

        found = pd.DataFrame()

        if not cache_df.empty and inchikey and "inchikey" in cache_df.columns:
            found = cache_df[
                cache_df["inchikey"].astype(str).str.strip() == inchikey
            ].copy()

        if found.empty and not cache_df.empty and canonical_smiles and "canonical_smiles" in cache_df.columns:
            found = cache_df[
                cache_df["canonical_smiles"].astype(str).str.strip() == canonical_smiles
            ].copy()

        if found.empty:
            report["without_spectrum"] += 1
            report["descriptor_cache_misses"] += 1

            if progress_callback is not None:
                try:
                    progress_callback(
                        processed_count,
                        total_compounds,
                        "no_ready_descriptors",
                        {
                            "inchikey": inchikey,
                            "canonical_smiles": canonical_smiles,
                        }
                    )
                except Exception:
                    pass

            continue

        cached_row = found.iloc[0].to_dict()
        spectrum_record = spectral_spectrum_record_from_cached_descriptor_row(
            cached_row,
            spectrum_type=spectrum_type
        )
        desc = spectral_prepare_cached_descriptor_row(
            cached_row,
            compound,
            spectrum_record,
            spectrum_record.get("processed_file", "")
        )

        if desc is None:
            report["parse_errors"] += 1
            continue

        rows.append(desc)
        report["with_spectrum"] += 1
        report["descriptor_cache_hits"] += 1
        report["used_spectra"].append(spectrum_record.get("spectrum_id", ""))

        values_norm_cached = spectral_values_norm_from_descriptor_row(
            desc,
            spectrum_type=spectrum_type
        )

        if values_norm_cached is not None:
            matrix_rows.append(values_norm_cached)
            matrix_meta.append({
                "row_index": compound.get("row_index", ""),
                "compound_id": compound.get("compound_id", ""),
                "name": compound.get("name", ""),
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "spectrum_id": spectrum_record.get("spectrum_id", ""),
            })

        used_phase = str(
            spectrum_record.get("spectrum_phase_norm", "unknown")
        ).strip() or "unknown"
        report["used_phases"][used_phase] = report["used_phases"].get(used_phase, 0) + 1

        selection_reason = str(
            spectrum_record.get("spectrum_selection_reason", "selected_ready_descriptor")
        ).strip() or "selected_ready_descriptor"
        report["spectrum_selection_reasons"][selection_reason] = (
            report["spectrum_selection_reasons"].get(selection_reason, 0) + 1
        )

        if progress_callback is not None:
            try:
                progress_callback(
                    processed_count,
                    total_compounds,
                    "done",
                    {
                        "inchikey": inchikey,
                        "canonical_smiles": canonical_smiles,
                        "spectrum_id": spectrum_record.get("spectrum_id", ""),
                        "descriptor_source": "ready_descriptor_cache",
                    }
                )
            except Exception:
                pass

    descriptors_df = pd.DataFrame(rows)

    if use_svd and len(matrix_rows) >= 2 and not descriptors_df.empty:
        X_spec = np.vstack(matrix_rows)
        max_components = min(
            int(svd_components),
            X_spec.shape[0] - 1,
            X_spec.shape[1] - 1
        )

        if max_components >= 1:
            try:
                svd = TruncatedSVD(n_components=max_components, random_state=42)
                svd_values = svd.fit_transform(X_spec)
                svd_df = pd.DataFrame(matrix_meta)
                prefix = spectra_normalize_spectrum_type(spectrum_type)

                for i in range(max_components):
                    svd_df[f"{prefix}_SVD_{i + 1}"] = svd_values[:, i]

                descriptors_df = descriptors_df.merge(
                    svd_df,
                    on=[
                        "row_index",
                        "compound_id",
                        "name",
                        "canonical_smiles",
                        "inchikey",
                        "spectrum_id",
                    ],
                    how="left"
                )
                report["svd_components_created"] = max_components
                report["svd_explained_variance_sum"] = float(
                    np.sum(svd.explained_variance_ratio_)
                )
            except Exception as e:
                report["svd_error"] = str(e)

    report["without_spectrum"] = (
        report["total_compounds"]
        - report["with_spectrum"]
        - report["parse_errors"]
    )

    if report["without_spectrum"] < 0:
        report["without_spectrum"] = 0

    if descriptors_df.empty:
        return descriptors_df, report

    return descriptors_df.reset_index(drop=True), report


def spectral_build_descriptor_cache_for_all_indexed_spectra(
    spectrum_type="IR",
    wn_min=550,
    wn_max=3798,
    step=4,
    normalization="min-max",
    invert_signal=False,
    use_grid=True,
    use_binary_fp=True,
    use_binned_numeric=True,
    binary_window=20,
    binary_threshold=0.10,
    numeric_window=100,
    active_only=True,
    skip_existing_inchikey=True,
    autosave_every=10,
    progress_callback=None
):
    """
    Calculates ready per-spectrum descriptors for all indexed local spectra.
    The result is appended to the project descriptor cache file.
    """
    spectrum_type_norm = spectra_normalize_spectrum_type(spectrum_type)
    descriptor_settings = spectral_descriptor_cache_settings(
        spectrum_type=spectrum_type_norm,
        wn_min=wn_min,
        wn_max=wn_max,
        step=step,
        normalization=normalization,
        invert_signal=invert_signal,
        use_grid=use_grid,
        use_binary_fp=use_binary_fp,
        use_binned_numeric=use_binned_numeric,
        binary_window=binary_window,
        binary_threshold=binary_threshold,
        numeric_window=numeric_window,
    )

    index_df = spectra_load_index()

    report = {
        "spectrum_type": spectrum_type_norm,
        "total_index_rows": 0,
        "processed": 0,
        "cached": 0,
        "missing_processed_file": 0,
        "parse_errors": 0,
        "empty_grid": 0,
        "skipped_existing_inchikey": 0,
        "cache_rows_before": 0,
        "cache_rows_added": 0,
        "cache_rows_after": 0,
        "autosave_every": int(autosave_every or 0),
        "autosave_count": 0,
        "cache_file": spectral_descriptor_cache_file(spectrum_type_norm),
        "descriptor_settings": descriptor_settings,
    }

    if index_df.empty:
        return pd.DataFrame(), report

    existing_cache_df = spectral_load_descriptor_cache(spectrum_type_norm)
    report["cache_rows_before"] = int(len(existing_cache_df))

    existing_inchikeys = set()

    if (
        skip_existing_inchikey
        and existing_cache_df is not None
        and not existing_cache_df.empty
        and "inchikey" in existing_cache_df.columns
    ):
        existing_inchikeys = set(
            existing_cache_df["inchikey"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace("", np.nan)
            .dropna()
            .tolist()
        )

    work = index_df.copy()

    for col in [
        "spectrum_type",
        "active",
        "processed_file",
        "spectrum_id",
        "canonical_smiles",
        "inchikey",
        "compound_id",
        "name",
        "source",
        "source_database",
        "phase",
        "intensity_type",
        "is_experimental",
        "is_quantitative",
    ]:
        if col not in work.columns:
            work[col] = ""

    work["_spectrum_type_norm"] = work["spectrum_type"].apply(
        spectra_normalize_spectrum_type
    )
    work = work[work["_spectrum_type_norm"] == spectrum_type_norm].copy()

    if active_only:
        active_values = ["true", "1", "yes", "y", "РґР°", "active", ""]
        work = work[
            work["active"].astype(str).str.lower().str.strip().isin(active_values)
        ].copy()

    work = work.reset_index(drop=True)
    report["total_index_rows"] = int(len(work))
    pending_rows = []
    cache_df = existing_cache_df
    autosave_every = max(int(autosave_every or 0), 1)
    total = len(work)

    def _flush_pending_rows():
        nonlocal pending_rows, cache_df

        if not pending_rows:
            return

        cache_df, merge_report = spectral_merge_descriptor_cache_rows(
            pending_rows,
            descriptor_settings,
            spectrum_type=spectrum_type_norm
        )

        report["cache_rows_added"] += int(merge_report.get("cache_rows_added", 0) or 0)
        report["cache_rows_after"] = int(merge_report.get("cache_rows_after", len(cache_df)) or 0)
        report["autosave_count"] += 1
        pending_rows = []

    for pos, (_, spectrum_record_row) in enumerate(work.iterrows(), start=1):
        spectrum_record = spectrum_record_row.to_dict()
        processed_file = str(spectrum_record.get("processed_file", "")).strip()
        spectrum_inchikey = str(spectrum_record.get("inchikey", "")).strip().upper()

        if spectrum_inchikey in ["", "NAN", "NONE", "NULL"]:
            spectrum_inchikey = ""

        if not spectrum_inchikey and processed_file:
            filename_inchikey, filename_inchikey_mode = spectra_extract_inchikey_from_filename(
                processed_file
            )

            if filename_inchikey_mode == "full":
                spectrum_inchikey = str(filename_inchikey).strip().upper()
                spectrum_record["inchikey"] = spectrum_inchikey

        if skip_existing_inchikey and spectrum_inchikey and spectrum_inchikey in existing_inchikeys:
            report["skipped_existing_inchikey"] += 1

            if progress_callback is not None:
                try:
                    progress_callback(
                        pos,
                        total,
                        "skipped_existing_inchikey",
                        {
                            "spectrum_id": spectrum_record.get("spectrum_id", ""),
                            "processed_file": processed_file,
                            "inchikey": spectrum_inchikey,
                        }
                    )
                except Exception:
                    pass

            continue

        if progress_callback is not None:
            try:
                progress_callback(
                    pos - 1,
                    total,
                    "loading_spectrum",
                    {
                        "spectrum_id": spectrum_record.get("spectrum_id", ""),
                        "processed_file": processed_file,
                    }
                )
            except Exception:
                pass

        real_processed_path = spectral_resolve_processed_spectrum_path(
            processed_file,
            spectrum_record=spectrum_record,
            allow_remote=False
        )

        if not real_processed_path:
            report["missing_processed_file"] += 1
            continue

        spectrum_df = spectral_load_processed_spectrum(
            real_processed_path,
            spectrum_record=spectrum_record
        )

        if spectrum_df.empty:
            report["parse_errors"] += 1
            continue

        grid, values = spectral_interpolate_to_grid(
            spectrum_df,
            wn_min=wn_min,
            wn_max=wn_max,
            step=step
        )

        if len(grid) == 0:
            report["empty_grid"] += 1
            continue

        values_norm = spectral_normalize_values(
            values,
            method=normalization,
            invert=invert_signal
        )

        compound = {
            "row_index": "",
            "compound_id": spectrum_record.get("compound_id", ""),
            "name": spectrum_record.get("name", ""),
            "input_smiles": spectrum_record.get("canonical_smiles", ""),
            "canonical_smiles": spectrum_record.get("canonical_smiles", ""),
            "inchikey": spectrum_record.get("inchikey", ""),
        }

        desc = spectral_make_descriptor_row_metadata(
            compound,
            spectrum_record,
            processed_file
        )

        if use_grid:
            desc.update(
                spectral_make_grid_descriptors(
                    grid,
                    values_norm,
                    prefix=f"{spectrum_type_norm}_GRID"
                )
            )

        if use_binary_fp:
            desc.update(
                spectral_make_binary_fp(
                    grid,
                    values_norm,
                    window_size=binary_window,
                    threshold=binary_threshold,
                    prefix=f"{spectrum_type_norm}_BIN"
                )
            )

        if use_binned_numeric:
            desc.update(
                spectral_make_binned_numeric_descriptors(
                    grid,
                    values_norm,
                    window_size=numeric_window,
                    prefix=f"{spectrum_type_norm}_BAND"
                )
            )

        pending_rows.append(desc)
        report["processed"] += 1
        report["cached"] += 1

        if spectrum_inchikey:
            existing_inchikeys.add(spectrum_inchikey)

        if progress_callback is not None:
            try:
                progress_callback(
                    pos,
                    total,
                    "done",
                    {
                        "spectrum_id": spectrum_record.get("spectrum_id", ""),
                        "processed_file": processed_file,
                    }
                )
            except Exception:
                pass

        if len(pending_rows) >= autosave_every:
            _flush_pending_rows()

            if progress_callback is not None:
                try:
                    progress_callback(
                        pos,
                        total,
                        "autosaved",
                        {
                            "cache_file": report.get("cache_file", ""),
                            "cache_rows_after": report.get("cache_rows_after", 0),
                            "autosave_count": report.get("autosave_count", 0),
                        }
                    )
                except Exception:
                    pass

    _flush_pending_rows()

    if report["cache_rows_after"] == 0:
        report["cache_rows_after"] = int(len(cache_df))

    return cache_df, report


def spectral_get_active_spectrum_for_compound(inchikey, canonical_smiles="", spectrum_type="IR"):
    """
    Возвращает активную запись спектра для вещества.
    """
    return spectra_find_in_bank(
        inchikey=inchikey,
        canonical_smiles=canonical_smiles,
        spectrum_type=spectrum_type
    )

def spectral_normalize_phase_value(phase):
    """
    Нормализует фазу / состояние образца для выбора спектров.

    Примеры:
    gas, vapor, vapour -> gas
    liquid -> liquid
    solution -> solution
    KBr, kbr pellet -> kbr
    Nujol -> nujol
    film -> film
    solid -> solid
    """
    x = str(phase).strip().lower()

    if x in ["", "nan", "none"]:
        return "unknown"

    if "gas" in x:
        return "gas"

    if "vapor" in x or "vapour" in x:
        return "gas"

    if "liquid" in x:
        return "liquid"

    if "solution" in x:
        return "solution"

    if "film" in x:
        return "film"

    if "kbr" in x:
        return "kbr"

    if "nujol" in x:
        return "nujol"

    if "solid" in x:
        return "solid"

    return spectra_safe_filename_part(x).lower()
    
def spectral_phase_priority_score(phase):
    """
    Возвращает числовой приоритет фазы.
    Чем меньше число, тем выше приоритет.

    По умолчанию предпочитаем газофазные спектры.
    """
    phase_norm = spectral_normalize_phase_value(phase)

    priority = {
        "gas": 0,
        "liquid": 10,
        "film": 20,
        "solution": 30,
        "solid": 40,
        "kbr": 50,
        "nujol": 60,
        "unknown": 90,
    }

    return priority.get(phase_norm, 80)
    
def spectral_normalize_source_value(source):
    """
    Нормализует название источника спектра для фильтрации.
    """
    x = str(source).strip()

    if x == "" or x.lower() in ["nan", "none"]:
        return "unknown"

    return x
    
def spectral_find_candidate_spectra_for_compound(
    inchikey,
    canonical_smiles="",
    spectrum_type="IR"
):
    """
    Возвращает все активные спектры вещества из spectra_bank
    для заданного типа спектра.

    В отличие от spectra_find_in_bank, не возвращает первый найденный,
    а отдаёт таблицу кандидатов.
    """
    index_df = spectra_load_index()

    if index_df.empty:
        return pd.DataFrame()

    work = index_df.copy()

    required_cols = [
        "inchikey",
        "canonical_smiles",
        "spectrum_type",
        "active",
        "processed_file",
        "phase",
        "source",
        "source_database",
        "intensity_type",
        "is_experimental",
        "is_quantitative",
        "confidence_level",
        "status",
        "spectrum_id",
    ]

    for col in required_cols:
        if col not in work.columns:
            work[col] = ""

    work["inchikey"] = work["inchikey"].astype(str).str.strip()
    work["canonical_smiles"] = work["canonical_smiles"].astype(str).str.strip()
    work["spectrum_type"] = work["spectrum_type"].astype(str).str.strip()
    work["active"] = work["active"].astype(str).str.strip()
    work["processed_file"] = work["processed_file"].astype(str).str.strip()

    work["_spectrum_type_norm"] = work["spectrum_type"].apply(
        spectra_normalize_spectrum_type
    )

    requested_type = spectra_normalize_spectrum_type(spectrum_type)

    work = work[
        work["_spectrum_type_norm"] == requested_type
    ].copy()

    if work.empty:
        return pd.DataFrame()

    active_values = ["true", "1", "yes", "y", "да", "active", ""]
    work["_active_norm"] = (
        work["active"]
        .astype(str)
        .str.lower()
        .isin(active_values)
    )

    work = work[work["_active_norm"]].copy()

    if work.empty:
        return pd.DataFrame()

    inchikey = str(inchikey).strip()
    canonical_smiles = str(canonical_smiles).strip()

    masks = []

    if inchikey:
        masks.append(work["inchikey"] == inchikey)

    if canonical_smiles:
        masks.append(work["canonical_smiles"] == canonical_smiles)

    if not masks:
        return pd.DataFrame()

    combined_mask = masks[0]

    for m in masks[1:]:
        combined_mask = combined_mask | m

    found = work[combined_mask].copy()

    if found.empty:
        return pd.DataFrame()

    found["_phase_norm"] = found["phase"].apply(spectral_normalize_phase_value)
    found["_phase_priority"] = found["_phase_norm"].apply(spectral_phase_priority_score)

    found["_source_norm"] = found["source"].apply(spectral_normalize_source_value)
    found["_intensity_type_norm"] = (
        found["intensity_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", "unknown")
    )

    return found.reset_index(drop=True)
    
def spectral_get_best_spectrum_for_compound(
    inchikey,
    canonical_smiles="",
    spectrum_type="IR",
    phase_mode="prefer_gas",
    allowed_phases=None,
    allowed_sources=None,
    allowed_intensity_types=None,
    prefer_quantitative=False,
    experimental_only=True,
    descriptor_settings=None,
    progress_callback=None
):
    """
    Выбирает лучший спектр вещества для расчёта дескрипторов.

    phase_mode:
    - "prefer_gas"       : предпочитать gas, если есть, иначе следующий по приоритету
    - "only_gas"         : использовать только gas
    - "any"              : использовать любой активный спектр
    - "manual"           : использовать только allowed_phases

    Возвращает:
    - dict записи спектра
    - None, если подходящий спектр не найден
    """
    candidates = spectral_find_candidate_spectra_for_compound(
        inchikey=inchikey,
        canonical_smiles=canonical_smiles,
        spectrum_type=spectrum_type
    )

    if candidates.empty:
        return None

    work = candidates.copy()

    # ------------------------------------------------------------
    # Фильтр: только экспериментальные, если требуется

    if experimental_only and "is_experimental" in work.columns:
        exp_norm = work["is_experimental"].astype(str).str.lower().str.strip()

        experimental_values = ["true", "1", "yes", "y", "да", "experimental", ""]

        work = work[
            exp_norm.isin(experimental_values)
        ].copy()

        if work.empty:
            return None

    # ------------------------------------------------------------
    # Фильтр по источникам

    if allowed_sources is not None:
        allowed_sources_norm = [
            spectral_normalize_source_value(x)
            for x in allowed_sources
        ]

        work = work[
            work["_source_norm"].isin(allowed_sources_norm)
        ].copy()

        if work.empty:
            return None

    # ------------------------------------------------------------
    # Фильтр по типам интенсивности

    if allowed_intensity_types is not None:
        allowed_intensity_types_norm = [
            str(x).strip().lower()
            for x in allowed_intensity_types
        ]

        work = work[
            work["_intensity_type_norm"].isin(allowed_intensity_types_norm)
        ].copy()

        if work.empty:
            return None

    # ------------------------------------------------------------
    # Фильтр / режим по фазам

    phase_mode = str(phase_mode).strip().lower()

    if phase_mode == "only_gas":
        work = work[
            work["_phase_norm"] == "gas"
        ].copy()

        if work.empty:
            return None

    elif phase_mode == "manual":
        if allowed_phases is None or len(allowed_phases) == 0:
            return None

        allowed_phases_norm = [
            spectral_normalize_phase_value(x)
            for x in allowed_phases
        ]

        work = work[
            work["_phase_norm"].isin(allowed_phases_norm)
        ].copy()

        if work.empty:
            return None

    elif phase_mode == "any":
        pass

    else:
        # prefer_gas:
        # ничего не фильтруем, просто сортируем по фазовому приоритету
        phase_mode = "prefer_gas"

    # ------------------------------------------------------------
    # Приоритет количественных спектров, если пользователь выбрал

    if "is_quantitative" in work.columns:
        quant_norm = work["is_quantitative"].astype(str).str.lower().str.strip()
        work["_quant_priority"] = np.where(
            quant_norm.isin(["true", "1", "yes", "y", "да"]),
            0 if prefer_quantitative else 1,
            1 if prefer_quantitative else 0
        )
    else:
        work["_quant_priority"] = 0

    # ------------------------------------------------------------
    # Приоритет confidence_level

    if "confidence_level" in work.columns:
        conf = work["confidence_level"].astype(str).str.lower().str.strip()

        confidence_priority_map = {
            "high": 0,
            "spectral_page_found": 10,
            "medium": 20,
            "low": 40,
            "error": 90,
            "": 50,
        }

        work["_confidence_priority"] = conf.map(confidence_priority_map).fillna(50)
    else:
        work["_confidence_priority"] = 50

    # ------------------------------------------------------------
    # Проверка наличия processed_file

    work["_descriptor_exists_priority"] = 1
    work["_processed_exists_priority"] = 1

    for idx, row in work.iterrows():
        row_dict = row.to_dict()

        if descriptor_settings:
            cached_row = spectral_find_cached_descriptor_row(
                row_dict,
                descriptor_settings=descriptor_settings
            )

            if cached_row is not None:
                work.loc[idx, "_descriptor_exists_priority"] = 0

        processed_file = str(row.get("processed_file", "")).strip()

        if processed_file:
            exists = bool(
                spectral_resolve_processed_spectrum_path(
                    processed_file,
                    spectrum_record=row_dict,
                    allow_remote=False
                )
            )

            if not exists:
                exists = spectra_find_remote_manifest_record(
                    processed_file,
                    spectrum_record=row_dict
                ) is not None

            if exists:
                work.loc[idx, "_processed_exists_priority"] = 0

    # ------------------------------------------------------------
    # Сортировка

    work = work.sort_values(
        by=[
            "_descriptor_exists_priority",
            "_processed_exists_priority",
            "_phase_priority",
            "_quant_priority",
            "_confidence_priority",
            "spectrum_id",
        ],
        ascending=[True, True, True, True, True, True]
    ).reset_index(drop=True)

    selected = work.iloc[0].to_dict()

    selected_phase = selected.get("_phase_norm", "unknown")

    if phase_mode == "prefer_gas":
        if selected_phase == "gas":
            reason = "selected_gas_by_priority"
        else:
            reason = f"gas_not_available_selected_{selected_phase}"
    elif phase_mode == "only_gas":
        reason = "selected_only_gas"
    elif phase_mode == "manual":
        reason = f"selected_from_manual_phases_{selected_phase}"
    else:
        reason = f"selected_any_active_{selected_phase}"

    selected["spectrum_selection_reason"] = reason
    selected["spectrum_phase_norm"] = selected_phase

    return selected

def spectral_build_descriptors_for_dataset(
    input_df,
    smiles_col="SMILES",
    spectrum_type="IR",
    wn_min=550,
    wn_max=3798,
    step=4,
    normalization="min-max",
    invert_signal=False,
    use_grid=True,
    use_binary_fp=True,
    use_binned_numeric=True,
    binary_window=20,
    binary_threshold=0.10,
    numeric_window=100,
    use_svd=True,
    svd_components=10,
    spectrum_phase_mode="prefer_gas",
    allowed_phases=None,
    allowed_sources=None,
    allowed_intensity_types=None,
    prefer_quantitative=False,
    experimental_only=True,
    progress_callback=None
):
    """
    Главная функция:
    текущий датасет + spectra_bank -> таблица спектральных дескрипторов.

    Возвращает:
    descriptors_df, report_dict
    """
    compounds = spectra_prepare_compounds_from_df(
        input_df,
        smiles_col=smiles_col
    )
    descriptor_settings = spectral_descriptor_cache_settings(
        spectrum_type=spectrum_type,
        wn_min=wn_min,
        wn_max=wn_max,
        step=step,
        normalization=normalization,
        invert_signal=invert_signal,
        use_grid=use_grid,
        use_binary_fp=use_binary_fp,
        use_binned_numeric=use_binned_numeric,
        binary_window=binary_window,
        binary_threshold=binary_threshold,
        numeric_window=numeric_window,
    )

    rows = []
    matrix_rows = []
    matrix_meta = []

    report = {
        "total_compounds": len(compounds),
        "with_spectrum": 0,
        "without_spectrum": 0,
        "parse_errors": 0,
        "used_spectra": [],
        "used_phases": {},
        "spectrum_selection_reasons": {},
        "descriptor_cache_hits": 0,
        "descriptor_cache_misses": 0,
        "spectra_bank_status": spectra_remote_bank_status(),
    }

    total_compounds = len(compounds)

    for processed_count, (_, compound) in enumerate(compounds.iterrows(), start=1):
        inchikey = compound.get("inchikey", "")
        canonical_smiles = compound.get("canonical_smiles", "")

        if progress_callback is not None:
            try:
                progress_callback(
                    processed_count - 1,
                    total_compounds,
                    "searching",
                    {
                        "inchikey": inchikey,
                        "canonical_smiles": canonical_smiles,
                    }
                )
            except Exception:
                pass

        spectrum_record = spectral_get_best_spectrum_for_compound(
            inchikey=inchikey,
            canonical_smiles=canonical_smiles,
            spectrum_type=spectrum_type,
            phase_mode=spectrum_phase_mode,
            allowed_phases=allowed_phases,
            allowed_sources=allowed_sources,
            allowed_intensity_types=allowed_intensity_types,
            prefer_quantitative=prefer_quantitative,
            experimental_only=experimental_only,
            descriptor_settings=descriptor_settings
        )

        if spectrum_record is None:
            report["without_spectrum"] += 1
            if progress_callback is not None:
                try:
                    progress_callback(
                        processed_count,
                        total_compounds,
                        "no_spectrum",
                        {
                            "inchikey": inchikey,
                            "canonical_smiles": canonical_smiles,
                        }
                    )
                except Exception:
                    pass
            continue

        processed_file = spectrum_record.get("processed_file", "")

        cached_descriptor_row = spectral_find_cached_descriptor_row(
            spectrum_record,
            descriptor_settings=descriptor_settings
        )

        if cached_descriptor_row is not None:
            desc = spectral_prepare_cached_descriptor_row(
                cached_descriptor_row,
                compound,
                spectrum_record,
                processed_file
            )

            if desc is not None:
                rows.append(desc)

                values_norm_cached = spectral_values_norm_from_descriptor_row(
                    desc,
                    spectrum_type=spectrum_type
                )

                if values_norm_cached is not None:
                    matrix_rows.append(values_norm_cached)
                    matrix_meta.append({
                        "row_index": compound.get("row_index", ""),
                        "compound_id": compound.get("compound_id", ""),
                        "name": compound.get("name", ""),
                        "canonical_smiles": canonical_smiles,
                        "inchikey": inchikey,
                        "spectrum_id": spectrum_record.get("spectrum_id", ""),
                    })

                report["with_spectrum"] += 1
                report["descriptor_cache_hits"] += 1
                report["used_spectra"].append(spectrum_record.get("spectrum_id", ""))

                used_phase = str(
                    spectrum_record.get(
                        "spectrum_phase_norm",
                        spectrum_record.get("phase", "unknown")
                    )
                ).strip()

                if not used_phase:
                    used_phase = "unknown"

                report["used_phases"][used_phase] = (
                    report["used_phases"].get(used_phase, 0) + 1
                )

                selection_reason = str(
                    spectrum_record.get("spectrum_selection_reason", "unknown")
                ).strip()

                if not selection_reason:
                    selection_reason = "unknown"

                report["spectrum_selection_reasons"][selection_reason] = (
                    report["spectrum_selection_reasons"].get(selection_reason, 0) + 1
                )

                if progress_callback is not None:
                    try:
                        progress_callback(
                            processed_count,
                            total_compounds,
                            "done",
                            {
                                "inchikey": inchikey,
                                "canonical_smiles": canonical_smiles,
                                "spectrum_id": spectrum_record.get("spectrum_id", ""),
                                "processed_file": processed_file,
                                "descriptor_source": "cached_descriptor",
                            }
                        )
                    except Exception:
                        pass

                continue

        report["descriptor_cache_misses"] += 1

        if progress_callback is not None:
            try:
                progress_callback(
                    processed_count - 1,
                    total_compounds,
                    "loading_spectrum",
                    {
                        "inchikey": inchikey,
                        "canonical_smiles": canonical_smiles,
                        "spectrum_id": spectrum_record.get("spectrum_id", ""),
                        "processed_file": processed_file,
                    }
                )
            except Exception:
                pass

        spectrum_df = spectral_load_processed_spectrum(
            processed_file,
            spectrum_record=spectrum_record
        )

        if spectrum_df.empty:
            report["parse_errors"] += 1
            if progress_callback is not None:
                try:
                    progress_callback(
                        processed_count,
                        total_compounds,
                        "parse_error",
                        {
                            "inchikey": inchikey,
                            "canonical_smiles": canonical_smiles,
                            "spectrum_id": spectrum_record.get("spectrum_id", ""),
                            "processed_file": processed_file,
                        }
                    )
                except Exception:
                    pass
            continue

        grid, values = spectral_interpolate_to_grid(
            spectrum_df,
            wn_min=wn_min,
            wn_max=wn_max,
            step=step
        )

        if len(grid) == 0:
            report["parse_errors"] += 1
            if progress_callback is not None:
                try:
                    progress_callback(
                        processed_count,
                        total_compounds,
                        "empty_grid",
                        {
                            "inchikey": inchikey,
                            "canonical_smiles": canonical_smiles,
                            "spectrum_id": spectrum_record.get("spectrum_id", ""),
                            "processed_file": processed_file,
                        }
                    )
                except Exception:
                    pass
            continue

        values_norm = spectral_normalize_values(
            values,
            method=normalization,
            invert=invert_signal
        )

        desc = spectral_make_descriptor_row_metadata(
            compound,
            spectrum_record,
            processed_file
        )

        if use_grid:
            desc.update(
                spectral_make_grid_descriptors(
                    grid,
                    values_norm,
                    prefix=f"{spectra_normalize_spectrum_type(spectrum_type)}_GRID"
                )
            )

        if use_binary_fp:
            desc.update(
                spectral_make_binary_fp(
                    grid,
                    values_norm,
                    window_size=binary_window,
                    threshold=binary_threshold,
                    prefix=f"{spectra_normalize_spectrum_type(spectrum_type)}_BIN"
                )
            )

        if use_binned_numeric:
            desc.update(
                spectral_make_binned_numeric_descriptors(
                    grid,
                    values_norm,
                    window_size=numeric_window,
                    prefix=f"{spectra_normalize_spectrum_type(spectrum_type)}_BAND"
                )
            )

        rows.append(desc)
        spectral_update_descriptor_cache(
            desc,
            descriptor_settings,
            spectrum_type=spectrum_type
        )

        matrix_rows.append(values_norm)
        matrix_meta.append({
            "row_index": compound.get("row_index", ""),
            "compound_id": compound.get("compound_id", ""),
            "name": compound.get("name", ""),
            "canonical_smiles": canonical_smiles,
            "inchikey": inchikey,
            "spectrum_id": spectrum_record.get("spectrum_id", ""),
        })

        report["with_spectrum"] += 1
        report["used_spectra"].append(spectrum_record.get("spectrum_id", ""))
        
        used_phase = str(
            spectrum_record.get(
                "spectrum_phase_norm",
                spectrum_record.get("phase", "unknown")
            )
        ).strip()

        if not used_phase:
            used_phase = "unknown"

        report["used_phases"][used_phase] = (
            report["used_phases"].get(used_phase, 0) + 1
        )

        selection_reason = str(
            spectrum_record.get("spectrum_selection_reason", "unknown")
        ).strip()

        if not selection_reason:
            selection_reason = "unknown"

        report["spectrum_selection_reasons"][selection_reason] = (
            report["spectrum_selection_reasons"].get(selection_reason, 0) + 1
        )

        if progress_callback is not None:
            try:
                progress_callback(
                    processed_count,
                    total_compounds,
                    "done",
                    {
                        "inchikey": inchikey,
                        "canonical_smiles": canonical_smiles,
                        "spectrum_id": spectrum_record.get("spectrum_id", ""),
                        "processed_file": processed_file,
                    }
                )
            except Exception:
                pass

    descriptors_df = pd.DataFrame(rows)

    # SVD-компоненты
    if use_svd and len(matrix_rows) >= 2 and not descriptors_df.empty:
        X_spec = np.vstack(matrix_rows)

        max_components = min(
            int(svd_components),
            X_spec.shape[0] - 1,
            X_spec.shape[1] - 1
        )

        if max_components >= 1:
            try:
                svd = TruncatedSVD(
                    n_components=max_components,
                    random_state=42
                )

                svd_values = svd.fit_transform(X_spec)

                svd_df = pd.DataFrame(matrix_meta)

                prefix = spectra_normalize_spectrum_type(spectrum_type)

                for i in range(max_components):
                    svd_df[f"{prefix}_SVD_{i + 1}"] = svd_values[:, i]

                descriptors_df = descriptors_df.merge(
                    svd_df,
                    on=[
                        "row_index",
                        "compound_id",
                        "name",
                        "canonical_smiles",
                        "inchikey",
                        "spectrum_id",
                    ],
                    how="left"
                )

                report["svd_components_created"] = max_components
                report["svd_explained_variance_sum"] = float(
                    np.sum(svd.explained_variance_ratio_)
                )

            except Exception as e:
                report["svd_error"] = str(e)

    report["without_spectrum"] = (
        report["total_compounds"]
        - report["with_spectrum"]
        - report["parse_errors"]
    )

    if report["without_spectrum"] < 0:
        report["without_spectrum"] = 0

    if descriptors_df.empty:
        return descriptors_df, report

    return descriptors_df.reset_index(drop=True), report


def spectral_save_descriptors(descriptors_df, spectrum_type="IR"):
    """
    Сохраняет таблицу спектральных дескрипторов.
    """
    if descriptors_df is None or descriptors_df.empty:
        return ""

    dirs = spectra_get_dirs_by_type(spectrum_type)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filepath = os.path.join(
        dirs["descriptors"],
        f"{spectra_normalize_spectrum_type(spectrum_type)}_spectral_descriptors_{timestamp}.csv"
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    descriptors_df.to_csv(filepath, index=False)

    return filepath
