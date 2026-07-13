# -*- coding: utf-8 -*-
"""
Модуль интернационализации (i18n) для Augur QSPR.
Поддерживает русский (ru), английский (en), казахский (kk).
"""

import ast
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Iterable, List, Set

LOCALES_DIR = os.path.join(os.path.dirname(__file__), '..', 'locales')
CURRENT_LANG = 'ru'

_cache = {}
_cache_mtime = {}


def _lookup(data: Dict[str, Any], key: str):
    value = data
    for part in key.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _humanize_key(key: str) -> str:
    """Безопасная подпись вместо показа технического !key!."""
    leaf = key.rsplit('.', 1)[-1]
    return leaf.replace('_', ' ').strip().capitalize() or key


def _is_corrupted_translation(value: Any) -> bool:
    return isinstance(value, str) and "???" in value

def load_language(lang: str) -> Dict[str, Any]:
    """Загружает JSON-файл с переводами для языка с обработкой ошибок."""
    file_path = os.path.join(LOCALES_DIR, f"{lang}.json")
    try:
        file_mtime = os.path.getmtime(file_path)
    except OSError:
        file_mtime = None

    if lang in _cache and _cache_mtime.get(lang) == file_mtime:
        return _cache[lang]

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            _cache[lang] = json.load(f)
            _cache_mtime[lang] = file_mtime
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        if lang == 'ru':
            _cache[lang] = {}
            _cache_mtime[lang] = file_mtime
        else:
            _cache[lang] = load_language('ru')
            _cache_mtime[lang] = file_mtime
    return _cache[lang]

def set_language(lang: str):
    global CURRENT_LANG
    CURRENT_LANG = lang

def gettext(key: str, **kwargs) -> str:
    """Возвращает перевод по ключу с подстановкой параметров."""
    data = load_language(CURRENT_LANG)
    value = _lookup(data, key)
    if _is_corrupted_translation(value):
        value = None
        if CURRENT_LANG != 'en':
            value = _lookup(load_language('en'), key)
    if value is None and CURRENT_LANG != 'ru':
        value = _lookup(load_language('ru'), key)
        if _is_corrupted_translation(value):
            value = _lookup(load_language('en'), key)
    if _is_corrupted_translation(value):
        value = _lookup(load_language('en'), key)
    if value is None:
        return _humanize_key(key)
    if isinstance(value, str):
        if kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                return value
        return value
    return str(value)


@lru_cache(maxsize=4)
def collect_translation_keys(source_root: str) -> Set[str]:
    """Собирает статические ключи из вызовов t("...") и gettext("...")."""
    root = Path(source_root).resolve()
    source_files = [root / "qspr_app.py"]
    source_files.extend(sorted((root / "modules").glob("*.py")))
    keys: Set[str] = set()

    for source_file in source_files:
        if not source_file.is_file():
            continue
        try:
            tree = ast.parse(
                source_file.read_text(encoding="utf-8-sig"),
                filename=str(source_file),
            )
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in {"t", "gettext"}:
                continue
            first_arg = node.args[0]
            if (
                isinstance(first_arg, ast.Constant)
                and isinstance(first_arg.value, str)
            ):
                keys.add(first_arg.value)
    return keys


def _flatten_translation_keys(
    data: Dict[str, Any],
    prefix: str = "",
) -> Set[str]:
    keys: Set[str] = set()
    for name, value in data.items():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            keys.update(_flatten_translation_keys(value, full_name))
        else:
            keys.add(full_name)
    return keys


def validate_translation_keys(
    source_root: str,
    languages: Iterable[str] = ("ru", "en", "kk"),
) -> Dict[str, List[str]]:
    """Возвращает отсутствующие статические ключи для каждого языка."""
    required_keys = collect_translation_keys(source_root)
    missing: Dict[str, List[str]] = {}
    for lang in languages:
        available_keys = _flatten_translation_keys(load_language(lang))
        lang_missing = sorted(required_keys - available_keys)
        if lang_missing:
            missing[lang] = lang_missing
    return missing

# Для использования в коде (избегаем конфликта с переменной _)
t = gettext
