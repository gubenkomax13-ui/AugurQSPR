# -*- coding: utf-8 -*-
"""Command-line localization audit for Augur QSPR."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from modules.i18n import collect_translation_keys, validate_translation_keys


LOCALES = ("ru", "en", "kk")
PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            values.update(_flatten(value, full_key))
        elif isinstance(value, str):
            values[full_key] = value
    return values


def _has_cyrillic(value: str) -> bool:
    return any("CYRILLIC" in unicodedata.name(char, "") for char in value)


def _load_locale_files(project_root: Path) -> dict[str, dict[str, str]]:
    locale_values: dict[str, dict[str, str]] = {}
    for language in LOCALES:
        path = project_root / "locales" / f"{language}.json"
        with path.open("r", encoding="utf-8") as handle:
            locale_values[language] = _flatten(json.load(handle))
    return locale_values


def _placeholder_issues(values: dict[str, dict[str, str]]) -> list[str]:
    issues: list[str] = []
    all_keys = set().union(*(set(locale_values) for locale_values in values.values()))
    for key in sorted(all_keys):
        placeholders_by_language = {
            language: sorted(PLACEHOLDER_RE.findall(locale_values.get(key, "")))
            for language, locale_values in values.items()
        }
        if len({tuple(placeholders) for placeholders in placeholders_by_language.values()}) > 1:
            issues.append(f"{key}: {placeholders_by_language}")
    return issues


def _quality_warnings(values: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    warnings: dict[str, list[str]] = {
        "en_contains_cyrillic": [],
        "suspicious_markers": [],
        "en_equals_ru": [],
        "kk_equals_ru": [],
    }

    en_values = values.get("en", {})
    ru_values = values.get("ru", {})
    kk_values = values.get("kk", {})

    for key, value in sorted(en_values.items()):
        if _has_cyrillic(value):
            warnings["en_contains_cyrillic"].append(key)
        if "??" in value or "\ufffd" in value:
            warnings["suspicious_markers"].append(f"en:{key}")
        if key in ru_values and value == ru_values[key] and _has_cyrillic(value):
            warnings["en_equals_ru"].append(key)

    for key, value in sorted(kk_values.items()):
        if "??" in value or "\ufffd" in value:
            warnings["suspicious_markers"].append(f"kk:{key}")
        if key in ru_values and value == ru_values[key] and _has_cyrillic(value):
            warnings["kk_equals_ru"].append(key)

    return warnings


def _print_limited(title: str, items: list[str], limit: int = 40) -> None:
    print(f"{title}: {len(items)}")
    for item in items[:limit]:
        print(f"  - {item}")
    if len(items) > limit:
        print(f"  ... {len(items) - limit} more")


def main() -> int:
    project_root = Path(__file__).resolve().parent
    required_keys = collect_translation_keys(str(project_root))
    completeness_issues = validate_translation_keys(str(project_root))
    locale_values = _load_locale_files(project_root)
    placeholder_issues = _placeholder_issues(locale_values)
    warnings = _quality_warnings(locale_values)

    print(f"Required keys checked: {len(required_keys)}")
    if completeness_issues:
        print("Missing localization keys:")
        for language, missing_keys in completeness_issues.items():
            _print_limited(f"  {language}", list(missing_keys))
    else:
        print("Completeness: OK")

    if placeholder_issues:
        _print_limited("Placeholder mismatches", placeholder_issues)
    else:
        print("Placeholders: OK")

    print("Quality warnings:")
    for warning_name, warning_items in warnings.items():
        _print_limited(f"  {warning_name}", warning_items)

    return 1 if completeness_issues or placeholder_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
