# -*- coding: utf-8 -*-
"""Командная проверка полноты локализаций QSPR Forge."""

from pathlib import Path

from modules.i18n import collect_translation_keys, validate_translation_keys


def main() -> int:
    project_root = Path(__file__).resolve().parent
    required_keys = collect_translation_keys(str(project_root))
    issues = validate_translation_keys(str(project_root))

    print(f"Проверено ключей: {len(required_keys)}")
    if not issues:
        print("Локализации ru/en/kk полны.")
        return 0

    for language, missing_keys in issues.items():
        print(f"\n{language}: отсутствует {len(missing_keys)}")
        for key in missing_keys:
            print(f"  - {key}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
