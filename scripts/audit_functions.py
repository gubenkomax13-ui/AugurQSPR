#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit function definitions and references in the Augur QSPR codebase."""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOTS = ("qspr_app.py", "modules", "tests")
USER_ENTRY_FILES = {"qspr_app.py"}
REPORT_HINTS = ("report", "excel", "download", "generate_full_report")
IGNORED_NAME_PARTS = (" 2106", "++", "-.py")


def iter_python_files(root: Path, roots: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for item in roots:
        path = root / item
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    files = [
        path for path in files
        if not any(part in path.name for part in IGNORED_NAME_PARTS)
    ]
    return sorted({path.resolve() for path in files})


def parse_file(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def audit_functions(root: Path, roots: tuple[str, ...]) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    files = iter_python_files(root, roots)
    definitions: dict[str, list[str]] = defaultdict(list)
    imports: dict[str, list[str]] = defaultdict(list)
    calls: dict[str, list[str]] = defaultdict(list)
    ui_refs: dict[str, list[str]] = defaultdict(list)
    report_refs: dict[str, list[str]] = defaultdict(list)

    for path in files:
        tree = parse_file(path)
        if tree is None:
            continue
        rel = path.relative_to(root.resolve()).as_posix()

        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[node.name].append(f"{rel}:{node.lineno}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports[alias.asname or alias.name].append(f"{rel}:{node.lineno}")

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[-1]].append(f"{rel}:{node.lineno}")

            elif isinstance(node, ast.Call):
                name = call_name(node.func)
                if name:
                    calls[name].append(f"{rel}:{node.lineno}")
                    if rel in USER_ENTRY_FILES or rel.endswith("_ui.py"):
                        ui_refs[name].append(f"{rel}:{node.lineno}")
                    if any(hint in rel.lower() or hint in name.lower() for hint in REPORT_HINTS):
                        report_refs[name].append(f"{rel}:{node.lineno}")

    rows: list[dict[str, str]] = []
    duplicates: dict[str, list[str]] = {}
    for name, locations in sorted(definitions.items()):
        files_for_name = [
            location.rsplit(":", 1)[0]
            for location in locations
        ]
        if len(files_for_name) != len(set(files_for_name)):
            duplicates[name] = locations
        used = bool(calls.get(name) or imports.get(name))
        accessible = bool(ui_refs.get(name))
        in_report = bool(report_refs.get(name))
        if accessible or in_report:
            decision = "keep"
        elif used:
            decision = "review_internal"
        else:
            decision = "remove_candidate"
        rows.append(
            {
                "function": name,
                "defined_at": "; ".join(locations),
                "used": "yes" if used else "no",
                "called_at": "; ".join(calls.get(name, [])) or "-",
                "imported_at": "; ".join(imports.get(name, [])) or "-",
                "user_accessible": "yes" if accessible else "no",
                "report_path": "yes" if in_report else "no",
                "decision": decision,
            }
        )
    return rows, duplicates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="")
    parser.add_argument("--fail-on-duplicates", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows, duplicates = audit_functions(root, DEFAULT_ROOTS)

    fieldnames = [
        "function",
        "defined_at",
        "used",
        "called_at",
        "imported_at",
        "user_accessible",
        "report_path",
        "decision",
    ]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if duplicates:
        print("Duplicate function definitions:")
        for name, locations in duplicates.items():
            print(f"{name}: {'; '.join(locations)}")
    else:
        print("Duplicate function definitions: none")

    return 1 if duplicates and args.fail_on_duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
