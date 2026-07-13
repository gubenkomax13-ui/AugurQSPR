#!/usr/bin/env python3
"""PowerShell-safe wrapper around ripgrep for multi-pattern searches.

Use this instead of putting regex alternation or shell-sensitive characters in
the command line. Patterns are passed to rg as repeated -e arguments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def build_command(args: argparse.Namespace) -> list[str]:
    command = ["rg", "-n", "--color", "never"]

    if not args.regex:
        command.append("-F")
    if args.ignore_case:
        command.append("-i")

    for glob in args.glob or []:
        command.extend(["-g", glob])

    for pattern in args.pattern:
        command.extend(["-e", pattern])

    paths = []
    paths.extend(args.path or [])
    paths.extend(args.paths or [])
    command.extend(paths or ["."])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search with rg without exposing patterns to PowerShell parsing."
    )
    parser.add_argument(
        "-p",
        "--pattern",
        action="append",
        required=True,
        help="Search pattern. Repeat this option for multiple patterns.",
    )
    parser.add_argument(
        "--regex",
        action="store_true",
        help="Treat patterns as regular expressions. Defaults to fixed-string search.",
    )
    parser.add_argument(
        "-i",
        "--ignore-case",
        action="store_true",
        help="Case-insensitive search.",
    )
    parser.add_argument(
        "-g",
        "--glob",
        action="append",
        help="rg glob filter. Repeat for multiple globs.",
    )
    parser.add_argument(
        "--path",
        action="append",
        help="File or directory to search. Repeat for multiple paths.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to search. Defaults to the current directory.",
    )
    args = parser.parse_args(argv)

    try:
        completed = subprocess.run(build_command(args), check=False)
    except FileNotFoundError:
        print("safe_rg.py: rg was not found on PATH.", file=sys.stderr)
        return 127

    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
