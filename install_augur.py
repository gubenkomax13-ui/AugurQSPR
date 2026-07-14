#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External dependency installer for Augur QSPR.

This script intentionally uses only the Python standard library so it can run
before Streamlit, scikit-learn, RDKit, or other application dependencies exist.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIREMENT_FILES = {
    "base": "requirements.txt",
    "local": "requirements-local.txt",
    "full": "requirements-full.txt",
    "lock": "requirements-lock.txt",
}


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.check_call(command, cwd=str(ROOT))


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Augur QSPR dependencies.")
    parser.add_argument(
        "--profile",
        choices=sorted(REQUIREMENT_FILES),
        default="local",
        help="Dependency profile to install.",
    )
    parser.add_argument(
        "--upgrade-pip",
        action="store_true",
        help="Upgrade pip before installing requirements.",
    )
    parser.add_argument(
        "--user",
        action="store_true",
        help="Install into the user site-packages. Do not use inside a virtual environment.",
    )
    args = parser.parse_args()

    req_file = ROOT / REQUIREMENT_FILES[args.profile]
    if not req_file.exists():
        print(f"Requirement file not found: {req_file}", file=sys.stderr)
        return 2

    if args.user and in_virtualenv():
        print(
            "--user cannot be used inside an active virtual environment; "
            "it can install packages outside the environment that runs Augur.",
            file=sys.stderr,
        )
        return 2

    if args.upgrade_pip:
        command = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if args.user:
            command.append("--user")
        command.append("pip")
        run(command)

    command = [sys.executable, "-m", "pip", "install"]
    if args.user:
        command.append("--user")
    command.extend(["-r", str(req_file)])
    run(command)

    print()
    print("Installation command completed.")
    print("Restart the Streamlit/Python process before running Augur again.")
    print("Run the app with:")
    print(f"  {sys.executable} -m streamlit run qspr_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
