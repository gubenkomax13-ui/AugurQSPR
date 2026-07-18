#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External dependency installer for Augur QSPR.

This script intentionally uses only the Python standard library so it can run
before Streamlit, scikit-learn, RDKit, or other application dependencies exist.
"""

from __future__ import annotations

import argparse
import importlib
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


def pip_command(args: argparse.Namespace, *extra: str) -> list[str]:
    command = [sys.executable, "-m", "pip"]
    command.extend(extra)
    if extra and extra[0] == "install":
        for host in args.trusted_host:
            command.extend(["--trusted-host", host])
    return command


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
    parser.add_argument(
        "--trusted-host",
        action="append",
        default=[],
        help=(
            "Pass a trusted host to pip. Useful on Windows machines where "
            "local SSL certificates prevent downloads from PyPI."
        ),
    )
    parser.add_argument(
        "--prewarm-pysr",
        action="store_true",
        help="Import PySR once after installation so juliacall can prepare its Julia environment.",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Print optional dependency status and the number of available model candidates.",
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
        command = pip_command(args, "install", "--upgrade")
        if args.user:
            command.append("--user")
        command.append("pip")
        run(command)

    command = pip_command(args, "install")
    if args.user:
        command.append("--user")
    command.extend(["-r", str(req_file)])
    run(command)

    if args.prewarm_pysr:
        try:
            print()
            print("Prewarming PySR/Juliacall environment...")
            importlib.import_module("pysr")
            print("PySR import completed.")
        except Exception as exc:
            print(
                "PySR Python package is installed, but its Julia runtime "
                f"could not be prepared automatically: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if args.check_models:
        try:
            print()
            print("Checking Augur model availability...")
            from modules import qspr_core

            statuses = [
                ("xgboost", qspr_core.xgboost_available, qspr_core.xgboost_status),
                ("lightgbm", qspr_core.lightgbm_available, qspr_core.lightgbm_status),
                ("catboost", qspr_core.catboost_available, qspr_core.catboost_status),
                ("pysr", qspr_core.pysr_available, qspr_core.pysr_status),
            ]
            for package, available, status in statuses:
                detail = ""
                if isinstance(status, dict) and status.get("error_message"):
                    detail = f" ({status.get('error_type')}: {status.get('error_message')})"
                print(f"  {package}: {'OK' if available else 'missing'}{detail}")
            total = sum(len(v) for v in qspr_core.qspr_available_model_options().values())
            print(f"  available model candidates: {total}")
        except Exception as exc:
            print(
                f"Model availability check failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    print()
    print("Installation command completed.")
    print("Restart the Streamlit/Python process before running Augur again.")
    print("Run the app with:")
    print(f"  {sys.executable} -m streamlit run qspr_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
