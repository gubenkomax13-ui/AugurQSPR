# -*- coding: utf-8 -*-
"""Runtime installation diagnostics for optional Augur QSPR dependencies."""

from __future__ import annotations

import importlib.util
import importlib
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum


class InstallStatusCode(str, Enum):
    OK = "OK"
    MISSING = "MISSING"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class InstallationDiagnostic:
    component: str
    status: InstallStatusCode
    details: str = ""
    import_status: str = "not_applicable"
    functional_status: str = "not_checked"
    functional_details: str = ""


def _classify_import_failure(module_name: str, exc: Exception) -> str:
    message = str(exc).lower()
    if importlib.util.find_spec(module_name) is None:
        return "not_installed"
    if isinstance(exc, OSError):
        if any(token in message for token in ["dll", "shared object", "cannot open", "library"]):
            return "runtime_dependency_missing"
        return "binary_incompatibility"
    if any(token in message for token in ["numpy.dtype size changed", "binary incompat", "abi"]):
        return "binary_incompatibility"
    if isinstance(exc, ImportError):
        return "runtime_dependency_missing"
    return "import_failed"


def _module_import_status(module_name: str) -> tuple[bool, str, str]:
    if importlib.util.find_spec(module_name) is None:
        return False, "not_installed", "module spec not found"
    try:
        importlib.import_module(module_name)
        return True, "ok", ""
    except ModuleNotFoundError as exc:
        status = "not_installed" if getattr(exc, "name", "") == module_name else "runtime_dependency_missing"
        return False, status, str(exc)
    except Exception as exc:
        return False, _classify_import_failure(module_name, exc), str(exc)


def _command_available(command: str) -> bool:
    return shutil.which(command) is not None


def _java_available() -> tuple[bool, str]:
    if not _command_available("java"):
        return False, "java executable is not on PATH"
    try:
        completed = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    version_text = (completed.stderr or completed.stdout or "").splitlines()
    details = version_text[0] if version_text else "java executable found"
    return completed.returncode == 0, details


def _julia_available() -> tuple[bool, str]:
    if not _command_available("julia"):
        return False, "julia executable is not on PATH"
    try:
        completed = subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    details = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, details


def _rdkit_functional() -> tuple[bool, str]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        if mol is None:
            return False, "RDKit could not parse CCO"
        return True, Chem.MolToSmiles(mol, canonical=True)
    except Exception as exc:
        return False, str(exc)


def _padel_functional(java_ok: bool) -> tuple[bool, str]:
    if not java_ok:
        return False, "Java is required for PaDEL"
    try:
        from padelpy import from_smiles
        result = from_smiles(["CCO"], fingerprints=False)
        if result and isinstance(result, list) and result[0]:
            return True, f"{len(result[0])} descriptors"
        return False, "PaDEL returned no descriptors"
    except Exception as exc:
        return False, str(exc)


def _xtb_functional() -> tuple[bool, str]:
    try:
        from modules.qspr_core import qspr_calc_xtb_descriptors_single
        result = qspr_calc_xtb_descriptors_single(
            "CCO",
            conformer_mode="fast",
            conformer_count=1,
            max_embed_attempts=3,
        )
        status = str(result.get("xtb_status", ""))
        return status in {"complete", "ok"}, status or "no status"
    except Exception as exc:
        return False, str(exc)


def _kaleido_functional() -> tuple[bool, str]:
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Scatter(x=[0, 1], y=[0, 1]))
        png = fig.to_image(format="png", width=320, height=240)
        return bool(png), f"{len(png)} bytes"
    except Exception as exc:
        return False, str(exc)


def _pysr_functional(julia_ok: bool) -> tuple[bool, str]:
    if not julia_ok:
        return False, "Julia is required for PySR"
    try:
        import pysr  # noqa: F401
        return True, "pysr import and Julia executable found"
    except Exception as exc:
        return False, str(exc)


def _combined_status(import_ok: bool, functional_ok: bool | None) -> InstallStatusCode:
    if functional_ok is True:
        return InstallStatusCode.OK
    if import_ok:
        return InstallStatusCode.MISSING
    return InstallStatusCode.UNAVAILABLE


def collect_installation_diagnostics() -> list[InstallationDiagnostic]:
    """Return deterministic diagnostics for installed and optional tools."""
    java_ok, java_details = _java_available()
    julia_ok, julia_details = _julia_available()
    xtb_command = _command_available("xtb")
    rdkit_import, rdkit_import_status, rdkit_import_details = _module_import_status("rdkit")
    rdkit_func, rdkit_func_details = _rdkit_functional() if rdkit_import else (False, rdkit_import_details or "rdkit is not importable")
    padel_import, padel_import_status, padel_import_details = _module_import_status("padelpy")
    padel_func, padel_func_details = _padel_functional(java_ok) if padel_import else (False, padel_import_details or "padelpy is not importable")
    mordred_import, mordred_import_status, mordred_import_details = _module_import_status("mordred")
    kaleido_import, kaleido_import_status, kaleido_import_details = _module_import_status("kaleido")
    kaleido_func, kaleido_func_details = _kaleido_functional() if kaleido_import else (False, kaleido_import_details or "kaleido is not importable")
    pysr_import, pysr_import_status, pysr_import_details = _module_import_status("pysr")
    pysr_func, pysr_func_details = _pysr_functional(julia_ok) if pysr_import else (False, pysr_import_details or "pysr is not importable")
    xtb_import, xtb_import_status, xtb_import_details = _module_import_status("xtb")
    xtb_func, xtb_func_details = _xtb_functional() if xtb_import else (False, xtb_import_details or "xtb Python package is not importable")

    checks = [
        InstallationDiagnostic(
            "RDKit",
            _combined_status(rdkit_import, rdkit_func),
            rdkit_import_details or "rdkit Python package",
            rdkit_import_status,
            "yes" if rdkit_func else "no",
            rdkit_func_details,
        ),
        InstallationDiagnostic(
            "Java",
            InstallStatusCode.OK if java_ok else InstallStatusCode.MISSING,
            java_details,
            "not_applicable",
            "yes" if java_ok else "no",
            java_details,
        ),
        InstallationDiagnostic(
            "PaDEL",
            _combined_status(padel_import, padel_func),
            "padelpy + Java" if java_ok else "padelpy requires Java",
            padel_import_status,
            "yes" if padel_func else "no",
            padel_func_details,
        ),
        InstallationDiagnostic(
            "xTB",
            _combined_status(xtb_import or xtb_command, xtb_func),
            "xtb executable" if xtb_command else "xtb executable is not on PATH",
            xtb_import_status,
            "yes" if xtb_func else "no",
            xtb_func_details,
        ),
        InstallationDiagnostic(
            "Mordred",
            InstallStatusCode.OK if mordred_import else InstallStatusCode.MISSING,
            mordred_import_details or "mordred Python package",
            mordred_import_status,
            "not_checked",
            "descriptor object construction is not required at startup",
        ),
        InstallationDiagnostic(
            "Kaleido",
            _combined_status(kaleido_import, kaleido_func),
            "required by Plotly static image export",
            kaleido_import_status,
            "yes" if kaleido_func else "no",
            kaleido_func_details,
        ),
        InstallationDiagnostic(
            "PySR",
            _combined_status(pysr_import, pysr_func),
            "optional symbolic regression package",
            pysr_import_status,
            "yes" if pysr_func else "no",
            pysr_func_details,
        ),
        InstallationDiagnostic(
            "Julia",
            InstallStatusCode.OK if julia_ok else InstallStatusCode.UNAVAILABLE,
            julia_details,
            "not_applicable",
            "yes" if julia_ok else "no",
            julia_details,
        ),
    ]
    return checks
