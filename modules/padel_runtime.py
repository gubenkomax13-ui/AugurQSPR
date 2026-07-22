# -*- coding: utf-8 -*-
"""Runtime preparation for padelpy on Windows paths containing spaces.

padelpy builds the Java command as a string without quoting the path to
PaDEL-Descriptor.jar.  A project installed in e.g. ``QSPR Forge`` therefore
fails despite a successful import.  PaDEL's JAR also needs its adjacent
libraries, so the complete bundled directory is copied to a temporary path
without spaces and padelpy is pointed there.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def prepare_padel_runtime() -> dict:
    """Make padelpy's bundled Java runtime safe for a path with spaces."""
    try:
        from padelpy import wrapper
    except Exception as exc:
        return {"ready": False, "prepared": False, "message": str(exc)}

    source_jar = Path(str(wrapper._PADEL_PATH)).resolve()
    if not source_jar.exists():
        return {
            "ready": False,
            "prepared": False,
            "message": f"PaDEL JAR not found: {source_jar}",
        }
    if " " not in str(source_jar):
        return {"ready": True, "prepared": False, "message": "PaDEL path has no spaces."}

    runtime_root = Path(tempfile.gettempdir()) / "augur_padel_runtime"
    # A temporary folder with a space would not fix padelpy's quoting defect.
    if " " in str(runtime_root):
        return {
            "ready": False,
            "prepared": False,
            "message": f"Temporary PaDEL path contains spaces: {runtime_root}",
        }

    target_jar = runtime_root / source_jar.name
    try:
        copy_needed = (
            not target_jar.exists()
            or target_jar.stat().st_size != source_jar.stat().st_size
            or target_jar.stat().st_mtime_ns < source_jar.stat().st_mtime_ns
        )
        if copy_needed:
            shutil.copytree(source_jar.parent, runtime_root, dirs_exist_ok=True)
        wrapper._PADEL_PATH = str(target_jar)
        return {
            "ready": True,
            "prepared": True,
            "message": f"PaDEL runtime prepared at {runtime_root}",
            "runtime_path": str(target_jar),
        }
    except Exception as exc:
        return {"ready": False, "prepared": False, "message": str(exc)}
