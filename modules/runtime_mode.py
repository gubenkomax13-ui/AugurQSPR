# -*- coding: utf-8 -*-
"""Runtime-mode detection kept outside calculation cores."""

from __future__ import annotations

import os


ONLINE_MODES = {"online", "demo", "cloud", "public"}
LOCAL_MODES = {"local", "full", "desktop"}


def _normalise_mode(value):
    value = str(value or "").strip().lower()
    if value in ONLINE_MODES:
        return "online"
    if value in LOCAL_MODES:
        return "local"
    return ""


def qspr_runtime_mode():
    for source in (os.environ.get("AUGUR_RUNTIME_MODE"), os.environ.get("AUGUR_MODE")):
        mode = _normalise_mode(source)
        if mode:
            return mode

    try:
        import streamlit as st  # Localised UI/runtime dependency.

        mode = _normalise_mode(st.secrets.get("AUGUR_RUNTIME_MODE", ""))
        if mode:
            return mode

        mode = _normalise_mode(st.secrets.get("AUGUR_MODE", ""))
        if mode:
            return mode

        context = getattr(st, "context", None)
        headers = getattr(context, "headers", {}) if context is not None else {}
        host = str(headers.get("host") or headers.get("Host") or "").lower()
        url = str(getattr(context, "url", "") or "").lower()
    except Exception:
        host = ""
        url = ""

    if any(marker in host or marker in url for marker in ("streamlit.app", "share.streamlit.io")):
        return "online"
    return "local"


def qspr_is_online_mode():
    return qspr_runtime_mode() == "online"
