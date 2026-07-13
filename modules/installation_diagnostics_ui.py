# -*- coding: utf-8 -*-
"""Streamlit UI for installation diagnostics."""

import pandas as pd
import streamlit as st

from modules.installation_diagnostics import (
    InstallStatusCode,
    collect_installation_diagnostics,
)
from modules.i18n import t
from modules.qspr_core import qspr_csv_download_bytes


STATUS_LABELS = {
    InstallStatusCode.OK: "works",
    InstallStatusCode.MISSING: "installed_but_not_verified",
    InstallStatusCode.UNAVAILABLE: "unavailable",
}


MODULE_MATURITY_ROWS = [
    {"Module": "RDKit descriptors", "Status": "stable", "Scope": "core descriptor calculation"},
    {"Module": "Standard regression", "Status": "stable", "Scope": "baseline QSPR modeling"},
    {"Module": "SAOD for alkanes", "Status": "validation", "Scope": "rule discovery for alkane series"},
    {"Module": "Universal SAOD", "Status": "experimental", "Scope": "automatic chemical-series discovery"},
    {"Module": "SAOD-SAR", "Status": "in_development", "Scope": "R-group and transformation analysis"},
    {"Module": "xTB", "Status": "advanced_local", "Scope": "local quantum descriptors"},
    {"Module": "PySR", "Status": "experimental", "Scope": "symbolic regression"},
    {"Module": "Chemical space", "Status": "diagnostic", "Scope": "similarity/projection diagnostics"},
]


def _status_renderer(item):
    line = (
        f"{item.component}: import={item.import_status}, "
        f"functional_test={item.functional_status}, status={STATUS_LABELS[item.status]}"
    )
    if item.status == InstallStatusCode.OK:
        st.success(line)
    elif item.status == InstallStatusCode.MISSING:
        st.warning(line)
    else:
        st.info(line)


def render_installation_diagnostics_section():
    """Render a standalone installation diagnostics page."""
    st.header(t("installation_diagnostics.header"))
    st.caption(t("installation_diagnostics.caption"))

    diagnostics = collect_installation_diagnostics()
    rows = [
        {
            "Tool": item.component,
            "Import": item.import_status,
            "Functional test": item.functional_status,
            "Status": STATUS_LABELS[item.status],
            "Code": item.status.value,
            "Details": item.details,
            "Functional details": item.functional_details,
        }
        for item in diagnostics
    ]

    for item in diagnostics:
        _status_renderer(item)

    with st.expander(t("installation_diagnostics.details"), expanded=True):
        display = pd.DataFrame(rows)
        st.dataframe(display, width="stretch", hide_index=True)

    with st.expander(t("installation_diagnostics.maturity_matrix"), expanded=True):
        maturity = pd.DataFrame(MODULE_MATURITY_ROWS)
        st.dataframe(maturity, width="stretch", hide_index=True)
        st.download_button(
            t("installation_diagnostics.download_maturity_csv"),
            qspr_csv_download_bytes(maturity),
            "module_maturity_matrix.csv",
            "text/csv",
            key="download_module_maturity_matrix_csv",
        )

    st.download_button(
        t("installation_diagnostics.download_diagnostics_csv"),
        qspr_csv_download_bytes(pd.DataFrame(rows)),
        "installation_diagnostics.csv",
        "text/csv",
        key="download_installation_diagnostics_csv",
    )
