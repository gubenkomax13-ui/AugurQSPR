# -*- coding: utf-8 -*-
"""Local-development runtime mode.

The public Streamlit app is maintained on the separate `main` branch.  The
`local-dev` branch is the full desktop/local application, so runtime checks keep
returning local mode even if environment variables or Streamlit headers look
cloud-like.
"""

from __future__ import annotations

def qspr_runtime_mode():
    return "local"


def qspr_is_online_mode():
    return qspr_runtime_mode() == "online"
