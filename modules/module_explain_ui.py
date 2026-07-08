# -*- coding: utf-8 -*-

"""Shared Streamlit renderer for module descriptions."""

import streamlit as st

from modules.i18n import t
from modules.module_registry import (
    get_module_description,
    iter_module_blocks,
    module_anchor_id,
    tool_anchor_id,
)


def _anchor(anchor_id):
    st.markdown(f'<span id="{anchor_id}"></span>', unsafe_allow_html=True)


KIND_LABEL_KEYS = {
    "step": "module_explain.kind_step",
    "control": "module_explain.kind_control",
    "analysis": "module_explain.kind_analysis",
    "procedure": "module_explain.kind_procedure",
    "tool": "module_explain.kind_tool",
}


def _render_item(module_key, item, anchor_number):
    _anchor(tool_anchor_id(module_key, anchor_number))
    kind = item.get("kind", "tool")
    label = t(KIND_LABEL_KEYS.get(kind, "module_explain.kind_item"))
    update_text = (
        t("module_explain.updates", updates=item["updates"])
        if item.get("updates")
        else ""
    )
    st.markdown(
        t(
            "module_explain.item_line",
            label=label,
            name=item["name"],
            purpose=item["purpose"],
            updates=update_text,
        )
    )


def render_module_explanation(module_key, expanded=False):
    """Render a module intro plus compact help about its tools."""
    info = get_module_description(module_key)
    if not info:
        return

    _anchor(module_anchor_id(module_key))
    title = info.get("title") or t("module_explain.fallback_module")

    with st.expander(t("module_explain.expander_title", title=title), expanded=expanded):
        st.markdown(t("module_explain.goal_line", goal=info["goal"]))

        anchor_number = 1
        for block in iter_module_blocks(info):
            _anchor(tool_anchor_id(module_key, anchor_number))
            block_name = block.get("title") or block.get("name")
            purpose = block.get("purpose")
            if purpose:
                st.markdown(
                    t(
                        "module_explain.block_line",
                        block_name=block_name,
                        purpose=purpose,
                    )
                )
            else:
                st.markdown(t("module_explain.block_title", block_name=block_name))
            anchor_number += 1

            for item in block.get("items", []):
                _render_item(module_key, item, anchor_number)
                anchor_number += 1
