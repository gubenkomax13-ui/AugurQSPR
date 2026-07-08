# -*- coding: utf-8 -*-

"""Shared Streamlit renderer for module descriptions."""

import streamlit as st

from modules.module_registry import (
    get_module_description,
    iter_module_blocks,
    module_anchor_id,
    tool_anchor_id,
)


def _anchor(anchor_id):
    st.markdown(f'<span id="{anchor_id}"></span>', unsafe_allow_html=True)


KIND_LABELS = {
    "step": "Шаг",
    "control": "Контроль",
    "analysis": "Анализ",
    "procedure": "Процедура",
    "tool": "Инструмент",
}


def _render_item(module_key, item, anchor_number):
    _anchor(tool_anchor_id(module_key, anchor_number))
    kind = item.get("kind", "tool")
    label = KIND_LABELS.get(kind, "Пункт")
    update_text = f" Обновляет: `{item['updates']}`." if item.get("updates") else ""
    st.markdown(f"- **{label}: {item['name']}** — {item['purpose']}{update_text}")


def render_module_explanation(module_key, expanded=False):
    """Render a module intro plus compact help about its tools."""
    info = get_module_description(module_key)
    if not info:
        return

    _anchor(module_anchor_id(module_key))
    title = info.get("title") or "модуль"

    with st.expander(f"Что делает модуль: {title}", expanded=expanded):
        st.markdown(f"**Назначение модуля:** {info['goal']}")

        anchor_number = 1
        for block in iter_module_blocks(info):
            _anchor(tool_anchor_id(module_key, anchor_number))
            block_name = block.get("title") or block.get("name")
            purpose = block.get("purpose")
            if purpose:
                st.markdown(f"**Блок: {block_name}** — {purpose}")
            else:
                st.markdown(f"**Блок: {block_name}**")
            anchor_number += 1

            for item in block.get("items", []):
                _render_item(module_key, item, anchor_number)
                anchor_number += 1
