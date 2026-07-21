"""Small Streamlit helpers for showing elapsed time during long calculations."""

from __future__ import annotations

from html import escape
from time import perf_counter

import streamlit as st
import streamlit.components.v1 as components


def format_elapsed_time(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def show_last_elapsed_time(timer_key: str, label: str) -> None:
    elapsed_seconds = st.session_state.get(f"{timer_key}_elapsed_seconds")
    if elapsed_seconds is not None:
        st.caption(f"{label}: {format_elapsed_time(elapsed_seconds)}")


def start_elapsed_timer(timer_key: str, label: str):
    """Start a browser-side live timer and return its start time and placeholder."""
    st.session_state.pop(f"{timer_key}_elapsed_seconds", None)
    placeholder = st.empty()
    element_id = f"elapsed-{escape(timer_key, quote=True)}"
    safe_label = escape(str(label))
    with placeholder.container():
        components.html(
            f"""
            <div style="font: 600 14px sans-serif; color: #1f8cff; padding: 0.2rem 0;">
              {safe_label}: <span id="{element_id}">00:00</span>
            </div>
            <script>
              const startedAt = Date.now();
              const output = document.getElementById("{element_id}");
              const update = () => {{
                const total = Math.floor((Date.now() - startedAt) / 1000);
                const hours = Math.floor(total / 3600);
                const minutes = Math.floor((total % 3600) / 60);
                const seconds = total % 60;
                output.textContent = hours
                  ? String(hours).padStart(2, "0") + ":" + String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0")
                  : String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0");
              }};
              update();
              setInterval(update, 1000);
            </script>
            """,
            height=32,
        )
    return perf_counter(), placeholder


def finish_elapsed_timer(timer_key: str, started_at: float, placeholder, label: str) -> float:
    elapsed_seconds = perf_counter() - started_at
    st.session_state[f"{timer_key}_elapsed_seconds"] = elapsed_seconds
    placeholder.empty()
    st.caption(f"{label}: {format_elapsed_time(elapsed_seconds)}")
    return elapsed_seconds
