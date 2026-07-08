# -*- coding: utf-8 -*-
"""Streamlit UI for chemical diversity diagnostics."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from modules.chemical_diversity_core import (
    Butina,
    DBSCAN,
    analyze_chemical_diversity,
    analyze_structural_communities,
)


def _safe_float_text(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _result_signature(data, smiles_col, descriptor_df):
    try:
        smiles = data[smiles_col].astype(str).fillna("").head(10000).tolist()
    except Exception:
        smiles = []
    payload = "\n".join(smiles).encode("utf-8", errors="replace")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    desc_shape = tuple(descriptor_df.shape) if isinstance(descriptor_df, pd.DataFrame) else None
    return f"{smiles_col}:{len(data)}:{digest}:{desc_shape}"


def _summary_table(summary):
    rows = [
        ("Всего молекул", summary.get("total_rows")),
        ("Валидных структур", summary.get("valid_structures")),
        ("Невалидных / пустых SMILES", summary.get("invalid_structures")),
        ("Всего пар структур", summary.get("total_pairs")),
        ("Использовано пар в расчёте", summary.get("pairs_used")),
        ("Среднее Tanimoto-сходство", _safe_float_text(summary.get("mean_tanimoto"))),
        ("Медианное Tanimoto-сходство", _safe_float_text(summary.get("median_tanimoto"))),
        ("Минимальное Tanimoto-сходство", _safe_float_text(summary.get("min_tanimoto"))),
        ("Максимальное Tanimoto-сходство", _safe_float_text(summary.get("max_tanimoto"))),
        ("Почти дубли / очень близкие пары (>0.95)", summary.get("pairs_gt_0_95")),
        ("Близкие структурные аналоги (>0.85)", summary.get("pairs_gt_0_85")),
        ("Одиночные вещества (max similarity <0.30)", summary.get("unique_molecules_lt_0_30")),
        ("Структурных кластеров", summary.get("n_clusters")),
        ("Крупнейший кластер", summary.get("largest_cluster_size")),
        ("Крупнейший кластер, %", _safe_float_text(summary.get("largest_cluster_percent"), digits=1)),
        ("Одиночных кластеров", summary.get("singleton_clusters")),
        ("Dense area", summary.get("csa_dense_area")),
        ("Moderate area", summary.get("csa_moderate_area")),
        ("Sparse area", summary.get("csa_sparse_area")),
        ("Singleton / outlier", summary.get("csa_singleton_outlier")),
        ("Exact duplicates", summary.get("csa_exact_duplicates")),
        ("Near duplicates", summary.get("csa_near_duplicates")),
        ("Connected components", summary.get("csa_connected_components")),
        ("Размер крупнейшей компоненты", summary.get("csa_largest_component_size")),
    ]
    return pd.DataFrame(rows, columns=["Показатель", "Значение"])


def _descriptor_summary_table(descriptor_space):
    if not isinstance(descriptor_space, dict) or not descriptor_space:
        return pd.DataFrame()
    rows = []
    for key, label in [
        ("n_descriptor_rows", "Строк в descriptor-space"),
        ("n_descriptor_columns", "Числовых дескрипторов"),
        ("median_nearest_distance", "Медианная дистанция до ближайшего соседа"),
        ("mean_nearest_distance", "Средняя дистанция до ближайшего соседа"),
        ("max_nearest_distance", "Максимальная дистанция до ближайшего соседа"),
        ("pca_explained_variance_1", "PCA PC1 объясняет дисперсии"),
        ("pca_explained_variance_2", "PCA PC2 объясняет дисперсии"),
        ("status", "Статус"),
    ]:
        if key in descriptor_space and key != "pca_coordinates":
            value = descriptor_space[key]
            if isinstance(value, float):
                value = _safe_float_text(value, digits=4)
            rows.append({"Показатель": label, "Значение": value})
    return pd.DataFrame(rows)


def _show_compact_figure(fig, width=720):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    buffer.seek(0)
    st.image(buffer, width=int(width))
    plt.close(fig)


def _make_final_chemical_space_figure(map_df, edges_df, color_by="csa_class", size_by="close_analog_count", show_outlier_labels=True):
    if not isinstance(map_df, pd.DataFrame) or map_df.empty:
        return None

    plot_df = map_df.copy()
    plot_df["marker_size"] = pd.to_numeric(plot_df.get(size_by, 0), errors="coerce").fillna(0.0)
    plot_df["marker_size"] = 9.0 + np.sqrt(plot_df["marker_size"].clip(lower=0.0) + 1.0) * 5.0

    class_colors = {
        "Dense area": "#2563eb",
        "Moderate area": "#16a34a",
        "Sparse area": "#f59e0b",
        "Singleton / outlier": "#dc2626",
    }
    fig = go.Figure()

    if isinstance(edges_df, pd.DataFrame) and not edges_df.empty:
        coords = plot_df.reset_index(drop=True)
        edge_x = []
        edge_y = []
        for _, edge in edges_df.iterrows():
            try:
                src = coords.iloc[int(edge["source"])]
                dst = coords.iloc[int(edge["target"])]
            except Exception:
                continue
            edge_x.extend([src["csa_x"], dst["csa_x"], None])
            edge_y.extend([src["csa_y"], dst["csa_y"], None])
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.7, color="rgba(120, 130, 145, 0.32)"),
                hoverinfo="skip",
                showlegend=False,
                name="Близкие аналоги",
            ))

    color_values = plot_df.get(color_by, plot_df.get("csa_class", ""))
    if color_by == "csa_class":
        for csa_class, group in plot_df.groupby("csa_class", dropna=False):
            labels = np.where(
                show_outlier_labels & group["is_structural_outlier"].astype(bool),
                group["name"],
                "",
            )
            fig.add_trace(go.Scatter(
                x=group["csa_x"],
                y=group["csa_y"],
                mode="markers+text" if show_outlier_labels else "markers",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=group["marker_size"],
                    color=class_colors.get(str(csa_class), "#64748b"),
                    opacity=0.88,
                    line=dict(width=0.6, color="white"),
                ),
                customdata=np.stack([
                    group["name"].astype(str),
                    group["SMILES"].astype(str),
                    group["nearest_neighbor"].astype(str),
                    group["nearest_neighbor_tanimoto"].astype(str),
                    group["close_analog_count"].astype(str),
                    group["local_density"].astype(str),
                    group["connected_component"].astype(str),
                    group["canonical_smiles"].astype(str),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "SMILES: %{customdata[1]}<br>"
                    "CSA-class: " + str(csa_class) + "<br>"
                    "Ближайший аналог: %{customdata[2]}<br>"
                    "Tanimoto: %{customdata[3]}<br>"
                    "Близких аналогов: %{customdata[4]}<br>"
                    "local_density: %{customdata[5]}<br>"
                    "connected_component: %{customdata[6]}<br>"
                    "canonical SMILES: %{customdata[7]}"
                    "<extra></extra>"
                ),
                name=str(csa_class),
            ))
    else:
        numeric_color = pd.to_numeric(color_values, errors="coerce")
        fig.add_trace(go.Scatter(
            x=plot_df["csa_x"],
            y=plot_df["csa_y"],
            mode="markers",
            marker=dict(
                size=plot_df["marker_size"],
                color=numeric_color,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_by),
                opacity=0.88,
                line=dict(width=0.6, color="white"),
            ),
            text=plot_df["name"],
            customdata=np.stack([
                plot_df["SMILES"].astype(str),
                plot_df["csa_class"].astype(str),
                plot_df["nearest_neighbor"].astype(str),
                plot_df["nearest_neighbor_tanimoto"].astype(str),
                plot_df["close_analog_count"].astype(str),
                plot_df["local_density"].astype(str),
                plot_df["connected_component"].astype(str),
                plot_df["canonical_smiles"].astype(str),
            ], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "SMILES: %{customdata[0]}<br>"
                "CSA-class: %{customdata[1]}<br>"
                "Ближайший аналог: %{customdata[2]}<br>"
                "Tanimoto: %{customdata[3]}<br>"
                "Близких аналогов: %{customdata[4]}<br>"
                "local_density: %{customdata[5]}<br>"
                "connected_component: %{customdata[6]}<br>"
                "canonical SMILES: %{customdata[7]}"
                "<extra></extra>"
            ),
            name=color_by,
        ))

    fig.update_layout(
        title="Итоговая карта химического пространства",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=650,
        template="plotly_white",
        legend_title_text="CSA-class",
        margin=dict(l=20, r=20, t=70, b=35),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig


def _render_final_chemical_space(result):
    summary = result.get("summary", {})
    final_space = result.get("final_chemical_space", {})
    if not isinstance(final_space, dict):
        return
    map_df = final_space.get("map", pd.DataFrame())
    edges_df = final_space.get("edges", pd.DataFrame())
    if not isinstance(map_df, pd.DataFrame) or map_df.empty:
        st.info("Итоговая карта химического пространства недоступна: нужно хотя бы одно валидное SMILES.")
        return

    st.markdown("### Итоговая карта химического пространства")
    st.caption(
        "Карта построена по структурному сходству Morgan/Tanimoto и показывает распределение молекул "
        "в химическом пространстве датасета."
    )
    st.caption(
        "Итоговая карта химического пространства построена на основе Morgan fingerprints и Tanimoto similarity. "
        "Чем ближе расположены точки, тем выше структурное сходство молекул. Изолированные точки и малые компоненты "
        "указывают на участки химического пространства, слабо представленные аналогами."
    )

    if final_space.get("sampled"):
        st.warning(
            "Набор большой, поэтому итоговая карта построена по воспроизводимой выборке "
            f"{final_space.get('displayed_structures')} из {final_space.get('total_valid_structures')} валидных структур."
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Всего молекул", summary.get("total_rows", "—"))
    metric_cols[1].metric("Валидных SMILES", summary.get("valid_structures", "—"))
    metric_cols[2].metric("Невалидных SMILES", summary.get("invalid_structures", "—"))
    metric_cols[3].metric("Компонент", final_space.get("n_components", "—"))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Dense area", summary.get("csa_dense_area", 0))
    metric_cols[1].metric("Moderate area", summary.get("csa_moderate_area", 0))
    metric_cols[2].metric("Sparse area", summary.get("csa_sparse_area", 0))
    metric_cols[3].metric("Singleton / outlier", summary.get("csa_singleton_outlier", 0))

    controls = st.columns([1.2, 1.0, 1.0])
    with controls[0]:
        color_options = ["csa_class"]
        if "experimental_value" in map_df.columns and pd.to_numeric(map_df["experimental_value"], errors="coerce").notna().any():
            color_options.append("experimental_value")
        color_by = st.selectbox("Окраска точек", color_options, key="chemical_space_final_color_by")
    with controls[1]:
        size_by = st.selectbox(
            "Размер точек",
            ["close_analog_count", "local_density"],
            key="chemical_space_final_size_by",
        )
    with controls[2]:
        show_labels = st.checkbox(
            "Показывать подписи выбросов",
            value=True,
            key="chemical_space_final_show_outlier_labels",
        )

    fig = _make_final_chemical_space_figure(
        map_df=map_df,
        edges_df=edges_df,
        color_by=color_by,
        size_by=size_by,
        show_outlier_labels=show_labels,
    )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

        html = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
        csv = map_df.to_csv(index=False).encode("utf-8-sig")
        dl_cols = st.columns(3)
        dl_cols[0].download_button(
            "Скачать карту HTML",
            data=html,
            file_name="final_chemical_space_map.html",
            mime="text/html",
            key="download_final_chemical_space_html",
        )
        try:
            png = fig.to_image(format="png", scale=2)
        except Exception:
            png = None
        if png:
            dl_cols[1].download_button(
                "Скачать карту PNG",
                data=png,
                file_name="final_chemical_space_map.png",
                mime="image/png",
                key="download_final_chemical_space_png",
            )
        else:
            dl_cols[1].caption("PNG-экспорт недоступен: не установлен kaleido.")
        dl_cols[2].download_button(
            "Скачать CSA-таблицу CSV",
            data=csv,
            file_name="chemical_space_csa_table.csv",
            mime="text/csv",
            key="download_final_chemical_space_csv",
        )

    with st.expander("Таблица ближайших аналогов", expanded=False):
        nearest_df = final_space.get("nearest_neighbors", pd.DataFrame())
        if isinstance(nearest_df, pd.DataFrame) and not nearest_df.empty:
            st.dataframe(nearest_df, width="stretch", hide_index=True)
        else:
            st.info("Ближайшие аналоги не рассчитаны.")

    with st.expander("Дубликаты и почти дубликаты", expanded=False):
        duplicate_df = final_space.get("duplicates", pd.DataFrame())
        if isinstance(duplicate_df, pd.DataFrame) and not duplicate_df.empty:
            st.dataframe(duplicate_df, width="stretch", hide_index=True)
        else:
            st.info("Дубликаты и почти дубликаты не найдены.")

    with st.expander("Одиночные вещества и структурные выбросы", expanded=False):
        outliers = map_df[map_df["is_structural_outlier"].astype(bool)].copy()
        if not outliers.empty:
            st.dataframe(
                outliers[[
                    "name",
                    "SMILES",
                    "nearest_neighbor",
                    "nearest_neighbor_tanimoto",
                    "close_analog_count",
                    "local_density",
                    "connected_component",
                    "csa_class",
                ]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Структурные выбросы по текущему порогу не найдены.")


def _make_structural_communities_figure(nodes_df, edges_df, color_by, size_by, show_singleton_labels, show_all_labels):
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return None

    plot_df = nodes_df.copy()
    if "method" not in plot_df.columns:
        plot_df["method"] = ""
    if "component_id" not in plot_df.columns:
        plot_df["component_id"] = plot_df.get("group_id", "")
    if size_by == "одинаковый":
        plot_df["marker_size"] = 13.0
    else:
        size_col = {
            "по размеру группы": "group_size",
            "по числу связей": "degree",
            "по числу близких аналогов": "close_analog_count",
        }.get(size_by, "group_size")
        values = pd.to_numeric(plot_df.get(size_col, 1), errors="coerce").fillna(1.0).clip(lower=0.0)
        plot_df["marker_size"] = 9.0 + np.sqrt(values + 1.0) * 5.0

    color_col = {
        "по cluster/component id": "group_id",
        "по размеру группы": "group_size",
        "по singleton status": "is_singleton_selected",
        "по числу связей": "degree",
        "по nearest_neighbor_tanimoto": "nearest_neighbor_tanimoto",
        "по csa_class": "csa_class",
    }.get(color_by, "group_id")

    fig = go.Figure()
    if isinstance(edges_df, pd.DataFrame) and not edges_df.empty:
        coords = plot_df.set_index("node_index", drop=False)
        edge_x = []
        edge_y = []
        for _, edge in edges_df.iterrows():
            try:
                src = coords.loc[int(edge["source"])]
                dst = coords.loc[int(edge["target"])]
            except Exception:
                continue
            edge_x.extend([src["csa_x"], dst["csa_x"], None])
            edge_y.extend([src["csa_y"], dst["csa_y"], None])
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.8, color="rgba(90, 100, 115, 0.30)"),
                hoverinfo="skip",
                showlegend=False,
                name="Связи аналогов",
            ))

    label_mask = plot_df["is_singleton_selected"].astype(bool) | plot_df["is_small_isolated_group"].astype(bool) | plot_df["is_noise"].astype(bool)
    labels = np.where(show_all_labels, plot_df["name"], np.where(show_singleton_labels & label_mask, plot_df["name"], ""))

    if color_col in {"csa_class", "is_singleton_selected"}:
        for value, group in plot_df.groupby(color_col, dropna=False):
            group_labels = pd.Series(labels, index=plot_df.index).loc[group.index]
            color = "#dc2626" if bool(value) and color_col == "is_singleton_selected" else None
            fig.add_trace(go.Scatter(
                x=group["csa_x"],
                y=group["csa_y"],
                mode="markers+text" if (show_singleton_labels or show_all_labels) else "markers",
                text=group_labels,
                textposition="top center",
                marker=dict(
                    size=group["marker_size"],
                    color=color,
                    opacity=0.88,
                    line=dict(width=0.8, color=np.where(group["is_singleton_selected"].astype(bool), "#dc2626", "white")),
                ),
                customdata=np.stack([
                    group["name"].astype(str),
                    group["SMILES"].astype(str),
                    group["method"].astype(str),
                    group["group_id"].astype(str),
                    group["group_size"].astype(str),
                    group["degree"].astype(str),
                    group["nearest_neighbor"].astype(str),
                    group["nearest_neighbor_tanimoto"].astype(str),
                    group["is_singleton_selected"].astype(str),
                    group["is_noise"].astype(str),
                    group["csa_class"].astype(str),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "SMILES: %{customdata[1]}<br>"
                    "Метод: %{customdata[2]}<br>"
                    "cluster/component id: %{customdata[3]}<br>"
                    "cluster size: %{customdata[4]}<br>"
                    "degree: %{customdata[5]}<br>"
                    "nearest neighbor: %{customdata[6]}<br>"
                    "nearest Tanimoto: %{customdata[7]}<br>"
                    "singleton: %{customdata[8]}<br>"
                    "noise: %{customdata[9]}<br>"
                    "csa_class: %{customdata[10]}"
                    "<extra></extra>"
                ),
                name=str(value),
            ))
    else:
        numeric_color = pd.to_numeric(plot_df[color_col], errors="coerce")
        fig.add_trace(go.Scatter(
            x=plot_df["csa_x"],
            y=plot_df["csa_y"],
            mode="markers+text" if (show_singleton_labels or show_all_labels) else "markers",
            text=labels,
            textposition="top center",
            marker=dict(
                size=plot_df["marker_size"],
                color=numeric_color,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title=color_col),
                opacity=0.88,
                line=dict(width=0.8, color=np.where(plot_df["is_singleton_selected"].astype(bool), "#dc2626", "white")),
            ),
            customdata=np.stack([
                plot_df["SMILES"].astype(str),
                plot_df["method"].astype(str),
                plot_df["group_id"].astype(str),
                plot_df["group_size"].astype(str),
                plot_df["degree"].astype(str),
                plot_df["nearest_neighbor"].astype(str),
                plot_df["nearest_neighbor_tanimoto"].astype(str),
                plot_df["is_singleton_selected"].astype(str),
                plot_df["is_noise"].astype(str),
                plot_df["csa_class"].astype(str),
            ], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "SMILES: %{customdata[0]}<br>"
                "Метод: %{customdata[1]}<br>"
                "cluster/component id: %{customdata[2]}<br>"
                "cluster size: %{customdata[3]}<br>"
                "degree: %{customdata[4]}<br>"
                "nearest neighbor: %{customdata[5]}<br>"
                "nearest Tanimoto: %{customdata[6]}<br>"
                "singleton: %{customdata[7]}<br>"
                "noise: %{customdata[8]}<br>"
                "csa_class: %{customdata[9]}"
                "<extra></extra>"
            ),
            name=color_col,
        ))

    fig.update_layout(
        title="Карта структурных сообществ и одиночных веществ",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=620,
        template="plotly_white",
        legend_title_text=color_by,
        margin=dict(l=20, r=20, t=70, b=35),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def _cached_structural_communities(
    map_df,
    similarity_matrix,
    method,
    threshold,
    top_k,
    min_cluster_size,
    butina_cutoff,
    dbscan_eps,
    dbscan_min_samples,
    singleton_criterion,
):
    return analyze_structural_communities(
        map_df=map_df,
        similarity_matrix=np.asarray(similarity_matrix, dtype=float),
        method=method,
        threshold=float(threshold),
        top_k=int(top_k),
        min_cluster_size=int(min_cluster_size),
        butina_cutoff=float(butina_cutoff),
        dbscan_eps=float(dbscan_eps),
        dbscan_min_samples=int(dbscan_min_samples),
        singleton_criterion=singleton_criterion,
    )


def _render_structural_communities_block(result):
    final_space = result.get("final_chemical_space", {})
    if not isinstance(final_space, dict):
        return
    map_df = final_space.get("map", pd.DataFrame())
    similarity_matrix = final_space.get("similarity_matrix")
    if not isinstance(map_df, pd.DataFrame) or map_df.empty or similarity_matrix is None:
        return

    st.markdown("### Карта структурных сообществ и одиночных веществ")
    st.caption(
        "График показывает разбиение датасета на структурные группы и выделяет вещества, "
        "не имеющие достаточно близких аналогов."
    )
    st.caption(
        "Этот график показывает не только положение молекул в химическом пространстве, но и их структурные сообщества. "
        "В зависимости от выбранного метода вещества разбиваются на группы по структурному сходству, а одиночные вещества "
        "и малые изолированные группы выделяются отдельно."
    )
    st.info(
        "Важно: принадлежность к группе определяется не визуальным расстоянием между точками на 2D-карте, "
        "а исходным структурным сходством между молекулами."
    )

    method_options = ["Connected components"]
    if Butina is not None:
        method_options.append("Butina clustering")
    if DBSCAN is not None:
        method_options.append("DBSCAN")
    method_options.extend(["Similarity network", "Singletons only"])
    method = st.selectbox("Способ группировки", method_options, key="chemical_space_communities_method")
    if Butina is None:
        st.caption("Butina clustering недоступен в текущем окружении RDKit.")
    if DBSCAN is None:
        st.caption("DBSCAN недоступен: scikit-learn не найден.")

    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = st.slider("Tanimoto threshold", 0.50, 0.95, 0.75, 0.01, key="chemical_space_communities_threshold")
    with c2:
        top_k = st.slider("top-k neighbors", 1, 10, 5, 1, key="chemical_space_communities_top_k")
    with c3:
        small_limit = st.slider("Размер малой группы", 1, 5, 3, 1, key="chemical_space_communities_small_limit")

    c4, c5, c6 = st.columns(3)
    with c4:
        butina_cutoff = st.slider("Butina distance cutoff", 0.05, 0.50, 0.20, 0.01, key="chemical_space_communities_butina_cutoff")
    with c5:
        dbscan_eps = st.slider("DBSCAN eps", 0.05, 0.70, 0.25, 0.01, key="chemical_space_communities_dbscan_eps")
    with c6:
        dbscan_min_samples = st.slider("DBSCAN min_samples", 1, 10, 2, 1, key="chemical_space_communities_dbscan_min_samples")

    c7, c8, c9 = st.columns(3)
    with c7:
        singleton_criterion = st.selectbox(
            "Критерий одиночества",
            [
                "combined",
                "component size == 1",
                "no neighbors above threshold",
                "cluster size <= N",
                "DBSCAN noise",
            ],
            key="chemical_space_communities_singleton_criterion",
        )
    with c8:
        color_by = st.selectbox(
            "Окраска точек",
            [
                "по cluster/component id",
                "по размеру группы",
                "по singleton status",
                "по числу связей",
                "по nearest_neighbor_tanimoto",
                "по csa_class",
            ],
            key="chemical_space_communities_color_by",
        )
    with c9:
        size_by = st.selectbox(
            "Размер точек",
            ["одинаковый", "по размеру группы", "по числу связей", "по числу близких аналогов"],
            key="chemical_space_communities_size_by",
        )

    f1, f2, f3, f4 = st.columns(4)
    only_singletons = f1.checkbox("Только singleton", value=(method == "Singletons only"), key="chemical_space_communities_only_singletons")
    only_small = f2.checkbox("Только малые группы", value=False, key="chemical_space_communities_only_small")
    only_large = f3.checkbox("Только крупные группы", value=False, key="chemical_space_communities_only_large")
    show_singleton_labels = f4.checkbox("Подписи одиночных", value=True, key="chemical_space_communities_singleton_labels")
    show_all_labels = st.checkbox(
        "Показывать подписи всех точек",
        value=False,
        disabled=len(map_df) > 80,
        key="chemical_space_communities_all_labels",
    )

    communities = _cached_structural_communities(
        map_df,
        similarity_matrix,
        method,
        float(threshold),
        int(top_k),
        int(small_limit),
        float(butina_cutoff),
        float(dbscan_eps),
        int(dbscan_min_samples),
        singleton_criterion,
    )
    nodes = communities.get("nodes", pd.DataFrame())
    edges = communities.get("edges", pd.DataFrame())
    summary = communities.get("summary", {})

    if isinstance(nodes, pd.DataFrame) and not nodes.empty:
        filtered_nodes = nodes.copy()
        if only_singletons:
            filtered_nodes = filtered_nodes[filtered_nodes["is_singleton_selected"].astype(bool)].copy()
        if only_small:
            filtered_nodes = filtered_nodes[filtered_nodes["group_size"] <= int(small_limit)].copy()
        if only_large:
            filtered_nodes = filtered_nodes[filtered_nodes["group_size"] > int(small_limit)].copy()
        visible = set(filtered_nodes["node_index"].astype(int).tolist())
        if isinstance(edges, pd.DataFrame) and not edges.empty:
            filtered_edges = edges[
                edges["source"].astype(int).isin(visible)
                & edges["target"].astype(int).isin(visible)
            ].copy()
        else:
            filtered_edges = edges
    else:
        filtered_nodes = nodes
        filtered_edges = edges

    metrics = st.columns(4)
    metrics[0].metric("Групп / кластеров", summary.get("n_groups", "—"))
    metrics[1].metric("Singleton", summary.get("n_singletons", "—"))
    metrics[2].metric("Малых групп", summary.get("n_small_groups", "—"))
    metrics[3].metric("Крупнейшая группа", summary.get("largest_group_size", "—"))
    metrics = st.columns(4)
    metrics[0].metric("Доля крупнейшей", f"{float(summary.get('largest_group_fraction', 0.0)) * 100:.1f}%")
    metrics[1].metric("Noise", summary.get("noise_points", "—"))
    metrics[2].metric("Средний degree", _safe_float_text(summary.get("mean_degree")))
    metrics[3].metric("Без близких соседей", summary.get("no_close_neighbors", "—"))
    st.info(str(summary.get("interpretation", "")))

    fig = _make_structural_communities_figure(
        filtered_nodes,
        filtered_edges,
        color_by=color_by,
        size_by=size_by,
        show_singleton_labels=show_singleton_labels,
        show_all_labels=show_all_labels,
    )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Для выбранных фильтров нет точек для отображения.")

    with st.expander("Таблица структурных сообществ", expanded=False):
        groups = communities.get("groups", pd.DataFrame())
        if isinstance(groups, pd.DataFrame) and not groups.empty:
            st.dataframe(groups, width="stretch", hide_index=True)
        else:
            st.info("Структурные сообщества не найдены.")

    with st.expander("Таблица одиночных веществ", expanded=False):
        singletons = communities.get("singletons", pd.DataFrame())
        if isinstance(singletons, pd.DataFrame) and not singletons.empty:
            st.dataframe(singletons, width="stretch", hide_index=True)
        else:
            st.info("Одиночные вещества по выбранному критерию не найдены.")

    with st.expander("Малые изолированные группы", expanded=False):
        small_groups = communities.get("small_groups", pd.DataFrame())
        if isinstance(small_groups, pd.DataFrame) and not small_groups.empty:
            st.dataframe(small_groups, width="stretch", hide_index=True)
        else:
            st.info("Малые изолированные группы по выбранному лимиту не найдены.")


def _render_exact_pattern_map_block(result):
    final_space = result.get("final_chemical_space", {})
    if not isinstance(final_space, dict):
        return
    exact_payload = final_space.get("exact_patterns", {})
    if not isinstance(exact_payload, dict):
        return
    groups = exact_payload.get("groups", pd.DataFrame())
    if not isinstance(groups, pd.DataFrame) or groups.empty:
        st.info("Карта точных структурных паттернов недоступна: не найдено распознанных паттернов алканов.")
        return

    st.markdown("### Карта точных структурных паттернов")
    st.caption(
        "График показывает распределение датасета по точным структурным сериям алканов. "
        "Размер узла пропорционален числу веществ в серии."
    )
    st.caption(
        "Этот график агрегирует не отдельные молекулы, а точные структурные паттерны алканов. "
        "В отличие от обычной карты химического пространства, здесь показана химическая классификация "
        "по точному типу замещения: n-alkanes, 2-methylalkanes, 2,3-dimethylalkanes, "
        "2,2,4-trimethylalkanes и т.д."
    )
    st.info(
        "Эта визуализация помогает найти singleton-patterns, малочисленные паттерны и нераспознанные структуры, "
        "которые не образуют полноценной группы в пространстве."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        chart_type = st.selectbox(
            "Тип графика",
            ["Tree map", "Радиальная иерархия", "Bubble chart"],
            key="chemical_space_exact_pattern_chart_type",
        )
    with c2:
        color_by = st.selectbox(
            "Цвет узлов",
            ["по крупной серии", "по числу веществ", "по редкости группы", "по среднему значению свойства"],
            key="chemical_space_exact_pattern_color_by",
        )
    with c3:
        small_threshold = st.slider(
            "small group threshold",
            1,
            5,
            2,
            1,
            key="chemical_space_exact_pattern_small_threshold",
        )
    with c4:
        rare_threshold = st.slider(
            "rare group threshold",
            1,
            8,
            3,
            1,
            key="chemical_space_exact_pattern_rare_threshold",
        )

    plot_df = groups.copy()
    plot_df["small_group"] = plot_df["group_size"] <= int(small_threshold)
    plot_df["rare_group"] = plot_df["group_size"] <= int(rare_threshold)
    only_rare = st.checkbox("Показать только rare/singleton/unclassified groups", value=False, key="chemical_space_exact_pattern_only_rare")
    if only_rare:
        plot_df = plot_df[
            plot_df["rare_group"].astype(bool)
            | plot_df["singleton_group"].astype(bool)
            | plot_df["unclassified_group"].astype(bool)
        ].copy()
    if plot_df.empty:
        st.info("Нет паттернов для выбранного фильтра.")
        return

    color_column = {
        "по крупной серии": "broad_series",
        "по числу веществ": "group_size",
        "по редкости группы": "rare_group",
        "по среднему значению свойства": "mean_property",
    }.get(color_by, "broad_series")
    if color_column == "mean_property" and not pd.to_numeric(plot_df["mean_property"], errors="coerce").notna().any():
        color_column = "broad_series"
        st.caption("Среднее значение свойства недоступно, используется окраска по крупной серии.")

    hover_data = [
        "broad_series",
        "exact_pattern",
        "group_size",
        "dataset_fraction",
        "rare_group",
        "singleton_group",
        "unclassified_group",
        "representative_examples",
    ]
    if chart_type == "Tree map":
        fig = px.treemap(
            plot_df,
            path=["broad_series", "exact_pattern"],
            values="group_size",
            color=color_column,
            hover_data=hover_data,
            title="Карта точных структурных паттернов",
        )
    elif chart_type == "Радиальная иерархия":
        fig = px.sunburst(
            plot_df,
            path=["broad_series", "exact_pattern"],
            values="group_size",
            color=color_column,
            hover_data=hover_data,
            title="Карта точных структурных паттернов",
        )
    else:
        bubble_df = plot_df.sort_values("group_size", ascending=False).reset_index(drop=True)
        bubble_df["x"] = np.arange(len(bubble_df))
        series_codes = {name: idx for idx, name in enumerate(sorted(bubble_df["broad_series"].astype(str).unique()))}
        bubble_df["y"] = bubble_df["broad_series"].astype(str).map(series_codes)
        labels = np.where(
            bubble_df["rare_group"].astype(bool) | bubble_df["singleton_group"].astype(bool),
            bubble_df["exact_pattern"],
            "",
        )
        fig = px.scatter(
            bubble_df,
            x="x",
            y="y",
            size="group_size",
            color=color_column,
            text=labels,
            hover_data=hover_data,
            title="Карта точных структурных паттернов",
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(series_codes.values()),
            ticktext=list(series_codes.keys()),
            title="broad_series",
        )
        fig.update_xaxes(title="exact_pattern groups")
        fig.update_traces(textposition="top center")

    fig.update_layout(height=620, margin=dict(l=20, r=20, t=70, b=35))
    st.info(str(exact_payload.get("interpretation", "")))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Таблица точных структурных паттернов", expanded=False):
        st.dataframe(groups, width="stretch", hide_index=True)

    with st.expander("Малочисленные и одиночные паттерны", expanded=False):
        rare_groups = exact_payload.get("rare_groups", pd.DataFrame())
        if isinstance(rare_groups, pd.DataFrame) and not rare_groups.empty:
            st.dataframe(rare_groups, width="stretch", hide_index=True)
        else:
            st.info("Малочисленные, одиночные или нераспознанные паттерны не найдены.")

    with st.expander("Состав паттерна", expanded=False):
        members = exact_payload.get("members", pd.DataFrame())
        if isinstance(members, pd.DataFrame) and not members.empty:
            selected = st.selectbox(
                "exact_pattern",
                sorted(members["exact_pattern"].astype(str).unique()),
                key="chemical_space_exact_pattern_members_select",
            )
            st.dataframe(
                members[members["exact_pattern"].astype(str) == selected],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Состав паттернов недоступен.")


def _render_similarity_histogram(hist_df):
    st.caption("Гистограмма показывает общее распределение попарного Tanimoto-сходства в датасете.")
    if not isinstance(hist_df, pd.DataFrame) or hist_df.empty:
        st.info("Гистограмма сходства недоступна.")
        return
    chart_df = hist_df.copy()
    chart_df["Диапазон"] = chart_df.apply(
        lambda row: f"{row['similarity_from']:.2f}-{row['similarity_to']:.2f}",
        axis=1,
    )
    st.bar_chart(chart_df.set_index("Диапазон")["count"])


def _render_pca_map(pca_df):
    st.caption("Карта пространства показывает химические области: каждая точка — молекула, координаты построены PCA по Morgan fingerprints.")
    if not isinstance(pca_df, pd.DataFrame) or pca_df.empty:
        st.info("PCA-карта недоступна: нужно хотя бы 3 валидные структуры и scikit-learn.")
        return

    color_mode = "cluster_id"
    if "target" in pca_df.columns and pd.to_numeric(pca_df["target"], errors="coerce").notna().any():
        color_mode = st.radio(
            "Окраска точек",
            ["cluster_id", "target"],
            horizontal=True,
            key="chemical_diversity_pca_color_mode",
        )
    else:
        st.caption("Окраска: cluster_id. Окраска по target появится, если передано целевое свойство.")

    plot_df = pca_df.copy()
    plot_df[color_mode] = pd.to_numeric(plot_df[color_mode], errors="coerce")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    scatter = ax.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        c=plot_df[color_mode],
        cmap="viridis",
        s=24,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlabel("PCA 1", fontsize=9)
    ax.set_ylabel("PCA 2", fontsize=9)
    ax.set_title("Карта химического пространства", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, label=color_mode, shrink=0.82)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    _show_compact_figure(fig, width=720)

    with st.expander("Точки карты", expanded=False):
        st.dataframe(plot_df, width="stretch", hide_index=True)


def _render_similarity_heatmap(heatmap_payload):
    st.caption("Heatmap показывает блоки близких структур; молекулы отсортированы по cluster_id, цветовая шкала Tanimoto от 0 до 1.")
    if not isinstance(heatmap_payload, dict):
        st.info("Матрица сходства недоступна.")
        return
    matrix_df = heatmap_payload.get("matrix", pd.DataFrame())
    molecules_df = heatmap_payload.get("molecules", pd.DataFrame())
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        st.info("Матрица сходства недоступна.")
        return
    if heatmap_payload.get("sampled"):
        st.warning("В датасете больше 300 валидных структур, поэтому heatmap построен по воспроизводимой выборке.")

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    image = ax.imshow(matrix_df.values, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title("Тепловая карта Tanimoto-сходства", fontsize=10)
    ax.set_xlabel("Молекулы по cluster_id", fontsize=9)
    ax.set_ylabel("Молекулы по cluster_id", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, label="Tanimoto", shrink=0.82)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    _show_compact_figure(fig, width=620)

    with st.expander("Порядок молекул в heatmap", expanded=False):
        st.dataframe(molecules_df, width="stretch", hide_index=True)


def _render_clusters(cluster_summary, summary):
    st.caption("Кластеры показывают фрагментацию датасета: размер серий аналогов и долю крупнейшего химического семейства.")
    if not isinstance(cluster_summary, pd.DataFrame) or cluster_summary.empty:
        st.info("Кластеры не рассчитаны.")
        return

    col_a, col_b = st.columns(2)
    col_a.metric("Одиночных кластеров", summary.get("singleton_clusters", "—"))
    col_b.metric("Доля крупнейшего кластера", f"{_safe_float_text(summary.get('largest_cluster_percent'), digits=1)}%")

    chart_df = cluster_summary.sort_values("n", ascending=False).copy()
    chart_df["cluster_id"] = chart_df["cluster_id"].astype(str)
    st.bar_chart(chart_df.set_index("cluster_id")["n"])
    st.dataframe(chart_df, width="stretch", hide_index=True)


def _render_analogue_network(pca_df, network_edges):
    st.caption("Сеть показывает серии аналогов: ребро проводится между молекулами с Tanimoto выше порога близких аналогов.")
    if not isinstance(network_edges, pd.DataFrame) or network_edges.empty:
        st.info("Близких аналогов для построения сети не найдено.")
        return
    if len(network_edges) >= 500:
        st.warning("Сеть ограничена первыми 500 наиболее близкими связями, чтобы граф оставался читаемым.")
    if not isinstance(pca_df, pd.DataFrame) or pca_df.empty:
        st.dataframe(network_edges, width="stretch", hide_index=True)
        return

    node_df = pca_df.set_index("row")
    edge_df = network_edges[
        network_edges["source_row"].isin(node_df.index)
        & network_edges["target_row"].isin(node_df.index)
    ].copy()
    if edge_df.empty:
        st.info("Связи не сопоставились с PCA-картой.")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for _, edge in edge_df.iterrows():
        src = node_df.loc[int(edge["source_row"])]
        dst = node_df.loc[int(edge["target_row"])]
        ax.plot(
            [src["PC1"], dst["PC1"]],
            [src["PC2"], dst["PC2"]],
            color="#9ca3af",
            alpha=min(0.75, max(0.15, float(edge["tanimoto"]) - 0.55)),
            linewidth=0.6,
        )

    scatter = ax.scatter(
        node_df["PC1"],
        node_df["PC2"],
        c=pd.to_numeric(node_df["cluster_id"], errors="coerce"),
        cmap="viridis",
        s=24,
        alpha=0.9,
        edgecolors="none",
    )
    ax.set_title("Сеть близких аналогов на PCA-карте", fontsize=10)
    ax.set_xlabel("PCA 1", fontsize=9)
    ax.set_ylabel("PCA 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, label="cluster_id", shrink=0.82)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    _show_compact_figure(fig, width=720)

    with st.expander("Рёбра сети близких аналогов", expanded=False):
        st.dataframe(edge_df, width="stretch", hide_index=True)


def _render_pairs_and_unique(result):
    st.caption("Таблицы показывают почти дубли, близкие аналоги и самые уникальные вещества с минимальной близостью к соседям.")
    duplicate_pairs = result.get("duplicate_pairs", pd.DataFrame())
    analogue_pairs = result.get("analogue_pairs", pd.DataFrame())
    unique_table = result.get("unique_molecules", pd.DataFrame())

    st.markdown("#### Почти дубли: Tanimoto > 0.95")
    if isinstance(duplicate_pairs, pd.DataFrame) and not duplicate_pairs.empty:
        st.dataframe(duplicate_pairs, width="stretch", hide_index=True)
    else:
        st.info("Почти дубли не найдены.")

    st.markdown("#### Близкие аналоги: Tanimoto > 0.85")
    if isinstance(analogue_pairs, pd.DataFrame) and not analogue_pairs.empty:
        st.dataframe(analogue_pairs, width="stretch", hide_index=True)
    else:
        st.info("Близкие аналоги не найдены.")

    st.markdown("#### Самые уникальные вещества")
    if isinstance(unique_table, pd.DataFrame) and not unique_table.empty:
        st.dataframe(unique_table, width="stretch", hide_index=True)
    else:
        st.info("Таблица уникальных веществ недоступна.")


def render_chemical_diversity_section(
    data,
    smiles_col,
    label_col=None,
    target_col=None,
    descriptor_df=None,
    expanded=False,
):
    """Render pre-modeling chemical diversity diagnostics."""
    if not isinstance(data, pd.DataFrame) or data.empty or not smiles_col or smiles_col not in data.columns:
        return

    if not label_col:
        for candidate in ("Name", "name", "compound_id", "Compound ID", "CAS", "cas"):
            if candidate in data.columns and candidate != smiles_col:
                label_col = candidate
                break

    st.markdown("### Близость молекул и химическое разнообразие датасета")
    st.caption(
        "Диагностика показывает, является ли выборка однотипной, смешанной, "
        "перегруженной близкими аналогами или содержит одиночные структуры."
    )
    st.caption(
        "Модуль химического пространства анализирует структурную организацию датасета до построения модели. "
        "Итоговая карта показывает, является ли датасет однородным или разнородным, есть ли в нём плотные группы "
        "аналогов, одиночные вещества, структурные выбросы, дубликаты и почти дубликаты."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        cluster_threshold = st.slider(
            "Порог кластеризации Tanimoto",
            min_value=0.30,
            max_value=0.90,
            value=0.60,
            step=0.05,
            key="chemical_diversity_cluster_threshold",
        )
    with col_b:
        duplicate_threshold = st.slider(
            "Порог почти дублей",
            min_value=0.85,
            max_value=1.00,
            value=0.95,
            step=0.01,
            key="chemical_diversity_duplicate_threshold",
        )
    with col_c:
        analogue_threshold = st.slider(
            "Порог близких аналогов",
            min_value=0.60,
            max_value=0.99,
            value=0.85,
            step=0.01,
            key="chemical_diversity_analogue_threshold",
        )

    col_d, col_e, col_f, col_g = st.columns(4)
    with col_d:
        projection_method = st.selectbox(
            "Метод проекции",
            ["auto", "UMAP", "MDS", "t-SNE"],
            index=0,
            key="chemical_diversity_projection_method",
        )
    with col_e:
        radius = st.number_input(
            "Morgan radius",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
            key="chemical_diversity_morgan_radius",
        )
    with col_f:
        n_bits = st.selectbox(
            "nBits",
            [1024, 2048, 4096],
            index=1,
            key="chemical_diversity_morgan_n_bits",
        )
    with col_g:
        map_top_k = st.number_input(
            "k ближайших рёбер",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="chemical_diversity_map_top_k",
        )

    st.caption(
        "До 2000 валидных структур считается полная матрица пар. "
        "Для больших наборов используется воспроизводимая выборка пар, чтобы интерфейс не зависал."
    )

    signature = (
        f"{_result_signature(data, smiles_col, descriptor_df)}:"
        f"cluster={cluster_threshold:.2f}:dup={duplicate_threshold:.2f}:analog={analogue_threshold:.2f}:"
        f"projection={projection_method}:radius={int(radius)}:bits={int(n_bits)}:topk={int(map_top_k)}"
    )
    cached = st.session_state.get("chemical_diversity_result")
    cached_signature = st.session_state.get("chemical_diversity_signature")

    run_clicked = st.button(
        "Рассчитать химическое разнообразие",
        type="primary",
        key="run_chemical_diversity",
    )

    if run_clicked or (cached is not None and cached_signature == signature):
        if run_clicked:
            with st.spinner("Считаю Morgan fingerprints, Tanimoto-сходство и кластеры..."):
                result = analyze_chemical_diversity(
                    data=data,
                    smiles_col=smiles_col,
                    label_col=label_col,
                    target_col=target_col,
                    descriptor_df=descriptor_df,
                    radius=int(radius),
                    n_bits=int(n_bits),
                    duplicate_threshold=duplicate_threshold,
                    analogue_threshold=analogue_threshold,
                    cluster_similarity_threshold=cluster_threshold,
                    projection_method=projection_method,
                    map_edge_threshold=analogue_threshold,
                    map_edge_top_k=int(map_top_k),
                )
                st.session_state.chemical_diversity_result = result
                st.session_state.chemical_diversity_signature = signature
        else:
            result = cached

        summary = result.get("summary", {})
        status = str(summary.get("status", "не рассчитано"))
        reasons = str(summary.get("status_reasons", ""))

        if status in {"низкое разнообразие", "неоднородный датасет"}:
            st.warning(f"Статус: {status}. {reasons}")
        elif status in {"высокое разнообразие", "умеренное разнообразие"}:
            st.success(f"Статус: {status}. {reasons}")
        else:
            st.info(f"Статус: {status}. {reasons}")

        if summary.get("pairwise_mode") == "sampled":
            st.info(
                "Набор большой, поэтому пары Tanimoto посчитаны по выборке. "
                "Счётчики близких пар показаны как оценка по sampled pairs."
            )
        if summary.get("cluster_sampled"):
            st.info(
                "Кластеры построены на выборке до 2000 структур. "
                "Для полного кластерного отчёта уменьшите набор или запускайте локально отдельным расчётом."
            )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Среднее Tanimoto", _safe_float_text(summary.get("mean_tanimoto")))
        metric_cols[1].metric("Пары >0.95", summary.get("pairs_gt_0_95", "—"))
        metric_cols[2].metric("Кластеры", summary.get("n_clusters", "—"))
        metric_cols[3].metric("Одиночные", summary.get("unique_molecules_lt_0_30", "—"))

        st.dataframe(_summary_table(summary), width="stretch", hide_index=True)
        (
            tab_distribution,
            tab_space,
            tab_heatmap,
            tab_clusters,
            tab_network,
            tab_pairs,
        ) = st.tabs([
            "Распределение сходства",
            "Карта химического пространства",
            "Тепловая карта сходства",
            "Кластеры",
            "Сеть близких аналогов",
            "Пары и одиночные вещества",
        ])

        with tab_distribution:
            _render_similarity_histogram(result.get("similarity_histogram", pd.DataFrame()))
            descriptor_space = result.get("descriptor_space", {})
            descriptor_table = _descriptor_summary_table(descriptor_space)
            if not descriptor_table.empty:
                with st.expander("Descriptor-space diversity", expanded=False):
                    st.dataframe(descriptor_table, width="stretch", hide_index=True)
                    coords = descriptor_space.get("pca_coordinates") if isinstance(descriptor_space, dict) else None
                    if isinstance(coords, pd.DataFrame) and {"PC1", "PC2"}.issubset(coords.columns):
                        st.scatter_chart(coords[["PC1", "PC2"]])

        with tab_space:
            _render_pca_map(result.get("fingerprint_pca", pd.DataFrame()))

        with tab_heatmap:
            _render_similarity_heatmap(result.get("similarity_heatmap", {}))

        with tab_clusters:
            _render_clusters(result.get("cluster_summary", pd.DataFrame()), summary)

        with tab_network:
            _render_analogue_network(
                result.get("fingerprint_pca", pd.DataFrame()),
                result.get("network_edges", pd.DataFrame()),
            )

        with tab_pairs:
            _render_pairs_and_unique(result)
            invalid_df = result.get("invalid_structures", pd.DataFrame())
            if isinstance(invalid_df, pd.DataFrame) and not invalid_df.empty:
                with st.expander("Невалидные SMILES", expanded=False):
                    st.dataframe(invalid_df, width="stretch", hide_index=True)

        _render_final_chemical_space(result)
        _render_structural_communities_block(result)
        _render_exact_pattern_map_block(result)
