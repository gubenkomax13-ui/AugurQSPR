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
from modules.i18n import t
from modules.chemical_diversity_core import Butina, DBSCAN, analyze_chemical_diversity, analyze_structural_communities

def _safe_float_text(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return '—'
    if not np.isfinite(value):
        return '—'
    return f'{value:.{digits}f}'

def _result_signature(data, smiles_col, descriptor_df):
    try:
        smiles = data[smiles_col].astype(str).fillna('').head(10000).tolist()
    except Exception:
        smiles = []
    payload = '\n'.join(smiles).encode('utf-8', errors='replace')
    digest = hashlib.sha1(payload).hexdigest()[:12]
    desc_shape = tuple(descriptor_df.shape) if isinstance(descriptor_df, pd.DataFrame) else None
    return f'{smiles_col}:{len(data)}:{digest}:{desc_shape}'

def _summary_table(summary):
    rows = [(t('chemical_diversity.text_10dd2c1e6e'), summary.get('total_rows')), (t('chemical_diversity.text_9afccebb18'), summary.get('valid_structures')), (t('chemical_diversity.text_ae171e42c2'), summary.get('invalid_structures')), (t('chemical_diversity.text_0b9edeed53'), summary.get('total_pairs')), (t('chemical_diversity.text_ee43026d14'), summary.get('pairs_used')), (t('chemical_diversity.text_afcb806109'), _safe_float_text(summary.get('mean_tanimoto'))), (t('chemical_diversity.text_227e99c26c'), _safe_float_text(summary.get('median_tanimoto'))), (t('chemical_diversity.text_e2f0a5ae0b'), _safe_float_text(summary.get('min_tanimoto'))), (t('chemical_diversity.text_6893a74204'), _safe_float_text(summary.get('max_tanimoto'))), (t('chemical_diversity.text_4c27e37460'), summary.get('pairs_gt_0_95')), (t('chemical_diversity.text_966232685c'), summary.get('pairs_gt_0_85')), (t('chemical_diversity.text_f95557eeb4'), summary.get('unique_molecules_lt_0_30')), (t('chemical_diversity.text_e72e9c2b87'), summary.get('n_clusters')), (t('chemical_diversity.text_736fff1164'), summary.get('largest_cluster_size')), (t('chemical_diversity.text_ef342379ee'), _safe_float_text(summary.get('largest_cluster_percent'), digits=1)), (t('chemical_diversity.text_0dec38a6a7'), summary.get('singleton_clusters')), ('Dense area', summary.get('csa_dense_area')), ('Moderate area', summary.get('csa_moderate_area')), ('Sparse area', summary.get('csa_sparse_area')), ('Singleton / outlier', summary.get('csa_singleton_outlier')), ('Exact duplicates', summary.get('csa_exact_duplicates')), ('Near duplicates', summary.get('csa_near_duplicates')), ('Connected components', summary.get('csa_connected_components')), (t('chemical_diversity.text_c109a2a5d3'), summary.get('csa_largest_component_size'))]
    return pd.DataFrame(rows, columns=[t('chemical_diversity.text_bce6874778'), t('chemical_diversity.text_7b46d8ccf6')])

def _descriptor_summary_table(descriptor_space):
    if not isinstance(descriptor_space, dict) or not descriptor_space:
        return pd.DataFrame()
    rows = []
    for (key, label) in [('n_descriptor_rows', t('chemical_diversity.text_08a4a61409')), ('n_descriptor_columns', t('chemical_diversity.text_61981fe2c3')), ('median_nearest_distance', t('chemical_diversity.text_097f13e16c')), ('mean_nearest_distance', t('chemical_diversity.text_7f2a9fa02a')), ('max_nearest_distance', t('chemical_diversity.text_fd3615a75d')), ('pca_explained_variance_1', t('chemical_diversity.text_da1ab80301')), ('pca_explained_variance_2', t('chemical_diversity.text_0432843a3d')), ('status', t('chemical_diversity.text_7203f7a4ff'))]:
        if key in descriptor_space and key != 'pca_coordinates':
            value = descriptor_space[key]
            if isinstance(value, float):
                value = _safe_float_text(value, digits=4)
            rows.append({t('chemical_diversity.text_bce6874778'): label, t('chemical_diversity.text_7b46d8ccf6'): value})
    return pd.DataFrame(rows)

def _show_compact_figure(fig, width=720):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    st.image(buffer, width=int(width))
    plt.close(fig)

def _make_final_chemical_space_figure(map_df, edges_df, color_by='csa_class', size_by='close_analog_count', show_outlier_labels=True):
    if not isinstance(map_df, pd.DataFrame) or map_df.empty:
        return None
    plot_df = map_df.copy()
    plot_df['marker_size'] = pd.to_numeric(plot_df.get(size_by, 0), errors='coerce').fillna(0.0)
    plot_df['marker_size'] = 9.0 + np.sqrt(plot_df['marker_size'].clip(lower=0.0) + 1.0) * 5.0
    class_colors = {'Dense area': '#2563eb', 'Moderate area': '#16a34a', 'Sparse area': '#f59e0b', 'Singleton / outlier': '#dc2626'}
    fig = go.Figure()
    if isinstance(edges_df, pd.DataFrame) and (not edges_df.empty):
        coords = plot_df.reset_index(drop=True)
        edge_x = []
        edge_y = []
        for (_, edge) in edges_df.iterrows():
            try:
                src = coords.iloc[int(edge['source'])]
                dst = coords.iloc[int(edge['target'])]
            except Exception:
                continue
            edge_x.extend([src['csa_x'], dst['csa_x'], None])
            edge_y.extend([src['csa_y'], dst['csa_y'], None])
        if edge_x:
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.7, color='rgba(120, 130, 145, 0.32)'), hoverinfo='skip', showlegend=False, name=t('chemical_diversity.text_2649168f1e')))
    color_values = plot_df.get(color_by, plot_df.get('csa_class', ''))
    if color_by == 'csa_class':
        for (csa_class, group) in plot_df.groupby('csa_class', dropna=False):
            labels = np.where(show_outlier_labels & group['is_structural_outlier'].astype(bool), group['name'], '')
            fig.add_trace(go.Scatter(x=group['csa_x'], y=group['csa_y'], mode='markers+text' if show_outlier_labels else 'markers', text=labels, textposition='top center', marker=dict(size=group['marker_size'], color=class_colors.get(str(csa_class), '#64748b'), opacity=0.88, line=dict(width=0.6, color='white')), customdata=np.stack([group['name'].astype(str), group['SMILES'].astype(str), group['nearest_neighbor'].astype(str), group['nearest_neighbor_tanimoto'].astype(str), group['close_analog_count'].astype(str), group['local_density'].astype(str), group['connected_component'].astype(str), group['canonical_smiles'].astype(str)], axis=-1), hovertemplate='<b>%{customdata[0]}</b><br>SMILES: %{customdata[1]}<br>CSA-class: ' + str(csa_class) + t('chemical_diversity.text_560094861d'), name=str(csa_class)))
    else:
        numeric_color = pd.to_numeric(color_values, errors='coerce')
        fig.add_trace(go.Scatter(x=plot_df['csa_x'], y=plot_df['csa_y'], mode='markers', marker=dict(size=plot_df['marker_size'], color=numeric_color, colorscale='Viridis', showscale=True, colorbar=dict(title=color_by), opacity=0.88, line=dict(width=0.6, color='white')), text=plot_df['name'], customdata=np.stack([plot_df['SMILES'].astype(str), plot_df['csa_class'].astype(str), plot_df['nearest_neighbor'].astype(str), plot_df['nearest_neighbor_tanimoto'].astype(str), plot_df['close_analog_count'].astype(str), plot_df['local_density'].astype(str), plot_df['connected_component'].astype(str), plot_df['canonical_smiles'].astype(str)], axis=-1), hovertemplate=t('chemical_diversity.text_db7498efc9'), name=color_by))
    fig.update_layout(title=t('chemical_diversity.text_e46d8938dc'), xaxis_title='Dimension 1', yaxis_title='Dimension 2', height=650, template='plotly_white', legend_title_text='CSA-class', margin=dict(l=20, r=20, t=70, b=35))
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig

def _render_final_chemical_space(result):
    summary = result.get('summary', {})
    final_space = result.get('final_chemical_space', {})
    if not isinstance(final_space, dict):
        return
    map_df = final_space.get('map', pd.DataFrame())
    edges_df = final_space.get('edges', pd.DataFrame())
    if not isinstance(map_df, pd.DataFrame) or map_df.empty:
        st.info(t('chemical_diversity.text_02d3396d06'))
        return
    st.markdown(t('chemical_diversity.text_c608c45b2a'))
    st.caption(t('chemical_diversity.text_3c33ebf200'))
    st.caption(t('chemical_diversity.text_d8e6a5ec9b'))
    if final_space.get('sampled'):
        st.warning(t(
            'chemical_diversity.final_map_sampled_warning',
            displayed=final_space.get('displayed_structures'),
            total=final_space.get('total_valid_structures'),
        ))
    metric_cols = st.columns(4)
    metric_cols[0].metric(t('chemical_diversity.text_10dd2c1e6e'), summary.get('total_rows', '—'))
    metric_cols[1].metric(t('chemical_diversity.text_91fc951aca'), summary.get('valid_structures', '—'))
    metric_cols[2].metric(t('chemical_diversity.text_9804d4a5c0'), summary.get('invalid_structures', '—'))
    metric_cols[3].metric(t('chemical_diversity.text_49dec84b23'), final_space.get('n_components', '—'))
    metric_cols = st.columns(4)
    metric_cols[0].metric('Dense area', summary.get('csa_dense_area', 0))
    metric_cols[1].metric('Moderate area', summary.get('csa_moderate_area', 0))
    metric_cols[2].metric('Sparse area', summary.get('csa_sparse_area', 0))
    metric_cols[3].metric('Singleton / outlier', summary.get('csa_singleton_outlier', 0))
    controls = st.columns([1.2, 1.0, 1.0])
    with controls[0]:
        color_options = ['csa_class']
        if 'experimental_value' in map_df.columns and pd.to_numeric(map_df['experimental_value'], errors='coerce').notna().any():
            color_options.append('experimental_value')
        color_by = st.selectbox(t('chemical_diversity.text_d11b85826f'), color_options, key='chemical_space_final_color_by')
    with controls[1]:
        size_by = st.selectbox(t('chemical_diversity.text_686b1c751c'), ['close_analog_count', 'local_density'], key='chemical_space_final_size_by')
    with controls[2]:
        show_labels = st.checkbox(t('chemical_diversity.text_498d3ea747'), value=True, key='chemical_space_final_show_outlier_labels')
    fig = _make_final_chemical_space_figure(map_df=map_df, edges_df=edges_df, color_by=color_by, size_by=size_by, show_outlier_labels=show_labels)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        html = fig.to_html(include_plotlyjs='cdn', full_html=True).encode('utf-8')
        csv = map_df.to_csv(index=False).encode('utf-8-sig')
        dl_cols = st.columns(3)
        dl_cols[0].download_button(t('chemical_diversity.text_f11db0c93d'), data=html, file_name='final_chemical_space_map.html', mime='text/html', key='download_final_chemical_space_html')
        try:
            png = fig.to_image(format='png', scale=2)
        except Exception:
            png = None
        if png:
            dl_cols[1].download_button(t('chemical_diversity.text_b1eb9744e7'), data=png, file_name='final_chemical_space_map.png', mime='image/png', key='download_final_chemical_space_png')
        else:
            dl_cols[1].caption(t('chemical_diversity.text_1ad04dcdaf'))
        dl_cols[2].download_button(t('chemical_diversity.text_9e8188bf73'), data=csv, file_name='chemical_space_csa_table.csv', mime='text/csv', key='download_final_chemical_space_csv')
    with st.expander(t('chemical_diversity.text_da1f5c0444'), expanded=False):
        nearest_df = final_space.get('nearest_neighbors', pd.DataFrame())
        if isinstance(nearest_df, pd.DataFrame) and (not nearest_df.empty):
            st.dataframe(nearest_df, width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_60f4b6240f'))
    with st.expander(t('chemical_diversity.text_9505fbdde4'), expanded=False):
        duplicate_df = final_space.get('duplicates', pd.DataFrame())
        if isinstance(duplicate_df, pd.DataFrame) and (not duplicate_df.empty):
            st.dataframe(duplicate_df, width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_ca09a4bf8a'))
    with st.expander(t('chemical_diversity.text_c99adacc8d'), expanded=False):
        outliers = map_df[map_df['is_structural_outlier'].astype(bool)].copy()
        if not outliers.empty:
            st.dataframe(outliers[['name', 'SMILES', 'nearest_neighbor', 'nearest_neighbor_tanimoto', 'close_analog_count', 'local_density', 'connected_component', 'csa_class']], width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_a15ddd267f'))

def _make_structural_communities_figure(nodes_df, edges_df, color_by, size_by, show_singleton_labels, show_all_labels):
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return None
    plot_df = nodes_df.copy()
    if 'method' not in plot_df.columns:
        plot_df['method'] = ''
    if 'component_id' not in plot_df.columns:
        plot_df['component_id'] = plot_df.get('group_id', '')
    if size_by == t('chemical_diversity.text_5ab8758886'):
        plot_df['marker_size'] = 13.0
    else:
        size_col = {t('chemical_diversity.text_caa3d4dd04'): 'group_size', t('chemical_diversity.text_71d09e557e'): 'degree', t('chemical_diversity.text_400481e52f'): 'close_analog_count'}.get(size_by, 'group_size')
        values = pd.to_numeric(plot_df.get(size_col, 1), errors='coerce').fillna(1.0).clip(lower=0.0)
        plot_df['marker_size'] = 9.0 + np.sqrt(values + 1.0) * 5.0
    color_col = {t('chemical_diversity.text_5c2c62c55f'): 'group_id', t('chemical_diversity.text_caa3d4dd04'): 'group_size', t('chemical_diversity.text_c0577d0edf'): 'is_singleton_selected', t('chemical_diversity.text_71d09e557e'): 'degree', t('chemical_diversity.text_59dc3f712d'): 'nearest_neighbor_tanimoto', t('chemical_diversity.text_9177d9bf27'): 'csa_class'}.get(color_by, 'group_id')
    fig = go.Figure()
    if isinstance(edges_df, pd.DataFrame) and (not edges_df.empty):
        coords = plot_df.set_index('node_index', drop=False)
        edge_x = []
        edge_y = []
        for (_, edge) in edges_df.iterrows():
            try:
                src = coords.loc[int(edge['source'])]
                dst = coords.loc[int(edge['target'])]
            except Exception:
                continue
            edge_x.extend([src['csa_x'], dst['csa_x'], None])
            edge_y.extend([src['csa_y'], dst['csa_y'], None])
        if edge_x:
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.8, color='rgba(90, 100, 115, 0.30)'), hoverinfo='skip', showlegend=False, name=t('chemical_diversity.text_15ac7536fd')))
    label_mask = plot_df['is_singleton_selected'].astype(bool) | plot_df['is_small_isolated_group'].astype(bool) | plot_df['is_noise'].astype(bool)
    labels = np.where(show_all_labels, plot_df['name'], np.where(show_singleton_labels & label_mask, plot_df['name'], ''))
    if color_col in {'csa_class', 'is_singleton_selected'}:
        for (value, group) in plot_df.groupby(color_col, dropna=False):
            group_labels = pd.Series(labels, index=plot_df.index).loc[group.index]
            color = '#dc2626' if bool(value) and color_col == 'is_singleton_selected' else None
            fig.add_trace(go.Scatter(x=group['csa_x'], y=group['csa_y'], mode='markers+text' if show_singleton_labels or show_all_labels else 'markers', text=group_labels, textposition='top center', marker=dict(size=group['marker_size'], color=color, opacity=0.88, line=dict(width=0.8, color=np.where(group['is_singleton_selected'].astype(bool), '#dc2626', 'white'))), customdata=np.stack([group['name'].astype(str), group['SMILES'].astype(str), group['method'].astype(str), group['group_id'].astype(str), group['group_size'].astype(str), group['degree'].astype(str), group['nearest_neighbor'].astype(str), group['nearest_neighbor_tanimoto'].astype(str), group['is_singleton_selected'].astype(str), group['is_noise'].astype(str), group['csa_class'].astype(str)], axis=-1), hovertemplate=t('chemical_diversity.text_1b37ba0007'), name=str(value)))
    else:
        numeric_color = pd.to_numeric(plot_df[color_col], errors='coerce')
        fig.add_trace(go.Scatter(x=plot_df['csa_x'], y=plot_df['csa_y'], mode='markers+text' if show_singleton_labels or show_all_labels else 'markers', text=labels, textposition='top center', marker=dict(size=plot_df['marker_size'], color=numeric_color, colorscale='Turbo', showscale=True, colorbar=dict(title=color_col), opacity=0.88, line=dict(width=0.8, color=np.where(plot_df['is_singleton_selected'].astype(bool), '#dc2626', 'white'))), customdata=np.stack([plot_df['SMILES'].astype(str), plot_df['method'].astype(str), plot_df['group_id'].astype(str), plot_df['group_size'].astype(str), plot_df['degree'].astype(str), plot_df['nearest_neighbor'].astype(str), plot_df['nearest_neighbor_tanimoto'].astype(str), plot_df['is_singleton_selected'].astype(str), plot_df['is_noise'].astype(str), plot_df['csa_class'].astype(str)], axis=-1), hovertemplate=t('chemical_diversity.text_5c23e7ef89'), name=color_col))
    fig.update_layout(title=t('chemical_diversity.text_3999391893'), xaxis_title='Dimension 1', yaxis_title='Dimension 2', height=620, template='plotly_white', legend_title_text=color_by, margin=dict(l=20, r=20, t=70, b=35))
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig

@st.cache_data(show_spinner=False)
def _cached_structural_communities(map_df, similarity_matrix, method, threshold, top_k, min_cluster_size, butina_cutoff, dbscan_eps, dbscan_min_samples, singleton_criterion):
    return analyze_structural_communities(map_df=map_df, similarity_matrix=np.asarray(similarity_matrix, dtype=float), method=method, threshold=float(threshold), top_k=int(top_k), min_cluster_size=int(min_cluster_size), butina_cutoff=float(butina_cutoff), dbscan_eps=float(dbscan_eps), dbscan_min_samples=int(dbscan_min_samples), singleton_criterion=singleton_criterion)

def _render_structural_communities_block(result):
    final_space = result.get('final_chemical_space', {})
    if not isinstance(final_space, dict):
        return
    map_df = final_space.get('map', pd.DataFrame())
    similarity_matrix = final_space.get('similarity_matrix')
    if not isinstance(map_df, pd.DataFrame) or map_df.empty or similarity_matrix is None:
        return
    st.markdown(t('chemical_diversity.text_d46a6d9635'))
    st.caption(t('chemical_diversity.text_b6386d92b9'))
    st.caption(t('chemical_diversity.text_2440bb6377'))
    st.info(t('chemical_diversity.text_a39b9efee1'))
    method_options = ['Connected components']
    if Butina is not None:
        method_options.append('Butina clustering')
    if DBSCAN is not None:
        method_options.append('DBSCAN')
    method_options.extend(['Similarity network', 'Singletons only'])
    method = st.selectbox(t('chemical_diversity.text_cff5dfb6f8'), method_options, key='chemical_space_communities_method')
    if Butina is None:
        st.caption(t('chemical_diversity.text_120c01bc3d'))
    if DBSCAN is None:
        st.caption(t('chemical_diversity.text_0aae75b8c7'))
    (c1, c2, c3) = st.columns(3)
    with c1:
        threshold = st.slider('Tanimoto threshold', 0.5, 0.95, 0.75, 0.01, key='chemical_space_communities_threshold')
    with c2:
        top_k = st.slider('top-k neighbors', 1, 10, 5, 1, key='chemical_space_communities_top_k')
    with c3:
        small_limit = st.slider(t('chemical_diversity.text_078fab33c2'), 1, 5, 3, 1, key='chemical_space_communities_small_limit')
    (c4, c5, c6) = st.columns(3)
    with c4:
        butina_cutoff = st.slider('Butina distance cutoff', 0.05, 0.5, 0.2, 0.01, key='chemical_space_communities_butina_cutoff')
    with c5:
        dbscan_eps = st.slider('DBSCAN eps', 0.05, 0.7, 0.25, 0.01, key='chemical_space_communities_dbscan_eps')
    with c6:
        dbscan_min_samples = st.slider('DBSCAN min_samples', 1, 10, 2, 1, key='chemical_space_communities_dbscan_min_samples')
    (c7, c8, c9) = st.columns(3)
    with c7:
        singleton_criterion = st.selectbox(t('chemical_diversity.text_1c309142ed'), ['combined', 'component size == 1', 'no neighbors above threshold', 'cluster size <= N', 'DBSCAN noise'], key='chemical_space_communities_singleton_criterion')
    with c8:
        color_by = st.selectbox(t('chemical_diversity.text_d11b85826f'), [t('chemical_diversity.text_5c2c62c55f'), t('chemical_diversity.text_caa3d4dd04'), t('chemical_diversity.text_c0577d0edf'), t('chemical_diversity.text_71d09e557e'), t('chemical_diversity.text_59dc3f712d'), t('chemical_diversity.text_9177d9bf27')], key='chemical_space_communities_color_by')
    with c9:
        size_by = st.selectbox(t('chemical_diversity.text_686b1c751c'), [t('chemical_diversity.text_5ab8758886'), t('chemical_diversity.text_caa3d4dd04'), t('chemical_diversity.text_71d09e557e'), t('chemical_diversity.text_400481e52f')], key='chemical_space_communities_size_by')
    (f1, f2, f3, f4) = st.columns(4)
    only_singletons = f1.checkbox(t('chemical_diversity.text_3a3186e085'), value=method == 'Singletons only', key='chemical_space_communities_only_singletons')
    only_small = f2.checkbox(t('chemical_diversity.text_932ac048f9'), value=False, key='chemical_space_communities_only_small')
    only_large = f3.checkbox(t('chemical_diversity.text_6e0740daf4'), value=False, key='chemical_space_communities_only_large')
    show_singleton_labels = f4.checkbox(t('chemical_diversity.text_f4b8644b2f'), value=True, key='chemical_space_communities_singleton_labels')
    show_all_labels = st.checkbox(t('chemical_diversity.text_b5e91a4d0e'), value=False, disabled=len(map_df) > 80, key='chemical_space_communities_all_labels')
    communities = _cached_structural_communities(map_df, similarity_matrix, method, float(threshold), int(top_k), int(small_limit), float(butina_cutoff), float(dbscan_eps), int(dbscan_min_samples), singleton_criterion)
    nodes = communities.get('nodes', pd.DataFrame())
    edges = communities.get('edges', pd.DataFrame())
    summary = communities.get('summary', {})
    if isinstance(nodes, pd.DataFrame) and (not nodes.empty):
        filtered_nodes = nodes.copy()
        if only_singletons:
            filtered_nodes = filtered_nodes[filtered_nodes['is_singleton_selected'].astype(bool)].copy()
        if only_small:
            filtered_nodes = filtered_nodes[filtered_nodes['group_size'] <= int(small_limit)].copy()
        if only_large:
            filtered_nodes = filtered_nodes[filtered_nodes['group_size'] > int(small_limit)].copy()
        visible = set(filtered_nodes['node_index'].astype(int).tolist())
        if isinstance(edges, pd.DataFrame) and (not edges.empty):
            filtered_edges = edges[edges['source'].astype(int).isin(visible) & edges['target'].astype(int).isin(visible)].copy()
        else:
            filtered_edges = edges
    else:
        filtered_nodes = nodes
        filtered_edges = edges
    metrics = st.columns(4)
    metrics[0].metric(t('chemical_diversity.text_7c16ae8c7a'), summary.get('n_groups', '—'))
    metrics[1].metric('Singleton', summary.get('n_singletons', '—'))
    metrics[2].metric(t('chemical_diversity.text_51f205c8a3'), summary.get('n_small_groups', '—'))
    metrics[3].metric(t('chemical_diversity.text_517b5bb8b0'), summary.get('largest_group_size', '—'))
    metrics = st.columns(4)
    metrics[0].metric(t('chemical_diversity.text_3145630c55'), f"{float(summary.get('largest_group_fraction', 0.0)) * 100:.1f}%")
    metrics[1].metric('Noise', summary.get('noise_points', '—'))
    metrics[2].metric(t('chemical_diversity.text_ed299969be'), _safe_float_text(summary.get('mean_degree')))
    metrics[3].metric(t('chemical_diversity.text_5a246e5093'), summary.get('no_close_neighbors', '—'))
    st.info(str(summary.get('interpretation', '')))
    fig = _make_structural_communities_figure(filtered_nodes, filtered_edges, color_by=color_by, size_by=size_by, show_singleton_labels=show_singleton_labels, show_all_labels=show_all_labels)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t('chemical_diversity.text_b6beebf9a2'))
    with st.expander(t('chemical_diversity.text_fc11d33558'), expanded=False):
        groups = communities.get('groups', pd.DataFrame())
        if isinstance(groups, pd.DataFrame) and (not groups.empty):
            st.dataframe(groups, width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_0aebaa87dc'))
    with st.expander(t('chemical_diversity.text_62175fc385'), expanded=False):
        singletons = communities.get('singletons', pd.DataFrame())
        if isinstance(singletons, pd.DataFrame) and (not singletons.empty):
            st.dataframe(singletons, width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_02a581a409'))
    with st.expander(t('chemical_diversity.text_f22da00101'), expanded=False):
        small_groups = communities.get('small_groups', pd.DataFrame())
        if isinstance(small_groups, pd.DataFrame) and (not small_groups.empty):
            st.dataframe(small_groups, width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_50bbc50525'))

def _render_exact_pattern_map_block(result):
    final_space = result.get('final_chemical_space', {})
    if not isinstance(final_space, dict):
        return
    exact_payload = final_space.get('exact_patterns', {})
    if not isinstance(exact_payload, dict):
        return
    groups = exact_payload.get('groups', pd.DataFrame())
    if not isinstance(groups, pd.DataFrame) or groups.empty:
        st.info(t('chemical_diversity.text_e2dc339c23'))
        return
    st.markdown(t('chemical_diversity.text_a64d545cfb'))
    st.caption(t('chemical_diversity.text_3e28b59c95'))
    st.caption(t('chemical_diversity.text_33ed55496a'))
    st.info(t('chemical_diversity.text_86c94d310f'))
    (c1, c2, c3, c4) = st.columns(4)
    with c1:
        chart_type = st.selectbox(t('chemical_diversity.text_a6aa4d8f5e'), ['Tree map', t('chemical_diversity.text_aa192846a5'), 'Bubble chart'], key='chemical_space_exact_pattern_chart_type')
    with c2:
        color_by = st.selectbox(t('chemical_diversity.text_c5b6ba6451'), [t('chemical_diversity.text_a0aa67d654'), t('chemical_diversity.text_666e9c37f7'), t('chemical_diversity.text_bd4dd06981'), t('chemical_diversity.text_3b4d2673e1')], key='chemical_space_exact_pattern_color_by')
    with c3:
        small_threshold = st.slider('small group threshold', 1, 5, 2, 1, key='chemical_space_exact_pattern_small_threshold')
    with c4:
        rare_threshold = st.slider('rare group threshold', 1, 8, 3, 1, key='chemical_space_exact_pattern_rare_threshold')
    plot_df = groups.copy()
    plot_df['small_group'] = plot_df['group_size'] <= int(small_threshold)
    plot_df['rare_group'] = plot_df['group_size'] <= int(rare_threshold)
    only_rare = st.checkbox(t('chemical_diversity.text_b1c99699fe'), value=False, key='chemical_space_exact_pattern_only_rare')
    if only_rare:
        plot_df = plot_df[plot_df['rare_group'].astype(bool) | plot_df['singleton_group'].astype(bool) | plot_df['unclassified_group'].astype(bool)].copy()
    if plot_df.empty:
        st.info(t('chemical_diversity.text_a759040fc2'))
        return
    color_column = {t('chemical_diversity.text_a0aa67d654'): 'broad_series', t('chemical_diversity.text_666e9c37f7'): 'group_size', t('chemical_diversity.text_bd4dd06981'): 'rare_group', t('chemical_diversity.text_3b4d2673e1'): 'mean_property'}.get(color_by, 'broad_series')
    if color_column == 'mean_property' and (not pd.to_numeric(plot_df['mean_property'], errors='coerce').notna().any()):
        color_column = 'broad_series'
        st.caption(t('chemical_diversity.text_7ab7d9c0cd'))
    hover_data = ['broad_series', 'exact_pattern', 'group_size', 'dataset_fraction', 'rare_group', 'singleton_group', 'unclassified_group', 'representative_examples']
    if chart_type == 'Tree map':
        fig = px.treemap(plot_df, path=['broad_series', 'exact_pattern'], values='group_size', color=color_column, hover_data=hover_data, title=t('chemical_diversity.text_108c9eccf1'))
    elif chart_type == t('chemical_diversity.text_aa192846a5'):
        fig = px.sunburst(plot_df, path=['broad_series', 'exact_pattern'], values='group_size', color=color_column, hover_data=hover_data, title=t('chemical_diversity.text_108c9eccf1'))
    else:
        bubble_df = plot_df.sort_values('group_size', ascending=False).reset_index(drop=True)
        bubble_df['x'] = np.arange(len(bubble_df))
        series_codes = {name: idx for (idx, name) in enumerate(sorted(bubble_df['broad_series'].astype(str).unique()))}
        bubble_df['y'] = bubble_df['broad_series'].astype(str).map(series_codes)
        labels = np.where(bubble_df['rare_group'].astype(bool) | bubble_df['singleton_group'].astype(bool), bubble_df['exact_pattern'], '')
        fig = px.scatter(bubble_df, x='x', y='y', size='group_size', color=color_column, text=labels, hover_data=hover_data, title=t('chemical_diversity.text_108c9eccf1'))
        fig.update_yaxes(tickmode='array', tickvals=list(series_codes.values()), ticktext=list(series_codes.keys()), title='broad_series')
        fig.update_xaxes(title='exact_pattern groups')
        fig.update_traces(textposition='top center')
    fig.update_layout(height=620, margin=dict(l=20, r=20, t=70, b=35))
    st.info(str(exact_payload.get('interpretation', '')))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander(t('chemical_diversity.text_643698d77f'), expanded=False):
        st.dataframe(groups, width='stretch', hide_index=True)
    with st.expander(t('chemical_diversity.text_c151368324'), expanded=False):
        rare_groups = exact_payload.get('rare_groups', pd.DataFrame())
        if isinstance(rare_groups, pd.DataFrame) and (not rare_groups.empty):
            st.dataframe(rare_groups, width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_fa7d35293f'))
    with st.expander(t('chemical_diversity.text_700b2e4759'), expanded=False):
        members = exact_payload.get('members', pd.DataFrame())
        if isinstance(members, pd.DataFrame) and (not members.empty):
            selected = st.selectbox('exact_pattern', sorted(members['exact_pattern'].astype(str).unique()), key='chemical_space_exact_pattern_members_select')
            st.dataframe(members[members['exact_pattern'].astype(str) == selected], width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_70ce63a1a3'))

def _render_similarity_histogram(hist_df):
    st.caption(t('chemical_diversity.text_93287c8a30'))
    if not isinstance(hist_df, pd.DataFrame) or hist_df.empty:
        st.info(t('chemical_diversity.text_6e4e7e7796'))
        return
    chart_df = hist_df.copy()
    chart_df[t('chemical_diversity.text_e65071ce2c')] = chart_df.apply(lambda row: f"{row['similarity_from']:.2f}-{row['similarity_to']:.2f}", axis=1)
    st.bar_chart(chart_df.set_index(t('chemical_diversity.text_e65071ce2c'))['count'])

def _render_pca_map(pca_df):
    st.caption(t('chemical_diversity.text_2bbfda4cab'))
    if not isinstance(pca_df, pd.DataFrame) or pca_df.empty:
        st.info(t('chemical_diversity.text_d131ad5424'))
        return
    color_mode = 'cluster_id'
    if 'target' in pca_df.columns and pd.to_numeric(pca_df['target'], errors='coerce').notna().any():
        color_mode = st.radio(t('chemical_diversity.text_d11b85826f'), ['cluster_id', 'target'], horizontal=True, key='chemical_diversity_pca_color_mode')
    else:
        st.caption(t('chemical_diversity.text_06b088d288'))
    plot_df = pca_df.copy()
    plot_df[color_mode] = pd.to_numeric(plot_df[color_mode], errors='coerce')
    (fig, ax) = plt.subplots(figsize=(5.2, 3.4))
    scatter = ax.scatter(plot_df['PC1'], plot_df['PC2'], c=plot_df[color_mode], cmap='viridis', s=24, alpha=0.85, edgecolors='none')
    ax.set_xlabel('PCA 1', fontsize=9)
    ax.set_ylabel('PCA 2', fontsize=9)
    ax.set_title(t('chemical_diversity.text_ff058db7da'), fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, label=color_mode, shrink=0.82)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    _show_compact_figure(fig, width=720)
    with st.expander(t('chemical_diversity.text_21f131008c'), expanded=False):
        st.dataframe(plot_df, width='stretch', hide_index=True)

def _render_similarity_heatmap(heatmap_payload):
    st.caption(t('chemical_diversity.text_993a9d3427'))
    if not isinstance(heatmap_payload, dict):
        st.info(t('chemical_diversity.text_9eccc3fd57'))
        return
    matrix_df = heatmap_payload.get('matrix', pd.DataFrame())
    molecules_df = heatmap_payload.get('molecules', pd.DataFrame())
    if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        st.info(t('chemical_diversity.text_9eccc3fd57'))
        return
    if heatmap_payload.get('sampled'):
        st.warning(t('chemical_diversity.text_1c13d34b85'))
    (fig, ax) = plt.subplots(figsize=(4.8, 3.8))
    image = ax.imshow(matrix_df.values, cmap='viridis', vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_title(t('chemical_diversity.text_0667369e27'), fontsize=10)
    ax.set_xlabel(t('chemical_diversity.text_f62f40d1bd'), fontsize=9)
    ax.set_ylabel(t('chemical_diversity.text_f62f40d1bd'), fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, label='Tanimoto', shrink=0.82)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    _show_compact_figure(fig, width=620)
    with st.expander(t('chemical_diversity.text_07e0e29473'), expanded=False):
        st.dataframe(molecules_df, width='stretch', hide_index=True)

def _render_clusters(cluster_summary, summary):
    st.caption(t('chemical_diversity.text_d923e23835'))
    if not isinstance(cluster_summary, pd.DataFrame) or cluster_summary.empty:
        st.info(t('chemical_diversity.text_932c1891f7'))
        return
    (col_a, col_b) = st.columns(2)
    col_a.metric(t('chemical_diversity.text_0dec38a6a7'), summary.get('singleton_clusters', '—'))
    col_b.metric(t('chemical_diversity.text_9cbcd7a3eb'), f"{_safe_float_text(summary.get('largest_cluster_percent'), digits=1)}%")
    chart_df = cluster_summary.sort_values('n', ascending=False).copy()
    chart_df['cluster_id'] = chart_df['cluster_id'].astype(str)
    st.bar_chart(chart_df.set_index('cluster_id')['n'])
    st.dataframe(chart_df, width='stretch', hide_index=True)

def _render_analogue_network(pca_df, network_edges):
    st.caption(t('chemical_diversity.text_c05892029d'))
    if not isinstance(network_edges, pd.DataFrame) or network_edges.empty:
        st.info(t('chemical_diversity.text_93777b4e20'))
        return
    if len(network_edges) >= 500:
        st.warning(t('chemical_diversity.text_75e62f50ed'))
    if not isinstance(pca_df, pd.DataFrame) or pca_df.empty:
        st.dataframe(network_edges, width='stretch', hide_index=True)
        return
    node_df = pca_df.set_index('row')
    edge_df = network_edges[network_edges['source_row'].isin(node_df.index) & network_edges['target_row'].isin(node_df.index)].copy()
    if edge_df.empty:
        st.info(t('chemical_diversity.text_d7b3f7703a'))
        return
    (fig, ax) = plt.subplots(figsize=(5.2, 3.4))
    for (_, edge) in edge_df.iterrows():
        src = node_df.loc[int(edge['source_row'])]
        dst = node_df.loc[int(edge['target_row'])]
        ax.plot([src['PC1'], dst['PC1']], [src['PC2'], dst['PC2']], color='#9ca3af', alpha=min(0.75, max(0.15, float(edge['tanimoto']) - 0.55)), linewidth=0.6)
    scatter = ax.scatter(node_df['PC1'], node_df['PC2'], c=pd.to_numeric(node_df['cluster_id'], errors='coerce'), cmap='viridis', s=24, alpha=0.9, edgecolors='none')
    ax.set_title(t('chemical_diversity.text_04e8ff9fb0'), fontsize=10)
    ax.set_xlabel('PCA 1', fontsize=9)
    ax.set_ylabel('PCA 2', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, label='cluster_id', shrink=0.82)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    _show_compact_figure(fig, width=720)
    with st.expander(t('chemical_diversity.text_46b5ef0e11'), expanded=False):
        st.dataframe(edge_df, width='stretch', hide_index=True)

def _render_pairs_and_unique(result):
    st.caption(t('chemical_diversity.text_cdfd580648'))
    duplicate_pairs = result.get('duplicate_pairs', pd.DataFrame())
    analogue_pairs = result.get('analogue_pairs', pd.DataFrame())
    unique_table = result.get('unique_molecules', pd.DataFrame())
    st.markdown(t('chemical_diversity.text_4ac8c02ce6'))
    if isinstance(duplicate_pairs, pd.DataFrame) and (not duplicate_pairs.empty):
        st.dataframe(duplicate_pairs, width='stretch', hide_index=True)
    else:
        st.info(t('chemical_diversity.text_c25f33c0c2'))
    st.markdown(t('chemical_diversity.text_56d599ac1b'))
    if isinstance(analogue_pairs, pd.DataFrame) and (not analogue_pairs.empty):
        st.dataframe(analogue_pairs, width='stretch', hide_index=True)
    else:
        st.info(t('chemical_diversity.text_537656b68b'))
    st.markdown(t('chemical_diversity.text_e4fc47ef5c'))
    if isinstance(unique_table, pd.DataFrame) and (not unique_table.empty):
        st.dataframe(unique_table, width='stretch', hide_index=True)
    else:
        st.info(t('chemical_diversity.text_9eefa4a7ca'))

def _localized_diversity_status(status):
    status_text = str(status or "").strip()
    status_map = {
        t('chemical_diversity.text_fc4dc2a373'): t('chemical_diversity.text_fc4dc2a373'),
        t('chemical_diversity.text_c8571b5d1b'): t('chemical_diversity.text_c8571b5d1b'),
        t('chemical_diversity.text_b9282269a3'): t('chemical_diversity.text_b9282269a3'),
        t('chemical_diversity.text_0d428eae71'): t('chemical_diversity.text_0d428eae71'),
        'низкое разнообразие': t('chemical_diversity.text_fc4dc2a373'),
        'неоднородный датасет': t('chemical_diversity.text_c8571b5d1b'),
        'высокое разнообразие': t('chemical_diversity.text_b9282269a3'),
        'умеренное разнообразие': t('chemical_diversity.text_0d428eae71'),
        'не рассчитано': t('chemical_diversity.text_de32203cf2'),
    }
    return status_map.get(status_text, status_text or t('chemical_diversity.text_de32203cf2'))


def _localized_diversity_reasons(reasons):
    reason_text = str(reasons or "").strip()
    replacements = {
        'среднее Tanimoto-сходство высокое': t('chemical_diversity.reason_mean_tanimoto_high'),
        'среднее Tanimoto-сходство низкое и кластеров много': t('chemical_diversity.reason_mean_tanimoto_low_many_clusters'),
        'структуры образуют несколько областей без экстремального сходства': t('chemical_diversity.reason_moderate_regions'),
        'есть крупный кластер и заметная доля одиночных кластеров': t('chemical_diversity.reason_large_cluster_singletons'),
        'есть почти дублирующиеся или очень близкие пары': t('chemical_diversity.reason_near_duplicates'),
        'явных причин не выделено': t('chemical_diversity.reason_no_clear_causes'),
    }
    for source, localized in replacements.items():
        reason_text = reason_text.replace(source, localized)
    return reason_text


def render_chemical_diversity_section(data, smiles_col, label_col=None, target_col=None, descriptor_df=None, expanded=False):
    """Render pre-modeling chemical diversity diagnostics."""
    if not isinstance(data, pd.DataFrame) or data.empty or (not smiles_col) or (smiles_col not in data.columns):
        return
    if not label_col:
        for candidate in ('Name', 'name', 'compound_id', 'Compound ID', 'CAS', 'cas'):
            if candidate in data.columns and candidate != smiles_col:
                label_col = candidate
                break
    st.markdown(t('chemical_diversity.text_c45333a0db'))
    st.caption(t('chemical_diversity.text_e9cf048210'))
    st.caption(t('chemical_diversity.text_244d21577b'))
    (col_a, col_b, col_c) = st.columns(3)
    with col_a:
        cluster_threshold = st.slider(t('chemical_diversity.text_a3cd2ce9d4'), min_value=0.3, max_value=0.9, value=0.6, step=0.05, key='chemical_diversity_cluster_threshold')
    with col_b:
        duplicate_threshold = st.slider(t('chemical_diversity.text_2eb512500a'), min_value=0.85, max_value=1.0, value=0.95, step=0.01, key='chemical_diversity_duplicate_threshold')
    with col_c:
        analogue_threshold = st.slider(t('chemical_diversity.text_11a8e7a92f'), min_value=0.6, max_value=0.99, value=0.85, step=0.01, key='chemical_diversity_analogue_threshold')
    (col_d, col_e, col_f, col_g) = st.columns(4)
    with col_d:
        projection_method = st.selectbox(t('chemical_diversity.text_2a09bff718'), ['auto', 'UMAP', 'MDS', 't-SNE'], index=0, key='chemical_diversity_projection_method')
    with col_e:
        radius = st.number_input(t('chemical_diversity.morgan_radius_label'), min_value=1, max_value=4, value=2, step=1, key='chemical_diversity_morgan_radius')
    with col_f:
        n_bits = st.selectbox(t('chemical_diversity.morgan_nbits_label'), [1024, 2048, 4096], index=1, key='chemical_diversity_morgan_n_bits')
    with col_g:
        map_top_k = st.number_input(t('chemical_diversity.text_3a641834de'), min_value=1, max_value=20, value=5, step=1, key='chemical_diversity_map_top_k')
    st.caption(t('chemical_diversity.text_8db9159eba'))
    signature = f'{_result_signature(data, smiles_col, descriptor_df)}:cluster={cluster_threshold:.2f}:dup={duplicate_threshold:.2f}:analog={analogue_threshold:.2f}:projection={projection_method}:radius={int(radius)}:bits={int(n_bits)}:topk={int(map_top_k)}'
    cached = st.session_state.get('chemical_diversity_result')
    cached_signature = st.session_state.get('chemical_diversity_signature')
    run_clicked = st.button(t('chemical_diversity.text_f1c4ac0147'), type='primary', key='run_chemical_diversity')
    if run_clicked or (cached is not None and cached_signature == signature):
        if run_clicked:
            with st.spinner(t('chemical_diversity.text_df201cd4e3')):
                result = analyze_chemical_diversity(data=data, smiles_col=smiles_col, label_col=label_col, target_col=target_col, descriptor_df=descriptor_df, radius=int(radius), n_bits=int(n_bits), duplicate_threshold=duplicate_threshold, analogue_threshold=analogue_threshold, cluster_similarity_threshold=cluster_threshold, projection_method=projection_method, map_edge_threshold=analogue_threshold, map_edge_top_k=int(map_top_k))
                st.session_state.chemical_diversity_result = result
                st.session_state.chemical_diversity_signature = signature
        else:
            result = cached
        summary = result.get('summary', {})
        raw_status = str(summary.get('status', t('chemical_diversity.text_de32203cf2')))
        status = _localized_diversity_status(raw_status)
        reasons = _localized_diversity_reasons(summary.get('status_reasons', ''))
        if raw_status in {'низкое разнообразие', 'неоднородный датасет'}:
            st.warning(t('chemical_diversity.status_message', status=status, reasons=reasons))
        elif raw_status in {'высокое разнообразие', 'умеренное разнообразие'}:
            st.success(t('chemical_diversity.status_message', status=status, reasons=reasons))
        else:
            st.info(t('chemical_diversity.status_message', status=status, reasons=reasons))
        if summary.get('pairwise_mode') == 'sampled':
            st.info(t('chemical_diversity.text_d403523606'))
        if summary.get('cluster_sampled'):
            st.info(t('chemical_diversity.text_a38981340f'))
        metric_cols = st.columns(4)
        metric_cols[0].metric(t('chemical_diversity.text_a742738aa1'), _safe_float_text(summary.get('mean_tanimoto')))
        metric_cols[1].metric(t('chemical_diversity.text_a7f99b1892'), summary.get('pairs_gt_0_95', '—'))
        metric_cols[2].metric(t('chemical_diversity.text_90cd825cac'), summary.get('n_clusters', '—'))
        metric_cols[3].metric(t('chemical_diversity.text_1b605ecac0'), summary.get('unique_molecules_lt_0_30', '—'))
        st.dataframe(_summary_table(summary), width='stretch', hide_index=True)
        (tab_distribution, tab_space, tab_heatmap, tab_clusters, tab_network, tab_pairs) = st.tabs([t('chemical_diversity.text_b22b11bf82'), t('chemical_diversity.text_ff058db7da'), t('chemical_diversity.text_addc12944d'), t('chemical_diversity.text_90cd825cac'), t('chemical_diversity.text_986a21123c'), t('chemical_diversity.text_c2a0c6990c')])
        with tab_distribution:
            _render_similarity_histogram(result.get('similarity_histogram', pd.DataFrame()))
            descriptor_space = result.get('descriptor_space', {})
            descriptor_table = _descriptor_summary_table(descriptor_space)
            if not descriptor_table.empty:
                with st.expander(t('chemical_diversity.descriptor_space_expander'), expanded=False):
                    st.dataframe(descriptor_table, width='stretch', hide_index=True)
                    coords = descriptor_space.get('pca_coordinates') if isinstance(descriptor_space, dict) else None
                    if isinstance(coords, pd.DataFrame) and {'PC1', 'PC2'}.issubset(coords.columns):
                        st.scatter_chart(coords[['PC1', 'PC2']])
        with tab_space:
            _render_pca_map(result.get('fingerprint_pca', pd.DataFrame()))
        with tab_heatmap:
            _render_similarity_heatmap(result.get('similarity_heatmap', {}))
        with tab_clusters:
            _render_clusters(result.get('cluster_summary', pd.DataFrame()), summary)
        with tab_network:
            _render_analogue_network(result.get('fingerprint_pca', pd.DataFrame()), result.get('network_edges', pd.DataFrame()))
        with tab_pairs:
            _render_pairs_and_unique(result)
            invalid_df = result.get('invalid_structures', pd.DataFrame())
            if isinstance(invalid_df, pd.DataFrame) and (not invalid_df.empty):
                with st.expander(t('chemical_diversity.text_1ea1ac107e'), expanded=False):
                    st.dataframe(invalid_df, width='stretch', hide_index=True)
        _render_final_chemical_space(result)
        _render_structural_communities_block(result)
        _render_exact_pattern_map_block(result)
