"""Streamlit UI for chemical diversity diagnostics."""
from __future__ import annotations
import hashlib
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    px = None
    PLOTLY_AVAILABLE = False
from modules.i18n import t
from modules.chemical_diversity_core import (
    Butina,
    DBSCAN,
    CHEMICAL_SPACE_ALGORITHM_VERSION,
    CSA_DENSE_LABEL,
    CSA_ISOLATED_LABEL,
    CSA_MODERATE_LABEL,
    CSA_SPARSE_LABEL,
    analyze_chemical_diversity,
    analyze_structural_communities,
    coerce_boolean_series,
)
from modules.analysis_state import current_analysis_parameters_table, update_analysis_bundle

def _safe_float_text(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return '—'
    if not np.isfinite(value):
        return '—'
    return f'{value:.{digits}f}'

def _series_digest(data, col):
    if not col or not isinstance(data, pd.DataFrame) or col not in data.columns:
        return 'none'
    digest = hashlib.sha1()
    values = data[col].astype(str).fillna('')
    digest.update(f'{col}:{len(values)}\n'.encode('utf-8', errors='replace'))
    for value in values:
        digest.update(str(value).encode('utf-8', errors='replace'))
        digest.update(b'\0')
    return digest.hexdigest()[:16]


def _result_signature(data, smiles_col, descriptor_df, label_col=None, target_col=None):
    digest = hashlib.sha1()
    try:
        smiles = data[smiles_col].astype(str).fillna('')
        digest.update(f'smiles_col={smiles_col}\nrows={len(smiles)}\n'.encode('utf-8'))
        for value in smiles:
            digest.update(str(value).encode('utf-8', errors='replace'))
            digest.update(b'\0')
    except Exception:
        digest.update(b'smiles_error')
    smiles_digest = digest.hexdigest()[:16]

    desc_shape = tuple(descriptor_df.shape) if isinstance(descriptor_df, pd.DataFrame) else None
    desc_digest = 'none'
    if isinstance(descriptor_df, pd.DataFrame):
        try:
            desc_hash = pd.util.hash_pandas_object(
                descriptor_df.reset_index(drop=True),
                index=True
            )
            desc_value_digest = hashlib.sha1(desc_hash.values.tobytes()).hexdigest()[:16]
            desc_col_digest = hashlib.sha1(
                '\0'.join(map(str, descriptor_df.columns)).encode('utf-8', errors='replace')
            ).hexdigest()[:16]
            desc_digest = f'{desc_value_digest}:{desc_col_digest}'
        except Exception:
            desc_digest = 'hash_error'
    label_digest = _series_digest(data, label_col)
    target_digest = _series_digest(data, target_col)
    return f'{smiles_col}:{len(data)}:{smiles_digest}:label={label_col}:{label_digest}:target={target_col}:{target_digest}:{desc_shape}:{desc_digest}'


def _render_current_analysis_parameters(rows, key_suffix):
    params_df = pd.DataFrame(rows or [])
    if params_df.empty:
        return
    params_df = params_df.rename(
        columns={
            "parameter": t("chemical_diversity.analysis_parameter"),
            "value": t("chemical_diversity.analysis_value"),
            "module": t("chemical_diversity.analysis_module"),
        }
    )
    with st.expander(t("chemical_diversity.analysis_parameters_title"), expanded=False):
        st.dataframe(params_df, width="stretch", hide_index=True)
        st.download_button(
            t("chemical_diversity.analysis_parameters_download"),
            params_df.to_csv(index=False).encode("utf-8-sig"),
            f"analysis_parameters_{key_suffix}.csv",
            "text/csv",
            key=f"analysis_parameters_download_{key_suffix}",
        )

def _summary_table(summary):
    rows = [(t('chemical_diversity.text_10dd2c1e6e'), summary.get('total_rows')), (t('chemical_diversity.text_9afccebb18'), summary.get('valid_structures')), (t('chemical_diversity.text_ae171e42c2'), summary.get('invalid_structures')), (t('chemical_diversity.text_0b9edeed53'), summary.get('total_pairs')), (t('chemical_diversity.text_ee43026d14'), summary.get('pairs_used')), (t('chemical_diversity.text_afcb806109'), _safe_float_text(summary.get('mean_tanimoto'))), (t('chemical_diversity.text_227e99c26c'), _safe_float_text(summary.get('median_tanimoto'))), (t('chemical_diversity.text_e2f0a5ae0b'), _safe_float_text(summary.get('min_tanimoto'))), (t('chemical_diversity.text_6893a74204'), _safe_float_text(summary.get('max_tanimoto'))), (t('chemical_diversity.text_4c27e37460'), summary.get('pairs_gt_0_95')), (t('chemical_diversity.text_966232685c'), summary.get('pairs_gt_0_85')), (t('chemical_diversity.text_f95557eeb4'), summary.get('unique_molecules_lt_0_30')), (t('chemical_diversity.text_e72e9c2b87'), summary.get('n_clusters')), (t('chemical_diversity.text_736fff1164'), summary.get('largest_cluster_size')), (t('chemical_diversity.text_ef342379ee'), _safe_float_text(summary.get('largest_cluster_percent'), digits=1)), (t('chemical_diversity.text_0dec38a6a7'), summary.get('singleton_clusters')), ('Плотная область', summary.get('csa_dense_area')), ('Умеренная область', summary.get('csa_moderate_area')), ('Разреженная область', summary.get('csa_sparse_area')), ('Изолированная структура', summary.get('csa_singleton_outlier')), ('Точные дубли', summary.get('csa_exact_duplicates')), ('Близкие дубли', summary.get('csa_near_duplicates')), ('Связные компоненты', summary.get('csa_connected_components')), (t('chemical_diversity.text_c109a2a5d3'), summary.get('csa_largest_component_size')), ('Источник структуры для fingerprints', summary.get('fingerprint_structure_source')), ('Конфигурация fingerprint', f"{summary.get('fingerprint_type')} radius={summary.get('fingerprint_radius')} bits={summary.get('fingerprint_bits')}"), ('Примечание к Tanimoto', summary.get('tanimoto_threshold_note'))]
    rows.extend([
        ('Chemical space random seed', summary.get('random_seed')),
        ('Fraction of all pairs used', summary.get('pair_sample_fraction')),
        ('Mean Tanimoto 95% bootstrap CI', (
            f"{_safe_float_text(summary.get('mean_tanimoto_bootstrap_ci95_low'))} - "
            f"{_safe_float_text(summary.get('mean_tanimoto_bootstrap_ci95_high'))}"
        )),
    ])
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

def _show_compact_figure(fig, width=720, close_after_render=True):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    st.image(buffer, width=int(width))
    if close_after_render:
        plt.close(fig)


CSA_CLASS_COLORS = {
    CSA_DENSE_LABEL: '#2563eb',
    CSA_MODERATE_LABEL: '#16a34a',
    CSA_SPARSE_LABEL: '#f59e0b',
    CSA_ISOLATED_LABEL: '#dc2626',
}

CSA_CLASS_PALETTES = {
    'Balanced': {
        CSA_DENSE_LABEL: '#2563eb',
        CSA_MODERATE_LABEL: '#16a34a',
        CSA_SPARSE_LABEL: '#f59e0b',
        CSA_ISOLATED_LABEL: '#dc2626',
    },
    'Colorblind safe': {
        CSA_DENSE_LABEL: '#0072B2',
        CSA_MODERATE_LABEL: '#009E73',
        CSA_SPARSE_LABEL: '#E69F00',
        CSA_ISOLATED_LABEL: '#D55E00',
    },
    'Scientific': {
        CSA_DENSE_LABEL: '#3B4CC0',
        CSA_MODERATE_LABEL: '#00A087',
        CSA_SPARSE_LABEL: '#F39B7F',
        CSA_ISOLATED_LABEL: '#E64B35',
    },
    'High contrast': {
        CSA_DENSE_LABEL: '#1D4ED8',
        CSA_MODERATE_LABEL: '#059669',
        CSA_SPARSE_LABEL: '#D97706',
        CSA_ISOLATED_LABEL: '#BE123C',
    },
    'Muted print': {
        CSA_DENSE_LABEL: '#4C78A8',
        CSA_MODERATE_LABEL: '#54A24B',
        CSA_SPARSE_LABEL: '#F58518',
        CSA_ISOLATED_LABEL: '#E45756',
    },
}

CONTINUOUS_COLOR_SCALES = [
    'Viridis',
    'Cividis',
    'Plasma',
    'Inferno',
    'Magma',
    'Turbo',
    'IceFire',
    'RdBu',
    'Portland',
    'YlGnBu',
]
MATPLOTLIB_CMAP_BY_PLOTLY = {
    'Viridis': 'viridis',
    'Cividis': 'cividis',
    'Plasma': 'plasma',
    'Inferno': 'inferno',
    'Magma': 'magma',
    'Turbo': 'turbo',
    'IceFire': 'coolwarm',
    'RdBu': 'RdBu',
    'Portland': 'Spectral',
    'YlGnBu': 'YlGnBu',
}

CSA_CLASS_LEGACY_TO_CODE = {
    'Dense area': CSA_DENSE_LABEL,
    'Moderate area': CSA_MODERATE_LABEL,
    'Sparse area': CSA_SPARSE_LABEL,
    'Singleton / outlier': CSA_ISOLATED_LABEL,
    'Изолированная структура': CSA_ISOLATED_LABEL,
}

CLUSTER_CATEGORY_COLORS = [
    '#2563eb',
    '#dc2626',
    '#16a34a',
    '#f59e0b',
    '#7c3aed',
    '#0891b2',
    '#db2777',
    '#65a30d',
    '#ea580c',
    '#475569',
    '#0d9488',
    '#9333ea',
]
CLUSTER_OTHER_COLOR = '#9ca3af'


def _csa_class_code(value):
    text = str(value or '').strip()
    return CSA_CLASS_LEGACY_TO_CODE.get(text, text if text in CSA_CLASS_COLORS else CSA_ISOLATED_LABEL)


def _cluster_category_series(values):
    return pd.Series(values).astype(str).replace({
        'nan': 'unassigned',
        '<NA>': 'unassigned',
        'None': 'unassigned',
        '-1': 'unassigned',
    })


def _cluster_category_colors(cluster_values, top_n=12):
    clusters = _cluster_category_series(cluster_values)
    counts = clusters.value_counts(dropna=False)
    top_clusters = [
        cluster for cluster in counts.index.tolist()
        if cluster != 'unassigned'
    ][:int(top_n)]
    color_map = {
        cluster: CLUSTER_CATEGORY_COLORS[i % len(CLUSTER_CATEGORY_COLORS)]
        for i, cluster in enumerate(top_clusters)
    }
    color_map['unassigned'] = CLUSTER_OTHER_COLOR
    return clusters, color_map, top_clusters


def _cluster_label(cluster):
    if str(cluster) == 'unassigned':
        return 'Other / unassigned'
    return f'Cluster {cluster}'


def _csa_class_display(value):
    code = _csa_class_code(value)
    labels = {
        CSA_DENSE_LABEL: 'Dense area',
        CSA_MODERATE_LABEL: 'Moderate area',
        CSA_SPARSE_LABEL: 'Sparse area',
        CSA_ISOLATED_LABEL: 'Singleton / outlier',
    }
    return labels.get(code, str(value))


def _ensure_columns(df, defaults):
    work = df.copy()
    for column, default in defaults.items():
        if column not in work.columns:
            work[column] = default
    return work


def _safe_numeric_series(df, column, default=0.0):
    if column in df.columns:
        values = df[column]
    else:
        values = pd.Series(default, index=df.index)
    return pd.to_numeric(values, errors='coerce')


def _make_final_chemical_space_figure(map_df, edges_df, color_by='csa_class', size_by='close_analog_count', show_outlier_labels=True, palette_name='Balanced', continuous_scale='Viridis'):
    if not isinstance(map_df, pd.DataFrame) or map_df.empty:
        return None
    plot_df = map_df.copy()
    plot_df = _ensure_columns(
        plot_df,
        {
            'name': '',
            'SMILES': '',
            'nearest_neighbor': '',
            'nearest_neighbor_tanimoto': np.nan,
            'close_analog_count': 0,
            'local_density': 0,
            'connected_component': '',
            'canonical_smiles': '',
            'csa_class': CSA_ISOLATED_LABEL,
            'is_structural_outlier': False,
            'csa_x': np.nan,
            'csa_y': np.nan,
        },
    )
    if 'node_index' not in plot_df.columns:
        plot_df['node_index'] = np.arange(len(plot_df), dtype=int)
    plot_df['csa_class_code'] = plot_df['csa_class'].map(_csa_class_code)
    plot_df['csa_class_label'] = plot_df['csa_class_code'].map(_csa_class_display)
    plot_df['marker_size'] = _safe_numeric_series(plot_df, size_by, 0.0).fillna(0.0)
    plot_df['marker_size'] = 9.0 + np.sqrt(plot_df['marker_size'].clip(lower=0.0) + 1.0) * 5.0
    fig = go.Figure()
    class_colors = CSA_CLASS_PALETTES.get(palette_name, CSA_CLASS_COLORS)
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
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.7, color='rgba(120, 130, 145, 0.32)'), hoverinfo='skip', showlegend=False, name=t('chemical_diversity.text_2649168f1e')))
    color_values = plot_df.get(color_by, plot_df.get('csa_class', ''))
    if color_by == 'csa_class':
        for (csa_class, group) in plot_df.groupby('csa_class_code', dropna=False):
            labels = np.where(show_outlier_labels & coerce_boolean_series(group['is_structural_outlier']), group['name'], '')
            label = _csa_class_display(csa_class)
            fig.add_trace(go.Scatter(x=group['csa_x'], y=group['csa_y'], mode='markers+text' if show_outlier_labels else 'markers', text=labels, textposition='top center', marker=dict(size=group['marker_size'], color=class_colors.get(str(csa_class), '#64748b'), opacity=0.88, line=dict(width=0.6, color='white')), customdata=np.stack([group['name'].astype(str), group['SMILES'].astype(str), group['nearest_neighbor'].astype(str), group['nearest_neighbor_tanimoto'].astype(str), group['close_analog_count'].astype(str), group['local_density'].astype(str), group['connected_component'].astype(str), group['canonical_smiles'].astype(str)], axis=-1), hovertemplate='<b>%{customdata[0]}</b><br>SMILES: %{customdata[1]}<br>CSA-class: ' + label + t('chemical_diversity.text_560094861d'), name=label))
    else:
        numeric_color = pd.to_numeric(color_values, errors='coerce')
        fig.add_trace(go.Scatter(x=plot_df['csa_x'], y=plot_df['csa_y'], mode='markers', marker=dict(size=plot_df['marker_size'], color=numeric_color, colorscale=continuous_scale, showscale=True, colorbar=dict(title=color_by), opacity=0.88, line=dict(width=0.6, color='white')), text=plot_df['name'], customdata=np.stack([plot_df['SMILES'].astype(str), plot_df['csa_class_label'].astype(str), plot_df['nearest_neighbor'].astype(str), plot_df['nearest_neighbor_tanimoto'].astype(str), plot_df['close_analog_count'].astype(str), plot_df['local_density'].astype(str), plot_df['connected_component'].astype(str), plot_df['canonical_smiles'].astype(str)], axis=-1), hovertemplate=t('chemical_diversity.text_db7498efc9'), name=color_by))
    fig.update_layout(title=t('chemical_diversity.text_e46d8938dc'), xaxis_title='Координата проекции 1', yaxis_title='Координата проекции 2', height=650, template='plotly_white', legend_title_text='Класс области', margin=dict(l=20, r=20, t=70, b=35))
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
    st.caption(t('chemical_diversity.premodeling_scope_caption'))
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
    metric_cols[0].metric(t('chemical_diversity.dense_area'), summary.get('csa_dense_area', 0))
    metric_cols[1].metric(t('chemical_diversity.moderate_area'), summary.get('csa_moderate_area', 0))
    metric_cols[2].metric(t('chemical_diversity.sparse_area'), summary.get('csa_sparse_area', 0))
    metric_cols[3].metric(t('chemical_diversity.isolated_structure'), summary.get('csa_singleton_outlier', 0))
    controls = st.columns([1.2, 1.0, 1.0, 1.0])
    with controls[0]:
        color_options = ['csa_class']
        if 'experimental_value' in map_df.columns and pd.to_numeric(map_df['experimental_value'], errors='coerce').notna().any():
            color_options.append('experimental_value')
        color_by = st.selectbox(t('chemical_diversity.text_d11b85826f'), color_options, key='chemical_space_final_color_by')
    with controls[1]:
        size_by = st.selectbox(t('chemical_diversity.text_686b1c751c'), ['close_analog_count', 'local_density'], key='chemical_space_final_size_by')
        st.caption(t('chemical_diversity.point_size_note'))
    with controls[2]:
        show_labels = st.checkbox(t('chemical_diversity.text_498d3ea747'), value=True, key='chemical_space_final_show_outlier_labels')
    with controls[3]:
        palette_name = st.selectbox(t('chemical_diversity.palette_label'), list(CSA_CLASS_PALETTES.keys()), key='chemical_space_final_palette')
        continuous_scale = st.selectbox(t('chemical_diversity.numeric_scale_label'), CONTINUOUS_COLOR_SCALES, key='chemical_space_final_continuous_scale')
    fig = _make_final_chemical_space_figure(map_df=map_df, edges_df=edges_df, color_by=color_by, size_by=size_by, show_outlier_labels=show_labels, palette_name=palette_name, continuous_scale=continuous_scale)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t('chemical_diversity.projection_coordinates_note'))
        compact_html = fig.to_html(include_plotlyjs='cdn', full_html=True).encode('utf-8')
        standalone_html = fig.to_html(include_plotlyjs=True, full_html=True).encode('utf-8')
        csv = map_df.to_csv(index=False).encode('utf-8-sig')
        dl_cols = st.columns(4)
        dl_cols[0].download_button(t('chemical_diversity.compact_html_download'), data=compact_html, file_name='final_chemical_space_map_compact.html', mime='text/html', key='download_final_chemical_space_html_compact')
        dl_cols[1].download_button(t('chemical_diversity.standalone_html_download'), data=standalone_html, file_name='final_chemical_space_map_standalone.html', mime='text/html', key='download_final_chemical_space_html_standalone')
        try:
            png = fig.to_image(format='png', scale=2)
        except Exception:
            png = None
        if png:
            dl_cols[2].download_button(t('chemical_diversity.text_b1eb9744e7'), data=png, file_name='final_chemical_space_map.png', mime='image/png', key='download_final_chemical_space_png')
        else:
            dl_cols[2].caption(t('chemical_diversity.text_1ad04dcdaf'))
        dl_cols[3].download_button(t('chemical_diversity.text_9e8188bf73'), data=csv, file_name='chemical_space_csa_table.csv', mime='text/csv', key='download_final_chemical_space_csv')
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
        outlier_table = _ensure_columns(
            map_df,
            {
                'name': '',
                'SMILES': '',
                'nearest_neighbor': '',
                'nearest_neighbor_tanimoto': np.nan,
                'close_analog_count': 0,
                'local_density': 0,
                'connected_component': '',
                'csa_class': CSA_ISOLATED_LABEL,
                'is_structural_outlier': False,
            },
        )
        outlier_table['csa_class'] = outlier_table['csa_class'].map(_csa_class_display)
        outliers = outlier_table[coerce_boolean_series(outlier_table['is_structural_outlier'])].copy()
        if not outliers.empty:
            st.dataframe(outliers[['name', 'SMILES', 'nearest_neighbor', 'nearest_neighbor_tanimoto', 'close_analog_count', 'local_density', 'connected_component', 'csa_class']], width='stretch', hide_index=True)
        else:
            st.info(t('chemical_diversity.text_a15ddd267f'))

def _make_structural_communities_figure(nodes_df, edges_df, color_by, size_by, show_singleton_labels, show_all_labels, palette_name='Balanced', continuous_scale='Turbo'):
    if not isinstance(nodes_df, pd.DataFrame) or nodes_df.empty:
        return None
    plot_df = nodes_df.copy()
    plot_df = _ensure_columns(
        plot_df,
        {
            'name': '',
            'SMILES': '',
            'method': '',
            'node_index': np.arange(len(plot_df)),
            'group_id': '',
            'group_size': 1,
            'degree': 0,
            'nearest_neighbor': '',
            'nearest_neighbor_tanimoto': np.nan,
            'close_analog_count': 0,
            'is_singleton_selected': False,
            'is_small_isolated_group': False,
            'is_noise': False,
            'csa_class': CSA_ISOLATED_LABEL,
            'csa_x': np.nan,
            'csa_y': np.nan,
        },
    )
    if 'component_id' not in plot_df.columns:
        plot_df['component_id'] = plot_df['group_id']
    plot_df['csa_class_code'] = plot_df['csa_class'].map(_csa_class_code)
    plot_df['csa_class_label'] = plot_df['csa_class_code'].map(_csa_class_display)
    if size_by == t('chemical_diversity.text_5ab8758886'):
        plot_df['marker_size'] = 13.0
    else:
        size_col = {t('chemical_diversity.text_caa3d4dd04'): 'group_size', t('chemical_diversity.text_71d09e557e'): 'degree', t('chemical_diversity.text_400481e52f'): 'close_analog_count'}.get(size_by, 'group_size')
        values = _safe_numeric_series(plot_df, size_col, 1).fillna(1.0).clip(lower=0.0)
        plot_df['marker_size'] = 9.0 + np.sqrt(values + 1.0) * 5.0
    color_col = {t('chemical_diversity.text_5c2c62c55f'): 'group_id', t('chemical_diversity.text_caa3d4dd04'): 'group_size', t('chemical_diversity.text_c0577d0edf'): 'is_singleton_selected', t('chemical_diversity.text_71d09e557e'): 'degree', t('chemical_diversity.text_59dc3f712d'): 'nearest_neighbor_tanimoto', t('chemical_diversity.text_9177d9bf27'): 'csa_class'}.get(color_by, 'group_id')
    class_colors = CSA_CLASS_PALETTES.get(palette_name, CSA_CLASS_COLORS)
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
    singleton_mask = coerce_boolean_series(plot_df['is_singleton_selected'])
    small_isolated_mask = coerce_boolean_series(plot_df['is_small_isolated_group'])
    noise_mask = coerce_boolean_series(plot_df['is_noise'])
    label_mask = singleton_mask | small_isolated_mask | noise_mask
    labels = np.where(show_all_labels, plot_df['name'], np.where(show_singleton_labels & label_mask, plot_df['name'], ''))
    if color_col in {'csa_class', 'is_singleton_selected', 'group_id'}:
        group_col = 'csa_class_code' if color_col == 'csa_class' else color_col
        qualitative_palette = (
            px.colors.qualitative.Safe
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Dark24
        )
        for color_index, (value, group) in enumerate(plot_df.groupby(group_col, dropna=False)):
            group_labels = pd.Series(labels, index=plot_df.index).loc[group.index]
            if color_col == 'csa_class':
                color = class_colors.get(str(value))
                name = _csa_class_display(value)
            elif color_col == 'is_singleton_selected':
                is_singleton_value = bool(coerce_boolean_series([value]).iloc[0])
                color = '#dc2626' if is_singleton_value else '#64748b'
                name = str(value)
            else:
                color = qualitative_palette[color_index % len(qualitative_palette)]
                name = 'missing group' if pd.isna(value) else str(value)
            fig.add_trace(go.Scatter(x=group['csa_x'], y=group['csa_y'], mode='markers+text' if show_singleton_labels or show_all_labels else 'markers', text=group_labels, textposition='top center', marker=dict(size=group['marker_size'], color=color, opacity=0.88, line=dict(width=0.8, color=np.where(coerce_boolean_series(group['is_singleton_selected']), '#dc2626', 'white'))), customdata=np.stack([group['name'].astype(str), group['SMILES'].astype(str), group['method'].astype(str), group['group_id'].astype(str), group['group_size'].astype(str), group['degree'].astype(str), group['nearest_neighbor'].astype(str), group['nearest_neighbor_tanimoto'].astype(str), group['is_singleton_selected'].astype(str), group['is_noise'].astype(str), group['csa_class_label'].astype(str)], axis=-1), hovertemplate=t('chemical_diversity.text_1b37ba0007'), name=name))
    else:
        numeric_color = _safe_numeric_series(plot_df, color_col, np.nan)
        fig.add_trace(go.Scatter(x=plot_df['csa_x'], y=plot_df['csa_y'], mode='markers+text' if show_singleton_labels or show_all_labels else 'markers', text=labels, textposition='top center', marker=dict(size=plot_df['marker_size'], color=numeric_color, colorscale=continuous_scale, showscale=True, colorbar=dict(title=color_col), opacity=0.88, line=dict(width=0.8, color=np.where(singleton_mask, '#dc2626', 'white'))), customdata=np.stack([plot_df['SMILES'].astype(str), plot_df['method'].astype(str), plot_df['group_id'].astype(str), plot_df['group_size'].astype(str), plot_df['degree'].astype(str), plot_df['nearest_neighbor'].astype(str), plot_df['nearest_neighbor_tanimoto'].astype(str), plot_df['is_singleton_selected'].astype(str), plot_df['is_noise'].astype(str), plot_df['csa_class_label'].astype(str)], axis=-1), hovertemplate=t('chemical_diversity.text_5c23e7ef89'), name=color_col))
    fig.update_layout(title=t('chemical_diversity.text_3999391893'), xaxis_title='Координата проекции 1', yaxis_title='Координата проекции 2', height=620, template='plotly_white', legend_title_text=color_by, margin=dict(l=20, r=20, t=70, b=35))
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig

def _cached_structural_communities(map_df, similarity_matrix, method, threshold, top_k, min_cluster_size, butina_cutoff, dbscan_eps, dbscan_min_samples, singleton_criterion):
    return analyze_structural_communities(map_df=map_df, similarity_matrix=np.asarray(similarity_matrix, dtype=float), method=method, threshold=float(threshold), top_k=int(top_k), min_cluster_size=int(min_cluster_size), butina_cutoff=float(butina_cutoff), dbscan_eps=float(dbscan_eps), dbscan_min_samples=int(dbscan_min_samples), singleton_criterion=singleton_criterion)

def _render_structural_communities_block(result):
    final_space = result.get('final_chemical_space', {})
    if not isinstance(final_space, dict):
        return
    map_df = final_space.get('map', pd.DataFrame())
    similarity_matrix = final_space.get('similarity_matrix')
    if not isinstance(map_df, pd.DataFrame) or map_df.empty:
        return
    if similarity_matrix is None:
        st.info(
            "Structural communities require a full similarity matrix. "
            "This block is skipped for large/sampled chemical-space results."
        )
        return
    st.markdown(t('chemical_diversity.text_d46a6d9635'))
    st.caption(t('chemical_diversity.text_b6386d92b9'))
    st.caption(t('chemical_diversity.text_2440bb6377'))
    st.info(t('chemical_diversity.text_a39b9efee1'))
    method_options = ['connected_components']
    if Butina is not None:
        method_options.append('butina')
    if DBSCAN is not None:
        method_options.append('dbscan')
    method_options.extend(['similarity_network', 'singletons_only'])
    method_labels = {
        'connected_components': 'Connected components',
        'butina': 'Butina clustering',
        'dbscan': 'DBSCAN',
        'similarity_network': 'Similarity network',
        'singletons_only': 'Singletons only',
    }
    old_method = st.session_state.get('chemical_space_communities_method')
    method_aliases = {
        'Connected components': 'connected_components',
        'Butina clustering': 'butina',
        'DBSCAN': 'dbscan',
        'Similarity network': 'similarity_network',
        'Singletons only': 'singletons_only',
    }
    if old_method in method_aliases:
        st.session_state.chemical_space_communities_method = method_aliases[old_method]
    method = st.selectbox(
        t('chemical_diversity.text_cff5dfb6f8'),
        method_options,
        key='chemical_space_communities_method',
        format_func=lambda key: method_labels.get(key, str(key)),
    )
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
        singleton_options = ['combined', 'component_size_1', 'no_neighbors', 'cluster_size_lte_n', 'dbscan_noise']
        singleton_labels = {
            'combined': 'combined',
            'component_size_1': 'component size == 1',
            'no_neighbors': 'no neighbors above threshold',
            'cluster_size_lte_n': 'cluster size <= N',
            'dbscan_noise': 'DBSCAN noise',
        }
        old_singleton = st.session_state.get('chemical_space_communities_singleton_criterion')
        singleton_aliases = {label: code for code, label in singleton_labels.items()}
        if old_singleton in singleton_aliases:
            st.session_state.chemical_space_communities_singleton_criterion = singleton_aliases[old_singleton]
        singleton_criterion = st.selectbox(
            t('chemical_diversity.text_1c309142ed'),
            singleton_options,
            key='chemical_space_communities_singleton_criterion',
            format_func=lambda key: singleton_labels.get(key, str(key)),
        )
    with c8:
        color_by = st.selectbox(t('chemical_diversity.text_d11b85826f'), [t('chemical_diversity.text_5c2c62c55f'), t('chemical_diversity.text_caa3d4dd04'), t('chemical_diversity.text_c0577d0edf'), t('chemical_diversity.text_71d09e557e'), t('chemical_diversity.text_59dc3f712d'), t('chemical_diversity.text_9177d9bf27')], key='chemical_space_communities_color_by')
    with c9:
        size_by = st.selectbox(t('chemical_diversity.text_686b1c751c'), [t('chemical_diversity.text_5ab8758886'), t('chemical_diversity.text_caa3d4dd04'), t('chemical_diversity.text_71d09e557e'), t('chemical_diversity.text_400481e52f')], key='chemical_space_communities_size_by')
    (p1, p2) = st.columns(2)
    with p1:
        community_palette_name = st.selectbox(t('chemical_diversity.community_palette_label'), list(CSA_CLASS_PALETTES.keys()), key='chemical_space_communities_palette')
    with p2:
        community_continuous_scale = st.selectbox(t('chemical_diversity.community_numeric_scale_label'), CONTINUOUS_COLOR_SCALES, index=CONTINUOUS_COLOR_SCALES.index('Turbo'), key='chemical_space_communities_continuous_scale')
    (f1, f2) = st.columns([2, 1])
    display_filter = f1.radio(
        t('chemical_diversity.displayed_groups_filter'),
        ['all', 'singletons', 'small', 'large'],
        index=1 if method == 'singletons_only' else 0,
        horizontal=True,
        key='chemical_space_communities_display_filter',
        format_func=lambda value: {
            'all': t('chemical_diversity.display_all_groups'),
            'singletons': t('chemical_diversity.display_singletons_only'),
            'small': t('chemical_diversity.display_small_only'),
            'large': t('chemical_diversity.display_large_only'),
        }.get(value, value),
    )
    show_singleton_labels = f2.checkbox(t('chemical_diversity.text_f4b8644b2f'), value=True, key='chemical_space_communities_singleton_labels')
    show_all_labels = st.checkbox(t('chemical_diversity.text_b5e91a4d0e'), value=False, disabled=len(map_df) > 80, key='chemical_space_communities_all_labels')
    if len(map_df) > 80:
        show_all_labels = False
    communities = _cached_structural_communities(map_df, similarity_matrix, method, float(threshold), int(top_k), int(small_limit), float(butina_cutoff), float(dbscan_eps), int(dbscan_min_samples), singleton_criterion)
    nodes = communities.get('nodes', pd.DataFrame())
    edges = communities.get('edges', pd.DataFrame())
    summary = communities.get('summary', {})
    if isinstance(nodes, pd.DataFrame) and (not nodes.empty):
        filtered_nodes = nodes.copy()
        if display_filter == 'singletons':
            filtered_nodes = filtered_nodes[coerce_boolean_series(filtered_nodes['is_singleton_selected'])].copy()
        elif display_filter == 'small':
            filtered_nodes = filtered_nodes[filtered_nodes['group_size'] <= int(small_limit)].copy()
        elif display_filter == 'large':
            filtered_nodes = filtered_nodes[filtered_nodes['group_size'] > int(small_limit)].copy()
        visible = set(filtered_nodes['node_index'].astype(int).tolist())
        if isinstance(edges, pd.DataFrame) and (not edges.empty):
            filtered_edges = edges[edges['source'].astype(int).isin(visible) & edges['target'].astype(int).isin(visible)].copy()
        else:
            filtered_edges = edges
    else:
        filtered_nodes = nodes
        filtered_edges = edges
    if isinstance(filtered_nodes, pd.DataFrame) and not filtered_nodes.empty:
        group_sizes = pd.to_numeric(filtered_nodes.get('group_size', pd.Series(dtype=float)), errors='coerce')
        group_ids = filtered_nodes.get('group_id', pd.Series(index=filtered_nodes.index, dtype=object))
        display_summary = {
            'n_nodes': int(len(filtered_nodes)),
            'n_groups': int(pd.Series(group_ids).nunique(dropna=True)),
            'n_singletons': int(coerce_boolean_series(filtered_nodes.get('is_singleton_selected', pd.Series(False, index=filtered_nodes.index))).sum()),
            'largest_group_size': int(group_sizes.max()) if not group_sizes.dropna().empty else 0,
        }
    else:
        display_summary = {'n_nodes': 0, 'n_groups': 0, 'n_singletons': 0, 'largest_group_size': 0}
    st.markdown(t('chemical_diversity.full_analysis_metrics'))
    metrics = st.columns(4)
    metrics[0].metric(t('chemical_diversity.text_7c16ae8c7a'), summary.get('n_groups', '—'))
    metrics[1].metric(t('chemical_diversity.singleton_metric'), summary.get('n_singletons', '—'))
    metrics[2].metric(t('chemical_diversity.text_51f205c8a3'), summary.get('n_small_groups', '—'))
    metrics[3].metric(t('chemical_diversity.text_517b5bb8b0'), summary.get('largest_group_size', '—'))
    metrics = st.columns(4)
    metrics[0].metric(t('chemical_diversity.text_3145630c55'), f"{float(summary.get('largest_group_fraction', 0.0)) * 100:.1f}%")
    metrics[1].metric(t('chemical_diversity.noise_metric'), summary.get('noise_points', '—'))
    metrics[2].metric(t('chemical_diversity.text_ed299969be'), _safe_float_text(summary.get('mean_degree')))
    metrics[3].metric(t('chemical_diversity.text_5a246e5093'), summary.get('no_close_neighbors', '—'))
    st.markdown(t('chemical_diversity.current_display_metrics'))
    display_metrics = st.columns(4)
    display_metrics[0].metric(t('chemical_diversity.displayed_points'), display_summary.get('n_nodes', 0))
    display_metrics[1].metric(t('chemical_diversity.displayed_groups'), display_summary.get('n_groups', 0))
    display_metrics[2].metric(t('chemical_diversity.displayed_singletons'), display_summary.get('n_singletons', 0))
    display_metrics[3].metric(t('chemical_diversity.largest_displayed_group'), display_summary.get('largest_group_size', 0))
    st.info(str(summary.get('interpretation', '')))
    fig = _make_structural_communities_figure(filtered_nodes, filtered_edges, color_by=color_by, size_by=size_by, show_singleton_labels=show_singleton_labels, show_all_labels=show_all_labels, palette_name=community_palette_name, continuous_scale=community_continuous_scale)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t('chemical_diversity.communities_projection_note'))
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

    def exact_interpretation_message(value):
        if isinstance(value, dict):
            for key in ("message", "summary", "text", "status"):
                if value.get(key):
                    return str(value.get(key))
            return ""
        if isinstance(value, (list, tuple, set)):
            return "\n".join(str(item) for item in value if str(item).strip())
        if value is None:
            return ""
        return str(value)

    def tr_or_fallback(key, fallback):
        value = t(key)
        return fallback if value == key else value

    st.markdown(t('chemical_diversity.text_a64d545cfb'))
    st.caption(t('chemical_diversity.text_3e28b59c95'))
    st.caption(t('chemical_diversity.text_33ed55496a'))
    st.info(t('chemical_diversity.text_86c94d310f'))
    chart_options = {
        t('chemical_diversity.exact_chart_treemap'): "treemap",
        t('chemical_diversity.text_aa192846a5'): "sunburst",
        t('chemical_diversity.exact_chart_bubble'): "bubble",
    }
    (c1, c2, c3, c4) = st.columns(4)
    with c1:
        chart_label = st.selectbox(
            t('chemical_diversity.text_a6aa4d8f5e'),
            list(chart_options.keys()),
            key='chemical_space_exact_pattern_chart_type'
        )
        chart_type = chart_options.get(chart_label, "treemap")
    with c2:
        color_by = st.selectbox(t('chemical_diversity.text_c5b6ba6451'), [t('chemical_diversity.text_a0aa67d654'), t('chemical_diversity.text_666e9c37f7'), t('chemical_diversity.text_bd4dd06981'), t('chemical_diversity.text_3b4d2673e1')], key='chemical_space_exact_pattern_color_by')
    with c3:
        small_threshold = st.slider(
            t('chemical_diversity.exact_small_group_threshold'),
            1,
            8,
            2,
            1,
            key='chemical_space_exact_pattern_small_threshold'
        )
    with c4:
        rare_min = max(1, int(small_threshold))
        rare_threshold = st.slider(
            t('chemical_diversity.exact_rare_group_threshold'),
            rare_min,
            12,
            max(rare_min, 3),
            1,
            key='chemical_space_exact_pattern_rare_threshold'
        )
    st.caption(t('chemical_diversity.exact_threshold_order_note'))
    _render_current_analysis_parameters(
        [
            {
                "parameter": "Exact pattern singleton threshold",
                "value": 1,
                "module": "Chemical space",
            },
            {
                "parameter": "Exact pattern small group threshold",
                "value": int(small_threshold),
                "module": "Chemical space",
            },
            {
                "parameter": "Exact pattern rare group threshold",
                "value": int(rare_threshold),
                "module": "Chemical space",
            },
        ],
        "exact_patterns",
    )

    unclassified_series_label = tr_or_fallback(
        'chemical_diversity.unclassified_series',
        'Unclassified series'
    )
    unclassified_pattern_label = tr_or_fallback(
        'chemical_diversity.unclassified_pattern',
        'Unclassified'
    )

    def _fill_exact_category(series, fallback):
        values = series if isinstance(series, pd.Series) else pd.Series([], dtype=object)
        out = values.astype("object").where(values.notna(), fallback).astype(str)
        out = out.replace(
            {
                "": fallback,
                "nan": fallback,
                "NaN": fallback,
                "None": fallback,
                "<NA>": fallback,
            }
        )
        return out

    plot_df = groups.copy()
    if 'broad_series' not in plot_df.columns:
        plot_df['broad_series'] = unclassified_series_label
    if 'exact_pattern' not in plot_df.columns:
        plot_df['exact_pattern'] = unclassified_pattern_label
    if 'group_size' not in plot_df.columns:
        plot_df['group_size'] = 0
    for bool_col in ['singleton_group', 'unclassified_group']:
        if bool_col not in plot_df.columns:
            plot_df[bool_col] = False
    if 'representative_examples' not in plot_df.columns:
        plot_df['representative_examples'] = ''
    if 'dataset_fraction' not in plot_df.columns:
        plot_df['dataset_fraction'] = np.nan
    plot_df['broad_series'] = _fill_exact_category(
        plot_df['broad_series'],
        unclassified_series_label,
    )
    plot_df['exact_pattern'] = _fill_exact_category(
        plot_df['exact_pattern'],
        unclassified_pattern_label,
    )
    source_group_size_sum = float(pd.to_numeric(plot_df['group_size'], errors='coerce').fillna(0).sum())
    plot_df['small_group'] = plot_df['group_size'] <= int(small_threshold)
    plot_df['rare_group'] = plot_df['group_size'] <= int(rare_threshold)
    only_small = st.checkbox(
        tr_or_fallback(
            'chemical_diversity.exact_only_small_groups',
            'Only small groups'
        ),
        value=False,
        key='chemical_space_exact_pattern_only_small'
    )
    only_rare = st.checkbox(t('chemical_diversity.text_b1c99699fe'), value=False, key='chemical_space_exact_pattern_only_rare')
    if only_small:
        plot_df = plot_df[
            coerce_boolean_series(plot_df['small_group'])
            | coerce_boolean_series(plot_df['singleton_group'])
            | coerce_boolean_series(plot_df['unclassified_group'])
        ].copy()
    if only_rare:
        plot_df = plot_df[
            coerce_boolean_series(plot_df['rare_group'])
            | coerce_boolean_series(plot_df['singleton_group'])
            | coerce_boolean_series(plot_df['unclassified_group'])
        ].copy()
    if plot_df.empty:
        st.info(t('chemical_diversity.text_a759040fc2'))
        return
    plotted_group_size_sum = float(pd.to_numeric(plot_df['group_size'], errors='coerce').fillna(0).sum())
    color_column = {t('chemical_diversity.text_a0aa67d654'): 'broad_series', t('chemical_diversity.text_666e9c37f7'): 'group_size', t('chemical_diversity.text_bd4dd06981'): 'rare_group', t('chemical_diversity.text_3b4d2673e1'): 'mean_property'}.get(color_by, 'broad_series')
    if color_column == 'mean_property' and (
        'mean_property' not in plot_df.columns
        or not pd.to_numeric(plot_df['mean_property'], errors='coerce').notna().any()
    ):
        color_column = 'broad_series'
        st.caption(t('chemical_diversity.text_7ab7d9c0cd'))
    expected_hover_columns = ['broad_series', 'exact_pattern', 'group_size', 'dataset_fraction', 'small_group', 'rare_group', 'singleton_group', 'unclassified_group', 'representative_examples']
    hover_data = [
        column for column in expected_hover_columns
        if column in plot_df.columns
    ]
    if chart_type == "treemap":
        fig = px.treemap(plot_df, path=['broad_series', 'exact_pattern'], values='group_size', color=color_column, hover_data=hover_data, title=t('chemical_diversity.text_108c9eccf1'))
    elif chart_type == "sunburst":
        fig = px.sunburst(plot_df, path=['broad_series', 'exact_pattern'], values='group_size', color=color_column, hover_data=hover_data, title=t('chemical_diversity.text_108c9eccf1'))
    else:
        bubble_df = plot_df.sort_values('group_size', ascending=False).reset_index(drop=True)
        bubble_df['x'] = np.arange(len(bubble_df))
        series_codes = {name: idx for (idx, name) in enumerate(sorted(bubble_df['broad_series'].astype(str).unique()))}
        bubble_df['y'] = bubble_df['broad_series'].astype(str).map(series_codes)
        labels = np.where(
            coerce_boolean_series(bubble_df['rare_group'])
            | coerce_boolean_series(bubble_df['small_group'])
            | coerce_boolean_series(bubble_df['singleton_group']),
            bubble_df['exact_pattern'],
            ''
        )
        st.caption(tr_or_fallback(
            'chemical_diversity.exact_bubble_geometry_note',
            'Bubble positions are categorical: the X axis is group order, not chemical similarity distance.'
        ))
        fig = px.scatter(
            bubble_df,
            x='x',
            y='y',
            size='group_size',
            color=color_column,
            text=labels,
            hover_data=hover_data,
            title=tr_or_fallback(
                'chemical_diversity.exact_bubble_title',
                'Bubble chart of exact-pattern representation'
            )
        )
        fig.update_yaxes(tickmode='array', tickvals=list(series_codes.values()), ticktext=list(series_codes.keys()), title=t('chemical_diversity.exact_broad_series_axis'))
        fig.update_xaxes(title=t('chemical_diversity.exact_pattern_groups_axis'))
        fig.update_traces(textposition='top center')
    if plotted_group_size_sum > source_group_size_sum + 1e-9:
        st.warning("Exact-pattern plot size check failed: displayed group size exceeds source size.")
    fig.update_layout(height=620, margin=dict(l=20, r=20, t=70, b=35))
    interpretation_message = exact_interpretation_message(
        exact_payload.get('interpretation', '')
    )
    if interpretation_message:
        st.info(interpretation_message)
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
            members_display = members.copy()
            if 'exact_pattern' not in members_display.columns:
                members_display['exact_pattern'] = unclassified_pattern_label
            members_display['exact_pattern_display'] = _fill_exact_category(
                members_display['exact_pattern'],
                unclassified_pattern_label,
            )
            selected = st.selectbox(
                t('chemical_diversity.exact_pattern_select'),
                sorted(members_display['exact_pattern_display'].unique()),
                key='chemical_space_exact_pattern_members_select'
            )
            st.dataframe(
                members_display[members_display['exact_pattern_display'] == selected],
                width='stretch',
                hide_index=True
            )
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
    (fig, ax) = plt.subplots(figsize=(5.2, 3.4))
    if color_mode == 'cluster_id':
        clusters, color_map, top_clusters = _cluster_category_colors(plot_df.get('cluster_id', pd.Series(['unassigned'] * len(plot_df))))
        clusters.index = plot_df.index
        for cluster in list(top_clusters) + ['unassigned']:
            mask = clusters.eq(cluster)
            if not mask.any():
                continue
            ax.scatter(
                plot_df.loc[mask, 'PC1'],
                plot_df.loc[mask, 'PC2'],
                color=color_map.get(cluster, CLUSTER_OTHER_COLOR),
                s=24,
                alpha=0.85,
                edgecolors='none',
                label=_cluster_label(cluster),
            )
        other_mask = ~clusters.isin(top_clusters) & ~clusters.eq('unassigned')
        if other_mask.any():
            ax.scatter(
                plot_df.loc[other_mask, 'PC1'],
                plot_df.loc[other_mask, 'PC2'],
                color=CLUSTER_OTHER_COLOR,
                s=24,
                alpha=0.75,
                edgecolors='none',
                label='Other clusters',
            )
        ax.legend(fontsize=7, loc='best', frameon=False, ncol=1)
    else:
        plot_df[color_mode] = pd.to_numeric(plot_df[color_mode], errors='coerce')
        scatter = ax.scatter(plot_df['PC1'], plot_df['PC2'], c=plot_df[color_mode], cmap='viridis', s=24, alpha=0.85, edgecolors='none')
        cbar = fig.colorbar(scatter, ax=ax, label=color_mode, shrink=0.82)
        cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel('PCA 1', fontsize=9)
    ax.set_ylabel('PCA 2', fontsize=9)
    ax.set_title(t('chemical_diversity.text_ff058db7da'), fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
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
    order_mode = st.selectbox(
        t('chemical_diversity.heatmap_order_label'),
        ['By cluster', 'Original order', 'Hierarchical clustering', 'By exact pattern'],
        format_func=lambda value: {
            'By cluster': t('chemical_diversity.heatmap_order_cluster'),
            'Original order': t('chemical_diversity.heatmap_order_original'),
            'Hierarchical clustering': t('chemical_diversity.heatmap_order_hierarchical'),
            'By exact pattern': t('chemical_diversity.heatmap_order_exact_pattern'),
        }.get(value, value),
        key='chemical_diversity_heatmap_order',
    )
    heatmap_scale = st.selectbox(
        t('chemical_diversity.heatmap_color_scale'),
        CONTINUOUS_COLOR_SCALES,
        index=CONTINUOUS_COLOR_SCALES.index('Viridis'),
        key='chemical_diversity_heatmap_scale',
    )
    order = list(range(len(molecules_df)))
    if order_mode == 'Original order' and 'row' in molecules_df.columns:
        order = list(np.argsort(pd.to_numeric(molecules_df['row'], errors='coerce').fillna(10**12).to_numpy()))
    elif order_mode == 'By exact pattern' and 'exact_pattern' in molecules_df.columns:
        order = list(np.lexsort((
            pd.to_numeric(molecules_df.get('row', pd.Series(range(len(molecules_df)))), errors='coerce').fillna(10**12).to_numpy(),
            molecules_df['exact_pattern'].astype(str).to_numpy(),
        )))
    elif order_mode == 'Hierarchical clustering' and len(matrix_df) >= 3:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform
            dist = np.clip(1.0 - matrix_df.values.astype(float), 0.0, 1.0)
            np.fill_diagonal(dist, 0.0)
            order = list(leaves_list(linkage(squareform(dist, checks=False), method='average')))
        except Exception:
            order = list(range(len(molecules_df)))
    else:
        if 'cluster_id' in molecules_df.columns:
            order = list(np.lexsort((
                pd.to_numeric(molecules_df.get('row', pd.Series(range(len(molecules_df)))), errors='coerce').fillna(10**12).to_numpy(),
                pd.to_numeric(molecules_df['cluster_id'], errors='coerce').fillna(10**9).to_numpy(),
            )))
    matrix_df = matrix_df.iloc[order, order]
    molecules_df = molecules_df.iloc[order].reset_index(drop=True)
    (fig, ax) = plt.subplots(figsize=(4.8, 3.8))
    image = ax.imshow(matrix_df.values, cmap=MATPLOTLIB_CMAP_BY_PLOTLY.get(heatmap_scale, 'viridis'), vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_title(t('chemical_diversity.text_0667369e27'), fontsize=10)
    ax.set_xlabel(t('chemical_diversity.text_f62f40d1bd'), fontsize=9)
    ax.set_ylabel(t('chemical_diversity.text_f62f40d1bd'), fontsize=9)
    if len(matrix_df) <= 40:
        labels = molecules_df.get('label', pd.Series(matrix_df.index)).astype(str).tolist()
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if 'cluster_id' in molecules_df.columns:
        cluster_values = molecules_df['cluster_id'].astype(str).tolist()
        for idx in range(1, len(cluster_values)):
            if cluster_values[idx] != cluster_values[idx - 1]:
                ax.axhline(idx - 0.5, color='white', linewidth=0.45, alpha=0.8)
                ax.axvline(idx - 0.5, color='white', linewidth=0.45, alpha=0.8)
    cbar = fig.colorbar(image, ax=ax, label='Tanimoto', shrink=0.82)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    _show_compact_figure(fig, width=620)
    with st.expander(t('chemical_diversity.text_07e0e29473'), expanded=False):
        st.dataframe(molecules_df, width='stretch', hide_index=True)
    with st.expander(t('chemical_diversity.heatmap_submatrix_expander'), expanded=False):
        max_idx = max(1, len(matrix_df))
        start = st.number_input(t('chemical_diversity.heatmap_start_index'), 1, max_idx, 1, key='chemical_diversity_heatmap_start')
        size = st.number_input(t('chemical_diversity.heatmap_submatrix_size'), 5, min(80, max_idx), min(30, max_idx), key='chemical_diversity_heatmap_size')
        start0 = int(start) - 1
        stop0 = min(start0 + int(size), max_idx)
        sub = matrix_df.iloc[start0:stop0, start0:stop0]
        if not sub.empty:
            fig_sub = px.imshow(sub, color_continuous_scale=heatmap_scale, zmin=0.0, zmax=1.0, aspect='auto')
            st.plotly_chart(fig_sub, use_container_width=True)

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

def _render_analogue_network(pca_df, network_edges, analogue_threshold=0.85, max_edges=500):
    st.caption(t('chemical_diversity.text_c05892029d'))
    if not isinstance(network_edges, pd.DataFrame) or network_edges.empty:
        st.info(t('chemical_diversity.text_93777b4e20'))
        return
    if not isinstance(pca_df, pd.DataFrame) or pca_df.empty:
        st.dataframe(network_edges, width='stretch', hide_index=True)
        return
    node_df = pca_df.set_index('row')
    edge_df = network_edges[network_edges['source_row'].isin(node_df.index) & network_edges['target_row'].isin(node_df.index)].copy()
    if edge_df.empty:
        st.info(t('chemical_diversity.text_d7b3f7703a'))
        return
    edge_df['tanimoto'] = pd.to_numeric(edge_df['tanimoto'], errors='coerce')
    edge_df = edge_df.dropna(subset=['tanimoto']).sort_values('tanimoto', ascending=False)
    max_edges = int(max_edges)
    if len(edge_df) > max_edges:
        st.warning(
            f'Network drawing is limited to the top {max_edges} strongest edges '
            f'of {len(edge_df)} available edges.'
        )
        edge_df = edge_df.head(max_edges).copy()
    st.caption('Coordinates: descriptor PCA; edges: Morgan/Tanimoto similarity.')
    (fig, ax) = plt.subplots(figsize=(5.2, 3.4))
    segments = []
    edge_colors = []
    threshold = float(analogue_threshold)
    denom = max(1.0 - threshold, 1e-9)
    for (_, edge) in edge_df.iterrows():
        src = node_df.loc[int(edge['source_row'])]
        dst = node_df.loc[int(edge['target_row'])]
        segments.append([(src['PC1'], src['PC2']), (dst['PC1'], dst['PC2'])])
        alpha = 0.15 + 0.60 * ((float(edge['tanimoto']) - threshold) / denom)
        alpha = float(np.clip(alpha, 0.15, 0.75))
        edge_colors.append((0.61, 0.64, 0.69, alpha))
    if segments:
        ax.add_collection(LineCollection(segments, colors=edge_colors, linewidths=0.6))
    clusters, color_map, top_clusters = _cluster_category_colors(node_df.get('cluster_id', pd.Series(['unassigned'] * len(node_df), index=node_df.index)))
    clusters.index = node_df.index
    for cluster in list(top_clusters) + ['unassigned']:
        mask = clusters.eq(cluster)
        if not mask.any():
            continue
        ax.scatter(
            node_df.loc[mask, 'PC1'],
            node_df.loc[mask, 'PC2'],
            color=color_map.get(cluster, CLUSTER_OTHER_COLOR),
            s=24,
            alpha=0.9,
            edgecolors='none',
            label=_cluster_label(cluster),
        )
    other_mask = ~clusters.isin(top_clusters) & ~clusters.eq('unassigned')
    if other_mask.any():
        ax.scatter(
            node_df.loc[other_mask, 'PC1'],
            node_df.loc[other_mask, 'PC2'],
            color=CLUSTER_OTHER_COLOR,
            s=24,
            alpha=0.75,
            edgecolors='none',
            label='Other clusters',
        )
    ax.set_title(t('chemical_diversity.text_04e8ff9fb0'), fontsize=10)
    ax.set_xlabel('PCA 1', fontsize=9)
    ax.set_ylabel('PCA 2', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc='best', frameon=False, ncol=1)
    ax.autoscale()
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


def _diversity_status_code(raw_status, summary):
    code = str((summary or {}).get('status_code', '')).strip().upper()
    if code:
        return code
    status_text = str(raw_status or '').strip()
    status_codes = {
        'низкое разнообразие': 'LOW_DIVERSITY',
        'неоднородный датасет': 'HETEROGENEOUS_DATASET',
        'высокое разнообразие': 'HIGH_DIVERSITY',
        'умеренное разнообразие': 'MODERATE_DIVERSITY',
        'не рассчитано': 'NOT_CALCULATED',
        'недостаточно данных': 'INSUFFICIENT_DATA',
        'LOW_DIVERSITY': 'LOW_DIVERSITY',
        'HETEROGENEOUS_DATASET': 'HETEROGENEOUS_DATASET',
        'HIGH_DIVERSITY': 'HIGH_DIVERSITY',
        'MODERATE_DIVERSITY': 'MODERATE_DIVERSITY',
        'NOT_CALCULATED': 'NOT_CALCULATED',
        'INSUFFICIENT_DATA': 'INSUFFICIENT_DATA',
    }
    if status_text in status_codes:
        return status_codes[status_text]
    warning_statuses = {
        'РЅРёР·РєРѕРµ СЂР°Р·РЅРѕРѕР±СЂР°Р·РёРµ',
        'РЅРµРѕРґРЅРѕСЂРѕРґРЅС‹Р№ РґР°С‚Р°СЃРµС‚',
        'низкое разнообразие',
        'неоднородный датасет',
        'LOW_DIVERSITY',
        'HETEROGENEOUS_DATASET',
    }
    success_statuses = {
        'РІС‹СЃРѕРєРѕРµ СЂР°Р·РЅРѕРѕР±СЂР°Р·РёРµ',
        'СѓРјРµСЂРµРЅРЅРѕРµ СЂР°Р·РЅРѕРѕР±СЂР°Р·РёРµ',
        'высокое разнообразие',
        'умеренное разнообразие',
        'HIGH_DIVERSITY',
        'MODERATE_DIVERSITY',
    }
    if status_text in warning_statuses:
        return status_text if status_text in {'LOW_DIVERSITY', 'HETEROGENEOUS_DATASET'} else 'WARNING'
    if status_text in success_statuses:
        return status_text if status_text in {'HIGH_DIVERSITY', 'MODERATE_DIVERSITY'} else 'SUCCESS'
    return 'INFO'


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
    if not PLOTLY_AVAILABLE:
        st.warning(
            t('chemical_diversity.plotly_required')
        )
        return
    if not expanded:
        with st.expander(t('chemical_diversity.text_c45333a0db'), expanded=False):
            return render_chemical_diversity_section(
                data=data,
                smiles_col=smiles_col,
                label_col=label_col,
                target_col=target_col,
                descriptor_df=descriptor_df,
                expanded=True,
            )
    if not label_col:
        for candidate in ('Name', 'name', 'compound_id', 'Compound ID', 'CAS', 'cas'):
            if candidate in data.columns and candidate != smiles_col:
                label_col = candidate
                break
    st.markdown(t('chemical_diversity.text_c45333a0db'))
    st.caption(t('chemical_diversity.section_scope_caption'))
    st.caption(t('chemical_diversity.text_e9cf048210'))
    st.caption(t('chemical_diversity.text_244d21577b'))
    with st.expander(t('chemical_diversity.terms_glossary_title'), expanded=False):
        st.dataframe(
            pd.DataFrame([
                {t('chemical_diversity.term_col'): t('chemical_diversity.term_structural_series'), t('chemical_diversity.meaning_col'): t('chemical_diversity.meaning_structural_series')},
                {t('chemical_diversity.term_col'): t('chemical_diversity.term_cluster'), t('chemical_diversity.meaning_col'): t('chemical_diversity.meaning_cluster')},
                {t('chemical_diversity.term_col'): t('chemical_diversity.term_component'), t('chemical_diversity.meaning_col'): t('chemical_diversity.meaning_component')},
                {t('chemical_diversity.term_col'): t('chemical_diversity.term_pattern'), t('chemical_diversity.meaning_col'): t('chemical_diversity.meaning_pattern')},
                {t('chemical_diversity.term_col'): t('chemical_diversity.term_scaffold'), t('chemical_diversity.meaning_col'): t('chemical_diversity.meaning_scaffold')},
                {t('chemical_diversity.term_col'): t('chemical_diversity.term_isolated'), t('chemical_diversity.meaning_col'): t('chemical_diversity.meaning_isolated')},
            ]),
            width='stretch',
            hide_index=True,
        )
    (col_a, col_b, col_c) = st.columns(3)
    with col_a:
        cluster_threshold = st.slider(t('chemical_diversity.text_a3cd2ce9d4'), min_value=0.3, max_value=0.9, value=0.6, step=0.05, key='chemical_diversity_cluster_threshold')
    with col_b:
        duplicate_threshold = st.slider(t('chemical_diversity.text_2eb512500a'), min_value=0.85, max_value=1.0, value=0.95, step=0.01, key='chemical_diversity_duplicate_threshold')
    with col_c:
        analogue_threshold = st.slider(t('chemical_diversity.text_11a8e7a92f'), min_value=0.6, max_value=0.99, value=0.85, step=0.01, key='chemical_diversity_analogue_threshold')
    threshold_order_valid = (
        float(cluster_threshold) <= float(analogue_threshold) <= float(duplicate_threshold)
    )
    if not threshold_order_valid:
        st.warning(
            t('chemical_diversity.threshold_order_warning')
        )
    (col_d, col_e, col_f, col_g) = st.columns(4)
    with col_d:
        projection_options = {
            'AUTO': 'auto',
            'UMAP': 'UMAP',
            'MDS': 'MDS',
            'TSNE': 't-SNE',
        }
        old_projection_method = st.session_state.get('chemical_diversity_projection_method')
        projection_aliases = {'auto': 'AUTO', 't-SNE': 'TSNE', 'T-SNE': 'TSNE'}
        if old_projection_method in projection_aliases:
            st.session_state.chemical_diversity_projection_method = projection_aliases[old_projection_method]
        projection_method = st.selectbox(
            t('chemical_diversity.text_2a09bff718'),
            list(projection_options.keys()),
            index=0,
            format_func=lambda key: projection_options.get(key, str(key)),
            key='chemical_diversity_projection_method',
        )
    with col_e:
        radius = st.number_input(t('chemical_diversity.morgan_radius_label'), min_value=1, max_value=4, value=2, step=1, key='chemical_diversity_morgan_radius')
    with col_f:
        n_bits = st.selectbox(t('chemical_diversity.morgan_nbits_label'), [1024, 2048, 4096], index=1, key='chemical_diversity_morgan_n_bits')
    with col_g:
        map_top_k = st.number_input(t('chemical_diversity.text_3a641834de'), min_value=1, max_value=20, value=5, step=1, key='chemical_diversity_map_top_k')
    (col_h, col_i) = st.columns(2)
    with col_h:
        structure_source_label = st.selectbox(
            t('chemical_diversity.fingerprint_structure_source'),
            ['standardized_parent', 'original_smiles'],
            index=0,
            format_func=lambda value: {
                'standardized_parent': t('chemical_diversity.structure_standardized_parent'),
                'original_smiles': t('chemical_diversity.structure_original_smiles'),
            }.get(value, value),
            key='chemical_diversity_structure_source',
        )
    fingerprint_structure_source = structure_source_label
    with col_i:
        projection_seed = st.number_input(
            t('chemical_diversity.projection_seed_label'),
            value=42,
            step=1,
            key='chemical_diversity_projection_seed',
            help=t('chemical_diversity.projection_seed_help'),
        )
    st.caption(t('chemical_diversity.random_seed_caption', seed=int(projection_seed)))
    valid_smiles_estimate = int(data[smiles_col].astype(str).str.strip().ne("").sum())
    estimated_matrix_mb = (valid_smiles_estimate ** 2 * 8) / (1024 ** 2)
    if valid_smiles_estimate >= 5000:
        st.warning(
            f"Large chemical-space run: a full {valid_smiles_estimate} x {valid_smiles_estimate} "
            f"similarity matrix is approximately {estimated_matrix_mb:.0f} MB before copies. "
            "For large datasets use sampling/top-k graph mode when available."
        )
    st.caption(t('chemical_diversity.text_8db9159eba'))
    _render_current_analysis_parameters(
        current_analysis_parameters_table(st.session_state),
        "chemical_space",
    )
    signature = f'{CHEMICAL_SPACE_ALGORITHM_VERSION}:{_result_signature(data, smiles_col, descriptor_df, label_col=label_col, target_col=target_col)}:cluster={cluster_threshold:.2f}:dup={duplicate_threshold:.2f}:analog={analogue_threshold:.2f}:projection={projection_method}:radius={int(radius)}:bits={int(n_bits)}:topk={int(map_top_k)}:source={fingerprint_structure_source}:seed={int(projection_seed)}'
    cached = st.session_state.get('chemical_diversity_result')
    cached_signature = st.session_state.get('chemical_diversity_signature')
    if cached is not None and cached_signature != signature:
        st.session_state.chemical_diversity_result = None
        st.session_state.chemical_diversity_signature = None
        cached = None
        cached_signature = None
    run_clicked = st.button(t('chemical_diversity.text_f1c4ac0147'), type='primary', key='run_chemical_diversity')
    if run_clicked or (cached is not None and cached_signature == signature):
        if run_clicked:
            if not threshold_order_valid:
                st.error("Chemical diversity was not started because similarity thresholds are inconsistent.")
                st.stop()
            with st.spinner(t('chemical_diversity.text_df201cd4e3')):
                try:
                    result = analyze_chemical_diversity(data=data, smiles_col=smiles_col, label_col=label_col, target_col=target_col, descriptor_df=descriptor_df, radius=int(radius), n_bits=int(n_bits), duplicate_threshold=duplicate_threshold, analogue_threshold=analogue_threshold, cluster_similarity_threshold=cluster_threshold, projection_method=projection_method, fingerprint_structure_source=fingerprint_structure_source, map_edge_threshold=analogue_threshold, map_edge_top_k=int(map_top_k), random_state=int(projection_seed))
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "errors": [{
                            "stage": "analyze_chemical_diversity",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }],
                        "warnings": [],
                        "summary": {},
                    }
                if not isinstance(result, dict):
                    result = {
                        "status": "failed",
                        "errors": [{
                            "stage": "analyze_chemical_diversity",
                            "error_type": type(result).__name__,
                            "error_message": "Chemical diversity core returned a non-dict result.",
                        }],
                        "warnings": [],
                        "summary": {},
                    }
                result.setdefault("status", "success")
                result.setdefault("errors", [])
                result.setdefault("warnings", [])
                st.session_state.chemical_diversity_result = result
                st.session_state.chemical_diversity_signature = signature
                update_analysis_bundle(
                    st.session_state,
                    "chemical_space",
                    {
                        "ready": True,
                        "algorithm_version": CHEMICAL_SPACE_ALGORITHM_VERSION,
                        "signature": signature,
                        "fingerprint_radius": int(radius),
                        "fingerprint_bits": int(n_bits),
                        "fingerprint_structure_source": fingerprint_structure_source,
                        "label_col": label_col,
                        "label_values_hash": _series_digest(data, label_col),
                        "target_col": target_col,
                        "target_values_hash": _series_digest(data, target_col),
                        "projection_seed": int(projection_seed),
                        "analogue_threshold": float(analogue_threshold),
                        "duplicate_threshold": float(duplicate_threshold),
                    },
                )
        else:
            result = cached
        if not isinstance(result, dict):
            st.error(t('chemical_diversity.invalid_result_type'))
            st.session_state.chemical_diversity_result = None
            st.session_state.chemical_diversity_signature = None
            return
        result.setdefault("status", "success")
        result.setdefault("errors", [])
        result.setdefault("warnings", [])
        if result.get("status") == "failed":
            st.error(t('chemical_diversity.analysis_failed'))
            errors = result.get("errors", [])
            if errors:
                st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)
            return
        summary = result.get('summary', {})
        raw_status = str(summary.get('status', t('chemical_diversity.text_de32203cf2')))
        status_code = _diversity_status_code(raw_status, summary)
        status = _localized_diversity_status(raw_status)
        reasons = _localized_diversity_reasons(summary.get('status_reasons', ''))
        if status_code in {'LOW_DIVERSITY', 'HETEROGENEOUS_DATASET'}:
            st.warning(t('chemical_diversity.status_message', status=status, reasons=reasons))
        elif status_code in {'HIGH_DIVERSITY', 'MODERATE_DIVERSITY'}:
            st.success(t('chemical_diversity.status_message', status=status, reasons=reasons))
        else:
            st.info(t('chemical_diversity.status_message', status=status, reasons=reasons))
        if summary.get('pairwise_mode') == 'sampled':
            st.info(t('chemical_diversity.text_d403523606'))
            st.dataframe(
                pd.DataFrame([{
                    'pairs_used': summary.get('pairs_used'),
                    'total_pairs': summary.get('total_pairs'),
                    'fraction_of_all_pairs': summary.get('pair_sample_fraction'),
                    'random_seed': summary.get('random_seed'),
                    'mean_tanimoto': summary.get('mean_tanimoto'),
                    'mean_tanimoto_ci95_low': summary.get('mean_tanimoto_bootstrap_ci95_low'),
                    'mean_tanimoto_ci95_high': summary.get('mean_tanimoto_bootstrap_ci95_high'),
                }]),
                width='stretch',
                hide_index=True,
            )
        if summary.get('cluster_sampled'):
            st.info(t('chemical_diversity.text_a38981340f'))
        metric_cols = st.columns(4)
        metric_cols[0].metric(t('chemical_diversity.text_a742738aa1'), _safe_float_text(summary.get('mean_tanimoto')))
        metric_cols[1].metric(t('chemical_diversity.text_a7f99b1892'), summary.get('pairs_gt_0_95', '—'))
        metric_cols[2].metric(t('chemical_diversity.text_90cd825cac'), summary.get('n_clusters', '—'))
        metric_cols[3].metric(t('chemical_diversity.text_1b605ecac0'), summary.get('unique_molecules_lt_0_30', '—'))
        st.dataframe(_summary_table(summary), width='stretch', hide_index=True)
        final_space = result.get('final_chemical_space', {})
        projection_quality = final_space.get('projection_quality', {}) if isinstance(final_space, dict) else {}
        if isinstance(projection_quality, dict) and projection_quality:
            with st.expander(t('chemical_diversity.projection_quality_title'), expanded=False):
                st.caption(t('chemical_diversity.projection_quality_caption'))
                st.dataframe(pd.DataFrame([projection_quality]), width='stretch', hide_index=True)
        sensitivity_df = result.get('cluster_threshold_sensitivity', pd.DataFrame())
        if isinstance(sensitivity_df, pd.DataFrame) and not sensitivity_df.empty:
            with st.expander(t('chemical_diversity.cluster_sensitivity_title'), expanded=False):
                st.caption(t('chemical_diversity.cluster_sensitivity_caption'))
                st.dataframe(sensitivity_df, width='stretch', hide_index=True)
                st.line_chart(
                    sensitivity_df.set_index('tanimoto_threshold')[
                        ['n_clusters', 'n_singletons', 'largest_cluster_size']
                    ]
                )
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
                        st.scatter_chart(coords, x='PC1', y='PC2')
        with tab_space:
            _render_pca_map(result.get('fingerprint_pca', pd.DataFrame()))
        with tab_heatmap:
            _render_similarity_heatmap(result.get('similarity_heatmap', {}))
        with tab_clusters:
            _render_clusters(result.get('cluster_summary', pd.DataFrame()), summary)
        with tab_network:
            fingerprint_config = result.get('fingerprint_config', {})
            _render_analogue_network(
                result.get('fingerprint_pca', pd.DataFrame()),
                result.get('network_edges', pd.DataFrame()),
                analogue_threshold=float(fingerprint_config.get('analogue_threshold', 0.85)),
                max_edges=500,
            )
        with tab_pairs:
            _render_pairs_and_unique(result)
            invalid_df = result.get('invalid_structures', pd.DataFrame())
            if isinstance(invalid_df, pd.DataFrame) and (not invalid_df.empty):
                with st.expander(t('chemical_diversity.text_1ea1ac107e'), expanded=False):
                    reason_col = 'reason' if 'reason' in invalid_df.columns else 'reason_code'
                    if reason_col in invalid_df.columns:
                        reason_summary = (
                            invalid_df.groupby(reason_col, dropna=False)
                            .size()
                            .reset_index(name='count')
                            .sort_values('count', ascending=False)
                        )
                        st.dataframe(reason_summary, width='stretch', hide_index=True)
                    st.dataframe(invalid_df, width='stretch', hide_index=True)
        _render_final_chemical_space(result)
        _render_structural_communities_block(result)
        _render_exact_pattern_map_block(result)
