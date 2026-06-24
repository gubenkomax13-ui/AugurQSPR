# -*- coding: utf-8 -*-
"""Streamlit interface for graph-based QSPR error analysis."""

import numpy as np
import pandas as pd
import streamlit as st

from modules.error_analysis_core import (
    error_analysis_chemical_annotations,
    error_analysis_cluster_summary,
    error_analysis_group_summary,
    error_analysis_prepare_table,
    error_analysis_problem_molecules,
    error_analysis_select_cluster_members,
    error_analysis_select_group_members,
    error_analysis_structural_annotations,
    error_analysis_structural_series_summary,
    error_analysis_structure_clusters,
    error_analysis_substitution_effects,
)
from modules.i18n import t


def _source_payload(context, model_name):
    session = st.session_state
    sources = {}

    kfold = session.get("kfold_results_dict", {}).get(model_name)
    if isinstance(kfold, dict):
        sources["kfold"] = {
            "label": t("error_analysis.source_kfold", k=kfold.get("k", "")),
            "smiles": list(kfold.get("smiles", [])),
            "y_true": np.asarray(kfold.get("y", []), dtype=float),
            "y_pred": np.asarray(kfold.get("y_pred_cv", []), dtype=float),
            "indices": list(kfold.get(
                "valid_indices", range(len(kfold.get("y", [])))
            )),
        }

    loo = session.get("loo_results_dict", {}).get(model_name)
    if isinstance(loo, dict):
        sources["loo"] = {
            "label": t("error_analysis.source_loo"),
            "smiles": list(loo.get("smiles", [])),
            "y_true": np.asarray(loo.get("y", []), dtype=float),
            "y_pred": np.asarray(loo.get("y_pred_loo", []), dtype=float),
            "indices": list(loo.get(
                "valid_indices", range(len(loo.get("y", [])))
            )),
        }

    holdout = session.get("holdout_results_dict", {}).get(model_name)
    if isinstance(holdout, dict):
        sources["holdout"] = {
            "label": t("error_analysis.source_holdout"),
            "smiles": list(holdout.get("test_smiles", [])),
            "y_true": np.asarray(holdout.get("y_test", []), dtype=float),
            "y_pred": np.asarray(
                holdout.get("y_pred_test", []), dtype=float
            ),
            "indices": list(holdout.get(
                "test_orig_indices",
                range(len(holdout.get("y_test", []))),
            )),
        }

    trained = session.get("trained_models", {}).get(model_name)
    required = {
        "data", "smiles_col_current", "valid_indices_current", "y_all_current"
    }
    if isinstance(trained, dict) and required.issubset(context):
        indices = list(context["valid_indices_current"])
        sources["training"] = {
            "label": t("error_analysis.source_training"),
            "smiles": context["data"][
                context["smiles_col_current"]
            ].iloc[indices].astype(str).tolist(),
            "y_true": np.asarray(context["y_all_current"], dtype=float),
            "y_pred": np.asarray(trained.get("y_pred", []), dtype=float),
            "indices": indices,
        }
    return sources


def _group_labels(high_molwt, high_logp, high_tpsa):
    return {
        "overall": t("error_analysis.group_overall"),
        "group_hydrocarbon": t("error_analysis.group_hydrocarbon"),
        "group_aromatic": t("error_analysis.group_aromatic"),
        "group_non_aromatic": t("error_analysis.group_non_aromatic"),
        "group_cyclic": t("error_analysis.group_cyclic"),
        "group_acyclic": t("error_analysis.group_acyclic"),
        "group_heterocyclic": t("error_analysis.group_heterocyclic"),
        "group_contains_n": t("error_analysis.group_contains_n"),
        "group_contains_o": t("error_analysis.group_contains_o"),
        "group_contains_s": t("error_analysis.group_contains_s"),
        "group_contains_p": t("error_analysis.group_contains_p"),
        "group_halogenated": t("error_analysis.group_halogenated"),
        "group_charged": t("error_analysis.group_charged"),
        "group_high_molwt": t(
            "error_analysis.group_high_molwt", value=high_molwt
        ),
        "group_high_logp": t(
            "error_analysis.group_high_logp", value=high_logp
        ),
        "group_high_tpsa": t(
            "error_analysis.group_high_tpsa", value=high_tpsa
        ),
        "group_flexible": t("error_analysis.group_flexible"),
        "group_multiple_rings": t("error_analysis.group_multiple_rings"),
        "fg_hydroxyl": t("error_analysis.group_hydroxyl"),
        "fg_carboxylic_acid": t(
            "error_analysis.group_carboxylic_acid"
        ),
        "fg_ester": t("error_analysis.group_ester"),
        "fg_amide": t("error_analysis.group_amide"),
        "fg_amine": t("error_analysis.group_amine"),
        "fg_ether": t("error_analysis.group_ether"),
        "fg_carbonyl": t("error_analysis.group_carbonyl"),
        "fg_nitrile": t("error_analysis.group_nitrile"),
        "fg_nitro": t("error_analysis.group_nitro"),
    }


def _reliability_label(value):
    return {
        "adequate": t("error_analysis.reliability_adequate"),
        "small_group": t("error_analysis.reliability_small"),
        "insufficient": t("error_analysis.reliability_insufficient"),
    }.get(value, value)


def _show_molecules(context, table, title, key):
    renderer = context.get("show_molecule_grid_from_table")
    if callable(renderer) and not table.empty:
        renderer(
            table_df=table,
            title=title,
            target_col="absolute_error",
            smiles_col="SMILES",
            max_molecules=100,
            key_prefix=key,
        )


def _series_tab(context, result, model_name):
    summary = result["series_summary"]
    table = result["structural_error_table"]
    if summary.empty:
        st.info(t("error_analysis.no_series"))
        return

    columns = [
        "family", "scaffold", "substitution_scheme", "structural_series",
        "n", "size_range", "mae", "rmse", "bias",
        "mae_vs_overall_percent", "absolute_error_size_slope",
        "order_preservation", "reliability",
    ]
    display = summary[[column for column in columns if column in summary]].copy()
    display["reliability"] = display["reliability"].map(_reliability_label)
    display = display.rename(columns={
        "family": t("error_analysis.col_family"),
        "scaffold": t("error_analysis.col_scaffold"),
        "substitution_scheme": t("error_analysis.col_scheme"),
        "structural_series": t("error_analysis.col_series"),
        "n": t("error_analysis.col_n"),
        "size_range": t("error_analysis.col_size_range"),
        "mae": t("error_analysis.col_mae"),
        "rmse": t("error_analysis.col_rmse"),
        "bias": t("error_analysis.col_bias"),
        "mae_vs_overall_percent": t(
            "error_analysis.col_mae_vs_overall"
        ),
        "absolute_error_size_slope": t(
            "error_analysis.col_error_trend"
        ),
        "order_preservation": t(
            "error_analysis.col_order_preservation"
        ),
        "reliability": t("error_analysis.col_reliability"),
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    options = summary["series_id"].tolist()
    labels = dict(zip(summary["series_id"], summary["structural_series"]))
    selected = st.selectbox(
        t("error_analysis.select_series"),
        options,
        format_func=lambda value: labels.get(value, value),
        key=f"error_series_{model_name}",
    )
    members = table[table["series_id"] == selected].sort_values(
        ["size", "absolute_error"], ascending=[True, False]
    )
    detail = summary[summary["series_id"] == selected].iloc[0]
    st.caption(
        t(
            "error_analysis.series_trend_caption",
            experimental=detail["experimental_monotonicity"],
            predicted=detail["predicted_monotonicity"],
            correlation=(
                f"{detail['property_size_correlation']:.3f}"
                if np.isfinite(detail["property_size_correlation"]) else "—"
            ),
        )
    )
    member_columns = [
        "original_index", "SMILES", "size", "experimental", "predicted",
        "error", "absolute_error", "family", "scaffold",
        "substitution_scheme",
    ]
    st.dataframe(
        members[[column for column in member_columns if column in members]],
        use_container_width=True,
        hide_index=True,
    )
    if members["size"].nunique() > 1:
        chart = members.sort_values("size").set_index("size")[
            ["experimental", "predicted"]
        ]
        st.line_chart(chart)
    _show_molecules(
        context, members,
        t("error_analysis.series_structures", series=labels[selected]),
        f"error_series_molecules_{model_name}_{abs(hash(selected))}",
    )
    st.download_button(
        t("error_analysis.download_series"),
        summary.to_csv(index=False).encode("utf-8"),
        f"error_structural_series_{model_name}.csv",
        "text/csv",
        key=f"download_error_series_{model_name}",
    )


def _effects_tab(result, model_name):
    summary = result["effect_summary"]
    pairs = result["effect_pairs"]
    if summary.empty:
        st.info(t("error_analysis.no_effect_pairs"))
        return
    display = summary.copy()
    display["reliability"] = display["reliability"].map(_reliability_label)
    display = display.rename(columns={
        "family": t("error_analysis.col_family"),
        "scaffold": t("error_analysis.col_scaffold"),
        "substitution_scheme": t("error_analysis.col_scheme"),
        "structural_series": t("error_analysis.col_series"),
        "n": t("error_analysis.col_n"),
        "effect_mae": t("error_analysis.col_effect_mae"),
        "effect_rmse": t("error_analysis.col_effect_rmse"),
        "effect_bias": t("error_analysis.col_effect_bias"),
        "direction_accuracy": t("error_analysis.col_direction_accuracy"),
        "experimental_effect_slope": t(
            "error_analysis.col_experimental_effect_slope"
        ),
        "predicted_effect_slope": t(
            "error_analysis.col_predicted_effect_slope"
        ),
        "reliability": t("error_analysis.col_reliability"),
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    options = summary["effect_series_id"].tolist()
    labels = dict(zip(
        summary["effect_series_id"], summary["structural_series"]
    ))
    selected = st.selectbox(
        t("error_analysis.select_effect_series"),
        options,
        format_func=lambda value: labels.get(value, value),
        key=f"error_effect_series_{model_name}",
    )
    selected_pairs = pairs[pairs["effect_series_id"] == selected].sort_values(
        "comparison_size"
    )
    st.dataframe(selected_pairs, use_container_width=True, hide_index=True)
    if selected_pairs["comparison_size"].nunique() > 1:
        chart = selected_pairs.set_index("comparison_size")[
            ["delta_experimental", "delta_predicted"]
        ]
        st.line_chart(chart)
    st.download_button(
        t("error_analysis.download_effects"),
        pairs.to_csv(index=False).encode("utf-8"),
        f"error_substitution_effects_{model_name}.csv",
        "text/csv",
        key=f"download_error_effects_{model_name}",
    )


def _groups_tab(context, result, model_name):
    summary = result["group_summary"]
    table = result["annotated_error_table"]
    st.caption(t("error_analysis.secondary_groups_note"))
    if summary.empty:
        st.info(t("error_analysis.no_groups"))
        return
    display = summary.copy()
    display["reliability"] = display["reliability"].map(_reliability_label)
    st.dataframe(display, use_container_width=True, hide_index=True)
    selectable = summary[summary["group_id"] != "overall"]
    if not selectable.empty:
        labels = dict(zip(selectable["group_id"], selectable["group"]))
        selected = st.selectbox(
            t("error_analysis.select_group"),
            selectable["group_id"].tolist(),
            format_func=lambda value: labels.get(value, value),
            key=f"error_group_{model_name}",
        )
        members = error_analysis_select_group_members(
            table, selected
        ).sort_values("absolute_error", ascending=False)
        st.dataframe(members, use_container_width=True, hide_index=True)
        _show_molecules(
            context, members,
            t("error_analysis.group_structures", group=labels[selected]),
            f"error_group_molecules_{model_name}_{selected}",
        )
    st.download_button(
        t("error_analysis.download_groups"),
        summary.to_csv(index=False).encode("utf-8"),
        f"error_groups_{model_name}.csv",
        "text/csv",
        key=f"download_error_groups_{model_name}",
    )


def _clusters_tab(context, result, model_name):
    summary = result["cluster_summary"]
    table = result["clustered_error_table"]
    st.caption(t("error_analysis.cluster_explanation"))
    if summary.empty:
        st.info(t("error_analysis.no_clusters"))
        return
    display = summary.copy()
    display["reliability"] = display["reliability"].map(_reliability_label)
    st.dataframe(display, use_container_width=True, hide_index=True)
    selected = st.selectbox(
        t("error_analysis.select_cluster"),
        summary["cluster_id"].astype(int).tolist(),
        format_func=lambda value: t(
            "error_analysis.cluster_label", value=value
        ),
        key=f"error_cluster_{model_name}",
    )
    members = error_analysis_select_cluster_members(
        table, selected
    ).sort_values("absolute_error", ascending=False)
    st.dataframe(members, use_container_width=True, hide_index=True)
    _show_molecules(
        context, members,
        t("error_analysis.cluster_structures", cluster=selected),
        f"error_cluster_molecules_{model_name}_{selected}",
    )
    st.download_button(
        t("error_analysis.download_clusters"),
        summary.to_csv(index=False).encode("utf-8"),
        f"error_clusters_{model_name}.csv",
        "text/csv",
        key=f"download_error_clusters_{model_name}",
    )


def _problems_tab(context, result, model_name):
    problems = result["problem_molecules"]
    if problems.empty:
        st.info(t("error_analysis.no_problem_molecules"))
        return
    large_only = st.checkbox(
        t("error_analysis.large_errors_only"),
        value=True,
        key=f"error_large_only_{model_name}",
    )
    visible = (
        problems[problems["large_error"]]
        if large_only else problems
    )
    st.dataframe(visible, use_container_width=True, hide_index=True)
    _show_molecules(
        context, visible,
        t("error_analysis.problem_structures"),
        f"error_problem_molecules_{model_name}",
    )
    st.download_button(
        t("error_analysis.download_problems"),
        problems.to_csv(index=False).encode("utf-8"),
        f"error_problem_molecules_{model_name}.csv",
        "text/csv",
        key=f"download_error_problems_{model_name}",
    )


def render_error_analysis_section(context):
    """Render structural-series-first error analysis."""
    st.header(t("error_analysis.title"))
    st.markdown(t("error_analysis.description_structural"))

    model_name = st.session_state.get("last_model_algorithm", "")
    sources = _source_payload(context, model_name)
    if not sources:
        st.info(t("error_analysis.no_model"))
        return

    source_order = [
        value for value in ("kfold", "loo", "holdout", "training")
        if value in sources
    ]
    source_name = st.selectbox(
        t("error_analysis.source_selector"),
        source_order,
        format_func=lambda value: sources[value]["label"],
        key=f"error_source_selector_{model_name}",
    )
    source = sources[source_name]
    if source_name == "training":
        st.warning(t(
            "error_analysis.training_warning", source=source["label"]
        ))
    else:
        st.success(t(
            "error_analysis.independent_source", source=source["label"]
        ))

    object_count = len(source["y_true"])
    with st.expander(t("error_analysis.settings_expander"), expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_series_size = st.number_input(
                t("error_analysis.min_series_size"),
                min_value=1,
                max_value=max(1, object_count),
                value=min(5, max(1, object_count)),
                step=1,
                key=f"error_min_series_{model_name}_{source_name}",
            )
            bootstrap_repeats = st.number_input(
                t("error_analysis.bootstrap_repeats"),
                min_value=50, max_value=2000, value=500, step=50,
                key=f"error_bootstrap_{model_name}_{source_name}",
            )
            large_multiplier = st.number_input(
                t("error_analysis.large_error_multiplier"),
                min_value=1.0, max_value=10.0, value=2.0, step=0.25,
                key=f"error_large_multiplier_{model_name}_{source_name}",
            )
        with col2:
            high_molwt = st.number_input(
                t("error_analysis.high_molwt"),
                min_value=50.0, max_value=2000.0, value=300.0, step=25.0,
                key=f"error_molwt_{model_name}_{source_name}",
            )
            high_logp = st.number_input(
                t("error_analysis.high_logp"),
                min_value=-5.0, max_value=15.0, value=3.0, step=0.5,
                key=f"error_logp_{model_name}_{source_name}",
            )
            high_tpsa = st.number_input(
                t("error_analysis.high_tpsa"),
                min_value=0.0, max_value=500.0, value=90.0, step=10.0,
                key=f"error_tpsa_{model_name}_{source_name}",
            )
        with col3:
            similarity = st.slider(
                t("error_analysis.similarity_threshold"),
                min_value=0.30, max_value=0.90, value=0.60, step=0.05,
                key=f"error_similarity_{model_name}_{source_name}",
            )
            min_cluster_size = st.number_input(
                t("error_analysis.min_cluster_size"),
                min_value=1,
                max_value=max(1, object_count),
                value=min(5, max(1, object_count)),
                step=1,
                key=f"error_min_cluster_{model_name}_{source_name}",
            )

    result_key = f"error_analysis_result_{model_name}"
    if st.button(
        t("error_analysis.run_button"),
        type="primary",
        key=f"run_error_analysis_{model_name}_{source_name}",
    ):
        try:
            if not (
                len(source["smiles"])
                == len(source["y_true"])
                == len(source["y_pred"])
            ):
                raise ValueError(t("error_analysis.length_mismatch"))

            error_table = error_analysis_prepare_table(
                source["smiles"], source["y_true"], source["y_pred"],
                source["indices"],
            )
            structural = error_analysis_structural_annotations(
                source["smiles"]
            )
            series_summary, structural_table = (
                error_analysis_structural_series_summary(
                    error_table, structural,
                    min_series_size=int(min_series_size),
                    n_bootstrap=int(bootstrap_repeats),
                    large_error_multiplier=float(large_multiplier),
                )
            )
            effect_pairs, effect_summary = (
                error_analysis_substitution_effects(
                    structural_table,
                    min_series_size=int(min_series_size),
                )
            )
            chemical = error_analysis_chemical_annotations(
                source["smiles"],
                high_molwt=float(high_molwt),
                high_logp=float(high_logp),
                high_tpsa=float(high_tpsa),
            )
            group_summary, annotated_table = error_analysis_group_summary(
                error_table, chemical,
                group_labels=_group_labels(
                    high_molwt, high_logp, high_tpsa
                ),
                min_group_size=int(min_series_size),
                n_bootstrap=int(bootstrap_repeats),
                large_error_multiplier=float(large_multiplier),
            )
            assignments = error_analysis_structure_clusters(
                source["smiles"],
                similarity_threshold=float(similarity),
                min_cluster_size=int(min_cluster_size),
            )
            cluster_summary, clustered_table = error_analysis_cluster_summary(
                error_table, assignments,
                min_cluster_size=int(min_cluster_size),
                n_bootstrap=int(bootstrap_repeats),
                large_error_multiplier=float(large_multiplier),
            )
            if not cluster_summary.empty:
                cluster_summary["group"] = cluster_summary[
                    "cluster_id"
                ].apply(lambda value: t(
                    "error_analysis.cluster_label", value=int(value)
                ))

            st.session_state[result_key] = {
                "source": source_name,
                "source_label": source["label"],
                "series_summary": series_summary,
                "structural_error_table": structural_table,
                "effect_pairs": effect_pairs,
                "effect_summary": effect_summary,
                "group_summary": group_summary,
                "annotated_error_table": annotated_table,
                "cluster_summary": cluster_summary,
                "clustered_error_table": clustered_table,
                "problem_molecules": error_analysis_problem_molecules(
                    structural_table,
                    large_error_multiplier=float(large_multiplier),
                ),
            }
            st.rerun()
        except Exception as error:
            st.error(t("error_analysis.run_error", error=error))

    result = st.session_state.get(result_key)
    if (
        not isinstance(result, dict)
        or result.get("source") != source_name
    ):
        return

    series_summary = result["series_summary"]
    if not series_summary.empty:
        overall_n = int(series_summary["n"].sum())
        structural_table = result["structural_error_table"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("error_analysis.metric_objects"), overall_n)
        col2.metric(
            t("error_analysis.metric_mae"),
            f"{structural_table['absolute_error'].mean():.4g}",
        )
        col3.metric(
            t("error_analysis.metric_rmse"),
            f"{np.sqrt(structural_table['squared_error'].mean()):.4g}",
        )
        col4.metric(
            t("error_analysis.metric_bias"),
            f"{structural_table['error'].mean():+.4g}",
        )

    tabs = st.tabs([
        t("error_analysis.tab_series"),
        t("error_analysis.tab_effects"),
        t("error_analysis.tab_groups_secondary"),
        t("error_analysis.tab_clusters"),
        t("error_analysis.tab_problems"),
    ])
    with tabs[0]:
        _series_tab(context, result, model_name)
    with tabs[1]:
        _effects_tab(result, model_name)
    with tabs[2]:
        _groups_tab(context, result, model_name)
    with tabs[3]:
        _clusters_tab(context, result, model_name)
    with tabs[4]:
        _problems_tab(context, result, model_name)
