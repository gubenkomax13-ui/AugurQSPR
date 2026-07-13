# -*- coding: utf-8 -*-
"""Обучение QSPR-моделей и отображение результатов обучения."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import norm

from modules.i18n import t
from modules.analysis_state import analysis_result_hash, attach_result_cache_metadata, cached_result_is_current
from modules.module_explain_ui import render_module_explanation
from modules.model_catalog import (
    MODEL_GROUP_BOOSTING,
    MODEL_GROUP_KERNEL_SIMILARITY,
    MODEL_GROUP_LINEAR,
    MODEL_GROUP_META_ENSEMBLES,
    MODEL_GROUP_NEURAL,
    MODEL_GROUP_SPLINE,
    MODEL_GROUP_SYMBOLIC,
    MODEL_GROUP_TREES,
    MODEL_GROUP_TREE_ENSEMBLES,
    get_model_display_name,
    get_models_by_group,
    normalize_model_id,
)
from modules.qspr_core import (
    qspr_auto_select_and_tune,
    qspr_available_model_options,
    qspr_get_model_help,
    qspr_model_applicability_guidance,
    qspr_metrics,
    qspr_prediction_table,
    qspr_save_results_auto,
    qspr_seed_stability_holdout,
    qspr_train_analysis_model,
)
from modules.structural_filter_core import qspr_show_descriptor_meaning_table


def render_training_section(context):
    """Рендерит обучение и возвращает контекст выбранной модели."""
    globals().update(context)
    # ------------------------------------------------------------------
    # Model selection
    
    st.header(t('model_selection.header'))
    render_module_explanation("training")
    
    MODEL_FILTER_OPTIONS = {
        "small_datasets": t("model_catalog.filter_small_datasets"),
        "interpretation": t("model_catalog.filter_interpretation"),
        "nonlinear": t("model_catalog.filter_nonlinear"),
        "high_accuracy": t("model_catalog.filter_high_accuracy"),
        "beginners": t("model_catalog.filter_beginners"),
    }
    
    with st.expander(t("model_catalog.filters_title"), expanded=False):
        selected_model_filters = st.multiselect(
            t("model_catalog.filters_label"),
            options=list(MODEL_FILTER_OPTIONS),
            format_func=lambda key: MODEL_FILTER_OPTIONS[key],
            key="model_catalog_filters",
            placeholder=t("model_catalog.filters_placeholder"),
        )
    
        model_filter_match_mode = st.radio(
            t("model_catalog.match_mode_label"),
            options=["all", "any"],
            format_func=lambda mode: (
                t("model_catalog.match_all")
                if mode == "all"
                else t("model_catalog.match_any")
            ),
            horizontal=True,
            key="model_catalog_match_mode",
        )
    
    def qspr_app_model_options_compat(active_filters=None, match_mode="all"):
        """
        Совместимый вызов каталога моделей.
    
        Поддерживает:
        - новое qspr_core с active_filters/match_mode;
        - старое qspr_core + отдельный modules.model_catalog;
        - старое qspr_core без каталога (фильтры временно недоступны).
        """
        active_filters = list(active_filters or [])
    
        try:
            return qspr_available_model_options(
                active_filters=active_filters,
                match_mode=match_mode,
            )
        except TypeError:
            pass
    
        try:
            from modules.model_catalog import get_models_by_group
    
            availability = {
                "xgboost": bool(globals().get("xgboost_available", False)),
                "lightgbm": bool(globals().get("lightgbm_available", False)),
                "catboost": bool(globals().get("catboost_available", False)),
                "pysr": bool(globals().get("pysr_available", False)),
            }
            return get_models_by_group(
                active_filters=active_filters,
                match_mode=match_mode,
                availability=availability,
            )
        except Exception:
            return qspr_available_model_options()
    
    
    model_groups = qspr_app_model_options_compat(
        active_filters=selected_model_filters,
        match_mode=model_filter_match_mode,
    )
    
    filtered_model_count = sum(len(models) for models in model_groups.values())
    total_model_count = sum(
        len(models)
        for models in qspr_app_model_options_compat().values()
    )
    
    if filtered_model_count == 0:
        st.warning(t("model_catalog.no_matches"))
        model_groups = qspr_app_model_options_compat()
        filtered_model_count = total_model_count
    else:
        st.caption(t(
            "model_catalog.results_count",
            filtered=filtered_model_count,
            total=total_model_count,
        ))
    
    if st.session_state.get("model_group_radio") not in model_groups:
        st.session_state.model_group_radio = next(iter(model_groups))
    
    MODEL_GROUP_LABELS = {
        "ru": {
            MODEL_GROUP_LINEAR: "Линейные и регуляризованные методы",
            MODEL_GROUP_KERNEL_SIMILARITY: "Методы сходства и ядерные методы",
            MODEL_GROUP_SPLINE: "Кусочно-линейные и сплайновые методы",
            MODEL_GROUP_TREES: "Деревья решений",
            MODEL_GROUP_TREE_ENSEMBLES: "Ансамбли деревьев",
            MODEL_GROUP_BOOSTING: "Бустинговые методы",
            MODEL_GROUP_NEURAL: "Нейросетевые методы",
            MODEL_GROUP_META_ENSEMBLES: "Метаансамбли",
            MODEL_GROUP_SYMBOLIC: "Символическая и эволюционная регрессия",
        },
        "en": {
            MODEL_GROUP_LINEAR: "Linear and regularized methods",
            MODEL_GROUP_KERNEL_SIMILARITY: "Similarity-based and kernel methods",
            MODEL_GROUP_SPLINE: "Piecewise linear and spline methods",
            MODEL_GROUP_TREES: "Decision trees",
            MODEL_GROUP_TREE_ENSEMBLES: "Tree ensembles",
            MODEL_GROUP_BOOSTING: "Boosting methods",
            MODEL_GROUP_NEURAL: "Neural network methods",
            MODEL_GROUP_META_ENSEMBLES: "Meta-ensembles",
            MODEL_GROUP_SYMBOLIC: "Symbolic and evolutionary regression",
        },
        "kk": {
            MODEL_GROUP_LINEAR: "Сызықтық және регуляризацияланған әдістер",
            MODEL_GROUP_KERNEL_SIMILARITY: "Ұқсастыққа негізделген және ядролық әдістер",
            MODEL_GROUP_SPLINE: "Бөліктік-сызықтық және сплайндық әдістер",
            MODEL_GROUP_TREES: "Шешім ағаштары",
            MODEL_GROUP_TREE_ENSEMBLES: "Ағаш ансамбльдері",
            MODEL_GROUP_BOOSTING: "Бустинг әдістері",
            MODEL_GROUP_NEURAL: "Нейрондық желі әдістері",
            MODEL_GROUP_META_ENSEMBLES: "Метаансамбльдер",
            MODEL_GROUP_SYMBOLIC: "Символдық және эволюциялық регрессия",
        },
    }
    
    model_group_labels = MODEL_GROUP_LABELS.get(
        st.session_state.get("lang", "ru"),
        MODEL_GROUP_LABELS["ru"]
    )
    
    model_group = st.radio(
        t('model_selection.group_label'),
        list(model_groups.keys()),
        horizontal=True,
        key="model_group_radio",
        format_func=lambda group: model_group_labels.get(group, group)
    )
    
    model_options = model_groups[model_group]
    
    current_model_code = normalize_model_id(st.session_state.get("model_algorithm_radio"))
    if current_model_code not in model_options:
        current_model_code = model_options[0]
    st.session_state.model_algorithm_radio = current_model_code
    
    model_name = st.radio(
        t('model_selection.model_label'),
        model_options,
        horizontal=True,
        key="model_algorithm_radio",
        format_func=get_model_display_name,
    )
    model_id = normalize_model_id(model_name)
            
    st.session_state.last_model_group = model_group
    st.session_state.last_model_algorithm = model_name
    st.session_state.random_seed = int(
        st.number_input(
            t("model_params.random_seed_label"),
            min_value=0,
            max_value=2_147_483_647,
            value=int(st.session_state.get("random_seed", 42)),
            step=1,
            key="model_random_seed_input",
        )
    )

    is_online_mode = bool(globals().get("qspr_is_online_mode", lambda: False)())
    online_allowed_models = {
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "elastic_net",
        "random_forest",
        "svr",
    }
    online_model_locked = is_online_mode and model_id not in online_allowed_models
    if online_model_locked:
        st.info(
            t("model_params.online_model_locked_info")
        )
    
    if model_group == MODEL_GROUP_KERNEL_SIMILARITY:
        show_markdown_help(
            t('model_selection.kernel_help_title'),
            os.path.join(HELP_DIR, "model_group_kernel_help.md"),
            expanded=False
        )
    
    if model_group == MODEL_GROUP_SYMBOLIC:
        show_markdown_help(
            t('model_selection.genetic_help_title'),
            os.path.join(HELP_DIR, "model_group_genetic_help.md"),
            expanded=False
        )
    
    if model_group == MODEL_GROUP_NEURAL:
        show_markdown_help(
            t('model_selection.neural_help_title'),
            os.path.join(HELP_DIR, "model_group_neural_help.md"),
            expanded=False
        )
            
    # Model params UI
    if model_id == "pls_regression":
        max_pls = max(1, min(20, X_all_current.shape[1], len(y_all_current) - 1))
        st.session_state.pls_components = st.slider(
            t('model_params.pls_components_label'),
            min_value=1,
            max_value=max_pls,
            value=min(st.session_state.pls_components, max_pls),
            key="pls_components_slider"
        )
    
    elif model_id == "ridge_regression":
        st.session_state.ridge_alpha = st.number_input(
            t('model_params.ridge_alpha_label'),
            min_value=0.000001,
            value=float(st.session_state.ridge_alpha),
            step=0.1,
            format="%.6f",
            key="ridge_alpha_input"
        )
    
    elif model_id == "lasso_regression":
        st.session_state.lasso_alpha = st.number_input(
            t('model_params.lasso_alpha_label'),
            min_value=0.000001,
            value=float(st.session_state.lasso_alpha),
            step=0.01,
            format="%.6f",
            key="lasso_alpha_input"
        )
    
    elif model_id == "elastic_net":
        st.session_state.elastic_alpha = st.number_input(
            t('model_params.elastic_alpha_label'),
            min_value=0.000001,
            value=float(st.session_state.elastic_alpha),
            step=0.01,
            format="%.6f",
            key="elastic_alpha_input"
        )
        st.session_state.elastic_l1_ratio = st.slider(
            t('model_params.elastic_l1_ratio_label'),
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.elastic_l1_ratio),
            step=0.05,
            key="elastic_l1_ratio_slider"
        )
    
    elif model_id == "mlp_regression":
        st.session_state.mlp_hidden_layer_sizes = st.text_input(
            t('model_params.mlp_hidden_layers_label'),
            value=str(st.session_state.mlp_hidden_layer_sizes),
            key="mlp_hidden_layer_sizes_input",
            help=t('model_params.mlp_hidden_layers_help')
        )
    
        st.session_state.mlp_activation = st.selectbox(
            t('model_params.mlp_activation_label'),
            ["relu", "tanh", "logistic"],
            index=["relu", "tanh", "logistic"].index(st.session_state.mlp_activation)
            if st.session_state.mlp_activation in ["relu", "tanh", "logistic"] else 0,
            key="mlp_activation_select"
        )
    
        st.session_state.mlp_alpha = st.number_input(
            t('model_params.mlp_alpha_label'),
            min_value=0.000000001,
            value=float(st.session_state.mlp_alpha),
            step=0.0001,
            format="%.9f",
            key="mlp_alpha_input"
        )
    
        st.session_state.mlp_learning_rate_init = st.number_input(
            t('model_params.mlp_learning_rate_label'),
            min_value=0.0000001,
            value=float(st.session_state.mlp_learning_rate_init),
            step=0.0001,
            format="%.7f",
            key="mlp_learning_rate_init_input"
        )
    
        st.session_state.mlp_max_iter = st.slider(
            t('model_params.mlp_max_iter_label'),
            min_value=200,
            max_value=10000,
            value=int(st.session_state.mlp_max_iter),
            step=100,
            key="mlp_max_iter_slider"
        )
    
        st.warning(t('model_params.mlp_warning'))
    
    elif model_id == "lightgbm":
        st.session_state.lightgbm_n_estimators = st.slider(
            t('model_params.lightgbm_estimators_label'),
            50, 1000,
            int(st.session_state.lightgbm_n_estimators),
            50,
            key="lightgbm_n_estimators_slider"
        )
        st.session_state.lightgbm_learning_rate = st.number_input(
            t('model_params.lightgbm_learning_rate_label'),
            min_value=0.001,
            max_value=1.0,
            value=float(st.session_state.lightgbm_learning_rate),
            step=0.01,
            format="%.3f",
            key="lightgbm_learning_rate_input"
        )
        st.session_state.lightgbm_num_leaves = st.slider(
            t('model_params.lightgbm_num_leaves_label'),
            4, 128,
            int(st.session_state.lightgbm_num_leaves),
            1,
            key="lightgbm_num_leaves_slider"
        )
    
    elif model_id == "svr":
        st.session_state.svr_c = st.number_input(
            t('model_params.svr_c_label'),
            min_value=0.000001,
            value=float(st.session_state.svr_c),
            step=1.0,
            format="%.6f",
            key="svr_c_input"
        )
    
        st.session_state.svr_epsilon = st.number_input(
            t('model_params.svr_epsilon_label'),
            min_value=0.000001,
            value=float(st.session_state.svr_epsilon),
            step=0.01,
            format="%.6f",
            key="svr_epsilon_input"
        )
    
        st.session_state.svr_gamma = st.selectbox(
            t('model_params.svr_gamma_label'),
            ["scale", "auto"],
            index=0 if st.session_state.svr_gamma == "scale" else 1,
            key="svr_gamma_select"
        )
    
        st.info(t('model_params.svr_info'))
    
    elif model_id == "gpr":
        st.session_state.gpr_alpha = st.number_input(
            t('model_params.gpr_alpha_label'),
            min_value=0.000000001,
            value=float(st.session_state.gpr_alpha),
            step=0.000001,
            format="%.9f",
            key="gpr_alpha_input"
        )
    
        st.session_state.gpr_length_scale = st.number_input(
            t('model_params.gpr_length_scale_label'),
            min_value=0.000001,
            value=float(st.session_state.gpr_length_scale),
            step=0.1,
            format="%.6f",
            key="gpr_length_scale_input"
        )
    
        st.session_state.gpr_noise_level = st.number_input(
            t('model_params.gpr_noise_level_label'),
            min_value=0.000001,
            value=float(st.session_state.gpr_noise_level),
            step=0.01,
            format="%.6f",
            key="gpr_noise_level_input"
        )
    
        st.info(t('model_params.gpr_info'))
    
    elif model_id == "knn_regression":
        max_knn = max(1, min(30, len(y_all_current)))
    
        st.session_state.knn_n_neighbors = st.slider(
            t('model_params.knn_n_neighbors_label'),
            min_value=1,
            max_value=max_knn,
            value=min(int(st.session_state.knn_n_neighbors), max_knn),
            step=1,
            key="knn_n_neighbors_slider"
        )
    
        st.session_state.knn_weights = st.selectbox(
            t('model_params.knn_weights_label'),
            ["distance", "uniform"],
            index=0 if st.session_state.knn_weights == "distance" else 1,
            key="knn_weights_select"
        )
    
        st.info(t('model_params.knn_info'))
    
    elif model_id == "catboost":
        st.session_state.catboost_iterations = st.slider(
            t('model_params.catboost_iterations_label'),
            50, 1000,
            int(st.session_state.catboost_iterations),
            50,
            key="catboost_iterations_slider"
        )
        st.session_state.catboost_learning_rate = st.number_input(
            t('model_params.catboost_learning_rate_label'),
            min_value=0.001,
            max_value=1.0,
            value=float(st.session_state.catboost_learning_rate),
            step=0.01,
            format="%.3f",
            key="catboost_learning_rate_input"
        )
        st.session_state.catboost_depth = st.slider(
            t('model_params.catboost_depth_label'),
            2, 12,
            int(st.session_state.catboost_depth),
            1,
            key="catboost_depth_slider"
        )
    
    elif model_id == "adaboost":
        st.session_state.adaboost_n_estimators = st.slider(
            t('model_params.adaboost_estimators_label'),
            min_value=10,
            max_value=1000,
            value=int(st.session_state.get("adaboost_n_estimators", 300)),
            step=10,
            key="adaboost_n_estimators_slider"
        )
        st.session_state.adaboost_learning_rate = st.number_input(
            t('model_params.adaboost_learning_rate_label'),
            min_value=0.001,
            max_value=2.0,
            value=float(st.session_state.get("adaboost_learning_rate", 1.0)),
            step=0.01,
            format="%.3f",
            key="adaboost_learning_rate_input"
        )
        st.info(t('model_params.adaboost_info'))
    
    elif model_id == "stacking_regressor":
        st.session_state.stacking_cv = st.slider(
            t('model_params.stacking_cv_label'),
            3, 10,
            int(st.session_state.stacking_cv),
            1,
            key="stacking_cv_slider"
        )
        st.session_state.stacking_passthrough = st.checkbox(
            t('model_params.stacking_passthrough_label'),
            value=bool(st.session_state.stacking_passthrough),
            key="stacking_passthrough_checkbox"
        )
        st.info(t('model_params.stacking_info'))
        st.warning(
            t("model_params.stacking_method_warning")
        )
    
    elif model_id == "voting_regressor":
        st.caption(t("model_params.voting_weights_caption"))
        col_vote_1, col_vote_2, col_vote_3 = st.columns(3)
        with col_vote_1:
            st.session_state.voting_rf_weight = st.number_input(
                "Random Forest",
                min_value=0.0,
                value=float(st.session_state.voting_rf_weight),
                step=0.1,
                key="voting_rf_weight_input"
            )
        with col_vote_2:
            st.session_state.voting_extra_trees_weight = st.number_input(
                "Extra Trees",
                min_value=0.0,
                value=float(st.session_state.voting_extra_trees_weight),
                step=0.1,
                key="voting_extra_trees_weight_input"
            )
        with col_vote_3:
            st.session_state.voting_ridge_weight = st.number_input(
                "Ridge",
                min_value=0.0,
                value=float(st.session_state.voting_ridge_weight),
                step=0.1,
                key="voting_ridge_weight_input"
            )
    
        if (
            st.session_state.voting_rf_weight
            + st.session_state.voting_extra_trees_weight
            + st.session_state.voting_ridge_weight
        ) <= 0:
            st.error(t("model_params.voting_weight_error"))
        else:
            st.info(t("model_params.voting_info"))
    
    elif model_id == "cart_regression":
        st.session_state.cart_max_depth = st.slider(
            t('model_params.cart_max_depth_label'),
            min_value=1,
            max_value=30,
            value=int(st.session_state.cart_max_depth),
            step=1,
            key="cart_max_depth_slider"
        )
    
        st.session_state.cart_min_samples_leaf = st.slider(
            t('model_params.cart_min_samples_leaf_label'),
            min_value=1,
            max_value=20,
            value=int(st.session_state.cart_min_samples_leaf),
            step=1,
            key="cart_min_samples_leaf_slider"
        )
    
        st.info(t('model_params.cart_info'))
    
    elif model_id == "mars_like":
        st.session_state.mars_degree = st.slider(
            t('model_params.mars_degree_label'),
            min_value=1,
            max_value=3,
            value=int(st.session_state.mars_degree),
            step=1,
            key="mars_degree_slider"
        )
    
        st.session_state.mars_alpha = st.number_input(
            t('model_params.mars_alpha_label'),
            min_value=0.000001,
            value=float(st.session_state.mars_alpha),
            step=0.1,
            format="%.6f",
            key="mars_alpha_input"
        )
    
        st.info(t('model_params.mars_info'))
    
    elif model_id == "spline_regression":
        st.session_state.spline_n_knots = st.slider(
            t("model_params.spline_n_knots_label"),
            min_value=3,
            max_value=12,
            value=int(st.session_state.spline_n_knots),
            step=1,
            key="spline_n_knots_slider"
        )
        st.session_state.spline_degree = st.slider(
            t("model_params.spline_degree_label"),
            min_value=1,
            max_value=5,
            value=int(st.session_state.spline_degree),
            step=1,
            key="spline_degree_slider"
        )
        st.session_state.spline_alpha = st.number_input(
            t("model_params.spline_alpha_label"),
            min_value=0.000001,
            value=float(st.session_state.spline_alpha),
            step=0.1,
            format="%.6f",
            key="spline_alpha_input"
        )
        st.info(t("model_params.spline_info"))
    
    elif model_id == "gam_regression":
        st.session_state.gam_n_splines = st.slider(
            t("model_params.gam_n_splines_label"),
            min_value=3,
            max_value=15,
            value=int(st.session_state.gam_n_splines),
            step=1,
            key="gam_n_splines_slider"
        )
        st.session_state.gam_degree = st.slider(
            t("model_params.gam_degree_label"),
            min_value=1,
            max_value=5,
            value=int(st.session_state.gam_degree),
            step=1,
            key="gam_degree_slider"
        )
        st.session_state.gam_alpha = st.number_input(
            t("model_params.gam_alpha_label"),
            min_value=0.000001,
            value=float(st.session_state.gam_alpha),
            step=0.1,
            format="%.6f",
            key="gam_alpha_input"
        )
        st.info(t("model_params.gam_info"))
    
    elif model_id == "gep_symbolic":
        st.session_state.gep_population_size = st.slider(
            t('model_params.gep_population_size_label'),
            min_value=100,
            max_value=5000,
            value=int(st.session_state.gep_population_size),
            step=100,
            key="gep_population_size_slider"
        )
    
        st.session_state.gep_generations = st.slider(
            t('model_params.gep_generations_label'),
            min_value=5,
            max_value=200,
            value=int(st.session_state.gep_generations),
            step=5,
            key="gep_generations_slider"
        )
    
        st.session_state.gep_max_depth = st.slider(
            t('model_params.gep_max_depth_label'),
            min_value=1,
            max_value=6,
            value=int(st.session_state.get("gep_max_depth", 4)),
            step=1,
            key="gep_max_depth_slider"
        )
    
    elif model_id == "genetic_programming":
        st.session_state.gp_population_size = st.slider(
            t("model_params.gp_population_size_label"),
            min_value=100,
            max_value=5000,
            value=int(st.session_state.gp_population_size),
            step=100,
            key="gp_population_size_slider"
        )
        st.session_state.gp_generations = st.slider(
            t("model_params.gp_generations_label"),
            min_value=5,
            max_value=200,
            value=int(st.session_state.gp_generations),
            step=5,
            key="gp_generations_slider"
        )
        st.session_state.gp_max_depth = st.slider(
            t("model_params.gp_max_depth_label"),
            min_value=1,
            max_value=8,
            value=int(st.session_state.gp_max_depth),
            step=1,
            key="gp_max_depth_slider"
        )
        st.info(t("model_params.gp_info"))
    
    elif model_id == "pysr":
        st.session_state.pysr_niterations = st.slider(
            t("model_params.pysr_niterations_label"),
            min_value=10,
            max_value=500,
            value=int(st.session_state.pysr_niterations),
            step=10,
            key="pysr_niterations_slider"
        )
        st.session_state.pysr_populations = st.slider(
            t("model_params.pysr_populations_label"),
            min_value=1,
            max_value=32,
            value=int(st.session_state.pysr_populations),
            step=1,
            key="pysr_populations_slider"
        )
        st.session_state.pysr_maxsize = st.slider(
            t("model_params.pysr_maxsize_label"),
            min_value=5,
            max_value=50,
            value=int(st.session_state.pysr_maxsize),
            step=1,
            key="pysr_maxsize_slider"
        )
        st.warning(t("model_params.pysr_warning"))
    
    elif model_id == "extra_trees":
        st.session_state.et_n_estimators = st.slider(
            t('model_params.et_estimators_label'),
            min_value=50,
            max_value=1000,
            value=st.session_state.get("et_n_estimators", 300),
            step=50,
            key="et_n_estimators_slider"
        )
    
        # max_depth: None означает без ограничений
        et_max_depth_val = st.number_input(
            t('model_params.et_max_depth_label'),
            min_value=0,
            max_value=100,
            value=0 if st.session_state.get("et_max_depth") is None else st.session_state.get("et_max_depth", 0),
            step=1,
            key="et_max_depth_input"
        )
        st.session_state.et_max_depth = None if et_max_depth_val == 0 else int(et_max_depth_val)
    
        st.session_state.et_min_samples_split = st.slider(
            t('model_params.et_min_samples_split_label'),
            min_value=2,
            max_value=20,
            value=st.session_state.get("et_min_samples_split", 2),
            step=1,
            key="et_min_samples_split_slider"
        )
    
        st.session_state.et_min_samples_leaf = st.slider(
            t('model_params.et_min_samples_leaf_label'),
            min_value=1,
            max_value=20,
            value=st.session_state.get("et_min_samples_leaf", 1),
            step=1,
            key="et_min_samples_leaf_slider"
        )
    
        st.session_state.et_max_features = st.selectbox(
            t('model_params.et_max_features_label'),
            options=["sqrt", "log2", "auto", None],
            index=0 if st.session_state.get("et_max_features", "sqrt") == "sqrt" else 1,
            key="et_max_features_select"
        )
    
        st.info(t('model_params.et_info'))
    
    elif model_id == "hist_gradient_boosting":
        st.session_state.hgb_max_iter = st.slider(
            t('model_params.hgb_max_iter_label'),
            min_value=50,
            max_value=2000,
            value=int(st.session_state.hgb_max_iter),
            step=50,
            key="hgb_max_iter_slider"
        )
        st.session_state.hgb_learning_rate = st.number_input(
            t('model_params.hgb_learning_rate_label'),
            min_value=0.001,
            max_value=1.0,
            value=float(st.session_state.hgb_learning_rate),
            step=0.01,
            format="%.3f",
            key="hgb_learning_rate_input"
        )
        hgb_max_depth_val = st.number_input(
            t('model_params.hgb_max_depth_label'),
            min_value=0,
            max_value=100,
            value=0 if st.session_state.hgb_max_depth is None else st.session_state.hgb_max_depth,
            step=1,
            key="hgb_max_depth_input"
        )
        st.session_state.hgb_max_depth = None if hgb_max_depth_val == 0 else int(hgb_max_depth_val)
        st.session_state.hgb_min_samples_leaf = st.slider(
            t('model_params.hgb_min_samples_leaf_label'),
            min_value=1,
            max_value=100,
            value=int(st.session_state.hgb_min_samples_leaf),
            step=1,
            key="hgb_min_samples_leaf_slider"
        )
        st.session_state.hgb_l2_regularization = st.number_input(
            t('model_params.hgb_l2_regularization_label'),
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.hgb_l2_regularization),
            step=0.1,
            format="%.1f",
            key="hgb_l2_regularization_input"
        )
        st.info(t('model_params.hgb_info'))
    
    desc_names_current = st.session_state.get("desc_names", [])
    
    if desc_names_current is None:
        desc_names_current = []
    
        st.info(t('model_params.gep_info_repeated'))
    
    # ------------------------------------------------------------
    # Энциклопедия модели
    
    model_help = qspr_get_model_help(model_name)
    
    if model_help:
    
        st.markdown(t('model_encyclopedia.title'))
    
        with st.expander(
            f"ℹ {model_help.get('full_name', model_name)}",
            expanded=False
        ):
    
            st.markdown(f"### {model_help.get('full_name', model_name)}")
    
            st.markdown(
                t('model_encyclopedia.type') + f" {model_help.get('type','—')}"
            )
    
            st.markdown(
                t('model_encyclopedia.plain_language') + f"\n\n{model_help.get('plain_language','—')}"
            )
    
            st.markdown(
                t('model_encyclopedia.main_idea') + f"\n\n{model_help.get('main_idea','—')}"
            )
    
            st.markdown(
                t('model_encyclopedia.example') + f"\n\n{model_help.get('molecule_example','—')}"
            )
    
            st.markdown(
                t('model_encyclopedia.dependency') + f"\n\n{model_help.get('dependency_type','—')}"
            )
    
            st.markdown(t('model_encyclopedia.dataset_size'))
    
            st.write(
                t('model_encyclopedia.minimum'),
                model_help.get("minimum_dataset", "—")
            )
    
            st.write(
                t('model_encyclopedia.recommended'),
                model_help.get("recommended_dataset", "—")
            )
    
            st.write(
                t('model_encyclopedia.comfortable'),
                model_help.get("comfortable_dataset", "—")
            )
    
            st.markdown(t('model_encyclopedia.strengths'))
    
            for item in model_help.get("strengths", []):
                st.write("✅", item)
    
            st.markdown(t('model_encyclopedia.limitations'))
    
            for item in model_help.get("limitations", []):
                st.write("⚠️", item)
    
            st.markdown(t('model_encyclopedia.mistakes'))
    
            for item in model_help.get("typical_mistakes", []):
                st.write("❌", item)
    
            st.markdown(t('model_encyclopedia.when_use'))
    
            for item in model_help.get("when_use", []):
                st.write("✔", item)
    
            st.markdown(t('model_encyclopedia.when_not_use'))
    
            for item in model_help.get("when_not_use", []):
                st.write("✖", item)
    
            st.markdown(
                t('model_encyclopedia.interpretability')
                + " " + "★" * int(model_help.get("interpretability", 0))
            )
    
            st.markdown(
                t('model_encyclopedia.speed')
                + " " + "★" * int(model_help.get("speed", 0))
            )
    
            st.markdown(
                t('model_encyclopedia.complexity')
                + " " + "★" * int(model_help.get("complexity", 0))
            )
    
            tasks = model_help.get("typical_qspr_tasks", [])
            if tasks:
                st.markdown(t('model_encyclopedia.typical_tasks'))
                st.write(", ".join(tasks))
    
            # Гиперпараметры модели
            hyperparams = model_help.get("hyperparameters", [])
            if hyperparams:
                st.markdown(t('model_encyclopedia.hyperparams_title'))
                st.markdown(t('model_encyclopedia.hyperparams_desc'))
                for hp in hyperparams:
                    st.markdown(f"**{hp.get('name', '')}**")
                    st.markdown(f"*{hp.get('description', '')}*")
                    st.markdown(f"- **{t('model_encyclopedia.hp_effect')}:** {hp.get('effect', '')}")
                    st.markdown(f"- **{t('model_encyclopedia.hp_recommendation')}:** {hp.get('recommendation', '')}")
                    st.markdown("---")
    
    desc_names_current = st.session_state.get("desc_names", [])
    
    if desc_names_current is None:
        desc_names_current = []
        st.info(t('model_params.gep_info'))
    
    
    st.markdown(
        f'<div class="tool-badge">{t("auto_select.tool_badge")}</div>',
        unsafe_allow_html=True
    )
    with st.expander(t('auto_select.expander_title'), expanded=False):
        st.session_state.auto_feature_selection = st.checkbox(
            t('auto_select.enable_selection'),
            value=bool(st.session_state.get("auto_feature_selection", False)),
            key="auto_feature_selection_checkbox"
        )
    
        st.session_state.auto_hyperparameter_optimization = st.checkbox(
            t('auto_select.enable_optimization'),
            value=bool(st.session_state.get("auto_hyperparameter_optimization", False)),
            key="auto_hyperparameter_optimization_checkbox"
        )
    
        st.markdown(t('auto_select.mode_title'))
    
        # Переведённые названия методов
        feature_selection_label_to_value = {
            t('auto_select.method_none'): "none",
            t('auto_select.method_fast'): "fast",
            t('auto_select.method_f_regression'): "f_regression",
            t('auto_select.method_mutual_info'): "mutual_info",
            t('auto_select.method_lasso'): "lasso",
            t('auto_select.method_rf'): "random_forest",
            t('auto_select.method_rfe'): "rfe_ridge",
        }
    
        current_feature_selection_value = st.session_state.get(
            "auto_feature_selection_method",
            "fast"
        )
    
        label_values = list(feature_selection_label_to_value.values())
        label_names = list(feature_selection_label_to_value.keys())
    
        if current_feature_selection_value in label_values:
            current_feature_selection_index = label_values.index(current_feature_selection_value)
        else:
            current_feature_selection_index = label_values.index("fast")
    
        selected_feature_selection_label = st.selectbox(
            t('auto_select.method_label'),
            label_names,
            index=current_feature_selection_index,
            key="auto_feature_selection_method_select_label"
        )
    
        st.session_state.auto_feature_selection_method = feature_selection_label_to_value[
            selected_feature_selection_label
        ]
    
        max_possible_features = max(1, len(desc_names_current))
    
        st.session_state.auto_max_features = st.slider(
            t('auto_select.max_features_label'),
            min_value=1,
            max_value=max_possible_features,
            value=min(int(st.session_state.get("auto_max_features", 50)), max_possible_features),
            step=1,
            key="auto_max_features_slider"
        )
    
        st.markdown(t('auto_select.cleaning_title'))
    
        col_auto_clean_1, col_auto_clean_2, col_auto_clean_3 = st.columns(3)
    
        with col_auto_clean_1:
            st.session_state.auto_remove_constant_descriptors = st.checkbox(
                t('auto_select.remove_constant'),
                value=bool(st.session_state.get("auto_remove_constant_descriptors", True)),
                key="auto_remove_constant_descriptors_checkbox"
            )
    
        with col_auto_clean_2:
            st.session_state.auto_remove_correlated_descriptors = st.checkbox(
                t('auto_select.remove_correlated'),
                value=bool(st.session_state.get("auto_remove_correlated_descriptors", True)),
                key="auto_remove_correlated_descriptors_checkbox"
            )
    
        with col_auto_clean_3:
            st.session_state.auto_corr_threshold = st.slider(
                t('auto_select.corr_threshold_label'),
                min_value=0.70,
                max_value=0.999,
                value=float(st.session_state.get("auto_corr_threshold", 0.95)),
                step=0.01,
                format="%.3f",
                key="auto_corr_threshold_slider"
            )
    
        st.caption(t('auto_select.corr_caption'))
    
        st.markdown(t('auto_select.advanced_title'))
    
        col_auto_adv_1, col_auto_adv_2, col_auto_adv_3 = st.columns(3)
    
        with col_auto_adv_1:
            st.session_state.auto_lasso_selection_alpha = st.number_input(
                t('auto_select.lasso_alpha_label'),
                min_value=0.000001,
                value=float(st.session_state.get("auto_lasso_selection_alpha", 0.01)),
                step=0.01,
                format="%.6f",
                key="auto_lasso_selection_alpha_input"
            )
    
        with col_auto_adv_2:
            st.session_state.auto_rf_selection_estimators = st.slider(
                t('auto_select.rf_estimators_label'),
                min_value=50,
                max_value=1000,
                value=int(st.session_state.get("auto_rf_selection_estimators", 300)),
                step=50,
                key="auto_rf_selection_estimators_slider"
            )
    
        with col_auto_adv_3:
            st.session_state.auto_rfe_step = st.slider(
                t('auto_select.rfe_step_label'),
                min_value=0.05,
                max_value=0.50,
                value=float(st.session_state.get("auto_rfe_step", 0.2)),
                step=0.05,
                format="%.2f",
                key="auto_rfe_step_slider"
            )
    
        st.markdown(t('auto_select.cv_title'))
    
        st.session_state.auto_cv = st.slider(
            t('auto_select.cv_slider_label'),
            min_value=2,
            max_value=max(2, min(10, len(y_all_current))),
            value=min(int(st.session_state.get("auto_cv", 5)), max(2, min(10, len(y_all_current)))),
            step=1,
            key="auto_cv_slider"
        )
    
        st.session_state.auto_search_method = st.radio(
            t('auto_select.search_method_label'),
            ["grid", "random"],
            index=0 if st.session_state.get("auto_search_method", "grid") == "grid" else 1,
            horizontal=True,
            key="auto_search_method_radio"
        )
    
        show_markdown_help(
            t('auto_select.help_title'),
            os.path.join(HELP_DIR, "auto_feature_selection_help.md"),
            expanded=False
        )

    current_params_for_guidance = get_model_params_from_session()
    current_guidance = qspr_model_applicability_guidance(
        model_name,
        n_samples=len(y_all_current),
        n_features=X_all_current.shape[1],
        params=current_params_for_guidance,
        online_mode=is_online_mode,
    )
    model_resource_blocked = False
    for guidance in current_guidance:
        message = guidance.get("message", "")
        if guidance.get("topic") == "GPR applicability":
            estimated = guidance.get("estimated_fit_time_seconds")
            if estimated is not None and np.isfinite(float(estimated)):
                message = f"{message} Estimated fit time: {float(estimated):.1f} s."
        if guidance.get("topic") == "MLP data-to-parameter ratio":
            message = (
                f"{message} N={guidance.get('n_samples')}, "
                f"parameters~{guidance.get('n_parameters_estimated')}, "
                f"N/parameters={float(guidance.get('samples_per_parameter', 0.0)):.4f}."
            )
        if guidance.get("level") == "error":
            model_resource_blocked = True
            st.error(message)
        elif guidance.get("level") == "warning":
            st.warning(message)
        else:
            st.info(message)

    stochastic_models_for_seed_stability = {
        "random_forest",
        "extra_trees",
        "adaboost",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp_regression",
        "gep_symbolic",
        "genetic_programming",
    }
    if model_id in stochastic_models_for_seed_stability:
        with st.expander(t("model_params.seed_stability_title"), expanded=False):
            st.caption(
                t("model_params.seed_stability_caption")
            )
            seed_test_percent = st.slider(
                t("model_params.seed_stability_test_percent"),
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                key="seed_stability_test_percent",
            )
            if st.button(t("model_params.seed_stability_run"), key="run_seed_stability"):
                try:
                    smiles_for_seed = data[smiles_col_current].iloc[
                        valid_indices_current
                    ].values.tolist()
                    seed_result = qspr_seed_stability_holdout(
                        X=X_all_current,
                        y=y_all_current,
                        model_name=model_name,
                        seeds=[1, 7, 42, 101, 2026],
                        valid_indices=valid_indices_current,
                        smiles=smiles_for_seed,
                        test_size=float(seed_test_percent) / 100.0,
                        params=current_params_for_guidance,
                        scale=True,
                    )
                    st.session_state.seed_stability_result = seed_result
                except Exception as e:
                    st.error(t("model_params.seed_stability_failed", error=e))
            seed_result = st.session_state.get("seed_stability_result")
            if isinstance(seed_result, dict):
                st.dataframe(
                    seed_result.get("results_df", pd.DataFrame()),
                    width="stretch",
                    hide_index=True,
                )
                st.json(seed_result.get("summary", {}))
    
    # Training
    train_clicked = st.button(
        t('model_training.train_button'),
        type="primary",
        disabled=online_model_locked or model_resource_blocked,
    )
    
    if train_clicked:
        try:
            use_auto = (
                st.session_state.get("auto_feature_selection", False) or
                st.session_state.get("auto_hyperparameter_optimization", False)
            )
    
            if use_auto:
                with st.spinner(t('model_training.auto_spinner', model=model_name)):
                    if st.session_state.get("auto_feature_selection", False):
                        auto_method = st.session_state.get("auto_feature_selection_method", "fast")
                        auto_max_features = st.session_state.get("auto_max_features", 50)
                        auto_remove_constant = st.session_state.get("auto_remove_constant_descriptors", True)
                        auto_remove_correlated = st.session_state.get("auto_remove_correlated_descriptors", True)
                    else:
                        auto_method = "none"
                        auto_max_features = len(desc_names_current)
                        auto_remove_constant = False
                        auto_remove_correlated = False
    
                    auto_result = qspr_auto_select_and_tune(
                        X=X_all_current,
                        y=y_all_current,
                        desc_names=desc_names_current,
                        model_name=model_name,
                        params=get_model_params_from_session(),
                        scale=True,
                        feature_selection_method=auto_method,
                        max_features=auto_max_features,
                        optimize_hyperparams=st.session_state.get("auto_hyperparameter_optimization", False),
                        cv=st.session_state.get("auto_cv", 5),
                        search_method=st.session_state.get("auto_search_method", "grid"),
                        remove_constant=auto_remove_constant,
                        remove_correlated=auto_remove_correlated,
                        corr_threshold=st.session_state.get("auto_corr_threshold", 0.95),
                        lasso_selection_alpha=st.session_state.get("auto_lasso_selection_alpha", 0.01),
                        rf_selection_estimators=st.session_state.get("auto_rf_selection_estimators", 300),
                        rfe_step=st.session_state.get("auto_rfe_step", 0.2)
                    )
    
                    X_model_space = auto_result.get("X_model_space", X_all_current)
                    y_pred_all = np.ravel(
                        auto_result["model"].predict(X_all_current)
                    )
    
                    result_train = {
                        "model": auto_result["model"],
                        "scaler": None,
                        "X_scaled": X_model_space,
                        "X_original": X_all_current,
                        "y_pred": y_pred_all,
                        "metrics": qspr_metrics(y_all_current, y_pred_all),
                        "auto_result": auto_result,
                        "selected_desc_names": auto_result["selected_desc_names"],
                    }
    
                    st.session_state.model_used_descriptor_names = list(
                        auto_result["selected_desc_names"]
                    )
                    st.session_state.model_used_descriptor_model_name = model_name
    
                    train_hash = analysis_result_hash(
                        st.session_state,
                        model_name,
                        params=get_model_params_from_session(),
                        validation_settings={
                            "kind": "train",
                            "auto_feature_selection": True,
                            "auto_hyperparameter_optimization": st.session_state.get(
                                "auto_hyperparameter_optimization",
                                False,
                            ),
                        },
                        X=X_all_current,
                        y=y_all_current,
                        desc_names=desc_names_current,
                        valid_indices=valid_indices_current,
                    )
                    st.session_state.trained_models[model_name] = attach_result_cache_metadata(
                        result_train,
                        train_hash,
                    )
                    st.session_state.pop(
                        f"descriptor_importance_result_{model_name}",
                        None
                    )
                    st.session_state.pop(
                        f"descriptor_shap_result_{model_name}",
                        None
                    )
                    st.session_state.pop(
                        f"descriptor_importance_unified_{model_name}",
                        None
                    )
                    st.session_state.pop(
                        f"error_analysis_result_{model_name}",
                        None
                    )
                    st.session_state.holdout_results_dict.pop(model_name, None)
                    st.session_state.kfold_results_dict.pop(model_name, None)
                    st.session_state.loo_results_dict.pop(model_name, None)
                    st.session_state.auto_tuning_result = auto_result
    
                    add_event_log(
                        "FEATURE_SELECTION",
                        (
                            f"{model_name} auto-improvement: исходно {len(desc_names_current)} признаков, "
                            f"выбрано {len(auto_result['selected_desc_names'])}."
                        ),
                        level="info",
                        details={
                            "model": model_name,
                            "initial_features": len(desc_names_current),
                            "selected_features": len(auto_result["selected_desc_names"]),
                        },
                        event="auto_feature_selection_completed",
                    )
    
                    st.rerun()
    
            else:
                with st.spinner(t('model_training.train_spinner', model=model_name)):
                    result_train = qspr_train_analysis_model(
                        X_all_current,
                        y_all_current,
                        model_name,
                        params=get_model_params_from_session(),
                        scale=True
                    )
                    result_train["selected_desc_names"] = list(desc_names_current)
    
                    st.session_state.model_used_descriptor_names = list(desc_names_current)
                    st.session_state.model_used_descriptor_model_name = model_name
    
                    train_hash = analysis_result_hash(
                        st.session_state,
                        model_name,
                        params=get_model_params_from_session(),
                        validation_settings={"kind": "train"},
                        X=X_all_current,
                        y=y_all_current,
                        desc_names=desc_names_current,
                        valid_indices=valid_indices_current,
                    )
                    st.session_state.trained_models[model_name] = attach_result_cache_metadata(
                        result_train,
                        train_hash,
                    )
                    st.session_state.pop(
                        f"descriptor_importance_result_{model_name}",
                        None
                    )
                    st.session_state.pop(
                        f"descriptor_shap_result_{model_name}",
                        None
                    )
                    st.session_state.pop(
                        f"descriptor_importance_unified_{model_name}",
                        None
                    )
                    st.session_state.pop(
                        f"error_analysis_result_{model_name}",
                        None
                    )
                    st.session_state.holdout_results_dict.pop(model_name, None)
                    st.session_state.kfold_results_dict.pop(model_name, None)
                    st.session_state.loo_results_dict.pop(model_name, None)
                    add_log(t('model_training.log_trained', model=model_name))
                    st.rerun()
    
        except Exception as e:
            st.error(t('model_training.train_error', error=e))
    
    # ------------------------------------------------------------------
    # Display trained model
    if st.session_state.get("model_used_descriptor_names"):
        model_used_names = list(st.session_state.get("model_used_descriptor_names", []))
        model_used_name = st.session_state.get("model_used_descriptor_model_name", "")
    
        qspr_show_descriptor_meaning_table(
            desc_names=model_used_names,
            title=t('model_display.used_descriptors_title', model=model_used_name),
            status_label=t('model_display.used_in_model_status'),
            expanded=False,
            key_prefix="model_used_descriptor_meanings"
        )
    if model_name in st.session_state.trained_models:
        model_data = st.session_state.trained_models[model_name]
        expected_train_hash = analysis_result_hash(
            st.session_state,
            model_name,
            params=get_model_params_from_session(),
            validation_settings={"kind": "train"},
            X=X_all_current,
            y=y_all_current,
            desc_names=desc_names_current,
            valid_indices=valid_indices_current,
        )
        expected_auto_train_hash = analysis_result_hash(
            st.session_state,
            model_name,
            params=get_model_params_from_session(),
            validation_settings={
                "kind": "train",
                "auto_feature_selection": True,
                "auto_hyperparameter_optimization": st.session_state.get(
                    "auto_hyperparameter_optimization",
                    False,
                ),
            },
            X=X_all_current,
            y=y_all_current,
            desc_names=desc_names_current,
            valid_indices=valid_indices_current,
        )
        if (
            isinstance(model_data, dict)
            and not (
                cached_result_is_current(model_data, expected_train_hash)
                or cached_result_is_current(model_data, expected_auto_train_hash)
            )
        ):
            st.warning(
                t("model_params.stale_model_warning")
            )
            st.stop()
        model = model_data["model"]
        scaler = model_data.get("scaler", None)
        X_scaled = model_data.get("X_scaled", X_all_current)
        y_pred_all = np.ravel(model_data["y_pred"])
        core_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        if hasattr(core_model, "get_formula_complexity"):
            complexity = core_model.get_formula_complexity()
            if complexity:
                st.warning(
                    t("model_params.symbolic_formula_warning")
                )
                with st.expander(t("model_params.symbolic_formula_expander"), expanded=False):
                    st.json(complexity)
    
        smiles_current = data[smiles_col_current].iloc[valid_indices_current].values
    
        analysis_table = qspr_prediction_table(
            y_true=y_all_current,
            y_pred=y_pred_all,
            smiles=smiles_current,
            original_indices=valid_indices_current
        )
    
        auto_result_view = {}
        selection_summary_view = {}
        selection_table_view = pd.DataFrame()
    
        if isinstance(model_data, dict) and model_data.get("auto_result") is not None:
            auto_result_view = model_data.get("auto_result", {})
            selection_summary_view = auto_result_view.get("selection_summary", {})
            selection_table_view = auto_result_view.get("selection_table", pd.DataFrame())
    
        # ------------------------------------------------------------
        # Auto feature selection / tuning results
    
        if st.session_state.get("auto_tuning_result") is not None:
            auto_res = st.session_state.auto_tuning_result
    
            st.subheader(t('auto_results.subheader'))
    
            auto_cv_metrics = auto_res.get("cv_metrics", auto_res.get("metrics", {}))
            auto_fit_metrics = auto_res.get("fit_metrics", {})
    
            selected_desc_names_auto = auto_res.get("selected_desc_names", [])
            selection_summary_auto = auto_res.get("selection_summary", {})
            best_params_auto = auto_res.get("best_params", {})
            best_cv_rmse_auto = auto_res.get("best_cv_rmse", np.nan)
    
            col_auto_1, col_auto_2, col_auto_3, col_auto_4 = st.columns(4)
    
            with col_auto_1:
                st.metric(
                    t('auto_results.metric_selected_descriptors'),
                    len(selected_desc_names_auto)
                )
    
            with col_auto_2:
                st.metric(t('auto_results.metric_cv'), auto_res.get("cv", "—"))
    
            with col_auto_3:
                cv_r2_auto = auto_cv_metrics.get("R2", np.nan)
                if pd.notna(cv_r2_auto):
                    st.metric(t('auto_results.metric_cv_r2'), f"{float(cv_r2_auto):.3f}")
                else:
                    st.metric(t('auto_results.metric_cv_r2'), "—")
    
            with col_auto_4:
                cv_rmse_auto = auto_cv_metrics.get("RMSE", np.nan)
                if pd.notna(cv_rmse_auto):
                    st.metric(t('auto_results.metric_cv_rmse'), f"{float(cv_rmse_auto):.3f}")
                else:
                    st.metric(t('auto_results.metric_cv_rmse'), "—")
    
            col_auto_5, col_auto_6, col_auto_7, col_auto_8 = st.columns(4)
    
            with col_auto_5:
                fit_r2_auto = auto_fit_metrics.get("R2", np.nan)
                if pd.notna(fit_r2_auto):
                    st.metric(t('auto_results.metric_train_r2'), f"{float(fit_r2_auto):.3f}")
                else:
                    st.metric(t('auto_results.metric_train_r2'), "—")
    
            with col_auto_6:
                fit_rmse_auto = auto_fit_metrics.get("RMSE", np.nan)
                if pd.notna(fit_rmse_auto):
                    st.metric(t('auto_results.metric_train_rmse'), f"{float(fit_rmse_auto):.3f}")
                else:
                    st.metric(t('auto_results.metric_train_rmse'), "—")
    
            with col_auto_7:
                if pd.notna(best_cv_rmse_auto):
                    st.metric(t('auto_results.metric_best_cv_rmse'), f"{float(best_cv_rmse_auto):.3f}")
                else:
                    st.metric(t('auto_results.metric_best_cv_rmse'), "—")
    
            with col_auto_8:
                st.metric(
                    t('auto_results.metric_method'),
                    str(auto_res.get("feature_selection_method", selection_summary_auto.get("method", "—")))
                )
    
            if auto_res.get("cv_status") == "failed":
                st.warning(
                    t("model_params.cv_failed_model_warning")
                )
                st.caption(
                    f"{auto_res.get('failed_stage', 'CV')}: "
                    f"{auto_res.get('cv_error_type', '')} "
                    f"{auto_res.get('cv_error_message', '')}"
                )
            elif auto_res.get("model_validation_status"):
                st.caption(str(auto_res.get("model_validation_status")))
            target_quality_auto = auto_res.get("target_quality") or {}
            if target_quality_auto.get("warnings"):
                st.warning(
                    "Target property has weak variation: "
                    f"{target_quality_auto.get('n_unique_y')} unique values; "
                    f"dominant value fraction = {target_quality_auto.get('dominant_y_fraction', 0):.2f}."
                )
            feature_ratio_auto = auto_res.get("feature_ratio_diagnostics") or {}
            if feature_ratio_auto.get("warnings"):
                st.warning(
                    "Descriptor count is high relative to the number of objects: "
                    f"n={feature_ratio_auto.get('n')}, p={feature_ratio_auto.get('p')}, "
                    f"residual df={feature_ratio_auto.get('residual_degrees_of_freedom')}."
                )
            st.caption(str(auto_res.get("best_cv_rmse_label", "")))
            st.caption(str(auto_res.get("cv_metrics_label", "")))

            if best_params_auto:
                with st.expander(t('auto_results.best_params_expander'), expanded=False):
                    st.json(best_params_auto)
    
            if selected_desc_names_auto:
                selected_desc_df = pd.DataFrame({
                    t('auto_results.selected_descriptors_column'): selected_desc_names_auto
                })
    
                st.download_button(
                    t('auto_results.download_selected_button'),
                    selected_desc_df.to_csv(index=False).encode("utf-8"),
                    "selected_descriptors.csv",
                    "text/csv",
                    key=f"download_selected_descriptors_auto_tuning_{model_name}"
                )
    
            with st.expander(t('auto_results.report_expander'), expanded=False):
                if selection_summary_view:
                    summary_ru = pd.DataFrame([
                        {t('auto_results.report_prompt'): t('auto_results.report_method'), t('auto_results.report_value'): selection_summary_view.get("method", "")},
                        {t('auto_results.report_prompt'): t('auto_results.report_initial'), t('auto_results.report_value'): selection_summary_view.get("n_features_initial", "")},
                        {t('auto_results.report_prompt'): t('auto_results.report_removed_const'), t('auto_results.report_value'): selection_summary_view.get("n_removed_constant", "")},
                        {t('auto_results.report_prompt'): t('auto_results.report_after_const'), t('auto_results.report_value'): selection_summary_view.get("n_after_constant_filter", "")},
                        {t('auto_results.report_prompt'): t('auto_results.report_removed_corr'), t('auto_results.report_value'): selection_summary_view.get("n_removed_correlated", "")},
                        {t('auto_results.report_prompt'): t('auto_results.report_after_corr'), t('auto_results.report_value'): selection_summary_view.get("n_after_correlation_filter", "")},
                        {t('auto_results.report_prompt'): t('auto_results.report_final'), t('auto_results.report_value'): selection_summary_view.get("n_selected_final", "")},
                        {t('auto_results.report_prompt'): t('auto_results.report_corr_threshold'), t('auto_results.report_value'): selection_summary_view.get("corr_threshold", "")},
                    ])
    
                    st.dataframe(
                        summary_ru,
                        width="stretch",
                        hide_index=True
                    )
    
                cv_metrics_view = auto_result_view.get("cv_metrics", {})
    
                if cv_metrics_view:
                    cv_metrics_df = pd.DataFrame([
                        {t('auto_results.report_metric'): k, t('auto_results.report_metric_value'): v}
                        for k, v in cv_metrics_view.items()
                    ])
    
                    st.markdown(t('auto_results.cv_metrics_title'))
    
                    st.dataframe(
                        cv_metrics_df,
                        width="stretch",
                        hide_index=True
                    )
    
                best_params_view = auto_result_view.get("best_params", {})
    
                if best_params_view:
                    st.markdown(t('auto_results.best_params_title'))
    
                    best_params_df = pd.DataFrame([
                        {t('auto_results.report_param'): k, t('auto_results.report_value'): str(v)}
                        for k, v in best_params_view.items()
                    ])
    
                    st.dataframe(
                        best_params_df,
                        width="stretch",
                        hide_index=True
                    )
    
                selected_desc_view = auto_result_view.get("selected_desc_names", [])
    
                if selected_desc_view:
                    st.markdown(t('auto_results.final_selected_title'))
    
                    selected_desc_df = pd.DataFrame({
                        t('auto_results.report_number'): range(1, len(selected_desc_view) + 1),
                        t('auto_results.report_descriptor'): selected_desc_view
                    })
    
                    st.dataframe(
                        selected_desc_df,
                        width="stretch",
                        hide_index=True
                    )
                    # ----------------------------------------------------
                    # Тепловая карта корреляций первых выбранных дескрипторов
    
                    st.markdown(t('auto_results.heatmap_title'))
    
                    try:
                        heatmap_desc_names = list(selected_desc_view[:15])
    
                        if len(heatmap_desc_names) < 2:
                            st.info(t('auto_results.heatmap_min_descriptors'))
                        else:
                            desc_name_to_index = {
                                str(name): i
                                for i, name in enumerate(desc_names_current)
                            }
    
                            available_heatmap_desc = [
                                name for name in heatmap_desc_names
                                if str(name) in desc_name_to_index
                            ]
    
                            if len(available_heatmap_desc) < 2:
                                st.info(t('auto_results.heatmap_not_found'))
                            else:
                                heatmap_indices = [
                                    desc_name_to_index[str(name)]
                                    for name in available_heatmap_desc
                                ]
    
                                heatmap_values = np.asarray(
                                    X_all_current[:, heatmap_indices],
                                    dtype=float
                                )
    
                                heatmap_df = pd.DataFrame(
                                    heatmap_values,
                                    columns=available_heatmap_desc
                                )
    
                                heatmap_df[target_col] = np.asarray(
                                    y_all_current,
                                    dtype=float
                                )
    
                                heatmap_df = heatmap_df.replace(
                                    [np.inf, -np.inf],
                                    np.nan
                                )
    
                                heatmap_df = heatmap_df.dropna(axis=1, how="all")
    
                                for heat_col in heatmap_df.columns:
                                    if heatmap_df[heat_col].isna().any():
                                        median_val = heatmap_df[heat_col].median()
    
                                        if pd.isna(median_val):
                                            median_val = 0.0
    
                                        heatmap_df[heat_col] = heatmap_df[heat_col].fillna(
                                            median_val
                                        )
    
                                if heatmap_df.shape[1] < 3:
                                    st.info(t('auto_results.heatmap_insufficient_cols'))
                                else:
                                    corr_heatmap_selected = heatmap_df.corr(
                                        numeric_only=True
                                    )
    
                                    fig_auto_heat, ax_auto_heat = plt.subplots(
                                        figsize=(9, 7)
                                    )
    
                                    sns.heatmap(
                                        corr_heatmap_selected,
                                        annot=True,
                                        fmt=".2f",
                                        square=True,
                                        linewidths=0.5,
                                        cbar=True,
                                        ax=ax_auto_heat
                                    )
    
                                    ax_auto_heat.set_title(t('auto_results.heatmap_plot_title', col=target_col))
    
                                    fig_auto_heat.tight_layout()
    
                                    st.pyplot(fig_auto_heat)
                                    plt.close(fig_auto_heat)
    
                                    st.caption(t('auto_results.heatmap_caption'))
    
                                    with st.expander(t('auto_results.heatmap_matrix_expander'), expanded=False):
                                        st.dataframe(
                                            corr_heatmap_selected.round(4),
                                            width="stretch"
                                        )
    
                                        st.download_button(
                                            t('auto_results.heatmap_download_matrix'),
                                            corr_heatmap_selected.to_csv().encode("utf-8"),
                                            f"selected_descriptors_correlation_{model_name}.csv",
                                            "text/csv",
                                            key=f"download_selected_descriptor_corr_auto_report_{model_name}"
                                        )
    
                    except Exception as e:
                        st.warning(t('auto_results.heatmap_error', error=e))
    
                    if isinstance(selection_table_view, pd.DataFrame) and not selection_table_view.empty:
                        st.markdown(t('auto_results.full_selection_table_title'))
    
                        status_filter_options = [t('auto_results.filter_all')] + sorted(
                            selection_table_view["Статус"].dropna().astype(str).unique().tolist()
                        )
    
                        status_filter = st.selectbox(
                            t('auto_results.filter_by_status'),
                            status_filter_options,
                            key=f"auto_selection_status_filter_{model_name}"
                        )
    
                        if status_filter == t('auto_results.filter_all'):
                            selection_table_display = selection_table_view.copy()
                        else:
                            selection_table_display = selection_table_view[
                                selection_table_view["Статус"].astype(str) == status_filter
                            ].copy()
    
                        st.dataframe(
                            selection_table_display.head(1000),
                            width="stretch",
                            hide_index=True
                        )
    
                        csv_selection_report = selection_table_view.to_csv(index=False).encode("utf-8")
    
                        st.download_button(
                            t('auto_results.download_selection_report'),
                            csv_selection_report,
                            f"descriptor_selection_report_{model_name}.csv",
                            "text/csv",
                            key=f"download_descriptor_selection_report_{model_name}"
                        )
    
        st.subheader(t('analysis_results.table_subheader'))
        st.dataframe(analysis_table, width="stretch", hide_index=True)
    
        qspr_save_results_auto(
            analysis_table,
            "analysis",
            target_col,
            len(y_all_current)
        )
    
        col_graf1, col_graf2 = st.columns(2)
    
        with col_graf1:
            fig_anal, ax_anal = plt.subplots(figsize=(4, 4))
    
            ax_anal.scatter(
                y_all_current,
                y_pred_all,
                alpha=0.6,
                s=25
            )
    
            ax_anal.plot(
                [y_all_current.min(), y_all_current.max()],
                [y_all_current.min(), y_all_current.max()],
                "r--",
                lw=2
            )
    
            ax_anal.set_xlabel(t('analysis_results.plot_xlabel'))
            ax_anal.set_ylabel(t('analysis_results.plot_ylabel'))
            ax_anal.set_title(f"{model_name}")
    
            st.pyplot(fig_anal)
    
        with col_graf2:
            errors_anal = y_all_current - y_pred_all
    
            fig_hist_anal, ax_hist_anal = plt.subplots(figsize=(4, 3))
    
            safe_histplot(ax_hist_anal, errors_anal, kde=True, color='coral', edgecolor='black', alpha=0.8)
    
            mu_anal = np.mean(errors_anal)
            std_anal = np.std(errors_anal)
    
            x_anal = np.linspace(
                errors_anal.min(),
                errors_anal.max(),
                100
            )
    
            if std_anal > 1e-12:
                ax_hist_anal.plot(
                    x_anal,
                    norm.pdf(x_anal, mu_anal, std_anal) * len(errors_anal),
                    "r--",
                    label=t('analysis_results.hist_normal_label', mu=mu_anal, sigma=std_anal)
                )
    
            ax_hist_anal.set_title(t('analysis_results.hist_title'))
            ax_hist_anal.legend(fontsize=8)
    
            st.pyplot(fig_hist_anal)
    
        st.subheader(t('analysis_results.metrics_subheader'))
    
        metrics = model_data["metrics"]
    
        metric_table = pd.DataFrame({
            t('analysis_results.metrics_prompt'): list(metrics.keys()),
            t('analysis_results.metrics_value'): list(metrics.values())
        })
    
        st.dataframe(
            metric_table,
            width="stretch",
            hide_index=True
        )
    
        if model_id == "gpr":
            try:
                y_gpr_mean, y_gpr_std = model.predict(X_scaled, return_std=True)
    
                gpr_uncertainty_df = pd.DataFrame({
                    t('analysis_results.gpr_number'): range(1, len(y_all_current) + 1),
                    t('analysis_results.gpr_index'): valid_indices_current,
                    t('analysis_results.gpr_smiles'): smiles_current,
                    t('analysis_results.gpr_experiment'): y_all_current,
                    t('analysis_results.gpr_prediction'): y_gpr_mean,
                    t('analysis_results.gpr_std'): y_gpr_std,
                    t('analysis_results.gpr_lower'): y_gpr_mean - 1.96 * y_gpr_std,
                    t('analysis_results.gpr_upper'): y_gpr_mean + 1.96 * y_gpr_std,
                })
    
                with st.expander(t('analysis_results.gpr_expander'), expanded=False):
                    st.dataframe(
                        gpr_uncertainty_df,
                        width="stretch",
                        hide_index=True
                    )
    
                    csv_gpr = gpr_uncertainty_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        t('analysis_results.gpr_download_button'),
                        csv_gpr,
                        "gpr_uncertainty.csv",
                        "text/csv",
                        key="download_gpr_uncertainty"
                    )
    
            except Exception as e:
                st.warning(t('analysis_results.gpr_error', error=e))
    

    return locals()
