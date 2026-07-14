from collections import OrderedDict


MODEL_GROUP_LINEAR = "Линейные и регуляризованные методы"
MODEL_GROUP_KERNEL_SIMILARITY = "Методы сходства и ядерные методы"
MODEL_GROUP_SPLINE = "Кусочно-линейные и сплайновые методы"
MODEL_GROUP_TREES = "Деревья решений"
MODEL_GROUP_TREE_ENSEMBLES = "Ансамбли деревьев"
MODEL_GROUP_BOOSTING = "Бустинговые методы"
MODEL_GROUP_NEURAL = "Нейросетевые методы"
MODEL_GROUP_META_ENSEMBLES = "Метаансамбли"
MODEL_GROUP_SYMBOLIC = "Символическая и эволюционная регрессия"


MODEL_GROUPS = OrderedDict([
    ("linear", MODEL_GROUP_LINEAR),
    ("kernel_similarity", MODEL_GROUP_KERNEL_SIMILARITY),
    ("spline", MODEL_GROUP_SPLINE),
    ("trees", MODEL_GROUP_TREES),
    ("tree_ensembles", MODEL_GROUP_TREE_ENSEMBLES),
    ("boosting", MODEL_GROUP_BOOSTING),
    ("neural", MODEL_GROUP_NEURAL),
    ("meta_ensembles", MODEL_GROUP_META_ENSEMBLES),
    ("symbolic", MODEL_GROUP_SYMBOLIC),
])


FILTER_KEYS = (
    "small_datasets",
    "interpretation",
    "nonlinear",
    "high_accuracy",
    "beginners",
)


def _entry(
    runtime_name,
    display_name,
    group_id,
    encyclopedia_key,
    aliases=(),
    dependency=None,
    small_datasets=False,
    interpretation=False,
    nonlinear=False,
    high_accuracy=False,
    beginners=False,
):
    return {
        "runtime_name": runtime_name,
        "display_name": display_name,
        "group_id": group_id,
        "group": MODEL_GROUPS[group_id],
        "encyclopedia_key": encyclopedia_key,
        "aliases": tuple(dict.fromkeys((runtime_name, display_name, *aliases))),
        "dependency": dependency,
        "filters": {
            "small_datasets": bool(small_datasets),
            "interpretation": bool(interpretation),
            "nonlinear": bool(nonlinear),
            "high_accuracy": bool(high_accuracy),
            "beginners": bool(beginners),
        },
    }


MODEL_CATALOG = OrderedDict([
    ("linear_regression", _entry(
        "Множественная линейная регрессия (MLR)", "Linear Regression",
        "linear", "Linear Regression",
        aliases=("MLR", "Multiple Linear Regression"),
        small_datasets=True, interpretation=True, beginners=True,
    )),
    ("pls_regression", _entry(
        "PLS Regression", "PLS Regression", "linear", "PLS Regression",
        aliases=("PLS", "Partial Least Squares Regression"),
        small_datasets=True, interpretation=True, beginners=True,
    )),
    ("ridge_regression", _entry(
        "Ridge", "Ridge Regression", "linear", "Ridge Regression",
        aliases=("Ridge Regressor",),
        small_datasets=True, interpretation=True, beginners=True,
    )),
    ("lasso_regression", _entry(
        "LASSO", "LASSO Regression", "linear", "LASSO Regression",
        aliases=("Lasso", "Lasso Regression"),
        small_datasets=True, interpretation=True, beginners=True,
    )),
    ("elastic_net", _entry(
        "Elastic Net", "Elastic Net Regression", "linear",
        "Elastic Net Regression", aliases=("ElasticNet",),
        small_datasets=True, interpretation=True,
    )),
    ("svr", _entry(
        "SVR", "Support Vector Regression", "kernel_similarity", "SVR",
        aliases=("Support Vector Regressor",),
        small_datasets=True, nonlinear=True, high_accuracy=True,
    )),
    ("gpr", _entry(
        "Gaussian Process Regression (GPR)", "Gaussian Process Regression",
        "kernel_similarity", "GPR",
        aliases=("GPR", "Gaussian Process Regressor"),
        small_datasets=True, nonlinear=True,
    )),
    ("knn_regression", _entry(
        "KNN Regression", "KNN Regression", "kernel_similarity",
        "KNN Regression", aliases=("KNeighborsRegressor", "KNN Regressor"),
        small_datasets=True, interpretation=True, nonlinear=True,
        beginners=True,
    )),
    ("mars_like", _entry(
        "MARS-like Regression", "MARS-like Regression", "spline",
        "MARS-like Regression", aliases=("MARS Regression",),
        small_datasets=True, interpretation=True, nonlinear=True,
    )),
    ("spline_regression", _entry(
        "Spline Regression", "Spline Regression", "spline",
        "Spline Regression",
        small_datasets=True, interpretation=True, nonlinear=True,
        beginners=True,
    )),
    ("gam_regression", _entry(
        "GAM Regression", "GAM Regression", "spline", "GAM Regression",
        aliases=("Generalized Additive Model", "Generalized Additive Model Regression"),
        small_datasets=True, interpretation=True, nonlinear=True,
    )),
    ("cart_regression", _entry(
        "CART Regression", "CART Regression", "trees", "CART Regression",
        aliases=("Decision Tree Regression", "DecisionTreeRegressor"),
        small_datasets=True, interpretation=True, nonlinear=True,
        beginners=True,
    )),
    ("random_forest", _entry(
        "Random Forest", "Random Forest", "tree_ensembles", "Random Forest",
        aliases=("Random Forest Regressor", "RandomForestRegressor"),
        small_datasets=True, nonlinear=True, high_accuracy=True,
        beginners=True,
    )),
    ("extra_trees", _entry(
        "Extra Trees", "Extra Trees", "tree_ensembles", "Extra Trees",
        aliases=("Extra Trees Regressor", "ExtraTreesRegressor"),
        small_datasets=True, nonlinear=True, high_accuracy=True,
    )),
    ("adaboost", _entry(
        "AdaBoost Regressor", "AdaBoost Regression", "boosting",
        "AdaBoost Regressor", aliases=("AdaBoost", "AdaBoostRegressor"),
        small_datasets=True, nonlinear=True, beginners=True,
    )),
    ("hist_gradient_boosting", _entry(
        "HistGradientBoosting Regressor", "Histogram Gradient Boosting",
        "boosting", "HistGradientBoosting Regressor",
        aliases=("HistGradientBoosting", "HistGradientBoostingRegressor"),
        nonlinear=True, high_accuracy=True,
    )),
    ("xgboost", _entry(
        "XGBoost", "XGBoost", "boosting", "XGBoost",
        aliases=("XGBRegressor",), dependency="xgboost",
        small_datasets=True, nonlinear=True, high_accuracy=True,
    )),
    ("lightgbm", _entry(
        "LightGBM", "LightGBM", "boosting", "LightGBM",
        aliases=("LGBMRegressor",), dependency="lightgbm",
        small_datasets=True, nonlinear=True, high_accuracy=True,
    )),
    ("catboost", _entry(
        "CatBoost", "CatBoost", "boosting", "CatBoost",
        aliases=("CatBoostRegressor",), dependency="catboost",
        small_datasets=True, nonlinear=True, high_accuracy=True,
    )),
    ("mlp_regression", _entry(
        "MLP Regression", "MLP Regression", "neural", "MLP Regression",
        aliases=("MLP Regressor", "MLPRegressor"),
        nonlinear=True,
    )),
    ("voting_regressor", _entry(
        "Voting Regressor", "Voting Regressor", "meta_ensembles",
        "Voting Regressor", aliases=("Voting Regression", "VotingRegressor"),
        nonlinear=True, beginners=True,
    )),
    ("stacking_regressor", _entry(
        "Stacking", "Stacking Regressor", "meta_ensembles",
        "Stacking Regressor", aliases=("Stacking Regression", "StackingRegressor"),
        nonlinear=True, high_accuracy=True,
    )),
    ("gep_symbolic", _entry(
        "GEP Symbolic Regression", "GEP Symbolic Regression", "symbolic",
        "GEP Symbolic Regression", aliases=("GEP",),
        interpretation=True, nonlinear=True,
    )),
    ("genetic_programming", _entry(
        "Genetic Programming Regression", "Genetic Programming Regression",
        "symbolic", "Genetic Programming Regression",
        aliases=("GP Regression", "Genetic Programming Symbolic Regression"),
        interpretation=True, nonlinear=True,
    )),
    ("pysr", _entry(
        "PySR", "PySR Symbolic Regression", "symbolic",
        "PySR Symbolic Regression", aliases=("PySRRegressor",),
        dependency="pysr", interpretation=True, nonlinear=True,
    )),
])


def _normalized_lookup_key(value):
    return " ".join(str(value).strip().casefold().replace("_", " ").split())


MODEL_NAME_INDEX = {}
for model_id, model in MODEL_CATALOG.items():
    MODEL_NAME_INDEX[_normalized_lookup_key(model_id)] = model_id
    for alias in model["aliases"]:
        MODEL_NAME_INDEX[_normalized_lookup_key(alias)] = model_id


def get_model_catalog():
    return MODEL_CATALOG


def get_model(model_or_alias):
    model_id = normalize_model_id(model_or_alias)
    return MODEL_CATALOG.get(model_id)


def normalize_model_id(model_or_alias):
    key = _normalized_lookup_key(model_or_alias)
    return MODEL_NAME_INDEX.get(key, str(model_or_alias).strip())


def normalize_runtime_name(model_or_alias):
    model = get_model(model_or_alias)
    return model["runtime_name"] if model else str(model_or_alias).strip()


def get_model_display_name(model_or_alias):
    model = get_model(model_or_alias)
    return model["display_name"] if model else str(model_or_alias).strip()


def get_model_encyclopedia_key(model_or_alias):
    model = get_model(model_or_alias)
    return model["encyclopedia_key"] if model else str(model_or_alias).strip()


def get_model_group(model_or_alias):
    model = get_model(model_or_alias)
    return model["group"] if model else ""


def model_is_available(model, availability=None):
    dependency = model.get("dependency")
    if not dependency:
        return True
    if availability is None:
        return False
    return bool(availability.get(dependency, False))


def model_matches_filters(model, active_filters=None, match_mode="all"):
    active_filters = tuple(active_filters or ())
    if not active_filters:
        return True

    values = [
        bool(model["filters"].get(filter_key, False))
        for filter_key in active_filters
        if filter_key in FILTER_KEYS
    ]
    if not values:
        return True
    return any(values) if match_mode == "any" else all(values)


def get_models_by_group(
    active_filters=None,
    match_mode="all",
    availability=None,
    include_unavailable=False,
):
    grouped = OrderedDict((group_name, []) for group_name in MODEL_GROUPS.values())

    for model_id, model in MODEL_CATALOG.items():
        if not model_matches_filters(model, active_filters, match_mode):
            continue
        if not include_unavailable and not model_is_available(model, availability):
            continue
        grouped[model["group"]].append(model["runtime_name"])

    return OrderedDict(
        (group_name, models)
        for group_name, models in grouped.items()
        if models
    )


def count_models(active_filters=None, match_mode="all", availability=None):
    return sum(
        len(models)
        for models in get_models_by_group(
            active_filters=active_filters,
            match_mode=match_mode,
            availability=availability,
        ).values()
    )
