# -*- coding: utf-8 -*-

"""
Unified descriptor importance calculations for QSPR regression models.

The module contains no Streamlit code. It supports:
- coefficients for linear estimators;
- native feature importance for tree-based estimators;
- permutation importance for any fitted estimator;
- grouped permutation importance for correlated descriptor groups.
- one normalized result table combining all available methods.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _as_2d_float_array(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix.")
    return X


def _feature_names(feature_names, n_features):
    names = list(feature_names or [])
    if len(names) != n_features:
        names = [f"x{i}" for i in range(n_features)]
    return [str(name) for name in names]


def descriptor_importance_final_estimator(model):
    """Return the final fitted estimator from a Pipeline or the model itself."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def descriptor_importance_model_feature_names(model, feature_names):
    """
    Return names seen by the final estimator.

    A QSPR auto-selection Pipeline may contain a fitted ``preselect`` step with
    ``selected_names_``. Native coefficients and tree importances belong to
    those selected descriptors, while permutation importance for the complete
    Pipeline belongs to its original input columns.
    """
    if isinstance(model, Pipeline):
        preselect = model.named_steps.get("preselect")
        selected = list(getattr(preselect, "selected_names_", []) or [])
        if selected:
            return [str(name) for name in selected]
    return [str(name) for name in list(feature_names or [])]


def descriptor_coefficient_importance(model, feature_names):
    """Extract signed and absolute coefficients from a fitted estimator."""
    estimator = descriptor_importance_final_estimator(model)
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame()

    try:
        values = np.ravel(np.asarray(estimator.coef_, dtype=float))
    except Exception:
        return pd.DataFrame()

    names = descriptor_importance_model_feature_names(model, feature_names)
    if len(names) != len(values):
        return pd.DataFrame()

    result = pd.DataFrame({
        "descriptor": names,
        "coefficient": values,
        "absolute_importance": np.abs(values),
        "method": "standardized_coefficient",
        "coefficient_scale": "model_feature_scale",
        "coefficient_note": (
            "If preprocessing includes scaling, these coefficients apply to "
            "scaled descriptors, not raw descriptor units."
        ),
    })
    return result.sort_values(
        "absolute_importance", ascending=False
    ).reset_index(drop=True)


def descriptor_native_importance(model, feature_names):
    """Extract native ``feature_importances_`` from a fitted estimator."""
    estimator = descriptor_importance_final_estimator(model)
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    try:
        values = np.ravel(
            np.asarray(estimator.feature_importances_, dtype=float)
        )
    except Exception:
        return pd.DataFrame()

    names = descriptor_importance_model_feature_names(model, feature_names)
    if len(names) != len(values):
        return pd.DataFrame()

    result = pd.DataFrame({
        "descriptor": names,
        "importance": values,
        "absolute_importance": np.abs(values),
        "method": "native_feature_importance",
    })
    return result.sort_values(
        "absolute_importance", ascending=False
    ).reset_index(drop=True)


def descriptor_permutation_importance(
    model,
    X,
    y,
    feature_names,
    scoring="neg_root_mean_squared_error",
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
):
    """Calculate repeated permutation importance for any fitted estimator."""
    X = _as_2d_float_array(X)
    y = np.ravel(np.asarray(y, dtype=float))
    names = _feature_names(feature_names, X.shape[1])

    if len(y) != X.shape[0]:
        raise ValueError("X and y contain different numbers of rows.")
    if X.shape[0] < 2:
        raise ValueError("At least two objects are required.")

    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=max(2, int(n_repeats)),
        random_state=int(random_state),
        n_jobs=n_jobs,
    )

    table = pd.DataFrame({
        "descriptor": names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "positive_repeats_fraction": np.mean(result.importances > 0, axis=1),
        "method": "permutation_importance",
    })
    table["absolute_importance"] = np.abs(table["importance_mean"])
    return table.sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)


def descriptor_correlation_groups(
    X,
    feature_names,
    threshold=0.90,
    method="spearman",
):
    """
    Build connected groups of descriptors with |correlation| >= threshold.

    Connected components are used so that A-B and B-C correlations place all
    three descriptors in one group, even if A-C is slightly below threshold.
    """
    X = _as_2d_float_array(X)
    names = _feature_names(feature_names, X.shape[1])
    threshold = float(threshold)

    if not 0.0 < threshold <= 1.0:
        raise ValueError("Correlation threshold must be in (0, 1].")

    frame = pd.DataFrame(X, columns=names)
    corr = frame.corr(method=method).abs().fillna(0.0).to_numpy()
    n_features = len(names)
    parent = list(range(n_features))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(n_features):
        for right in range(left + 1, n_features):
            if corr[left, right] >= threshold:
                union(left, right)

    grouped = {}
    for index in range(n_features):
        grouped.setdefault(find(index), []).append(index)

    return [
        {
            "group_id": group_number,
            "indices": indices,
            "descriptors": [names[index] for index in indices],
        }
        for group_number, indices in enumerate(grouped.values(), start=1)
    ]


def descriptor_grouped_permutation_importance(
    model,
    X,
    y,
    feature_names,
    threshold=0.90,
    scoring="neg_root_mean_squared_error",
    n_repeats=30,
    random_state=42,
):
    """Permute correlated descriptor groups together using one row ordering."""
    X = _as_2d_float_array(X)
    y = np.ravel(np.asarray(y, dtype=float))
    names = _feature_names(feature_names, X.shape[1])

    if len(y) != X.shape[0]:
        raise ValueError("X and y contain different numbers of rows.")

    groups = descriptor_correlation_groups(
        X=X,
        feature_names=names,
        threshold=threshold,
    )
    scorer = get_scorer(scoring)
    baseline = float(scorer(model, X, y))
    rng = np.random.default_rng(int(random_state))
    rows = []

    for group in groups:
        decreases = []
        indices = group["indices"]

        for _ in range(max(2, int(n_repeats))):
            order = rng.permutation(X.shape[0])
            X_permuted = X.copy()
            X_permuted[:, indices] = X[order][:, indices]
            decreases.append(
                baseline - float(scorer(model, X_permuted, y))
            )

        decreases = np.asarray(decreases, dtype=float)
        rows.append({
            "group_id": group["group_id"],
            "descriptors": "; ".join(group["descriptors"]),
            "n_descriptors": len(indices),
            "importance_mean": float(np.mean(decreases)),
            "importance_std": float(np.std(decreases, ddof=1)),
            "positive_repeats_fraction": float(np.mean(decreases > 0)),
            "method": "grouped_permutation_importance",
        })

    return pd.DataFrame(rows).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)


def descriptor_refit_for_holdout(
    model,
    scaler,
    X_train,
    y_train,
):
    """
    Clone and refit the current model specification on a hold-out train split.

    Pipelines are cloned directly. For legacy models that keep their scaler
    separately, a new StandardScaler + model Pipeline is constructed.
    """
    X_train = _as_2d_float_array(X_train)
    y_train = np.ravel(np.asarray(y_train, dtype=float))

    if isinstance(model, Pipeline):
        validation_model = clone(model)
    elif scaler is not None:
        validation_model = Pipeline([
            ("scale", StandardScaler()),
            ("model", clone(model)),
        ])
    else:
        validation_model = clone(model)

    validation_model.fit(X_train, y_train)
    return validation_model


def descriptor_unified_importance_table(
    coefficient_table=None,
    native_table=None,
    permutation_table=None,
    shap_table=None,
):
    """
    Combine importance methods into one descriptor-level result table.

    Values from different methods are not directly comparable, therefore every
    method is normalized by its largest absolute value before aggregation.
    ``combined_score`` is the mean of available normalized scores.
    """
    method_frames = []

    def add_method(table, method_name, value_column, signed_column=None,
                   stability_column=None, use_absolute=True):
        if not isinstance(table, pd.DataFrame) or table.empty:
            return
        if "descriptor" not in table.columns or value_column not in table.columns:
            return

        frame = table[["descriptor", value_column]].copy()
        frame["descriptor"] = frame["descriptor"].astype(str)
        values = pd.to_numeric(frame[value_column], errors="coerce")
        score_values = (
            np.abs(values)
            if use_absolute
            else values.clip(lower=0.0)
        )
        denominator = (
            float(np.nanmax(score_values))
            if score_values.notna().any()
            else 0.0
        )
        frame[f"{method_name}_score"] = (
            score_values / denominator if denominator > 0 else 0.0
        )
        if signed_column and signed_column in table.columns:
            frame[f"{method_name}_signed"] = pd.to_numeric(
                table[signed_column], errors="coerce"
            ).values
        if stability_column and stability_column in table.columns:
            frame[f"{method_name}_stability"] = pd.to_numeric(
                table[stability_column], errors="coerce"
            ).values
        frame = frame.drop(columns=[value_column])
        method_frames.append(frame)

    add_method(
        coefficient_table,
        "coefficient",
        "absolute_importance",
        signed_column="coefficient",
    )
    add_method(
        native_table,
        "native",
        "absolute_importance",
    )
    add_method(
        permutation_table,
        "permutation",
        "importance_mean",
        signed_column="importance_mean",
        stability_column="positive_repeats_fraction",
        use_absolute=False,
    )
    add_method(
        shap_table,
        "shap",
        "mean_abs_shap",
        signed_column="mean_shap",
    )

    if not method_frames:
        return pd.DataFrame()

    result = method_frames[0]
    for frame in method_frames[1:]:
        result = result.merge(frame, on="descriptor", how="outer")

    score_columns = [
        column for column in result.columns if column.endswith("_score")
    ]
    result["methods_available"] = result[score_columns].notna().sum(axis=1)
    result["combined_score"] = result[score_columns].mean(axis=1, skipna=True)
    result["combined_rank"] = (
        result["combined_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    permutation_stability = result.get("permutation_stability")
    if permutation_stability is None:
        result["stability_status"] = "not_evaluated"
    else:
        result["stability_status"] = np.select(
            [
                permutation_stability >= 0.80,
                permutation_stability >= 0.60,
            ],
            ["stable", "moderate"],
            default="unstable",
        )
        result.loc[permutation_stability.isna(), "stability_status"] = (
            "not_evaluated"
        )

    ordered = [
        "combined_rank",
        "descriptor",
        "combined_score",
        "methods_available",
        "stability_status",
    ]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining].sort_values(
        ["combined_rank", "descriptor"],
        ascending=[True, True],
    ).reset_index(drop=True)
