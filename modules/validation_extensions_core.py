# -*- coding: utf-8 -*-

"""Additional validation routines for QSPR regression models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:
    Chem = None
    MurckoScaffold = None

from modules.prediction_uncertainty import (
    uncertainty_conformal_interval,
    uncertainty_oof_predictions,
)
from modules.qspr_core import (
    qspr_create_regression_model,
    qspr_metrics,
    qspr_prediction_table,
)


def _as_matrix(X):
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("X contains NaN or infinite values.")
    return matrix


def _as_vector(y):
    vector = np.ravel(np.asarray(y, dtype=float))
    if not np.isfinite(vector).all():
        raise ValueError("y contains NaN or infinite values.")
    return vector


def _normalise_optional_list(values, n, default=""):
    if values is None:
        return [default] * n
    result = list(values)
    if len(result) != n:
        return [default] * n
    return result


def _make_pipeline(model_name, n_samples, n_features, params=None, scale=True):
    model = qspr_create_regression_model(
        model_name,
        n_samples=int(n_samples),
        n_features=int(n_features),
        params=params,
    )
    if scale:
        return Pipeline([("scale", StandardScaler()), ("model", model)])
    return Pipeline([("model", model)])


def _prediction_rows(
    y_true,
    y_pred,
    row_indices,
    original_indices,
    smiles,
    dataset_label,
    extra=None,
):
    table = qspr_prediction_table(
        y_true=y_true,
        y_pred=y_pred,
        smiles=[smiles[i] for i in row_indices],
        original_indices=original_indices,
        dataset_label=dataset_label,
    )
    if extra:
        for key, value in extra.items():
            table[key] = value
    return table


def repeated_kfold_validation(
    X,
    y,
    model_name,
    valid_indices=None,
    smiles=None,
    k=5,
    n_repeats=5,
    params=None,
    scale=True,
    random_state=42,
    progress_callback=None,
):
    """Run Repeated K-Fold CV and keep both split metrics and predictions."""
    X = _as_matrix(X)
    y = _as_vector(y)
    if len(y) != X.shape[0]:
        raise ValueError("X and y contain different numbers of rows.")
    n = len(y)
    if n < 4:
        raise ValueError("Repeated K-Fold requires at least 4 compounds.")

    k = min(max(2, int(k)), n)
    n_repeats = max(1, int(n_repeats))
    valid_indices = _normalise_optional_list(valid_indices, n, default=None)
    smiles = _normalise_optional_list(smiles, n)

    splitter = RepeatedKFold(
        n_splits=k,
        n_repeats=n_repeats,
        random_state=int(random_state),
    )
    rows = []
    prediction_tables = []
    prediction_sum = np.zeros(n, dtype=float)
    prediction_count = np.zeros(n, dtype=int)
    total = k * n_repeats

    for split_index, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        repeat = (split_index - 1) // k + 1
        fold = (split_index - 1) % k + 1
        estimator = _make_pipeline(
            model_name,
            n_samples=len(train_idx),
            n_features=X.shape[1],
            params=params,
            scale=scale,
        )
        estimator.fit(X[train_idx], y[train_idx])
        pred = np.ravel(np.asarray(estimator.predict(X[test_idx]), dtype=float))
        metrics = qspr_metrics(y[test_idx], pred)

        rows.append({
            "repeat": repeat,
            "fold": fold,
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE_percent": metrics["MAPE_percent"],
        })
        prediction_sum[test_idx] += pred
        prediction_count[test_idx] += 1
        prediction_tables.append(
            _prediction_rows(
                y_true=y[test_idx],
                y_pred=pred,
                row_indices=test_idx,
                original_indices=[valid_indices[i] for i in test_idx],
                smiles=smiles,
                dataset_label="repeated-kfold-test",
                extra={"repeat": repeat, "fold": fold},
            )
        )
        if progress_callback is not None:
            progress_callback(split_index, total)

    mean_prediction = prediction_sum / np.maximum(prediction_count, 1)
    aggregate_metrics = qspr_metrics(y, mean_prediction)
    split_table = pd.DataFrame(rows)
    summary_rows = []
    for metric in ("R2", "RMSE", "MAE", "MAPE_percent"):
        values = pd.to_numeric(split_table[metric], errors="coerce").dropna()
        summary_rows.append({
            "metric": metric,
            "mean": float(values.mean()) if len(values) else np.nan,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()) if len(values) else np.nan,
            "max": float(values.max()) if len(values) else np.nan,
        })

    aggregate_table = qspr_prediction_table(
        y_true=y,
        y_pred=mean_prediction,
        smiles=smiles,
        original_indices=valid_indices,
        dataset_label="repeated-kfold-mean",
    )
    aggregate_table["prediction_count"] = prediction_count

    return {
        "method": "repeated_kfold",
        "model_name": model_name,
        "k": k,
        "n_repeats": n_repeats,
        "metrics": aggregate_metrics,
        "summary_table": pd.DataFrame(summary_rows),
        "split_table": split_table,
        "prediction_table": (
            pd.concat(prediction_tables, ignore_index=True)
            if prediction_tables else pd.DataFrame()
        ),
        "aggregate_prediction_table": aggregate_table,
        "y": y,
        "y_pred_cv_mean": mean_prediction,
    }


def murcko_scaffold_labels(smiles):
    """Return one group label per SMILES using Bemis-Murcko scaffolds."""
    if Chem is None or MurckoScaffold is None:
        raise ValueError("RDKit MurckoScaffold is not available.")
    labels = []
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            labels.append(f"invalid::{index}")
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        labels.append(scaffold if scaffold else f"acyclic::{index}")
    return np.asarray(labels, dtype=object)


def group_holdout_validation(
    X,
    y,
    model_name,
    groups,
    valid_indices=None,
    smiles=None,
    test_size=0.2,
    random_state=42,
    params=None,
    scale=True,
    group_label="group",
):
    """Validate transfer by holding out whole chemical groups."""
    X = _as_matrix(X)
    y = _as_vector(y)
    groups = np.asarray(groups, dtype=object)
    if len(y) != X.shape[0] or len(groups) != X.shape[0]:
        raise ValueError("X, y and groups must contain the same number of rows.")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two groups are required for group split.")

    n = len(y)
    valid_indices = _normalise_optional_list(valid_indices, n, default=None)
    smiles = _normalise_optional_list(smiles, n)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(test_size),
        random_state=int(random_state),
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    estimator = _make_pipeline(
        model_name,
        n_samples=len(train_idx),
        n_features=X.shape[1],
        params=params,
        scale=scale,
    )
    estimator.fit(X[train_idx], y[train_idx])
    y_pred_train = np.ravel(np.asarray(estimator.predict(X[train_idx]), dtype=float))
    y_pred_test = np.ravel(np.asarray(estimator.predict(X[test_idx]), dtype=float))

    train_table = _prediction_rows(
        y_true=y[train_idx],
        y_pred=y_pred_train,
        row_indices=train_idx,
        original_indices=[valid_indices[i] for i in train_idx],
        smiles=smiles,
        dataset_label="group-train",
        extra={group_label: groups[train_idx]},
    )
    test_table = _prediction_rows(
        y_true=y[test_idx],
        y_pred=y_pred_test,
        row_indices=test_idx,
        original_indices=[valid_indices[i] for i in test_idx],
        smiles=smiles,
        dataset_label="group-test",
        extra={group_label: groups[test_idx]},
    )

    return {
        "method": "group_holdout",
        "model_name": model_name,
        "model": estimator,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "groups": groups,
        "train_groups": sorted({str(groups[i]) for i in train_idx}),
        "test_groups": sorted({str(groups[i]) for i in test_idx}),
        "metrics_train": qspr_metrics(y[train_idx], y_pred_train),
        "metrics_test": qspr_metrics(y[test_idx], y_pred_test),
        "train_table": train_table,
        "test_table": test_table,
    }


def scaffold_holdout_validation(
    X,
    y,
    model_name,
    smiles,
    valid_indices=None,
    test_size=0.2,
    random_state=42,
    params=None,
    scale=True,
):
    """Group hold-out where groups are Bemis-Murcko scaffolds."""
    labels = murcko_scaffold_labels(smiles)
    return group_holdout_validation(
        X=X,
        y=y,
        model_name=model_name,
        groups=labels,
        valid_indices=valid_indices,
        smiles=smiles,
        test_size=test_size,
        random_state=random_state,
        params=params,
        scale=scale,
        group_label="scaffold",
    )


def learning_curve_validation(
    X,
    y,
    model_name,
    params=None,
    scale=True,
    k=5,
    train_sizes=None,
    random_state=42,
):
    """Calculate train/CV learning curves for RMSE and R2."""
    X = _as_matrix(X)
    y = _as_vector(y)
    if len(y) != X.shape[0]:
        raise ValueError("X and y contain different numbers of rows.")
    if len(y) < 5:
        raise ValueError("Learning curves require at least 5 compounds.")

    k = min(max(2, int(k)), len(y))
    if train_sizes is None:
        train_pool = len(y) - int(np.ceil(len(y) / k))
        min_fraction = min(1.0, max(0.2, 3.0 / max(train_pool, 1)))
        train_sizes = np.linspace(min_fraction, 1.0, 5)
    estimator = _make_pipeline(
        model_name,
        n_samples=max(2, len(y) - int(np.ceil(len(y) / k))),
        n_features=X.shape[1],
        params=params,
        scale=scale,
    )
    cv = KFold(n_splits=k, shuffle=True, random_state=int(random_state))

    sizes_abs, train_rmse_raw, test_rmse_raw = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        shuffle=True,
        random_state=int(random_state),
        error_score=np.nan,
    )
    _, train_r2, test_r2 = learning_curve(
        estimator=clone(estimator),
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="r2",
        shuffle=True,
        random_state=int(random_state),
        error_score=np.nan,
    )

    table = pd.DataFrame({
        "train_size": sizes_abs,
        "train_rmse_mean": -np.nanmean(train_rmse_raw, axis=1),
        "train_rmse_std": np.nanstd(-train_rmse_raw, axis=1, ddof=1),
        "cv_rmse_mean": -np.nanmean(test_rmse_raw, axis=1),
        "cv_rmse_std": np.nanstd(-test_rmse_raw, axis=1, ddof=1),
        "train_r2_mean": np.nanmean(train_r2, axis=1),
        "train_r2_std": np.nanstd(train_r2, axis=1, ddof=1),
        "cv_r2_mean": np.nanmean(test_r2, axis=1),
        "cv_r2_std": np.nanstd(test_r2, axis=1, ddof=1),
    })
    final = table.iloc[-1]
    rmse_gap = float(final["cv_rmse_mean"] - final["train_rmse_mean"])
    if rmse_gap > max(0.25 * abs(float(final["train_rmse_mean"])), 1e-12):
        diagnosis = "possible_overfitting"
    elif len(table) >= 2 and table["cv_rmse_mean"].iloc[-2] - table["cv_rmse_mean"].iloc[-1] > 1e-12:
        diagnosis = "more_data_may_help"
    else:
        diagnosis = "curve_plateau"

    return {
        "method": "learning_curve",
        "model_name": model_name,
        "k": k,
        "table": table,
        "diagnosis": diagnosis,
        "rmse_gap": rmse_gap,
    }


def prediction_interval_holdout_coverage(
    X,
    y,
    model_name,
    valid_indices=None,
    smiles=None,
    test_size=0.2,
    confidence=0.90,
    calibration_cv=5,
    random_state=42,
    params=None,
    scale=True,
):
    """Calibrate conformal intervals on train OOF residuals and audit hold-out coverage."""
    X = _as_matrix(X)
    y = _as_vector(y)
    if len(y) != X.shape[0]:
        raise ValueError("X and y contain different numbers of rows.")
    alpha = 1.0 - float(confidence)
    if not 0.0 < alpha < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    n = len(y)
    valid_indices = _normalise_optional_list(valid_indices, n, default=None)
    smiles = _normalise_optional_list(smiles, n)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(test_size),
        random_state=int(random_state),
    )
    groups = np.arange(n)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    estimator = _make_pipeline(
        model_name,
        n_samples=len(train_idx),
        n_features=X.shape[1],
        params=params,
        scale=scale,
    )
    estimator.fit(X[train_idx], y[train_idx])
    point = np.ravel(np.asarray(estimator.predict(X[test_idx]), dtype=float))
    oof = uncertainty_oof_predictions(
        estimator,
        scaler=None,
        X=X[train_idx],
        y=y[train_idx],
        cv=int(calibration_cv),
        random_state=int(random_state),
    )
    residuals = oof - y[train_idx]
    interval = uncertainty_conformal_interval(point, residuals, alpha=alpha)
    inside = (y[test_idx] >= interval["lower"]) & (y[test_idx] <= interval["upper"])
    coverage = float(np.mean(inside)) if len(inside) else np.nan

    table = _prediction_rows(
        y_true=y[test_idx],
        y_pred=point,
        row_indices=test_idx,
        original_indices=[valid_indices[i] for i in test_idx],
        smiles=smiles,
        dataset_label="interval-holdout-test",
        extra={
            "conformal_lower": interval["lower"],
            "conformal_upper": interval["upper"],
            "inside_interval": inside,
        },
    )

    return {
        "method": "prediction_interval_holdout_coverage",
        "model_name": model_name,
        "confidence": float(confidence),
        "coverage": coverage,
        "coverage_gap": coverage - float(confidence),
        "radius": float(interval["radius"]),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "metrics_test": qspr_metrics(y[test_idx], point),
        "table": table,
        "oof_residuals": residuals,
    }
