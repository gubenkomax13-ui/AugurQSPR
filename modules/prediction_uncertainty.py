# -*- coding: utf-8 -*-

"""
Unified uncertainty estimation for Augur QSPR regression predictions.

The module contains no Streamlit code. It combines:
- calibrated split-free conformal intervals from out-of-fold residuals;
- bootstrap distributions for each query object;
- local error estimates from nearest neighbours;
- disagreement between independently trained models;
- native Gaussian-process posterior standard deviation;
- distance, descriptor-range, leverage and structural-similarity AD signals;
- an explainable reliability score and status.

The conformal interval is the calibrated prediction interval. Ensemble
disagreement, bootstrap standard deviation and GPR standard deviation are
reported separately and must not be presented as prediction intervals.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except Exception:
    Chem = None
    DataStructs = None
    rdFingerprintGenerator = None


def _matrix(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _vector(values, name):
    array = np.ravel(np.asarray(values, dtype=float))
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _safe_clone(value):
    try:
        return clone(value)
    except Exception:
        return copy.deepcopy(value)


def uncertainty_make_estimator(model, scaler=None):
    """Create an unfitted estimator that reproduces model preprocessing."""
    if scaler is None:
        return _safe_clone(model)
    return Pipeline([
        ("scale", _safe_clone(scaler)),
        ("model", _safe_clone(model)),
    ])


def uncertainty_predict(model, X, scaler=None):
    """Predict with a fitted model and its optional external scaler."""
    X = _matrix(X, "X")
    X_model = scaler.transform(X) if scaler is not None else X
    return np.ravel(np.asarray(model.predict(X_model), dtype=float))


def uncertainty_oof_predictions(
    model,
    scaler,
    X,
    y,
    cv=5,
    random_state=42,
):
    """Return out-of-fold predictions using the complete preprocessing path."""
    X = _matrix(X, "X")
    y = _vector(y, "y")
    if len(y) != X.shape[0]:
        raise ValueError("X and y contain different numbers of objects.")
    n_splits = min(max(2, int(cv)), len(y))
    if n_splits < 2:
        raise ValueError("At least two objects are required for calibration.")

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )
    estimator = uncertainty_make_estimator(model, scaler)
    return np.ravel(cross_val_predict(estimator, X, y, cv=splitter, n_jobs=None))


def uncertainty_conformal_quantile(absolute_residuals, alpha=0.10):
    """
    Finite-sample conformal quantile using the conservative 'higher' rule.
    """
    residuals = _vector(absolute_residuals, "absolute_residuals")
    residuals = residuals[residuals >= 0]
    if len(residuals) < 2:
        raise ValueError("At least two calibration residuals are required.")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    level = math.ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals)
    level = min(1.0, max(0.0, level))
    try:
        return float(np.quantile(residuals, level, method="higher"))
    except TypeError:
        return float(np.quantile(residuals, level, interpolation="higher"))


def uncertainty_conformal_interval(point_prediction, oof_residuals, alpha=0.10):
    """Build symmetric calibrated intervals around point predictions."""
    point = _vector(point_prediction, "point_prediction")
    residuals = _vector(oof_residuals, "oof_residuals")
    radius = uncertainty_conformal_quantile(np.abs(residuals), alpha=alpha)
    return {
        "lower": point - radius,
        "upper": point + radius,
        "radius": radius,
        "confidence": 1.0 - float(alpha),
    }


def uncertainty_bootstrap_query(
    model,
    scaler,
    X_train,
    y_train,
    X_query,
    n_bootstrap=100,
    random_state=42,
    min_unique_fraction=0.35,
):
    """Refit the full model path on bootstrap samples and predict X_query."""
    X_train = _matrix(X_train, "X_train")
    X_query = _matrix(X_query, "X_query")
    y_train = _vector(y_train, "y_train")
    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train and y_train lengths differ.")
    if X_train.shape[1] != X_query.shape[1]:
        raise ValueError("X_train and X_query dimensions differ.")

    n_bootstrap = max(2, int(n_bootstrap))
    rng = np.random.default_rng(int(random_state))
    predictions = []
    failed = 0

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_train), size=len(y_train))
        if len(np.unique(indices)) < max(2, int(len(y_train) * min_unique_fraction)):
            failed += 1
            continue
        estimator = uncertainty_make_estimator(model, scaler)
        try:
            estimator.fit(X_train[indices], y_train[indices])
            predictions.append(
                np.ravel(np.asarray(estimator.predict(X_query), dtype=float))
            )
        except Exception:
            failed += 1

    if len(predictions) < 2:
        return {
            "predictions": np.empty((0, X_query.shape[0])),
            "mean": np.full(X_query.shape[0], np.nan),
            "std": np.full(X_query.shape[0], np.nan),
            "lower": np.full(X_query.shape[0], np.nan),
            "upper": np.full(X_query.shape[0], np.nan),
            "successful": len(predictions),
            "failed": failed,
        }

    values = np.asarray(predictions, dtype=float)
    return {
        "predictions": values,
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0, ddof=1),
        "lower": np.quantile(values, 0.05, axis=0),
        "upper": np.quantile(values, 0.95, axis=0),
        "successful": values.shape[0],
        "failed": failed,
    }


def uncertainty_local_neighbour_error(
    X_train,
    y_train,
    oof_prediction,
    X_query,
    k=5,
):
    """Estimate local predictive error from nearest training neighbours."""
    X_train = _matrix(X_train, "X_train")
    X_query = _matrix(X_query, "X_query")
    y_train = _vector(y_train, "y_train")
    oof_prediction = _vector(oof_prediction, "oof_prediction")
    if not (len(y_train) == len(oof_prediction) == X_train.shape[0]):
        raise ValueError("Training arrays have inconsistent lengths.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    query_scaled = scaler.transform(X_query)
    k = min(max(1, int(k)), len(y_train))
    neighbours = NearestNeighbors(n_neighbors=k)
    neighbours.fit(X_scaled)
    distances, indices = neighbours.kneighbors(query_scaled)
    residuals = oof_prediction - y_train

    rows = []
    for query_index in range(X_query.shape[0]):
        local_indices = indices[query_index]
        local_errors = residuals[local_indices]
        rows.append({
            "local_mae": float(np.mean(np.abs(local_errors))),
            "local_rmse": float(np.sqrt(np.mean(local_errors ** 2))),
            "local_bias": float(np.mean(local_errors)),
            "nearest_distance": float(distances[query_index, 0]),
            "mean_neighbour_distance": float(np.mean(distances[query_index])),
            "neighbour_indices": local_indices.tolist(),
            "neighbour_distances": distances[query_index].tolist(),
        })
    return rows


def uncertainty_distance_ad(X_train, X_query, quantile=0.95):
    """Nearest-neighbour distance AD in standardized descriptor space."""
    X_train = _matrix(X_train, "X_train")
    X_query = _matrix(X_query, "X_query")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    query_scaled = scaler.transform(X_query)

    train_k = 2 if len(X_train) > 1 else 1
    train_nn = NearestNeighbors(n_neighbors=train_k).fit(train_scaled)
    train_distances = train_nn.kneighbors(train_scaled)[0]
    reference = train_distances[:, -1]
    threshold = float(np.quantile(reference, float(quantile)))

    query_nn = NearestNeighbors(n_neighbors=1).fit(train_scaled)
    query_distance = query_nn.kneighbors(query_scaled)[0][:, 0]
    denominator = threshold if threshold > 1e-12 else 1.0
    return {
        "distance": query_distance,
        "threshold": threshold,
        "ratio": query_distance / denominator,
        "inside": query_distance <= threshold,
    }


def uncertainty_range_ad(X_train, X_query, tolerance=0.05):
    """Descriptor-range AD with a small tolerance around training ranges."""
    X_train = _matrix(X_train, "X_train")
    X_query = _matrix(X_query, "X_query")
    minimum = np.min(X_train, axis=0)
    maximum = np.max(X_train, axis=0)
    span = maximum - minimum
    margin = np.where(span > 1e-12, span * float(tolerance), 0.0)
    outside = (X_query < minimum - margin) | (X_query > maximum + margin)
    fraction = np.mean(outside, axis=1)
    return {
        "outside_fraction": fraction,
        "outside_count": np.sum(outside, axis=1),
        "inside": fraction == 0,
    }


def uncertainty_leverage_ad(X_train, X_query):
    """Classical leverage AD for query objects."""
    X_train = _matrix(X_train, "X_train")
    X_query = _matrix(X_query, "X_query")
    if X_train.shape[1] != X_query.shape[1]:
        raise ValueError("X_train and X_query dimensions differ.")
    n, p = X_train.shape
    train_augmented = np.column_stack([np.ones(n), X_train])
    query_augmented = np.column_stack([np.ones(len(X_query)), X_query])
    inverse = np.linalg.pinv(train_augmented.T @ train_augmented)
    leverage = np.sum(
        (query_augmented @ inverse) * query_augmented,
        axis=1,
    )
    threshold = 3.0 * (p + 1) / n
    return {
        "leverage": leverage,
        "threshold": float(threshold),
        "inside": leverage <= threshold,
    }


def _morgan_fingerprint(smiles):
    if Chem is None or rdFingerprintGenerator is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprint(mol)


def uncertainty_similarity_ad(train_smiles, query_smiles, threshold=0.50):
    """Maximum Morgan/Tanimoto similarity to the training set."""
    if Chem is None or DataStructs is None:
        return None
    train_values = list(train_smiles) if train_smiles is not None else []
    query_values = list(query_smiles) if query_smiles is not None else []
    if not query_values:
        return None
    train_fps = [_morgan_fingerprint(value) for value in train_values]
    train_fps = [fp for fp in train_fps if fp is not None]
    if not train_fps:
        return None

    maximum = []
    for value in query_values:
        fingerprint = _morgan_fingerprint(value)
        if fingerprint is None:
            maximum.append(np.nan)
        else:
            maximum.append(max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps)))
    maximum = np.asarray(maximum, dtype=float)
    return {
        "maximum_similarity": maximum,
        "threshold": float(threshold),
        "inside": maximum >= float(threshold),
    }


def uncertainty_gpr_std(model, X_query, scaler=None):
    """Return native GPR posterior std when the fitted estimator supports it."""
    X_query = _matrix(X_query, "X_query")
    try:
        if isinstance(model, Pipeline):
            transformed = X_query
            for _, step in model.steps[:-1]:
                transformed = step.transform(transformed)
            final = model.steps[-1][1]
            _, std = final.predict(transformed, return_std=True)
            return np.ravel(np.asarray(std, dtype=float))

        transformed = scaler.transform(X_query) if scaler is not None else X_query
        _, std = model.predict(transformed, return_std=True)
        return np.ravel(np.asarray(std, dtype=float))
    except Exception:
        return np.full(X_query.shape[0], np.nan)


def uncertainty_model_consensus(model_entries, X_query, weights=None):
    """
    Predict with multiple fitted models.

    model_entries is {name: {"model": fitted_model, "scaler": optional_scaler}}.
    """
    X_query = _matrix(X_query, "X_query")
    names = list(model_entries)
    predictions = []
    used_names = []
    for name in names:
        entry = model_entries[name]
        try:
            predictions.append(
                uncertainty_predict(
                    entry["model"],
                    X_query,
                    scaler=entry.get("scaler"),
                )
            )
            used_names.append(name)
        except Exception:
            continue

    if not predictions:
        raise ValueError("No model produced a prediction.")
    values = np.asarray(predictions, dtype=float)

    if weights is None:
        weight_values = np.ones(len(used_names), dtype=float)
    else:
        weight_values = np.asarray(
            [float(weights.get(name, 0.0)) for name in used_names],
            dtype=float,
        )
        if not np.isfinite(weight_values).all() or np.sum(weight_values) <= 0:
            weight_values = np.ones(len(used_names), dtype=float)
    weight_values = weight_values / np.sum(weight_values)

    mean = np.average(values, axis=0, weights=weight_values)
    if len(used_names) > 1:
        variance = np.average((values - mean) ** 2, axis=0, weights=weight_values)
        std = np.sqrt(variance)
    else:
        std = np.full(X_query.shape[0], np.nan)

    return {
        "names": used_names,
        "weights": weight_values,
        "predictions": values,
        "mean": mean,
        "median": np.median(values, axis=0),
        "std": std,
        "minimum": np.min(values, axis=0),
        "maximum": np.max(values, axis=0),
    }


def uncertainty_reliability(
    y_train,
    oof_residuals,
    conformal_radius,
    consensus_std,
    bootstrap_std,
    local_mae,
    gpr_std,
    distance_inside,
    range_inside,
    similarity_inside=None,
    leverage_inside=None,
):
    """Combine uncertainty and AD signals into an explainable status."""
    y_train = _vector(y_train, "y_train")
    residuals = _vector(oof_residuals, "oof_residuals")
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    target_iqr = float(np.subtract(*np.percentile(y_train, [75, 25])))
    scale = rmse if rmse > 1e-12 else max(target_iqr, np.std(y_train), 1.0)

    arrays = {
        "consensus": np.asarray(consensus_std, dtype=float),
        "bootstrap": np.asarray(bootstrap_std, dtype=float),
        "local": np.asarray(local_mae, dtype=float),
        "gpr": np.asarray(gpr_std, dtype=float),
        "distance": np.asarray(distance_inside, dtype=bool),
        "range": np.asarray(range_inside, dtype=bool),
    }
    n_query = len(arrays["distance"])
    scores = np.full(n_query, 100.0)
    reasons = [[] for _ in range(n_query)]

    interval_ratio = (2.0 * float(conformal_radius)) / max(target_iqr, scale, 1e-12)
    if interval_ratio > 2.0:
        scores -= 20
        for item in reasons:
            item.append("wide_conformal_interval")
    elif interval_ratio > 1.0:
        scores -= 10
        for item in reasons:
            item.append("moderate_conformal_interval")

    for index in range(n_query):
        if not arrays["distance"][index]:
            scores[index] -= 30
            reasons[index].append("outside_distance_ad")
        if not arrays["range"][index]:
            scores[index] -= 20
            reasons[index].append("outside_descriptor_range")

        if similarity_inside is not None and not bool(similarity_inside[index]):
            scores[index] -= 25
            reasons[index].append("low_structural_similarity")
        if leverage_inside is not None and not bool(leverage_inside[index]):
            scores[index] -= 30
            reasons[index].append("outside_leverage_ad")

        for key, penalty, warning_ratio in (
            ("consensus", 15, 1.0),
            ("bootstrap", 15, 1.0),
            ("local", 15, 1.5),
            ("gpr", 10, 1.0),
        ):
            value = arrays[key][index]
            if np.isfinite(value) and value / scale > warning_ratio:
                scores[index] -= penalty
                reasons[index].append(f"high_{key}_uncertainty")

    scores = np.clip(scores, 0.0, 100.0)
    status = np.where(scores >= 75, "high", np.where(scores >= 50, "medium", "low"))
    return {
        "score": scores,
        "status": status,
        "reasons": reasons,
        "oof_rmse": rmse,
        "interval_to_target_scale": interval_ratio,
    }


def uncertainty_full_analysis(
    primary_model,
    primary_scaler,
    X_train,
    y_train,
    X_query,
    model_entries=None,
    train_smiles=None,
    query_smiles=None,
    leverage=None,
    leverage_threshold=None,
    alpha=0.10,
    cv=5,
    n_bootstrap=100,
    k_neighbors=5,
    random_state=42,
):
    """Run the complete Augur QSPR uncertainty workflow."""
    X_train = _matrix(X_train, "X_train")
    X_query = _matrix(X_query, "X_query")
    y_train = _vector(y_train, "y_train")

    point = uncertainty_predict(primary_model, X_query, scaler=primary_scaler)
    oof = uncertainty_oof_predictions(
        primary_model,
        primary_scaler,
        X_train,
        y_train,
        cv=cv,
        random_state=random_state,
    )
    residuals = oof - y_train
    conformal = uncertainty_conformal_interval(point, residuals, alpha=alpha)
    bootstrap = uncertainty_bootstrap_query(
        primary_model,
        primary_scaler,
        X_train,
        y_train,
        X_query,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    local = uncertainty_local_neighbour_error(
        X_train,
        y_train,
        oof,
        X_query,
        k=k_neighbors,
    )
    distance_ad = uncertainty_distance_ad(X_train, X_query)
    range_ad = uncertainty_range_ad(X_train, X_query)
    similarity_ad = uncertainty_similarity_ad(train_smiles, query_smiles)
    gpr_std = uncertainty_gpr_std(primary_model, X_query, primary_scaler)

    consensus_cv_rmse = {}
    consensus_weights = {}
    if model_entries:
        for name, entry in model_entries.items():
            try:
                if (
                    entry.get("model") is primary_model
                    and entry.get("scaler") is primary_scaler
                ):
                    model_oof = oof
                else:
                    model_oof = uncertainty_oof_predictions(
                        entry["model"],
                        entry.get("scaler"),
                        X_train,
                        y_train,
                        cv=cv,
                        random_state=random_state,
                    )
                model_rmse = float(
                    np.sqrt(np.mean((model_oof - y_train) ** 2))
                )
                consensus_cv_rmse[name] = model_rmse
                consensus_weights[name] = 1.0 / max(model_rmse ** 2, 1e-12)
            except Exception:
                consensus_cv_rmse[name] = np.nan
                consensus_weights[name] = 0.0
        consensus = uncertainty_model_consensus(
            model_entries,
            X_query,
            weights=consensus_weights,
        )
    else:
        consensus = {
            "names": [],
            "weights": np.array([]),
            "predictions": np.empty((0, len(point))),
            "mean": point.copy(),
            "median": point.copy(),
            "std": np.full(len(point), np.nan),
            "minimum": point.copy(),
            "maximum": point.copy(),
        }

    leverage_inside = None
    leverage_values = np.full(len(point), np.nan)
    if leverage is not None and leverage_threshold is not None:
        leverage_values = np.ravel(np.asarray(leverage, dtype=float))
        leverage_inside = leverage_values <= float(leverage_threshold)

    local_mae = np.asarray([row["local_mae"] for row in local], dtype=float)
    similarity_inside = (
        similarity_ad["inside"] if similarity_ad is not None else None
    )
    reliability = uncertainty_reliability(
        y_train=y_train,
        oof_residuals=residuals,
        conformal_radius=conformal["radius"],
        consensus_std=consensus["std"],
        bootstrap_std=bootstrap["std"],
        local_mae=local_mae,
        gpr_std=gpr_std,
        distance_inside=distance_ad["inside"],
        range_inside=range_ad["inside"],
        similarity_inside=similarity_inside,
        leverage_inside=leverage_inside,
    )

    table = pd.DataFrame({
        "prediction": point,
        "conformal_lower": conformal["lower"],
        "conformal_upper": conformal["upper"],
        "conformal_confidence": conformal["confidence"],
        "consensus_mean": consensus["mean"],
        "consensus_std": consensus["std"],
        "bootstrap_mean": bootstrap["mean"],
        "bootstrap_std": bootstrap["std"],
        "bootstrap_p05": bootstrap["lower"],
        "bootstrap_p95": bootstrap["upper"],
        "local_mae": local_mae,
        "local_rmse": [row["local_rmse"] for row in local],
        "local_bias": [row["local_bias"] for row in local],
        "nearest_distance": distance_ad["distance"],
        "distance_ad_threshold": distance_ad["threshold"],
        "distance_ad_inside": distance_ad["inside"],
        "descriptor_range_outside_fraction": range_ad["outside_fraction"],
        "descriptor_range_inside": range_ad["inside"],
        "gpr_std": gpr_std,
        "leverage": leverage_values,
        "reliability_score": reliability["score"],
        "reliability_status": reliability["status"],
        "reliability_reasons": [
            "; ".join(items) if items else "no_major_warning"
            for items in reliability["reasons"]
        ],
    })
    for model_index, model_name in enumerate(consensus["names"]):
        safe_name = str(model_name).replace("\n", " ").strip()
        table[f"prediction_{safe_name}"] = consensus["predictions"][model_index]
    if similarity_ad is not None:
        table["maximum_tanimoto_similarity"] = similarity_ad["maximum_similarity"]
        table["similarity_ad_inside"] = similarity_ad["inside"]

    return {
        "table": table,
        "point_prediction": point,
        "oof_prediction": oof,
        "oof_residuals": residuals,
        "conformal": conformal,
        "bootstrap": bootstrap,
        "local": local,
        "consensus": consensus,
        "consensus_cv_rmse": consensus_cv_rmse,
        "consensus_weights": consensus_weights,
        "gpr_std": gpr_std,
        "distance_ad": distance_ad,
        "range_ad": range_ad,
        "similarity_ad": similarity_ad,
        "reliability": reliability,
    }
